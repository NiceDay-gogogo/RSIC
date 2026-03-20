#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate 30 Fig10-style minimal qualitative figures for Sydney-Captions.

Output: figures/fig10_sydney_XX_<scene>_id<image_id>.png (+ .spec.json)
"""

import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

from make_fig10_triple_compare import make_figure


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_preds(path: str) -> Dict[int, str]:
    arr = load_json(path)
    out: Dict[int, str] = {}
    for item in arr:
        if "image_id" not in item or "caption" not in item:
            continue
        out[int(item["image_id"])] = str(item["caption"])
    return out


def load_coco_gt(path: str) -> Tuple[Dict[int, List[str]], Dict[int, str]]:
    coco = load_json(path)
    id2caps: Dict[int, List[str]] = {}
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        cap = str(ann.get("caption", "")).strip()
        if cap:
            id2caps.setdefault(image_id, []).append(cap)

    id2file: Dict[int, str] = {}
    for im in coco.get("images", []):
        if "id" not in im:
            continue
        image_id = int(im["id"])
        fn = str(im.get("file_name", "")).strip()
        if fn:
            id2file[image_id] = fn

    return id2caps, id2file


_STOPWORDS = {
    "a",
    "an",
    "the",
    "there",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "with",
    "and",
    "or",
    "on",
    "in",
    "at",
    "to",
    "of",
    "for",
    "from",
    "by",
    "while",
    "beside",
    "besides",
    "next",
    "near",
    "some",
    "many",
    "lots",
    "it",
    "this",
    "that",
    "as",
    "over",
    "under",
    "into",
    "through",
    "between",
    "around",
    "along",
}


def iter_tokens_with_separators(text: str) -> Iterable[str]:
    for part in re.findall(r"[A-Za-z0-9]+|\s+|[^A-Za-z0-9\s]", text):
        yield part


def norm_token(tok: str) -> str:
    return re.sub(r"[^a-z0-9]", "", tok.lower())


def build_vocab(caps: List[str]) -> set:
    vocab = set()
    for cap in caps:
        for part in iter_tokens_with_separators(cap):
            t = norm_token(part)
            if t:
                vocab.add(t)
    return vocab


def invert_synonyms(syn: Dict[str, List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, vals in syn.items():
        k = norm_token(key)
        if not k:
            continue
        out[k] = k
        for v in vals:
            vv = norm_token(v)
            if vv:
                out[vv] = k
    return out


def colorize_text(
    text: str,
    gt_vocab: set,
    syn_inv: Dict[str, str],
    attribute_words: set,
    mode: str,
    pred_vocab: Optional[set] = None,
) -> List[Dict[str, str]]:
    segments: List[Dict[str, str]] = []
    if pred_vocab is None:
        pred_vocab = set()

    for part in iter_tokens_with_separators(text):
        nt = norm_token(part)
        if not nt:
            segments.append({"text": part, "color": "black"})
            continue
        if nt in _STOPWORDS:
            segments.append({"text": part, "color": "black"})
            continue

        if mode == "pred":
            if nt in gt_vocab:
                color = "green"
            else:
                canon = syn_inv.get(nt)
                if canon and canon in gt_vocab:
                    color = "blue"
                elif nt in attribute_words:
                    color = "orange"
                else:
                    color = "red"
        else:  # gt
            if nt in pred_vocab:
                color = "green"
            else:
                canon = syn_inv.get(nt)
                if canon and canon in pred_vocab:
                    color = "blue"
                else:
                    canon_attr = canon if canon else nt
                    if canon_attr in attribute_words:
                        color = "red"
                    else:
                        color = "black"

        segments.append({"text": part, "color": color})

    return segments


def score_prediction(pred: str, gt_vocab: set, syn_inv: Dict[str, str]) -> float:
    content: List[str] = []
    for part in iter_tokens_with_separators(pred):
        t = norm_token(part)
        if not t or t in _STOPWORDS:
            continue
        content.append(t)
    if not content:
        return -1e9
    matched = 0
    red = 0
    for t in content:
        if t in gt_vocab:
            matched += 1
        else:
            canon = syn_inv.get(t)
            if canon and canon in gt_vocab:
                matched += 1
            else:
                red += 1
    return matched - 1.2 * red - 0.05 * len(content)


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pick_scene(image_id: int, captions: List[str], scene_rules: List[Tuple[str, List[str]]]) -> str:
    text = normalize_text(" ".join(captions))
    for scene, keys in scene_rules:
        for k in keys:
            if k in text:
                return scene
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures", help="Output directory")
    ap.add_argument("--num", type=int, default=30, help="How many figures")
    ap.add_argument(
        "--our",
        default="eval_results/aoa_all_attr_prob_sydney_ce_test_predictions.json",
        help="Our predictions json",
    )
    ap.add_argument(
        "--baseline",
        default="eval_results/showtell_all_attr_prob_sydney_rl_test_test_predictions.json",
        help="Baseline predictions json",
    )
    ap.add_argument(
        "--gt",
        default="coco-caption/annotations/captions_sydney_test.json",
        help="COCO GT captions json",
    )
    args = ap.parse_args()

    gt_caps, id2file = load_coco_gt(args.gt)
    our_preds = load_preds(args.our)
    base_preds = load_preds(args.baseline)

    # Sydney attribute vocab for auto coloring
    syn = load_json("data/Sydney/attribute_synonyms_sydney40.json")
    syn_inv = invert_synonyms({str(k): [str(x) for x in v] for k, v in syn.items()})
    attribute_words = set(norm_token(w) for w in load_json("data/Sydney/attribute_words_sydney40.json"))

    gt_count = 5

    # Scene keyword rules (heuristic on GT captions)
    scene_rules: List[Tuple[str, List[str]]] = [
        ("airport", ["airport", "airfield", "aircraft", "runway", "runways", "terminal", "plane", "planes"]),
        ("residential", ["residential", "houses", "house", "hosues", "buildings", "building", "roofs"]),
        ("industrial", ["industrial", "factory", "warehouses", "warehouse"]),
        ("meadow", ["meadow", "lawn", "lawns", "bushes", "bush", "paths", "path"]),
        ("roads", ["roads", "road", "intersection", "highway"]),
        ("water", ["river", "water", "harbor", "harbour", "boat", "boats"]),
    ]

    # Candidates must have GT, file, and both predictions
    candidates = []
    for image_id, caps in gt_caps.items():
        if image_id not in id2file:
            continue
        if image_id not in our_preds or image_id not in base_preds:
            continue
        candidates.append(image_id)

    # Prefer distinct scenes and higher-quality predictions
    picked: List[Tuple[str, int]] = []
    used_scenes = set()
    used_ids = set()

    for image_id in sorted(candidates):
        scene = pick_scene(image_id, gt_caps.get(image_id, []), scene_rules)
        if scene == "other":
            continue
        if scene in used_scenes:
            continue
        gt_vocab = build_vocab(gt_caps.get(image_id, [])[:gt_count])
        s = score_prediction(our_preds.get(image_id, ""), gt_vocab, syn_inv)
        if s < 0.5:
            continue
        picked.append((scene, image_id))
        used_scenes.add(scene)
        used_ids.add(image_id)
        if len(picked) >= args.num:
            break

    if len(picked) < args.num:
        for image_id in sorted(candidates):
            if image_id in used_ids:
                continue
            scene = pick_scene(image_id, gt_caps.get(image_id, []), scene_rules)
            gt_vocab = build_vocab(gt_caps.get(image_id, [])[:gt_count])
            s = score_prediction(our_preds.get(image_id, ""), gt_vocab, syn_inv)
            if s < 0.5:
                continue
            picked.append((scene, image_id))
            used_ids.add(image_id)
            if len(picked) >= args.num:
                break

    os.makedirs(args.outdir, exist_ok=True)

    base_spec = {
        "dataset": "Sydney",
        "split": "test",
        "our_predictions": args.our,
        "baseline_predictions": args.baseline,
        "gt_coco": args.gt,
        "gt_count": gt_count,
        "show_title": False,
        "show_legend": False,
        "show_frame": False,
        "show_header": False,
        "show_labels": True,
        "repeat_gt_label": True,
        "show_image_border": False,
        "show_missing_image_text": False,
        "width": 1700,
        "margin": 50,
        "thumb_size": 240,
        "thumb_gap": 28,
        "image_root": "data/Sydney/imgs",
    }

    for idx, (scene, image_id) in enumerate(picked[: args.num], start=1):
        our_text = our_preds.get(image_id, "")
        base_text = base_preds.get(image_id, "")
        gts = gt_caps.get(image_id, [])[:gt_count]
        gt_vocab = build_vocab(gts)
        pred_vocab = build_vocab([our_text])

        markup = {
            "our": colorize_text(our_text, gt_vocab, syn_inv, attribute_words, mode="pred"),
            "baseline": colorize_text(base_text, gt_vocab, syn_inv, attribute_words, mode="pred"),
            "gt": [
                colorize_text(gt, gt_vocab, syn_inv, attribute_words, mode="gt", pred_vocab=pred_vocab)
                for gt in gts
            ],
            "label_style": "fig10",
        }

        spec = dict(base_spec)
        spec["samples"] = [
            {
                "image_id": image_id,
                "scenario": scene,
                "gt_pick": list(range(gt_count)),
                "markup": markup,
            }
        ]

        out_png = os.path.join(args.outdir, f"fig10_sydney_{idx:02d}_{scene}_id{image_id}.png")
        out_spec = os.path.join(args.outdir, f"fig10_sydney_{idx:02d}_{scene}_id{image_id}.spec.json")

        with open(out_spec, "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)

        make_figure(out_spec, out_png)

    print(f"generated: {min(len(picked), args.num)} figures -> {args.outdir}")


if __name__ == "__main__":
    main()
