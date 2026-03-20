#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate 20 AoAAllAttrProb heatmap montages on Sydney, then pick the best 4.

Same workflow as UCM:
1) Pick 20 candidate test images covering different scenes (keyword heuristic on GT captions).
2) Run `visualize_attention_heatmap_batch.py` once to generate per-image outputs under vis/<image_id>/.
3) Read `token_metrics.json` to score caption correctness + attention concentration.
4) Pick 4 best images with distinct scenes when possible and copy `montage.png` into figures.

Scoring:
- caption_f1: content-word bag-of-words F1 between generated caption and GT captions (max over GT captions)
- focus_score: attention focus on non-stopword tokens
- visual_prob_score: mean visual_prob on non-stopword tokens (aux)

final = 0.6*caption_f1 + 0.3*focus_score + 0.1*visual_prob_score

All 20 full results remain in the vis folder.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from typing import Dict, List, Tuple


STOPWORDS = {
    "a",
    "an",
    "the",
    "there",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "in",
    "on",
    "at",
    "with",
    "and",
    "some",
    "many",
    "lots",
    "few",
    "this",
    "that",
}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _content_tokens(text: str) -> List[str]:
    t = normalize_text(text)
    toks = [x for x in t.split(" ") if x]
    out: List[str] = []
    for w in toks:
        if w in STOPWORDS:
            continue
        if len(w) <= 2:
            continue
        out.append(w)
    return out


def _f1_overlap(pred: List[str], gt: List[str]) -> float:
    if not pred or not gt:
        return 0.0
    from collections import Counter

    cp = Counter(pred)
    cg = Counter(gt)
    inter = 0
    for k, v in cp.items():
        inter += min(v, cg.get(k, 0))
    if inter <= 0:
        return 0.0
    prec = inter / max(1, sum(cp.values()))
    rec = inter / max(1, sum(cg.values()))
    if prec + rec <= 1e-12:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def pick_scene(captions: List[str], scene_rules: List[Tuple[str, List[str]]]) -> str:
    text = normalize_text(" ".join(captions))
    for scene, keys in scene_rules:
        for k in keys:
            if k in text:
                return scene
    return "other"


def load_input_json_images(path: str, split: str) -> List[dict]:
    data = load_json(path)
    images = data.get("images", []) if isinstance(data, dict) else []
    out = []
    for im in images:
        if str(im.get("split", "")).strip() != split:
            continue
        if "id" not in im and "imgid" not in im:
            continue
        out.append(im)
    return out


def choose_candidates(
    images: List[dict],
    gt_caps: Dict[int, List[str]],
    scene_rules: List[Tuple[str, List[str]]],
    num: int,
) -> List[Tuple[str, int]]:
    pools: Dict[str, List[int]] = {}
    for im in images:
        image_id = int(im.get("id", im.get("imgid")))
        caps = gt_caps.get(image_id, [])
        if not caps:
            continue
        scene = pick_scene(caps, scene_rules)
        pools.setdefault(scene, []).append(image_id)

    for scene in list(pools.keys()):
        pools[scene] = sorted(set(pools[scene]))

    picked: List[Tuple[str, int]] = []
    used = set()

    scenes_main = [s for s in pools.keys() if s != "other"]
    scenes_other = ["other"] if "other" in pools else []
    scenes = sorted(scenes_main) + scenes_other

    per_scene_limit = 2 if len(scenes_main) >= (num // 2) else 3
    per_scene_count: Dict[str, int] = {s: 0 for s in scenes}

    while len(picked) < num:
        progressed = False
        for s in scenes:
            if len(picked) >= num:
                break
            if per_scene_count.get(s, 0) >= per_scene_limit and s != "other":
                continue
            ids = pools.get(s, [])
            chosen = None
            for image_id in ids:
                if image_id not in used:
                    chosen = image_id
                    break
            if chosen is None:
                continue
            picked.append((s, chosen))
            used.add(chosen)
            per_scene_count[s] = per_scene_count.get(s, 0) + 1
            progressed = True
        if not progressed:
            break

    # If diversity constraints cap us (e.g., too few images in rare scenes),
    # fill the remaining slots from any scene while keeping determinism.
    if len(picked) < num:
        remaining_by_scene: List[Tuple[int, str]] = []
        for s in scenes:
            remaining = 0
            for image_id in pools.get(s, []):
                if image_id not in used:
                    remaining += 1
            remaining_by_scene.append((remaining, s))

        # Prefer scenes with more remaining images (usually road/residential),
        # and keep stable ordering for reproducibility.
        remaining_by_scene.sort(key=lambda x: (-x[0], x[1]))
        for _remaining, s in remaining_by_scene:
            if len(picked) >= num:
                break
            for image_id in pools.get(s, []):
                if len(picked) >= num:
                    break
                if image_id in used:
                    continue
                picked.append((s, image_id))
                used.add(image_id)

    return picked[:num]


def run_batch(args, image_ids: List[int]) -> None:
    cmd = [
        args.python,
        "visualize_attention_heatmap_batch.py",
        "--model",
        args.model,
        "--infos_path",
        args.infos_path,
        "--input_json",
        args.input_json,
        "--input_fc_dir",
        args.input_fc_dir,
        "--input_att_dir",
        args.input_att_dir,
        "--image_root",
        args.image_root,
        "--split",
        args.split,
        "--image_ids",
        ",".join([str(int(x)) for x in image_ids]),
        "--attr_words",
        args.attr_words,
        "--out_dir",
        args.vis_out_dir,
        "--montage_cols",
        str(int(args.montage_cols)),
        "--montage_mode",
        args.montage_mode,
        "--montage_cell_size",
        str(int(args.montage_cell_size)),
        "--score_type",
        "visual_prob",
        "--skip_existing",
        "0",
        "--caption_with_end",
        "1",
    ]
    if args.vis_image_size and int(args.vis_image_size) > 0:
        cmd += ["--vis_image_size", str(int(args.vis_image_size))]

    subprocess.run(cmd, check=True)


def score_from_metrics(metrics: dict, gt_caps: List[str]) -> Tuple[float, float, float, float]:
    tokens = [str(x) for x in metrics.get("tokens", [])]
    focus = metrics.get("focus", [])
    vprob = metrics.get("visual_prob", [])

    gen_caption = str(metrics.get("caption", ""))
    pred_ctoks = _content_tokens(gen_caption)
    gt_scores = []
    for c in gt_caps:
        gt_scores.append(_f1_overlap(pred_ctoks, _content_tokens(str(c))))
    caption_f1 = float(max(gt_scores)) if gt_scores else 0.0

    focus_keep: List[float] = []
    vprob_keep: List[float] = []
    for t, tok in enumerate(tokens):
        w = tok.lower()
        if w in STOPWORDS:
            continue
        if w in ["<end>", "<start>"]:
            continue
        try:
            focus_keep.append(float(focus[t]))
        except Exception:
            pass
        try:
            vprob_keep.append(float(vprob[t]))
        except Exception:
            pass

    if not focus_keep:
        focus_keep = [float(x) for x in focus] if focus else [0.0]

    focus_keep_sorted = sorted(focus_keep, reverse=True)
    topk = focus_keep_sorted[:3] if len(focus_keep_sorted) >= 3 else focus_keep_sorted
    focus_score = sum(topk) / max(1, len(topk))

    vprob_score = 0.0
    if vprob_keep:
        vprob_score = sum(vprob_keep) / max(1, len(vprob_keep))

    final = 0.6 * float(caption_f1) + 0.3 * float(focus_score) + 0.1 * float(vprob_score)
    return float(final), float(caption_f1), float(focus_score), float(vprob_score)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--python", type=str, default="python")

    ap.add_argument(
        "--model",
        type=str,
        default="log_aoa_all_attr_prob_sydney/log_aoa_all_attr_prob_sydney/model-best.pth",
    )
    ap.add_argument(
        "--infos_path",
        type=str,
        default="log_aoa_all_attr_prob_sydney/log_aoa_all_attr_prob_sydney/infos_aoa_all_attr_prob_sydney-best.pkl",
    )

    ap.add_argument("--input_json", type=str, default="data/Sydney/sydney_with_attr_probs_40.json")
    ap.add_argument("--input_fc_dir", type=str, default="data/Sydney/sydneytalk_fc")
    ap.add_argument("--input_att_dir", type=str, default="data/Sydney/sydneytalk_att")
    ap.add_argument("--image_root", type=str, default="data/Sydney/imgs")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    ap.add_argument("--gt_coco", type=str, default="coco-caption/annotations/captions_sydney_test.json")
    ap.add_argument("--attr_words", type=str, default="data/Sydney/attribute_words_sydney40.json")

    ap.add_argument("--num_candidates", type=int, default=20)
    ap.add_argument("--num_pick", type=int, default=4)

    ap.add_argument("--vis_out_dir", type=str, default="vis/sydney_attr_heatmaps_20")
    ap.add_argument("--out_dir", type=str, default="figures/sydney_attr_heatmaps_best4")

    ap.add_argument("--montage_cols", type=int, default=5)
    ap.add_argument("--montage_cell_size", type=int, default=224)
    ap.add_argument("--montage_mode", type=str, default="heatmap", choices=["heatmap", "overlay"])
    ap.add_argument("--vis_image_size", type=int, default=0)

    args = ap.parse_args()

    os.makedirs(args.vis_out_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    gt_caps, _id2file = load_coco_gt(args.gt_coco)

    # Scene keyword rules (generic aerial/caption keywords; works for Sydney too)
    scene_rules: List[Tuple[str, List[str]]] = [
        ("airport", ["airport", "airplane", "aircraft", "plane", "runway"]),
        ("harbor", ["harbor", "harbour", "boat", "boats", "ship", "dock", "port"]),
        ("bridge", ["bridge"]),
        ("beach", ["beach", "shore", "coast", "sea"]),
        ("road", ["road", "highway", "freeway", "street", "cars", "vehicles", "intersection"]),
        ("residential", ["residential", "houses", "house", "buildings", "roofs", "neighborhood"]),
        ("forest", ["forest", "trees", "woods"]),
        ("farmland", ["farmland", "cropland", "field", "fields", "farm"]),
        ("water", ["river", "lake", "water"]),
        ("stadium", ["stadium"]),
        ("golf", ["golf"]),
        ("tennis", ["tennis"]),
        ("baseball", ["baseball", "diamond"]),
        ("parking", ["parking", "parked"]),
    ]

    images = load_input_json_images(args.input_json, args.split)
    candidates = choose_candidates(images, gt_caps, scene_rules, num=int(args.num_candidates))
    if len(candidates) < int(args.num_candidates):
        raise RuntimeError(f"Only got {len(candidates)} candidates.")

    run_batch(args, [image_id for _scene, image_id in candidates])

    all_rows = []
    for idx, (scene, image_id) in enumerate(candidates, start=1):
        out_dir = os.path.join(args.vis_out_dir, str(int(image_id)))
        metrics_path = os.path.join(out_dir, "token_metrics.json")
        montage_path = os.path.join(out_dir, "montage.png")
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(metrics_path)
        if not os.path.exists(montage_path):
            raise FileNotFoundError(montage_path)

        metrics = load_json(metrics_path)
        gt_list = gt_caps.get(int(image_id), [])
        s, cap_f1, focus_s, vprob_s = score_from_metrics(metrics, gt_list)
        all_rows.append(
            {
                "rank_in_run": idx,
                "scene": scene,
                "image_id": int(image_id),
                "score": float(s),
                "caption_f1": float(cap_f1),
                "focus_score": float(focus_s),
                "visual_prob_score": float(vprob_s),
                "caption": str(metrics.get("caption", "")),
                "vis_dir": out_dir,
            }
        )

    all_sorted = sorted(all_rows, key=lambda x: x["score"], reverse=True)

    picked = []
    used_scenes = set()
    for row in all_sorted:
        if row["scene"] not in used_scenes and row["scene"] != "other":
            picked.append(row)
            used_scenes.add(row["scene"])
        if len(picked) >= int(args.num_pick):
            break

    if len(picked) < int(args.num_pick):
        for row in all_sorted:
            if row in picked:
                continue
            picked.append(row)
            if len(picked) >= int(args.num_pick):
                break

    out_manifest = []
    for i, row in enumerate(picked, start=1):
        montage = os.path.join(row["vis_dir"], "montage.png")
        out_png = os.path.join(
            args.out_dir,
            f"{i:02d}_{row['scene']}_id{row['image_id']}_score{row['score']:.3f}.png",
        )
        shutil.copyfile(montage, out_png)
        out_manifest.append({**row, "png": out_png})

    with open(os.path.join(args.out_dir, "manifest_all_20.json"), "w", encoding="utf-8") as f:
        json.dump(all_sorted, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "manifest_picked_4.json"), "w", encoding="utf-8") as f:
        json.dump(out_manifest, f, ensure_ascii=False, indent=2)

    print("All 20 results saved under:", args.vis_out_dir)
    print("Picked 4 copied to:", args.out_dir)
    for row in out_manifest:
        print(f"- {row['scene']} image_id={row['image_id']} score={row['score']:.3f}")


if __name__ == "__main__":
    main()
