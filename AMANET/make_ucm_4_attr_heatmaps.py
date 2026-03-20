#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate 4 attention-montage heatmaps for AoAAllAttrProb (UCM).

Goal:
- Pick 4 different scenes (heuristic keyword match on captions).
- For each image_id, run `visualize_attention_heatmap.py` using AoAAllAttrProb model.
- Collect the resulting `montage.png` into one folder as 4 standalone PNGs.

This produces figures suitable for illustrating:
"Adding attribute-probability conditioning changes the decoder attention patterns".

Default paths assume the UCM AoAAllAttrProb experiment layout in this repo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from typing import Dict, List, Tuple


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


def choose_four_scenes(
    images: List[dict],
    gt_caps: Dict[int, List[str]],
    scene_rules: List[Tuple[str, List[str]]],
    prefer: List[str],
) -> List[Tuple[str, int]]:
    candidates: List[Tuple[str, int]] = []
    for im in images:
        image_id = int(im.get("id", im.get("imgid")))
        caps = gt_caps.get(image_id, [])
        if not caps:
            continue
        scene = pick_scene(caps, scene_rules)
        candidates.append((scene, image_id))

    # Build per-scene lists, stable order by image_id
    per_scene: Dict[str, List[int]] = {}
    for scene, image_id in sorted(candidates, key=lambda x: x[1]):
        per_scene.setdefault(scene, []).append(image_id)

    picked: List[Tuple[str, int]] = []
    used = set()

    # Prefer specific scenes first
    for s in prefer:
        ids = per_scene.get(s, [])
        if ids:
            picked.append((s, ids[0]))
            used.add(ids[0])
        if len(picked) >= 4:
            return picked

    # Fill with any other distinct scenes
    for scene, ids in per_scene.items():
        if scene in prefer or scene == "other":
            continue
        if not ids:
            continue
        if ids[0] in used:
            continue
        picked.append((scene, ids[0]))
        used.add(ids[0])
        if len(picked) >= 4:
            return picked

    # Last resort: fill by any remaining ids
    for scene, image_id in sorted(candidates, key=lambda x: x[1]):
        if image_id in used:
            continue
        picked.append((scene, image_id))
        used.add(image_id)
        if len(picked) >= 4:
            break

    return picked


def run_one(args, image_id: int, out_vis_dir: str) -> str:
    cmd = [
        args.python,
        "visualize_attention_heatmap.py",
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
        "--image_id",
        str(int(image_id)),
        "--attr_words",
        args.attr_words,
        "--out_dir",
        out_vis_dir,
        "--montage_cols",
        str(int(args.montage_cols)),
        "--montage_mode",
        args.montage_mode,
        "--montage_cell_size",
        str(int(args.montage_cell_size)),
        "--score_type",
        "visual_prob",
    ]
    if args.vis_image_size and int(args.vis_image_size) > 0:
        cmd += ["--vis_image_size", str(int(args.vis_image_size))]

    subprocess.run(cmd, check=True)

    montage = os.path.join(out_vis_dir, str(int(image_id)), "montage.png")
    if not os.path.exists(montage):
        raise FileNotFoundError(montage)
    return montage


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--python", type=str, default="python", help="Python executable to use (inside your conda env).")

    ap.add_argument(
        "--model",
        type=str,
        default="log_aoa_all_attr_prob_ucm/log_aoa_all_attr_prob_ucm/model-best.pth",
    )
    ap.add_argument(
        "--infos_path",
        type=str,
        default="log_aoa_all_attr_prob_ucm/log_aoa_all_attr_prob_ucm/infos_aoa_all_attr_prob_ucm-best.pkl",
    )

    ap.add_argument("--input_json", type=str, default="data/UCM/ucm_with_attr_probs_ucm40.json")
    ap.add_argument("--input_fc_dir", type=str, default="data/UCM/ucmtalk_fc")
    ap.add_argument("--input_att_dir", type=str, default="data/UCM/ucmtalk_att")
    ap.add_argument("--image_root", type=str, default="data/UCM/images")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    ap.add_argument("--gt_coco", type=str, default="coco-caption/annotations/captions_ucm_test.json")
    ap.add_argument("--attr_words", type=str, default="data/UCM/attribute_words_ucm.json")

    ap.add_argument("--out_dir", type=str, default="figures/ucm_attr_heatmaps_4")
    ap.add_argument("--vis_out_dir", type=str, default="vis/ucm_attr_heatmaps_4")

    ap.add_argument("--montage_cols", type=int, default=5)
    ap.add_argument("--montage_cell_size", type=int, default=224)
    ap.add_argument("--montage_mode", type=str, default="heatmap", choices=["heatmap", "overlay"])
    ap.add_argument("--vis_image_size", type=int, default=0)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.vis_out_dir, exist_ok=True)

    gt_caps, _id2file = load_coco_gt(args.gt_coco)

    # Scene keyword rules (same spirit as make_ucm_10_min_figs.py)
    scene_rules: List[Tuple[str, List[str]]] = [
        ("airport", ["airport", "airplane", "plane", "runway"]),
        ("harbor", ["harbor", "harbour", "boat", "boats", "ship", "dock"]),
        ("bridge", ["bridge"]),
        ("beach", ["beach", "shore"]),
        ("golf", ["golf"]),
        ("freeway", ["freeway", "highway", "road", "cars"]),
        ("parking", ["parking", "parked cars", "parked"]),
        ("residential", ["residential", "houses", "house", "buildings", "roofs"]),
        ("forest", ["forest", "trees", "woods"]),
        ("farmland", ["farmland", "cropland", "field", "fields", "formland"]),
        ("baseball", ["baseball", "diamond"]),
        ("tennis", ["tennis"]),
        ("stadium", ["stadium"]),
        ("river", ["river"]),
        ("intersection", ["intersection"]),
    ]

    prefer = ["airport", "harbor", "residential", "forest"]

    images = load_input_json_images(args.input_json, args.split)
    picked = choose_four_scenes(images, gt_caps, scene_rules, prefer)

    if len(picked) < 4:
        raise RuntimeError(f"Only picked {len(picked)} images; need 4.")

    manifest = []
    for i, (scene, image_id) in enumerate(picked, start=1):
        montage = run_one(args, image_id, args.vis_out_dir)
        out_png = os.path.join(args.out_dir, f"{i:02d}_{scene}_id{int(image_id)}.png")
        shutil.copyfile(montage, out_png)
        manifest.append({"index": i, "scene": scene, "image_id": int(image_id), "png": out_png})

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Saved 4 heatmaps to:", args.out_dir)
    for m in manifest:
        print(f"- {m['index']:02d} {m['scene']} image_id={m['image_id']} -> {os.path.basename(m['png'])}")


if __name__ == "__main__":
    main()
