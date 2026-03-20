#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate and save 20 attention heatmap montages on RSICD.

Workflow:
- Pick 20 images from a split (default: test) with lightweight scene diversity
  using the RSICD filename prefix (e.g. airport_123.jpg -> airport).
- Run visualize_attention_heatmap_batch.py once to generate outputs under vis/<image_id>/.
- Copy ALL 20 montage.png into figures/ and rename as: <filename_stem>_rsicd.png
- Write figures/rsicd_attr_heatmaps_20/manifest_all_20.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
from typing import Dict, List, Tuple


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scene_from_filename(file_name: str) -> str:
    base = os.path.basename(str(file_name))
    if "_" in base:
        return base.split("_", 1)[0].lower()
    stem = os.path.splitext(base)[0].lower()
    if stem.isdigit():
        return "unknown"
    return stem


def safe_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return stem or "image"


def load_input_images(path: str, split: str) -> List[dict]:
    data = load_json(path)
    images = data.get("images", []) if isinstance(data, dict) else []
    out: List[dict] = []
    for im in images:
        if str(im.get("split", "")).strip() != split:
            continue
        if "imgid" not in im and "id" not in im:
            continue
        if not str(im.get("filename", im.get("file_path", ""))).strip():
            continue
        out.append(im)
    return out


def choose_candidates(images: List[dict], num: int, seed: int) -> List[Tuple[str, int]]:
    rng = random.Random(int(seed))
    pools: Dict[str, List[int]] = {}
    for im in images:
        image_id = int(im.get("id", im.get("imgid")))
        fn = str(im.get("filename", im.get("file_path", "")))
        pools.setdefault(scene_from_filename(fn), []).append(image_id)

    for s in list(pools.keys()):
        ids = sorted(set(pools[s]))
        rng.shuffle(ids)
        pools[s] = ids

    scenes = sorted(pools.keys())
    per_scene_limit = 2
    per_scene_count: Dict[str, int] = {s: 0 for s in scenes}
    picked: List[Tuple[str, int]] = []
    used = set()

    while len(picked) < num:
        progressed = False
        for s in scenes:
            if len(picked) >= num:
                break
            if per_scene_count.get(s, 0) >= per_scene_limit:
                continue
            chosen = None
            for image_id in pools.get(s, []):
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

    if len(picked) < num:
        remaining: List[int] = []
        for s in scenes:
            for image_id in pools.get(s, []):
                if image_id not in used:
                    remaining.append(image_id)
        for image_id in remaining:
            if len(picked) >= num:
                break
            picked.append(("other", image_id))

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument(
        "--model",
        type=str,
        default="log_aoa_all_attr_prob_rsicd/log_aoa_attr_label_v1/model-best.pth",
    )
    ap.add_argument(
        "--infos_path",
        type=str,
        default="log_aoa_all_attr_prob_rsicd/log_aoa_attr_label_v1/infos_aoa_attr_label_v1-best.pkl",
    )

    ap.add_argument("--input_json", type=str, default="data/rsicd_with_attr_probs_new40.json")
    ap.add_argument("--input_fc_dir", type=str, default="data/rsicdtalk_fc")
    ap.add_argument("--input_att_dir", type=str, default="data/rsicdtalk_att")
    ap.add_argument("--image_root", type=str, default="data/RSICD_images")
    ap.add_argument("--attr_words", type=str, default="data/attribute_words_new40.json")

    ap.add_argument("--vis_out_dir", type=str, default="vis/rsicd_attr_heatmaps_20")
    ap.add_argument("--fig_out_dir", type=str, default="figures/rsicd_attr_heatmaps_20")

    ap.add_argument("--montage_cols", type=int, default=5)
    ap.add_argument("--montage_cell_size", type=int, default=224)
    ap.add_argument("--montage_mode", type=str, default="heatmap", choices=["heatmap", "overlay"])
    ap.add_argument("--vis_image_size", type=int, default=0)

    ap.add_argument("--python", type=str, default="python")

    args = ap.parse_args()

    os.makedirs(args.vis_out_dir, exist_ok=True)
    os.makedirs(args.fig_out_dir, exist_ok=True)

    images = load_input_images(args.input_json, args.split)
    if not images:
        raise RuntimeError(f"No images found for split={args.split} in {args.input_json}")

    picked = choose_candidates(images, args.num, seed=args.seed)
    image_ids = [image_id for _scene, image_id in picked]

    id2fn: Dict[int, str] = {}
    for im in images:
        image_id = int(im.get("id", im.get("imgid")))
        fn = str(im.get("filename", im.get("file_path", ""))).strip()
        if fn:
            id2fn[image_id] = fn

    run_batch(args, image_ids)

    manifest: List[dict] = []
    for idx, (scene, image_id) in enumerate(picked, start=1):
        src_dir = os.path.join(args.vis_out_dir, str(int(image_id)))
        montage_path = os.path.join(src_dir, "montage.png")
        metrics_path = os.path.join(src_dir, "token_metrics.json")
        if not os.path.exists(montage_path):
            raise FileNotFoundError(f"Missing montage: {montage_path}")

        metrics = {}
        if os.path.exists(metrics_path):
            try:
                metrics = load_json(metrics_path)
            except Exception:
                metrics = {}

        fn = id2fn.get(int(image_id), "")
        stem = safe_stem(fn) if fn else f"id{int(image_id)}"
        out_name = f"{stem}_rsicd.png"
        out_path = os.path.join(args.fig_out_dir, out_name)
        os.makedirs(args.fig_out_dir, exist_ok=True)
        shutil.copyfile(montage_path, out_path)

        manifest.append(
            {
                "rank": idx,
                "scene": scene,
                "image_id": int(image_id),
                "filename": fn,
                "out_file": out_name,
                "caption": metrics.get("caption", ""),
                "focus": metrics.get("focus_score", metrics.get("focus", None)),
                "visual_prob": metrics.get("visual_prob", None),
                "attn_max": metrics.get("attn_max", None),
            }
        )

    with open(os.path.join(args.fig_out_dir, "manifest_all_20.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"All {len(manifest)} results saved under: {args.vis_out_dir}")
    print(f"All {len(manifest)} montages copied to: {args.fig_out_dir}")


if __name__ == "__main__":
    main()
