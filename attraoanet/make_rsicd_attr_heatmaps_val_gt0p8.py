#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Select RSICD val images with visual_prob >= threshold, then save heatmaps.

Workflow:
1) Scan val split to compute per-image max visual_prob (fast mode, no token images).
2) Filter images where max visual_prob >= threshold.
3) Generate heatmap montages for the filtered images and copy to figures/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from typing import Dict, Iterable, List


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return stem or "image"


def load_split_images(path: str, split: str) -> List[dict]:
    data = load_json(path)
    images = data.get("images", []) if isinstance(data, dict) else []
    out: List[dict] = []
    for im in images:
        if str(im.get("split", "")).strip() != split:
            continue
        if "id" not in im and "imgid" not in im:
            continue
        out.append(im)
    return out


def chunked(items: List[int], size: int) -> Iterable[List[int]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_batch(args, image_ids: List[int], out_dir: str, make_montage: int, save_token_images: int) -> None:
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
        out_dir,
        "--score_type",
        "visual_prob",
        "--skip_existing",
        "1",
        "--caption_with_end",
        "1",
        "--make_montage",
        str(int(make_montage)),
        "--save_token_images",
        str(int(save_token_images)),
        "--montage_cols",
        str(int(args.montage_cols)),
        "--montage_mode",
        args.montage_mode,
        "--montage_cell_size",
        str(int(args.montage_cell_size)),
    ]
    if args.vis_image_size and int(args.vis_image_size) > 0:
        cmd += ["--vis_image_size", str(int(args.vis_image_size))]

    subprocess.run(cmd, check=True)


def max_visual_prob(metrics_path: str) -> float:
    if not os.path.exists(metrics_path):
        return 0.0
    try:
        data = load_json(metrics_path)
    except Exception:
        return 0.0
    vals = data.get("visual_prob", [])
    if not isinstance(vals, list):
        return 0.0
    nums = [x for x in vals if isinstance(x, (int, float))]
    return float(max(nums)) if nums else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--chunk_size", type=int, default=60)

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

    ap.add_argument("--scan_vis_dir", type=str, default="vis/rsicd_attr_heatmaps_val_scan")
    ap.add_argument("--vis_out_dir", type=str, default="vis/rsicd_attr_heatmaps_val_gt0p8")
    ap.add_argument("--fig_out_dir", type=str, default="figures/rsicd_attr_heatmaps_val_gt0p8")

    ap.add_argument("--montage_cols", type=int, default=5)
    ap.add_argument("--montage_cell_size", type=int, default=224)
    ap.add_argument("--montage_mode", type=str, default="heatmap", choices=["heatmap", "overlay"])
    ap.add_argument("--vis_image_size", type=int, default=0)

    ap.add_argument("--python", type=str, default="python")
    args = ap.parse_args()

    os.makedirs(args.scan_vis_dir, exist_ok=True)
    os.makedirs(args.vis_out_dir, exist_ok=True)
    os.makedirs(args.fig_out_dir, exist_ok=True)

    images = load_split_images(args.input_json, args.split)
    if not images:
        raise RuntimeError(f"No images found for split={args.split} in {args.input_json}")

    image_ids = [int(im.get("id", im.get("imgid"))) for im in images]
    id2fn: Dict[int, str] = {}
    for im in images:
        image_id = int(im.get("id", im.get("imgid")))
        fn = str(im.get("filename", im.get("file_path", ""))).strip()
        if fn:
            id2fn[image_id] = fn

    for chunk in chunked(image_ids, int(args.chunk_size)):
        run_batch(args, chunk, args.scan_vis_dir, make_montage=0, save_token_images=0)

    selected: List[int] = []
    scores: Dict[int, float] = {}
    for image_id in image_ids:
        metrics_path = os.path.join(args.scan_vis_dir, str(int(image_id)), "token_metrics.json")
        vmax = max_visual_prob(metrics_path)
        scores[image_id] = vmax
        if vmax >= float(args.threshold):
            selected.append(image_id)

    if not selected:
        print(f"No images meet threshold >= {args.threshold} on split={args.split}.")
        return

    run_batch(args, selected, args.vis_out_dir, make_montage=1, save_token_images=1)

    manifest: List[dict] = []
    for image_id in selected:
        src_dir = os.path.join(args.vis_out_dir, str(int(image_id)))
        montage_path = os.path.join(src_dir, "montage.png")
        if not os.path.exists(montage_path):
            continue
        fn = id2fn.get(int(image_id), "")
        stem = safe_stem(fn) if fn else f"id{int(image_id)}"
        out_name = f"{stem}_rsicd.png"
        out_path = os.path.join(args.fig_out_dir, out_name)
        shutil.copyfile(montage_path, out_path)
        manifest.append(
            {
                "image_id": int(image_id),
                "filename": fn,
                "out_file": out_name,
                "max_visual_prob": scores.get(image_id, 0.0),
            }
        )

    manifest_path = os.path.join(args.fig_out_dir, "manifest_gt0p8.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Selected {len(manifest)} images with max_visual_prob >= {args.threshold}.")
    print(f"- vis: {args.vis_out_dir}")
    print(f"- figures: {args.fig_out_dir}")
    print(f"- manifest: {manifest_path}")


if __name__ == "__main__":
    main()
