#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate 4 AoAAllAttrProb attention-heatmap montages for UCM.

This script is a thin wrapper around visualize_attention_heatmap.py.
It runs the AoAAllAttrProb model on a list of image_ids and saves each
result under:
  <out_dir>/<image_id>/montage.png

It also copies each montage into:
  <out_dir>/montage_id<image_id>.png
so you immediately get 4 standalone images.

Example:
  python make_ucm_4_attrprob_heatmaps.py \
    --model log_aoa_all_attr_prob_ucm/log_aoa_all_attr_prob_ucm/model-best.pth \
    --infos_path log_aoa_all_attr_prob_ucm/log_aoa_all_attr_prob_ucm/infos_aoa_all_attr_prob_ucm-best.pkl \
    --image_ids 191,190,490,1090 \
    --out_dir figures/ucm_attrprob_heatmaps
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import List


def _parse_ids(s: str) -> List[int]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        out.append(int(p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model .pth")
    ap.add_argument("--infos_path", required=True, help="Path to infos_*.pkl")

    ap.add_argument("--input_json", default="data/UCM/ucm_with_attr_probs_ucm40.json")
    ap.add_argument("--input_fc_dir", default="data/UCM/ucmtalk_fc")
    ap.add_argument("--input_att_dir", default="data/UCM/ucmtalk_att")
    ap.add_argument("--image_root", default="data/UCM/images")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])

    ap.add_argument("--attr_words", default="data/UCM/attribute_words_ucm.json")

    ap.add_argument(
        "--image_ids",
        default="191,190,490,1090",
        help="Comma-separated image ids (exactly 4 recommended).",
    )
    ap.add_argument("--out_dir", default="figures/ucm_attrprob_heatmaps")

    ap.add_argument("--montage_cols", type=int, default=5)
    ap.add_argument("--montage_mode", type=str, default="heatmap", choices=["heatmap", "overlay"])
    ap.add_argument(
        "--score_type",
        type=str,
        default="visual_prob",
        choices=["prob", "visual_prob", "attn_max", "focus", "none"],
    )
    ap.add_argument("--montage_cell_size", type=int, default=224)
    ap.add_argument("--vis_image_size", type=int, default=0)

    args = ap.parse_args()

    image_ids = _parse_ids(args.image_ids)
    if len(image_ids) == 0:
        raise SystemExit("--image_ids is empty")

    os.makedirs(args.out_dir, exist_ok=True)

    script = os.path.join(os.path.dirname(__file__), "visualize_attention_heatmap.py")

    for image_id in image_ids:
        cmd = [
            sys.executable,
            script,
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
            args.out_dir,
            "--montage_cols",
            str(int(args.montage_cols)),
            "--montage_mode",
            args.montage_mode,
            "--score_type",
            args.score_type,
            "--montage_cell_size",
            str(int(args.montage_cell_size)),
        ]

        if int(args.vis_image_size) > 0:
            cmd += ["--vis_image_size", str(int(args.vis_image_size))]

        print("[run]", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # Flatten copy for convenience
        src = os.path.join(args.out_dir, str(int(image_id)), "montage.png")
        dst = os.path.join(args.out_dir, f"montage_id{int(image_id)}.png")
        if os.path.exists(src):
            shutil.copyfile(src, dst)

    print("done ->", args.out_dir)


if __name__ == "__main__":
    main()
