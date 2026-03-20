#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Batch version of visualize_attention_heatmap.py.

Why this exists:
- Running visualize_attention_heatmap.py 20 times spawns 20 processes, each re-importing torch,
  initializing CUDA, loading the model checkpoint, and constructing the DataLoader.
- This script loads model+DataLoader once, then loops over multiple image_ids.

Outputs are identical to visualize_attention_heatmap.py:
- vis/<out_dir>/<image_id>/{montage.png, token_metrics.json, ...}

Example:
  python visualize_attention_heatmap_batch.py \
    --model log_aoa_all_attr_prob_ucm/log_aoa_all_attr_prob_ucm/model-best.pth \
    --infos_path log_aoa_all_attr_prob_ucm/log_aoa_all_attr_prob_ucm/infos_aoa_all_attr_prob_ucm-best.pkl \
    --input_json data/UCM/ucm_with_attr_probs_ucm40.json \
    --input_fc_dir data/UCM/ucmtalk_fc \
    --input_att_dir data/UCM/ucmtalk_att \
    --image_root data/UCM/images \
    --split test \
    --image_ids 80,190,490 \
    --attr_words data/UCM/attribute_words_ucm.json \
    --out_dir vis/ucm_attr_heatmaps_20
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional

# Avoid OpenMP SHM issues in restricted environments.
os.environ.setdefault("KMP_DISABLE_SHM", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import numpy as np
import torch
from PIL import Image

import misc.utils as utils
import models
from dataloader import DataLoader

# Reuse helpers from the single-image script to keep behavior consistent.
import visualize_attention_heatmap as vah


def _find_image_index(loader: DataLoader, image_id: int) -> int:
    target = int(image_id)
    for i, img in enumerate(loader.info["images"]):
        img_id = img.get("id", img.get("imgid"))
        if img_id == target:
            return i
    raise ValueError(f"image_id={target} not found in input_json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--infos_path", type=str, required=True)

    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--input_fc_dir", type=str, required=True)
    parser.add_argument("--input_att_dir", type=str, required=True)
    parser.add_argument("--input_box_dir", type=str, default="0")
    parser.add_argument("--input_label_h5", type=str, default="none")

    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--image_ids",
        type=str,
        required=True,
        help="Comma-separated list of image_ids to process.",
    )

    parser.add_argument("--out_dir", type=str, default="vis/attn_heatmaps")
    parser.add_argument("--max_words", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--vis_image_size", type=int, default=0)

    parser.add_argument(
        "--box_dir_for_vis",
        type=str,
        default=None,
        help="Optional directory containing bbox .npy files for visualization only.",
    )

    parser.add_argument("--grid_topk", type=int, default=0)

    parser.add_argument("--make_montage", type=int, default=1)
    parser.add_argument("--montage_show_start_end", type=int, default=1)
    parser.add_argument("--montage_cols", type=int, default=5)
    parser.add_argument("--montage_cell_size", type=int, default=224)
    parser.add_argument("--montage_mode", type=str, default="heatmap", choices=["heatmap", "overlay"])
    parser.add_argument(
        "--heatmap_blur",
        type=float,
        default=0.0,
        help="Gaussian blur radius for smoother heatmaps (0 disables).",
    )

    parser.add_argument(
        "--score_type",
        type=str,
        default="visual_prob",
        choices=["prob", "visual_prob", "attn_max", "focus", "none"],
    )
    parser.add_argument(
        "--save_token_images",
        type=int,
        default=1,
        help="If 0, skip saving per-token overlay images (still writes token_metrics.json).",
    )

    parser.add_argument("--attr_words", type=str, default="data/attribute_words_new40.json")
    parser.add_argument("--attr_topk", type=int, default=10)

    parser.add_argument(
        "--caption_with_end",
        type=int,
        default=0,
        help="If 1, append an explicit '<end>' token to caption.txt and token_metrics.json caption field.",
    )

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--attn_head_reduce", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--attn_power", type=float, default=1.0)
    parser.add_argument(
        "--skip_existing",
        type=int,
        default=1,
        help="If 1, skip images with existing montage.png and token_metrics.json.",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load infos
    with open(args.infos_path, "rb") as f:
        infos = utils.pickle_load(f)

    opt = infos["opt"]
    # Avoid multiprocessing DataLoader to prevent OpenMP SHM errors in sandboxed envs.
    opt.num_workers = 0

    # Override data/model paths
    opt.input_json = args.input_json
    opt.input_fc_dir = args.input_fc_dir
    opt.input_att_dir = args.input_att_dir
    opt.input_box_dir = args.input_box_dir
    opt.input_label_h5 = args.input_label_h5

    opt.use_attr_labels = True

    sample_opt = {
        "sample_method": "greedy",
        "beam_size": 1,
        "temperature": 1.0,
        "remove_bad_endings": 0,
        "block_trigrams": 0,
    }

    vocab = infos["vocab"]
    opt.vocab = vocab

    model = models.setup(opt)
    del opt.vocab

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    map_location = None if use_cuda else torch.device("cpu")
    state = torch.load(args.model, map_location=map_location)
    model.load_state_dict(state)
    if use_cuda:
        model.cuda()
    model.eval()

    loader = DataLoader(opt)
    loader.ix_to_word = infos["vocab"]

    # Precompute attr vocab indices for visual_prob
    attr_vocab_indices: List[int] = []
    if args.score_type == "visual_prob":
        try:
            attr_list = vah._load_attr_word_list(args.attr_words)
            attr_vocab_indices = vah._build_attr_vocab_indices(loader.get_vocab(), attr_list)
        except Exception:
            attr_vocab_indices = []

    image_ids = [int(x) for x in str(args.image_ids).split(",") if str(x).strip()]

    for image_id in image_ids:
        img_ix = _find_image_index(loader, image_id)
        img_info = loader.info["images"][img_ix]
        file_path = img_info.get("file_path", img_info.get("filename", ""))
        if not file_path:
            raise ValueError("Cannot find file_path/filename in input_json for selected image")

        img_path = os.path.join(args.image_root, file_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        out_dir = os.path.join(args.out_dir, str(int(image_id)))
        os.makedirs(out_dir, exist_ok=True)

        if int(args.skip_existing) == 1:
            if os.path.exists(os.path.join(out_dir, "montage.png")) and os.path.exists(
                os.path.join(out_dir, "token_metrics.json")
            ):
                print("Skip existing:", out_dir)
                continue

        img = Image.open(img_path).convert("RGB")
        img.save(os.path.join(out_dir, "image.jpg"))

        vis_img = img
        if int(args.vis_image_size) and int(args.vis_image_size) > 0:
            s = int(args.vis_image_size)
            vis_img = img.resize((s, s), resample=Image.BILINEAR)
            vis_img.save(os.path.join(out_dir, f"image_vis_{s}.jpg"))

        # Optional boxes for visualization only
        boxes_xyxy = None
        box_dir = args.box_dir_for_vis if args.box_dir_for_vis is not None else args.input_box_dir
        if box_dir and str(box_dir) not in ["0", "none", "None"]:
            candidate = os.path.join(str(box_dir), f"{int(image_id)}.npy")
            if os.path.exists(candidate):
                try:
                    boxes_xyxy = np.load(candidate)
                except Exception:
                    boxes_xyxy = None

        # Prepare feats
        fc_feat, att_feat, _seq, _ = loader.__getitem__(img_ix)
        fc_feats = torch.from_numpy(fc_feat).float().unsqueeze(0).to(device)
        att_feats = torch.from_numpy(att_feat).float().unsqueeze(0).to(device)
        att_masks = None

        # Attribute labels (prob vector)
        attr_labels_np = img_info.get("attribute_labels", None)
        attr_labels = None
        if attr_labels_np is not None:
            attr_labels = torch.tensor(attr_labels_np, dtype=torch.float32).unsqueeze(0).to(device)

        attr_probs_vis = None
        if attr_labels is not None and hasattr(model, "_process_attr_probs"):
            try:
                with torch.no_grad():
                    try:
                        attr_probs_vis = (
                            model._process_attr_probs(attr_labels, fc_feats)
                            .squeeze(0)
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        )
                    except TypeError:
                        attr_probs_vis = (
                            model._process_attr_probs(attr_labels)
                            .squeeze(0)
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        )
            except Exception:
                attr_probs_vis = None

        # Record per-step attention weights by wrapping get_logprobs_state
        attn_steps: List[Optional[torch.Tensor]] = []
        logprobs_steps: List[Optional[torch.Tensor]] = []
        original_get_logprobs_state = model.get_logprobs_state

        def wrapped_get_logprobs_state(it, *a, **kw):
            out = original_get_logprobs_state(it, *a, **kw)
            try:
                lp = out[0]
                logprobs_steps.append(lp.detach().float().cpu())
            except Exception:
                logprobs_steps.append(None)

            attn_tensor = None
            try:
                attn_mod = model.core.attention
                a_ = getattr(attn_mod, "attn", None)
                if a_ is not None:
                    if a_.dim() == 4:
                        if args.attn_head_reduce == "max":
                            a_ = a_.max(1).values
                        else:
                            a_ = a_.mean(1)
                        if a_.size(1) == 1:
                            a_ = a_.squeeze(1)
                    elif a_.dim() == 3 and a_.size(1) == 1:
                        a_ = a_.squeeze(1)
                    if a_.dim() == 3:
                        a_ = a_[:, -1, :]
                    attn_tensor = a_.detach().float().cpu()
            except Exception:
                attn_tensor = None

            attn_steps.append(attn_tensor)
            return out

        model.get_logprobs_state = wrapped_get_logprobs_state

        seqLogprobs = None
        with torch.no_grad():
            try:
                if attr_labels is not None:
                    out = model(fc_feats, att_feats, att_masks, attr_labels=attr_labels, opt=sample_opt, mode="sample")
                else:
                    out = model(fc_feats, att_feats, att_masks, opt=sample_opt, mode="sample")
            except TypeError:
                out = model(fc_feats, att_feats, att_masks, opt=sample_opt, mode="sample")

            if isinstance(out, (tuple, list)) and len(out) >= 2:
                seq, seqLogprobs = out[0], out[1]
            else:
                seq = out[0] if isinstance(out, (tuple, list)) else out

        model.get_logprobs_state = original_get_logprobs_state

        words = vah._decode_tokens(loader.get_vocab(), seq[0])
        sent_words = list(words)
        if int(args.caption_with_end) == 1:
            sent_words = sent_words + ["<end>"]
        sent = " ".join(sent_words)

        token_probs: List[Optional[float]] = []
        if seqLogprobs is not None:
            try:
                lp = seqLogprobs[0].detach().float().cpu().numpy().tolist()
                for t in range(len(words)):
                    token_probs.append(float(math.exp(float(lp[t]))))
            except Exception:
                token_probs = [None for _ in range(len(words))]
        else:
            token_probs = [None for _ in range(len(words))]

        visual_probs: List[Optional[float]] = [None for _ in range(len(words))]
        if args.score_type == "visual_prob" and attr_vocab_indices:
            for t in range(len(words)):
                lp_t = logprobs_steps[t] if t < len(logprobs_steps) else None
                if lp_t is None:
                    continue
                try:
                    probs = torch.exp(lp_t[0])
                    idx = torch.tensor(attr_vocab_indices, dtype=torch.long)
                    mass = float(probs.index_select(0, idx).sum().item())
                    visual_probs[t] = mass
                except Exception:
                    visual_probs[t] = None

        with open(os.path.join(out_dir, "caption.txt"), "w", encoding="utf-8") as f:
            f.write(sent + "\n")

        max_words = min(len(words), int(args.max_words))
        attn_np: List[np.ndarray] = []

        metrics_tokens: List[str] = []
        metrics_token_prob: List[float] = []
        metrics_visual_prob: List[float] = []
        metrics_attn_max: List[float] = []
        metrics_focus: List[float] = []

        montage_cells: List[Image.Image] = []
        montage_tokens: List[str] = []
        montage_scores: List[float] = []

        for t in range(max_words):
            token = words[t]
            a = attn_steps[t]
            if a is None:
                continue
            a1 = a[0].numpy()
            if float(args.attn_power) != 1.0:
                pwr = float(args.attn_power)
                if pwr > 0:
                    a1 = np.power(np.maximum(a1, 0.0), pwr)
                    s = float(a1.sum())
                    if s > 1e-12:
                        a1 = a1 / s
            attn_np.append(a1)

            metrics_tokens.append(token)
            tp = token_probs[t] if t < len(token_probs) and token_probs[t] is not None else 0.0
            vp = visual_probs[t] if t < len(visual_probs) and visual_probs[t] is not None else 0.0
            metrics_token_prob.append(float(tp))
            metrics_visual_prob.append(float(vp))
            metrics_attn_max.append(float(np.max(a1)) if a1.size > 0 else 0.0)
            metrics_focus.append(vah._attn_focus_score(a1))

            if int(args.make_montage) == 1:
                montage_tokens.append(token)
                if args.score_type == "prob":
                    p = token_probs[t] if t < len(token_probs) else None
                    montage_scores.append(float(p) if p is not None else 0.0)
                elif args.score_type == "visual_prob":
                    vp2 = visual_probs[t] if t < len(visual_probs) else None
                    montage_scores.append(float(vp2) if vp2 is not None else 0.0)
                elif args.score_type == "attn_max":
                    montage_scores.append(float(np.max(a1)) if a1.size > 0 else 0.0)
                elif args.score_type == "focus":
                    montage_scores.append(vah._attn_focus_score(a1))
                else:
                    montage_scores.append(0.0)

                cell_size = int(args.montage_cell_size)
                cell_size = cell_size if cell_size > 0 else max(1, min(vis_img.size))
                cell_wh = (cell_size, cell_size)
                if args.montage_mode == "overlay":
                    base_sq = vis_img.resize(cell_wh, resample=Image.BILINEAR)
                    montage_cells.append(
                        vah._build_attn_overlay(
                            base_sq, a1, alpha=float(args.alpha), heatmap_blur=float(args.heatmap_blur)
                        )
                    )
                else:
                    montage_cells.append(
                        vah._attn_to_heatmap_rgb(a1, cell_wh, heatmap_blur=float(args.heatmap_blur))
                    )

            overlay = None
            if boxes_xyxy is not None and boxes_xyxy.ndim == 2 and boxes_xyxy.shape[1] >= 4:
                try:
                    overlay = vah._build_bbox_overlay(vis_img, boxes_xyxy[:, :4], a1, alpha=float(args.alpha))
                except Exception:
                    overlay = None

            if overlay is None:
                overlay = vah._build_attn_overlay(
                    vis_img, a1, alpha=float(args.alpha), heatmap_blur=float(args.heatmap_blur)
                )
                att_size = int(a1.size)
                grid = int(round(math.sqrt(att_size)))
                if grid * grid != att_size and boxes_xyxy is None:
                    vah._save_attn_strip(a1, os.path.join(out_dir, f"{t:02d}_attn_strip.png"))

        if int(args.save_token_images) == 1:
            safe_token = "".join([c if c.isalnum() or c in ["-", "_"] else "_" for c in token])
            os.makedirs(out_dir, exist_ok=True)
            overlay.save(os.path.join(out_dir, f"{t:02d}_{safe_token}.png"))

            if int(args.grid_topk) and int(args.grid_topk) > 0:
                try:
                    boxes_overlay = vah._build_grid_topk_boxes(
                        vis_img, a1, topk=int(args.grid_topk), alpha=float(args.alpha)
                    )
                    boxes_overlay.save(
                        os.path.join(out_dir, f"{t:02d}_{safe_token}_gridtop{int(args.grid_topk)}.png")
                    )
                except Exception:
                    pass

        if attn_np:
            np.save(os.path.join(out_dir, "attn_weights.npy"), np.stack(attn_np, axis=0))

        try:
            with open(os.path.join(out_dir, "token_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "image_id": int(image_id),
                        "caption": sent,
                        "tokens": metrics_tokens,
                        "token_prob": metrics_token_prob,
                        "visual_prob": metrics_visual_prob,
                        "attn_max": metrics_attn_max,
                        "focus": metrics_focus,
                        "score_type_used_for_montage": args.score_type,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass

        # Append <end> cell (EOS step) to match classic layouts.
        if (
            int(args.make_montage) == 1
            and int(args.montage_show_start_end) == 1
            and len(attn_steps) > len(words)
        ):
            a_end = attn_steps[len(words)]
            if a_end is not None:
                a1_end = a_end[0].numpy()
                if float(args.attn_power) != 1.0:
                    pwr = float(args.attn_power)
                    if pwr > 0:
                        a1_end = np.power(np.maximum(a1_end, 0.0), pwr)
                        s_end = float(a1_end.sum())
                        if s_end > 1e-12:
                            a1_end = a1_end / s_end

                montage_tokens.append("<end>")
                if args.score_type in ["prob", "visual_prob"]:
                    montage_scores.append(0.0)
                elif args.score_type == "attn_max":
                    montage_scores.append(float(np.max(a1_end)) if a1_end.size > 0 else 0.0)
                elif args.score_type == "focus":
                    montage_scores.append(vah._attn_focus_score(a1_end))
                else:
                    montage_scores.append(0.0)

                cell_size = int(args.montage_cell_size)
                cell_size = cell_size if cell_size > 0 else max(1, min(vis_img.size))
                cell_wh = (cell_size, cell_size)
                if args.montage_mode == "overlay":
                    base_sq = vis_img.resize(cell_wh, resample=Image.BILINEAR)
                    montage_cells.append(
                        vah._build_attn_overlay(
                            base_sq, a1_end, alpha=float(args.alpha), heatmap_blur=float(args.heatmap_blur)
                        )
                    )
                else:
                    montage_cells.append(
                        vah._attn_to_heatmap_rgb(a1_end, cell_wh, heatmap_blur=float(args.heatmap_blur))
                    )

        if int(args.make_montage) == 1 and montage_cells:
            try:
                import matplotlib.pyplot as plt

                cols = max(1, int(args.montage_cols))
                total = 1 + len(montage_cells)
                rows = int(math.ceil(total / float(cols)))

                fig = plt.figure(figsize=(cols * 2.2, rows * 2.6))

                ax0 = fig.add_subplot(rows, cols, 1)
                ax0.imshow(np.array(vis_img))
                ax0.set_title("<start>" if int(args.montage_show_start_end) == 1 else "<img>", fontsize=10, pad=2)
                if args.score_type != "none" and int(args.montage_show_start_end) == 1:
                    ax0.text(0.5, -0.03, f"{0.0:.2f}", transform=ax0.transAxes, ha="center", va="top", fontsize=12, color="green")
                ax0.axis("off")

                for i, (cell, tok, sc) in enumerate(zip(montage_cells, montage_tokens, montage_scores), start=2):
                    ax = fig.add_subplot(rows, cols, i)
                    ax.imshow(np.array(cell))
                    ax.set_title(tok, fontsize=10, pad=2)
                    if args.score_type != "none":
                        ax.text(0.5, -0.03, f"{sc:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=12, color="green")
                    ax.axis("off")

                plt.tight_layout(pad=0.1, h_pad=0.6, w_pad=0.2)
                fig.savefig(os.path.join(out_dir, "montage.png"), dpi=160)
                plt.close(fig)
            except Exception:
                pass

        if attr_labels_np is not None:
            try:
                attr_names = vah._load_attr_word_list(args.attr_words)
                probs_for_plot = attr_probs_vis if attr_probs_vis is not None else np.array(attr_labels_np, dtype=np.float32)
                vah._save_attr_topk(probs_for_plot, attr_names, os.path.join(out_dir, "attributes_topk.png"), topk=int(args.attr_topk))
            except Exception:
                pass

        print("Saved to:", out_dir)
        print("Caption:", sent)


if __name__ == "__main__":
    main()
