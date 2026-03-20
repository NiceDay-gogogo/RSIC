#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize per-word attention heatmaps for Att/AoA captioning models.

This script runs greedy decoding (beam_size=1) on a single image and saves
an overlay heatmap per generated word.

It also optionally plots the attribute probability distribution (top-k)
when the dataset JSON contains `attribute_labels`.

Example (AoAAllAttrProb RSICD):
  python visualize_attention_heatmap.py \
    --model log_aoa_all_attr_prob/log_aoa_all_attr_prob/model-best.pth \
    --infos_path log_aoa_all_attr_prob/log_aoa_all_attr_prob/infos_aoa_all_attr_prob-best.pkl \
    --input_json data/rsicd_with_attr_probs_new40.json \
    --input_fc_dir data/rsicdtalk_fc \
    --input_att_dir data/rsicdtalk_att \
    --image_root data/RSICD_images \
    --split test \
    --image_id 10051
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

# Avoid OpenMP SHM issues in restricted environments.
os.environ.setdefault("KMP_DISABLE_SHM", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

import misc.utils as utils
import models
from dataloader import DataLoader


def _load_attr_word_list(path: str) -> List[str]:
    """Load attribute word list from json.

    Supports list[str], dict[str|int,str], or dict with 'words'/'attributes' fields.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        if "words" in data and isinstance(data["words"], list):
            return [str(x) for x in data["words"]]
        if "attributes" in data and isinstance(data["attributes"], list):
            return [str(x) for x in data["attributes"]]
        # Common format: {"0":"airport", "1":"airplane", ...}
        vals = list(data.values())
        if all(isinstance(v, (str, int, float)) for v in vals):
            return [str(v) for v in vals]
    return []


def _build_attr_vocab_indices(vocab: Dict[Any, str], attr_words: List[str]) -> List[int]:
    """Map attribute words to vocab indices.

    - Uses exact match on vocab tokens.
    - For multi-word attributes, splits on whitespace and includes parts.
    - Adds simple plural/singular variants when present in vocab.
    """
    token_to_ix: Dict[str, int] = {}
    for k, v in vocab.items():
        try:
            ix = int(k)
        except Exception:
            continue
        token_to_ix[str(v).lower()] = ix

    indices: List[int] = []
    seen: set = set()

    def _maybe_add(tok: str) -> None:
        t = tok.strip().lower()
        if not t:
            return
        if t in token_to_ix:
            ix = int(token_to_ix[t])
            if ix not in seen:
                seen.add(ix)
                indices.append(ix)

    for w in attr_words:
        if not isinstance(w, str):
            continue
        parts = [p for p in w.replace("-", " ").split() if p]
        if not parts:
            continue
        for p in parts:
            _maybe_add(p)
            # plural/singular heuristics
            if p.endswith("s") and len(p) > 3:
                _maybe_add(p[:-1])
            else:
                _maybe_add(p + "s")

    return indices


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_ix_to_word(vocab: Dict[Any, str], ix: int) -> str:
    if ix in vocab:
        return vocab[ix]
    s = str(int(ix))
    if s in vocab:
        return vocab[s]
    return "<unk>"


def _decode_tokens(vocab: Dict[Any, str], seq_1d: torch.Tensor) -> List[str]:
    words: List[str] = []
    for token in seq_1d.tolist():
        if int(token) == 0:
            break
        words.append(_safe_ix_to_word(vocab, int(token)))
    return words


def _find_image_index(loader: DataLoader, image_id: Optional[int], ix: Optional[int], split: str) -> int:
    if ix is not None:
        return int(ix)

    if image_id is None:
        # Default: first image of requested split (fallback to 0)
        candidates = loader.split_ix.get(split, [])
        return int(candidates[0]) if candidates else 0

    target = int(image_id)
    for i, img in enumerate(loader.info["images"]):
        img_id = img.get("id", img.get("imgid"))
        if img_id == target:
            return i
    raise ValueError(f"image_id={target} not found in input_json")


def _build_attn_overlay(
    img: Image.Image, attn_vec: np.ndarray, alpha: float = 0.45, heatmap_blur: float = 0.0
) -> Image.Image:
    # attn_vec: (att_size,)
    # Some datasets use a fixed spatial grid (e.g., 14x14=196), while others
    # use object proposals (e.g., 36). For non-square sizes, pad into a
    # near-square rectangle so we can still visualize.
    att_size = int(attn_vec.size)

    def _infer_hw(n: int) -> Tuple[int, int]:
        g = int(round(math.sqrt(n)))
        g = max(g, 1)
        best_h, best_w = 1, n
        best_pad = best_h * best_w - n
        # Try a few candidates around sqrt(n)
        for h in range(max(1, g - 5), g + 6):
            w = int(math.ceil(n / float(h)))
            pad = h * w - n
            if pad < best_pad:
                best_h, best_w, best_pad = h, w, pad
        return best_h, best_w

    h, w = _infer_hw(att_size)
    if h * w != att_size:
        padded = np.zeros((h * w,), dtype=np.float32)
        padded[:att_size] = attn_vec.astype(np.float32)
        attn = padded.reshape(h, w)
    else:
        attn = attn_vec.reshape(h, w).astype(np.float32)

    attn = attn - attn.min()
    denom = float(attn.max() - attn.min())
    if denom > 1e-8:
        attn = attn / denom

    # Smooth in low-res space, then upsample for softer heatmaps.
    heat_low = Image.fromarray(np.uint8(attn * 255.0), mode="L")
    if float(heatmap_blur) > 0:
        heat_low = heat_low.filter(ImageFilter.GaussianBlur(radius=float(heatmap_blur)))
    heat = heat_low.resize(img.size, resample=Image.BICUBIC)

    # Colorize (simple 'jet'-like via matplotlib if available; fallback to red channel)
    try:
        import matplotlib.cm as cm

        cmap = cm.get_cmap("jet")
        heat_rgba = (cmap(np.array(heat, dtype=np.float32) / 255.0) * 255).astype(np.uint8)
        heat_rgb = Image.fromarray(heat_rgba[:, :, :3], mode="RGB")
    except Exception:
        heat_rgb = Image.merge("RGB", (heat, Image.new("L", img.size, 0), Image.new("L", img.size, 0)))

    base = img.convert("RGB")
    return Image.blend(base, heat_rgb, alpha)


def _attn_to_heatmap_rgb(attn_vec: np.ndarray, out_size: Tuple[int, int], heatmap_blur: float = 0.0) -> Image.Image:
    """Convert a 1D attention vector to a colored heatmap image (no background image)."""
    att_size = int(attn_vec.size)
    h = int(round(math.sqrt(att_size)))
    w = h
    if h * w != att_size:
        # Fall back to near-square padding (same as _build_attn_overlay)
        def _infer_hw(n: int) -> Tuple[int, int]:
            g = int(round(math.sqrt(n)))
            g = max(g, 1)
            best_h, best_w = 1, n
            best_pad = best_h * best_w - n
            for hh in range(max(1, g - 5), g + 6):
                ww = int(math.ceil(n / float(hh)))
                pad = hh * ww - n
                if pad < best_pad:
                    best_h, best_w, best_pad = hh, ww, pad
            return best_h, best_w

        h, w = _infer_hw(att_size)
        padded = np.zeros((h * w,), dtype=np.float32)
        padded[:att_size] = attn_vec.astype(np.float32)
        attn = padded.reshape(h, w)
    else:
        attn = attn_vec.reshape(h, w).astype(np.float32)

    attn = attn - float(attn.min())
    denom = float(attn.max() - attn.min())
    if denom > 1e-8:
        attn = attn / denom

    heat_low = Image.fromarray(np.uint8(attn * 255.0), mode="L")
    if float(heatmap_blur) > 0:
        heat_low = heat_low.filter(ImageFilter.GaussianBlur(radius=float(heatmap_blur)))
    heat_l = heat_low.resize(out_size, resample=Image.BICUBIC)
    try:
        import matplotlib.cm as cm

        cmap = cm.get_cmap("jet")
        heat_rgba = (cmap(np.array(heat_l, dtype=np.float32) / 255.0) * 255).astype(np.uint8)
        return Image.fromarray(heat_rgba[:, :, :3], mode="RGB")
    except Exception:
        return Image.merge("RGB", (heat_l, Image.new("L", out_size, 0), Image.new("L", out_size, 0)))


def _attn_focus_score(attn_vec: np.ndarray) -> float:
    """A 0..1 score: higher means more spatially focused attention.

    Defined as 1 - H(p)/log(K), where H is entropy and K is number of regions.
    """
    p = attn_vec.reshape(-1).astype(np.float64)
    p = np.maximum(p, 0.0)
    s = float(p.sum())
    if s <= 1e-12:
        return 0.0
    p = p / s
    eps = 1e-12
    h = -float(np.sum(p * np.log(p + eps)))
    k = int(p.size)
    if k <= 1:
        return 1.0
    hmax = float(np.log(k))
    if hmax <= 1e-12:
        return 1.0
    score = 1.0 - (h / hmax)
    return float(max(0.0, min(1.0, score)))


def _build_bbox_overlay(
    img: Image.Image,
    boxes_xyxy: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Overlay region attention using bounding boxes.

    boxes_xyxy: (K, 4) in absolute pixel coords (x1, y1, x2, y2)
    weights: (K,) attention weights
    """
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
        raise ValueError(f"boxes_xyxy must be (K,4), got {boxes_xyxy.shape}")
    w = weights.reshape(-1).astype(np.float32)
    if w.size != boxes_xyxy.shape[0]:
        raise ValueError(f"weights length {w.size} != num boxes {boxes_xyxy.shape[0]}")

    w = w - float(w.min())
    denom = float(w.max() - w.min())
    if denom > 1e-8:
        w = w / denom

    base = img.convert("RGB")
    overlay = Image.new("RGB", base.size, (0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    img_w, img_h = base.size
    for i in range(boxes_xyxy.shape[0]):
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        x1 = max(0, min(img_w - 1, int(round(x1))))
        y1 = max(0, min(img_h - 1, int(round(y1))))
        x2 = max(0, min(img_w - 1, int(round(x2))))
        y2 = max(0, min(img_h - 1, int(round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        a = int(round(255 * float(w[i])))
        draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, a), outline=(255, 0, 0, min(255, a + 40)))

    return Image.blend(base, overlay, alpha)


def _save_attn_strip(attn_vec: np.ndarray, out_path: str, height: int = 24) -> None:
    """Save a simple 1D attention visualization (useful when regions are not a spatial grid)."""
    w = attn_vec.reshape(-1).astype(np.float32)
    w = w - float(w.min())
    denom = float(w.max() - w.min())
    if denom > 1e-8:
        w = w / denom
    width = int(w.size)
    if width <= 0:
        return
    row = (w * 255.0).clip(0, 255).astype(np.uint8)[None, :]
    strip = np.repeat(row, repeats=max(1, int(height)), axis=0)
    im = Image.fromarray(strip, mode="L")
    try:
        import matplotlib.cm as cm

        cmap = cm.get_cmap("jet")
        rgba = (cmap(np.array(im, dtype=np.float32) / 255.0) * 255).astype(np.uint8)
        Image.fromarray(rgba[:, :, :3], mode="RGB").save(out_path)
    except Exception:
        im.save(out_path)


def _build_grid_topk_boxes(
    img: Image.Image,
    attn_vec: np.ndarray,
    topk: int = 5,
    alpha: float = 0.35,
) -> Image.Image:
    """Draw top-k attention grid cells as translucent boxes.

    This is meaningful when attention corresponds to a spatial grid (e.g., 14x14).
    """
    w = attn_vec.reshape(-1).astype(np.float32)
    att_size = int(w.size)
    grid = int(round(math.sqrt(att_size)))
    if grid * grid != att_size:
        raise ValueError(f"Attention size {att_size} is not a perfect square; cannot draw grid boxes")
    k = max(1, min(int(topk), att_size))

    w = w - float(w.min())
    denom = float(w.max() - w.min())
    if denom > 1e-8:
        w = w / denom

    idx = np.argsort(-w)[:k]
    base = img.convert("RGB")
    overlay = Image.new("RGB", base.size, (0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    img_w, img_h = base.size
    cell_w = img_w / float(grid)
    cell_h = img_h / float(grid)

    for j in idx.tolist():
        r = int(j // grid)
        c = int(j % grid)
        x1 = int(round(c * cell_w))
        y1 = int(round(r * cell_h))
        x2 = int(round((c + 1) * cell_w))
        y2 = int(round((r + 1) * cell_h))
        a = int(round(255 * float(w[j])))
        draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, a), outline=(255, 255, 255, min(255, a + 60)))

    return Image.blend(base, overlay, float(alpha))


def _save_attr_topk(attr_probs: np.ndarray, attr_names: List[str], out_path: str, topk: int = 10) -> None:
    if attr_probs.ndim != 1:
        attr_probs = attr_probs.reshape(-1)

    k = min(int(topk), int(attr_probs.size))
    idx = np.argsort(-attr_probs)[:k]
    names = [attr_names[i] if i < len(attr_names) else str(i) for i in idx]
    vals = attr_probs[idx]

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, max(3, 0.4 * k)))
        y = np.arange(k)
        plt.barh(y, vals)
        plt.yticks(y, names)
        plt.gca().invert_yaxis()
        plt.xlabel("prob")
        plt.title(f"Top-{k} attributes")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception:
        # If matplotlib is unavailable, just skip plotting.
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--infos_path", type=str, required=True)

    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--input_fc_dir", type=str, required=True)
    parser.add_argument("--input_att_dir", type=str, required=True)
    parser.add_argument("--input_box_dir", type=str, default="0")
    parser.add_argument("--input_label_h5", type=str, default="none")

    parser.add_argument("--image_root", type=str, default="data/RSICD_images")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--image_id", type=int, default=None)
    parser.add_argument("--ix", type=int, default=None)

    parser.add_argument("--out_dir", type=str, default="vis/attn_heatmaps")
    parser.add_argument("--max_words", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.45)

    parser.add_argument(
        "--vis_image_size",
        type=int,
        default=0,
        help="If >0, also visualize on a square resized image of this size (helps align with 14x14 grid features).",
    )

    parser.add_argument(
        "--box_dir_for_vis",
        type=str,
        default=None,
        help="Optional directory containing bbox .npy files for visualization only (does not affect model input).",
    )

    parser.add_argument(
        "--grid_topk",
        type=int,
        default=0,
        help="If >0 and attention is a square grid (e.g., 14x14), also save a top-k grid-box overlay per word.",
    )

    parser.add_argument(
        "--make_montage",
        type=int,
        default=1,
        help="If 1, also save a single montage figure showing per-word attention maps.",
    )
    parser.add_argument(
        "--montage_show_start_end",
        type=int,
        default=1,
        help="If 1, show <start> (original image) and append <end> cell in montage (classic layout).",
    )
    parser.add_argument("--montage_cols", type=int, default=5)
    parser.add_argument(
        "--montage_cell_size",
        type=int,
        default=224,
        help="Montage cell size in pixels for attention maps (square).",
    )
    parser.add_argument(
        "--montage_mode",
        type=str,
        default="heatmap",
        choices=["heatmap", "overlay"],
        help="Render montage cells as pure heatmaps or as overlays on the (vis) image.",
    )
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
        help=(
            "What number to show under each token in montage: "
            "prob (token probability), visual_prob (P[next token is a visual attribute word]), "
            "attn_max (max attention weight for that token), "
            "focus (entropy-based focus), or none."
        ),
    )

    parser.add_argument("--attr_words", type=str, default="data/attribute_words_new40.json")
    parser.add_argument("--attr_topk", type=int, default=10)

    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument(
        "--caption_with_end",
        type=int,
        default=0,
        help="If 1, append an explicit '<end>' token to caption.txt and token_metrics.json caption field.",
    )

    parser.add_argument(
        "--attn_head_reduce",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="How to reduce multi-head attention weights across heads when available.",
    )
    parser.add_argument(
        "--attn_power",
        type=float,
        default=1.0,
        help="Optional power transform to sharpen attention weights before visualization (>=1 recommended).",
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

    # Ensure attr labels are loaded if present
    opt.use_attr_labels = True

    # Force greedy decoding (beam search attention alignment is non-trivial)
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

    # Choose one image
    img_ix = _find_image_index(loader, args.image_id, args.ix, args.split)
    img_info = loader.info["images"][img_ix]
    file_path = img_info.get("file_path", img_info.get("filename", ""))
    if not file_path:
        raise ValueError("Cannot find file_path/filename in input_json for selected image")

    img_path = os.path.join(args.image_root, file_path)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    image_id = img_info.get("id", img_info.get("imgid", img_ix))

    out_dir = os.path.join(args.out_dir, str(image_id))
    _ensure_dir(out_dir)

    img = Image.open(img_path).convert("RGB")
    img.save(os.path.join(out_dir, "image.jpg"))

    vis_img = img
    if int(args.vis_image_size) and int(args.vis_image_size) > 0:
        s = int(args.vis_image_size)
        vis_img = img.resize((s, s), resample=Image.BILINEAR)
        vis_img.save(os.path.join(out_dir, f"image_vis_{s}.jpg"))

    # Optional: load boxes for visualization (does NOT change model input dims)
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

    # Attention masks: for visualization we can safely use None (treat all regions as valid).
    # This also avoids a PyTorch constraint where pack_padded_sequence expects CPU lengths.
    att_masks = None

    # Attribute labels (prob vector)
    attr_labels_np = img_info.get("attribute_labels", None)
    attr_labels = None
    if attr_labels_np is not None:
        attr_labels = torch.tensor(attr_labels_np, dtype=torch.float32).unsqueeze(0).to(device)
    attr_probs_vis = None
    if attr_labels is not None and hasattr(model, "_process_attr_probs"):
        # Support different signatures:
        # - AoAAllAttrProbModel._process_attr_probs(attr_labels, ref_tensor)
        # - (older variants) _process_attr_probs(attr_labels)
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
        # out[0] is logprobs over vocab for the next token
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
                # Multi-head dot attention: (B, H, Q, K)
                if a_.dim() == 4:
                    if args.attn_head_reduce == "max":
                        a_ = a_.max(1).values
                    else:
                        a_ = a_.mean(1)  # avg heads -> (B, Q, K)
                    if a_.size(1) == 1:
                        a_ = a_.squeeze(1)  # (B, K)
                elif a_.dim() == 3 and a_.size(1) == 1:
                    a_ = a_.squeeze(1)
                # Sometimes attention may still be (B, Q, K) with Q>1; take last query.
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
        # Some models accept attr_labels kwarg, others don't.
        try:
            if attr_labels is not None:
                out = model(fc_feats, att_feats, att_masks, attr_labels=attr_labels, opt=sample_opt, mode="sample")
            else:
                out = model(fc_feats, att_feats, att_masks, opt=sample_opt, mode="sample")
        except TypeError:
            # Fallback for models whose _sample signature doesn't include attr_labels.
            out = model(fc_feats, att_feats, att_masks, opt=sample_opt, mode="sample")

        # AttModel._sample returns (seq, seqLogprobs)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            seq, seqLogprobs = out[0], out[1]
        else:
            seq = out[0] if isinstance(out, (tuple, list)) else out

    # Restore
    model.get_logprobs_state = original_get_logprobs_state

    words = _decode_tokens(loader.get_vocab(), seq[0])
    sent_words = list(words)
    if int(args.caption_with_end) == 1:
        sent_words = sent_words + ["<end>"]
    sent = " ".join(sent_words)

    # Per-token probability of selected token (aligns with words[t])
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

    # Visual probability per step: P[next token is a visual attribute word]
    attr_vocab_indices: List[int] = []
    if args.score_type == "visual_prob":
        try:
            attr_list = _load_attr_word_list(args.attr_words)
            attr_vocab_indices = _build_attr_vocab_indices(loader.get_vocab(), attr_list)
        except Exception:
            attr_vocab_indices = []

    visual_probs: List[Optional[float]] = [None for _ in range(len(words))]
    if args.score_type == "visual_prob" and attr_vocab_indices:
        for t in range(len(words)):
            lp_t = logprobs_steps[t] if t < len(logprobs_steps) else None
            if lp_t is None:
                continue
            try:
                # lp_t: (B, V). Use batch 0.
                probs = torch.exp(lp_t[0])
                idx = torch.tensor(attr_vocab_indices, dtype=torch.long)
                mass = float(probs.index_select(0, idx).sum().item())
                visual_probs[t] = mass
            except Exception:
                visual_probs[t] = None

    with open(os.path.join(out_dir, "caption.txt"), "w", encoding="utf-8") as f:
        f.write(sent + "\n")

    # Save attention arrays
    # Align: attn_steps[t] corresponds to generated token at position t
    max_words = min(len(words), int(args.max_words))
    attn_np: List[np.ndarray] = []

    # Per-token metrics (aligned with saved attention vectors)
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
        a1 = a[0].numpy()  # (att_size,)
        # Optional sharpening (keeps it model-derived but makes peaks clearer)
        if float(args.attn_power) != 1.0:
            p = float(args.attn_power)
            if p > 0:
                a1 = np.power(np.maximum(a1, 0.0), p)
                s = float(a1.sum())
                if s > 1e-12:
                    a1 = a1 / s
        attn_np.append(a1)

        # Record metrics for downstream selection
        metrics_tokens.append(token)
        tp = token_probs[t] if t < len(token_probs) and token_probs[t] is not None else 0.0
        vp = visual_probs[t] if t < len(visual_probs) and visual_probs[t] is not None else 0.0
        metrics_token_prob.append(float(tp))
        metrics_visual_prob.append(float(vp))
        metrics_attn_max.append(float(np.max(a1)) if a1.size > 0 else 0.0)
        metrics_focus.append(_attn_focus_score(a1))

        # Prepare montage assets
        if int(args.make_montage) == 1:
            montage_tokens.append(token)
            if args.score_type == "prob":
                p = token_probs[t] if t < len(token_probs) else None
                montage_scores.append(float(p) if p is not None else 0.0)
            elif args.score_type == "visual_prob":
                vp = visual_probs[t] if t < len(visual_probs) else None
                montage_scores.append(float(vp) if vp is not None else 0.0)
            elif args.score_type == "attn_max":
                montage_scores.append(float(np.max(a1)) if a1.size > 0 else 0.0)
            elif args.score_type == "focus":
                montage_scores.append(_attn_focus_score(a1))
            else:
                montage_scores.append(0.0)

            cell_size = int(args.montage_cell_size)
            cell_size = cell_size if cell_size > 0 else max(1, min(vis_img.size))
            cell_wh = (cell_size, cell_size)

            if args.montage_mode == "overlay":
                base_sq = vis_img.resize(cell_wh, resample=Image.BILINEAR)
                montage_cells.append(
                    _build_attn_overlay(
                        base_sq, a1, alpha=float(args.alpha), heatmap_blur=float(args.heatmap_blur)
                    )
                )
            else:
                montage_cells.append(_attn_to_heatmap_rgb(a1, cell_wh, heatmap_blur=float(args.heatmap_blur)))

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
                # By convention in classic visualizations, <end> is shown with 0.00.
                montage_scores.append(0.0)
            elif args.score_type == "attn_max":
                montage_scores.append(float(np.max(a1_end)) if a1_end.size > 0 else 0.0)
            elif args.score_type == "focus":
                montage_scores.append(_attn_focus_score(a1_end))
            else:
                montage_scores.append(0.0)

            cell_size = int(args.montage_cell_size)
            cell_size = cell_size if cell_size > 0 else max(1, min(vis_img.size))
            cell_wh = (cell_size, cell_size)
            if args.montage_mode == "overlay":
                base_sq = vis_img.resize(cell_wh, resample=Image.BILINEAR)
                montage_cells.append(
                    _build_attn_overlay(
                        base_sq, a1_end, alpha=float(args.alpha), heatmap_blur=float(args.heatmap_blur)
                    )
                )
            else:
                montage_cells.append(
                    _attn_to_heatmap_rgb(a1_end, cell_wh, heatmap_blur=float(args.heatmap_blur))
                )

        overlay = None
        if boxes_xyxy is not None and boxes_xyxy.ndim == 2 and boxes_xyxy.shape[1] >= 4:
            try:
                overlay = _build_bbox_overlay(vis_img, boxes_xyxy[:, :4], a1, alpha=float(args.alpha))
            except Exception:
                overlay = None

        if overlay is None:
            overlay = _build_attn_overlay(
                vis_img, a1, alpha=float(args.alpha), heatmap_blur=float(args.heatmap_blur)
            )
            # If regions are not a perfect square and we have no boxes, the 2D overlay is only a heuristic.
            att_size = int(a1.size)
            grid = int(round(math.sqrt(att_size)))
            if grid * grid != att_size and boxes_xyxy is None:
                _save_attn_strip(a1, os.path.join(out_dir, f"{t:02d}_attn_strip.png"))
        safe_token = "".join([c if c.isalnum() or c in ["-", "_"] else "_" for c in token])
        os.makedirs(out_dir, exist_ok=True)
        overlay.save(os.path.join(out_dir, f"{t:02d}_{safe_token}.png"))

        if int(args.grid_topk) and int(args.grid_topk) > 0:
            try:
                boxes_overlay = _build_grid_topk_boxes(vis_img, a1, topk=int(args.grid_topk), alpha=float(args.alpha))
                boxes_overlay.save(os.path.join(out_dir, f"{t:02d}_{safe_token}_gridtop{int(args.grid_topk)}.png"))
            except Exception:
                pass

    if attn_np:
        np.save(os.path.join(out_dir, "attn_weights.npy"), np.stack(attn_np, axis=0))

    # Save metrics for later ranking/selection
    try:
        with open(os.path.join(out_dir, "token_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image_id": int(image_id) if image_id is not None else None,
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

    # Montage figure (similar to classic per-word attention visualizations)
    if int(args.make_montage) == 1 and montage_cells:
        try:
            import matplotlib.pyplot as plt

            cols = max(1, int(args.montage_cols))
            total = 1 + len(montage_cells)  # include original image as first cell
            rows = int(math.ceil(total / float(cols)))

            # More compact overall height.
            fig = plt.figure(figsize=(cols * 2.2, rows * 2.6))

            # First: original image
            ax0 = fig.add_subplot(rows, cols, 1)
            ax0.imshow(np.array(vis_img))
            ax0.set_title(
                "<start>" if int(args.montage_show_start_end) == 1 else "<img>",
                fontsize=10,
                pad=2,
            )
            if args.score_type != "none" and int(args.montage_show_start_end) == 1:
                ax0.text(
                    0.5,
                    -0.03,
                    f"{0.0:.2f}",
                    transform=ax0.transAxes,
                    ha="center",
                    va="top",
                    fontsize=12,
                    color="green",
                )
            ax0.axis("off")

            for i, (cell, tok, sc) in enumerate(zip(montage_cells, montage_tokens, montage_scores), start=2):
                ax = fig.add_subplot(rows, cols, i)
                ax.imshow(np.array(cell))
                ax.set_title(tok, fontsize=10, pad=2)
                if args.score_type != "none":
                    ax.text(
                        0.5,
                        -0.03,
                        f"{sc:.2f}",
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=12,
                        color="green",
                    )
                ax.axis("off")

            # Increase vertical padding so that the green score under a token
            # does not overlap with the title of the next row.
            # Make per-row spacing larger (avoid title/score overlap), while keeping score closer to the image.
            # Reduce outer margins and inter-row spacing (roughly half).
            plt.tight_layout(pad=0.1, h_pad=0.6, w_pad=0.2)
            fig.savefig(os.path.join(out_dir, "montage.png"), dpi=160)
            plt.close(fig)
        except Exception:
            pass

    # Attribute distribution plot
    if attr_labels_np is not None:
        try:
            with open(args.attr_words, "r", encoding="utf-8") as f:
                attr_names = json.load(f)
            probs_for_plot = attr_probs_vis if attr_probs_vis is not None else np.array(attr_labels_np, dtype=np.float32)
            _save_attr_topk(probs_for_plot, attr_names, os.path.join(out_dir, "attributes_topk.png"), topk=args.attr_topk)
            with open(os.path.join(out_dir, "attributes.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "attribute_labels_raw": attr_labels_np,
                        "attribute_probs_used_by_model": attr_probs_vis.tolist() if attr_probs_vis is not None else None,
                    },
                    f,
                    ensure_ascii=False,
                )
        except Exception:
            pass

    print("Saved to:", out_dir)
    print("Caption:", sent)


if __name__ == "__main__":
    main()
