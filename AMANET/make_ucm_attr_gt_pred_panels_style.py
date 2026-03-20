#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate UCM panels with image + GT/Prediction attribute lists.

Style target: single image on top, GT and Predictions text lines below,
similar to the provided sample layout.
"""

from __future__ import annotations

import json
import os
from typing import Callable, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def unique_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for w in seq:
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def pick_font(size: int):
    # Match the font used in figure renderers.
    for name in ["DejaVuSans.ttf", "DejaVuSansMono.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_labeled_list(
    draw: ImageDraw.ImageDraw,
    label: str,
    words: List[str],
    x0: int,
    y0: int,
    max_w: int,
    line_h: int,
    font: ImageFont.ImageFont,
    color_func: Callable[[str], Tuple[int, int, int]],
    measure_only: bool = False,
) -> Tuple[int, int]:
    """Draw label + word list with wrapping. Returns (lines_used, last_y)."""
    x = x0
    y = y0
    lines = 1

    def text_w(text: str) -> int:
        return int(draw.textlength(text, font=font))

    # Draw label
    if not measure_only:
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    x += text_w(label)

    # Add a space after label
    space = " "
    if x + text_w(space) > max_w:
        x = x0
        y += line_h
        lines += 1
    else:
        x += text_w(space)

    for idx, w in enumerate(words):
        token = w + ("," if idx < len(words) - 1 else "")
        token_text = token + " "
        wlen = text_w(token_text)
        if x + wlen > max_w:
            x = x0
            y += line_h
            lines += 1
        if not measure_only:
            draw.text((x, y), token_text, fill=color_func(w), font=font)
        x += wlen

    return lines, y


def main() -> None:
    vis_dir = "vis/ucm_attr_heatmaps_filenames_10"
    img_dir = "data/UCM/images"
    out_dir = "figures/ucm_attr_gt_pred_panels_style"
    os.makedirs(out_dir, exist_ok=True)

    ucm = load_json("data/UCM/dataset_ucm.json")
    ucm_attr = load_json("data/UCM/ucm_with_attr_probs_ucm40.json")
    attr_words = load_json("data/UCM/attribute_words_ucm.json")

    attr_set = {w.lower() for w in attr_words}
    img_by_id = {int(im["imgid"]): im for im in ucm["images"]}
    attr_by_id = {int(im["imgid"]): im for im in ucm_attr["images"]}

    ids = sorted(int(d) for d in os.listdir(vis_dir) if d.isdigit())

    font = pick_font(16)
    bbox = font.getbbox("Ag")
    line_h = int(bbox[3] - bbox[1] + 4)
    pad = 8
    max_w = 320

    for image_id in ids:
        im = img_by_id.get(image_id)
        if not im:
            continue
        fn = im.get("filename", "")
        img_path = os.path.join(img_dir, fn)
        if not os.path.isfile(img_path):
            continue

        # GT attribute words from 5 captions
        gt_words: List[str] = []
        for s in im.get("sentences", []):
            for tok in s.get("tokens", []):
                w = str(tok).lower()
                if w in attr_set:
                    gt_words.append(w)
        gt_words = unique_preserve(gt_words)

        # Predicted attrs with prob >= 0.8
        attr_im = attr_by_id.get(image_id, {})
        probs = attr_im.get("attribute_probs", [])
        pred_words: List[str] = []
        for idx, p in enumerate(probs):
            try:
                pv = float(p)
            except Exception:
                continue
            if pv >= 0.8 and idx < len(attr_words):
                pred_words.append(str(attr_words[idx]).lower())

        img = Image.open(img_path).convert("RGB")
        if img.width > max_w:
            new_h = int(img.height * (max_w / img.width))
            img = img.resize((max_w, new_h), resample=Image.BICUBIC)

        # Measure text height
        dummy = Image.new("RGB", (img.width, 100), color="white")
        ddraw = ImageDraw.Draw(dummy)
        lines = 0
        l1, y1 = draw_labeled_list(
            ddraw,
            "GT:",
            gt_words,
            x0=6,
            y0=pad,
            max_w=img.width - 6,
            line_h=line_h,
            font=font,
            color_func=lambda _w: (0, 0, 0),
            measure_only=True,
        )
        lines += l1
        l2, _y2 = draw_labeled_list(
            ddraw,
            "Predictions:",
            pred_words,
            x0=6,
            y0=pad + l1 * line_h,
            max_w=img.width - 6,
            line_h=line_h,
            font=font,
            color_func=lambda _w: (0, 0, 0),
            measure_only=True,
        )
        lines += l2
        text_h = pad * 2 + lines * line_h

        canvas = Image.new("RGB", (img.width, img.height + text_h), color="white")
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)

        # Draw GT line (black)
        draw_labeled_list(
            draw,
            "GT:",
            gt_words,
            x0=6,
            y0=img.height + pad,
            max_w=img.width - 6,
            line_h=line_h,
            font=font,
            color_func=lambda _w: (0, 0, 0),
        )

        gt_set = set(gt_words)

        # Draw Predictions line (words not in GT -> red)
        draw_labeled_list(
            draw,
            "Predictions:",
            pred_words,
            x0=6,
            y0=img.height + pad + line_h,
            max_w=img.width - 6,
            line_h=line_h,
            font=font,
            color_func=lambda w: (220, 0, 0) if w not in gt_set else (0, 0, 0),
        )

        out_name = f"{os.path.splitext(os.path.basename(fn))[0]}_attrs.png"
        canvas.save(os.path.join(out_dir, out_name))

    print("saved", out_dir)


if __name__ == "__main__":
    main()
