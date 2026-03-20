#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


Color = Tuple[int, int, int]


PALETTE: Dict[str, Color] = {
    "black": (0, 0, 0),
    "green": (44, 160, 44),
    "orange": (255, 127, 14),
    "blue": (31, 119, 180),
    "red": (214, 39, 40),
    "gray": (120, 120, 120),
}


def _safe_get_palette(name: str) -> Color:
    return PALETTE.get(name, PALETTE["black"])


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


def load_coco_captions(path: str) -> Dict[int, List[str]]:
    coco = load_json(path)
    id2caps: Dict[int, List[str]] = {}
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        cap = str(ann.get("caption", ""))
        if cap:
            id2caps.setdefault(image_id, []).append(cap)
    return id2caps


def load_coco_image_filenames(path: str) -> Dict[int, str]:
    coco = load_json(path)
    out: Dict[int, str] = {}
    for im in coco.get("images", []):
        if "id" not in im:
            continue
        image_id = int(im["id"])
        file_name = str(im.get("file_name", ""))
        if file_name:
            out[image_id] = file_name
    return out


@dataclass
class Segment:
    text: str
    color: str


def segments_from_plain(text: str, color: str = "black") -> List[Segment]:
    return [Segment(text=text, color=color)]


def segments_from_spec(spec_list) -> List[Segment]:
    segs: List[Segment] = []
    for seg in spec_list:
        segs.append(Segment(text=str(seg.get("text", "")), color=str(seg.get("color", "black"))))
    return segs


def pick_font(size: int):
    from PIL import ImageFont

    # Match the user's reference renderer: prefer DejaVuSans / DejaVuSansMono.
    # The actual weight comes from the font file (regular, not bold).
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


def pick_bold_font(size: int):
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def measure_text(draw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def wrap_segments(draw, segments: List[Segment], font, max_width: int) -> List[List[Segment]]:
    lines: List[List[Segment]] = [[]]
    current_width = 0

    for seg in segments:
        if not seg.text:
            continue

        # Keep whitespace with previous token if possible; but split long runs by characters.
        remaining = seg.text
        while remaining:
            w = measure_text(draw, remaining, font)
            if current_width + w <= max_width:
                lines[-1].append(Segment(text=remaining, color=seg.color))
                current_width += w
                remaining = ""
                continue

            # Try split by last space within fit
            split_at = None
            for i in range(len(remaining), 0, -1):
                if remaining[i - 1].isspace():
                    candidate = remaining[:i]
                    if current_width + measure_text(draw, candidate, font) <= max_width:
                        split_at = i
                        break

            if split_at is None:
                # Hard split by characters
                lo, hi = 1, len(remaining)
                best = 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    candidate = remaining[:mid]
                    if current_width + measure_text(draw, candidate, font) <= max_width:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                split_at = best

            head = remaining[:split_at]
            tail = remaining[split_at:]
            if head:
                lines[-1].append(Segment(text=head, color=seg.color))
            lines.append([])
            current_width = 0
            remaining = tail.lstrip("\n")

    # Trim empty last line
    if lines and not any(s.text for s in lines[-1]):
        lines.pop()

    return lines


def draw_segmented_text(draw, x: int, y: int, segments: List[Segment], font, max_width: int, line_gap: int) -> int:
    lines = wrap_segments(draw, segments, font, max_width)
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
    cursor_y = y
    for i, line in enumerate(lines):
        cursor_x = x
        for seg in line:
            color = _safe_get_palette(seg.color)
            draw.text((cursor_x, cursor_y), seg.text, font=font, fill=color)
            cursor_x += measure_text(draw, seg.text, font)
        cursor_y += line_height
        if i < len(lines) - 1:
            cursor_y += line_gap
    return cursor_y


def _infer_image_roots(spec: dict) -> List[str]:
    roots: List[str] = []
    if spec.get("image_root"):
        roots.append(str(spec["image_root"]))

    dataset = str(spec.get("dataset", "")).strip()
    if dataset:
        roots.extend(
            [
                os.path.join("data", dataset, "images"),
                os.path.join("data", dataset, "Images"),
                os.path.join("data", dataset, "imgs"),
                os.path.join("data", dataset, "Imgs"),
            ]
        )

    # Common paths in this repo
    roots.extend(
        [
            os.path.join("data", "UCM", "images"),
            os.path.join("data", "Sydney", "imgs"),
            os.path.join("data", "RSICD", "images"),
            os.path.join("data", "RSICD_images"),
        ]
    )

    # De-dup while preserving order
    seen = set()
    out = []
    for r in roots:
        r = os.path.normpath(r)
        if r not in seen:
            out.append(r)
            seen.add(r)
    return out


def find_image_path(file_name: str, roots: List[str]) -> str:
    file_name = file_name.lstrip("/\\")
    for r in roots:
        p = os.path.join(r, file_name)
        if os.path.exists(p):
            return p
    return ""


def make_figure(spec_path: str, out_path: str, dpi: int = 200):
    from PIL import Image, ImageDraw

    spec = load_json(spec_path)

    show_title = bool(spec.get("show_title", False))
    show_legend = bool(spec.get("show_legend", False))
    show_frame = bool(spec.get("show_frame", False))
    show_header = bool(spec.get("show_header", False))
    show_labels = bool(spec.get("show_labels", False))
    repeat_gt_label = bool(spec.get("repeat_gt_label", False))
    show_image_border = bool(spec.get("show_image_border", False))
    show_missing_image_text = bool(spec.get("show_missing_image_text", False))
    gt_count = int(spec.get("gt_count", 5))

    label_style = str(spec.get("label_style", "fig10")).strip().lower()

    def get_labels(gt_n: int) -> Tuple[str, str, List[str]]:
        if label_style in {"fig10", "paper", "ours_gt"}:
            return "Ours:", "Baseline:", [f"GT{i+1}:" for i in range(gt_n)]
        return "Predict:", "Baseline:", ["Ground truth:" for _ in range(gt_n)]

    ours_label, baseline_label, gt_labels = get_labels(gt_count)

    our_preds = load_preds(spec["our_predictions"])
    base_preds = load_preds(spec["baseline_predictions"])
    gt_caps = load_coco_captions(spec["gt_coco"])
    id2file = load_coco_image_filenames(spec["gt_coco"])
    image_roots = _infer_image_roots(spec)

    samples = spec.get("samples", [])

    # Layout (left image, right text)
    width = int(spec.get("width", 1700))
    margin = int(spec.get("margin", 50))
    thumb = int(spec.get("thumb_size", 240))
    gap = int(spec.get("thumb_gap", 28))
    right_w = width - margin * 2 - thumb - gap

    # Match the reference renderer's font weight (regular) and typical sizes.
    # Match the reference renderer: fixed regular font sizes.
    title_font = pick_font(22)
    header_font = pick_font(22)
    label_font = pick_font(18)
    body_font = pick_font(18)
    small_font = pick_font(18)

    # First pass: estimate height
    tmp = Image.new("RGB", (width, 200), (255, 255, 255))
    tmp_draw = ImageDraw.Draw(tmp)

    y = margin
    title = str(spec.get("title", "Fig.10  UCM 三方句子对比展示"))
    if show_title:
        y += 42

    block_gap = int(spec.get("block_gap", 26))
    # Match the reference renderer's default line spacing.
    line_gap = int(spec.get("line_gap", 6))
    # Extra whitespace between entries (Ours/Baseline/GT*). This is distinct
    # from `line_gap`, which applies only between wrapped lines within one entry.
    entry_gap = int(spec.get("entry_gap", 12))
    inner_pad = int(spec.get("inner_pad", 16))

    # Keep the right text block within the thumbnail box and align vertically.
    align_text_to_thumb = bool(spec.get("align_text_to_thumb", True))

    def _line_h(font) -> int:
        b = font.getbbox("Ag")
        return int(b[3] - b[1])

    def _lines_h(n_lines: int, font, gap: int) -> int:
        n = max(1, int(n_lines))
        lh = _line_h(font)
        return n * lh + max(0, n - 1) * int(gap)

    def est_text_height(label: str, segs: List[Segment], max_w: int) -> int:
        if show_labels and label:
            label_w = measure_text(tmp_draw, label, label_font)
            avail = max(50, max_w - label_w - 8)
        else:
            avail = max_w
        lines = wrap_segments(tmp_draw, segs, body_font, avail)
        return _lines_h(len(lines), body_font, line_gap)

    def est_block_height(sample) -> int:
        image_id = int(sample["image_id"])
        markup = sample.get("markup", {})

        our_text = our_preds.get(image_id, "")
        base_text = base_preds.get(image_id, "")
        gts = gt_caps.get(image_id, [])
        gt_pick = sample.get("gt_pick")
        if isinstance(gt_pick, list) and gt_pick:
            picked = [gts[i] for i in gt_pick if 0 <= i < len(gts)]
        else:
            picked = gts[:gt_count]
        picked = picked[:gt_count]

        our_segs = segments_from_spec(markup.get("our", [])) if markup.get("our") else segments_from_plain(our_text)
        base_segs = (
            segments_from_spec(markup.get("baseline", [])) if markup.get("baseline") else segments_from_plain(base_text)
        )

        gt_marked = markup.get("gt")
        gt_lines: List[List[Segment]] = []
        for i in range(gt_count):
            if gt_marked and i < len(gt_marked):
                gt_lines.append(segments_from_spec(gt_marked[i]))
            elif i < len(picked):
                gt_lines.append(segments_from_plain(picked[i]))
            else:
                gt_lines.append(segments_from_plain(""))

        entries: List[Tuple[str, List[Segment]]] = []
        if show_labels:
            entries = [(ours_label, our_segs), (baseline_label, base_segs)]

            if label_style in {"fig10", "paper", "ours_gt"}:
                for i, gt in enumerate(gt_lines):
                    lab = gt_labels[i] if i < len(gt_labels) else f"GT{i+1}:"
                    entries.append((lab, gt))
            else:
                gt_label = gt_labels[0]
                if repeat_gt_label:
                    for gt in gt_lines:
                        entries.append((gt_label, gt))
                else:
                    entries.append((gt_label, gt_lines[0] if gt_lines else segments_from_plain("")))
                    for gt in gt_lines[1:]:
                        entries.append(("", gt))
        else:
            entries = [("", our_segs), ("", base_segs)]
            for gt in gt_lines:
                entries.append(("", gt))

        h = 0
        if show_header:
            h += _line_h(header_font) + 10

        max_w = right_w - inner_pad * 2
        for i, (lab, segs) in enumerate(entries):
            h += est_text_height(lab, segs, max_w)
            if i < len(entries) - 1:
                h += entry_gap

        h = max(h, thumb)
        h += inner_pad * 2
        return h

    total_h = margin
    if show_title:
        total_h += 60
    for s in samples:
        total_h += est_block_height(s) + block_gap

    if show_legend:
        total_h += 80

    img = Image.new("RGB", (width, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    y = margin
    if show_title:
        draw.text((margin, y), title, font=title_font, fill=PALETTE["black"])
        y += 48

        subtitle = f"Dataset: {spec.get('dataset','')}  /  split: {spec.get('split','')}"
        draw.text((margin, y), subtitle, font=small_font, fill=PALETTE["gray"])
        y += 26

    def draw_labeled_line(x: int, y: int, label: str, segs: List[Segment], max_w: int) -> int:
        draw.text((x, y), label, font=label_font, fill=PALETTE["black"])
        label_w = measure_text(draw, label, label_font)
        start_x = x + label_w + 8
        avail = max(50, max_w - label_w - 8)
        return draw_segmented_text(draw, start_x, y, segs, body_font, avail, line_gap)

    def draw_indented_line(x: int, y: int, indent_px: int, segs: List[Segment], max_w: int) -> int:
        start_x = x + indent_px
        avail = max(50, max_w - indent_px)
        return draw_segmented_text(draw, start_x, y, segs, body_font, avail, line_gap)

    def wrap_for_entry(draw_obj, label: str, segs: List[Segment], lab_font, txt_font, max_w: int) -> int:
        lab = label or ""
        if lab:
            lw = measure_text(draw_obj, lab, lab_font)
            avail = max(50, max_w - lw - 8)
        else:
            avail = max_w
        lines = wrap_segments(draw_obj, segs, txt_font, avail)
        return max(1, len(lines))

    def choose_layout(
        draw_obj,
        entries: List[Tuple[str, List[Segment]]],
        max_w: int,
        target_h: int,
        base_gap: int,
        entry_gap_px: int,
    ):
        # Match draw_segmented_text: only add gap BETWEEN lines (not after the last line).
        # Strategy: use as much gap as possible (up to base_gap) to fill available height.
        # Keep the search range aligned with the reference renderer (base size 18).
        for font_size in [18, 17, 16, 15]:
            lab_f = pick_font(font_size)
            txt_f = pick_font(font_size)
            lh = _line_h(txt_f)
            n_lines = 0
            for lab, segs in entries:
                n_lines += wrap_for_entry(draw_obj, lab, segs, lab_f, txt_f, max_w)

            extra_entries_h = max(0, len(entries) - 1) * int(entry_gap_px)

            if n_lines <= 0:
                continue
            if n_lines == 1:
                gap = 0
                total_h = lh + extra_entries_h
            else:
                # Use full base_gap if it fits; otherwise compute max gap that fits.
                ideal_h = n_lines * lh + (n_lines - 1) * base_gap
                if ideal_h <= target_h:
                    gap = base_gap
                    total_h = ideal_h + extra_entries_h
                else:
                    remaining = target_h - n_lines * lh
                    if remaining < 0:
                        continue
                    gap = max(0, remaining // (n_lines - 1))
                    total_h = n_lines * lh + (n_lines - 1) * gap + extra_entries_h

            if total_h <= target_h:
                return lab_f, txt_f, gap, total_h

        # Last resort: smallest font, zero gap.
        lab_f = pick_font(16)
        txt_f = pick_font(16)
        lh = _line_h(txt_f)
        n_lines = 0
        for lab, segs in entries:
            n_lines += wrap_for_entry(draw_obj, lab, segs, lab_f, txt_f, max_w)
        total_h = max(1, n_lines) * lh + max(0, len(entries) - 1) * int(entry_gap_px)
        return lab_f, txt_f, 0, total_h

    # Blocks
    for idx, sample in enumerate(samples, start=1):
        image_id = int(sample["image_id"])
        scenario = str(sample.get("scenario", f"Sample {idx}"))
        markup = sample.get("markup", {})

        block_h = est_block_height(sample)
        block_x = margin
        block_y = y

        # Block frame (optional)
        if show_frame:
            frame_w = width - margin * 2
            draw.rounded_rectangle(
                (block_x, block_y, block_x + frame_w, block_y + block_h),
                radius=12,
                outline=(220, 220, 220),
                width=2,
                fill=(255, 255, 255),
            )

        # Image thumb
        img_x = block_x + inner_pad
        img_y = block_y + inner_pad
        file_name = id2file.get(image_id, "")
        img_path = find_image_path(file_name, image_roots) if file_name else ""
        if img_path:
            try:
                im = Image.open(img_path).convert("RGB")
                im.thumbnail((thumb, thumb))
                thumb_bg = Image.new("RGB", (thumb, thumb), (255, 255, 255))
                px = (thumb - im.size[0]) // 2
                py = (thumb - im.size[1]) // 2
                thumb_bg.paste(im, (px, py))
                img.paste(thumb_bg, (img_x, img_y))
                if show_image_border:
                    draw.rectangle((img_x, img_y, img_x + thumb, img_y + thumb), outline=(200, 200, 200), width=2)
            except Exception:
                if show_image_border:
                    draw.rectangle((img_x, img_y, img_x + thumb, img_y + thumb), outline=(200, 200, 200), width=2)
                if show_missing_image_text:
                    draw.text((img_x + 10, img_y + 10), "(image load failed)", font=small_font, fill=PALETTE["gray"])
        else:
            if show_image_border:
                draw.rectangle((img_x, img_y, img_x + thumb, img_y + thumb), outline=(200, 200, 200), width=2)
            if show_missing_image_text:
                draw.text((img_x + 10, img_y + 10), "(image not found)", font=small_font, fill=PALETTE["gray"])

        # Text area
        text_x = img_x + thumb + gap
        text_y = block_y + inner_pad
        text_max_w = width - margin - inner_pad - text_x

        if show_header:
            header = f"({idx}) {scenario}"
            draw.text((text_x, text_y), header, font=header_font, fill=PALETTE["black"])
            text_y += _line_h(header_font) + 10

        our_text = our_preds.get(image_id, "")
        base_text = base_preds.get(image_id, "")
        gts = gt_caps.get(image_id, [])
        gt_pick = sample.get("gt_pick")
        if isinstance(gt_pick, list) and gt_pick:
            picked = [gts[i] for i in gt_pick if 0 <= i < len(gts)]
        else:
            picked = gts[:gt_count]
        picked = picked[:gt_count]

        our_segs = segments_from_spec(markup.get("our", [])) if markup.get("our") else segments_from_plain(our_text)
        base_segs = segments_from_spec(markup.get("baseline", [])) if markup.get("baseline") else segments_from_plain(base_text)

        gt_marked = markup.get("gt")
        gt_lines: List[List[Segment]] = []
        for i in range(gt_count):
            if gt_marked and i < len(gt_marked):
                gt_lines.append(segments_from_spec(gt_marked[i]))
            elif i < len(picked):
                gt_lines.append(segments_from_plain(picked[i]))
            else:
                gt_lines.append(segments_from_plain(""))

        if show_labels:
            entries: List[Tuple[str, List[Segment]]] = [(ours_label, our_segs), (baseline_label, base_segs)]

            if label_style in {"fig10", "paper", "ours_gt"}:
                for i, gt in enumerate(gt_lines):
                    lab = gt_labels[i] if i < len(gt_labels) else f"GT{i+1}:"
                    entries.append((lab, gt))
            else:
                gt_label = gt_labels[0]
                if repeat_gt_label:
                    for gt in gt_lines:
                        entries.append((gt_label, gt))
                else:
                    entries.append((gt_label, gt_lines[0] if gt_lines else segments_from_plain("")))
                    for gt in gt_lines[1:]:
                        entries.append(("", gt))

            # Top-align text with the thumbnail.
            if align_text_to_thumb and not show_header:
                text_y = img_y

            # Fixed fonts/gaps (match the reference renderer)
            lab_f, txt_f, gap_used = label_font, body_font, line_gap

            def draw_entry(cur_y: int, label: str, segs: List[Segment]) -> int:
                lab = label or ""
                if lab:
                    draw.text((text_x, cur_y), lab, font=lab_f, fill=PALETTE["black"])
                    lw = measure_text(draw, lab, lab_f)
                    start_x = text_x + lw + 8
                    avail = max(50, text_max_w - lw - 8)
                else:
                    start_x = text_x
                    avail = text_max_w
                return draw_segmented_text(draw, start_x, cur_y, segs, txt_f, avail, gap_used)

            for i, (lab, segs) in enumerate(entries):
                text_y = draw_entry(text_y, lab, segs)
                if i < len(entries) - 1:
                    text_y += entry_gap
        else:
            entry_gap_px = 2
            text_y = draw_segmented_text(draw, text_x, text_y, our_segs, body_font, text_max_w, line_gap)
            text_y += entry_gap_px
            text_y = draw_segmented_text(draw, text_x, text_y, base_segs, body_font, text_max_w, line_gap)
            text_y += entry_gap_px
            for gt in gt_lines:
                text_y = draw_segmented_text(draw, text_x, text_y, gt, body_font, text_max_w, line_gap)
                text_y += entry_gap_px

        y += block_h + block_gap

    # Legend (optional)
    if not show_legend:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)
        return

    draw.text((margin, y), "Legend:", font=label_font, fill=PALETTE["black"])
    y += 26

    legend_items = [
        ("green", "与 GT 一致/正确"),
        ("orange", "GT 未出现但语义正确（补充信息）"),
        ("blue", "同义/近义表达"),
        ("red", "错误或 GT 关键缺失"),
    ]

    x0 = margin
    for color, text in legend_items:
        draw.text((x0, y), "■", font=small_font, fill=_safe_get_palette(color))
        draw.text((x0 + 20, y), text, font=small_font, fill=PALETTE["black"])
        x0 += 360

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="Path to spec json")
    ap.add_argument("--out", required=True, help="Output image path")
    args = ap.parse_args()

    make_figure(args.spec, args.out)


if __name__ == "__main__":
    main()
