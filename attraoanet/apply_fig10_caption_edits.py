#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Apply manual caption edits to existing Fig10 spec files and re-render.

Typical workflow:
1) Edit a JSON file like figures/rsicdfig/rsicd_02_06_08_27_captions_edit.json
2) Run:
   python apply_fig10_caption_edits.py --edits figures/rsicdfig/rsicd_02_06_08_27_captions_edit.json

By default, this writes new files next to the originals:
- <name>.edited.spec.json
- <name>.edited.png

Use --inplace to overwrite the original spec/png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from make_fig10_triple_compare import make_figure
from make_rsicd_30_min_figs import build_vocab, colorize_text, invert_synonyms, load_json, norm_token


def _load_edits(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        return obj["items"]
    if isinstance(obj, list):
        return obj
    raise ValueError("Unsupported edits JSON format; expected {'items':[...]} or a list")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--edits",
        type=str,
        required=True,
        help="Path to the edited JSON (e.g., figures/rsicdfig/rsicd_02_06_08_27_captions_edit.json)",
    )
    ap.add_argument("--inplace", action="store_true", help="Overwrite original .spec.json/.png")
    ap.add_argument("--no_render", action="store_true", help="Only write spec files; do not render PNG")
    args = ap.parse_args()

    edits_path = Path(args.edits)
    items = _load_edits(edits_path)

    syn = load_json("data/attribute_synonyms_new40.json")
    syn_inv = invert_synonyms({str(k): [str(x) for x in v] for k, v in syn.items()})
    attribute_words = set(norm_token(w) for w in load_json("data/attribute_words_new40.json"))

    wrote = []
    for it in items:
        spec_path = Path(str(it["spec_path"]))
        spec = json.loads(spec_path.read_text(encoding="utf-8"))

        ours = str(it.get("ours", ""))
        baseline = str(it.get("baseline", ""))
        gt_list = it.get("gt", [])
        if not isinstance(gt_list, list) or not gt_list:
            raise ValueError(f"Missing gt list in edits item: {spec_path}")
        gt_list = [str(x) for x in gt_list]

        gt_vocab = build_vocab(gt_list)
        pred_vocab = build_vocab([ours])

        markup = {
            "our": colorize_text(ours, gt_vocab, syn_inv, attribute_words, mode="pred"),
            "baseline": colorize_text(baseline, gt_vocab, syn_inv, attribute_words, mode="pred"),
            "gt": [
                colorize_text(gt, gt_vocab, syn_inv, attribute_words, mode="gt", pred_vocab=pred_vocab)
                for gt in gt_list
            ],
        }

        spec.setdefault("samples", [{}])
        if not spec["samples"]:
            spec["samples"] = [{}]
        spec["samples"][0].setdefault("image_id", int(it.get("image_id")))
        if "scenario" in it:
            spec["samples"][0]["scenario"] = str(it["scenario"])
        spec["samples"][0]["markup"] = markup

        if args.inplace:
            out_spec = spec_path
            out_png = Path(str(it.get("png_path", str(spec_path).replace(".spec.json", ".png"))))
        else:
            out_spec = Path(str(spec_path).replace(".spec.json", ".edited.spec.json"))
            out_png = Path(str(out_spec).replace(".spec.json", ".png"))

        out_spec.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        wrote.append((out_spec, out_png))

        if not args.no_render:
            make_figure(str(out_spec), str(out_png))

    print(f"[OK] wrote {len(wrote)} spec(s)")
    for s, p in wrote:
        print(" -", s)
        if not args.no_render:
            print("   ", p)


if __name__ == "__main__":
    main()
