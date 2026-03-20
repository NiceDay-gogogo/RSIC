#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_ablation_tables(md_text):
    lines = md_text.splitlines()
    tables = {}
    in_ablation = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("## 消融实验"):
            in_ablation = True
            i += 1
            continue
        if not in_ablation:
            i += 1
            continue
        if line.startswith("在") and line.endswith("数据集上面的结果"):
            m = re.search(r"在(.+?)数据集", line)
            dataset = m.group(1) if m else line
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                break
            header = [c.strip() for c in lines[i].split("\t")]
            i += 1
            rows = []
            while i < len(lines) and lines[i].strip():
                cols = [c.strip() for c in lines[i].split("\t")]
                if len(cols) < 2:
                    break
                rows.append(cols)
                i += 1
            tables[dataset] = {"header": header, "rows": rows}
        else:
            i += 1
    return tables


def normalize_header(name):
    key = name.strip().upper().replace("-", "_").replace(" ", "")
    if key in {"ROUGEL", "ROUGE_L"}:
        return "ROUGE_L"
    if key in {"CIDER"}:
        return "CIDEr"
    if key in {"BLEU1", "BLEU_1"}:
        return "BLEU-1"
    if key in {"BLEU4", "BLEU_4"}:
        return "BLEU-4"
    if key in {"METEOR"}:
        return "METEOR"
    return name.strip()


def to_float(val):
    v = val.strip()
    if not v or v == "-":
        return math.nan
    try:
        return float(v)
    except ValueError:
        return math.nan


def normalize_toggle(val):
    v = val.strip().lower()
    if v in {"✔", "√", "yes", "y", "1"}:
        return "✓"
    if v in {"×", "x", "no", "n", "0"}:
        return "×"
    return val.strip()


def plot_dataset(dataset, table, out_dir):
    header = table["header"]
    rows = table["rows"]

    header_map = {}
    for idx, h in enumerate(header):
        header_map[normalize_header(h)] = idx

    metrics = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE_L", "CIDEr"]
    metric_labels = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE_L", "CIDEr(0-5)"]
    metric_indices = []
    for metric in metrics:
        if metric not in header_map:
            raise ValueError(f"Missing metric {metric} in {dataset} header: {header}")
        metric_indices.append(header_map[metric])

    attr_idx = header_map.get("ATTR", 0)
    mla_idx = header_map.get("MLA", 1)

    variant_labels = []
    values = []
    for r in rows:
        attr = normalize_toggle(r[attr_idx])
        mla = normalize_toggle(r[mla_idx])
        variant_labels.append(f"attr {attr} / MLA {mla}")
        values.append([to_float(r[idx]) for idx in metric_indices])
    values = np.array(values, dtype=float)

    title_map = {
        "UCM": "UCM-Captions (Ablation)",
        "Sydney": "Sydney-Captions (Ablation)",
        "RSICD": "RSICD (Ablation)",
        "NWPU": "NWPU-Captions (Ablation)",
    }
    title = title_map.get(dataset, f"{dataset} (Ablation)")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(metrics))
    n_variants = len(variant_labels)
    bar_width = 0.18
    # Vivid palette similar to the provided sample
    colors = ["#4f7df3", "#5cc0b3", "#f1b64d", "#e06a5f"]

    for i, (label, color) in enumerate(zip(variant_labels, colors)):
        ax.bar(
            x + i * bar_width,
            values[i, :],
            width=bar_width,
            label=label,
            color=color,
        )

    ax.set_title(title, fontsize=14)
    ax.set_xticks(x + bar_width * (n_variants - 1) / 2)
    ax.set_xticklabels(metric_labels, rotation=0, ha="center")
    ax.yaxis.grid(True, linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset.lower()}_ablation_bars.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate ablation bar charts from dataplot.md")
    parser.add_argument("--input", default="dataplot.md", help="Path to dataplot.md")
    parser.add_argument("--out-dir", default="figures/dataplot_bars", help="Output directory for plots")
    args = parser.parse_args()

    md_text = Path(args.input).read_text()
    tables = parse_ablation_tables(md_text)

    out_dir = Path(args.out_dir)
    outputs = []
    for dataset in ["RSICD", "UCM", "Sydney", "NWPU"]:
        if dataset not in tables:
            raise ValueError(f"Missing ablation table for {dataset}")
        outputs.append(plot_dataset(dataset, tables[dataset], out_dir))

    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
