#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


ITER_LOSS_RE = re.compile(
    r"iter\s+(?P<iter>\d+).*?(?:train_)?loss\s*=\s*(?P<loss>[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)
EPOCH_LOSS_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+).*?(?:train_)?loss\s*=\s*(?P<loss>[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)
LOSS_ONLY_RE = re.compile(
    r"(?:train_)?loss\s*=\s*(?P<loss>[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)


def parse_log(path: Path):
    points = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = ITER_LOSS_RE.search(line)
            if match:
                points.append((int(match.group("iter")), float(match.group("loss"))))
                continue

            match = EPOCH_LOSS_RE.search(line)
            if match:
                points.append((int(match.group("epoch")), float(match.group("loss"))))
                continue

            match = LOSS_ONLY_RE.search(line)
            if match:
                points.append((len(points), float(match.group("loss"))))

    return points


def smooth(values, window):
    if window <= 1:
        return values
    smoothed = []
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
            smoothed.append(total / window)
        else:
            smoothed.append(total / (i + 1))
    return smoothed


def main():
    parser = argparse.ArgumentParser(
        description="Plot training loss curves from one or more log files."
    )
    parser.add_argument("logs", nargs="+", help="Path(s) to training log files.")
    parser.add_argument("-o", "--output", default="train_loss.png", help="Output image.")
    parser.add_argument("--title", default=None, help="Plot title.")
    parser.add_argument("--labels", default=None, help="Comma-separated labels.")
    parser.add_argument("--smooth", type=int, default=0, help="Moving average window.")
    args = parser.parse_args()

    labels = None
    if args.labels:
        labels = [label.strip() for label in args.labels.split(",")]

    plt.figure(figsize=(10, 5))
    for idx, log_path in enumerate(args.logs):
        path = Path(log_path)
        points = parse_log(path)
        if not points:
            print(f"[WARN] No loss points found in {path}")
            continue

        xs, ys = zip(*points)
        ys = smooth(list(ys), args.smooth)
        label = path.stem if not labels or idx >= len(labels) else labels[idx]
        plt.plot(xs, ys, label=label)

    plt.xlabel("step")
    plt.ylabel("train loss")
    if args.title:
        plt.title(args.title)
    if len(args.logs) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"[OK] Saved: {args.output}")


if __name__ == "__main__":
    main()
