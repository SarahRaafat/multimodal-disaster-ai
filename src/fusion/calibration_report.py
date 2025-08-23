"""Produce reliability curves and basic calibration stats for text/image/fused models.

Saves a PNG with the curves and prints ECE/Brier numbers.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def reliability_curve(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    accs, confs, sizes = [], [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        confs.append(float(p[m].mean()))
        accs.append(float(((p[m] >= 0.5).astype(int) == y[m]).mean()))
        sizes.append(int(m.sum()))
    return np.array(confs), np.array(accs), np.array(sizes)


def plot_reliability(ax, y, p, title):
    conf, acc, sizes = reliability_curve(y, p, bins=10)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.plot(conf, acc, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str, default="artifacts/fusion")
    args = ap.parse_args()
    d = Path(args.in_dir)

    y = np.load(d / "_labels_test.npy")
    p_txt = np.load(d / "_probs_text_test.npy")
    p_img = np.load(d / "_probs_image_test.npy")
    p_fuse = np.load(d / "_probs_fused_test.npy")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=140)
    plot_reliability(axes[0], y, p_txt, "Text (calibrated)")
    plot_reliability(axes[1], y, p_img, "Image (calibrated)")
    plot_reliability(axes[2], y, p_fuse, "Fused")
    fig.tight_layout()
    fig.savefig(d / "reliability_curves.png")
    print(f"[saved] {d/'reliability_curves.png'}")

    # also dump a quick table with ECE/Brier from previous step if present
    cal_json = d / "calibration_test.json"
    if cal_json.exists():
        print(cal_json.read_text())


if __name__ == "__main__":
    main()
