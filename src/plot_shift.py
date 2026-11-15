"""Week 3 figure: what noise shift and machine shift do to the detector.

Reads results/eval_fan_v1.csv (python src/evaluate.py --model fan_v1) and
results/shift_machine.csv (python src/shift_machine.py); writes results/week3_shift.png.
All bars: CNN embedding + diagonal Gaussian, mean over the four machine IDs.

Usage:
    python src/plot_shift.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHOD = "emb/diag"
BARS = [("auc", "ROC-AUC", "C0"), ("fpr", "false-alarm rate", "C3"), ("tpr", "detection rate", "C2")]


def panel(ax, table: pd.DataFrame, labels: list[str], title: str):
    x = np.arange(len(table))
    for i, (col, name, color) in enumerate(BARS):
        b = ax.bar(x + (i - 1) * 0.27, table[col], 0.27, label=name, color=color)
        ax.bar_label(b, fmt="%.2f", fontsize=8, padding=1)
    ax.axhline(0.05, color="C3", ls=":", lw=1)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=11)


def main():
    res = ROOT / "results"
    noise = pd.read_csv(res / "eval_fan_v1.csv")
    noise = noise[noise.method == METHOD].groupby("condition")[["auc", "fpr", "tpr"]].mean()
    conds = ["6dB>6dB", "6dB>0dB", "6dB>-6dB", "0dB>0dB", "-6dB>-6dB"]
    mach = pd.read_csv(res / "shift_machine.csv")
    mach = mach[(mach.method == METHOD) & (mach.snr == "6dB")]
    mach = mach.groupby("scenario")[["auc", "fpr", "tpr"]].mean()
    scen = ["seen", "single", "pooled", "refit"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, constrained_layout=True)
    panel(a, noise.loc[conds],
          ["fit 6 dB\ntest 6 dB", "fit 6 dB\ntest 0 dB", "fit 6 dB\ntest -6 dB",
           "refit\n0 dB", "refit\n-6 dB"],
          "Noise shift (same machine, CNN trained on 6 + 0 dB)")
    panel(b, mach.loc[scen],
          ["machine seen\nin CNN training", "unseen machine,\nGaussian from\none other fan",
           "unseen machine,\nGaussian from\n3 other fans", "unseen machine,\nrefit on it"],
          "Machine shift at 6 dB (CNN never heard the test fan)")
    a.set_ylabel("mean over the four fans")
    a.legend(loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.2))
    fig.suptitle("Without adaptation the alarm threshold is useless; the dotted line is the "
                 "5% false-alarm target", fontsize=11)
    out = res / "week3_shift.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
