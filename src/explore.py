"""Week 1 exploration plots for the slides.

Makes, for one noise level:
  1. results/<snr>_id00_normal_vs_abnormal.png   log-mel spectrograms side by side
  2. results/<snr>_id00_vs_id02_normal.png        two machines, both normal
  3. results/<snr>_mean_spectrum.png              mean dB per mel band, all four groups
                                                  (median and 25-75% band over many clips)

Usage:
    python src/explore.py               # 6dB
    python src/explore.py --snr -6dB
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import audio_config as C
from features import load_wav, log_mel

ROOT = Path(__file__).resolve().parents[1]


def first_wav(folder: Path) -> Path:
    wavs = sorted(folder.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No wav files in {folder}. Run src/check_data.py first.")
    return wavs[0]


def band_profile(folder: Path, n: int) -> np.ndarray:
    """Per-clip mean dB per mel band for the first n clips, shape (n, N_MELS)."""
    wavs = sorted(folder.glob("*.wav"))[:n]
    return np.stack([log_mel(load_wav(w)).mean(axis=1) for w in wavs])


def spec_pair(paths, titles, out: Path, suptitle: str):
    specs = [log_mel(load_wav(p)) for p in paths]
    vmin = min(np.percentile(s, 1) for s in specs)
    vmax = max(s.max() for s in specs)
    fig, axes = plt.subplots(1, len(specs), figsize=(6 * len(specs), 3.6), sharey=True,
                             constrained_layout=True)
    for ax, s, t, p in zip(axes, specs, titles, paths):
        dur = len(load_wav(p)) / C.SR
        im = ax.imshow(s, origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
                       cmap="magma", extent=[0, dur, 0, C.N_MELS])
        ax.set_title(f"{t}\n{p.name}", fontsize=10)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("mel band")
    fig.colorbar(im, ax=axes, label="dB (ref = 1.0)")
    fig.suptitle(suptitle)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "data", type=Path)
    ap.add_argument("--snr", default="6dB", choices=["6dB", "0dB", "-6dB"])
    ap.add_argument("--a", default="id_00", help="first machine id")
    ap.add_argument("--b", default="id_02", help="second machine id")
    ap.add_argument("--n", type=int, default=50, help="clips per group for the mean spectrum")
    ap.add_argument("--out", default=ROOT / "results", type=Path)
    args = ap.parse_args()

    fan = args.data / args.snr / "fan"
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.snr}_{args.a.replace('_', '')}"

    spec_pair([first_wav(fan / args.a / "normal"), first_wav(fan / args.a / "abnormal")],
              ["normal", "abnormal"], args.out / f"{tag}_normal_vs_abnormal.png",
              f"Fan {args.a}, {args.snr}: normal vs abnormal")

    spec_pair([first_wav(fan / args.a / "normal"), first_wav(fan / args.b / "normal")],
              [f"{args.a} normal", f"{args.b} normal"],
              args.out / f"{args.snr}_{args.a.replace('_', '')}_vs_{args.b.replace('_', '')}_normal.png",
              f"Fan, {args.snr}: two machines, both normal")

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bands = np.arange(C.N_MELS)
    for mid, label, color, ls in [(args.a, "normal", "C0", "-"), (args.a, "abnormal", "C3", "-"),
                                  (args.b, "normal", "C0", "--"), (args.b, "abnormal", "C3", "--")]:
        prof = band_profile(fan / mid / label, args.n)
        q25, med, q75 = np.percentile(prof, [25, 50, 75], axis=0)
        ax.plot(bands, med, color=color, ls=ls, label=f"{mid} {label} (n={len(prof)})")
        ax.fill_between(bands, q25, q75, color=color, alpha=0.12)
    ax.set_xlabel("mel band")
    ax.set_ylabel("mean dB per clip (median, 25-75% band)")
    ax.set_title(f"Fan, {args.snr}: average spectrum per group")
    ax.legend()
    out = args.out / f"{args.snr}_mean_spectrum.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
