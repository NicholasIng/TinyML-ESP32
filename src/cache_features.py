"""Pre-compute 1 s log-mel windows for every clip and save them as .npy.

Output (git-ignored): data/cache/<snr>/<id>_<label>.npy   shape (clips, 10, 64, 30) float32
                      data/cache/<snr>/<id>_<label>.txt   one wav file name per clip (same order)

Usage:
    python src/cache_features.py               # 6dB
    python src/cache_features.py --snr=-6dB
"""

import argparse
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

import audio_config as C
from features import load_wav, windows

ROOT = Path(__file__).resolve().parents[1]
WINS_PER_CLIP = 10   # MIMII clips are 10 s


def clip_windows(path: Path) -> np.ndarray:
    w = windows(load_wav(path))
    if len(w) < WINS_PER_CLIP:
        raise ValueError(f"{path}: only {len(w)} windows, expected {WINS_PER_CLIP}")
    return w[:WINS_PER_CLIP]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "data", type=Path)
    ap.add_argument("--snr", default="6dB", choices=["6dB", "0dB", "-6dB"])
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    fan = args.data / args.snr / "fan"
    out = args.data / "cache" / args.snr
    out.mkdir(parents=True, exist_ok=True)
    for d in sorted(p for p in fan.iterdir() if p.is_dir()):
        for label in ("normal", "abnormal"):
            wavs = sorted((d / label).glob("*.wav"))
            x = np.stack(Parallel(n_jobs=args.jobs)(delayed(clip_windows)(w) for w in wavs))
            assert x.shape[1:] == (WINS_PER_CLIP, C.N_MELS, C.FRAMES_PER_WIN), x.shape
            np.save(out / f"{d.name}_{label}.npy", x)
            (out / f"{d.name}_{label}.txt").write_text("\n".join(w.name for w in wavs))
            print(f"{args.snr} {d.name} {label:8s} {x.shape}")


if __name__ == "__main__":
    main()
