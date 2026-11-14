"""Cached features + the fixed train/val/test split, shared by train.py and evaluate.py.

Split per machine ID (same clip indices at every noise level, because the three MIMII
noise levels contain the same recordings under the same file names):
  test  = all abnormal clips + the same number of normal clips (random, seed 0)
  val   = 10% of the remaining normal clips (only for watching the classifier train)
  train = the rest of the normal clips
Abnormal audio is never used for training or for fitting the anomaly scorer.
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IDS = ["id_00", "id_02", "id_04", "id_06"]
SPLIT_SEED = 0
VAL_FRACTION = 0.1


def load(snr: str, mid: str, label: str, data: Path = ROOT / "data") -> np.ndarray:
    """All clips of one group, shape (clips, 10, 64, 30). Run cache_features.py first."""
    f = data / "cache" / snr / f"{mid}_{label}.npy"
    if not f.exists():
        raise SystemExit(f"{f} missing. Run: python src/cache_features.py --snr={snr}")
    return np.load(f)


def split_indices(n_normal: int, n_abnormal: int, mid: str) -> dict[str, np.ndarray]:
    """Clip indices into the (sorted-by-file-name) normal folder of one machine ID."""
    rng = np.random.default_rng([SPLIT_SEED, IDS.index(mid)])
    perm = rng.permutation(n_normal)
    n_val = int(round(VAL_FRACTION * (n_normal - n_abnormal)))
    return {"test": np.sort(perm[:n_abnormal]),
            "val": np.sort(perm[n_abnormal:n_abnormal + n_val]),
            "train": np.sort(perm[n_abnormal + n_val:])}


def load_split(snr: str, mid: str, data: Path = ROOT / "data") -> dict[str, np.ndarray]:
    """Dict with train / val / test_normal / test_abnormal, each (clips, 10, 64, 30)."""
    normal, abnormal = load(snr, mid, "normal", data), load(snr, mid, "abnormal", data)
    idx = split_indices(len(normal), len(abnormal), mid)
    return {"train": normal[idx["train"]], "val": normal[idx["val"]],
            "test_normal": normal[idx["test"]], "test_abnormal": abnormal}
