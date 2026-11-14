"""Anomaly detection scores for one trained embedding model: ROC-AUC and pAUC (FPR <= 0.1).

For each machine ID, a Gaussian is fitted on NORMAL training windows only and every test
clip is scored by its Mahalanobis distance, averaged over the ten 1 s windows of the clip.

Features:  mel64 = mean dB per mel band of the window (no CNN, reference baseline)
           emb   = the 32-number embedding from the CNN
Covariance: diag   = per-dimension variance only (cheapest on the ESP32)
            shrink = Ledoit-Wolf shrinkage covariance (no plain full covariance, see rules)

Conditions are written fit>test: the noise level the Gaussian is fitted on, then the one
it is tested on. "6dB>-6dB" is a noise shift with no adaptation.

Usage:
    python src/evaluate.py --model base_6dB
    python src/evaluate.py --model base_6dB --conditions "6dB>6dB" "6dB>0dB"
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from tensorflow import keras

import audio_config as C
from dataset import IDS, ROOT, load_split

DEFAULT_CONDITIONS = ["6dB>6dB", "0dB>0dB", "-6dB>-6dB", "6dB>-6dB"]
METHODS = ["mel64/diag", "mel64/shrink", "emb/diag", "emb/shrink"]


class Gaussian:
    """Mean + inverse covariance of the normal data; score = squared Mahalanobis distance."""

    def __init__(self, x: np.ndarray, kind: str):
        self.mu = x.mean(axis=0)
        if kind == "diag":
            self.prec = np.diag(1.0 / (x.var(axis=0) + 1e-6))
        else:
            self.prec = LedoitWolf().fit(x).precision_

    def score(self, x: np.ndarray) -> np.ndarray:
        d = x - self.mu
        return np.einsum("ni,ij,nj->n", d, self.prec, d)


def aucs(normal: np.ndarray, abnormal: np.ndarray) -> tuple[float, float]:
    y = np.r_[np.zeros(len(normal)), np.ones(len(abnormal))]
    s = np.r_[normal, abnormal]
    return roc_auc_score(y, s), roc_auc_score(y, s, max_fpr=0.1)


def evaluate(model_name: str, conditions=DEFAULT_CONDITIONS) -> pd.DataFrame:
    """Long table: condition, id, method, auc, pauc."""
    net = keras.models.load_model(ROOT / "models" / f"{model_name}.keras")
    cache = {}

    def feats(snr, mid, part):
        """Per-window features of one group: {"mel64": (n, 64), "emb": (n, 32)}, n = clips*10."""
        if (snr, mid, part) not in cache:
            w = load_split(snr, mid)[part].reshape(-1, C.N_MELS, C.FRAMES_PER_WIN)
            cache[snr, mid, part] = {"mel64": w.mean(axis=2),
                                     "emb": net.predict(w[..., None], batch_size=256, verbose=0)}
        return cache[snr, mid, part]

    rows = []
    for cond in conditions:
        fit_snr, test_snr = cond.split(">")
        for mid in IDS:
            fit = feats(fit_snr, mid, "train")
            fn, fa = feats(test_snr, mid, "test_normal"), feats(test_snr, mid, "test_abnormal")
            for method in METHODS:
                feat, kind = method.split("/")
                g = Gaussian(fit[feat], kind)
                clip = lambda f: g.score(f[feat]).reshape(-1, 10).mean(axis=1)
                rows.append((cond, mid, method, *aucs(clip(fn), clip(fa))))
    return pd.DataFrame(rows, columns=["condition", "id", "method", "auc", "pauc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="name used in train.py --name")
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    args = ap.parse_args()

    df = evaluate(args.model, args.conditions)
    for metric in ("auc", "pauc"):
        t = df.pivot_table(index="method", columns="condition", values=metric)
        t = t.loc[METHODS, args.conditions]
        print(f"\nmean {metric.upper()} over the four machine IDs, model = {args.model}")
        print(t.round(3).to_string())
    first = args.conditions[0]
    t = df[df.condition == first].pivot(index="method", columns="id", values="auc").loc[METHODS]
    print(f"\nAUC per machine ID, condition {first}")
    print(t.round(3).to_string())
    out = ROOT / "results" / f"eval_{args.model}.csv"
    df.assign(model=args.model).round(4).to_csv(out, index=False)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
