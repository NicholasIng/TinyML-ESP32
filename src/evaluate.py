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
    python src/evaluate.py --model base_6dB --conditions="6dB>6dB,6dB>0dB,6dB>-6dB"
    python src/evaluate.py --model fan_v1_int8      # int8 TFLite model (convert_tflite.py)
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

DEFAULT_CONDITIONS = ["6dB>6dB", "0dB>0dB", "-6dB>-6dB", "6dB>0dB", "6dB>-6dB"]
METHODS = ["mel64/diag", "mel64/shrink", "emb/diag", "emb/shrink"]
TARGET_FPR = 0.05   # false-alarm rate the fixed threshold aims at in the fit condition


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


def feature_bank(model_name: str):
    """Returns feats(snr, mid, part) -> {"mel64": (n, 64), "emb": (n, 32)}, n = clips*10.
    Per-window features of one group, computed once and then cached."""
    tfl = ROOT / "models" / f"{model_name}.tflite"
    if tfl.exists():   # int8 model from convert_tflite.py, run as the ESP32 runs it
        from tflite_runner import Int8Embedder
        net = Int8Embedder(tfl)
    else:
        net = keras.models.load_model(ROOT / "models" / f"{model_name}.keras")
    kw = {} if tfl.exists() else {"batch_size": 256, "verbose": 0}
    cache = {}

    def feats(snr, mid, part):
        if (snr, mid, part) not in cache:
            w = load_split(snr, mid)[part].reshape(-1, C.N_MELS, C.FRAMES_PER_WIN)
            cache[snr, mid, part] = {"mel64": w.mean(axis=2),
                                     "emb": net.predict(w[..., None], **kw)}
        return cache[snr, mid, part]
    return feats


METRICS = ["auc", "pauc", "fpr", "tpr"]


def score_methods(fit: dict, test_normal: dict, test_abnormal: dict) -> list[tuple]:
    """One row (method, auc, pauc, fpr, tpr) per method. `fit` = normal windows the Gaussian
    is fitted on; the alarm threshold is also set from them, as a device would: the
    (1 - TARGET_FPR) quantile of their clip scores."""
    rows = []
    for method in METHODS:
        feat, kind = method.split("/")
        g = Gaussian(fit[feat], kind)
        clip = lambda f: g.score(f[feat]).reshape(-1, 10).mean(axis=1)
        sn, sa = clip(test_normal), clip(test_abnormal)
        thr = np.quantile(clip(fit), 1 - TARGET_FPR)
        rows.append((method, *aucs(sn, sa), (sn > thr).mean(), (sa > thr).mean()))
    return rows


def evaluate(model_name: str, conditions=DEFAULT_CONDITIONS) -> pd.DataFrame:
    """Long table: condition, id, method, auc, pauc, fpr, tpr."""
    feats = feature_bank(model_name)
    rows = []
    for cond in conditions:
        fit_snr, test_snr = cond.split(">")
        for mid in IDS:
            for r in score_methods(feats(fit_snr, mid, "train"), feats(test_snr, mid, "test_normal"),
                                   feats(test_snr, mid, "test_abnormal")):
                rows.append((cond, mid, *r))
    return pd.DataFrame(rows, columns=["condition", "id", "method", *METRICS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="name used in train.py --name")
    ap.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS),
                    help="comma-separated fit>test pairs; write it with = (see Usage)")
    args = ap.parse_args()
    args.conditions = args.conditions.split(",")

    df = evaluate(args.model, args.conditions)
    titles = {"auc": "ROC-AUC", "pauc": "pAUC (FPR <= 0.1)",
              "fpr": f"false-alarm rate at the fixed threshold (target {TARGET_FPR})",
              "tpr": "detection rate at the fixed threshold"}
    for metric in ("auc", "pauc", "fpr", "tpr"):
        t = df.pivot_table(index="method", columns="condition", values=metric)
        t = t.loc[METHODS, args.conditions]
        print(f"\nmean {titles[metric]} over the four machine IDs, model = {args.model}")
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
