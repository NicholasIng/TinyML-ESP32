"""Calibration methods for week 4, written so they port directly to C++ on the ESP32.

Everything works on plain float arrays of fixed size: D numbers per 1 s window (D = 32 for
the CNN embedding), at most 30 calibration windows. No matrix inverse, no dynamic memory:
a model is just  mu[D], var[D], thr.

A window's anomaly score is the diagonal Mahalanobis distance
    score(x) = mean_d (x[d] - mu[d])^2 / var[d]
and a clip is flagged when the mean score of its windows is above thr.

Factory (source) data gives SourceModel. On the new machine, `calibrate(method, src, xc)`
turns n verified-normal windows xc (n, D) into the (mu, var, thr) to use there.
Only `calibrate` and `score` run on the device; `fit_source` runs on the PC and its
outputs (mu, var, thr, k and two small threshold tables) are constants in the firmware.
"""

from dataclasses import dataclass, field

import numpy as np

TARGET_FPR = 0.05
VAR_FLOOR = 1e-6
N0 = 30.0          # prior weight of the factory variances, in "windows" (see _mixed_var)
WINS = 10          # 1 s windows per MIMII clip
SECONDS = [5, 10, 20, 30]

METHODS = ["none", "thr_retune", "centroid", "centroid_cal", "centroid_var_cal"]
LABELS = {"none": "no adaptation",
          "thr_retune": "threshold re-tune",
          "centroid": "centroid shift, factory threshold",
          "centroid_cal": "centroid shift + threshold table",
          "centroid_var_cal": "centroid + variance + threshold table",
          "oracle": "full refit (all target data)"}


@dataclass
class SourceModel:
    mu: np.ndarray    # (D,) mean of the factory normal windows
    var: np.ndarray   # (D,) variance around each factory machine's own mean
    thr: float        # clip threshold giving TARGET_FPR on the factory data
    k: float          # threshold rule for re-tuning: thr = mean + k * std of window scores
    # Thresholds for a model whose mean (and variances) come from only n seconds of audio,
    # found by rehearsing the calibration on factory data: {method: {seconds: thr}}
    thr_table: dict = field(default_factory=dict)


def score(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Window scores, shape (n,). x is (n, D)."""
    return ((x - mu) ** 2 / var).mean(axis=1)


def draw(pool: np.ndarray, seconds: int, rng):
    """`seconds` 1 s windows of normal audio from pool (n, D): whole clips, the last one cut.
    Returns (windows, indices of the clips used)."""
    clips = pool.reshape(-1, WINS, pool.shape[1])
    pick = rng.choice(len(clips), size=int(np.ceil(seconds / WINS)), replace=False)
    return clips[pick].reshape(-1, pool.shape[1])[:seconds], pick


def _mixed_var(src_var, xc, n0):
    """Weighted mix of factory and measured variances. With few windows the factory value
    dominates; n0 is how many windows the factory value is "worth"."""
    n = len(xc)
    return (n0 * src_var + n * xc.var(axis=0)) / (n0 + n) + VAR_FLOOR


def fit_source(groups: list[np.ndarray], n0: float = N0, draws: int = 40, seed: int = 0) -> SourceModel:
    """groups: one (n_i, D) array of normal windows per factory machine (one or more).
    Variances and the threshold are measured around each machine's own mean, so they
    describe how much ONE healthy machine varies, not how much machines differ."""
    centered = [g - g.mean(axis=0) for g in groups]
    var = np.concatenate(centered).var(axis=0) + VAR_FLOOR
    zero = np.zeros_like(var)
    ws = np.concatenate([score(c, zero, var) for c in centered])
    clip = ws.reshape(-1, WINS).mean(axis=1)
    thr = float(np.quantile(clip, 1 - TARGET_FPR))
    src = SourceModel(np.concatenate(groups).mean(axis=0), var, thr,
                      float((thr - ws.mean()) / ws.std()))

    # Rehearse the calibration on factory data: calibrate on n seconds of one machine,
    # score that machine's OTHER clips, and take the threshold that gives TARGET_FPR.
    # A mean estimated from a few seconds is a little off, so this threshold is higher
    # than src.thr; the table tells the device by how much.
    rng = np.random.default_rng(seed)
    for method in ("centroid_cal", "centroid_var_cal") if draws else ():
        src.thr_table[method] = {}
        for sec in SECONDS:
            scores = []
            for g in groups:
                n_clips = len(g) // WINS
                for _ in range(draws):
                    xc, used = draw(g, sec, rng)
                    mu = xc.mean(axis=0)
                    v = var if method == "centroid_cal" else _mixed_var(var, xc, n0)
                    rest = np.delete(np.arange(n_clips), used)
                    rest = rng.choice(rest, size=min(100, len(rest)), replace=False)
                    x = g.reshape(n_clips, WINS, -1)[rest].reshape(-1, g.shape[1])
                    scores.append(score(x, mu, v).reshape(-1, WINS).mean(axis=1))
            src.thr_table[method][sec] = float(np.quantile(np.concatenate(scores), 1 - TARGET_FPR))
    return src


def calibrate(method: str, src: SourceModel, xc: np.ndarray, n0: float = N0):
    """Returns (mu, var, thr) for the new machine from n calibration windows xc (n, D)."""
    n = len(xc)
    if method == "none":
        return src.mu, src.var, src.thr
    if method == "thr_retune":
        # Keep the factory Gaussian, move only the alarm threshold.
        s = score(xc, src.mu, src.var)
        return src.mu, src.var, s.mean() + src.k * s.std()

    mu = xc.mean(axis=0)
    if method == "centroid":
        # New mean, factory variances, factory threshold.
        return mu, src.var, src.thr
    if method == "centroid_cal":
        # Same, but with the threshold the factory rehearsal found for n seconds.
        return mu, src.var, src.thr_table[method][n]
    if method == "centroid_var_cal":
        return mu, _mixed_var(src.var, xc, n0), src.thr_table[method][n]
    raise ValueError(method)
