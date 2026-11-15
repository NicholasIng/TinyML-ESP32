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
outputs (mu, var, thr, k and a few small tables) are constants in the firmware.

Burn-in (the "burn_*" methods): a few clips cannot show how much the new room varies from
minute to minute, so the alarm threshold from the table is too low in a noisy room. After
calibration the device therefore listens for 1-5 more minutes (assumed normal, no alarms),
keeps only the count, sum and sum of squares of its clip scores, and sets
    thr = mean + k * std of those clip scores,      k from the factory table k_burn.
The Gaussian (mu, var) still comes only from the verified calibration audio.

Voting (the "*_vote" methods): one clip above the threshold is weak evidence, because noise
bursts come and go while a fault stays. The device therefore raises the alarm only when at
least VOTE_M of the last VOTE_N clip scores (2 minutes) are above the threshold. Because a
vote is much stricter than a single clip, its threshold is set lower (VOTE_FPR per clip
instead of TARGET_FPR), which catches weaker faults. On the device: VOTE_N flags in a ring
buffer and one counter.

The "retune_*" methods are the fair threshold re-tune baselines: factory Gaussian, alarm
threshold from the same 5 minutes of burn-in audio (and the same vote).
"""

from dataclasses import dataclass, field

import numpy as np

TARGET_FPR = 0.05
VAR_FLOOR = 1e-6
N0 = 30.0          # prior weight of the factory variances, in "windows" (see _mixed_var)
WINS = 10          # 1 s windows per MIMII clip
SECONDS = [5, 10, 20, 30]
BURN = {"burn_1min": 6, "burn_5min": 30}   # burn-in length in 10 s clips
VOTE_N, VOTE_M = 12, 9   # alarm when 9 of the last 12 clips (2 min) are above the threshold
VOTE_FPR = 0.20          # per-clip false-alarm rate of the threshold used with the vote
VOTED = ["burn_5min_vote", "retune_5min_vote"]

METHODS = ["none", "thr_retune", "centroid", "centroid_cal", "centroid_var_cal", *BURN,
           "burn_5min_vote", "retune_5min", "retune_5min_vote"]
LABELS = {"none": "no adaptation",
          "thr_retune": "threshold re-tune",
          "centroid": "centroid shift, factory threshold",
          "centroid_cal": "centroid shift + threshold table",
          "centroid_var_cal": "centroid + variance + threshold table",
          "burn_1min": "centroid + variance + 1 min burn-in threshold",
          "burn_5min": "centroid + variance + 5 min burn-in threshold",
          "burn_5min_vote": "same + 9-of-12 vote",
          "retune_5min": "threshold re-tune from 5 min burn-in",
          "retune_5min_vote": "threshold re-tune from 5 min burn-in + 9-of-12 vote",
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
    # k for the burn-in threshold rule, also from the rehearsal: {method: {seconds: k}}
    k_burn: dict = field(default_factory=dict)
    k_vote: dict = field(default_factory=dict)     # same, for the lower threshold of the vote
    k_retune: dict = field(default_factory=dict)   # {"clip": k, "vote": k} for the retune_* baselines


def score(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Window scores, shape (n,). x is (n, D)."""
    return ((x - mu) ** 2 / var).mean(axis=1)


def clip_scores(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Mean window score per clip. x is (clips * WINS, D)."""
    return score(x, mu, var).reshape(-1, WINS).mean(axis=1)


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

    # Rehearse the burn-in: calibrate on n seconds, set mean and std from the next clips,
    # and see how many stds above the mean the other clips of that machine reach.
    for method, nb in BURN.items() if draws else ():
        src.k_burn[method] = {}
        for sec in SECONDS:
            z = []
            for g in groups:
                clips = g.reshape(len(g) // WINS, WINS, -1)
                k0 = int(np.ceil(sec / WINS))
                for _ in range(draws):
                    pick = rng.permutation(len(clips))
                    xc = clips[pick[:k0]].reshape(-1, g.shape[1])[:sec]
                    mu, v = xc.mean(axis=0), _mixed_var(var, xc, n0)
                    sb = clip_scores(clips[pick[k0:k0 + nb]].reshape(-1, g.shape[1]), mu, v)
                    st = clip_scores(clips[pick[k0 + nb:k0 + nb + 100]].reshape(-1, g.shape[1]), mu, v)
                    z.append((st - sb.mean()) / sb.std())
            src.k_burn[method][sec] = float(np.quantile(np.concatenate(z), 1 - TARGET_FPR))
            src.k_vote.setdefault(method, {})[sec] = float(np.quantile(np.concatenate(z), 1 - VOTE_FPR))

    # Same rehearsal for the fair baseline: factory mean and variances, threshold from burn-in.
    if draws:
        nb, z = BURN["burn_5min"], []
        for g in groups:
            clips = g.reshape(len(g) // WINS, WINS, -1)
            for _ in range(draws):
                pick = rng.permutation(len(clips))
                sb = clip_scores(clips[pick[:nb]].reshape(-1, g.shape[1]), src.mu, var)
                st = clip_scores(clips[pick[nb:nb + 100]].reshape(-1, g.shape[1]), src.mu, var)
                z.append((st - sb.mean()) / sb.std())
        z = np.concatenate(z)
        src.k_retune = {"clip": float(np.quantile(z, 1 - TARGET_FPR)),
                        "vote": float(np.quantile(z, 1 - VOTE_FPR))}
    return src


def burn_in_threshold(scores: np.ndarray, k: float) -> float:
    """Alarm threshold from the clip scores seen during burn-in. On the device this needs
    only three running numbers: count, sum and sum of squares."""
    return float(scores.mean() + k * scores.std())


def vote_alarm(scores: np.ndarray, thr: float, m: int = VOTE_M, n: int = VOTE_N) -> np.ndarray:
    """Alarm state after each clip, from the n-th clip on: True when at least m of the last
    n clip scores are above thr. scores must be in time order."""
    c = np.cumsum(np.r_[0, scores > thr])
    return (c[n:] - c[:-n]) >= m


def vote_statistic(scores: np.ndarray, m: int = VOTE_M, n: int = VOTE_N) -> np.ndarray:
    """m-th largest of the last n clip scores. vote_alarm(scores, thr) == (this > thr);
    used on the PC only, to compute a ROC-AUC for the voted decision."""
    w = np.lib.stride_tricks.sliding_window_view(scores, n)
    return np.sort(w, axis=1)[:, n - m]


def calibrate(method: str, src: SourceModel, xc: np.ndarray, n0: float = N0, burn=None):
    """Returns (mu, var, thr) for the new machine from n calibration windows xc (n, D).
    burn: for the burn-in methods, the windows heard after calibration (clips * WINS, D)."""
    n = len(xc)
    if method == "none":
        return src.mu, src.var, src.thr
    if method == "thr_retune":
        # Keep the factory Gaussian, move only the alarm threshold.
        s = score(xc, src.mu, src.var)
        return src.mu, src.var, s.mean() + src.k * s.std()
    if method in ("retune_5min", "retune_5min_vote"):
        s = clip_scores(burn[:BURN["burn_5min"] * WINS], src.mu, src.var)
        k = src.k_retune["vote" if method in VOTED else "clip"]
        return src.mu, src.var, burn_in_threshold(s, k)

    mu = xc.mean(axis=0)
    if method == "centroid":
        # New mean, factory variances, factory threshold.
        return mu, src.var, src.thr
    if method == "centroid_cal":
        # Same, but with the threshold the factory rehearsal found for n seconds.
        return mu, src.var, src.thr_table[method][n]
    if method == "centroid_var_cal":
        return mu, _mixed_var(src.var, xc, n0), src.thr_table[method][n]
    if method in BURN:
        var = _mixed_var(src.var, xc, n0)
        s = clip_scores(burn[:BURN[method] * WINS], mu, var)
        return mu, var, burn_in_threshold(s, src.k_burn[method][n])
    if method == "burn_5min_vote":
        var = _mixed_var(src.var, xc, n0)
        s = clip_scores(burn[:BURN["burn_5min"] * WINS], mu, var)
        return mu, var, burn_in_threshold(s, src.k_vote["burn_5min"][n])
    raise ValueError(method)
