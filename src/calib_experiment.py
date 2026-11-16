"""Week 4: does a few seconds of normal audio fix the domain shift?

For each shift scenario and machine, n seconds (5/10/20/30) of verified-normal audio from
the NEW condition are drawn at random from its training clips (never from test clips),
each calibration method in calibration.py produces (mu, var, thr), and the new condition's
test clips are scored. Repeated over many random draws. The burn-in methods also get the
next 1 or 5 minutes of the new condition's training clips (never the calibration clips).

Scenarios (factory -> new condition):
  noise_0dB      same fan, 6 dB -> 0 dB           (model fan_v1)
  noise_-6dB     same fan, 6 dB -> -6 dB          (model fan_v1; -6 dB never seen by the CNN)
  machine        three other fans -> unseen fan, 6 dB   (leave-one-out models)
  machine+noise  three other fans at 6 dB -> unseen fan at 0 dB

The "*_vote" methods decide on the last 12 clips (2 minutes) instead of one clip. Their
false-alarm and detection rates are counted over every run of 12 consecutive test clips in
file order. In MIMII, neighbouring abnormal clips are the same kind of fault (their scores
are correlated, r = 0.6-0.8 at 6 dB) while normal clips show no correlation in time.

Features: emb = CNN embedding (32), mel64 = mean dB per mel band (64, no CNN).
Metrics: ROC-AUC, false-alarm rate, detection rate, balanced accuracy = (TPR + 1 - FPR) / 2.

Usage:
    python src/calib_experiment.py     # writes results/calibration.csv and two figures
    python src/calib_experiment.py --suffix _int8 --tag _int8    # same with the int8 models
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import calibration as cal
from dataset import IDS, ROOT
from evaluate import feature_bank

SECONDS = cal.SECONDS
SCENARIOS = ["noise_0dB", "noise_-6dB", "machine", "machine+noise"]
FEATURES = ["emb", "mel64"]
WINS = cal.WINS


def metrics(model, test_normal, test_abnormal, voted=False):
    mu, var, thr = model
    sn = cal.clip_scores(test_normal, mu, var)
    sa = cal.clip_scores(test_abnormal, mu, var)
    if voted:   # decision = at least VOTE_M of the last VOTE_N clips above thr
        sn, sa = cal.vote_statistic(sn), cal.vote_statistic(sa)
    auc = roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(sa))], np.r_[sn, sa])
    fpr, tpr = (sn > thr).mean(), (sa > thr).mean()
    return auc, fpr, tpr, (tpr + 1 - fpr) / 2


def run(trials: int, n0: float, seed: int, suffix: str = "") -> pd.DataFrame:
    seen = feature_bank("fan_v1" + suffix)
    rows = []
    for h in IDS:
        loo = feature_bank("loo_" + h + suffix)
        others = [m for m in IDS if m != h]
        setups = {   # scenario: (feature bank, factory groups as (snr, id), target snr)
            "noise_0dB": (seen, [("6dB", h)], "0dB"),
            "noise_-6dB": (seen, [("6dB", h)], "-6dB"),
            "machine": (loo, [("6dB", m) for m in others], "6dB"),
            "machine+noise": (loo, [("6dB", m) for m in others], "0dB"),
        }
        for scen, (bank, factory, tsnr) in setups.items():
            for feat in FEATURES:
                src = cal.fit_source([bank(s, m, "train")[feat] for s, m in factory], n0)
                pool = bank(tsnr, h, "train")[feat]
                tn, ta = bank(tsnr, h, "test_normal")[feat], bank(tsnr, h, "test_abnormal")[feat]
                o = cal.fit_source([pool], draws=0)
                rows.append((scen, h, feat, "oracle", 0, 0, *metrics((o.mu, o.var, o.thr), tn, ta)))
                rng = np.random.default_rng([seed, IDS.index(h)])
                rng_burn = np.random.default_rng([seed, IDS.index(h), 1])
                clips = pool.reshape(-1, WINS, pool.shape[1])
                for t in range(trials):
                    for sec in SECONDS:
                        xc, used = cal.draw(pool, sec, rng)
                        rest = rng_burn.choice(np.delete(np.arange(len(clips)), used),
                                               size=max(cal.BURN.values()), replace=False)
                        burn = clips[rest].reshape(-1, pool.shape[1])
                        for method in cal.METHODS:
                            m = metrics(cal.calibrate(method, src, xc, n0, burn), tn, ta,
                                        method in cal.VOTED)
                            rows.append((scen, h, feat, method, sec, t, *m))
    return pd.DataFrame(rows, columns=["scenario", "id", "feature", "method", "seconds", "trial",
                                       "auc", "fpr", "tpr", "bacc"])


def plot(summary: pd.DataFrame, out, metric="bacc", ylabel="balanced accuracy"):
    colors = dict(zip(cal.METHODS, ["0.5", "C1", "C0", "C2", "C3", "C4", "C5", "k", "C8", "C6"]))
    fig, axes = plt.subplots(len(FEATURES), len(SCENARIOS), figsize=(16, 7.8), sharey=True,
                             constrained_layout=True)
    for r, feat in enumerate(FEATURES):
        for c, scen in enumerate(SCENARIOS):
            ax, d = axes[r, c], summary[(summary.feature == feat) & (summary.scenario == scen)]
            for method in cal.METHODS:
                m = d[d.method == method].sort_values("seconds")
                if metric == "fpr":   # log axis; rates below 0.01% are drawn at 0.01%
                    ax.plot(m.seconds, m.fpr.clip(lower=1e-4), marker="o", ms=4,
                            color=colors[method], label=cal.LABELS[method])
                else:
                    ax.errorbar(m.seconds, m[metric], yerr=m[metric + "_std"], marker="o", ms=4,
                                capsize=2, color=colors[method], label=cal.LABELS[method])
            ax.axhline(d[d.method == "oracle"][metric].iloc[0], color="k", ls="--", lw=1,
                       label=cal.LABELS["oracle"])
            ax.set_xticks(SECONDS)
            if metric == "fpr":
                ax.axhline(cal.TARGET_FPR, color="C3", ls=":", lw=1)
                ax.set_yscale("log")
                ax.set_ylim(7e-5, 1.5)
            else:
                ax.set_ylim(0.45, 1.0)
            ax.set_title(f"{scen}  [{feat}]", fontsize=10)
            if r == 1:
                ax.set_xlabel("calibration audio (s)")
        axes[r, 0].set_ylabel(ylabel)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4,
               fontsize=8)
    fig.suptitle("Calibration with a few seconds of normal audio (mean over 4 fans, "
                 "bars = std over random draws)")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--n0", type=float, default=cal.N0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--suffix", default="", help="model name suffix, e.g. _int8 for the "
                                                 "TFLite models from convert_tflite.py")
    args = ap.parse_args()

    df = run(args.trials, args.n0, args.seed, args.suffix)
    cols = ["auc", "fpr", "tpr", "bacc"]
    # mean over the four fans per trial, then mean and std over trials
    per_trial = df.groupby(["scenario", "feature", "method", "seconds", "trial"])[cols].mean()
    g = per_trial.groupby(["scenario", "feature", "method", "seconds"])
    summary = g.mean().join(g.std().add_suffix("_std").fillna(0)).reset_index()

    pd.set_option("display.width", 200)
    for feat in FEATURES:
        for scen in SCENARIOS:
            d = summary[(summary.feature == feat) & (summary.scenario == scen)]
            print(f"\n=== {scen} [{feat}]   oracle: " + "  ".join(
                f"{c}={d[d.method == 'oracle'][c].iloc[0]:.3f}" for c in cols))
            for metric in ("bacc", "fpr", "tpr", "auc"):
                t = d[d.method != "oracle"].pivot(index="method", columns="seconds", values=metric)
                print(f"{metric}:")
                print(t.loc[cal.METHODS].round(4).to_string())
    res = ROOT / "results"
    summary.round(5).to_csv(res / f"calibration{args.tag}.csv", index=False)
    print("\nwrote", res / f"calibration{args.tag}.csv")
    plot(summary, res / f"week4_calibration{args.tag}.png")
    plot(summary, res / f"week4_false_alarms{args.tag}.png", "fpr",
         "false-alarm rate, log scale (dotted: 5%)")


if __name__ == "__main__":
    main()
