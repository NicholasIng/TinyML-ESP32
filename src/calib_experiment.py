"""Week 4: does a few seconds of normal audio fix the domain shift?

For each shift scenario and machine, n seconds (5/10/20/30) of verified-normal audio from
the NEW condition are drawn at random from its training clips (never from test clips),
each calibration method in calibration.py produces (mu, var, thr), and the new condition's
test clips are scored. Repeated over many random draws.

Scenarios (factory -> new condition):
  noise_0dB      same fan, 6 dB -> 0 dB           (model fan_v1)
  noise_-6dB     same fan, 6 dB -> -6 dB          (model fan_v1; -6 dB never seen by the CNN)
  machine        three other fans -> unseen fan, 6 dB   (leave-one-out models)
  machine+noise  three other fans at 6 dB -> unseen fan at 0 dB

Features: emb = CNN embedding (32), mel64 = mean dB per mel band (64, no CNN).
Metrics: ROC-AUC, false-alarm rate, detection rate, balanced accuracy = (TPR + 1 - FPR) / 2.

Usage:
    python src/calib_experiment.py              # writes results/calibration.csv + .png
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


def metrics(model, test_normal, test_abnormal):
    mu, var, thr = model
    sn = cal.score(test_normal, mu, var).reshape(-1, WINS).mean(axis=1)
    sa = cal.score(test_abnormal, mu, var).reshape(-1, WINS).mean(axis=1)
    auc = roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(sa))], np.r_[sn, sa])
    fpr, tpr = (sn > thr).mean(), (sa > thr).mean()
    return auc, fpr, tpr, (tpr + 1 - fpr) / 2


def run(trials: int, n0: float, seed: int) -> pd.DataFrame:
    seen = feature_bank("fan_v1")
    rows = []
    for h in IDS:
        loo = feature_bank("loo_" + h)
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
                for t in range(trials):
                    for sec in SECONDS:
                        xc, _ = cal.draw(pool, sec, rng)
                        for method in cal.METHODS:
                            m = metrics(cal.calibrate(method, src, xc, n0), tn, ta)
                            rows.append((scen, h, feat, method, sec, t, *m))
    return pd.DataFrame(rows, columns=["scenario", "id", "feature", "method", "seconds", "trial",
                                       "auc", "fpr", "tpr", "bacc"])


def plot(summary: pd.DataFrame, out):
    colors = dict(zip(cal.METHODS, ["0.5", "C1", "C0", "C2", "C3"]))
    fig, axes = plt.subplots(len(FEATURES), len(SCENARIOS), figsize=(16, 7), sharey=True,
                             constrained_layout=True)
    for r, feat in enumerate(FEATURES):
        for c, scen in enumerate(SCENARIOS):
            ax, d = axes[r, c], summary[(summary.feature == feat) & (summary.scenario == scen)]
            for method in cal.METHODS:
                m = d[d.method == method].sort_values("seconds")
                ax.errorbar(m.seconds, m.bacc, yerr=m.bacc_std, marker="o", ms=4, capsize=2,
                            color=colors[method], label=cal.LABELS[method])
            ax.axhline(d[d.method == "oracle"].bacc.iloc[0], color="k", ls="--", lw=1,
                       label=cal.LABELS["oracle"])
            ax.set_xticks(SECONDS)
            ax.set_ylim(0.45, 1.0)
            ax.set_title(f"{scen}  [{feat}]", fontsize=10)
            if r == 1:
                ax.set_xlabel("calibration audio (s)")
        axes[r, 0].set_ylabel("balanced accuracy")
    axes[0, 0].legend(fontsize=7, loc="lower right")
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
    args = ap.parse_args()

    df = run(args.trials, args.n0, args.seed)
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
                print(t.loc[cal.METHODS].round(3).to_string())
    res = ROOT / "results"
    summary.round(4).to_csv(res / f"calibration{args.tag}.csv", index=False)
    print("\nwrote", res / f"calibration{args.tag}.csv")
    plot(summary, res / f"week4_calibration{args.tag}.png")


if __name__ == "__main__":
    main()
