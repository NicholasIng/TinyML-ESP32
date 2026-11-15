"""Week 3: machine shift. What happens on a fan the CNN has never heard?

Needs four leave-one-out models, each trained without one machine:
    python src/train.py --name loo_id_00 --holdout-id id_00 --snrs 6dB 0dB --classes id_snr --mixup 0.5
    (same for id_02, id_04, id_06)

For each held-out machine H, with the model that never saw H, the same H test clips are
scored with a Gaussian + alarm threshold taken from different places:
  refit   normal training clips of H itself  (upper bound: full calibration on the new machine)
  pooled  the three other machines together  (a "factory default" shipped with the device)
  single  one other machine                  (mean over the three possible source machines)
The noise level is the same for fit and test, so only the machine changes.
"seen" is the reference: the fan_v1 model, whose CNN was trained on all four machines.

Usage:
    python src/shift_machine.py
"""

import argparse

import numpy as np
import pandas as pd

from dataset import IDS, ROOT
from evaluate import METHODS, METRICS, feature_bank, score_methods

SNRS = ["6dB", "0dB"]
SCENARIOS = ["seen", "refit", "pooled", "single"]


def pool(groups: list[dict]) -> dict:
    return {k: np.concatenate([g[k] for g in groups]) for k in groups[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seen-model", default="fan_v1")
    ap.add_argument("--loo-prefix", default="loo_")
    args = ap.parse_args()

    rows = []
    seen = feature_bank(args.seen_model)
    for h in IDS:
        loo = feature_bank(args.loo_prefix + h)
        others = [m for m in IDS if m != h]
        for snr in SNRS:
            def add(scenario, feats, fit):
                test = feats(snr, h, "test_normal"), feats(snr, h, "test_abnormal")
                for r in score_methods(fit, *test):
                    rows.append((scenario, snr, h, *r))

            add("seen", seen, seen(snr, h, "train"))
            add("refit", loo, loo(snr, h, "train"))
            add("pooled", loo, pool([loo(snr, m, "train") for m in others]))
            for m in others:
                add("single", loo, loo(snr, m, "train"))

    df = pd.DataFrame(rows, columns=["scenario", "snr", "id", "method", *METRICS])
    df = df.groupby(["scenario", "snr", "id", "method"], as_index=False)[METRICS].mean()
    titles = {"auc": "ROC-AUC", "fpr": "false-alarm rate (threshold from the fit data)",
              "tpr": "detection rate"}
    for snr in SNRS:
        for metric, title in titles.items():
            t = df[df.snr == snr].pivot_table(index="method", columns="scenario", values=metric)
            print(f"\n{snr}: mean {title} over the four held-out machines")
            print(t.loc[METHODS, SCENARIOS].round(3).to_string())
    t = df[(df.snr == "6dB") & (df.method == "emb/diag")].pivot(index="id", columns="scenario",
                                                                 values="auc")
    print("\n6dB, emb/diag: ROC-AUC per held-out machine")
    print(t[SCENARIOS].round(3).to_string())
    out = ROOT / "results" / "shift_machine.csv"
    df.round(4).to_csv(out, index=False)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
