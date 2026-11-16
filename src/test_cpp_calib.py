"""Week 5: checks that the C++ calibration (firmware/lib/zt_calib) gives the same numbers
as src/calibration.py.

Builds the library with g++ on the PC, then for many cases (machine, scenario, calibration
length, random draw, method) feeds the same int8-model embeddings to both, one window at a
time in C++, and compares: calibrated means and variances, the threshold after calibration
and after burn-in, every clip score, and every alarm decision.

Methods compared: centroid_var_cal (table threshold, no burn-in), burn_1min, burn_5min
(single-clip alarm) and burn_5min_vote (9-of-12 vote).
One more build uses the generated header from export_factory.py instead of constants
passed at run time and must give bit-identical output.

Usage:
    python src/test_cpp_calib.py            # two fans + one unseen-fan model, ~1 min
    python src/test_cpp_calib.py --all      # all four fans and all four loo_ models
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import calibration as cal
from dataset import IDS, ROOT
from evaluate import feature_bank
from export_factory import factory_fields, header

LIB = ROOT / "firmware" / "lib" / "zt_calib"
HARNESS = ROOT / "firmware" / "test_pc" / "test_calib.cpp"
# method: (burn-in clips, vote)
METHODS = {"centroid_var_cal": (0, False), "burn_1min": (6, False),
           "burn_5min": (30, False), "burn_5min_vote": (30, True)}
RTOL = 1e-4   # C++ and numpy both use 32-bit floats but add in a different order


def build(tmp: Path, name: str, extra=()) -> Path:
    gxx = shutil.which("g++") or r"C:\mingw64\bin\g++.exe"
    exe = tmp / (name + ".exe")
    cmd = [gxx, "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror", "-static",
           "-I", str(LIB), *extra, str(LIB / "zt_calib.cpp"), str(HARNESS), "-o", str(exe)]
    subprocess.run(cmd, check=True)
    return exe


def run_cpp(exe: Path, tmp: Path, fields: dict, vote: bool, xc, stream):
    """Returns mu, var, thr after calibration, final thr, sizeof(Detector), (clips, 3)."""
    parts = [np.float32([vote, len(xc), len(stream)]), fields["mu"], fields["var"],
             np.float32([fields["n0"], fields["burn_clips"]]), fields["seconds"],
             fields["thr_table"], fields["k_clip"], fields["k_vote"], xc.ravel(), stream.ravel()]
    np.concatenate([np.asarray(p, np.float32).ravel() for p in parts]).tofile(tmp / "case.bin")
    subprocess.run([str(exe), str(tmp / "case.bin"), str(tmp / "out.bin")], check=True)
    o = np.fromfile(tmp / "out.bin", np.float32)
    d = len(fields["mu"])
    return o[:d], o[d:2 * d], o[2 * d], o[2 * d + 1], int(o[2 * d + 2]), o[2 * d + 3:].reshape(-1, 3)


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - b) / np.abs(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every fan and every loo_ model")
    ap.add_argument("--trials", type=int, default=3, help="random draws per case")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # (model, factory groups, target noise level, target machine)
    fans = IDS if args.all else IDS[:2]
    setups = [("fan_v1_int8", [("6dB", h)], t, h) for h in fans for t in ("0dB", "-6dB")]
    for h in (IDS if args.all else IDS[2:3]):
        setups.append((f"loo_{h}_int8", [("6dB", m) for m in IDS if m != h], "6dB", h))

    worst = {"mu": 0.0, "var": 0.0, "thr_cal": 0.0, "thr": 0.0, "score": 0.0}
    n_cases = n_clips = n_alarm = flips = 0
    banks, size = {}, 0
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        exe = build(tmp, "test_calib")
        rng = np.random.default_rng(args.seed)
        for model, factory, tsnr, h in setups:
            if model not in banks:
                banks[model] = feature_bank(model)
            bank = banks[model]
            src = cal.fit_source([bank(s, m, "train")["emb"] for s, m in factory])
            pool = bank(tsnr, h, "train")["emb"]
            test = np.concatenate([bank(tsnr, h, "test_normal")["emb"],
                                   bank(tsnr, h, "test_abnormal")["emb"]])
            clips = pool.reshape(-1, cal.WINS, pool.shape[1])
            for sec in cal.SECONDS:
                for _ in range(args.trials):
                    xc, used = cal.draw(pool, sec, rng)
                    rest = rng.choice(np.delete(np.arange(len(clips)), used), size=30, replace=False)
                    burn = clips[rest].reshape(-1, pool.shape[1])
                    for method, (nb, vote) in METHODS.items():
                        fields = factory_fields(src, burn_clips=nb)
                        stream = np.concatenate([burn[:nb * cal.WINS], test])
                        mu, var, thr_cal, thr, size, got = run_cpp(exe, tmp, fields, vote, xc, stream)

                        e_mu, e_var, e_thr = cal.calibrate(method, src, xc, burn=burn)
                        e_score = cal.clip_scores(test, e_mu, e_var)
                        e_above = e_score > e_thr
                        # the device counts missing clips as "not above" until 12 have arrived
                        e_alarm = cal.vote_alarm(np.r_[np.full(cal.VOTE_N - 1, -np.inf), e_score],
                                                 e_thr) if vote else e_above
                        worst["mu"] = max(worst["mu"], float(np.max(np.abs(mu - e_mu))))
                        worst["var"] = max(worst["var"], rel(var, e_var))
                        worst["thr_cal"] = max(worst["thr_cal"],
                                               rel(thr_cal, src.thr_table["centroid_var_cal"][sec]))
                        worst["thr"] = max(worst["thr"], rel(thr, e_thr))
                        worst["score"] = max(worst["score"], rel(got[:, 0], e_score))
                        assert len(got) == len(e_score), "number of clips differs"
                        # a decision may only differ when the score sits exactly on the threshold
                        diff = (got[:, 1] != e_above) | (got[:, 2] != e_alarm)
                        near = np.abs(e_score - e_thr) < RTOL * e_thr
                        if vote:   # one borderline clip can change the vote for the next 12
                            near = np.convolve(near, np.ones(cal.VOTE_N), "full")[:len(near)] > 0
                        assert not (diff & ~near).any(), f"decisions differ: {model} {tsnr} {h} {sec}s {method}"
                        flips += int(diff.sum())
                        n_cases, n_clips, n_alarm = n_cases + 1, n_clips + len(got), n_alarm + int(got[:, 2].sum())

        # The generated header must carry exactly the same constants as the run-time struct.
        model, factory, tsnr, h = setups[0]
        bank = banks[model]
        src = cal.fit_source([bank(s, m, "train")["emb"] for s, m in factory])
        fields = factory_fields(src, burn_clips=30)
        meta = json.loads((ROOT / "models" / f"{model}.json").read_text())
        (tmp / "zt_factory.h").write_text(header(fields, meta, "test"), encoding="utf-8")
        exe_h = build(tmp, "test_calib_header", ["-DZT_USE_HEADER", "-I", str(tmp)])
        pool = bank(tsnr, h, "train")["emb"]
        stream = np.concatenate([pool[300:600], bank(tsnr, h, "test_abnormal")["emb"]])
        a = run_cpp(exe, tmp, fields, True, pool[:30], stream)
        b = run_cpp(exe_h, tmp, fields, True, pool[:30], stream)
        same = all(np.array_equal(x, y) for x, y in zip(a, b))

    for k, v in worst.items():
        assert v < RTOL, f"{k} differs by {v:.2e}"
    assert same, "generated header gives different results"
    print(f"{n_cases} cases, {n_clips} clip decisions ({n_alarm} alarms), "
          f"{flips} decisions differ (allowed only on the threshold)")
    print("largest difference C++ vs Python")
    print(f"  calibrated mean             {worst['mu']:.1e} (absolute)")
    print(f"  calibrated variance         {worst['var']:.1e} (relative)")
    print(f"  threshold after calibration {worst['thr_cal']:.1e}")
    print(f"  threshold after burn-in     {worst['thr']:.1e}")
    print(f"  clip score                  {worst['score']:.1e}")
    print(f"generated header: identical results; sizeof(zt::Detector) = {size} bytes on this PC")
    print("PASS")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, subprocess.CalledProcessError) as e:
        print("FAIL:", e)
        sys.exit(1)
