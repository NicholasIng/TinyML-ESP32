"""Check that the MIMII data is unzipped into the expected layout.

Expected: data/<snr>/fan/id_XX/{normal,abnormal}/*.wav  with snr in 6dB, 0dB, -6dB

Usage:
    python src/check_data.py                 # all noise levels
    python src/check_data.py --snr 6dB       # one level
    python src/check_data.py --sample 0      # skip the per-file header check
"""

import argparse
import random
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SNRS = ["6dB", "0dB", "-6dB"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "data", type=Path)
    ap.add_argument("--snr", choices=SNRS, nargs="*", default=SNRS)
    ap.add_argument("--sample", type=int, default=5,
                    help="wav headers to check per folder (0 = skip)")
    args = ap.parse_args()

    problems = []
    for snr in args.snr:
        fan = args.data / snr / "fan"
        print(f"\n== {snr}: {fan}")
        if not fan.is_dir():
            problems.append(f"missing folder {fan}")
            print("   MISSING")
            continue
        ids = sorted(p for p in fan.iterdir() if p.is_dir())
        print(f"   {'id':8s} {'normal':>7s} {'abnormal':>9s}   sr / ch / seconds (sampled)")
        for d in ids:
            counts, info = {}, set()
            for label in ("normal", "abnormal"):
                wavs = sorted((d / label).glob("*.wav"))
                counts[label] = len(wavs)
                if not wavs:
                    problems.append(f"no wavs in {d / label}")
                for w in random.Random(0).sample(wavs, min(args.sample, len(wavs))):
                    i = sf.info(str(w))
                    info.add((i.samplerate, i.channels, round(i.duration, 2)))
            info_s = ", ".join(f"{sr}/{ch}/{dur}" for sr, ch, dur in sorted(info)) or "-"
            print(f"   {d.name:8s} {counts['normal']:7d} {counts['abnormal']:9d}   {info_s}")
            for sr, ch, _ in info:
                if sr != 16000:
                    problems.append(f"{d}: sample rate {sr}, expected 16000")
                if ch != 8:
                    problems.append(f"{d}: {ch} channels, expected 8")

    print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("OK: layout looks right.")


if __name__ == "__main__":
    main()
