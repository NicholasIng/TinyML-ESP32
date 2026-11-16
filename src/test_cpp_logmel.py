"""Week 5: checks that the C++ log-mel front end (firmware/lib/zt_logmel) gives the same
model input as src/features.py, and that the small differences do not change the result.

Builds the library with g++ on the PC and runs it on real MIMII clips (16-bit samples of
channel 0, exactly what the wav file holds). Compared with features.windows():
  1. the dB values of every frame,
  2. the int8 window the model gets, with the audio fed in chunks of different sizes,
  3. the int8 embedding from that window, and the anomaly score of each clip.
Also checks that the committed table header matches src/audio_config.py, and the level
offset for a microphone with another sensitivity.

Usage:
    python src/test_cpp_logmel.py               # 48 clips (2 normal + 2 abnormal per fan and noise level)
    python src/test_cpp_logmel.py --clips 10    # 240 clips
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import audio_config as C
import calibration as cal
from dataset import IDS, ROOT
from evaluate import feature_bank
from export_logmel_tables import OUT as TABLES, tables_text
from features import load_wav, windows
from tflite_runner import Int8Embedder

LIB = ROOT / "firmware" / "lib" / "zt_logmel"
HARNESS = ROOT / "firmware" / "test_pc" / "test_logmel.cpp"
MODEL = "fan_v1_int8"
SNRS = ["6dB", "0dB", "-6dB"]
CHUNKS = [160, 512, 1000, 4096, 16000, 333]   # samples per call, as a driver might deliver
WINS = cal.WINS
# Limits for PASS. The measured values are far below them (see notes.md).
MAX_DB = 0.01            # largest dB difference of any mel value
MAX_INPUT_FLIPS = 0.01   # share of int8 input values that differ (never by more than 1)
MAX_SCORE = 0.02         # largest relative difference of a clip's anomaly score


def build(tmp: Path) -> Path:
    gxx = shutil.which("g++") or r"C:\mingw64\bin\g++.exe"
    exe = tmp / "test_logmel.exe"
    subprocess.run([gxx, "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror", "-static",
                    "-I", str(LIB), str(LIB / "zt_logmel.cpp"), str(HARNESS), "-o", str(exe)],
                   check=True)
    return exe


def run_cpp(exe: Path, tmp: Path, pcm: np.ndarray, chunk: int, net: Int8Embedder, offset_db=0.0):
    """Returns (float dB windows, int8 windows), each (n, 64, 30)."""
    pcm.astype("<i2").tofile(tmp / "audio.raw")
    subprocess.run([str(exe), str(tmp / "audio.raw"), str(tmp / "out"), str(chunk),
                    repr(float(net.in_scale)), str(int(net.in_zero)), repr(float(offset_db))],
                   check=True)
    shape = (-1, C.N_MELS, C.FRAMES_PER_WIN)
    return (np.fromfile(tmp / "out.db", np.float32).reshape(shape),
            np.fromfile(tmp / "out.q", np.int8).reshape(shape))


def to_pcm(y: np.ndarray) -> np.ndarray:
    v = y * 32768.0
    assert np.all(v == np.round(v)), "wav is not 16-bit"
    return v.astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=2, help="normal and abnormal clips per fan and noise level")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert TABLES.read_text(encoding="utf-8") == tables_text(), \
        "zt_logmel_tables.h is out of date: run python src/export_logmel_tables.py"

    net = Int8Embedder(ROOT / "models" / f"{MODEL}.tflite")
    bank = feature_bank(MODEL)
    rng = np.random.default_rng(args.seed)
    db_diff, flips, flip_max, n_val = [], 0, 0, 0
    emb_flips, emb_max, n_emb = 0, 0, 0
    score_rel, decisions, n_clip = [], 0, 0
    per_snr = {}
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        exe = build(tmp)
        for mid in IDS:
            # factory model of this fan, as in the noise scenarios of week 4
            src = cal.fit_source([bank("6dB", mid, "train")["emb"]], draws=0)
            for snr in SNRS:
                for label in ("normal", "abnormal"):
                    wavs = sorted((ROOT / "data" / snr / "fan" / mid / label).glob("*.wav"))
                    for i in rng.choice(len(wavs), size=args.clips, replace=False):
                        y = load_wav(wavs[i])
                        ref = windows(y)[:WINS]
                        db, q = run_cpp(exe, tmp, to_pcm(y), CHUNKS[n_clip % len(CHUNKS)], net)
                        assert db.shape == ref.shape == q.shape, (db.shape, ref.shape, q.shape)
                        d = np.abs(db - ref)
                        db_diff.append(d.max())
                        per_snr.setdefault(snr, []).append(d.max())
                        q_ref = net.quantize(ref)
                        dq = np.abs(q.astype(int) - q_ref.astype(int))
                        flips, flip_max, n_val = flips + int((dq > 0).sum()), max(flip_max, int(dq.max())), n_val + dq.size

                        e, e_ref = net.predict_int8(q[..., None]), net.predict_int8(q_ref[..., None])
                        de = np.abs(e.astype(int) - e_ref.astype(int))
                        emb_flips, emb_max, n_emb = emb_flips + int((de > 0).sum()), max(emb_max, int(de.max())), n_emb + de.size
                        f = lambda z: (z.astype(np.float32) - net.out_zero) * net.out_scale
                        s = cal.clip_scores(f(e), src.mu, src.var)[0]
                        s_ref = cal.clip_scores(f(e_ref), src.mu, src.var)[0]
                        score_rel.append(abs(s - s_ref) / s_ref)
                        decisions += int((s > src.thr) != (s_ref > src.thr))
                        n_clip += 1

        # Level offset: the same audio twice as loud, corrected with offset_db = -20*log10(2),
        # must give the original windows again.
        y = load_wav(sorted((ROOT / "data" / "6dB" / "fan" / "id_00" / "normal").glob("*.wav"))[0])
        assert np.abs(y).max() < 0.5
        db2, _ = run_cpp(exe, tmp, to_pcm(y * 2), 512, net, offset_db=-20 * np.log10(2))
        offset_err = float(np.abs(db2 - windows(y)[:WINS]).max())

    print(f"{n_clip} clips, {n_clip * WINS} windows, {n_val} mel values")
    print("C++ vs Python (features.py)")
    print(f"  dB value, largest difference          {max(db_diff):.1e} dB "
          f"({', '.join(f'{k}: {max(v):.1e}' for k, v in per_snr.items())})")
    print(f"  int8 model input values that differ   {flips} of {n_val} ({100 * flips / n_val:.3f}%), "
          f"never by more than {flip_max}")
    print(f"  int8 embedding values that differ     {emb_flips} of {n_emb} ({100 * emb_flips / n_emb:.3f}%), "
          f"never by more than {emb_max}")
    print(f"  clip anomaly score, largest difference {100 * max(score_rel):.3f}% "
          f"(mean {100 * np.mean(score_rel):.4f}%)")
    print(f"  clip decisions that differ            {decisions} of {n_clip}")
    print(f"  level offset check                    {offset_err:.1e} dB")
    assert max(db_diff) < MAX_DB, "dB values differ too much"
    assert flip_max <= 1 and flips / n_val < MAX_INPUT_FLIPS, "int8 input differs too much"
    assert max(score_rel) < MAX_SCORE, "anomaly scores differ too much"
    assert offset_err < MAX_DB, "offset_db is wrong"
    print("PASS")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, subprocess.CalledProcessError) as e:
        print("FAIL:", e)
        sys.exit(1)
