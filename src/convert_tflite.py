"""Week 5: turn a trained embedding CNN into a fully int8 TFLite model for the ESP32-S3.

Every weight and every activation becomes int8, input and output included, so the
microcontroller needs no float maths inside the network. The value ranges are measured on
a few hundred normal training windows (the "representative data") from the noise levels
and machines the CNN was trained on, never from a held-out machine.
Each representative window is shifted by a random level within +/- 6 dB (--gain-db). Without
this the int8 output range fits 6 dB and 0 dB audio exactly and clips 7% of the embedding
values at -6 dB; with it the int8 model matches the float model at every noise level.

Writes models/<name>_int8.tflite and models/<name>_int8.json (size, input/output scale and
zero point, and how close the int8 embedding is to the float one). Model files are not
committed.

Usage:
    python src/convert_tflite.py --model fan_v1
    python src/evaluate.py --model fan_v1_int8          # same tables, int8 embedding
"""

import argparse
import json
import os
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras

import audio_config as C
from dataset import IDS, ROOT, load_split
from tflite_runner import Int8Embedder

REP_PER_GROUP = 100    # representative windows per (noise level, machine)
CHECK_PER_GROUP = 300  # windows per group for the float-vs-int8 comparison


def windows(snr, mid, part, n, rng):
    w = load_split(snr, mid)[part].reshape(-1, C.N_MELS, C.FRAMES_PER_WIN, 1)
    return w[rng.choice(len(w), size=min(n, len(w)), replace=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="name used in train.py --name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gain-db", type=float, default=6.0,
                    help="shift each representative window by a random level within +/- this "
                         "many dB, so the int8 ranges leave room for a louder or quieter room")
    ap.add_argument("--suffix", default="_int8", help="output name = <model><suffix>")
    args = ap.parse_args()

    mdir = ROOT / "models"
    meta = json.loads((mdir / f"{args.model}.json").read_text())
    net = keras.models.load_model(mdir / f"{args.model}.keras")
    ids = [m for m in IDS if m != meta.get("holdout_id")]
    rng = np.random.default_rng(args.seed)
    rep = np.concatenate([windows(s, m, "train", REP_PER_GROUP, rng)
                          for s in meta["snrs"] for m in ids])
    if args.gain_db:
        rep = rep + rng.uniform(-args.gain_db, args.gain_db, size=(len(rep), 1, 1, 1)).astype(np.float32)
    print(f"representative data: {len(rep)} windows from {meta['snrs']} x {ids}, "
          f"{rep.min():.1f} to {rep.max():.1f} dB")

    with tempfile.TemporaryDirectory() as tmp:
        net.export(tmp, verbose=False)   # SavedModel: the route that works with Keras 3
        conv = tf.lite.TFLiteConverter.from_saved_model(tmp)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: ([w[None]] for w in rep)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        blob = conv.convert()
    out = mdir / f"{args.model}{args.suffix}.tflite"
    out.write_bytes(blob)

    emb = Int8Embedder(out)
    ops = sorted({d["op_name"] for d in emb.itp._get_ops_details()} - {"DELEGATE"})
    print(f"wrote {out}: {len(blob) / 1024:.1f} kB")
    print(f"input : int8, dB = (q - {emb.in_zero}) * {emb.in_scale:.5f}  "
          f"-> covers {(-128 - emb.in_zero) * emb.in_scale:.1f} to "
          f"{(127 - emb.in_zero) * emb.in_scale:.1f} dB")
    print(f"output: int8, emb = (q - {emb.out_zero}) * {emb.out_scale:.5f}")
    print("ops   :", ", ".join(ops))

    # How close is the int8 embedding to the float one? Checked on test windows, including
    # -6 dB, which neither the CNN nor the quantization ranges have seen.
    report = {}
    print("\nfloat vs int8 embedding on normal test windows")
    print("noise   cosine similarity   error / spread   input clipped   output clipped")
    for snr in ["6dB", "0dB", "-6dB"]:
        x = np.concatenate([windows(snr, m, "test_normal", CHECK_PER_GROUP, rng) for m in IDS])
        f, q8 = net.predict(x, batch_size=256, verbose=0), emb.predict_int8(emb.quantize(x))
        q = (q8.astype(np.float32) - emb.out_zero) * emb.out_scale
        cos = float(np.mean(np.sum(f * q, axis=1) /
                            (np.linalg.norm(f, axis=1) * np.linalg.norm(q, axis=1))))
        # RMS error per dimension relative to how much that dimension varies over windows
        rel = float(np.mean(np.sqrt(((f - q) ** 2).mean(axis=0)) / f.std(axis=0)))
        lo, hi = (-128 - emb.in_zero) * emb.in_scale, (127 - emb.in_zero) * emb.in_scale
        clip_in = float(((x < lo) | (x > hi)).mean())
        clip_out = float((np.abs(q8.astype(int)) >= 127).mean())
        report[snr] = {"cosine": cos, "rel_error": rel, "input_clipped": clip_in,
                       "output_clipped": clip_out}
        print(f"{snr:>5s}   {cos:.4f}              {rel:.3f}            {100 * clip_in:.2f}%"
              f"           {100 * clip_out:.2f}%")

    (mdir / f"{args.model}{args.suffix}.json").write_text(json.dumps(
        {**meta, "name": args.model + args.suffix, "source": args.model, "bytes": len(blob),
         "gain_db": args.gain_db,
         "in_scale": float(emb.in_scale), "in_zero": int(emb.in_zero),
         "out_scale": float(emb.out_scale), "out_zero": int(emb.out_zero),
         "ops": ops, "float_vs_int8": report}, indent=2))


if __name__ == "__main__":
    main()
