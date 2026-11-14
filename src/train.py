"""Train the small CNN on a classification task using NORMAL clips only.

Only the part up to the layer "embedding" (32 numbers) is saved: that is what the anomaly
scorer and, later, the ESP32 run. The classification head is a training aid and is thrown
away. Only int8-friendly layers are used in the saved part.

Options that make the training task harder (a harder task gives a more useful embedding):
  --snrs 6dB 0dB      train on several noise levels
  --classes id_snr    one class per (machine ID, noise level) instead of per machine ID
  --mixup 0.5         blend pairs of windows and their labels (Beta(0.5, 0.5))
  --head cos          cosine-margin head (CosFace) instead of plain softmax

Output (git-ignored): models/<name>.keras, models/<name>.json

Usage:
    python src/train.py --name base_6dB
    python src/train.py --name best --snrs 6dB 0dB --classes id_snr --mixup 0.5 --head cos
"""

import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import audio_config as C
from dataset import IDS, ROOT, load_split

EMB_DIM = 32
COS_SCALE, COS_MARGIN = 20.0, 0.2


class CosineHead(layers.Layer):
    """Cosine similarity between the embedding and one learned direction per class."""

    def __init__(self, n_classes, **kw):
        super().__init__(**kw)
        self.n_classes = n_classes

    def build(self, shape):
        self.w = self.add_weight(shape=(shape[-1], self.n_classes), initializer="glorot_uniform")

    def call(self, x):
        return tf.matmul(tf.math.l2_normalize(x, axis=1), tf.math.l2_normalize(self.w, axis=0))


def cos_margin_loss(y, cos):
    return keras.losses.categorical_crossentropy(y, COS_SCALE * (cos - COS_MARGIN * y),
                                                 from_logits=True)


def build(mean: float, std: float, n_classes: int, head: str):
    x = inp = keras.Input((C.N_MELS, C.FRAMES_PER_WIN, 1), name="logmel")
    # Fixed input scaling (a constant, not per-clip): folds into int8 input quantization.
    x = layers.Rescaling(1.0 / std, offset=-mean / std, name="scale")(x)
    for i, ch in enumerate([16, 32, 64, 64]):
        x = layers.Conv2D(ch, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.ReLU(name=f"relu{i}")(x)
        if i < 3:
            x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    emb = layers.Dense(EMB_DIM, name="embedding")(x)
    embedder = keras.Model(inp, emb, name="fan_cnn")
    if head == "cos":
        out = CosineHead(n_classes, name="head")(emb)
    else:
        out = layers.Dense(n_classes, activation="softmax", name="head")(layers.Dropout(0.2)(emb))
    return embedder, keras.Model(inp, out)


def stack(snrs, part, classes):
    xs, ys = [], []
    for s, snr in enumerate(snrs):
        for k, mid in enumerate(IDS):
            x = load_split(snr, mid)[part].reshape(-1, C.N_MELS, C.FRAMES_PER_WIN, 1)
            xs.append(x)
            ys.append(np.full(len(x), k * len(snrs) + s if classes == "id_snr" else k))
    return np.concatenate(xs), np.concatenate(ys)


def mixup(alpha):
    def f(x, y):
        n = tf.shape(x)[0]
        g1, g2 = tf.random.gamma([n], alpha), tf.random.gamma([n], alpha)
        lam = g1 / (g1 + g2 + 1e-8)                       # Beta(alpha, alpha) per sample
        lx, ly = tf.reshape(lam, [-1, 1, 1, 1]), tf.reshape(lam, [-1, 1])
        return lx * x + (1 - lx) * x[::-1], ly * y + (1 - ly) * y[::-1]
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="model name, saved as models/<name>.keras")
    ap.add_argument("--snrs", nargs="+", default=["6dB"])
    ap.add_argument("--classes", default="id", choices=["id", "id_snr"])
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--head", default="softmax", choices=["softmax", "cos"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all)")
    args = ap.parse_args()
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    keras.utils.set_random_seed(args.seed)

    xtr, ytr = stack(args.snrs, "train", args.classes)
    xva, yva = stack(args.snrs, "val", args.classes)
    n_classes = int(ytr.max()) + 1
    mean, std = float(xtr.mean()), float(xtr.std())
    print(f"train {xtr.shape}, val {xva.shape}, {n_classes} classes, mean {mean:.2f}, std {std:.2f}")

    embedder, model = build(mean, std, n_classes, args.head)
    assert embedder.count_params() < 100_000
    batch = 64
    ds = tf.data.Dataset.from_tensor_slices((xtr, tf.one_hot(ytr, n_classes)))
    ds = ds.shuffle(len(xtr), seed=args.seed).batch(batch)
    if args.mixup > 0:
        ds = ds.map(mixup(args.mixup))
    ds = ds.prefetch(2)
    steps = args.epochs * int(np.ceil(len(xtr) / batch))
    model.compile(optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(1e-3, steps)),
                  loss=cos_margin_loss if args.head == "cos" else "categorical_crossentropy",
                  metrics=["accuracy"])
    hist = model.fit(ds, validation_data=(xva, tf.one_hot(yva, n_classes)), epochs=args.epochs,
                     verbose=2)

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    embedder.save(out / f"{args.name}.keras")
    (out / f"{args.name}.json").write_text(json.dumps(
        {**{k: v for k, v in vars(args).items() if k != "threads"}, "mean": mean, "std": std,
         "params": embedder.count_params(), "emb_dim": EMB_DIM, "n_classes": n_classes,
         "val_accuracy": float(hist.history["val_accuracy"][-1])}, indent=2))
    print(f"saved {out / (args.name + '.keras')}  ({embedder.count_params()} params)")


if __name__ == "__main__":
    main()
