"""Runs an int8 TFLite embedding model on the PC exactly as the ESP32 will run it.

The model takes int8 and returns int8. The float log-mel window (dB) is turned into int8
with the model's own input scale and zero point, and the int8 output is turned back into
floats with the output scale and zero point. The firmware has to do the same two steps:
    q   = clamp(round(dB / in_scale) + in_zero, -128, 127)
    emb = (q_out - out_zero) * out_scale
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


class Int8Embedder:
    def __init__(self, path, batch: int = 256, reference: bool = False):
        """reference=True uses TFLite's plain reference kernels instead of the fast default
        (XNNPACK), which round some results differently (see notes.md, week 6b)."""
        self.path, self.batch, self.reference = str(path), batch, reference
        self._open(batch)
        i, o = self.itp.get_input_details()[0], self.itp.get_output_details()[0]
        assert i["dtype"] == np.int8 and o["dtype"] == np.int8, "model is not fully int8"
        self.in_scale, self.in_zero = i["quantization"]
        self.out_scale, self.out_zero = o["quantization"]

    def _open(self, n: int):
        kw = {"experimental_op_resolver_type": tf.lite.experimental.OpResolverType.BUILTIN_REF} \
            if self.reference else {"num_threads": os.cpu_count()}
        self.itp = tf.lite.Interpreter(model_path=self.path, **kw)
        i = self.itp.get_input_details()[0]
        self.itp.resize_tensor_input(i["index"], [n, *i["shape"][1:]])
        self.itp.allocate_tensors()
        self.i_idx = i["index"]
        self.o_idx = self.itp.get_output_details()[0]["index"]
        self.n = n

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Float dB -> int8, the way the firmware does it."""
        return np.clip(np.round(x / self.in_scale) + self.in_zero, -128, 127).astype(np.int8)

    def predict_int8(self, q: np.ndarray) -> np.ndarray:
        """int8 (n, 64, 30, 1) -> int8 (n, 32)."""
        out = []
        for a in range(0, len(q), self.batch):
            part = q[a:a + self.batch]
            if len(part) != self.n:
                self._open(len(part))
            self.itp.set_tensor(self.i_idx, part)
            self.itp.invoke()
            out.append(self.itp.get_tensor(self.o_idx).copy())
        return np.concatenate(out)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Float dB (n, 64, 30, 1) -> float embedding (n, 32)."""
        q = self.predict_int8(self.quantize(x))
        return ((q.astype(np.float32) - self.out_zero) * self.out_scale).astype(np.float32)
