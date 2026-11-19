"""Week 6: records audio from the ESP32-S3 + INMP441 (firmware/mic_test) over USB.

The board sends text lines (see firmware/mic_test/src/main.c):
  "L ..."                          once per second, the level of both I2S slots
  "A <seq> <checksum> <base64>"    256 samples (16 ms) of 16-bit audio
This script shows the levels, saves the audio as a wav file, and compares its level with
the MIMII data the model was trained on. The difference is offset_db, the one number the
firmware needs to move the microphone's level to the training level.

Usage:
    python src/mic_record.py --monitor                 # level lines only, Ctrl+C stops
    python src/mic_record.py --seconds 10 --name test  # data/recordings/test.wav + report
    python src/mic_record.py --check data/recordings/test.wav   # report for a saved file
Add --port COM5 if more than one serial port is connected.
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import audio_config as C
from dataset import ROOT, IDS
from features import load_wav, windows

OUT = ROOT / "data" / "recordings"
BAUD = 921600                 # used by a USB-UART bridge; the native USB port ignores it
LINE_SAMPLES = 256
MODEL_JSON = ROOT / "models" / "fan_v1_int8.json"
# USB vendor ids of the ESP32-S3's own USB port and the usual USB-UART bridge chips
ESP_VIDS = {0x303A: "Espressif USB", 0x1A86: "WCH CH34x", 0x10C4: "Silicon Labs CP210x",
            0x0403: "FTDI"}


def find_port() -> str:
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    esp = [p for p in ports if p.vid in ESP_VIDS]
    if len(esp) == 1:
        return esp[0].device
    listing = ", ".join(f"{p.device} ({p.description})" for p in ports) or "none"
    sys.exit(f"Could not pick the board's serial port. Ports found: {listing}. Use --port.")


def open_port(port: str):
    import serial
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, BAUD, 0.5
    s.dtr = s.rts = False     # set before opening, so opening the port does not reset the board
    s.open()
    return s


def decode(line: str):
    """Returns (seq, samples) of an "A" line, or None if the line is damaged."""
    parts = line.split()
    if len(parts) != 4:
        return None
    try:
        seq, check = int(parts[1]), int(parts[2])
        raw = base64.b64decode(parts[3], validate=True)
    except ValueError:
        return None
    if len(raw) != 2 * LINE_SAMPLES or (sum(raw) & 0xFFFF) != check:
        return None
    return seq, np.frombuffer(raw, "<i2")


def monitor(port: str):
    with open_port(port) as s:
        print(f"listening on {port} (Ctrl+C stops)")
        while True:
            line = s.readline().decode("ascii", "replace").strip()
            if line.startswith(("L ", "I ")):
                print(line)


def record(port: str, seconds: float):
    """Returns (int16 samples, number of samples lost)."""
    want = int(seconds * C.SR)
    chunks, lost, bad, last, levels = [], 0, 0, None, []
    with open_port(port) as s:
        s.reset_input_buffer()
        t0 = time.time()
        while sum(len(c) for c in chunks) < want:
            if time.time() - t0 > seconds + 10 and not chunks:
                sys.exit("No audio lines from the board. Is firmware/mic_test flashed, and is "
                         "this the right port? Try --monitor.")
            line = s.readline().decode("ascii", "replace").strip()
            if line.startswith("L "):
                levels.append(line)
                print(f"  {line}")
                continue
            if not line.startswith("A "):
                continue
            pkt = decode(line)
            if pkt is None:
                # The first line is usually cut off by opening the port: not a loss. Later
                # damaged lines show up as a gap in the sequence numbers below.
                bad += last is not None
                continue
            seq, x = pkt
            if last is not None and seq != last + 1:
                gap = (seq - last - 1) % 2**32
                lost += gap * LINE_SAMPLES
                chunks.append(np.zeros(gap * LINE_SAMPLES, np.int16))   # keep the timing
            last = seq
            chunks.append(x)
    if bad:
        print(f"  {bad} damaged lines (included in the lost samples)")
    return np.concatenate(chunks)[:want], lost


def mimii_reference(n_clips: int = 10):
    """Mean log-mel dB per band of normal 6 dB fan clips, or None without the data."""
    rows = []
    for mid in IDS:
        wavs = sorted((ROOT / "data" / "6dB" / "fan" / mid / "normal").glob("*.wav"))[:n_clips]
        rows += [windows(load_wav(w)).mean(axis=(0, 2)) for w in wavs]
    return np.mean(rows, axis=0) if rows else None


def report(pcm: np.ndarray, name: str):
    y = pcm.astype(np.float32) / 32768.0
    if len(y) < C.WIN_SAMPLES:
        sys.exit("Need at least 1 s of audio.")
    w = windows(y)                                  # (n, 64, 30), the model's input in dB
    meta = json.loads(MODEL_JSON.read_text()) if MODEL_JSON.exists() else {}
    train_mean = meta.get("mean", -34.63)
    scale, zero = meta.get("in_scale", 0.27714), meta.get("in_zero", 125)
    lo, hi = (-128 - zero) * scale, (127 - zero) * scale
    rec_mean = float(w.mean())
    offset = train_mean - rec_mean
    shifted = w + offset
    peak = int(np.abs(pcm.astype(np.int32)).max())

    print(f"\n{name}: {len(y) / C.SR:.1f} s, {len(w)} windows")
    print(f"  sample peak            {20 * np.log10(max(peak, 1) / 32768):6.1f} dBFS"
          f"  ({int((np.abs(pcm.astype(np.int32)) >= 32767).sum())} samples at full scale)")
    print(f"  mean log-mel level     {rec_mean:6.1f} dB   (5% {np.percentile(w, 5):.1f}, "
          f"95% {np.percentile(w, 95):.1f})")
    print(f"  training data level    {train_mean:6.1f} dB   (MIMII fans, mean)")
    print(f"  offset_db to use       {offset:+6.1f} dB")
    print(f"  model input range      {lo:.1f} .. {hi:.1f} dB; with the offset "
          f"{100 * (shifted < lo).mean():.2f}% of values below, {100 * (shifted > hi).mean():.2f}% above")
    print("  Note: offset_db should come from the machine you will monitor, running normally,")
    print("  with the microphone where it will stay. A recording of a quiet room only shows")
    print("  that the microphone works.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ref = mimii_reference()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(w.mean(axis=(0, 2)), label=f"{name}, as recorded")
    ax.plot(shifted.mean(axis=(0, 2)), label=f"{name}, + offset_db ({offset:+.1f} dB)")
    if ref is not None:
        ax.plot(ref, "k--", label="MIMII fans, normal, 6 dB")
    ax.axhline(lo, color="gray", lw=0.8)
    ax.axhline(hi, color="gray", lw=0.8)
    ax.set_xlabel("mel band")
    ax.set_ylabel("mean dB")
    ax.set_title("Average spectrum (grey lines: model input range)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = OUT / f"{name}_spectrum.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=120)
    print(f"  spectrum plot          {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="serial port, e.g. COM5 (found automatically if only one board)")
    ap.add_argument("--monitor", action="store_true", help="print the level lines only")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--name", default="test", help="saved as data/recordings/<name>.wav")
    ap.add_argument("--check", help="report for an existing 16 kHz wav instead of recording")
    args = ap.parse_args()

    if args.check:
        y = load_wav(args.check)
        report(np.round(y * 32768).clip(-32768, 32767).astype(np.int16), Path(args.check).stem)
        return
    port = args.port or find_port()
    if args.monitor:
        monitor(port)
        return
    print(f"recording {args.seconds:g} s from {port}")
    pcm, lost = record(port, args.seconds)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.name}.wav"
    sf.write(path, pcm, C.SR, subtype="PCM_16")
    print(f"saved {path} ({lost} samples lost{'' if lost == 0 else ', replaced by silence'})")
    report(pcm, args.name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
