"""Audio loading and log-mel features, shared by every script."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

import audio_config as C


def load_wav(path: str | Path) -> np.ndarray:
    """Load a wav as float32 mono at 16 kHz (channel 0 for multichannel files)."""
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y[:, C.CHANNEL]
    if sr != C.SR:
        # MIMII is already 16 kHz; this only triggers for other data.
        y = librosa.resample(y, orig_sr=sr, target_sr=C.SR)
    return y


def log_mel(y: np.ndarray) -> np.ndarray:
    """Log-mel spectrogram in dB, shape (N_MELS, frames)."""
    S = librosa.feature.melspectrogram(
        y=y, sr=C.SR, n_fft=C.N_FFT, hop_length=C.HOP, win_length=C.N_FFT,
        window=C.WINDOW, center=C.CENTER, power=C.POWER, n_mels=C.N_MELS,
        fmin=C.FMIN, fmax=C.FMAX, htk=C.MEL_HTK, norm=C.MEL_NORM,
    )
    return librosa.power_to_db(S, ref=C.DB_REF, amin=C.DB_AMIN, top_db=None).astype(np.float32)


def windows(y: np.ndarray, hop_s: float = 1.0) -> np.ndarray:
    """Cut a clip into 1 s log-mel windows, shape (n_win, N_MELS, FRAMES_PER_WIN)."""
    step = int(hop_s * C.SR)
    starts = range(0, len(y) - C.WIN_SAMPLES + 1, step)
    return np.stack([log_mel(y[s:s + C.WIN_SAMPLES]) for s in starts])
