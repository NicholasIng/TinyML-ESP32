"""Fixed audio front-end settings.

DO NOT CHANGE these without updating notes.md and the ESP32 firmware:
the C++ mel-spectrogram must reproduce exactly these numbers.
"""

SR = 16_000          # Hz, mono
CHANNEL = 0          # MIMII files have 8 channels; we use the first one only
N_FFT = 1024
HOP = 512
N_MELS = 64
WIN_SAMPLES = SR     # ~1 s model input window

# Choices that are easy to get wrong when porting to C++ (see notes.md):
CENTER = False       # no padding at the edges; a streaming device can't pad the future
WINDOW = "hann"      # periodic Hann (scipy/librosa default)
POWER = 2.0          # power spectrogram before the mel filterbank
FMIN = 0.0
FMAX = SR / 2
MEL_HTK = False      # librosa default (Slaney mel scale)
MEL_NORM = "slaney"  # librosa default filter normalization
DB_REF = 1.0         # fixed reference: no per-clip normalization
DB_AMIN = 1e-10      # floor before log10

# Frames in one 1 s window with CENTER=False: 1 + (16000 - 1024) // 512 = 30
FRAMES_PER_WIN = 1 + (WIN_SAMPLES - N_FFT) // HOP
