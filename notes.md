# Notes

## Fixed audio settings (ESP32 must match; source of truth: src/audio_config.py)
| setting | value |
|---|---|
| sample rate | 16 kHz, mono, channel 0 of the 8 MIMII channels |
| n_fft / win_length | 1024 (periodic Hann) |
| hop | 512 |
| mel bands | 64, Slaney scale, Slaney norm (librosa defaults), 0–8000 Hz |
| spectrum | power (|X|^2) → mel → 10·log10(max(x, 1e-10)), ref = 1.0, no top_db |
| center | False (no edge padding) |
| model input | 1 s = 16000 samples → 64 × 30 frames |

### Decisions made 2026-11-13 (please review as a group)
- **center=False**: librosa's default pads half a window at both ends; the ESP32 streams and
  cannot pad, so we don't either. Gives exactly 30 frames per 1 s window.
- **dB with a fixed ref (1.0), not ref=np.max**: per-clip max normalization hides loudness
  changes (a real fault cue) and is awkward on-device. Any normalization belongs in the model
  or the calibration step, where it is explicit.
- **Slaney mel + Slaney norm**: librosa default. The C++ filterbank must use the same
  formula; we'll export the 64×513 filter matrix from librosa to a header rather than
  re-deriving it.
- **Data layout** includes the noise level: `data/<snr>/fan/id_XX/...`, because the three
  MIMII zips use identical file names.

## Data check (fill in after running src/check_data.py)
| snr | id | normal | abnormal |
|---|---|---|---|
| 6dB | id_00 | | |
| 6dB | id_02 | | |
| 6dB | id_04 | | |
| 6dB | id_06 | | |

## Observations
- (week 1) spectrogram plots: what differs between normal and abnormal? between id_00 and id_02?
  Does -6 dB noise hide the difference?
