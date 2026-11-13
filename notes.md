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

## Data check (src/check_data.py, run 2026-11-14)
All three noise levels have identical counts (same recordings, different noise mix).
Every sampled file: 16000 Hz, 8 channels, 10.0 s.

| snr | id | normal | abnormal |
|---|---|---|---|
| 6dB, 0dB, -6dB | id_00 | 1011 | 407 |
| 6dB, 0dB, -6dB | id_02 | 1016 | 359 |
| 6dB, 0dB, -6dB | id_04 | 1033 | 348 |
| 6dB, 0dB, -6dB | id_06 | 1015 | 361 |
| **total per snr** | | **4075** | **1475** |

Unzipped size of all three levels: ~40 GB.

## Observations (week 1, plots in results/)
Mean spectrum = median over 50 clips of each clip's mean dB per mel band.

- **id_00 normal vs abnormal (6 dB)**: abnormal has a strong extra peak in the lowest mel
  bands 0-1 (about +9 dB), and slightly less energy around bands 35-45. The clear tone at
  band ~20 is in both. So the fault cue is mostly low-frequency and stationary over the 10 s.
- **id_00 vs id_02, both normal (6 dB)**: id_02 is 5-10 dB louder from band ~12 upward and
  has a different set of tonal lines. The gap between two *normal* machines is bigger than
  the gap between normal and abnormal of one machine. This is the domain-shift problem: a
  threshold/model fitted on one machine will not transfer, which is what calibration fixes.
- **id_02 abnormal is quieter than id_02 normal** (closer to id_00 normal). A fault is not
  always "more energy"; keeping a fixed dB reference (no per-clip max) keeps this cue.
- **-6 dB**: background noise lifts and flattens everything; the four curves almost overlap.
  The id_00 low-band peak survives, but the id_02 normal/abnormal gap shrinks to 2-3 dB and
  the id_00 vs id_02 difference mostly disappears. Expect much lower AUC at -6 dB.
- Note: the single-clip spectrograms (`*_normal_vs_abnormal.png`) show one file only; trust
  the mean-spectrum plots for group differences.

## Environment notes
- Machine used 2026-11-14 had only Python 3.13 (not 3.11); all week-1 requirements install
  and run fine on 3.13. Check TensorFlow's supported Python versions before week 2.
- `--snr -6dB` fails (argparse reads `-6dB` as a flag); use `--snr=-6dB`.
