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

## Week 2: CNN embedding + Mahalanobis baseline (2026-11-15)
Pipeline: `cache_features.py` → `train.py` → `evaluate.py` (split is in `dataset.py`).

- **Split per machine ID** (seed 0, same clips at every noise level): test = all abnormal
  clips + the same number of normal clips; 10% of the remaining normal clips = validation;
  rest = train. Train sizes: id_00 544, id_02 591, id_04 617, id_06 589 clips (×10 windows).
- **Model**: 4× (conv 3×3 → batch norm → ReLU), max-pool after the first three, global
  average pool, dense 32 (= embedding, linear), dense 4 softmax. **62,964 parameters.**
  Input scaling is one fixed mean/std from the training set (not per clip).
- **Training**: machine-ID classification on normal 6 dB clips only, 30 epochs, CPU, ~7 min.
  Validation accuracy reaches 100% after a few epochs.
- **Scoring**: per ID, Gaussian fitted on normal training windows; clip score = mean
  Mahalanobis distance of its ten windows. `diag` = variances only, `shrink` = Ledoit-Wolf.
  `mel64` = mean dB per mel band (no CNN). `emb/softmax` = −log P(correct ID); it was
  clearly worse, so it has since been removed from `evaluate.py`.

### ROC-AUC, CNN trained on 6 dB, Gaussian fitted and tested on 6 dB (one seed)
| method | id_00 | id_02 | id_04 | id_06 | mean | mean pAUC(0.1) |
|---|---|---|---|---|---|---|
| mel64 / diag (no CNN) | 0.785 | 0.984 | 0.957 | 0.999 | 0.931 | 0.822 |
| mel64 / shrink (no CNN) | 0.841 | 0.993 | 0.967 | 0.999 | 0.950 | 0.847 |
| emb / diag | 0.792 | 1.000 | 0.961 | 0.986 | 0.935 | 0.866 |
| emb / shrink | 0.761 | 0.994 | 0.949 | 0.999 | 0.926 | 0.846 |
| emb / softmax | 0.516 | 0.999 | 0.718 | 0.850 | 0.771 | 0.794 |

### Same frozen 6 dB CNN at other noise levels (mean ROC-AUC over the four IDs)
| Gaussian fitted on → tested on | mel64/diag | mel64/shrink | emb/diag | emb/shrink |
|---|---|---|---|---|
| 6 dB → 6 dB | 0.931 | 0.950 | 0.935 | 0.926 |
| 0 dB → 0 dB | 0.797 | 0.850 | 0.802 | 0.786 |
| −6 dB → −6 dB (refit on new noise) | 0.634 | 0.691 | 0.601 | 0.649 |
| 6 dB → −6 dB (no adaptation) | 0.551 | 0.700 | 0.572 | 0.546 |

Full per-ID numbers: `results/eval_base_6dB.csv` (model name `base_6dB`).

### What this says
- The pipeline works: 0.93 mean AUC at 6 dB with a 62k-parameter model and a diagonal Gaussian.
- **The CNN embedding does not beat the no-CNN baseline yet.** AUC is a tie (0.935 vs 0.931
  diag; the shrinkage mel64 baseline is best at 0.950); the embedding only wins on pAUC with
  diag (0.866 vs 0.822). Likely reason: telling four fans apart is too easy (100% accuracy
  in a few epochs), so the network is not pushed to learn fine detail. This must improve
  before week 4, otherwise calibration could be done on the 64 mel means without a CNN.
- **id_00 is the hard machine** (~0.79), not id_02 as guessed from the week-1 mean spectra.
  id_02 and id_06 are almost perfectly separable at 6 dB.
- The softmax score is clearly worse than Mahalanobis on the embedding; drop it.
- Noise hurts a lot: 0.93 → 0.80 → 0.60 (emb/diag) from 6 dB to −6 dB even when the Gaussian
  is refitted on the new noise level. With no adaptation (6 dB → −6 dB) the embedding falls to
  ~0.55-0.57, near chance. Refitting on the full −6 dB training set only recovers to 0.60-0.65,
  which is the ceiling for any week-4 calibration with this CNN at −6 dB.

### Ideas to make the embedding better (to try next)
- Harder training task: classify (machine ID × noise level) or add other MIMII machine
  types, mixup between windows, or train on all three noise levels so the CNN sees noise.
- Larger / multi-scale context, or a margin loss (ArcFace-style) instead of plain softmax.
- Repeat with 3-5 seeds before trusting differences of ±0.02.

### In progress (2026-11-15)
`train.py` now has options to make the training task harder (`--snrs`, `--classes id_snr`,
`--mixup`, `--head cos`) and saves only the embedding part of the network. Five variants are
being trained on 6 dB + 0 dB; −6 dB is kept out of CNN training so it stays an unseen noise
level for week 3. Results will be added here.

Note: the `base_6dB` numbers above come from the first version of `train.py` (30 epochs, early
stopping). Re-running `train.py --name base_6dB` now uses 12 epochs with cosine learning-rate
decay, so numbers will differ slightly.
