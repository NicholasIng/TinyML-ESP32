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

Note: the `base_6dB` numbers above come from the first version of `train.py` (30 epochs, early
stopping). Re-running `train.py --name base_6dB` now uses 12 epochs with cosine learning-rate
decay, so numbers will differ slightly.

## Week 2b: a better embedding (2026-11-15)
`train.py` got options to make the training task harder and now saves only the embedding
part of the network (62,832 parameters; the classification head is thrown away).
−6 dB is never used to train the CNN, so it stays an unseen noise level for week 3.

| run | `train.py` options |
|---|---|
| a | `--snrs 6dB 0dB` (4 classes = machine ID) |
| b | `--snrs 6dB 0dB --classes id_snr` (8 classes = machine ID × noise level) |
| c | b + `--mixup 0.5` |
| d | c + `--head cos` (cosine-margin head) |
| e | `--snrs 6dB --mixup 0.5 --head cos` (6 dB only) |

### Mean ROC-AUC over the four IDs, embedding + diagonal Gaussian (seed 0)
Columns are "Gaussian fitted on > tested on".

| model | 6>6 dB | 0>0 dB | −6>−6 dB | 6>−6 dB (no adaptation) | id_00 at 6 dB |
|---|---|---|---|---|---|
| no CNN, mel64 / shrink | 0.950 | 0.850 | 0.691 | 0.700 | 0.841 |
| base_6dB | 0.935 | 0.802 | 0.601 | 0.572 | 0.792 |
| a | 0.929 | 0.825 | 0.724 | 0.666 | 0.832 |
| b | 0.965 | 0.912 | 0.688 | 0.531 | 0.867 |
| **c** | **0.993** | **0.912** | 0.701 | 0.552 | **0.987** |
| d | 0.982 | 0.895 | 0.702 | 0.539 | 0.948 |
| e | 0.947 | 0.817 | 0.625 | 0.521 | 0.845 |

Mean pAUC(0.1) at 6>6 dB: base 0.866, a 0.851, b 0.952, c 0.972, d 0.934, e 0.863.
Per-ID numbers and the shrinkage variant: `results/eval_<model>.csv`.

### What this says
- **The CNN now beats the no-CNN baseline** at 6 dB (0.993 vs 0.950) and 0 dB (0.912 vs 0.850).
- **A harder task is what helped.** Making the network also tell the noise level apart
  (8 classes) and adding mixup lifted the hard machine id_00 from 0.79 to 0.99.
  The cosine-margin head gave nothing on top of mixup: keep plain softmax.
- **For the embedding, diagonal covariance is as good as or better than shrinkage** in the
  matched conditions (c: 0.993 vs 0.970). Good news for the ESP32: 32 means + 32 variances.
- **Trade-off:** b/c/d are best when the Gaussian is fitted on the matching noise level but
  near chance with no adaptation (6>−6 dB ≈ 0.55). Run a (noise level not a class) is more
  robust unadapted (0.666) but weaker when matched. We keep **c**: it is the "accurate when
  calibrated, breaks under shift" case that the week-4 calibration has to fix.
- **−6 dB stays hard for everything** (~0.70 even when refitted on all −6 dB training data).
  That is the ceiling for calibration at −6 dB with these models.
- Caveat: the best run was picked by test-set AUC, so 0.993 is a little optimistic.

### Run c with three seeds (embedding + diagonal Gaussian, mean over the four IDs)
The alarm threshold is the 95th percentile of the clip scores of the normal clips the
Gaussian was fitted on (target: 5% false alarms). Values are mean ± std over seeds 0, 1, 2.

| fitted on > tested on | ROC-AUC | false-alarm rate | detection rate |
|---|---|---|---|
| 6 > 6 dB | 0.994 ± 0.001 | 0.043 ± 0.002 | 0.961 ± 0.010 |
| 0 > 0 dB | 0.906 ± 0.006 | 0.071 ± 0.003 | 0.647 ± 0.017 |
| −6 > −6 dB | 0.703 ± 0.002 | 0.063 ± 0.003 | 0.231 ± 0.017 |
| 6 > 0 dB (no adaptation) | 0.705 ± 0.039 | 1.000 | 1.000 |
| 6 > −6 dB (no adaptation) | 0.539 ± 0.013 | 1.000 | 1.000 |

- The improvement is not seed luck: id_00 at 6 dB is 0.986-0.987 in all three runs.
- **This is already the week-3 domain-shift result for noise:** a threshold set at 6 dB
  alarms on *every* clip at 0 dB and −6 dB. Refitting on the new noise level brings false
  alarms back to 6-7%. Week 4 has to get close to the refit rows using only 10-30 s of audio.
- Detection still falls with noise after a full refit (96% → 65% → 23%).
- The chosen model is saved as `models/fan_v1.keras` (= run c, seed 0):
  `python src/train.py --name fan_v1 --snrs 6dB 0dB --classes id_snr --mixup 0.5`

## Week 3: domain shift (2026-11-15)
Figure for the slides: `results/week3_shift.png` (`python src/plot_shift.py`).
All numbers: CNN embedding + diagonal Gaussian unless stated, mean over the four fans.
The alarm threshold always comes from the normal clips the Gaussian was fitted on (5% target).

### Noise shift (model `fan_v1`, same machine, Gaussian fitted at 6 dB)
See the three-seed table in week 2b. Tested at 0 dB: AUC 0.99 → 0.71, false alarms 4% → 100%.
Tested at −6 dB: AUC 0.54, false alarms 100%. Refit on the new noise level: false alarms 6-7%.

### Machine shift (`src/shift_machine.py`, results in `results/shift_machine.csv`)
Four leave-one-out CNNs (`train.py --holdout-id`), each trained without one fan, same recipe
as `fan_v1`. The held-out fan is then tested with the Gaussian + threshold taken from:
its own normal clips (refit), the three other fans pooled, or one other fan (mean of three).

| 6 dB | ROC-AUC | false-alarm rate | detection rate |
|---|---|---|---|
| fan seen in CNN training (`fan_v1`) | 0.993 | 0.043 | 0.951 |
| unseen fan, Gaussian from one other fan | 0.578 | 1.000 | 1.000 |
| unseen fan, Gaussian from three other fans | 0.507 | 0.754 | 0.790 |
| unseen fan, refit on its own normal clips | 0.931 | 0.036 | 0.768 |

Same at 0 dB: seen 0.912, one other fan 0.569 (95% false alarms), pooled 0.648 (27%),
refit 0.834 (6% false alarms, 46% detection).

ROC-AUC per held-out fan at 6 dB: id_00 0.987 → 0.814 after refit, id_02 1.000 → 0.989,
id_04 0.987 → 0.921, id_06 1.000 → 1.000 (seen → unseen + refit).

### What this says
- **No adaptation = unusable.** Moving the detector to another fan, or to a noisier room,
  makes it alarm on (almost) everything. A "factory default" pooled over three fans does not
  help (75% false alarms at 6 dB). This is the problem statement for week 4.
- **Refitting on the new machine fixes the false alarms** (4%) and most of the AUC (0.93).
  That refit uses ~590 clips (98 minutes) of normal audio; week 4 must get close with 10-30 s.
- **Open problem: on an unseen fan the CNN is no better than no CNN.** After refit the
  embedding gets 0.931 (diag) / 0.946 (shrink); the 64 mel means get 0.931 / 0.950. The big
  gain of week 2b (0.993) only holds for fans the CNN was trained on. id_00 is the fan that
  loses most (0.987 → 0.814). With only three training fans the CNN learns those fans, not
  "fans in general". Options: (1) add more machines to CNN training (other MIMII types, MIMII
  DUE), (2) accept it and state that the CNN's value is for known machine models while
  calibration handles the room/noise, (3) run week 4 calibration on both features and report both.
- Memory note: do not start four trainings at once on this PC (24 GB): three crashed with
  out-of-memory while loading. Two at a time is fine.

## Week 4: calibration with 5-30 s of normal audio (2026-11-15)
Code: `src/calibration.py` (the methods, plain arrays, portable to C++) and
`src/calib_experiment.py` (the experiment). Results: `results/calibration.csv`,
figure `results/week4_calibration.png`.

**Setup.** Model per machine = `mu[32]`, `var[32]`, `thr`. Window score = mean over the 32
dimensions of (x − mu)² / var; a clip alarms if the mean score of its ten windows > thr.
Calibration audio is drawn at random from the new condition's *training* clips (whole 10 s
clips, so 20 s = 2 clips, 30 s = 3 clips), 30 random draws, mean over the four fans.

| scenario | factory data | new condition | CNN |
|---|---|---|---|
| noise_0dB | same fan, 6 dB | 0 dB | fan_v1 |
| noise_-6dB | same fan, 6 dB | −6 dB (never seen by the CNN) | fan_v1 |
| machine | three other fans, 6 dB | unseen fan, 6 dB | leave-one-out |
| machine+noise | three other fans, 6 dB | unseen fan, 0 dB | leave-one-out |

| method | what changes on the device |
|---|---|
| none | nothing |
| thr_retune | only thr (= mean + k·std of the calibration window scores; k from the factory) |
| centroid | mu = mean of the calibration windows; factory var and thr |
| centroid_cal | centroid, but thr from a small factory table indexed by calibration length |
| centroid_var_cal | centroid_cal + var = (n0·var_factory + n·var_measured)/(n0 + n), n0 = 30 |

The threshold table comes from rehearsing the calibration on factory data only (calibrate on
n seconds of a factory fan, score its other clips, take the 95% quantile). It corrects for
the mean being slightly off when it is estimated from a few seconds. No target data is used.

### Results, CNN embedding (balanced accuracy = (detection + 1 − false alarms)/2)
| scenario | method | 5 s | 10 s | 20 s | 30 s | ROC-AUC at 30 s | false alarms at 30 s |
|---|---|---|---|---|---|---|---|
| noise_0dB | thr_retune | 0.664 | 0.682 | 0.696 | 0.703 | 0.662 | 0.055 |
| noise_0dB | centroid_var_cal | 0.798 | 0.779 | 0.809 | 0.821 | 0.904 | 0.190 |
| noise_-6dB | thr_retune | 0.543 | 0.545 | 0.556 | 0.563 | 0.552 | 0.129 |
| noise_-6dB | centroid_var_cal | 0.559 | 0.565 | 0.604 | 0.626 | 0.682 | 0.404 |
| machine | thr_retune | 0.592 | 0.607 | 0.633 | 0.642 | 0.538 | 0.098 |
| machine | centroid_var_cal | 0.805 | 0.812 | 0.841 | 0.838 | 0.916 | 0.090 |
| machine+noise | thr_retune | 0.595 | 0.613 | 0.633 | 0.646 | 0.558 | 0.095 |
| machine+noise | centroid_var_cal | 0.658 | 0.662 | 0.703 | 0.730 | 0.817 | 0.289 |

No adaptation: balanced accuracy 0.500 everywhere (alarms on every clip).
Full refit on all target training data, ROC-AUC: 0.912 / 0.701 / 0.931 / 0.834.

### What this says
- **The rule is met: calibration beats threshold re-tune** in all four scenarios at every
  length, on balanced accuracy and by a wide margin on ROC-AUC (e.g. machine: 0.92 vs 0.54).
  Threshold re-tune cannot change the ranking of clips, so its AUC stays at the unadapted value.
- **The centroid shift does most of the work, and 5 s is already enough for it.** ROC-AUC with
  10 s is within 0.02-0.06 of a full refit on ~98 minutes of audio.
- **The threshold is the weak point.** In the machine scenario false alarms are ~9% (target
  5%). When the new room is noisier they are 19% (0 dB) and 40% (−6 dB) even with 30 s:
  noisier audio varies more from clip to clip than the factory rehearsal assumed. Mixing in
  measured variances helps (centroid_var_cal vs centroid_cal: 0.30 → 0.19 at 0 dB) but does
  not close the gap. Threshold re-tune has fewer false alarms but misses over half the faults.
- **5 s and 10 s are equal; 20 s and 30 s are better.** 5 and 10 s come from one clip, 20/30 s
  from 2-3 different clips. Variety over time matters more than seconds: on the device,
  spread the calibration recording out (e.g. 3 × 10 s some minutes apart).
- **The CNN earns its place here.** With calibration the embedding beats the 64 mel means in
  every scenario (30 s, balanced accuracy: 0.82 vs 0.69, 0.63 vs 0.56, 0.84 vs 0.72, 0.73 vs
  0.64), including on an unseen fan, where the two were tied after a full refit (week 3).
- n0 between 10 and 100 changes balanced accuracy by less than 0.02: not a sensitive setting.
- Tried and dropped: re-tuning the threshold from leave-one-out scores of the calibration
  windows. Windows from the same clip are too similar, so the threshold comes out too low.
- Caveat: balanced accuracy compares methods at different operating points; the full refit
  can score lower on it than a calibrated model because it sits at ~6% false alarms.

### Open for week 5 / later
- Lower the false alarms in noisier rooms: e.g. a safety factor on the table threshold that
  grows with the measured/factory variance ratio, or a short second calibration pass.
- Port `calibrate()` and `score()` to C++ (fixed arrays: 32 means, 32 variances, 4 thresholds).
