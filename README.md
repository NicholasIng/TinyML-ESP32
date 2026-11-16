# TinyML-ESP32

Zhuanti project: TinyML machine sound anomaly detection.

On-device feature-space calibration for machine sound anomaly detection on ESP32-S3.
A frozen int8 CNN produces a 16–32 number embedding; a small C++ calibration step
uses 10–30 s of verified normal audio to adapt to a new machine or environment.

## Layout
```
src/        Python: audio_config.py (fixed settings), features.py, check_data.py, explore.py
results/    plots and tables for slides
firmware/   ESP32-S3 C++ (later)
data/       git-ignored, see below
notes.md    settings, decisions, observations
```

## Setup (once per person)
Needs Python 3.11.

Windows (PowerShell):
```
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
macOS / Linux:
```
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
In VS Code: Ctrl+Shift+P → "Python: Select Interpreter" → pick `venv`.

## Data (not in git)
Download from MIMII (https://zenodo.org/record/3384388): `6_dB_fan.zip`, `0_dB_fan.zip`,
`-6_dB_fan.zip`. Each zip contains a `fan/` folder. Unzip each into its own noise-level folder:

```
data/6dB/fan/id_00/{normal,abnormal}/*.wav
data/0dB/fan/...
data/-6dB/fan/...
```
Windows: `Expand-Archive 6_dB_fan.zip -DestinationPath data\6dB` (same for the others).
macOS/Linux: `unzip 6_dB_fan.zip -d data/6dB`.

Check it:
```
python src/check_data.py
```

## Week 1 plots
```
python src/explore.py --snr 6dB
python src/explore.py --snr=-6dB   # "=" needed: argparse reads a bare -6dB as a flag
```
Writes PNGs into `results/`.

## Week 2 baseline
```
python src/cache_features.py --snr 6dB    # once per noise level, ~1 min, ~400 MB in data/cache
python src/train.py --name base_6dB       # ~3 min on CPU, saves models/base_6dB.keras
python src/evaluate.py --model base_6dB   # AUC / pAUC tables, writes results/eval_base_6dB.csv
```
Results and discussion are in `notes.md`.

Current best model (`fan_v1`, needs the 6dB and 0dB caches, ~10 min on CPU):
```
python src/cache_features.py --snr 0dB
python src/train.py --name fan_v1 --snrs 6dB 0dB --classes id_snr --mixup 0.5
python src/evaluate.py --model fan_v1
python src/evaluate.py --model fan_v1 --conditions="6dB>6dB,6dB>-6dB"   # fit>test pairs
```

## Week 3 domain shift
```
python src/train.py --name loo_id_00 --holdout-id id_00 --snrs 6dB 0dB --classes id_snr --mixup 0.5
# ...same for id_02, id_04, id_06 (one or two at a time: four at once runs out of memory)
python src/shift_machine.py     # unseen-machine table, writes results/shift_machine.csv
python src/plot_shift.py        # results/week3_shift.png (needs eval_fan_v1.csv too)
```

## Week 4 calibration
```
python src/calib_experiment.py   # needs fan_v1 and the four loo_ models; ~3 min
```
Writes `results/calibration.csv`, `results/week4_calibration.png` (balanced accuracy) and
`results/week4_false_alarms.png`. The methods are in `src/calibration.py`, written to be
ported to C++: calibration from 5-30 s of verified normal audio, then a 1-5 min burn-in that
sets the alarm threshold from the device's own clip scores. The alarm itself is a vote:
at least 9 of the last 12 clips (2 minutes) above the threshold.

## Week 5 int8 model
```
python src/convert_tflite.py --model fan_v1      # models/fan_v1_int8.tflite, ~72 kB
python src/evaluate.py --model fan_v1_int8       # same tables with the int8 embedding
python src/calib_experiment.py --suffix _int8 --tag _int8   # needs the loo_ models converted too
```
The int8 model matches the float one within 0.01 on every metric (see notes.md).

## Week 5 C++ calibration
```
python src/test_cpp_calib.py       # builds firmware/lib/zt_calib with g++, compares with calibration.py
python src/export_factory.py       # writes firmware/generated/zt_factory.h (not committed)
```
`firmware/lib/zt_calib/` is the on-device calibration, burn-in and 9-of-12 vote: fixed
arrays only, 448 bytes of state. The test needs g++ on the PATH (MinGW-w64 on Windows) and
the int8 models; add `--all` for every fan.

## Week 5 C++ log-mel
```
python src/test_cpp_logmel.py           # builds firmware/lib/zt_logmel with g++, compares with features.py
python src/export_logmel_tables.py      # only after changing src/audio_config.py
```
`firmware/lib/zt_logmel/` turns 16-bit audio into the int8 window the model reads (own
1024-point FFT, 64 mel filters, dB). The test needs g++, the wav files in `data/` and
`fan_v1_int8`.

## Rules
- Never commit `data/`, `venv/`, or model files (.gitignore covers them).
- Audio settings live only in `src/audio_config.py`; change them only with a note in notes.md.
