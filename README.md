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

## Rules
- Never commit `data/`, `venv/`, or model files (.gitignore covers them).
- Audio settings live only in `src/audio_config.py`; change them only with a note in notes.md.
