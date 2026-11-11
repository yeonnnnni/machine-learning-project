# Lab 6 Reproduction

Student ID: 20231879
Name: LeeJeongyeon

## Environment

- Python 3.10.x; CPU only; seed 42.
- Install the packages specified in the original assignment's `requirements.txt`, including `numpy>=1.26,<2`.

## Reproduce results

Use the original distributed `lab06_rnn_timeseries` directory, which contains
`requirements.txt`, `data/cafe_daily.csv`, `tests/`, and `check_submission.py`.
Place `20231879_LeeJeongyeon_lab06.zip` in that directory and run the following
commands there, using Python 3.10:

```bash
unzip -o 20231879_LeeJeongyeon_lab06.zip
python -m pip install -r requirements.txt
python src/lab06.py
python -m pytest tests/ -q
```

The submitted `src/lab06.py` replaces the skeleton. It locates
`data/cafe_daily.csv` relative to its own location, sorts the observations by
date, and compares persistence with a GRU on the same test windows.

The experiment uses a 14-day input window, a one-day horizon, seven-day embargo
gaps, validation and test fractions of 0.15 each, GRU hidden size 32, 150 epochs,
batch size 64, and learning rate 0.01. Standardisation constants and the MASE
reference value are computed from the training segment only. Each segment is
windowed separately, without shuffling.

`python src/lab06.py` prints the results and writes `results.json` in the
assignment directory. Running it again reproduces the same values except for
`runtime_seconds`:

```bash
python src/lab06.py
```

## Recorded results

| Forecaster | MAE (cups) | RMSE (cups) | MASE |
|------------|------------|-------------|------|
| Persistence | 15.5067 | 20.0934 | 0.9554 |
| GRU | 9.2924 | 11.6670 | 0.5725 |

Split sizes in windows: train 739, validation 150, test 150.
Training mean: 170.4754; training population standard deviation: 25.5496;
MASE reference value (`naive_scale`): 16.2314 cups.

## Check the submission ZIP

From the original assignment directory:

```bash
python check_submission.py 20231879_LeeJeongyeon_lab06.zip
```

The ZIP contains only `src/`, `report.pdf`, `results.json`, and this `README.md`.
The dataset, dependency file, tests, checker, notebook, and separate figure
images are not included. Figures 1-3 are embedded in `report.pdf`.
