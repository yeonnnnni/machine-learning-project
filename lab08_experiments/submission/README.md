# Lab 8 Reproduction

Student ID: 20231879  
Name: LeeJeongyeon

## Environment

- Python 3.10.x; CPU only.
- The author's Conda environment is named `MLP`.
- The fixed seed is 42; repeated experiments use seeds 42, 43, 44, 45, and 46.
- Install the packages specified in the original assignment's `requirements.txt`,
  including `numpy>=1.26,<2` and `scipy>=1.10,<1.15`.

## Reproduce results

Use the original distributed `lab08_experiments` directory, which contains
`requirements.txt`, `data/delivery_late.csv`, `tests/`, and
`check_submission.py`. Place `20231879_LeeJeongyeon_lab08.zip` in that directory
and run the following commands there with Python 3.10:

```bash
unzip -o 20231879_LeeJeongyeon_lab08.zip
python -m pip install -r requirements.txt
python src/lab08.py
python -m pytest tests/ -q
```

If using the author's existing Conda environment, run `conda activate MLP`
before these commands.

Extract into the original assignment directory rather than running the code
directly from `submission/`. The submitted `src/lab08.py` locates
`data/delivery_late.csv` relative to its own location. The dataset is supplied
with the assignment and is not included in this ZIP.

`python src/lab08.py` prints the results and writes `results.json` in the
assignment directory. The ablation evaluates six variants over five seeds,
producing 30 rows. Results are deterministic in the same environment except
for `runtime_seconds`, which varies between executions.

## Check the submission ZIP

From the original assignment directory:

```bash
python check_submission.py 20231879_LeeJeongyeon_lab08.zip
```

The ZIP contains only `src/`, `report.pdf`, `results.json`, and this `README.md`.
Figures 1-3 and the AI usage table are embedded in `report.pdf`.
