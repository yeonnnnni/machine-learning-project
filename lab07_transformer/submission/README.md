# Lab 7 Reproduction

Student ID: 20231879
Name: LeeJeongyeon

## Environment

- Python 3.10.x; CPU only; seed 42.
- The author's Conda environment is named `MLP`.
- Install the packages specified in the original assignment's `requirements.txt`,
  including `numpy>=1.26,<2`.

## Reproduce results

Use the original distributed `lab07_transformer` directory, which contains
`requirements.txt`, `data/machine_events.npz`, `tests/`, and
`check_submission.py`. Place `20231879_LeeJeongyeon_lab07.zip` in that directory
and run the following commands there, using Python 3.10:

```bash
unzip -o 20231879_LeeJeongyeon_lab07.zip
python -m pip install -r requirements.txt
python src/lab07.py
python -m pytest tests/ -q
```

The submitted `src/lab07.py` replaces the skeleton and locates
`data/machine_events.npz` relative to its own location. Extract into the original
assignment directory rather than running the code directly from `submission/`.
The dataset is supplied with the assignment and is not included in this ZIP.

The experiment compares `mean_pool`, `encoder_nopos`, and `encoder_pos` using
the same 3,000 training logs and 1,000 validation logs. The fixed settings are
maximum sequence length 16, vocabulary size 12, embedding dimension 32,
4 attention heads, feed-forward dimension 64, Adam learning rate 0.003,
batch size 64, and 12 epochs. The random seed is reset to 42 before each model
is constructed.

`python src/lab07.py` prints the results and writes `results.json` in the
assignment directory. Run it again to check reproducibility:

```bash
python src/lab07.py
```

Two runs in the same environment should produce identical results except for
`runtime_seconds`. Small numerical differences across PyTorch versions,
processors, or thread counts are expected, as noted in the assignment.

## Recorded results

| Configuration | Validation accuracy | Validation loss |
|---------------|---------------------|-----------------|
| `mean_pool` | 0.471 | 0.7018 |
| `encoder_nopos` | 0.490 | 0.7003 |
| `encoder_pos` | 0.970 | 0.1335 |

Validation positive rate: 0.517. Positional-encoding accuracy gap
(`encoder_pos` minus `encoder_nopos`): 0.480.

## Check the submission ZIP

From the original assignment directory:

```bash
python check_submission.py 20231879_LeeJeongyeon_lab07.zip
```

The ZIP contains only `src/`, `report.pdf`, `results.json`, and this `README.md`.
The dataset, dependency file, tests, checker, notebook, separate figure images,
and cache directories are not included. Figures 1-3 are embedded in `report.pdf`.
