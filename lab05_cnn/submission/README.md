# Lab 5 Reproduction

Student ID: 20231879
Name: LeeJeongyeon

## Environment

- Python 3.10.x; CPU only; seed 42.
- Install the packages specified in the original assignment's `requirements.txt`.
- Validation environment: Python 3.10.21, NumPy 1.26.4, SciPy 1.14.1,
  scikit-learn 1.7.2, and PyTorch 2.14.1.

## Reproduce results

Use the original distributed `lab05_cnn` directory, which contains
`requirements.txt`, `data/parts_shapes.npz`, `data/backbone_shapes6.pt`,
`tests/`, and `check_submission.py`.
Place `20231879_LeeJeongyeon_lab05.zip` in that directory and run these
commands there, using Python 3.10:

```bash
unzip -o 20231879_LeeJeongyeon_lab05.zip
python -m pip install -r requirements.txt
python src/lab05.py
python -m pytest tests/ -q
```

The submitted `src/lab05.py` replaces the skeleton. It finds the dataset and
pretrained weights in `data/` relative to its own location. The experiment uses
240 labelled training images, 960 validation images, 8 epochs, batch size 32,
and learning rate 0.001 for both CNNs.

`python src/lab05.py` prints the results and writes `results.json` in the
assignment directory. Run it again to check reproducibility:

```bash
python src/lab05.py
```

All values must match between the two runs except `runtime_seconds`.

## Recorded results

| Model | Validation accuracy | Trainable parameters |
|-------|---------------------|----------------------|
| Majority-class baseline | 0.2573 | 0 |
| CNN from scratch | 0.8219 | 7,940 |
| Frozen backbone + new head | 0.9469 | 2,052 |

## Check the submission ZIP

From the original assignment directory:

```bash
python check_submission.py 20231879_LeeJeongyeon_lab05.zip
```

The ZIP contains only `src/`, `report.pdf`, `results.json`, and this `README.md`.
The original dataset, pretrained weights, dependency file, tests, checker,
notebook, and separate figure images are not included. Figures 1-3 are embedded
in `report.pdf`.
