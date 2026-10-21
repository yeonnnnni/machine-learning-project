# Lab 4-1 — Reproduction

Student ID: 20231879
Name: LeeJeongyeon

## Environment

- Python 3.10.x
- CPU only
- Seed: 42

## Reproduce results

Extract the submission zip into the original `lab04_1_gradient_descent` course folder, replacing the skeleton implementation with the submitted `src/lab04_1.py`. Keep the course-provided `requirements.txt`, `data/press_qc.csv`, and `tests/` in that folder; they are not included in the submission zip.

From that folder, run:

```bash
python -m pip install -r requirements.txt
python src/lab04_1.py
```

The second command generates `results.json` and prints the gradient-check result and the learning-rate comparison. The experiment uses 8 hidden units, 400 epochs, a batch size of 32, and learning rates 0.01, 1.0, and 100.0.

## Run public tests

```bash
python -m pytest tests/ -q
```

For the unstable run at learning rate 100.0, final losses can vary between machines; compare `max_train_loss` and the overall behavior rather than requiring identical final losses.
