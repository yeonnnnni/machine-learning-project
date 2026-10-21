# Lab 4-2 Reproduction

Student ID: 20231879
Name: LeeJeongyeon

## Environment

- Python 3.10.x
- CPU only
- Seed: 42
- Packages specified in the original assignment's `requirements.txt`

## Reproduce results

Use the original distributed `lab04_2_pytorch` directory, which contains
`requirements.txt`, `data/motor_health.csv`, and `tests/`.
Place `20231879_LeeJeongyeon_lab04_2.zip` in that directory and run the following commands there:

```bash
unzip -o 20231879_LeeJeongyeon_lab04_2.zip
python -m pip install -r requirements.txt
python src/lab04_2.py
python -m pytest tests/ -q
```

`python src/lab04_2.py` writes `results.json` in the assignment directory.
Run it a second time to check reproducibility:

```bash
python src/lab04_2.py
```

All values must match between the two runs except `runtime_seconds`.
The selected configuration is hidden size 32
and learning rate 0.1.
Its test loss is 0.370398 and test accuracy is
0.847500.

The dataset, dependency file, tests, notebook, and separate figure images are
provided by the original assignment or embedded in the report; they are not
included in this submission ZIP, as required by the assignment instructions.
