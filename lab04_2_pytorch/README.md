# Lab 4-2 — PyTorch Training and Hyperparameter Tuning

**Machine Learning Project (53744-01) · Week 5 · Due: Wednesday, 7 October 2026,**
**11:59 PM (KST), via e-Class**
**No late submission is accepted. A submission after the deadline scores 0 points.**

> **Before the lab session, read the concept note in this folder**: `lab04_2_concepts.pdf`.
> It covers everything this lab assumes. Everything else specific to this lab is in this README.
> Course-wide policy — grading and AI use — was given in the policy slides of the first class.

## 1. Goal

Write the PyTorch training loop by hand, then tune it **honestly**: an early-stopped training run
on a validation set, a small hyperparameter grid selected on validation only, and the test set
measured exactly once at the very end. Lab 4-1 built a network with NumPy; here the framework does
the same arithmetic for you.

Dataset: `data/motor_health.csv` — 2,000 (synthetic) daily readings from factory motors, 7 numeric
features, binary target `needs_service` (~30% positive). **No download needed; the data ships in
this folder.**

## 2. Environment

| Item | Value |
|------|-------|
| Python | 3.10.x |
| Install | `pip install -r requirements.txt` |
| Hardware | CPU only (no GPU, no CUDA) |
| Expected runtime | a few seconds for all six configurations (3–6 s on the machines we measured) |
| Expected effort | 5–6.5 hours |
| Seed | 42 (already set in the skeleton) |

Your code must run top to bottom with a single command on a clean machine that has only
`requirements.txt` installed.

**Install exactly what `requirements.txt` pins.** It holds `numpy<2` on purpose: the PyTorch build
that `pip` picks on some machines (notably Intel Macs) is compiled against NumPy 1.x, and pairing it
with NumPy 2 breaks every conversion between arrays and tensors. If you already have NumPy 2 in the
environment, `pip install -r requirements.txt` will downgrade it — let it.

## 3. Tasks

Open `src/lab04_2.py` and complete the four TODO blocks (specs are in the docstrings):

| Task | Function | Points share |
|------|----------|--------------|
| 1 | `build_model` — a five-module `nn.Sequential` MLP that outputs one logit | ~15% of correctness |
| 2 | `train_one_epoch` + `evaluate` — the update step, and read-only measurement | ~35% |
| 3 | `train_model` — the epoch loop, early stopping, best weights restored | ~25% |
| 4 | `run_experiment` — the grid, selected on validation, test touched once | ~25% |

**Performance target.** This lab has a fixed baseline: a model that answers "no service needed"
for every row scores **0.7025** on the test set (and 0.700 on the validation set). Your selected
configuration has to beat that, and the reference implementation reaches **0.8475 test accuracy**.
Both are absolute numbers, the same for everyone. They are not a ranking against your classmates,
and not something to chase by adding hyperparameters the spec does not ask for.

Then open `notebooks/lab04_2_visualization.ipynb` — a **working tool, not a deliverable**. It
imports your finished functions from `src/lab04_2.py`, so it only runs once your implementation
is correct. Launch Jupyter from inside `notebooks/`; the first cell uses relative paths like
`../src`. Complete the three plot cells and **export the figures into your report**:

| Report figure | Content |
|---------------|---------|
| Figure 1 | training and validation loss per epoch for the selected configuration, with the best epoch marked |
| Figure 2 | best validation loss of all six grid configurations |
| Figure 3 | validation and test loss and accuracy for the selected configuration |

You do **not** submit the notebook — only the figures, inside `report.pdf`.

**Skeleton rules** — the automated tests depend on these:

- Write your code **only inside the TODO blocks**. You may add private helper functions.
- **Do not rename functions or change their arguments and return types.** The tests call them
  directly; a renamed function scores 0 on those tests.
- Do **not** change the layer list in `build_model`, and do **not** append `nn.Sigmoid()`.
  `nn.BCEWithLogitsLoss` expects raw logits and applies the sigmoid itself; adding one yourself
  applies it twice and the tests will see the wrong numbers.
- **Never train on the validation set, and evaluate the test set exactly once**, in
  `run_experiment`, after the search is finished. Hidden tests count how many times the test
  loader is used.
- Do not modify anything marked `# DO NOT MODIFY` (`set_seed()`, `load_data()`, `prepare_data()`,
  `main()`) or anything in `tests/`.
- **Do not hard-code answers.** Hidden tests run on different data, so a value copied from the
  shipped dataset will fail there. Hard-coding is treated as academic dishonesty.

## 4. Run & self-check

Fill in `STUDENT_ID` / `STUDENT_NAME` at the top of `src/lab04_2.py`, then:

```bash
pip install -r requirements.txt
python src/lab04_2.py          # writes results.json
python -m pytest tests/ -q     # 6 public tests (hidden tests run at grading)
```

Run `python src/lab04_2.py` twice. The numbers in `results.json` must be identical both times,
apart from `runtime_seconds`. If they are not, something in your code is unseeded.

## 5. What to submit

Submit **exactly one zip file**, named:

```
{student_id}_{name}_lab04_2.zip      e.g., 20261234_홍길동_lab04_2.zip
                                          20261234_HongGildong_lab04_2.zip
```

Your name in **Korean or roman letters**, written as one word — **no spaces, digits, or symbols**
(`Hong Gildong`, `Hong-Gildong`, `hong2` all fail the checker).

containing **exactly these four items at the top level** — nothing else:

```
20261234_HongGildong_lab04_2.zip
├── src/            # your completed lab04_2.py (.py files only)
├── report.pdf      # analysis + Figures 1–3 + AI-usage table (see §6)
├── results.json    # written by `python src/lab04_2.py` — do not edit by hand
└── README.md       # the exact commands to reproduce your results
```

| # | Item | What must be true |
|---|------|-------------------|
| 1 | `src/` | `.py` only. Function names, signatures, and return types unchanged. No notebooks, no data. |
| 2 | `results.json` | generated by `python src/lab04_2.py`; `student_id` matches the zip name; `seed` is 42; not hand-edited |
| 3 | `report.pdf` | ≤ 3 pages, text-based (not a scan); Figures 1–3 embedded and referred to by number in your text |
| 4 | `README.md` | a NEW file you write (not this instruction sheet): the commands a grader runs to reproduce your numbers |

**Do not include**: notebooks (`.ipynb`), the dataset, virtual environments, `__pycache__`,
`.DS_Store`, or any file over 10 MB.

Validate before submitting:

```bash
python check_submission.py 20261234_HongGildong_lab04_2.zip
```

The checker enforces the zip name, the four items, the `results.json` schema, seed, and lab
number, and the 10 MB limit. It does **not** open `report.pdf`; that part is graded by a person.

## 6. The report (`report.pdf`)

An analysis, not a diary. **0.5–1 page of text plus Figures 1–3, ≤ 3 pages total.** Fill in
`report_template.md`, then export to PDF with any tool (VS Code/Typora export, `pandoc`,
Word/HWP). The PDF must be text-based, not a photo or a scan.

Required sections, in this order:

1. **What you did** — 2–3 sentences, only where you deviated from or extended the skeleton.
2. **Results** — Table 1: the six grid configurations, then the selected one's test numbers.
3. **Figures** — Figures 1–3 from §3, embedded.
4. **Interpretation** — *why* the numbers and figures look the way they do.
5. **Limitations** — one thing that would change your conclusion.
6. **AI & external-code usage table** — every AI tool and external source you used, in the
   table at the end of `report_template.md`. An empty table means "I used nothing".

## 7. Questions

e-Class Q&A board (preferred, since answers benefit everyone — do not post solution code).
