# Lab 4-1 — Gradient Descent and a NumPy MLP

**Machine Learning Project (53744-01) · Week 5 · Due: Wednesday, 7 October 2026,**
**11:59 PM (KST), via e-Class**
**No late submission is accepted. A submission after the deadline scores 0 points.**

> **Before the lab session, read the concept note in this folder**: `lab04_1_concepts.pdf`.
> It covers everything this lab assumes. Everything else specific to this lab is in this README.
> Course-wide policy — grading and AI use — was given in the policy slides of the first class.

## 1. Goal

Write a small neural network **from scratch in NumPy**: the forward pass, the analytic
gradients, a numerical gradient check that proves those gradients are right, the mini-batch
gradient-descent loop, and a controlled comparison of three learning rates. No PyTorch, no
sklearn models, no autograd — you compute every derivative yourself.

Dataset: `data/press_qc.csv` — 800 (synthetic) quality-control records from a heat press.
Two numeric features (`temp_c`, `pressure_kpa`), binary target `pass` (~30% positive). A part
passes only when **both** settings sit inside their acceptance window, so the region of passing
parts is a rectangle. No straight line separates a rectangle from everything around it, which is
exactly why a hidden layer is needed. **No download needed; the data ships in this folder.**

## 2. Environment

| Item | Value |
|------|-------|
| Python | 3.10.x |
| Install | `pip install -r requirements.txt` |
| Hardware | CPU only |
| Expected runtime | < 1 minute |
| Expected effort | 4–5.5 hours |
| Seed | 42 (already set in the skeleton) |

Your graded code `src/lab04_1.py` may import **numpy, pandas, and the standard library only** —
no `torch`, no `tensorflow`, no `jax`, and no sklearn model, optimizer, or metric. Writing the
derivatives yourself is the entire point of this lab.

Your code must run top to bottom with a single command on a clean machine that has only
`requirements.txt` installed.

## 3. Tasks

Open `src/lab04_1.py` and complete the four TODO blocks (specs are in the docstrings):

| Task | Function | Points share |
|------|----------|--------------|
| 1 | `sigmoid` + `forward` + `bce_loss` — the forward pass, numerically stable | ~25% of correctness |
| 2 | `backward` + `gradient_check` — analytic gradients, verified against finite differences | ~35% |
| 3 | `train` — mini-batch gradient descent, recording the loss history | ~25% |
| 4 | `run_experiment` — three learning rates under identical conditions | ~15% |

The network is fixed: 2 inputs → 8 sigmoid hidden units → 1 sigmoid output, binary
cross-entropy loss. `init_params()` gives everyone the same starting weights, so the two usable
runs should match a classmate's closely. The **diverging run (learning rate 100)** is the
exception: once the loss explodes, rounding differences between machines grow with it, so its
`final_train_loss` and `final_val_loss` legitimately differ from one computer to the next.
`max_train_loss` is the column that is stable and the one that matters there.

Then open `notebooks/lab04_1_visualization.ipynb` — a **working tool, not a deliverable**. It
imports your finished functions from `src/lab04_1.py`, so it only runs once your implementation
is correct. Launch Jupyter from inside `notebooks/`; the first cell uses relative paths like
`../src`. Complete the three plot cells and **export the figures into your report**:

| Report figure | Content |
|---------------|---------|
| Figure 1 | the two features, one marker per class, showing the acceptance rectangle |
| Figure 2 | training-loss curves for the three learning rates, on one pair of axes |
| Figure 3 | the trained network's decision boundary over the same scatter |

You do **not** submit the notebook — only the figures, inside `report.pdf`.

**Skeleton rules** — the automated tests depend on these:

- Write your code **only inside the TODO blocks**. You may add private helper functions.
- **Do not rename functions or change their arguments and return types.** The tests call them
  directly; a renamed function scores 0 on those tests.
- **NumPy only for the maths.** Do not import `torch`, `tensorflow`, `jax`, or any sklearn
  model, optimizer, or metric.
- **`train()` should call the module-level `forward()` and `backward()`, and `run_experiment()`
  must call `train()`.** The graders instrument these functions by name to check that an epoch
  covers every row and that the validation set never reaches the update step.
- Do not modify anything marked `# DO NOT MODIFY`. In this lab that is the whole provided-helper
  block (`set_seed()`, `load_data()`, `train_val_split()`, `init_params()`, `predict()`,
  `accuracy()`) and `main()`, plus anything in `tests/`.
- **Do not hard-code answers.** Hidden tests run on different data, so a value copied from the
  shipped dataset will fail there. Hard-coding is treated as academic dishonesty.

## 4. Run & self-check

Fill in `STUDENT_ID` / `STUDENT_NAME` at the top of `src/lab04_1.py`, then:

```bash
pip install -r requirements.txt
python src/lab04_1.py          # writes results.json
python -m pytest tests/ -q     # 6 public tests (hidden tests run at grading)
```

## 5. What to submit

Submit **exactly one zip file**, named:

```
{student_id}_{name}_lab04_1.zip      e.g., 20261234_홍길동_lab04_1.zip
                                          20261234_HongGildong_lab04_1.zip
```

Your name in **Korean or roman letters**, written as one word — **no spaces, digits, or symbols**
(`Hong Gildong`, `Hong-Gildong`, `hong2` all fail the checker).

containing **exactly these four items at the top level** — nothing else:

```
20261234_HongGildong_lab04_1.zip
├── src/            # your completed lab04_1.py (.py files only)
├── report.pdf      # analysis + Figures 1–3 + AI-usage table (see §6)
├── results.json    # written by `python src/lab04_1.py` — do not edit by hand
└── README.md       # the exact commands to reproduce your results
```

| # | Item | What must be true |
|---|------|-------------------|
| 1 | `src/` | `.py` only. Function names, signatures, and return types unchanged. No notebooks, no data. |
| 2 | `results.json` | generated by `python src/lab04_1.py`; `student_id` matches the zip name; `seed` is 42; not hand-edited |
| 3 | `report.pdf` | ≤ 3 pages, text-based (not a scan); Figures 1–3 embedded and referred to by number in your text |
| 4 | `README.md` | a NEW file you write (not this instruction sheet): the commands a grader runs to reproduce your numbers |

**Do not include**: notebooks (`.ipynb`), the dataset, virtual environments, `__pycache__`,
`.DS_Store`, or any file over 10 MB.

Validate before submitting:

```bash
python check_submission.py 20261234_HongGildong_lab04_1.zip
```

The checker enforces the zip name, the four items, the `results.json` schema, seed, and lab
number, and the 10 MB limit. It does **not** open `report.pdf`; that part is graded by a person.

## 6. The report (`report.pdf`)

An analysis, not a diary. **0.5–1 page of text plus Figures 1–3, ≤ 3 pages total.** Fill in
`report_template.md`, then export to PDF with any tool (VS Code/Typora export, `pandoc`,
Word/HWP). The PDF must be text-based, not a photo or a scan.

Required sections, in this order:

1. **What you did** — 2–3 sentences, only where you deviated from or extended the skeleton.
2. **Results** — Table 1 (the gradient check) and Table 2 (the three learning rates).
3. **Figures** — Figures 1–3 from §3, embedded.
4. **Interpretation** — *why* the numbers and figures look the way they do.
5. **Limitations** — one thing that would change your conclusion.
6. **AI & external-code usage table** — every AI tool and external source you used, in the
   table at the end of `report_template.md`. An empty table means "I used nothing".

## 7. Questions

e-Class Q&A board (preferred, since answers benefit everyone — do not post solution code).
