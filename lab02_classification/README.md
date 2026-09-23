# Lab 2 — Classification & Regression with scikit-learn

**Machine Learning Project (53744-01) · Week 3 · Due: Wednesday, 23 September 2026,**
**11:59 PM (KST), via e-Class**
**No late submission is accepted. A submission after the deadline scores 0 points.**

> **Before the lab session, read the concept note in this folder**: `lab02_concepts.pdf`.
> It covers everything this lab assumes. Everything else specific to this lab is in this README.
> Course-wide policy — grading and AI use — was given in the policy slides of the first class.

## 1. Goal

Build **correct** supervised pipelines for the two basic problem types:

- **Classification** — stratified leak-free split → metrics implemented from scratch →
  majority-class baseline → controlled comparison of two models.
- **Regression** — the same discipline on a continuous target: MAE/RMSE/R² from scratch →
  predict-the-mean baseline → linear regression vs. kNN under identical conditions.

Datasets (both synthetic, both shipped — **no download needed**):

- `data/clinic_noshow.csv` — 1,200 clinic appointments, 6 numeric features, binary target
  `no_show` (~35% positive).
- `data/clinic_wait.csv` — 900 visits to the same clinic, 5 numeric features, continuous target
  `wait_minutes` (how long the patient waited).

## 2. Environment

| Item | Value |
|------|-------|
| Python | 3.10.x |
| Install | `pip install -r requirements.txt` |
| Hardware | CPU only |
| Expected runtime | < 1 minute |
| Expected effort | 4–5 hours |
| Seed | 42 (already set in the skeleton) |

Your code must run top to bottom with a single command on a clean machine that has only
`requirements.txt` installed.

## 3. Tasks

Open `src/lab02.py` and complete the six TODO blocks (specs are in the docstrings):

| Task | Function | Points share |
|------|----------|--------------|
| 1 | `split_data` — stratified, seeded, leak-free train/val split | ~15% of correctness |
| 2 | `compute_metrics` — accuracy/precision/recall/f1 **with numpy only** | ~25% |
| 3 | `build_baseline` + `build_model` — Dummy baseline; Scaler→clf Pipelines | ~15% |
| 4 | `run_experiment` — baseline vs logreg vs knn under identical conditions | ~15% |
| 5 | `compute_regression_metrics` — MAE/RMSE/R² **with numpy only** | ~15% |
| 6 | `build_regression_model` + `run_regression_experiment` — mean baseline vs linreg vs knn | ~15% |

Then open `notebooks/lab02_visualization.ipynb` — a **working tool, not a deliverable**. It
imports your finished functions from `src/lab02.py`, so it only runs once your implementation is
correct. Launch Jupyter from inside `notebooks/`; the first cell uses relative paths like
`../src`. Complete the three plot cells and **export the figures into your report**:

| Report figure | Content |
|---------------|---------|
| Figure 1 | class balance of `no_show` |
| Figure 2 | `days_ahead` distribution by class |
| Figure 3 | accuracy and F1 for baseline / logreg / knn |

You do **not** submit the notebook — only the figures, inside `report.pdf`.

**Skeleton rules** — the automated tests depend on these:

- Write your code **only inside the TODO blocks**. You may add private helper functions.
- **Do not rename functions or change their arguments and return types.** The tests call them
  directly; a renamed function scores 0 on those tests.
- Do **not** call `sklearn.metrics` inside `compute_metrics` or `compute_regression_metrics`.
  The point is to show you know what the numbers mean; the tests compare your values against
  sklearn's.
- Do not modify anything marked `# DO NOT MODIFY` (`set_seed()`, `load_data()`,
  `load_regression_data()`, `main()`) or anything in `tests/`.
- **Do not hard-code answers.** Hidden tests run on different data, so a value copied from the
  shipped dataset will fail there. Hard-coding is treated as academic dishonesty.

## 4. Run & self-check

Fill in `STUDENT_ID` / `STUDENT_NAME` at the top of `src/lab02.py`, then:

```bash
pip install -r requirements.txt
python src/lab02.py            # writes results.json
python -m pytest tests/ -q     # 9 public tests (hidden tests run at grading)
```

## 5. What to submit

Submit **exactly one zip file**, named:

```
{student_id}_{name}_lab02.zip        e.g., 20261234_홍길동_lab02.zip
                                          20261234_HongGildong_lab02.zip
```

Your name in **Korean or roman letters**, written as one word — **no spaces, digits, or symbols**
(`Hong Gildong`, `Hong-Gildong`, `hong2` all fail the checker).

containing **exactly these four items at the top level** — nothing else:

```
20261234_HongGildong_lab02.zip
├── src/            # your completed lab02.py (.py files only)
├── report.pdf      # analysis + Figures 1–3 + AI-usage table (see §6)
├── results.json    # written by `python src/lab02.py` — do not edit by hand
└── README.md       # the exact commands to reproduce your results
```

| # | Item | What must be true |
|---|------|-------------------|
| 1 | `src/` | `.py` only. Function names, signatures, and return types unchanged. No notebooks, no data. |
| 2 | `results.json` | generated by `python src/lab02.py`; `student_id` matches the zip name; `seed` is 42; not hand-edited |
| 3 | `report.pdf` | ≤ 3 pages, text-based (not a scan); Figures 1–3 embedded and referred to by number in your text |
| 4 | `README.md` | a NEW file you write (not this instruction sheet): the commands a grader runs to reproduce your numbers |

**Do not include**: notebooks (`.ipynb`), the datasets, virtual environments, `__pycache__`,
`.DS_Store`, or any file over 10 MB.

Validate before submitting:

```bash
python check_submission.py 20261234_HongGildong_lab02.zip
```

The checker enforces the zip name, the four items, the `results.json` schema, seed, and lab
number, and the 10 MB limit. It does **not** open `report.pdf`; that part is graded by a person.

## 6. The report (`report.pdf`)

An analysis, not a diary. **0.5–1 page of text plus Figures 1–3, ≤ 3 pages total.** Fill in
`report_template.md`, then export to PDF with any tool (VS Code/Typora export, `pandoc`,
Word/HWP). The PDF must be text-based, not a photo or a scan.

Required sections, in this order:

1. **What you did** — 2–3 sentences, only where you deviated from or extended the skeleton.
2. **Results** — Table 1 (classification) and Table 2 (regression) from `results.json`.
3. **Figures** — Figures 1–3 from §3, embedded.
4. **Interpretation** — *why* the numbers and figures look the way they do.
5. **Limitations** — one thing that would change your conclusion.
6. **AI & external-code usage table** — every AI tool and external source you used, in the
   table at the end of `report_template.md`. An empty table means "I used nothing".

## 7. Questions

e-Class Q&A board (preferred, since answers benefit everyone — do not post solution code).
