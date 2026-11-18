# Lab 8 — Experiment Management and Error Analysis

**Machine Learning Project (53744-01) · Week 10 · Due: Wednesday, 18 November 2026,**
**11:59 PM (KST), via e-Class**
**No late submission is accepted. A submission after the deadline scores 0 points.**

> **Before the lab session, read the concept note in this folder**: `lab08_concepts.pdf`.
> It covers everything this lab assumes. Everything else specific to this lab is in this README.
> Course-wide policy — grading and AI use — was given in the policy slides of the first class.

## 1. Goal

Run experiments so that the conclusions survive scrutiny, and find out **where** a model fails
rather than only **how much**. Four pieces: a runner that emits one record per run, a one-factor-
at-a-time ablation over several seeds, per-slice error analysis, and an aggregation that reports
mean and spread instead of a single number. This is the machinery the team project runs on.

The model is deliberately cheap — logistic regression and a shallow tree — because the model is
not the subject. Thirty runs finish in under a second.

Dataset: `data/delivery_late.csv` — 2,400 (synthetic) parcel deliveries, 7 numeric features, a
`region` column used as the slice label, binary target `late` (~40% positive). **No download
needed.**

## 2. Environment

| Item | Value |
|------|-------|
| Python | 3.10.x |
| Install | `pip install -r requirements.txt` |
| Hardware | CPU only |
| Expected runtime | `python src/lab08.py` a few seconds; `pytest tests/` a few seconds |
| Expected effort | 4–5.5 hours |
| Memory | < 1 GB |
| Seed | 42 (already set in the skeleton) |

Your code must run top to bottom with a single command on a clean machine that has only
`requirements.txt` installed.

**Install exactly what `requirements.txt` pins.** Two pins are there on purpose:

- `scipy<1.15` — on a macOS 27 (Apple silicon) machine we tested, the SciPy 1.15 wheels for
  Python 3.10 failed to load, so `import sklearn` crashed before any lab code ran.
- `numpy<2` — with NumPy 2 on the same machine, scikit-learn printed hundreds of spurious
  floating-point warnings during the logistic-regression fits. The numbers were right, but the
  noise buries real errors.

If your environment already has newer versions, `pip install -r requirements.txt` will downgrade
them — let it.

## 3. Tasks

Complete the four TODO blocks in `src/lab08.py` (full specs in docstrings):

| Task | Function | Points share |
|------|----------|--------------|
| 1 | `run_config` — one configuration, one seed, one complete record | ~25% of correctness |
| 2 | `run_ablation` — one-factor-at-a-time variants over several seeds, as a tidy table | ~30% |
| 3 | `error_analysis` — per-slice metrics, slice sizes, the worst slice worth reporting | ~25% |
| 4 | `summarize` — mean ± std per configuration, ranked, with the overlap flag | ~20% |

The helpers around them (`load_data`, `split_data`, `build_estimator`, `feature_columns`,
`compute_metrics`, `config_id`, `base_predictions`, `main`) are given. Use them; do not
reimplement them.

Then open `notebooks/lab08_experiments.ipynb` — a **working tool, not a deliverable**. It
imports your finished functions from `src/lab08.py`, so it only runs once your implementation is
correct. Launch Jupyter from inside `notebooks/`; the first cell uses relative paths like
`../src`. Complete the three plot cells and **export the figures into your report**:

| Report figure | Content |
|---------------|---------|
| Figure 1 | accuracy per slice against the overall accuracy, with each slice's sample count |
| Figure 2 | ablation: mean F1 per variant with ±1 std over seeds, against the base band |
| Figure 3 | the five per-seed values for the base and the challenger configuration |

You do **not** submit the notebook — only the figures, inside `report.pdf`.

**Skeleton rules** — the automated tests depend on these:

- Write your code **only inside the TODO blocks**. You may add private helper functions.
- **Do not rename functions or change their arguments and return types.** The tests call them
  directly; a renamed function scores 0 on those tests.
- Keys, column names, column order and sort order are part of the spec. The tests compare them
  exactly, so a table with the right numbers under different column names still fails.
- Do not modify anything marked `# DO NOT MODIFY` (`set_seed()`, `load_data()`,
  `feature_columns()`, `build_estimator()`, `split_data()`, `compute_metrics()`, `config_id()`,
  `base_predictions()`, `main()`) or anything in `tests/`.
- **Do not hard-code answers.** Hidden tests run on different data, so a value copied from the
  shipped dataset will fail there. Hard-coding is treated as academic dishonesty.
- All logic goes in the `.py` file. It is the only code you submit and the only code graded.

## 4. Run & self-check

Fill in `STUDENT_ID` / `STUDENT_NAME` at the top of `src/lab08.py`, then:

```bash
pip install -r requirements.txt
python src/lab08.py            # writes results.json
python -m pytest tests/ -q     # 6 public tests (hidden tests run at grading)
```

## 5. What to submit

Submit **exactly one zip file**, named:

```
{student_id}_{name}_lab08.zip        e.g., 20261234_홍길동_lab08.zip
                                          20261234_HongGildong_lab08.zip
```

Your name in **Korean or roman letters**, written as one word — **no spaces, digits, or symbols**
(`Hong Gildong`, `Hong-Gildong`, `hong2` all fail the checker).

containing **exactly these four items at the top level** — nothing else:

```
20261234_HongGildong_lab08.zip
├── src/            # lab08.py (your completed .py files ONLY)
├── report.pdf      # analysis + Figures 1-3 (see §3) + AI-usage table
├── results.json    # written by `python src/lab08.py` — do not edit by hand
└── README.md       # the exact commands to reproduce your results
```

| # | Item | Where it comes from | What must be true |
|---|------|---------------------|-------------------|
| 1 | `src/` | your edited `src/lab08.py` | `.py` only. Function names, signatures, and return types unchanged. No notebooks, no data. |
| 2 | `results.json` | generated by `python src/lab08.py` | `student_id` matches the zip name, `seed` is 42, not hand-edited |
| 3 | `report.pdf` | fill in `report_template.md`, then export to PDF | ≤ 3 pages, text-based (searchable) — not a photo or scan. Figures 1–3 embedded, each referred to by number in your text. |
| 4 | `README.md` | you write it (a NEW file — not this instruction sheet) | the exact commands a grader runs to reproduce your numbers, plus anything they must know |

The **AI & external-code usage table is inside `report.pdf`** (last section of
`report_template.md`). Leaving it empty means "I used nothing", and you will be held to that.

**Do not include**: notebooks (`.ipynb`), the dataset, virtual environments, `__pycache__`,
`.DS_Store`, or any file over 10 MB. The notebook is a working tool, not a deliverable —
its figures belong in `report.pdf`, where they are graded.

Validate before submitting:

```bash
python check_submission.py 20261234_HongGildong_lab08.zip
```

The checker enforces the zip name, the four items, the `results.json` schema, seed, and lab
number, and the 10 MB limit. It does **not** open `report.pdf`; that part is graded by a person.

## 6. The report (`report.pdf`)

Not a diary, an analysis. **0.5–1 page of text plus Figures 1–3, ≤ 3 pages total.** Write it from
`report_template.md`, then export to PDF (VS Code/Typora export, `pandoc report.md -o report.pdf`,
or Word/HWP export — the tool does not matter). The PDF must be **text-based (searchable)**, not a
photo or a scan of handwriting.

Required sections, in this order:

1. **What you did** — 2–3 sentences, only where you deviated from or extended the skeleton.
2. **Results** — the key numbers from `results.json`, in a small table.
3. **Figures** — Figures 1–3 from §3, embedded.
4. **Interpretation** — *why* the numbers and figures look the way they do.
5. **Limitations** — one thing that would change your conclusion.
6. **AI & external-code usage table** — every AI tool and external source you used, in the
   table at the end of `report_template.md`. An empty table means "I used nothing".

## 7. Questions

e-Class Q&A board (preferred, since answers benefit everyone — do not post solution code).
