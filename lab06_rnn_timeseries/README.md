# Lab 6 — RNNs and Time-Series Forecasting

**Machine Learning Project (53744-01) · Week 7 · Due: Wednesday, 28 October 2026,**
**11:59 PM (KST), via e-Class**
**No late submission is accepted. A submission after the deadline scores 0 points.**

> **Before the lab session, read the concept note in this folder**: `lab06_concepts.pdf`.
> It covers everything this lab assumes. Everything else specific to this lab is in this README.
> Course-wide policy — grading and AI use — was given in the policy slides of the first class.

## 1. Goal

Forecast a time series **without letting the future leak into the past**. Every lab so far
relied on rows being independent, and Lab 2's stratified random split was the right thing to do.
On a time series that same split is leakage. Here you build the honest version: windows that
never contain their own target, a chronological split with an embargo, a small GRU, and a
**persistence baseline** (repeat yesterday's value) that is much harder to beat than it looks.

Dataset: `data/cafe_daily.csv` — 1,095 (synthetic) days of cups sold at a campus cafe, one row
per day, columns `date` and `cups`. It has a drifting level, a day-of-week pattern, a slow
annual swing, and noise. **No download needed.**

## 2. Environment

| Item | Value |
|------|-------|
| Python | 3.10.x |
| Install | `pip install -r requirements.txt` |
| Hardware | CPU only (no GPU needed, none used) |
| Expected runtime | `python src/lab06.py` a few seconds; `pytest tests/` ≈ 10 s |
| Expected effort | 4–5 hours |
| Memory | < 1 GB |
| Seed | 42 (already set in the skeleton) |

Your code must run top to bottom with a single command on a clean machine that has only
`requirements.txt` installed.

**Install exactly what `requirements.txt` pins.** It holds `numpy<2` on purpose: the PyTorch build
that `pip` picks on some machines (notably Intel Macs) is compiled against NumPy 1.x, and pairing it
with NumPy 2 breaks every conversion between arrays and tensors. If you already have NumPy 2 in the
environment, `pip install -r requirements.txt` will downgrade it — let it.

## 3. Tasks

Complete the four TODO blocks in `src/lab06.py` (full specs in docstrings):

| Task | Function | Points share |
|------|----------|--------------|
| 1 | `make_windows` — (input window, target) pairs, no lookahead | ~25% of correctness |
| 2 | `temporal_split` — chronological train/val/test split with an embargo | ~20% |
| 3 | `ForecastRNN` + `build_rnn` — GRU and a linear head on the last time step | ~25% |
| 4 | `run_experiment` — persistence baseline against the RNN, same conditions | ~30% |

**Performance target.** This lab has a fixed baseline: the persistence forecast (repeat the last
value) scores a test **MASE of 0.9554**. Lower is better, and your RNN has to come in below that;
the reference implementation reaches **0.5725**. Both are absolute numbers, the same for everyone.
They are not a ranking against your classmates, and not something to chase by changing the
window, epochs, or learning rate the skeleton fixes.

The training loop (`train_model`), the forecast helper (`predict`) and the metrics
(`compute_metrics`) are **given** and marked `# DO NOT MODIFY`. You wrote a training loop in
Lab 4-2; this lab is not about writing it again. It is about the four decisions above, each of
which can silently invent an impossible forecast.

Then open `notebooks/lab06_visualization.ipynb` — a **working tool, not a deliverable**. It
imports your finished functions from `src/lab06.py`, so it only runs once your implementation is
correct. Launch Jupyter from inside `notebooks/`; the first cell uses relative paths like
`../src`. Complete the three plot cells and **export the figures into your report**:

| Report figure | Content |
|---------------|---------|
| Figure 1 | the whole series with the train / validation / test segments and the embargo gaps |
| Figure 2 | actual, persistence and RNN forecasts over the last 60 test days |
| Figure 3 | MAE, RMSE and MASE for the two forecasters |

You do **not** submit the notebook — only the figures, inside `report.pdf`.

**Skeleton rules** — the automated tests depend on these:

- Write your code **only inside the TODO blocks**. You may add private helper functions.
- **Do not rename functions or change their arguments and return types.** The tests call them
  directly; a renamed function scores 0 on those tests. `ForecastRNN` must keep the attribute
  names `self.rnn` and `self.head`, because the tests look them up.
- Do **not** use `train_test_split`, `KFold`, or any shuffling anywhere in this lab. A random
  split of a time series trains on the future, and the hidden tests are built to catch it.
- Do not modify anything marked `# DO NOT MODIFY` (`set_seed()`, `load_data()`, `train_model()`,
  `predict()`, `compute_metrics()`, `main()`) or anything in `tests/`.
- **Do not hard-code answers.** Hidden tests run on a different series, so a value copied from
  the shipped dataset will fail there. Hard-coding is treated as academic dishonesty.
- All logic goes in the `.py` file. It is the only code you submit and the only code graded.
- Running `src/lab06.py` twice must print **the same numbers**. Call `set_seed()` at the top of
  `run_experiment`, before the model is built. Both the public and the hidden tests check this.

## 4. Run & self-check

Fill in `STUDENT_ID` / `STUDENT_NAME` at the top of `src/lab06.py`, then:

```bash
pip install -r requirements.txt
python src/lab06.py            # writes results.json
python -m pytest tests/ -q     # 6 public tests (hidden tests run at grading)
```

## 5. What to submit

Submit **exactly one zip file**, named:

```
{student_id}_{name}_lab06.zip        e.g., 20261234_홍길동_lab06.zip
                                          20261234_HongGildong_lab06.zip
```

Your name in **Korean or roman letters**, written as one word — **no spaces, digits, or symbols**
(`Hong Gildong`, `Hong-Gildong`, `hong2` all fail the checker).

containing **exactly these four items at the top level** — nothing else:

```
20261234_HongGildong_lab06.zip
├── src/            # lab06.py (your completed .py files ONLY)
├── report.pdf      # analysis + Figures 1-3 (see §3) + AI-usage table
├── results.json    # written by `python src/lab06.py` — do not edit by hand
└── README.md       # the exact commands to reproduce your results
```

| # | Item | Where it comes from | What must be true |
|---|------|---------------------|-------------------|
| 1 | `src/` | your edited `src/lab06.py` | `.py` only. Function names, signatures, and return types unchanged. No notebooks, no data. |
| 2 | `results.json` | generated by `python src/lab06.py` | `student_id` matches the zip name, `seed` is 42, not hand-edited |
| 3 | `report.pdf` | fill in `report_template.md`, then export to PDF | ≤ 3 pages, text-based (searchable) — not a photo or scan. Figures 1–3 embedded, each referred to by number in your text. |
| 4 | `README.md` | you write it (a NEW file — not this instruction sheet) | the exact commands a grader runs to reproduce your numbers, plus anything they must know |

The **AI & external-code usage table is inside `report.pdf`** (last section of
`report_template.md`). Leaving it empty means "I used nothing", and you will be held to that.

**Do not include**: notebooks (`.ipynb`), the dataset, virtual environments, `__pycache__`,
`.DS_Store`, or any file over 10 MB. The notebook is a working tool, not a deliverable —
its figures belong in `report.pdf`, where they are graded.

Validate before submitting:

```bash
python check_submission.py 20261234_HongGildong_lab06.zip
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
