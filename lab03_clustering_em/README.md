# Lab 3 — K-means, GMM, and EM

**Machine Learning Project (53744-01) · Week 4 · Due: Wednesday, 30 September 2026,**
**11:59 PM (KST), via e-Class**
**No late submission is accepted. A submission after the deadline scores 0 points.**

> **Before the lab session, read the concept note in this folder**: `lab03_concepts.pdf`.
> It covers everything this lab assumes. Everything else specific to this lab is in this README.
> Course-wide policy — grading and AI use — was given in the policy slides of the first class.

## 1. Goal

Cluster a 3D scene **without labels**: implement K-means with k-means++ seeding, then the
full-covariance Gaussian mixture fitted by EM (soft responsibilities in log space, closed-form
M-step), and pick the number of clusters with BIC. The scene is several objects' surface
points merged into one unlabeled cloud; the objects are deliberately elongated and flat, which
is exactly where K-means' isotropic distance breaks and the GMM's per-cluster covariances win.
Grading rewards an EM that matches the update equations exactly — the hidden tests check one
E-step and one M-step against your own responsibilities — and an experiment you can interpret.

Dataset: `data/lab03_scene.npz` — `X`: 1614 surface points of 4 objects (an elongated box, a
flat plate, a spherical shell, a thin rod), merged and shuffled; `labels`: the ground-truth
object index per point, shipped **for evaluation only** (see the Task 4 rule below).
**No download needed; the data ships in this folder.**

## 2. Environment

| Item | Value |
|------|-------|
| Python | 3.10.x |
| Install | `pip install -r requirements.txt` |
| Hardware | CPU only |
| Expected runtime | < 1 minute |
| Expected effort | 5–6 hours |
| Seed | 42 (already set in the skeleton) |

Your graded code `src/lab03.py` may import **numpy and the standard library only** — no
sklearn, no scipy. Both ship finished K-means and GMM implementations; the point of this lab
is that you can build them, log-sum-exp and all.

Your code must run top to bottom with a single command on a clean machine that has only
`requirements.txt` installed.

## 3. Tasks

Open `src/lab03.py` and complete the four TODO blocks (specs are in the docstrings):

| Task | Function | Points share |
|------|----------|--------------|
| 1 | `kmeans(X, k, rng, n_init, max_iter)` — k-means++ D² seeding (rng call order per the docstring, so that your numbers reproduce ours), Lloyd iterations, the empty-cluster re-seeding rule, best of `n_init` restarts by inertia | ~30% of correctness |
| 2 | `gmm_em(X, k, rng, max_iter, reg)` — full-covariance GMM: log-space E-step with log-sum-exp, closed-form M-step, `reg * I` on every covariance, log-likelihood trace | ~30% |
| 3 | `bic(ll, n_params, n)` + `select_k(X, k_range, rng)` — BIC with the full-covariance parameter count, one GMM fit per candidate k, argmin | ~20% |
| 4 | `run_experiment(data, rng)` — K-means vs GMM ARI on the scene, the EM monotonicity check, the BIC curve over k = 2–8; feeds `results.json` | ~20% |

Then open `notebooks/lab03_visualization.ipynb` — a **working tool, not a deliverable**. It
imports your finished functions from `src/lab03.py`, so it only runs once your implementation is
correct. Launch Jupyter from inside `notebooks/`; the first cell uses relative paths like
`../src`. Complete the three plot cells and **export the figures into your report**:

| Report figure | Content |
|---------------|---------|
| Figure 1 | the 3D scene colored by K-means labels and by GMM labels, side by side |
| Figure 2 | the EM log-likelihood trace vs iteration (from `gmm_em` on the scene) |
| Figure 3 | the BIC curve over k = 2–8 with its minimum marked (from `results.json`) |

You do **not** submit the notebook — only the figures, inside `report.pdf`.

**Skeleton rules** — the automated tests depend on these:

- Write your code **only inside the TODO blocks**. You may add private helper functions.
- **Do not rename functions or change their arguments and return types.** The tests call them
  directly; a renamed function scores 0 on those tests.
- Do **not** import `sklearn` or `scipy` in `src/lab03.py` — numpy and the standard library
  only. `adjusted_rand_index` is **provided** in the skeleton; use it as-is.
- The ground-truth `labels` may be used **only** inside `run_experiment`'s evaluation step
  (the two `adjusted_rand_index` calls). They must never reach `kmeans`, `gmm_em`, or
  `select_k` in any form — the hidden tests rerun your `run_experiment` with permuted labels
  and require every fit-derived number to be unchanged.
- Do not modify anything marked `# DO NOT MODIFY` (`set_seed()`, `load_data()`,
  `adjusted_rand_index()`, `main()`) or anything in `tests/`.
- **Do not hard-code answers.** Hidden tests run on different data, so a value copied from the
  shipped dataset will fail there. Hard-coding is treated as academic dishonesty.

## 4. Run & self-check

Fill in `STUDENT_ID` / `STUDENT_NAME` at the top of `src/lab03.py`, then:

```bash
pip install -r requirements.txt
python src/lab03.py            # writes results.json
python -m pytest tests/ -q     # 6 public tests (hidden tests run at grading)
```

## 5. What to submit

Submit **exactly one zip file**, named:

```
{student_id}_{name}_lab03.zip        e.g., 20261234_홍길동_lab03.zip
                                          20261234_HongGildong_lab03.zip
```

Your name in **Korean or roman letters**, written as one word — **no spaces, digits, or symbols**
(`Hong Gildong`, `Hong-Gildong`, `hong2` all fail the checker).

containing **exactly these four items at the top level** — nothing else:

```
20261234_HongGildong_lab03.zip
├── src/            # your completed lab03.py (.py files only)
├── report.pdf      # analysis + Figures 1–3 + AI-usage table (see §6)
├── results.json    # written by `python src/lab03.py` — do not edit by hand
└── README.md       # the exact commands to reproduce your results
```

| # | Item | What must be true |
|---|------|-------------------|
| 1 | `src/` | `.py` only. Function names, signatures, and return types unchanged. No notebooks, no data. |
| 2 | `results.json` | generated by `python src/lab03.py`; `student_id` matches the zip name; `seed` is 42; not hand-edited |
| 3 | `report.pdf` | ≤ 3 pages, text-based (not a scan); Figures 1–3 embedded and referred to by number in your text |
| 4 | `README.md` | a NEW file you write (not this instruction sheet): the commands a grader runs to reproduce your numbers |

**Do not include**: notebooks (`.ipynb`), the dataset, virtual environments, `__pycache__`,
`.DS_Store`, or any file over 10 MB.

Validate before submitting:

```bash
python check_submission.py 20261234_HongGildong_lab03.zip
```

The checker enforces the zip name, the four items, the `results.json` schema, seed, and lab
number, and the 10 MB limit. It does **not** open `report.pdf`; that part is graded by a person.

## 6. The report (`report.pdf`)

An analysis, not a diary. **0.5–1 page of text plus Figures 1–3, ≤ 3 pages total.** Fill in
`report_template.md`, then export to PDF with any tool (VS Code/Typora export, `pandoc`,
Word/HWP). The PDF must be text-based, not a photo or a scan.

Required sections, in this order:

1. **What you did** — 2–3 sentences, only where you deviated from or extended the skeleton.
2. **Results** — Table 1: the key numbers from `results.json`.
3. **Figures** — Figures 1–3 from §3, embedded.
4. **Interpretation** — *why* the numbers and figures look the way they do.
5. **Limitations** — one thing that would change your conclusion.
6. **AI & external-code usage table** — every AI tool and external source you used, in the
   table at the end of `report_template.md`. An empty table means "I used nothing".

## 7. Questions

e-Class Q&A board (preferred, since answers benefit everyone — do not post solution code).
