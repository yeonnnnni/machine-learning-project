# Lab 9 — GANs and Diffusion Models

**Machine Learning Project (53744-01) · Week 11 · Due: Wednesday, 25 November 2026,**
**11:59 PM (KST), via e-Class**
**No late submission is accepted. A submission after the deadline scores 0 points.**

> **Before the lab session, read the concept note in this folder**: `lab09_concepts.pdf`.
> It covers everything this lab assumes. Everything else specific to this lab is in this README.
> Course-wide policy — grading and AI use — was given in the policy slides of the first class.

## 1. Goal

A generative model learns to **sample** from a distribution, not to classify its points. In
this lab you build the training core of the two families behind today's generative AI
systems (image generation, and the language model of Lab 10): a **GAN**, which learns from
adversarial feedback and can silently drop modes, and a **DDPM diffusion model**, which
learns to denoise every noise level and covers all modes at the cost of a slow, iterative
sampler. The data is a 2D ring of 8 Gaussians — small enough that both behaviours are
visible in a scatter plot and measurable: you train both models under fixed budgets and
compare mode coverage, high-quality sample fraction, and sampling wall-clock. Grading
rewards a correct, exactly reproducible implementation and an analysis that explains the
coverage and quality you measure for each family, and what each family pays for its strength.

Dataset: `data/lab09_ring.npz` — 2,000 points (250 per mode) from 8 Gaussians of standard
deviation 0.03 placed every 45° on a circle of radius 2, plus the 8 true `mode_centers`
(**for evaluation only** — training must never read them; hidden tests check this).
**No download needed; the data ships in this folder.**

## 2. Environment

| Item | Value |
|------|-------|
| Python | 3.10.x |
| Install | `pip install -r requirements.txt` |
| Hardware | CPU only (PyTorch CPU build is enough) |
| Expected runtime | < 1 minute (`python src/lab09.py` trains both models in about 10 seconds) |
| Expected effort | 4–5 hours |
| Memory | < 1 GB |
| Seed | 42 (already set in the skeleton) |

**Environment self-check** — run this line before writing any code:

```bash
python -c "import numpy, torch; print('numpy', numpy.__version__, '/ torch', torch.__version__)"
```

The three small networks (`GenNet`, `DiscNet`, `DenoiseNet`) are provided complete — this
lab is about the *training steps* and the *diffusion math*, not architecture design. Every
docstring states tensor shapes explicitly: `(B, 2)` means B points in the xy-plane.

Your code must run top to bottom with a single command on a clean machine that has only
`requirements.txt` installed.

**Install exactly what `requirements.txt` pins.** It holds `numpy<2` on purpose: the PyTorch build
that `pip` picks on some machines (notably Intel Macs) is compiled against NumPy 1.x, and pairing it
with NumPy 2 breaks every conversion between arrays and tensors. If you already have NumPy 2 in the
environment, `pip install -r requirements.txt` will downgrade it — let it.

## 3. Tasks

Complete the four TODO blocks in `src/lab09.py` (full specs in the docstrings):

| Task | Function | Points share |
|------|----------|--------------|
| 1 | `gan_step(G, D, optG, optD, real_batch, rng_noise)` — one alternating GAN update: D on real + **detached** fake with BCE, then G with the **non-saturating** loss | ~25% of correctness |
| 2 | `make_schedule(T, beta_min, beta_max)` + `q_sample(x0, t, noise, alphabar)` — linear β schedule with α and ᾱ (t ∈ {0..T−1}), and the closed-form forward noising √ᾱ·x₀ + √(1−ᾱ)·ε | ~25% |
| 3 | `ddpm_train_loss(model, x0_batch, t_batch, noise)` + `ddpm_sample(model, n, rng)` — the ε-prediction MSE and the reverse ancestral loop (σₜ = √βₜ, **no noise at t = 0**) | ~25% |
| 4 | `run_experiment(data, rng)` — train both models under the fixed budgets, sample from each, measure mode coverage, high-quality fraction, and sampling wall-clock; feeds `results.json` | ~25% |

Then open `notebooks/lab09_visualization.ipynb` — a **working tool, not a deliverable**. It
imports your finished functions from `src/lab09.py`, so it only runs once your implementation is
correct. Launch Jupyter from inside `notebooks/`; the first cell uses relative paths like
`../src`. Complete the three plot cells and **export the figures into your report**:

| Report figure | Content |
|---------------|---------|
| Figure 1 | real data vs GAN samples vs DDPM samples, three scatter panels side by side |
| Figure 2 | training loss curves: discriminator and generator, and the DDPM denoising loss |
| Figure 3 | per-mode sample share of both models as a bar chart, with the coverage threshold marked |

You do **not** submit the notebook — only the figures, inside `report.pdf`.

**Skeleton rules** — the automated tests depend on these:

- Write your code **only inside the TODO blocks**. You may add private helper functions.
- **Do not rename functions or change their arguments and return types.** The tests call them
  directly; a renamed function scores 0 on those tests.
- Do **not** import `cv2`, `open3d`, `scipy`, `pandas`, `sklearn`, `skimage`, `PIL`, `numba`, or `kornia` in
  `src/lab09.py` — numpy and torch are all the graded path needs (checked at grading time).
- Keep the fixed constants (`SEED`, `Z_DIM`, `T_DIFF`, `BETA_MIN`, `BETA_MAX`, `BATCH`,
  `GAN_STEPS`, `DDPM_STEPS`, `LR_GAN`, `LR_DDPM`, `N_GEN`, `LOG_EVERY`, `HQ_RADIUS`,
  `MODE_SHARE_MIN`) — the comparison is only fair because both models keep their fixed
  budgets, and the tests check every constant.
- `data["mode_centers"]` is **for evaluation only** (`mode_stats` inside `run_experiment`).
  Using the centers anywhere in training — or snapping samples toward them — is caught by
  hidden tests and treated as hard-coding.
- Do not modify anything marked `# DO NOT MODIFY` (`set_seed()`, `load_data()`, `GenNet`,
  `DiscNet`, `DenoiseNet`, `mode_stats()`, `main()`) or anything in `tests/`.
- **Do not hard-code answers.** Hidden tests run on different data, so a value copied from the
  shipped dataset will fail there. Hard-coding is treated as academic dishonesty.
- All logic goes in the `.py` file. It is the only code you submit and the only code graded.


## 4. Run & self-check

Fill in `STUDENT_ID` / `STUDENT_NAME` at the top of `src/lab09.py`, then:

```bash
pip install -r requirements.txt
python src/lab09.py            # writes results.json
python -m pytest tests/ -q     # 6 public tests (hidden tests run at grading)
```

Running `src/lab09.py` twice must reproduce every metric byte-for-byte, with one exception:
`runtime_seconds` and the two `*_sample_seconds` entries are wall-clock measurements and will
differ between runs. The tests exclude exactly those fields from the determinism check.

## 5. What to submit

Submit **exactly one zip file**, named:

```
{student_id}_{name}_lab09.zip        e.g., 20261234_홍길동_lab09.zip
                                          20261234_HongGildong_lab09.zip
```

Your name in **Korean or roman letters**, written as one word — **no spaces, digits, or symbols**
(`Hong Gildong`, `Hong-Gildong`, `hong2` all fail the checker).

containing **exactly these four items at the top level** — nothing else:

```
20261234_HongGildong_lab09.zip
├── src/            # lab09.py (your completed .py files ONLY)
├── report.pdf      # analysis + Figures 1-3 (see §3) + AI-usage table
├── results.json    # written by `python src/lab09.py` — do not edit by hand
└── README.md       # the exact commands to reproduce your results
```

| # | Item | Where it comes from | What must be true |
|---|------|---------------------|-------------------|
| 1 | `src/` | your edited `src/lab09.py` | `.py` only. Function names, signatures, and return types unchanged. No notebooks, no data. |
| 2 | `results.json` | generated by `python src/lab09.py` | `student_id` matches the zip name, `seed` is 42, not hand-edited |
| 3 | `report.pdf` | fill in `report_template.md`, then export to PDF | ≤ 3 pages, text-based (searchable) — not a photo or scan. Figures 1–3 embedded, each referred to by number in your text. |
| 4 | `README.md` | you write it (a NEW file — not this instruction sheet) | the exact commands a grader runs to reproduce your numbers, plus anything they must know |

The **AI & external-code usage table is inside `report.pdf`** (last section of
`report_template.md`). Leaving it empty means "I used nothing", and you will be held to that.

**Do not include**: notebooks (`.ipynb`), the dataset, virtual environments, `__pycache__`,
`.DS_Store`, or any file over 10 MB. The notebook is a working tool, not a deliverable —
its figures belong in `report.pdf`, where they are graded.

Validate before submitting:

```bash
python check_submission.py 20261234_HongGildong_lab09.zip
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
