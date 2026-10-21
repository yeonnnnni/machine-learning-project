# Lab 5 — CNNs and Transfer Learning

**Machine Learning Project (53744-01) · Week 6 · Due: Wednesday, 21 October 2026,**
**11:59 PM (KST), via e-Class**
**No late submission is accepted. A submission after the deadline scores 0 points.**

> **Before the lab session, read the concept note in this folder**: `lab05_concepts.pdf`.
> It covers everything this lab assumes. Everything else specific to this lab is in this README.
> Course-wide policy — grading and AI use — was given in the policy slides of the first class.

## 1. Goal

Get the shape arithmetic right, build a small convolutional network, and reuse a **frozen
pretrained backbone** for a new task with few labels. Then compare the two models under
identical conditions and explain the difference.

Dataset: `data/parts_shapes.npz` — 1,200 (synthetic) 32×32 grayscale images of stamped
parts in 4 shape classes (`diamond`, `ellipse`, `ell`, `tee`). Only **240** of them are used
for training; the rest are validation. **No download needed.**

Pretrained weights: `data/backbone_shapes6.pt` — the `state_dict` of the same architecture
trained on a **different** task (6 other shape classes, 9,000 labelled images). You get the
weights, not that dataset. **No download needed.**

## 2. Environment

| Item | Value |
|------|-------|
| Python | 3.10.x |
| Install | `pip install -r requirements.txt` |
| Hardware | CPU only (no GPU, no CUDA) |
| Expected runtime | `python src/lab05.py` a few seconds; `pytest tests/` ≈ 10 s |
| Expected effort | 4–5 hours |
| Memory | < 1 GB |
| Seed | 42 (already set in the skeleton) |

Your code must run top to bottom with a single command on a clean machine that has only
`requirements.txt` installed.

**Install exactly what `requirements.txt` pins.** Two pins are there on purpose:

- `numpy<2` — the PyTorch build that `pip` picks on some machines (notably Intel Macs) is compiled
  against NumPy 1.x, and pairing it with NumPy 2 breaks every conversion between arrays and tensors.
- `scipy<1.15` — on a macOS 27 (Apple silicon) machine we tested, the SciPy 1.15 wheels for
  Python 3.10 failed to load, so `import sklearn` crashed before any lab code ran.

If your environment already has newer versions, `pip install -r requirements.txt` will downgrade
them — let it.

## 3. Tasks

Complete the four TODO blocks in `src/lab05.py` (full specs in docstrings):

| Task | Function | Points share |
|------|----------|--------------|
| 1 | `conv_output_shape` — output height and width of a conv or pool layer | ~15% of correctness |
| 2 | `SmallCNN` + `build_cnn` — conv/ReLU/pool ×3, flatten, one linear head | ~30% |
| 3 | `freeze_backbone` + `build_transfer_model` — load the backbone, freeze it, keep a fresh head | ~30% |
| 4 | `run_experiment` — baseline vs scratch vs transfer under identical conditions | ~25% |

**Performance target.** This lab has a fixed baseline for the transfer model: the from-scratch
CNN of the reference implementation, which reaches **0.8240** validation accuracy. Your
frozen-backbone model has to beat that, and the reference transfer model reaches **0.9469**.
Both are absolute numbers, the same for everyone. They are not a ranking against your classmates,
and not something to chase by changing the epochs, learning rate, or split the skeleton fixes.

Then open `notebooks/lab05_visualization.ipynb` — a **working tool, not a deliverable**. It
imports your finished functions from `src/lab05.py`, so it only runs once your implementation is
correct. Launch Jupyter from inside `notebooks/`; the first cell uses relative paths like
`../src`. Complete the three plot cells and **export the figures into your report**:

| Report figure | Content |
|---------------|---------|
| Figure 1 | a grid of sample images, one row per class |
| Figure 2 | validation accuracy against the number of labelled training images (60–900, 5 budgets — the notebook loop retrains both models per budget), plus a dashed line at the majority-class baseline |
| Figure 3 | confusion matrices of the two models side by side |

You do **not** submit the notebook — only the figures, inside `report.pdf`.

**Skeleton rules** — the automated tests depend on these:

- Write your code **only inside the TODO blocks**. You may add private helper functions.
- **Do not rename functions or change their arguments and return types.** The tests call them
  directly; a renamed function scores 0 on those tests. The attribute names `features` and
  `head` are part of the contract too: the pretrained weights are keyed by them.
- Do **not** build a layer inside `conv_output_shape` to read the shape off a dummy tensor.
  A hidden test calls it with an input far too large to allocate.
- Do not modify anything marked `# DO NOT MODIFY` (`set_seed()`, `load_data()`, `split_data()`,
  `make_optimizer()`, `train_model()`, `predict()`, `accuracy()`, `count_trainable()`,
  `main()`) or anything in `tests/`.
- **Do not hard-code answers.** Hidden tests run on different data and with a different number
  of classes, so a value copied from the shipped dataset will fail there. Hard-coding is
  treated as academic dishonesty.
- All logic goes in the `.py` file. It is the only code you submit and the only code graded.

## 4. Run & self-check

Fill in `STUDENT_ID` / `STUDENT_NAME` at the top of `src/lab05.py`, then:

```bash
pip install -r requirements.txt
python src/lab05.py            # writes results.json
python -m pytest tests/ -q     # 6 public tests (hidden tests run at grading)
```

Run `python src/lab05.py` twice. The numbers in `results.json` must be identical both times,
apart from `runtime_seconds`. If they are not, something in your code is unseeded.

## 5. What to submit

Submit **exactly one zip file**, named:

```
{student_id}_{name}_lab05.zip        e.g., 20261234_홍길동_lab05.zip
                                          20261234_HongGildong_lab05.zip
```

Your name in **Korean or roman letters**, written as one word — **no spaces, digits, or symbols**
(`Hong Gildong`, `Hong-Gildong`, `hong2` all fail the checker).

containing **exactly these four items at the top level** — nothing else:

```
20261234_HongGildong_lab05.zip
├── src/            # lab05.py (your completed .py files ONLY)
├── report.pdf      # analysis + Figures 1-3 (see §3) + AI-usage table
├── results.json    # written by `python src/lab05.py` — do not edit by hand
└── README.md       # the exact commands to reproduce your results
```

| # | Item | Where it comes from | What must be true |
|---|------|---------------------|-------------------|
| 1 | `src/` | your edited `src/lab05.py` | `.py` only. Function names, signatures, and return types unchanged. No notebooks, no data, no `.pt` weights. |
| 2 | `results.json` | generated by `python src/lab05.py` | `student_id` matches the zip name, `seed` is 42, not hand-edited |
| 3 | `report.pdf` | fill in `report_template.md`, then export to PDF | ≤ 3 pages, text-based (searchable) — not a photo or scan. Figures 1–3 embedded, each referred to by number in your text. |
| 4 | `README.md` | you write it (a NEW file — not this instruction sheet) | the exact commands a grader runs to reproduce your numbers, plus anything they must know |

The **AI & external-code usage table is inside `report.pdf`** (last section of
`report_template.md`). Leaving it empty means "I used nothing", and you will be held to that.

**Do not include**: notebooks (`.ipynb`), the dataset, the `.pt` weight file, virtual
environments, `__pycache__`, `.DS_Store`, or any file over 10 MB. The notebook is a working
tool, not a deliverable — its figures belong in `report.pdf`, where they are graded.

Validate before submitting:

```bash
python check_submission.py 20261234_HongGildong_lab05.zip
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
