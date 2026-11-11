# Lab 7 — Implementing a Transformer

**Machine Learning Project (53744-01) · Week 9 · Due: Wednesday, 11 November 2026,**
**11:59 PM (KST), via e-Class**
**No late submission is accepted. A submission after the deadline scores 0 points.**

> **Before the lab session, read the concept note in this folder**: `lab07_concepts.pdf`.
> It covers everything this lab assumes. Everything else specific to this lab is in this README.
> Course-wide policy — grading and AI use — was given in the policy slides of the first class.

## 1. Goal

Build the attention mechanism **from scratch** in PyTorch: scaled dot-product attention with
masking, multi-head attention, sinusoidal positional encoding, and one encoder block. Then run a
controlled ablation that shows what each piece buys you. Attention is a short sequence of matrix
operations you can check by hand, and this lab checks them.

Dataset: `data/machine_events.npz` — 4,000 (synthetic) machine event logs, 8–16 events each,
padded to 16, vocabulary of 12 token ids. Every log contains exactly one heat alarm (token 1) and
one pressure alarm (token 2); the label is 1 when the **heat alarm came first**. **No download needed.**

## 2. Environment

| Item | Value |
|------|-------|
| Python | 3.10.x |
| Install | `pip install -r requirements.txt` |
| Hardware | CPU only |
| Expected runtime | `python src/lab07.py` a few seconds for all three models; `pytest tests/` ≈ 10 s |
| Expected effort | 4–5 hours |
| Memory | < 1 GB |
| Seed | 42 (already set in the skeleton) |

Your code must run top to bottom with a single command on a clean machine that has only
`requirements.txt` installed.

**Install exactly what `requirements.txt` pins.** It holds `numpy<2` on purpose: the PyTorch build
that `pip` picks on some machines (notably Intel Macs) is compiled against NumPy 1.x, and pairing it
with NumPy 2 breaks every conversion between arrays and tensors. If you already have NumPy 2 in the
environment, `pip install -r requirements.txt` will downgrade it — let it.

Two runs in the same environment must give identical numbers. Across PyTorch versions, processors,
or thread counts the accuracies can move by about 0.01 and the losses from the second decimal.
That is expected and not graded as an error.

## 3. Tasks

Complete the four TODO blocks in `src/lab07.py` (full specs in docstrings):

| Task | Function | Points share |
|------|----------|--------------|
| 1 | `scaled_dot_product_attention` — scores, `1/sqrt(d_k)`, mask, softmax, weighted sum | ~30% of correctness |
| 2 | `MultiHeadAttention` — `split_heads`, `merge_heads`, `forward` | ~30% |
| 3 | `positional_encoding` + `EncoderBlock.forward` — post-norm residual order | ~25% |
| 4 | `run_experiment` — `mean_pool` vs `encoder_nopos` vs `encoder_pos` | ~15% |

Then open `notebooks/lab07_visualization.ipynb` — a **working tool, not a deliverable**. It
imports your finished functions from `src/lab07.py`, so it only runs once your implementation is
correct. Launch Jupyter from inside `notebooks/`; the first cell uses relative paths like
`../src`. Complete the three plot cells and **export the figures into your report**:

| Report figure | Content |
|---------------|---------|
| Figure 1 | the positional encoding matrix, positions × dimensions |
| Figure 2 | the attention weight matrix of one validation log, queries × keys |
| Figure 3 | validation accuracy of the three configurations, with the chance line |

You do **not** submit the notebook — only the figures, inside `report.pdf`.

**Skeleton rules** — the automated tests depend on these:

- Write your code **only inside the TODO blocks**. You may add private helper functions.
- **Do not rename functions or change their arguments and return types.** The tests call them
  directly; a renamed function scores 0 on those tests. The same holds for the attribute names
  created in the two `__init__` methods that are given to you (`w_q`, `w_k`, `w_v`, `w_o`,
  `attn`, `ff`, `norm1`, `norm2`).
- Do **not** use `torch.nn.MultiheadAttention`, `torch.nn.TransformerEncoderLayer`,
  `torch.nn.TransformerEncoder`, or `torch.nn.functional.scaled_dot_product_attention`.
  Implementing them is the lab; the hidden tests compare your modules against PyTorch's.
- Do not modify anything marked `# DO NOT MODIFY` (`set_seed()`, `load_data()`, `masked_mean()`,
  `SeqClassifier`, `train_model()`, `main()`), the constants above the tasks, or anything
  in `tests/`.
- **Do not hard-code answers.** Hidden tests run on different data, so a value copied from the
  shipped dataset will fail there. Hard-coding is treated as academic dishonesty.
- All logic goes in the `.py` file. It is the only code you submit and the only code graded.

## 4. Run & self-check

Fill in `STUDENT_ID` / `STUDENT_NAME` at the top of `src/lab07.py`, then:

```bash
pip install -r requirements.txt
python src/lab07.py            # writes results.json
python -m pytest tests/ -q     # 6 public tests (hidden tests run at grading)
```

Run `python src/lab07.py` twice. The numbers in `results.json` must be **identical** both times
except `runtime_seconds`. If they are not, your seeding is wrong and the hidden determinism test
will fail.

## 5. What to submit

Submit **exactly one zip file**, named:

```
{student_id}_{name}_lab07.zip        e.g., 20261234_홍길동_lab07.zip
                                          20261234_HongGildong_lab07.zip
```

Your name in **Korean or roman letters**, written as one word — **no spaces, digits, or symbols**
(`Hong Gildong`, `Hong-Gildong`, `hong2` all fail the checker).

containing **exactly these four items at the top level** — nothing else:

```
20261234_HongGildong_lab07.zip
├── src/            # lab07.py (your completed .py files ONLY)
├── report.pdf      # analysis + Figures 1-3 (see §3) + AI-usage table
├── results.json    # written by `python src/lab07.py` — do not edit by hand
└── README.md       # the exact commands to reproduce your results
```

| # | Item | Where it comes from | What must be true |
|---|------|---------------------|-------------------|
| 1 | `src/` | your edited `src/lab07.py` | `.py` only. Function names, signatures, and return types unchanged. No notebooks, no data. |
| 2 | `results.json` | generated by `python src/lab07.py` | `student_id` matches the zip name, `seed` is 42, not hand-edited |
| 3 | `report.pdf` | fill in `report_template.md`, then export to PDF | ≤ 3 pages, text-based (searchable) — not a photo or scan. Figures 1–3 embedded, each referred to by number in your text. |
| 4 | `README.md` | you write it (a NEW file — not this instruction sheet) | the exact commands a grader runs to reproduce your numbers, plus anything they must know |

The **AI & external-code usage table is inside `report.pdf`** (last section of
`report_template.md`). Leaving it empty means "I used nothing", and you will be held to that.

**Do not include**: notebooks (`.ipynb`), the dataset, virtual environments, `__pycache__`,
`.DS_Store`, or any file over 10 MB. The notebook is a working tool, not a deliverable —
its figures belong in `report.pdf`, where they are graded.

Validate before submitting:

```bash
python check_submission.py 20261234_HongGildong_lab07.zip
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
