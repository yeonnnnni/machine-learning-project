> Fill in this template, then export to **report.pdf** for submission
> (e.g., VS Code/Typora export, or `pandoc report.md -o report.pdf`).
>
> **Budget**: 0.5–1 page of text **plus** Figures 1–3, ≤ 3 pages total. Text-based PDF, not a scan.
>
> **Caption format is the same as the concept note**: number, a short title, then one or two
> sentences of explanation. Table captions go **above** the table, figure captions **below** the
> figure. Every figure and table must be referred to by number somewhere in your text — an
> unreferenced figure earns nothing.
>
> Delete these instruction lines before exporting.

# Lab 9 Report — {Student ID} {Name}

## 1. What I did

(2–3 sentences. Only what is not obvious from the skeleton.)

## 2. Results

**Table 1. Both generative models under identical data, seed, and evaluation.** (One sentence:
which number your interpretation in §4 hangs on.)

| Metric | GAN | DDPM |
|--------|-----|------|
| Mode coverage (of 8) |  |  |
| High-quality fraction |  |  |
| Sampling wall-clock for 2,000 samples (s) |  |  |
| Final training loss(es) | D: / G: |  |

## 3. Figures

(Export from `notebooks/lab09_visualization.ipynb` and paste the images here. Axis labels must
be legible.)

{image}

**Figure 1. Real data vs GAN samples vs DDPM samples.** (One or two sentences: which modes
each model produced, and how their spreads compare.)

{image}

**Figure 2. Training loss curves.** (One or two sentences: how the two
trainings behave over time, and what each curve does and does not tell you.)

{image}

**Figure 3. Per-mode sample share.** (One or two sentences: which bars fall below the coverage
threshold, and for which model.)

## 4. Interpretation

(Refer to figures and the table by number. For example: how do the two models compare on
coverage and on the high-quality fraction, and what does each metric reward? Could you have
predicted the coverage from the loss curves alone? How many network passes does each model spend
per sample, and where does that cost come from in `ddpm_sample`? What would change if
`MODE_SHARE_MIN` were doubled?)

## 5. Limitations

(One concrete thing that would change your conclusions, and why. What does one seed and one
budget say, and not say, about each family in general?)

## AI & external-code usage

(Required. Leave the table empty only if you used nothing — see README §6.)

| Tool / Source | Part used for | What I modified & verified myself |
|---------------|---------------|-----------------------------------|
|  |  |  |
