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

# Lab 4-1 Report — {Student ID} {Name}

## 1. What I did

(2–3 sentences. Only what is not obvious from the skeleton.)

## 2. Results

**Table 1. Gradient check.** (One sentence: what the number is, and what it lets you conclude about
the analytic gradients before you look at any training curve.)

| Quantity | Value |
|----------|-------|
| max relative error, analytic vs numerical gradient |  |
| pass threshold | 1e-6 |

**Table 2. Three learning rates, one split, one seed, one initialization.** (One sentence: which
column separates the three runs most clearly, and why the final training loss alone is not enough.)
Note: `majority_baseline_accuracy` in `results.json` is computed over all 800 rows (0.7038). The
baseline row below is the majority rate of your **validation split**, which is the number the other
rows have to beat.

| Learning rate | Final train loss | Max train loss | Final val loss | Final val accuracy | Best val accuracy |
|---------------|------------------|----------------|----------------|--------------------|-------------------|
| 0.01 |  |  |  |  |  |
| 1.0 |  |  |  |  |  |
| 100.0 |  |  |  |  |  |
| majority-class baseline (validation split) | n/a | n/a | n/a |  | n/a |

## 3. Figures

(Export from `notebooks/lab04_1_visualization.ipynb` and paste the images here. Axis labels must be
legible.)

{image}

**Figure 1. The two press settings, coloured by outcome.** (One or two sentences: where the passing
parts sit, and what shape their region has.)

{image}

**Figure 2. Training loss against epoch for three learning rates.** (One or two sentences: how the
three curves differ in shape, not just in final value.)

{image}

**Figure 3. Decision boundary of the trained network.** (One or two sentences: how the boundary
compares with the region you described in Figure 1, and where it is wrong.)

## 4. Interpretation

(Refer to the figures and tables by number. Why does a rectangular acceptance region need a hidden
layer at all, and where do you see that in Figures 1 and 3? What does each of the three curves in
Figure 2 tell you about the step size, too small, usable, or too large? Why does the run at
lr = 100 end with a training loss that looks almost ordinary even though the run failed? Why is a
gradient check worth running before any of these experiments?)

## 5. Limitations

(One concrete thing that would change your conclusions — for example, a single split and a single
initialization, or the fact that only three learning rates were tried.)

## AI & external-code usage

(Required. Leave the table empty only if you used nothing — see README §6.)

| Tool / Source | Part used for | What I modified & verified myself |
|---------------|---------------|-----------------------------------|
|  |  |  |
