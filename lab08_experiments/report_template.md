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

# Lab 8 Report — {Student ID} {Name}

## 1. What I did

(2–3 sentences. Only what is not obvious from the skeleton.)

## 2. Results

**Table 1. Each factor changed on its own, five seeds each.** (One sentence: which column you read
first, and why the std column is there at all.)

| Variant | Mean F1 | Std F1 | Difference from base | Larger than the base spread? |
|---------|---------|--------|----------------------|------------------------------|
| base |  |  | 0.0000 | — |
| features=with_region |  |  |  |  |
| features=core |  |  |  |  |
| model=tree |  |  |  |  |
| class_weight=balanced |  |  |  |  |
| scale=False |  |  |  |  |

**Table 2. The base config sliced by region, seed 42.** (One sentence: what the `n` column changes
about how you read the accuracy column.)

| Region | n | Accuracy | Precision | Recall | F1 |
|--------|---|----------|-----------|--------|----|
| island |  |  |  |  |  |
| metro |  |  |  |  |  |
| overseas |  |  |  |  |  |
| rural |  |  |  |  |  |
| suburb |  |  |  |  |  |
| **overall** |  |  |  |  |  |

## 3. Figures

(Export from `notebooks/lab08_experiments.ipynb` and paste the images here. Axis labels must be
legible.)

{image}

**Figure 1. Accuracy per region against the overall accuracy.** (One or two sentences: which slice
is furthest below the line, and which slice you refuse to draw a conclusion from and why.)

{image}

**Figure 2. One factor changed at a time, mean ±1 std over five seeds.** (One or two sentences:
which bars clear the base band and which do not.)

{image}

**Figure 3. Five seeds each for the base and the challenger.** (One or two sentences: do the two
sets of points separate, and what does that allow you to claim?)

## 4. Interpretation

(Refer to the tables and figures by number. Which factor produced the largest change, and how many
base standard deviations is it? Which factor landed inside the seed noise, and what would you have
concluded from a single seed? Which region is worst among the slices large enough to judge, and does
any factor in the ablation address it? Which row of Table 2 would you refuse to draw a conclusion
from, and why? Find a region where accuracy and recall tell different stories, and explain how both
can be true.)

## 5. Limitations

(One concrete thing that would change your conclusions, and why.)

## AI & external-code usage

(Required. Leave the table empty only if you used nothing — see README §6.)

| Tool / Source | Part used for | What I modified & verified myself |
|---------------|---------------|-----------------------------------|
|  |  |  |
