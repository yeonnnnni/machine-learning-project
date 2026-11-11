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

# Lab 7 Report — {Student ID} {Name}

## 1. What I did

(2–3 sentences. Only what is not obvious from the skeleton.)

## 2. Results

**Table 1. Three configurations, one dataset, one seed, one training recipe.** (One sentence: which
factor changes between the rows, and what stays fixed.)

| Configuration | Attention | Positional encoding | Val accuracy | Val loss |
|---------------|-----------|---------------------|--------------|----------|
| `mean_pool` | no | no |  |  |
| `encoder_nopos` | yes | no |  |  |
| `encoder_pos` | yes | yes |  |  |

## 3. Figures

(Export from `notebooks/lab07_visualization.ipynb` and paste the images here. Axis labels must be
legible.)

{image}

**Figure 1. The positional encoding matrix.** (One or two sentences: what the axes are, and why
neighbouring rows look similar while distant rows do not.)

{image}

**Figure 2. Attention weights for one validation log.** (One or two sentences: which row and column
you are looking at, and where the padded positions went.)

{image}

**Figure 3. Validation accuracy of the three configurations.** (One or two sentences: where the
chance line sits, and how each bar compares with it.)

## 4. Interpretation

(Refer to the figures and the table by number. Which configurations stay near the chance line,
and is that a training failure or something the architecture cannot represent? What
exactly does adding the positional encoding change about the function the model can compute? In
Figure 2, do the attention weights on the padded positions match what the mask should do? Why is
the scores matrix divided by `sqrt(d_k)`, and what would you expect to see in training if you
removed that division?)

## 5. Limitations

(One concrete thing that would change your conclusions, and why.)

## AI & external-code usage

(Required. Leave the table empty only if you used nothing — see README §6.)

| Tool / Source | Part used for | What I modified & verified myself |
|---------------|---------------|-----------------------------------|
|  |  |  |
