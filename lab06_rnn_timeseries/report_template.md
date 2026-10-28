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

# Lab 6 Report — {Student ID} {Name}

## 1. What I did

(2–3 sentences. Only what is not obvious from the skeleton.)

## 2. Results

**Table 1. Both forecasters on the same test windows, one split, one seed.** (One sentence: the
persistence row is what the RNN has to beat, so say what it tells you before reading the rest.)

| Forecaster | MAE (cups) | RMSE (cups) | MASE |
|------------|------------|-------------|------|
| persistence (yesterday) |  |  |  |
| GRU (window 14) |  |  |  |

Split sizes (windows): train ___ / validation ___ / test ___.
Scaling constants, from the training segment only: mean ___ , std ___ .

## 3. Figures

(Export from `notebooks/lab06_visualization.ipynb` and paste the images here. Axis labels must be
legible.)

{image}

**Figure 1. The series and the chronological split.** (One or two sentences: where the three
segments sit, and what the two white strips between them are for.)

{image}

**Figure 2. Actual, persistence and RNN forecasts over the last 60 test days.** (One or two
sentences: where the persistence line is furthest from the truth, and whether those days have
anything in common.)

{image}

**Figure 3. Test error for the two forecasters.** (One or two sentences: what the MASE bars say
that the MAE bars do not.)

## 4. Interpretation

(Refer to the figures and Table 1 by number. Why is persistence already this good on this series?
Which days does it get wrong, and what does the RNN use to do better there — you can see it in
Figure 2. Why is MASE the number to quote when someone asks whether your MAE in cups is good? What
would have happened to all three numbers if you had shuffled the days before splitting?)

## 5. Limitations

(One concrete thing that would change your conclusions, and why.)

## AI & external-code usage

(Required. Leave the table empty only if you used nothing — see README §6.)

| Tool / Source | Part used for | What I modified & verified myself |
|---------------|---------------|-----------------------------------|
|  |  |  |
