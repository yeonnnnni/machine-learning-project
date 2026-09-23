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

# Lab 2 Report — {Student ID} {Name}

## 1. What I did

(2–3 sentences. Only what is not obvious from the skeleton.)

## 2. Results

**Table 1. All three models under one split, one seed, one metric definition.** (One sentence: the
baseline row is what the other two have to beat, so say what it tells you before reading the rest.)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| baseline (majority) |  |  |  |  |
| logistic regression |  |  |  |  |
| kNN (k=15) |  |  |  |  |

**Table 2. Regression on `wait_minutes`, same discipline.** (One sentence: what the mean-predicting
baseline's R² ≈ 0 means, and what beating it proves.)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| baseline (mean) |  |  |  |
| linear regression |  |  |  |
| kNN (k=15) |  |  |  |

## 3. Figures

(Export from `notebooks/lab02_visualization.ipynb` and paste the images here. Axis labels must be
legible.)

{image}

**Figure 1. Class balance of `no_show`.** (One or two sentences: the positive rate, and what it
predicts about the baseline's accuracy before you train anything.)

{image}

**Figure 2. `days_ahead` distribution by class.** (One or two sentences: does this feature separate
the two classes, and how much do they overlap?)

{image}

**Figure 3. Accuracy and F1 by model.** (One or two sentences: what the baseline bar tells you about
the other two, and why the two metrics disagree.)

## 4. Interpretation

(Refer to figures and the table by number. Why is baseline accuracy already around 0.65, and where
do you see that in Figure 1? Why is its F1 exactly 0? Which model wins on F1, and does Figure 2 help
explain why? Why must the scaler live inside the Pipeline rather than run before the split?
In Table 2, why is the baseline's R² close to 0 by construction, and why does linear regression
beat kNN on this particular dataset?)

## 5. Limitations

(One concrete thing that would change your conclusions — for example, a single split versus
cross-validation.)

## AI & external-code usage

(Required. Leave the table empty only if you used nothing — see README §6.)

| Tool / Source | Part used for | What I modified & verified myself |
|---------------|---------------|-----------------------------------|
|  |  |  |
