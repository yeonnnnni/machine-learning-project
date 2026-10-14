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

# Lab 4-2 Report — {Student ID} {Name}

## 1. What I did

(2–3 sentences. Only what is not obvious from the skeleton.)

## 2. Results

**Table 1. Six configurations scored on validation, one of them scored on test.** (One sentence:
say which row was selected and on which number, before the reader compares the columns.)

| Hidden size | Learning rate | Best epoch | Epochs run | Validation loss | Validation accuracy |
|-------------|---------------|------------|------------|-----------------|---------------------|
| 8 | 0.01 |  |  |  |  |
| 8 | 0.1 |  |  |  |  |
| 8 | 1.0 |  |  |  |  |
| 32 | 0.01 |  |  |  |  |
| 32 | 0.1 |  |  |  |  |
| 32 | 1.0 |  |  |  |  |

Selected configuration: hidden size ____, learning rate ____.
Test loss ____, test accuracy ____ (measured once, after the table above was complete).

## 3. Figures

(Export from `notebooks/lab04_2_visualization.ipynb` and paste the images here. Axis labels must be
legible.)

{image}

**Figure 1. Training and validation loss of the selected configuration.** (One or two sentences:
where the two curves separate, and which epoch early stopping kept.)

{image}

**Figure 2. Validation loss across the grid.** (One or two sentences: which factor moves the loss
more, the learning rate or the hidden size, and by how much.)

{image}

**Figure 3. Validation against test for the selected configuration.** (One or two sentences: the
size and the direction of the gap.)

## 4. Interpretation

(Refer to figures and the table by number. Two rows of your table should land at exactly the
accuracy of a model that never predicts a service call: which rows are they, what is that number
on your validation split, and what does a learning rate have to do with it? In Figure 1, what
would have happened to the validation loss if training had run to the full epoch budget? Explain
what selecting on validation loss, rather than on validation accuracy, gains you here.
Why is the test number in Figure 3 worse than the validation number for the very same weights?)

## 5. Limitations

(One concrete thing that would change your conclusions — for example, the best hidden size sitting
at the edge of the grid, or a single validation split rather than cross-validation.)

## AI & external-code usage

(Required. Leave the table empty only if you used nothing — see README §6.)

| Tool / Source | Part used for | What I modified & verified myself |
|---------------|---------------|-----------------------------------|
|  |  |  |
