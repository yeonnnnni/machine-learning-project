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

# Lab 5 Report — {Student ID} {Name}

## 1. What I did

(2–3 sentences. Only what is not obvious from the skeleton.)

## 2. Results

**Table 1. Both models under one split, one seed, one training budget.** (One sentence: the
baseline row is what the other two have to beat, and the parameter column is the second axis of
the comparison, so say what the table is meant to show before the numbers.)

| Model | Validation accuracy | Trainable parameters |
|-------|---------------------|----------------------|
| baseline (most frequent class) |  | 0 |
| CNN from scratch |  |  |
| frozen backbone + new head |  |  |

## 3. Figures

(Export from `notebooks/lab05_visualization.ipynb` and paste the images here. Axis labels must be
legible.)

{image}

**Figure 1. Sample images, one row per class.** (One or two sentences: what varies between images
of the same class, and which two classes you would expect the model to confuse.)

{image}

**Figure 2. Accuracy against the label budget.** (One or two sentences: how the two curves move
as labels are added, and where they would meet.)

{image}

**Figure 3. Confusion matrices, scratch against transfer.** (One or two sentences: which pair of
classes carries most of the errors, and whether Figure 1 explains it.)

## 4. Interpretation

(Refer to the figures and the table by number. What fraction of the weights does the transfer
model update, and how does its accuracy compare with the scratch model's? Explain the difference.
What accuracy does the baseline reach, and what does that number say about the classes? Where in
the network does the gradient stop for the transfer model, and what would change if you unfroze
the backbone? Which classes does Figure 3 say are hard, and does Figure 1 explain that?)

## 5. Limitations

(One concrete thing that would change your conclusions. Figure 2 hands you one: say what the gap
does as labels are added, and what that implies about when transfer learning is worth using.)

## AI & external-code usage

(Required. Leave the table empty only if you used nothing — see README §6.)

| Tool / Source | Part used for | What I modified & verified myself |
|---------------|---------------|-----------------------------------|
|  |  |  |
