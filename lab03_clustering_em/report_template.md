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

# Lab 3 Report — {Student ID} {Name}

## 1. What I did

(2–3 sentences. Only what is not obvious from the skeleton.)

## 2. Results

**Table 1. Key numbers from `results.json`.** (One sentence: which number your interpretation
in §4 hangs on.)

| Metric | Value |
|--------|-------|
| ARI, K-means / GMM |  |
| K-means inertia |  |
| Log-likelihood, first / final / min consecutive change |  |
| BIC at k = 3 / 4 / 5 |  |
| Selected k (BIC minimum) |  |

## 3. Figures

(Export from `notebooks/lab03_visualization.ipynb` and paste the images here. Axis labels must be
legible.)

{image}

**Figure 1. The scene colored by K-means labels and by GMM labels.** (One or two sentences:
which objects each method separates cleanly, and where K-means cuts or merges them.)

{image}

**Figure 2. EM log-likelihood vs iteration.** (One or two sentences: the shape of the trace and
what the guarantee behind it is.)

{image}

**Figure 3. BIC vs number of components k.** (One or two sentences: where the minimum sits and
what the two sides of the curve trade off.)

## 4. Interpretation

(Refer to figures and the table by number. For example: which objects does K-means cut in half
or merge, and why does the isotropic distance force that? Why do full covariances fix exactly
those mistakes? Why does the log-likelihood alone keep improving as k grows, and what does the
BIC penalty add? What would the responsibilities look like for a point midway between two
objects?)

## 5. Limitations

(One concrete thing that would change your conclusions — for example, what happens to the
K-means-vs-GMM comparison if every object were a sphere, or to the BIC minimum if two objects
touched each other.)

## AI & external-code usage

(Required. Leave the table empty only if you used nothing — see README §6.)

| Tool / Source | Part used for | What I modified & verified myself |
|---------------|---------------|-----------------------------------|
|  |  |  |
