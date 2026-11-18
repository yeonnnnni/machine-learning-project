"""Lab 8 — Experiment management and error analysis.

Machine Learning Project (53744-01), Fall 2026.

The model here is deliberately cheap. What is graded is the machinery around
it: a runner that produces one record per run, an ablation that changes one
factor at a time, a per-slice error analysis, and an aggregation that reports
mean and spread over repeated seeds.

Fill in every TODO block. Do NOT rename functions or change their
signatures/return types — automated (public + hidden) tests call them directly.
Run:        python src/lab08.py
Self-check: python -m pytest tests/ -q
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---- Fill in your information (used in results.json) ----
STUDENT_ID = "00000000"   # TODO: your student id
STUDENT_NAME = "None"     # TODO: your name in Korean or roman letters — "홍길동" / "HongGildong"

SEED = 42  # fixed for the whole course — DO NOT CHANGE
SEEDS = (42, 43, 44, 45, 46)  # the seeds every configuration is repeated over
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "delivery_late.csv"
TARGET = "late"
GROUP_COLUMN = "region"

NUMERIC_FEATURES = ["distance_km", "parcel_weight_kg", "items", "dispatch_hour",
                    "courier_load", "weather_index", "is_weekend"]
CORE_FEATURES = ["distance_km", "parcel_weight_kg", "items"]

# The configuration every experiment starts from. Its keys are the only keys a
# config may have; run_config rejects anything else.
BASE_CONFIG = {
    "model": "logreg",       # "logreg" | "tree"
    "features": "numeric",   # "core" | "numeric" | "with_region"
    "scale": True,           # standardize inside the pipeline or not
    "class_weight": None,    # None | "balanced"
}

# One-factor-at-a-time ablation: key -> the alternative values to try.
FACTORS = {
    "features": ["with_region", "core"],
    "model": ["tree"],
    "class_weight": ["balanced"],
    "scale": [False],
}

# The single change compared against the base config over all SEEDS in main().
CHALLENGER_CONFIG = {**BASE_CONFIG, "class_weight": "balanced"}


def set_seed(seed: int = SEED) -> None:
    """DO NOT MODIFY."""
    np.random.seed(seed)


def load_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """DO NOT MODIFY. Returns (X, y, groups).

    X holds the numeric features AND one-hot columns named ``region_*``.
    A column being present in X does not mean a model uses it — the feature set
    named in the config decides that (see feature_columns).
    groups is the raw ``region`` column, used for the per-slice error analysis.
    """
    df = pd.read_csv(path)
    dummies = pd.get_dummies(df[GROUP_COLUMN], prefix="region", drop_first=True).astype(int)
    X = pd.concat([df[NUMERIC_FEATURES], dummies], axis=1)
    return X, df[TARGET], df[GROUP_COLUMN]


def feature_columns(feature_set: str, X: pd.DataFrame) -> list[str]:
    """DO NOT MODIFY. Column names selected by a feature-set name."""
    if feature_set == "core":
        return list(CORE_FEATURES)
    if feature_set == "numeric":
        return list(NUMERIC_FEATURES)
    if feature_set == "with_region":
        return list(NUMERIC_FEATURES) + [c for c in X.columns if c.startswith("region_")]
    raise ValueError(f"unknown feature set: {feature_set}")


def build_estimator(config: dict, seed: int = SEED) -> Pipeline:
    """DO NOT MODIFY. Turns a config dict into an UNFITTED sklearn Pipeline.

    Everything random inside the estimator is tied to ``seed``, so a run is
    fully described by (config, seed). The tree looks at a random 70% of the
    features at each split, so its fit really does move with the seed.
    """
    if config["model"] == "logreg":
        clf = LogisticRegression(random_state=seed, max_iter=1000,
                                 class_weight=config["class_weight"])
    elif config["model"] == "tree":
        clf = DecisionTreeClassifier(random_state=seed, max_depth=4, max_features=0.7,
                                     class_weight=config["class_weight"])
    else:
        raise ValueError(f"unknown model: {config['model']}")
    steps = [("scaler", StandardScaler())] if config["scale"] else []
    steps.append(("clf", clf))
    return Pipeline(steps)


def split_data(X: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int = SEED):
    """DO NOT MODIFY. Stratified 75/25 holdout split (this was Lab 2, Task 1).

    Returns (X_train, X_val, y_train, y_val, groups_train, groups_val).
    """
    return train_test_split(X, y, groups, test_size=0.25, random_state=seed,
                            shuffle=True, stratify=y)


def compute_metrics(y_true, y_pred) -> dict:
    """DO NOT MODIFY. The four metrics from Lab 2, Task 2.

    Returns {"accuracy", "precision", "recall", "f1"} as plain Python floats
    rounded to 4 decimals. Precision, recall and f1 are 0.0 when their
    denominator is 0.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {k: float(round(v, 4)) for k, v in
            {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}.items()}


def config_id(config: dict) -> str:
    """DO NOT MODIFY. A stable, readable name for a configuration.

    ``{"model": "tree", "scale": False}`` -> ``"model=tree|scale=False"``.
    Keys are sorted, so two dicts with the same content get the same id.
    """
    return "|".join(f"{k}={config[k]}" for k in sorted(config))


def base_predictions(seed: int = SEED):
    """DO NOT MODIFY. Validation-set predictions of BASE_CONFIG.

    Returns (y_val, y_pred, groups_val) as numpy arrays, ready for
    error_analysis. The notebook uses this too.
    """
    X, y, groups = load_data(DATA_PATH)
    X_train, X_val, y_train, y_val, _, groups_val = split_data(X, y, groups, seed)
    cols = feature_columns(BASE_CONFIG["features"], X)
    est = build_estimator(BASE_CONFIG, seed).fit(X_train[cols], y_train)
    return np.asarray(y_val), est.predict(X_val[cols]), np.asarray(groups_val)


# ================= TODO (Task 1): one run in, one record out =================
def run_config(config: dict, seed: int = SEED, data=None) -> dict:
    """Train and evaluate ONE configuration with ONE seed, return ONE record.

    Steps:
      1. Raise ValueError if ``set(config) != set(BASE_CONFIG)``. A config with
         a missing or misspelled key must fail loudly, not silently fall back
         to a default.
      2. If ``data`` is None, load it with ``load_data(DATA_PATH)``. Otherwise
         ``data`` is the ``(X, y, groups)`` tuple to use as-is.
      3. Start the clock with ``time.perf_counter()``.
      4. ``X_train, X_val, y_train, y_val, _, _ = split_data(X, y, groups, seed)``
      5. ``cols = feature_columns(config["features"], X)``
      6. ``est = build_estimator(config, seed)``, fit on ``(X_train[cols],
         y_train)``, predict on ``X_val[cols]``.
      7. ``metrics = compute_metrics(y_val, y_pred)``

    Returns a dict with EXACTLY these keys:

      ================  =====================================================
      "config_id"       config_id(config)
      "config"          dict(config), a COPY (the caller must not be able to
                        mutate the record by changing its own dict)
      "seed"            int(seed)
      "metrics"         the compute_metrics dict
      "n_train"         number of training rows, int
      "n_val"           number of validation rows, int
      "runtime_seconds" elapsed seconds, round(..., 3)
      ================  =====================================================

    The record is the unit of an experiment log: everything needed to reproduce
    the number sits next to the number. Do not print, do not write files, do not
    read or write module-level MUTABLE state — the constants ``DATA_PATH`` and
    ``BASE_CONFIG`` are fine to read (step 2 requires ``DATA_PATH``).
    ``runtime_seconds`` is the only field that may differ between two runs of
    the same (config, seed).
    """
    raise NotImplementedError
# ============================ END TODO (Task 1) ==============================


# ============ TODO (Task 2): one-factor-at-a-time ablation table =============
def run_ablation(base_config: dict, factors: dict, seeds=SEEDS, data=None) -> pd.DataFrame:
    """Run every one-factor variant of ``base_config`` over every seed.

    ``factors`` maps a config key to the list of alternative values to try, for
    example ``{"features": ["with_region", "core"], "scale": [False]}``.

    Variants, in this order:
      1. ``base_config`` itself, named ``"base"``.
      2. For each key of ``factors`` in dict order, for each value in its list
         in order: a copy of ``base_config`` with THAT ONE key replaced. The
         variant name is ``f"{key}={value}"``.

    Every variant differs from the base in exactly one key. Two changes in one
    row make the row uninterpretable, and the hidden tests check this.

    Run each variant with each seed of ``seeds``, in order, through
    ``run_config``. Return a tidy DataFrame with one row per (variant, seed):

      columns, in this order:
        "variant", "seed",
        one column PER KEY of base_config (column names = the keys sorted
        alphabetically; column VALUES = that row's own variant config, so the
        table is readable on its own — the variant column differs from base),
        "accuracy", "precision", "recall", "f1"
      rows: variant-major, then seed, in the order the runs were made
      index: a fresh RangeIndex, 0..n-1

    Load the data once and pass it to ``run_config`` through ``data``, so the
    CSV is not re-read for every run.
    """
    raise NotImplementedError
# ============================ END TODO (Task 2) ==============================


# =================== TODO (Task 3): per-slice error analysis =================
def error_analysis(y_true, y_pred, groups, min_count: int = 30) -> dict:
    """Where does the model fail, not just how much.

    ``groups`` has the same length as ``y_true``: one slice label per sample.
    All three arrive as numpy arrays (call np.asarray if you construct your
    own inputs some other way).

    Returns:

      ==============  =======================================================
      "overall"       compute_metrics over ALL samples
      "slices"        {label: {"n": int, "accuracy": float, "precision":
                      float, "recall": float, "f1": float}}, one entry per
                      label present in ``groups``, inserted in ascending label
                      order. Each slice's metrics come from that slice's rows
                      ONLY, so filter y_true and y_pred with the same mask.
      "worst_slice"   the label with the LOWEST accuracy among the slices with
                      ``n >= min_count``. Ties broken by ascending label.
                      None if no slice reaches ``min_count``.
      "worst_gap"     overall accuracy minus the worst slice's accuracy,
                      round(..., 4). 0.0 when "worst_slice" is None.
      ==============  =======================================================

    A slice with a handful of samples can land on any accuracy at all, so
    ``min_count`` keeps it out of the worst-slice answer. Such slices still
    appear in "slices" with their ``n``, which is the column to read first.
    """
    raise NotImplementedError
# ============================ END TODO (Task 3) ==============================


# ============= TODO (Task 4): aggregate repeated seeds and rank ==============
def summarize(records: list, metric: str = "f1") -> pd.DataFrame:
    """Turn a pile of records into one row per configuration.

    Group ``records`` (as returned by ``run_config``) by ``"config_id"``. The
    value being aggregated is ``record["metrics"][metric]``.

    Per group:
      "config_id"      the id
      "n_seeds"        how many records the group has, int
      "mean"           mean of the metric, round(..., 4)
      "std"            SAMPLE standard deviation, ddof=1, round(..., 4).
                       One record cannot estimate a spread: use float("nan"),
                       not 0.0.
      "min", "max"     round(..., 4)

    Sort by "mean" descending, ties by "config_id" ascending. Then add:
      "rank"           1 for the first row, 2 for the next, and so on
      "overlaps_best"  True when this row's mean +- std interval touches the
                       rank-1 row's interval, that is
                           mean + s >= best_mean - best_s
                       where s is the row's std, using 0.0 in place of nan, and
                       using the rounded "mean" and "std" of this table. The
                       rank-1 row is always True.

    Columns in this order: "config_id", "n_seeds", "mean", "std", "min",
    "max", "rank", "overlaps_best". Index: a fresh RangeIndex.

    A row with ``overlaps_best`` True is not distinguishable from the best row
    at this number of seeds. Reporting such a row as an improvement is the
    mistake this table exists to prevent.
    """
    raise NotImplementedError
# ============================ END TODO (Task 4) ==============================


def main() -> dict:
    """DO NOT MODIFY. Writes results.json with the course-wide schema."""
    set_seed()
    t0 = time.perf_counter()
    X, y, groups = load_data(DATA_PATH)
    data = (X, y, groups)

    base_record = run_config(BASE_CONFIG, SEED, data=data)
    table = run_ablation(BASE_CONFIG, FACTORS, SEEDS, data=data)
    ablation = table.groupby("variant", sort=False)["f1"].agg(["mean", "std"]).round(4)
    best_variant = str(ablation["mean"].idxmax())

    y_val, y_pred, groups_val = base_predictions(SEED)
    errors = error_analysis(y_val, y_pred, groups_val)
    if errors["worst_slice"] is None:
        raise ValueError(
            "error_analysis returned worst_slice=None — with the shipped data at "
            "least one slice has n >= min_count, so check your min_count filter"
        )
    worst = errors["slices"][errors["worst_slice"]]
    smallest_name = min(errors["slices"], key=lambda k: (errors["slices"][k]["n"], k))
    smallest = errors["slices"][smallest_name]

    compare = [run_config(cfg, seed, data=data)
               for cfg in (BASE_CONFIG, CHALLENGER_CONFIG) for seed in SEEDS]
    ranking = summarize(compare)

    results = {
        "lab": "lab08",
        "student_id": STUDENT_ID,
        "name": STUDENT_NAME,
        "seed": SEED,
        "metrics": {
            "n_samples": int(len(X)),
            "positive_rate": float(round(y.mean(), 4)),
            "n_train": int(base_record["n_train"]),
            "n_val": int(base_record["n_val"]),
            "base_accuracy": float(base_record["metrics"]["accuracy"]),
            "base_f1": float(base_record["metrics"]["f1"]),
            "ablation_variants": int(table["variant"].nunique()),
            "ablation_rows": int(len(table)),
            "ablation_base_f1_mean": float(ablation.loc["base", "mean"]),
            "ablation_base_f1_std": float(ablation.loc["base", "std"]),
            "ablation_best_variant": best_variant,
            "ablation_best_f1_mean": float(ablation.loc[best_variant, "mean"]),
            "overall_accuracy": float(errors["overall"]["accuracy"]),
            "worst_slice": str(errors["worst_slice"]),
            "worst_slice_n": int(worst["n"]),
            "worst_slice_accuracy": float(worst["accuracy"]),
            "worst_gap": float(errors["worst_gap"]),
            "smallest_slice": str(smallest_name),
            "smallest_slice_n": int(smallest["n"]),
            "smallest_slice_accuracy": float(smallest["accuracy"]),
            "ranking_top": str(ranking.loc[0, "config_id"]),
            "ranking_top_mean": float(ranking.loc[0, "mean"]),
            "ranking_top_std": float(ranking.loc[0, "std"]),
            "ranking_second_mean": float(ranking.loc[1, "mean"]),
            "ranking_second_std": float(ranking.loc[1, "std"]),
            "ranking_second_overlaps_best": bool(ranking.loc[1, "overlaps_best"]),
        },
        "runtime_seconds": round(time.perf_counter() - t0, 2),
    }
    out = Path(__file__).resolve().parent.parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
