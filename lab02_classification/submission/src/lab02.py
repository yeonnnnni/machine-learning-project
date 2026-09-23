"""Lab 2 — Classification and Regression with scikit-learn: correct, leak-free pipelines.

Machine Learning Project (53744-01), Fall 2026.

Fill in every TODO block. Do NOT rename functions or change their
signatures/return types — automated (public + hidden) tests call them directly.
Run:      python src/lab02.py
Self-check: python -m pytest tests/ -q
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---- Fill in your information (used in results.json) ----
STUDENT_ID = "20231879"   # TODO: your student id
STUDENT_NAME = "LeeJeongyeon"     # TODO: your name in roman letters

SEED = 42  # fixed for the whole course — DO NOT CHANGE
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "clinic_noshow.csv"
TARGET = "no_show"
DATA_PATH_REG = Path(__file__).resolve().parent.parent / "data" / "clinic_wait.csv"
TARGET_REG = "wait_minutes"


def set_seed(seed: int = SEED) -> None:
    """DO NOT MODIFY."""
    np.random.seed(seed)


def load_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """DO NOT MODIFY. Returns (X, y)."""
    df = pd.read_csv(path)
    return df.drop(columns=[TARGET]), df[TARGET]


def load_regression_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """DO NOT MODIFY. Returns (X, y) for the regression half of the lab."""
    df = pd.read_csv(path)
    return df.drop(columns=[TARGET_REG]), df[TARGET_REG]


# ================= TODO (Task 1): stratified, leak-free split ================
def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
               seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train/validation sets.

    Requirements (checked by tests):
      - use sklearn.model_selection.train_test_split
      - STRATIFY on y (class ratios preserved in both splits)
      - random_state=seed, shuffle=True
      - return (X_train, X_val, y_train, y_val)

    Never fit anything (scaler, model, imputer) before this split — that is
    data leakage, and the hidden tests are built to catch it.
    """
    # raise NotImplementedError
    # input: test_size, seed, shuffle=True

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = test_size, random_state=seed, shuffle=True, stratify=y)

    return (X_train, X_val, y_train, y_val)

# ============================ END TODO (Task 1) ==============================


# ================== TODO (Task 2): metrics from scratch ======================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute binary-classification metrics WITH NUMPY ONLY.

    Do NOT call sklearn.metrics here — the point is to show you know what the
    numbers mean. (Tests compare your values against sklearn's.)

    The inputs may arrive as numpy arrays OR pandas Series (run_experiment
    passes the validation labels straight from the split). Convert both with
    np.asarray() first so positional indexing behaves.

    Positive class is 1. With TP/FP/TN/FN:
      accuracy  = (TP+TN) / N
      precision = TP / (TP+FP)   (define as 0.0 if TP+FP == 0)
      recall    = TP / (TP+FN)   (define as 0.0 if TP+FN == 0)
      f1        = 2*P*R / (P+R)  (define as 0.0 if P+R == 0)

    Returns:
        {"accuracy": float, "precision": float, "recall": float, "f1": float}
        — plain Python floats rounded to 4 decimals.
    """
    # raise NotImplementedError

    # input: y_ture, y_pred
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # TP, TN, FP, FN
    TP_arr = (y_true == 1) & (y_pred == 1)
    TP = np.sum(TP_arr)
    TN_arr = (y_true == 0) & (y_pred == 0)
    TN = np.sum(TN_arr)
    FP_arr = (y_true == 0) & (y_pred == 1)
    FP = np.sum(FP_arr)
    FN_arr = (y_true == 1) & (y_pred == 0)
    FN = np.sum(FN_arr)
    N = len(y_true)

    # accuracy  = (TP+TN) / N
    accuracy  = (TP+TN) / N
    
    # precision = TP / (TP+FP)   (define as 0.0 if TP+FP == 0)
    if (TP+FP == 0):
        precision = 0.0
    else:
        precision = TP / (TP+FP)
    
    # recall    = TP / (TP+FN)   (define as 0.0 if TP+FN == 0)
    if (TP+FN == 0):
        recall = 0.0
    else:
        recall = TP / (TP+FN)
    
    
    # f1        = 2*P*R / (P+R)  (define as 0.0 if P+R == 0)
    if (precision + recall == 0):
        f1 = 0.0
    else:
        f1 = 2*precision*recall / (precision + recall)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4)
    }

# ============================ END TODO (Task 2) ==============================


# ===================== TODO (Task 3): model pipelines ========================
def build_baseline() -> DummyClassifier:
    """Return an UNFITTED DummyClassifier with strategy="most_frequent".

    This is the sanity baseline every later model must beat.
    """
    # raise NotImplementedError

    model = DummyClassifier(strategy="most_frequent")

    return model


def build_model(model_name: str = "logreg") -> Pipeline:
    """Return an UNFITTED sklearn Pipeline: StandardScaler -> classifier.

    model_name == "logreg": LogisticRegression(random_state=SEED, max_iter=1000)
    model_name == "knn":    KNeighborsClassifier(n_neighbors=15)
    anything else:          raise ValueError

    The pipeline steps MUST be named exactly "scaler" and "clf".
    Scaling inside the Pipeline (fit on train only) is what keeps the
    validation data untouched during preprocessing.
    """
    # raise NotImplementedError

    # model_name == "logreg"
    if (model_name == "logreg"): 
        classifier = LogisticRegression(random_state=SEED, max_iter=1000)

    # model_name == "knn"
    elif (model_name == "knn"):
        classifier = KNeighborsClassifier(n_neighbors=15)


    # anything else:          raise ValueError
    else:
        raise ValueError
        
    model = Pipeline([ ("scaler", StandardScaler()), ("clf", classifier)])

    # Return an UNFITTED sklearn Pipeline: StandardScaler -> classifier.
    return model



# ============================ END TODO (Task 3) ==============================


# ================ TODO (Task 4): controlled model comparison =================
def run_experiment(X: pd.DataFrame, y: pd.Series) -> dict:
    """Compare baseline / logreg / knn under IDENTICAL conditions.

    Controlled experiment = one factor varies (the model); everything else is
    fixed: same split (split_data with default seed), same metrics
    (compute_metrics), same preprocessing (inside each pipeline).

    Steps:
      1. X_train, X_val, y_train, y_val = split_data(X, y)
      2. For each of {"baseline": build_baseline(),
                      "logreg": build_model("logreg"),
                      "knn": build_model("knn")}:
         fit on (X_train, y_train), predict on X_val,
         store compute_metrics(y_val, y_pred).

    Returns:
        {"baseline": {...}, "logreg": {...}, "knn": {...}}
        (each value is a compute_metrics dict).
    """
    # raise NotImplementedError

    # 1. X_train, X_val, y_train, y_val = split_data(X, y)
    X_train, X_val, y_train, y_val = split_data(X, y)

    # 2. For each of {"baseline": build_baseline(), "logreg": build_model("logreg"), "knn": build_model("knn")}:
    baseline = build_baseline()
    logreg = build_model("logreg")
    knn = build_model("knn")


    models = {
        "baseline": baseline,
        "logreg": logreg,
        "knn": knn
    }

    results = {}

    for name, model in models.items():
        # fit on (X_train, y_train), predict on X_val,
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # store compute_metrics(y_val, y_pred).
        metrics = compute_metrics(y_val, y_pred)
        
        results[name] = metrics

    return results


# ============================ END TODO (Task 4) ==============================


# ================ TODO (Task 5): regression metrics from scratch =============
def compute_regression_metrics(y_true, y_pred) -> dict:
    """Compute regression metrics WITH NUMPY ONLY.

    Do NOT call sklearn.metrics here — same rule as compute_metrics. The
    inputs may arrive as numpy arrays or pandas Series; convert both with
    np.asarray() first.

    With residuals e_i = y_true_i - y_pred_i and ybar = mean(y_true):
      mae  = mean(|e_i|)
      rmse = sqrt(mean(e_i ** 2))
      r2   = 1 - sum(e_i ** 2) / sum((y_true_i - ybar) ** 2)
             (define r2 as 0.0 if sum((y_true_i - ybar) ** 2) == 0;
              r2 CAN be negative — worse than predicting the mean)

    Returns:
        {"mae": float, "rmse": float, "r2": float}
        — plain Python floats rounded to 4 decimals.
    """
    # raise NotImplementedError

    # input: y_true, y_pred
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    error = y_true - y_pred

    # mae  = mean(|e_i|)
    mae = np.mean(np.abs(error))

    # rmse = sqrt(mean(e_i ** 2))
    rmse = np.sqrt(np.mean(error ** 2))

    # r2   = 1 - sum(e_i ** 2) / sum((y_true_i - ybar) ** 2)
    #         (define r2 as 0.0 if sum((y_true_i - ybar) ** 2) == 0;
    #         r2 CAN be negative — worse than predicting the mean)
    ybar = np.mean(y_true)
    if (np.sum((y_true - ybar) ** 2) == 0):
        r2 = 0.0
    else:
        r2   = 1 - np.sum(error ** 2) / np.sum((y_true - ybar) ** 2)

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4)
    }
    
# ============================ END TODO (Task 5) ==============================


# ============= TODO (Task 6): controlled regression comparison ===============
def build_regression_model(model_name: str = "baseline"):
    """Return an UNFITTED regressor.

    model_name == "baseline": DummyRegressor(strategy="mean")
                              — predicts the training mean for every row;
                              its r2 on new data is close to 0 by definition.
    model_name == "linreg":   Pipeline: StandardScaler -> LinearRegression()
    model_name == "knn":      Pipeline: StandardScaler ->
                              KNeighborsRegressor(n_neighbors=15)
    anything else:            raise ValueError

    Pipeline steps MUST be named exactly "scaler" and "reg". (The baseline is
    a bare DummyRegressor, not a Pipeline — it has nothing to scale.)
    """
    # raise NotImplementedError

    # model_name == "baseline": DummyRegressor(strategy="mean")
    #                           — predicts the training mean for every row;
    #                           its r2 on new data is close to 0 by definition.
    if model_name == "baseline":
        model = DummyRegressor(strategy="mean")

    # model_name == "linreg":   Pipeline: StandardScaler -> LinearRegression()
    elif model_name == "linreg":
        model = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])

    # model_name == "knn":      Pipeline: StandardScaler -> KNeighborsRegressor(n_neighbors=15)
    elif model_name == "knn":
        model = Pipeline([("scaler", StandardScaler()), ("reg", KNeighborsRegressor(n_neighbors=15))])

    # anything else:            raise ValueError
    else:
        raise ValueError

    return model

def run_regression_experiment(X: pd.DataFrame, y: pd.Series,
                              test_size: float = 0.2, seed: int = SEED) -> dict:
    """Compare baseline / linreg / knn on the waiting-time data.

    The target is continuous, so DO NOT reuse split_data here — stratifying on
    a continuous y is meaningless (and raises). Split with:

        train_test_split(X, y, test_size=test_size, random_state=seed,
                         shuffle=True)

    (no stratify). Then, exactly as in run_experiment: fit each model on the
    train split, predict on the validation split, and store
    compute_regression_metrics(y_val, y_pred).

    Returns:
        {"baseline": {...}, "linreg": {...}, "knn": {...}}
        (each value is a compute_regression_metrics dict).
    """
    # raise NotImplementedError

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True)

    models = {
        "baseline": build_regression_model("baseline"),
        "linreg": build_regression_model("linreg"),
        "knn": build_regression_model("knn"),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        metrics = compute_regression_metrics(y_val, y_pred)

        results[name] = metrics

    return results




# ============================ END TODO (Task 6) ==============================


def main() -> dict:
    """DO NOT MODIFY. Writes results.json with the course-wide schema."""
    set_seed()
    t0 = time.time()
    X, y = load_data(DATA_PATH)
    experiment = run_experiment(X, y)
    X_reg, y_reg = load_regression_data(DATA_PATH_REG)
    regression = run_regression_experiment(X_reg, y_reg)
    results = {
        "lab": "lab02",
        "student_id": STUDENT_ID,
        "name": STUDENT_NAME,
        "seed": SEED,
        "metrics": {
            "n_samples": int(len(X)),
            "positive_rate": float(round(y.mean(), 4)),
            **{f"{m}_{k}": v for m, d in experiment.items() for k, v in d.items()},
            "n_samples_reg": int(len(X_reg)),
            "wait_mean": float(round(y_reg.mean(), 4)),
            **{f"reg_{m}_{k}": v for m, d in regression.items() for k, v in d.items()},
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    out = Path(__file__).resolve().parent.parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
