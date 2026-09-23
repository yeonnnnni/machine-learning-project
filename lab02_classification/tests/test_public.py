"""Lab 2 public tests. DO NOT MODIFY.

Run: python -m pytest tests/ -q
Grading also runs hidden tests (different data, edge cases, determinism).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lab02  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "clinic_noshow.csv"
DATA_REG = Path(__file__).resolve().parent.parent / "data" / "clinic_wait.csv"


@pytest.fixture(scope="module")
def data():
    return lab02.load_data(DATA)


def test_split_shapes_and_disjoint(data):
    X, y = data
    X_tr, X_val, y_tr, y_val = lab02.split_data(X, y)
    assert len(X_tr) + len(X_val) == len(X)
    assert len(X_val) == int(round(0.2 * len(X)))
    assert set(X_tr.index).isdisjoint(set(X_val.index))          # no leakage
    assert (y_tr.index == X_tr.index).all() and (y_val.index == X_val.index).all()


def test_split_is_stratified_and_deterministic(data):
    X, y = data
    _, _, y_tr, y_val = lab02.split_data(X, y)
    assert abs(y_tr.mean() - y_val.mean()) < 0.02                # class ratios preserved
    a = lab02.split_data(X, y)[0]
    b = lab02.split_data(X, y)[0]
    assert (a.index == b.index).all()                            # same seed -> same split


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    m = lab02.compute_metrics(y_true, y_pred)
    assert set(m) == {"accuracy", "precision", "recall", "f1"}
    assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred), abs=1e-4)
    assert m["precision"] == pytest.approx(precision_score(y_true, y_pred), abs=1e-4)
    assert m["recall"] == pytest.approx(recall_score(y_true, y_pred), abs=1e-4)
    assert m["f1"] == pytest.approx(f1_score(y_true, y_pred), abs=1e-4)
    assert all(isinstance(v, float) for v in m.values())


def test_metrics_zero_division_convention():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])                              # never predicts positive
    m = lab02.compute_metrics(y_true, y_pred)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_pipeline_structure():
    p = lab02.build_model("logreg")
    assert isinstance(p, Pipeline)
    assert [name for name, _ in p.steps] == ["scaler", "clf"]
    with pytest.raises(ValueError):
        lab02.build_model("svm")


def test_experiment_beats_baseline(data):
    X, y = data
    r = lab02.run_experiment(X, y)
    assert set(r) == {"baseline", "logreg", "knn"}
    # majority-class baseline: accuracy == majority rate, f1 == 0
    assert r["baseline"]["f1"] == 0.0
    assert r["baseline"]["accuracy"] == pytest.approx(1 - y.mean(), abs=0.03)
    # a correct model must clearly beat the baseline on f1
    assert r["logreg"]["f1"] > 0.4
    assert r["logreg"]["accuracy"] > r["baseline"]["accuracy"]


def test_regression_metrics_match_sklearn_and_edge_cases():
    rng = np.random.default_rng(1)
    y_true = rng.normal(50, 10, 150)
    y_pred = y_true + rng.normal(0, 5, 150)
    m = lab02.compute_regression_metrics(y_true, y_pred)
    assert set(m) == {"mae", "rmse", "r2"}
    assert m["mae"] == pytest.approx(mean_absolute_error(y_true, y_pred), abs=1e-4)
    assert m["rmse"] == pytest.approx(mean_squared_error(y_true, y_pred) ** 0.5, abs=1e-4)
    assert m["r2"] == pytest.approx(r2_score(y_true, y_pred), abs=1e-4)
    assert all(isinstance(v, float) for v in m.values())
    # perfect prediction
    perfect = lab02.compute_regression_metrics(y_true, y_true)
    assert perfect["mae"] == 0.0 and perfect["rmse"] == 0.0 and perfect["r2"] == 1.0
    # constant target: the r2 denominator is zero -> 0.0 by convention
    const = lab02.compute_regression_metrics(np.full(10, 3.0), np.full(10, 4.0))
    assert const["r2"] == 0.0


def test_regression_pipeline_structure():
    from sklearn.dummy import DummyRegressor
    base = lab02.build_regression_model("baseline")
    assert isinstance(base, DummyRegressor)          # nothing to scale
    p = lab02.build_regression_model("linreg")
    assert isinstance(p, Pipeline)
    assert [name for name, _ in p.steps] == ["scaler", "reg"]
    with pytest.raises(ValueError):
        lab02.build_regression_model("forest")


def test_regression_experiment_beats_mean_baseline():
    X, y = lab02.load_regression_data(DATA_REG)
    r = lab02.run_regression_experiment(X, y)
    assert set(r) == {"baseline", "linreg", "knn"}
    # predicting the training mean explains (almost) none of the variance
    assert abs(r["baseline"]["r2"]) < 0.05
    # a correct linear model must clearly beat it on this data
    assert r["linreg"]["r2"] > 0.7
    assert r["linreg"]["mae"] < r["baseline"]["mae"]
