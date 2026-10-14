"""Lab 4-1 public tests. DO NOT MODIFY.

Run: python -m pytest tests/ -q
Grading also runs hidden tests (different data, edge cases, determinism, leakage).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lab04_1  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "press_qc.csv"


@pytest.fixture(scope="module")
def data():
    return lab04_1.load_data(DATA)


@pytest.fixture(scope="module")
def small(data):
    """A 40-row slice and matching fresh parameters — enough to check gradients."""
    X, y = data
    return X[:40], y[:40], lab04_1.init_params(X.shape[1], 5, seed=0)


def test_sigmoid_properties():
    z = np.array([-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0])
    s = lab04_1.sigmoid(z)
    assert s.shape == z.shape
    assert np.all((s > 0.0) & (s < 1.0))
    assert s[3] == pytest.approx(0.5)
    assert np.all(np.diff(s) > 0)                                  # strictly increasing
    assert np.allclose(lab04_1.sigmoid(-z), 1.0 - s)                 # sigma(-z) = 1 - sigma(z)
    # numerically stable: no overflow even far out in the tails
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        tails = lab04_1.sigmoid(np.array([-1000.0, 1000.0]))
    assert np.all(np.isfinite(tails))
    assert tails[0] == pytest.approx(0.0) and tails[1] == pytest.approx(1.0)


def test_forward_shapes_and_cache(data):
    X, _ = data
    params = lab04_1.init_params(X.shape[1], lab04_1.N_HIDDEN, seed=lab04_1.SEED)
    cache = lab04_1.forward(X, params)
    assert set(cache) == {"z1", "a1", "z2", "a2"}
    n, h = len(X), lab04_1.N_HIDDEN
    assert cache["z1"].shape == (n, h) and cache["a1"].shape == (n, h)
    assert cache["z2"].shape == (n, 1) and cache["a2"].shape == (n, 1)
    assert np.allclose(cache["a1"], lab04_1.sigmoid(cache["z1"]))
    assert np.allclose(cache["a2"], lab04_1.sigmoid(cache["z2"]))
    assert np.all((cache["a2"] > 0.0) & (cache["a2"] < 1.0))       # a2 is a probability


def test_bce_loss_values():
    y = np.array([[1.0], [1.0], [0.0], [0.0]])
    assert lab04_1.bce_loss(y, np.full((4, 1), 0.5)) == pytest.approx(np.log(2.0), abs=1e-9)
    confident = np.array([[0.9], [0.9], [0.1], [0.1]])
    assert lab04_1.bce_loss(y, confident) == pytest.approx(-np.log(0.9), abs=1e-9)
    # clipping keeps log(0) finite instead of returning inf
    hopeless = lab04_1.bce_loss(np.array([[1.0]]), np.array([[0.0]]))
    assert np.isfinite(hopeless) and hopeless > 20.0
    assert isinstance(lab04_1.bce_loss(y, confident), float)


def test_backward_shapes_and_gradient_check(small):
    X, y, params = small
    grads = lab04_1.backward(X, y, params, lab04_1.forward(X, params))
    assert set(grads) == set(params)
    for key in params:
        assert grads[key].shape == params[key].shape                # gradient matches parameter
    rel = lab04_1.gradient_check(X, y, params)
    assert set(rel) == set(params)
    for key, err in rel.items():
        assert err < 1e-6, f"analytic gradient for {key} disagrees with finite differences"


def test_train_history_and_convergence(data):
    X, y = data
    X_tr, X_val, y_tr, y_val = lab04_1.train_val_split(X, y)
    params, history = lab04_1.train(X_tr, y_tr, X_val, y_val, lr=1.0, epochs=40)
    assert set(history) == {"train_loss", "val_loss", "val_accuracy"}
    assert all(len(v) == 40 for v in history.values())
    assert all(isinstance(v, float) for v in history["train_loss"])
    assert history["train_loss"][-1] < history["train_loss"][0]     # it actually learns
    assert 0.0 <= min(history["val_accuracy"]) <= max(history["val_accuracy"]) <= 1.0
    assert set(params) == {"W1", "b1", "W2", "b2"}
    # same seed -> identical run
    _, again = lab04_1.train(X_tr, y_tr, X_val, y_val, lr=1.0, epochs=40)
    assert history == again


def test_run_experiment_compares_learning_rates(data):
    X, y = data
    r = lab04_1.run_experiment(X, y)
    assert set(r) == {str(lr) for lr in lab04_1.LEARNING_RATES}
    keys = {"final_train_loss", "max_train_loss", "final_val_loss",
            "final_val_accuracy", "best_val_accuracy"}
    for lr_key, rec in r.items():
        assert set(rec) == keys, f"{lr_key} record has the wrong keys"
        assert all(isinstance(v, float) for v in rec.values())
    majority = max(float(y.mean()), 1.0 - float(y.mean()))
    # the middle learning rate is the usable one: it must beat the majority baseline
    assert r["1.0"]["final_val_accuracy"] > majority + 0.05
    # too small: still stuck near the baseline after the whole budget
    assert r["0.01"]["final_train_loss"] > 0.5
    # too large: the loss blows up at some point during training
    assert r["100.0"]["max_train_loss"] > 5.0
