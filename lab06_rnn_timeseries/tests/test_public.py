"""Lab 6 public tests. DO NOT MODIFY.

Run: python -m pytest tests/ -q
Grading also runs hidden tests (different series, edge cases, leakage, determinism).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lab06  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "cafe_daily.csv"


@pytest.fixture(scope="module")
def series():
    return lab06.load_data(DATA)


@pytest.fixture(scope="module")
def experiment(series):
    return lab06.run_experiment(series)


def test_make_windows_worked_example():
    """The eight days of the concept note, §4: window 3, horizon 1, five pairs."""
    s = np.array([180.0, 190.0, 186.0, 182.0, 170.0, 140.0, 138.0, 188.0])
    X, y = lab06.make_windows(s, window=3, horizon=1)
    assert X.shape == (5, 3) and y.shape == (5,)
    assert np.allclose(X[0], [180.0, 190.0, 186.0]) and y[0] == 182.0
    assert np.allclose(X[4], [170.0, 140.0, 138.0]) and y[4] == 188.0
    assert X.dtype == np.float64 and y.dtype == np.float64


def test_make_windows_has_no_lookahead():
    """X[i] must end exactly `horizon` steps before y[i], for any horizon."""
    s = np.arange(20, dtype=float)                       # s[t] == t, so time is readable
    for horizon in (1, 2, 3):
        X, y = lab06.make_windows(s, window=5, horizon=horizon)
        assert len(X) == len(s) - 5 - horizon + 1
        assert np.allclose(X[:, -1] + horizon, y)        # target sits `horizon` after
        assert not any(y[i] in set(X[i]) for i in range(len(y)))   # never inside


def test_temporal_split_is_chronological():
    n = 200
    tr, va, te = lab06.temporal_split(n, val_frac=0.15, test_frac=0.15, gap=7)
    assert len(te) == round(0.15 * n) and len(va) == round(0.15 * n)
    assert len(tr) == n - len(va) - len(te) - 2 * 7
    assert tr.max() < va.min() < va.max() < te.min()     # strictly in time order
    assert te.max() == n - 1
    assert np.array_equal(tr, np.sort(tr))               # never shuffled
    assert va.min() - tr.max() - 1 == 7                  # embargo respected
    assert te.min() - va.max() - 1 == 7
    assert set(tr).isdisjoint(va) and set(va).isdisjoint(te)


def test_rnn_structure_and_last_hidden_state():
    torch.manual_seed(0)
    model = lab06.build_rnn(input_size=1, hidden_size=8, num_layers=1)
    assert isinstance(model.rnn, torch.nn.GRU) and model.rnn.batch_first
    assert model.rnn.hidden_size == 8
    assert isinstance(model.head, torch.nn.Linear)
    assert (model.head.in_features, model.head.out_features) == (8, 1)
    x = torch.randn(5, 12, 1)                            # batch 5 != seq_len 12
    out = model(x)
    assert out.shape == (5, 1)
    seq, _ = model.rnn(x)
    assert torch.allclose(out, model.head(seq[:, -1, :]), atol=1e-6)


def test_experiment_split_sizes_and_train_only_scaling(series, experiment):
    n = len(series)
    n_days_test = int(round(lab06.TEST_FRAC * n))
    n_days_val = int(round(lab06.VAL_FRAC * n))
    n_days_train = n - n_days_val - n_days_test - 2 * lab06.GAP
    per_segment = lambda d: d - lab06.WINDOW - lab06.HORIZON + 1   # noqa: E731
    assert experiment["n_train"] == per_segment(n_days_train)
    assert experiment["n_val"] == per_segment(n_days_val)
    assert experiment["n_test"] == per_segment(n_days_test)
    # the scaler saw the training segment only
    assert experiment["scale_mean"] == pytest.approx(series[:n_days_train].mean(), abs=1e-3)
    assert experiment["scale_mean"] != pytest.approx(series.mean(), abs=1e-3)
    # the metrics were measured on the LAST segment, in cups, with the window's
    # last value as the persistence forecast
    test_seg = series[n - n_days_test:]
    X_test, y_test = lab06.make_windows(test_seg, lab06.WINDOW, lab06.HORIZON)
    assert experiment["persistence"]["mae"] == pytest.approx(
        np.mean(np.abs(y_test - X_test[:, -1])), abs=1e-3)
    assert experiment["naive_scale"] == pytest.approx(
        np.mean(np.abs(np.diff(series[:n_days_train]))), abs=1e-3)


def test_rnn_beats_the_persistence_baseline(series, experiment):
    for name in ("persistence", "rnn"):
        assert set(experiment[name]) == {"mae", "rmse", "mase"}
    assert experiment["persistence"]["mase"] == pytest.approx(1.0, abs=0.2)
    assert experiment["rnn"]["mae"] < experiment["persistence"]["mae"]
    assert experiment["rnn"]["mase"] < 1.0                # better than repeating yesterday
    assert lab06.run_experiment(series) == experiment     # same seed -> same numbers
