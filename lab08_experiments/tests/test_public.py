"""Lab 8 public tests. DO NOT MODIFY.

Run: python -m pytest tests/ -q
Grading also runs hidden tests (different data, edge cases, determinism).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lab08  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "delivery_late.csv"

RECORD_KEYS = {"config_id", "config", "seed", "metrics",
               "n_train", "n_val", "runtime_seconds"}
METRIC_KEYS = {"accuracy", "precision", "recall", "f1"}


@pytest.fixture(scope="module")
def data():
    return lab08.load_data(DATA)


def _fake_records(config_id: str, values, metric: str = "f1"):
    return [{"config_id": config_id, "config": {}, "seed": i,
             "metrics": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, metric: v},
             "n_train": 10, "n_val": 5, "runtime_seconds": 0.0}
            for i, v in enumerate(values)]


def test_run_config_returns_one_complete_record(data):
    rec = lab08.run_config(lab08.BASE_CONFIG, 42, data=data)
    assert set(rec) == RECORD_KEYS
    assert set(rec["metrics"]) == METRIC_KEYS
    assert rec["seed"] == 42
    assert rec["config_id"] == lab08.config_id(lab08.BASE_CONFIG)
    assert rec["n_train"] + rec["n_val"] == len(data[0])
    assert rec["n_val"] == int(round(0.25 * len(data[0])))
    # the metrics must come from the held-out rows, not from the training rows
    y_val, y_pred, _ = lab08.base_predictions(42)
    assert rec["metrics"] == lab08.compute_metrics(y_val, y_pred)
    # the record owns its config: mutating the caller's dict must not touch it
    caller = dict(lab08.BASE_CONFIG)
    rec2 = lab08.run_config(caller, 42, data=data)
    caller["model"] = "tree"
    assert rec2["config"]["model"] == "logreg"
    # a config with an unknown or missing key must fail loudly
    with pytest.raises(ValueError):
        lab08.run_config({**lab08.BASE_CONFIG, "learning_rate": 0.1}, 42, data=data)
    with pytest.raises(ValueError):
        lab08.run_config({"model": "logreg"}, 42, data=data)


def test_run_config_is_deterministic_and_uses_the_seed(data):
    a = lab08.run_config(lab08.BASE_CONFIG, 42, data=data)
    b = lab08.run_config(lab08.BASE_CONFIG, 42, data=data)
    a.pop("runtime_seconds"), b.pop("runtime_seconds")
    assert a == b                                   # same (config, seed) -> same record
    c = lab08.run_config(lab08.BASE_CONFIG, 43, data=data)
    assert c["seed"] == 43
    assert c["metrics"] != a["metrics"]             # the seed really moves the split


def test_ablation_changes_exactly_one_factor(data):
    factors = {"model": ["tree"], "scale": [False]}
    seeds = (42, 43)
    table = lab08.run_ablation(lab08.BASE_CONFIG, factors, seeds, data=data)
    keys = sorted(lab08.BASE_CONFIG)

    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == (["variant", "seed"] + keys
                                  + ["accuracy", "precision", "recall", "f1"])
    assert list(table.index) == list(range(len(table)))
    assert len(table) == (1 + 2) * len(seeds)
    assert list(table["variant"].unique()) == ["base", "model=tree", "scale=False"]

    for _, row in table.iterrows():
        changed = [k for k in keys if row[k] != lab08.BASE_CONFIG[k]]
        expected = 0 if row["variant"] == "base" else 1
        assert len(changed) == expected, f"{row['variant']} changed {changed}"


def test_error_analysis_uses_each_slice_only(data):
    groups = np.array(["A"] * 6 + ["B"] * 4 + ["C"] * 2)
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])

    out = lab08.error_analysis(y_true, y_pred, groups, min_count=4)
    assert set(out) == {"overall", "slices", "worst_slice", "worst_gap"}
    assert list(out["slices"]) == ["A", "B", "C"]           # ascending label order
    assert [out["slices"][k]["n"] for k in "ABC"] == [6, 4, 2]
    assert out["overall"]["accuracy"] == pytest.approx(5 / 12, abs=1e-4)
    assert out["slices"]["A"]["accuracy"] == pytest.approx(5 / 6, abs=1e-4)
    assert out["slices"]["A"]["precision"] == pytest.approx(1.0, abs=1e-4)
    assert out["slices"]["A"]["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["slices"]["B"]["accuracy"] == 0.0
    # C is worse than nothing but has only 2 samples, so B is the answer
    assert out["worst_slice"] == "B"
    assert out["worst_gap"] == pytest.approx(5 / 12, abs=1e-4)
    # no slice large enough -> no claim
    empty = lab08.error_analysis(y_true, y_pred, groups, min_count=99)
    assert empty["worst_slice"] is None and empty["worst_gap"] == 0.0


def test_summarize_reports_spread_and_ranks(data):
    records = (_fake_records("A", [0.70, 0.80, 0.90])
               + _fake_records("B", [0.60, 0.62, 0.64])
               + _fake_records("C", [0.75]))
    out = lab08.summarize(records)
    assert list(out.columns) == ["config_id", "n_seeds", "mean", "std",
                                 "min", "max", "rank", "overlaps_best"]
    assert list(out["config_id"]) == ["A", "C", "B"]        # sorted by mean, descending
    assert list(out["rank"]) == [1, 2, 3]
    assert list(out["n_seeds"]) == [3, 1, 3]
    assert out.loc[0, "mean"] == pytest.approx(0.80, abs=1e-4)
    assert out.loc[0, "std"] == pytest.approx(0.10, abs=1e-4)   # sample std, ddof=1
    assert np.isnan(out.loc[1, "std"])                          # one seed is not a spread
    assert list(out["overlaps_best"]) == [True, True, False]


def test_the_lab_has_something_to_report(data):
    """The shipped data must produce a real finding and a real non-finding."""
    table = lab08.run_ablation(lab08.BASE_CONFIG, lab08.FACTORS, lab08.SEEDS, data=data)
    f1 = table.groupby("variant", sort=False)["f1"].agg(["mean", "std"])
    base_mean, base_std = f1.loc["base", "mean"], f1.loc["base", "std"]
    # one factor clearly clears the seed-to-seed spread
    assert f1.loc["features=with_region", "mean"] - base_mean > 3 * base_std
    # and one factor sits inside it
    assert abs(f1.loc["scale=False", "mean"] - base_mean) < base_std

    y_val, y_pred, groups_val = lab08.base_predictions(42)
    out = lab08.error_analysis(y_val, y_pred, groups_val)
    assert out["worst_slice"] == "island"
    assert out["slices"]["island"]["n"] >= 30
    assert out["worst_gap"] > 0.2                       # far worse than the overall number
    assert out["slices"]["overseas"]["n"] < 30          # too small to conclude from
