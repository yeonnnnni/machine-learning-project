"""Lab 4-2 public tests. DO NOT MODIFY.

Run: python -m pytest tests/ -q
Grading also runs hidden tests (different data, edge cases, determinism).
"""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lab04_2  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "motor_health.csv"


@pytest.fixture(scope="module")
def data():
    lab04_2.set_seed()
    X, y = lab04_2.load_data(DATA)
    return lab04_2.prepare_data(X, y)


def test_build_model_structure():
    lab04_2.set_seed()
    model = lab04_2.build_model(7, hidden_size=16)
    assert isinstance(model, nn.Sequential)
    kinds = [type(m) for m in model]
    assert kinds == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
    assert (model[0].in_features, model[0].out_features) == (7, 16)
    assert (model[2].in_features, model[2].out_features) == (16, 16)
    assert (model[4].in_features, model[4].out_features) == (16, 1)
    out = model(torch.zeros(5, 7))
    assert out.shape == (5, 1)                        # one logit per row


def test_train_one_epoch_updates_parameters(data):
    lab04_2.set_seed()
    model = lab04_2.build_model(data["input_dim"], 8)
    before = [p.detach().clone() for p in model.parameters()]
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = lab04_2.train_one_epoch(model, data["train_loader"], loss_fn, opt)
    assert isinstance(loss, float) and 0.0 < loss < 5.0
    assert model.training is True                     # model.train() was called
    after = list(model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    losses = [lab04_2.train_one_epoch(model, data["train_loader"], loss_fn, opt)
              for _ in range(3)]
    assert losses[-1] < loss                          # training makes progress


def test_evaluate_is_read_only(data):
    lab04_2.set_seed()
    model = lab04_2.build_model(data["input_dim"], 8)
    model.train()
    before = [p.detach().clone() for p in model.parameters()]
    seen_requires_grad = []
    handle = model.register_forward_hook(
        lambda mod, inp, out: seen_requires_grad.append(bool(out.requires_grad)))
    m = lab04_2.evaluate(model, data["val_loader"], nn.BCEWithLogitsLoss())
    handle.remove()
    assert set(m) == {"loss", "accuracy"}
    assert isinstance(m["loss"], float) and isinstance(m["accuracy"], float)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert model.training is False                    # model.eval() was called
    assert seen_requires_grad and not any(seen_requires_grad)   # torch.no_grad()
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_train_model_history_and_best_weights(data):
    lab04_2.set_seed()
    model = lab04_2.build_model(data["input_dim"], 8)
    h = lab04_2.train_model(model, data["train_loader"], data["val_loader"],
                          lr=0.1, max_epochs=12, patience=4)
    assert set(h) >= {"train_loss", "val_loss", "val_accuracy",
                      "best_epoch", "best_val_loss", "epochs_run"}
    n = h["epochs_run"]
    assert 1 <= n <= 12
    assert len(h["train_loss"]) == len(h["val_loss"]) == len(h["val_accuracy"]) == n
    assert 1 <= h["best_epoch"] <= n
    assert h["best_val_loss"] == min(h["val_loss"])
    assert h["val_loss"][h["best_epoch"] - 1] == h["best_val_loss"]
    again = lab04_2.evaluate(model, data["val_loader"], nn.BCEWithLogitsLoss())
    assert again["loss"] == pytest.approx(h["best_val_loss"], abs=1e-6)


def test_train_model_stops_early_and_keeps_best_weights(data):
    lab04_2.set_seed()
    model = lab04_2.build_model(data["input_dim"], 8)
    h = lab04_2.train_model(model, data["train_loader"], data["val_loader"],
                          lr=3.0, max_epochs=60, patience=2)
    assert h["epochs_run"] < 60                       # a bad lr must stop early
    assert h["epochs_run"] == h["best_epoch"] + 2     # exactly `patience` bad epochs
    assert h["best_epoch"] < h["epochs_run"]          # so best weights != last weights
    # the returned model must carry the weights of the BEST epoch
    again = lab04_2.evaluate(model, data["val_loader"], nn.BCEWithLogitsLoss())
    assert again["loss"] == pytest.approx(h["best_val_loss"], abs=1e-6)


def test_run_experiment_selects_on_validation(data):
    r = lab04_2.run_experiment(data)
    assert set(r) == {"grid", "best_config", "val", "test"}
    assert len(r["grid"]) == len(lab04_2.HIDDEN_SIZES) * len(lab04_2.LEARNING_RATES)
    assert [(g["hidden_size"], g["lr"]) for g in r["grid"]] == [
        (h, lr) for h in lab04_2.HIDDEN_SIZES for lr in lab04_2.LEARNING_RATES]
    best = min(r["grid"], key=lambda g: g["val_loss"])
    assert r["best_config"] == {"hidden_size": best["hidden_size"], "lr": best["lr"]}
    assert r["val"]["loss"] == pytest.approx(best["val_loss"], abs=1e-6)
    assert set(r["test"]) == {"loss", "accuracy"}
    # a trained network must beat the majority-class rate on the test set
    assert r["test"]["accuracy"] > max(data["positive_rate"], 1 - data["positive_rate"])
    # re-running the winning configuration must reproduce its grid record exactly,
    # and val_accuracy must be the accuracy AT the best epoch, not at the last one
    lab04_2.set_seed()
    again = lab04_2.build_model(data["input_dim"], best["hidden_size"])
    h = lab04_2.train_model(again, data["train_loader"], data["val_loader"], best["lr"])
    assert h["best_val_loss"] == pytest.approx(best["val_loss"], abs=1e-6)
    assert h["epochs_run"] == best["epochs_run"] and h["best_epoch"] == best["best_epoch"]
    assert h["val_accuracy"][h["best_epoch"] - 1] == pytest.approx(
        best["val_accuracy"], abs=1e-6)
