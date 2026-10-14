"""Lab 4-2 — Training a neural network in PyTorch, and tuning it honestly.

Machine Learning Project (53744-01), Fall 2026.

Fill in every TODO block. Do NOT rename functions or change their
signatures/return types — automated (public + hidden) tests call them directly.
Run:      python src/lab04_2.py
Self-check: python -m pytest tests/ -q
"""
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---- Fill in your information (used in results.json) ----
STUDENT_ID = "00000000"   # TODO: your student id
STUDENT_NAME = "None"     # TODO: your name in roman letters

SEED = 42  # fixed for the whole course — DO NOT CHANGE
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "motor_health.csv"
TARGET = "needs_service"

# ---- Fixed experiment settings. The tests read these; do not change them. ----
BATCH_SIZE = 64
MAX_EPOCHS = 80
PATIENCE = 10
LEARNING_RATES = (0.01, 0.1, 1.0)    # Task 4 grid, inner loop
HIDDEN_SIZES = (8, 32)               # Task 4 grid, outer loop


def set_seed(seed: int = SEED) -> None:
    """DO NOT MODIFY. Seeds python, numpy and torch in one call."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """DO NOT MODIFY. Returns (X, y) as float32 arrays, y of shape (n,)."""
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET]).to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    return X, y


def prepare_data(X: np.ndarray, y: np.ndarray, batch_size: int = BATCH_SIZE,
                 seed: int = SEED) -> dict:
    """DO NOT MODIFY. Three-way split, standardization, DataLoaders.

    Steps, in this order:
      1. stratified 60 / 20 / 20 split into train / validation / test
      2. standardize every column with the TRAINING mean and standard
         deviation only (validation and test are transformed, never fitted)
      3. wrap each part in a DataLoader; only the training loader shuffles

    Returns a dict with "train_loader", "val_loader", "test_loader",
    "input_dim", "n_train", "n_val", "n_test", "positive_rate".
    Every y batch has shape (batch, 1) to match the model output.
    """
    X_fit, X_tmp, y_fit, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=seed, shuffle=True, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, shuffle=True, stratify=y_tmp)

    mean = X_fit.mean(axis=0)
    std = X_fit.std(axis=0)
    std[std == 0] = 1.0

    def loader(Xp, yp, shuffle):
        ds = TensorDataset(
            torch.tensor((Xp - mean) / std, dtype=torch.float32),
            torch.tensor(yp, dtype=torch.float32).unsqueeze(1))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return {
        "train_loader": loader(X_fit, y_fit, True),
        "val_loader": loader(X_val, y_val, False),
        "test_loader": loader(X_test, y_test, False),
        "input_dim": int(X.shape[1]),
        "n_train": int(len(X_fit)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "positive_rate": float(round(float(y.mean()), 4)),
    }


# ==================== TODO (Task 1): the model ===============================
def build_model(input_dim: int, hidden_size: int = 32) -> nn.Sequential:
    """Return an UNTRAINED nn.Sequential with EXACTLY these five modules:

        1. nn.Linear(input_dim, hidden_size)
        2. nn.ReLU()
        3. nn.Linear(hidden_size, hidden_size)
        4. nn.ReLU()
        5. nn.Linear(hidden_size, 1)

    The last layer outputs ONE raw score per row (a logit), not a
    probability: shape (batch, 1). The loss function applies the sigmoid
    itself, so do not add nn.Sigmoid() here.

    Do not call set_seed() inside this function. Seeding is the caller's job
    (Task 4), which is what makes the three learning rates tried at a given
    hidden size start from identical weights.
    """
    raise NotImplementedError
# ============================ END TODO (Task 1) ==============================


# ============ TODO (Task 2): one training epoch, and evaluation ==============
def train_one_epoch(model: nn.Module, loader: DataLoader,
                    loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """Run ONE pass over `loader` in training mode and return the mean loss.

    Put the model in training mode first (model.train()). Then, for every
    (x_batch, y_batch) in the loader, in exactly this order:

      1. optimizer.zero_grad()   clear the gradients left by the last step
      2. logits = model(x_batch)
      3. loss = loss_fn(logits, y_batch)
      4. loss.backward()         accumulate d(loss)/d(parameter)
      5. optimizer.step()        apply the update

    Gradients ACCUMULATE in PyTorch. Skipping step 1, or moving it after
    step 4, silently trains on a sum of gradients from several batches.

    Returns:
        The average training loss per sample: sum over batches of
        `loss.item() * len(x_batch)`, divided by `len(loader.dataset)`,
        rounded to 6 decimals.
    """
    raise NotImplementedError


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> dict:
    """Measure loss and accuracy over `loader` WITHOUT training on it.

    Put the model in evaluation mode (model.eval()) and wrap the whole loop
    in `with torch.no_grad():`. Never call backward() or step() here.

    A row is predicted positive when its logit is greater than 0, which is
    the same as a sigmoid probability greater than 0.5.

    Returns:
        {"loss": float, "accuracy": float} — both averaged over SAMPLES,
        not over batches (the last batch is usually smaller), and rounded
        to 6 decimals:
          loss     = sum over batches of `loss.item() * len(x_batch)`,
                     divided by `len(loader.dataset)`
          accuracy = number of correct predictions / `len(loader.dataset)`
    """
    raise NotImplementedError
# ============================ END TODO (Task 2) ==============================


# ============= TODO (Task 3): the training loop with early stopping ==========
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                lr: float, max_epochs: int = MAX_EPOCHS,
                patience: int = PATIENCE) -> dict:
    """Train `model` with early stopping on the validation loss.

    Setup:
      loss_fn   = nn.BCEWithLogitsLoss()
      optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    For epoch = 1, 2, ..., max_epochs:
      1. train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
      2. val = evaluate(model, val_loader, loss_fn)
      3. append train_loss, val["loss"], val["accuracy"] to the history lists
      4. if val["loss"] is STRICTLY smaller than the best value so far:
           record it as the best, remember the epoch number, save a copy of
           the weights with copy.deepcopy(model.state_dict()), and reset the
           counter of epochs without improvement to 0
         otherwise:
           add 1 to that counter, and BREAK when it reaches `patience`

    AFTER the loop (not inside it): restore the saved best weights into
    `model` with model.load_state_dict(...).

    `model` is modified in place — when this function returns, it must carry
    the weights of the BEST epoch, not the last one. Those differ whenever
    early stopping fires. (The function itself returns the history dict below.)

    Never train on val_loader, and never look at the test set here.

    Returns:
        {"train_loss": [float, ...],       one entry per epoch actually run
         "val_loss": [float, ...],
         "val_accuracy": [float, ...],
         "best_epoch": int,                1-based, the epoch of the lowest val loss
         "best_val_loss": float,           that lowest validation loss
         "epochs_run": int}                epochs actually run (<= max_epochs)
    """
    raise NotImplementedError
# ============================ END TODO (Task 3) ==============================


# ========== TODO (Task 4): the hyperparameter grid, chosen on val ============
def run_experiment(data: dict) -> dict:
    """Search the (hidden size x learning rate) grid, then test ONCE.

    `data` is the dict returned by prepare_data().

    Loop over HIDDEN_SIZES on the outside and LEARNING_RATES on the inside,
    so the grid records come out in that order. For each configuration:
      1. set_seed()                      reset the RNG, so the learning rates
                                         tried at one hidden size share the
                                         same weights and the same batch order
      2. model = build_model(data["input_dim"], hidden_size)
      3. history = train_model(model, data["train_loader"], data["val_loader"], lr)
      4. append one record to the grid list:
           {"hidden_size": int, "lr": float,
            "val_loss": float,        history["best_val_loss"]
            "val_accuracy": float,    history["val_accuracy"] at the best epoch
            "best_epoch": int, "epochs_run": int}

    Then pick the configuration with the SMALLEST "val_loss" (on a tie, the
    one that comes first in the grid), and only then call evaluate() once on
    data["test_loader"] with the model of that configuration.

    The test set is measured exactly once, after the search is over. Comparing
    configurations on the test set turns it into a second validation set, and
    the number you report stops being an estimate of unseen performance. The
    hidden tests count how many times the test loader is evaluated.

    Returns:
        {"grid": [record, ...],                   one per configuration
         "best_config": {"hidden_size": int, "lr": float},
         "val":  {"loss": float, "accuracy": float},   of the chosen configuration
         "test": {"loss": float, "accuracy": float}}   from the single test pass
    """
    raise NotImplementedError
# ============================ END TODO (Task 4) ==============================


def main() -> dict:
    """DO NOT MODIFY. Writes results.json with the course-wide schema."""
    set_seed()
    t0 = time.time()
    X, y = load_data(DATA_PATH)
    data = prepare_data(X, y)
    experiment = run_experiment(data)
    best = experiment["best_config"]
    results = {
        "lab": "lab04_2",
        "student_id": STUDENT_ID,
        "name": STUDENT_NAME,
        "seed": SEED,
        "metrics": {
            "n_train": data["n_train"],
            "n_val": data["n_val"],
            "n_test": data["n_test"],
            "positive_rate": data["positive_rate"],
            "n_configs": len(experiment["grid"]),
            "best_hidden_size": best["hidden_size"],
            "best_lr": best["lr"],
            "val_loss": experiment["val"]["loss"],
            "val_accuracy": experiment["val"]["accuracy"],
            "test_loss": experiment["test"]["loss"],
            "test_accuracy": experiment["test"]["accuracy"],
            "grid": experiment["grid"],
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    out = Path(__file__).resolve().parent.parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
