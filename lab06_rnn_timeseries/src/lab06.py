"""Lab 6 — RNNs and time-series forecasting: a forecast that does not peek.

Machine Learning Project (53744-01), Fall 2026.

Fill in every TODO block. Do NOT rename functions or change their
signatures/return types — automated (public + hidden) tests call them directly.
Run:      python src/lab06.py
Self-check: python -m pytest tests/ -q
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

# ---- Fill in your information (used in results.json) ----
STUDENT_ID = "20231879"   # TODO: your student id
STUDENT_NAME = "LeeJeongyeon"     # TODO: your name in Korean or roman letters — "홍길동" / "HongGildong"

SEED = 42  # fixed for the whole course — DO NOT CHANGE
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cafe_daily.csv"
TARGET = "cups"

# ---- Experiment constants (the same for everyone; do not change them) ----
WINDOW = 14          # length of the input window, in days
HORIZON = 1          # how many steps ahead the target sits
GAP = 7              # embargo: days dropped between consecutive segments
VAL_FRAC = 0.15      # share of the series used for validation
TEST_FRAC = 0.15     # share of the series used for the final test
HIDDEN_SIZE = 32     # GRU hidden units
EPOCHS = 150
LEARNING_RATE = 0.01
BATCH_SIZE = 64


def set_seed(seed: int = SEED) -> None:
    """DO NOT MODIFY. Makes the whole run reproducible on CPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def load_data(path: str | Path) -> np.ndarray:
    """DO NOT MODIFY.

    Reads the csv, sorts it by date (never assume the file is ordered), and
    returns the `cups` column as a 1-D float64 numpy array in chronological
    order.
    """
    df = pd.read_csv(path).sort_values("date").reset_index(drop=True)
    return df[TARGET].to_numpy(dtype=np.float64)


# ================== TODO (Task 1): windows without lookahead =================
def make_windows(series: np.ndarray, window: int = WINDOW,
                 horizon: int = HORIZON) -> tuple[np.ndarray, np.ndarray]:
    """Cut a 1-D series into supervised (input window, target) pairs.

    For a series s of length N, pair i is
        X[i] = s[i : i+window]                  (`window` consecutive values)
        y[i] = s[i + window + horizon - 1]      (one value, `horizon` steps
                                                 after the end of the window)
    for i = 0, 1, ..., N - window - horizon, so there are
    N - window - horizon + 1 pairs. With horizon=1 the target is the point
    immediately after the window.

    Every value in X[i] must come strictly BEFORE y[i] in time. A window that
    contains its own target is lookahead, the model would score beautifully,
    and the hidden tests are built to catch it.

    Returns:
        (X, y) with X of shape (n_pairs, window) and y of shape (n_pairs,),
        both float64. If N < window + horizon, return two empty arrays with
        shapes (0, window) and (0,).
    """
    # raise NotImplementedError

    # 1. 입력 시계열을 float64 numpy 배열로 변환
    series = np.asarray(series, dtype=np.float64)
    N = len(series)

    # 2. 생성 가능한 입력-정답 쌍의 개수 계산
    n_pairs = N - window - horizon + 1

    # 3. 데이터 부족 시 지정된 shape의 빈 배열 반환
    if n_pairs <= 0:
        return np.empty((0, window), dtype=np.float64), np.empty((0,), dtype=np.float64)

    # 4. 입력 윈도와 정답을 저장할 float64 배열 생성
    X = np.empty((n_pairs, window), dtype=np.float64)
    y = np.empty((n_pairs,), dtype=np.float64)

    # 5. 시작 위치를 하루씩 이동하며 입력 윈도와 미래 정답 저장
    for i in range(n_pairs):
        X[i] = series[i:i + window]  # 입력 윈도
        y[i] = series[i + window + horizon - 1]  # 미래 정답

    # 6. 입력 배열 X와 정답 배열 y 반환
    return X, y


# ============================ END TODO (Task 1) ==============================


# ===================== TODO (Task 2): chronological split ====================
def temporal_split(n: int, val_frac: float = VAL_FRAC, test_frac: float = TEST_FRAC,
                   gap: int = GAP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the positions 0..n-1 of a series into train / val / test IN TIME ORDER.

    NEVER shuffle, and never use train_test_split here. Training must come
    first, then validation, then test, so that no training point sits after a
    validation or test point.

    Sizes (in this order, so that the three parts plus the gaps add up to n):
        n_test  = int(round(test_frac * n))
        n_val   = int(round(val_frac * n))
        n_train = n - n_val - n_test - 2*gap

    `gap` is the embargo: that many positions are dropped between train and
    val, and again between val and test, so the segments do not touch.

        train = 0 .. n_train-1
        (gap dropped)
        val   = n_train+gap .. n_train+gap+n_val-1
        (gap dropped)
        test  = n_train+2*gap+n_val .. n-1

    Raise ValueError if the three parts and the two gaps leave nothing for
    training (n_train <= 0).

    Returns:
        (train_idx, val_idx, test_idx), each a 1-D numpy array of ints in
        increasing order.
    """
    # raise NotImplementedError

    # 1. test와 validation의 날짜 수를 반올림하여 계산
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))

    # 2. 두 구간과 두 gap을 제외한 train 날짜 수 계산
    n_train = n - n_val - n_test - 2 * gap

    # 3. train 데이터가 남지 않는 경우 ValueError 발생
    if n_train <= 0:
        raise ValueError("n_train <= 0")

    # 4. 처음부터 train 종료 지점까지의 정수 인덱스 생성
    train_idx = np.arange(n_train)

    # 5. 첫 번째 gap 이후 validation 정수 인덱스 생성
    val_start = n_train + gap
    val_idx = np.arange(val_start, val_start + n_val)

    # 6. 두 번째 gap 이후 test 정수 인덱스 생성
    test_start = val_start + n_val + gap
    test_idx = np.arange(test_start, n)

    # 7. 시간순 train, validation, test 인덱스 배열 반환
    return train_idx, val_idx, test_idx

# ============================ END TODO (Task 2) ==============================


# ======================= TODO (Task 3): the recurrent model ==================
class ForecastRNN(nn.Module):
    """GRU over the input window, then a linear head on the LAST hidden state.

    Attribute names are fixed (the tests look them up):
      self.rnn  = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
      self.head = nn.Linear(hidden_size, 1)

    forward(x):
      x has shape (batch, seq_len, input_size)  — batch first.
      Run it through self.rnn, take the output of the LAST TIME STEP for every
      sequence in the batch, and pass that through self.head.
      Return shape (batch, 1).

    The whole task is picking the right slice. self.rnn returns
    `out` of shape (batch, seq_len, hidden_size), so the last time step is
    out[:, -1, :]. Indexing the batch dimension instead (out[-1]) gives the
    wrong shape when batch != seq_len, and — worse — passes the shape check
    while returning a completely different value when they happen to be equal.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = 1) -> None:
        super().__init__()
        # TODO: create self.rnn and self.head with the names above.
        # raise NotImplementedError
        # 1. 부모 클래스 초기화 유지
        super().__init__()

        # 2. 지정된 크기와 batch_first=True 설정으로 self.rnn 생성
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        # 3. hidden_size를 출력값 하나로 변환하는 self.head 생성
        self.head = nn.Linear(hidden_size, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: (batch, seq_len, input_size) -> (batch, 1)
        # raise NotImplementedError

        # 4. 입력 시퀀스를 GRU에 전달하여 전체 시점의 출력 획득
        out, _ = self.rnn(x)

        # 5. 모든 배치에서 마지막 시간 위치의 hidden state 선택
        last_hidden = out[:, -1, :]

        # 6. 선택한 hidden state를 head에 전달하여 예측값 반환
        return self.head(last_hidden)


def build_rnn(input_size: int = 1, hidden_size: int = HIDDEN_SIZE,
              num_layers: int = 1) -> ForecastRNN:
    """Return an UNTRAINED ForecastRNN with the given sizes."""
    # raise NotImplementedError

    # 7. 전달받은 크기로 미학습 ForecastRNN 모델 생성 및 반환
    return ForecastRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
# ============================ END TODO (Task 3) ==============================


def _to_tensor(X: np.ndarray, y: np.ndarray | None = None):
    """DO NOT MODIFY. numpy (n, window) -> torch (n, window, 1)."""
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32).unsqueeze(-1)
    if y is None:
        return Xt
    return Xt, torch.tensor(np.asarray(y), dtype=torch.float32).unsqueeze(-1)


def train_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, epochs: int = EPOCHS,
                lr: float = LEARNING_RATE) -> tuple[nn.Module, dict]:
    """DO NOT MODIFY. Adam + MSE, mini-batches in chronological order.

    Batches are NOT shuffled and no dropout is used, so the run consumes no
    randomness after the weights are initialised. The returned model is the
    checkpoint with the lowest validation loss, which is the only job the
    validation segment has in this lab.
    """
    Xt, yt = _to_tensor(X_train, y_train)
    Xv, yv = _to_tensor(X_val, y_val)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for start in range(0, len(Xt), BATCH_SIZE):
            xb, yb = Xt[start:start + BATCH_SIZE], yt[start:start + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        history["train_loss"].append(total / len(Xt))
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    return model, history


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """DO NOT MODIFY. Returns a 1-D numpy array of predictions, one per window."""
    model.eval()
    with torch.no_grad():
        return model(_to_tensor(X)).squeeze(-1).numpy().astype(np.float64)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    naive_scale: float) -> dict:
    """DO NOT MODIFY. Scale-appropriate error metrics, all in the original units.

      mae  = mean(|y_true - y_pred|)
      rmse = sqrt(mean((y_true - y_pred)^2))
      mase = mae / naive_scale

    `naive_scale` is the mean absolute one-step change of the TRAINING series,
    so mase < 1 means "off by less than the series typically moves in a day".
    Persistence itself need not land at 1 on the test segment. Values are
    plain Python floats rounded to 4 decimals.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4),
            "mase": round(mae / float(naive_scale), 4)}


# ============= TODO (Task 4): persistence baseline against the RNN ===========
def run_experiment(series: np.ndarray) -> dict:
    """Compare the persistence baseline with the RNN under IDENTICAL conditions.

    Same split, same windows, same targets, same metrics. Only the forecaster
    changes.

    Steps:
      1. set_seed()  — call it FIRST, before the model is built, so that two
         calls to run_experiment in one process give identical numbers.
      2. train_idx, val_idx, test_idx = temporal_split(len(series))
      3. Standardisation constants from the TRAINING SEGMENT ONLY:
             mean = np.mean(series[train_idx])
             std  = np.std(series[train_idx])        (population std, ddof=0)
         Fitting them on the whole series is leakage. The hidden tests read
         "scale_mean" back and compare it with the training segment.
      4. Standardise each segment with those two numbers: (segment - mean)/std.
      5. Window each standardised segment SEPARATELY with
         make_windows(segment, WINDOW, HORIZON). Never build a window that
         starts in one segment and ends in another.
      6. model = build_rnn(); model, _ = train_model(model, X_train, y_train,
         X_val, y_val).
      7. RNN forecast on the test windows, converted back to cups:
             pred_rnn = predict(model, X_test) * std + mean
         The true test targets in cups are y_test * std + mean.
      8. Persistence forecast: the last value of each test input window, in
         cups. It repeats the most recent observation and learns nothing.
      9. naive_scale = mean(|difference between consecutive values|) of
         series[train_idx], in cups.
     10. compute_metrics(y_true, y_pred, naive_scale) for both forecasters.

    Returns:
        {"n_train": int, "n_val": int, "n_test": int,      # windows, not days
         "scale_mean": float, "scale_std": float, "naive_scale": float,
         "persistence": {"mae":…, "rmse":…, "mase":…},
         "rnn":         {"mae":…, "rmse":…, "mase":…}}
        scale_mean, scale_std and naive_scale are rounded to 4 decimals
        (the metric dicts are already rounded by compute_metrics).
    """
    # raise NotImplementedError

    # 1. 함수 시작 시 난수 시드 초기화
    set_seed()

    # 2. 원본 시계열의 날짜 인덱스를 시간순으로 분할
    train_idx, val_idx, test_idx = temporal_split(len(series))

    # 3. 원본 단위의 train, validation, test 구간 추출
    train_segment = series[train_idx]
    val_segment = series[val_idx]
    test_segment = series[test_idx]

    # 4. 원본 train 구간에서만 평균과 모집단 표준편차 계산
    scale_mean = np.mean(train_segment)
    scale_std = np.std(train_segment, ddof=0)

    # 5. train의 평균과 표준편차로 세 구간 표준화
    train_standardized = (train_segment - scale_mean) / scale_std
    val_standardized = (val_segment - scale_mean) / scale_std
    test_standardized = (test_segment - scale_mean) / scale_std

    # 6. 각 구간에서 별도로 입력 윈도와 정답 생성
    X_train, y_train = make_windows(train_standardized, WINDOW, HORIZON)
    X_val, y_val = make_windows(val_standardized, WINDOW, HORIZON)
    X_test, y_test = make_windows(test_standardized, WINDOW, HORIZON)

    # 7. GRU 모델 생성 및 제공된 학습 함수로 학습
    model = build_rnn()
    model, _ = train_model(model, X_train, y_train, X_val, y_val)

    # 8. GRU 예측값과 test 정답을 원래 cups 단위로 복원
    pred_rnn = predict(model, X_test) * scale_std + scale_mean
    y_test_cups = y_test * scale_std + scale_mean

    # 9. 각 test 입력 윈도의 마지막 관측값을 cups 단위로 복원하여 persistence 예측 생성
    pred_persistence = X_test[:, -1] * scale_std + scale_mean

    # 10. 원본 train 구간의 평균 절대 일간 변화량으로 naive_scale 계산
    naive_scale = np.mean(np.abs(np.diff(train_segment)))

    # 11. 동일한 정답과 naive_scale로 persistence와 GRU의 평가지표 계산
    metrics_persistence = compute_metrics(y_test_cups, pred_persistence, naive_scale)
    metrics_rnn = compute_metrics(y_test_cups, pred_rnn, naive_scale)

    # 12. 윈도 수, 반환 시 반올림한 상수, 두 모델의 평가지표를 딕셔너리로 반환
    return {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "scale_mean": round(float(scale_mean), 4),
        "scale_std": round(float(scale_std), 4),
        "naive_scale": round(float(naive_scale), 4),
        "persistence": metrics_persistence,
        "rnn": metrics_rnn
    }
# ============================ END TODO (Task 4) ==============================


def main() -> dict:
    """DO NOT MODIFY. Writes results.json with the course-wide schema."""
    set_seed()
    t0 = time.time()
    series = load_data(DATA_PATH)
    experiment = run_experiment(series)
    results = {
        "lab": "lab06",
        "student_id": STUDENT_ID,
        "name": STUDENT_NAME,
        "seed": SEED,
        "metrics": {
            "n_points": int(len(series)),
            "window": WINDOW,
            "horizon": HORIZON,
            "gap": GAP,
            **{k: experiment[k] for k in
               ("n_train", "n_val", "n_test", "scale_mean", "scale_std", "naive_scale")},
            **{f"{m}_{k}": v for m in ("persistence", "rnn")
               for k, v in experiment[m].items()},
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    out = Path(__file__).resolve().parent.parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
