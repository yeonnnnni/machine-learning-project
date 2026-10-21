import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Fill in your information (used in results.json) ----
STUDENT_ID = "20231879"   # TODO: your student id
STUDENT_NAME = "LeeJeongyeon"     # TODO: your name in roman letters

SEED = 42  # fixed for the whole course — DO NOT CHANGE
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "press_qc.csv"
TARGET = "pass"

# Network and training configuration used by main(). Task 4 compares the three
# learning rates below while holding everything else fixed.
N_HIDDEN = 8
EPOCHS = 400
BATCH_SIZE = 32
LEARNING_RATES = (0.01, 1.0, 100.0)

# Machine spec sheet: the nominal setpoint and tolerance of the press.
# These are constants printed on the spec sheet, not statistics computed from
# the data, so scaling with them moves no information across the split.
TEMP_CENTER, TEMP_SCALE = 175.0, 25.0
PRESS_CENTER, PRESS_SCALE = 50.0, 30.0


# ===========================================================================
# Provided helpers — DO NOT MODIFY
# ===========================================================================
def set_seed(seed: int = SEED) -> None:
    """DO NOT MODIFY."""
    np.random.seed(seed)


def load_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """DO NOT MODIFY. Returns (X, y) as float64 arrays.

    X has shape (n, 2): temperature and pressure, each centred on the nominal
    setpoint and divided by the tolerance from the spec sheet, so both land in
    [-1, 1]. y has shape (n, 1) and holds 0.0 / 1.0.
    """
    df = pd.read_csv(path)
    X = np.column_stack([
        (df["temp_c"].to_numpy(dtype=np.float64) - TEMP_CENTER) / TEMP_SCALE,
        (df["pressure_kpa"].to_numpy(dtype=np.float64) - PRESS_CENTER) / PRESS_SCALE,
    ])
    y = df[TARGET].to_numpy(dtype=np.float64).reshape(-1, 1)
    return X, y


def train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2,
                    seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """DO NOT MODIFY. Stratified, shuffled, seeded holdout split (Lab 2, Task 1).

    Returns (X_train, X_val, y_train, y_val). Everyone gets the same split, so
    differences between runs come from the learning rate and nothing else.
    """
    rng = np.random.default_rng(seed)
    labels = y.ravel().astype(int)
    picked = []
    for c in (0, 1):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        picked.append(idx[:int(round(val_ratio * len(idx)))])
    mask = np.zeros(len(labels), dtype=bool)
    mask[np.concatenate(picked)] = True
    return X[~mask], X[mask], y[~mask], y[mask]


def init_params(n_features: int = 2, n_hidden: int = N_HIDDEN,
                seed: int = SEED) -> dict:
    """DO NOT MODIFY. Initial weights of a (n_features -> n_hidden -> 1) MLP.

    Returns {"W1": (n_features, n_hidden), "b1": (1, n_hidden),
             "W2": (n_hidden, 1),          "b2": (1, 1)}.
    Weights are Gaussian scaled by 1/sqrt(fan_in); biases start at zero.
    Same seed -> exactly the same starting point for everyone.
    """
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, 1.0, (n_features, n_hidden)) / np.sqrt(n_features),
        "b1": np.zeros((1, n_hidden)),
        "W2": rng.normal(0.0, 1.0, (n_hidden, 1)) / np.sqrt(n_hidden),
        "b2": np.zeros((1, 1)),
    }


def predict(X: np.ndarray, params: dict) -> np.ndarray:
    """DO NOT MODIFY. Hard 0/1 predictions, shape (n, 1), threshold 0.5."""
    return (forward(X, params)["a2"] >= 0.5).astype(np.float64)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """DO NOT MODIFY. Share of matching labels, as a plain Python float."""
    return float(np.mean(np.asarray(y_true).ravel() == np.asarray(y_pred).ravel()))
# ===========================================================================


# ==================== TODO (Task 1): the forward pass ======================
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Elementwise logistic function, sigma(z) = 1 / (1 + exp(-z)).

    Must be NUMERICALLY STABLE: sigmoid(-1000.0) and sigmoid(1000.0) must
    return 0.0 and 1.0 without an overflow warning and without nan.
    Hint: exp(-z) overflows for very negative z. Handle z >= 0 and z < 0
    separately, using exp(z)/(1+exp(z)) on the negative side.

    Returns an array of the same shape as z, with values in (0, 1).
    """
    # raise NotImplementedError

    # 입력과 같은 모양의 실수형 결과 배열 생성
    z = np.asarray(z, dtype=np.float64)
    result = np.empty_like(z, dtype=np.float64)

    # 입력이 0 이상인 위치와 0 미만인 위치 구분
    positive = z >= 0
    negative = ~positive

    # 0 이상인 위치는 1 / (1 + exp(-z))로 계산
    result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))

    # 0 미만인 위치는 exp(z) / (1 + exp(z))로 계산
    exp_z = np.exp(z[negative])
    result[negative] = exp_z / (1.0 + exp_z)

    # 입력과 같은 모양의 결과 배열 반환
    return result


def forward(X: np.ndarray, params: dict) -> dict:
    """Run the (d -> h -> 1) network forward and keep every intermediate value.

    With X of shape (n, d) and params from init_params():
        z1 = X  @ W1 + b1        shape (n, h)
        a1 = sigmoid(z1)         shape (n, h)
        z2 = a1 @ W2 + b2        shape (n, 1)
        a2 = sigmoid(z2)         shape (n, 1)   <- predicted P(y = 1)

    Returns:
        {"z1": z1, "a1": a1, "z2": z2, "a2": a2}
    The cache is not a convenience: backward() needs a1 and a2, and
    recomputing them there is where most from-scratch bugs come from.
    """
    # raise NotImplementedError

    # params에서 W1, b1, W2, b2 꺼냄
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    # z1 = 입력과 W1의 행렬 곱 + b1
    z1 = X @ W1 + b1

    # a1 = z1에 sigmoid 적용
    a1 = sigmoid(z1)

    # z2 = a1과 W2의 행렬 곱 + b2
    z2 = a1 @ W2 + b2

    # a2 = z2에 sigmoid 적용
    a2 = sigmoid(z2)

    # z1, a1, z2, a2를 딕셔너리로 반환
    return {"z1": z1, "a1": a1, "z2": z2, "a2": a2}


def bce_loss(y: np.ndarray, y_hat: np.ndarray, eps: float = 1e-12) -> float:
    """Mean binary cross-entropy over the batch, as a plain Python float.

        L = -(1/n) * sum( y*log(p) + (1-y)*log(1-p) ),
        where p = clip(y_hat, eps, 1 - eps).

    Clipping is what stops log(0) from producing inf when the network becomes
    very confident. y and y_hat both have shape (n, 1).
    """
    # raise NotImplementedError

    # 예측 확률을 eps 이상, 1 - eps 이하로 제한
    p = np.clip(y_hat, eps, 1.0 - eps)

    # 각 데이터의 손실 = -(정답 * log(확률)  + (1 - 정답) * log(1 - 확률))
    losses = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    # 전체 데이터의 평균 손실 계산
    loss = np.mean(losses)

    # 평균 손실을 Python float로 반환
    return float(loss)


# Task 1 검증:
# python -m pytest -q tests/test_public.py::test_sigmoid_properties tests/test_public.py::test_forward_shapes_and_cache tests/test_public.py::test_bce_loss_values
# ========================== END TODO (Task 1) ==============================


# =============== TODO (Task 2): gradients and a gradient check =============
def backward(X: np.ndarray, y: np.ndarray, params: dict, cache: dict) -> dict:
    """Analytic gradients of bce_loss with respect to every parameter.

    Derived in the concept note; transcribe it exactly. n = X.shape[0]:
        dz2 = (a2 - y) / n                        shape (n, 1)
        dW2 = a1.T @ dz2                          shape (h, 1)
        db2 = dz2.sum(axis=0, keepdims=True)      shape (1, 1)
        da1 = dz2 @ W2.T                          shape (n, h)
        dz1 = da1 * a1 * (1 - a1)                 shape (n, h)
        dW1 = X.T @ dz1                           shape (d, h)
        db1 = dz1.sum(axis=0, keepdims=True)      shape (1, h)

    The 1/n belongs in dz2 and appears exactly once. Putting it in twice, or
    leaving it out, is the most common error here and the gradient check
    below will show it.

    Returns:
        {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        — each gradient has the SAME shape as the parameter it belongs to.
    """
    # raise NotImplementedError

    # 현재 배치의 데이터 개수 n 확인
    n = X.shape[0]

    # cache에서 은닉층 출력 a1과 예측 확률 a2 꺼냄
    a1 = cache["a1"]
    a2 = cache["a2"]

    # params에서 출력층 가중치 W2 꺼냄
    W2 = params["W2"]

    # 출력층 오차 dz2 = (a2 - 정답) / n
    dz2 = (a2 - y) / n

    # dW2 = a1의 전치와 dz2의 행렬 곱
    dW2 = a1.T @ dz2

    # db2 = dz2를 행 방향으로 합산하고 2차원 모양 유지
    db2 = dz2.sum(axis=0, keepdims=True)

    # 은닉층으로 전달할 기울기 da1 = dz2와 W2 전치의 행렬 곱
    da1 = dz2 @ W2.T

    # 은닉층 오차 dz1 = da1 * a1 * (1 - a1)
    dz1 = da1 * a1 * (1.0 - a1)

    # dW1 = 입력의 전치와 dz1의 행렬 곱
    dW1 = X.T @ dz1

    # db1 = dz1을 행 방향으로 합산하고 2차원 모양 유지
    db1 = dz1.sum(axis=0, keepdims=True)

    # 평균 처리는 dz2에서 n으로 나누는 과정에 한 번만 적용
    # 각 기울기는 해당 파라미터와 같은 모양으로 유지
    # W1, b1, W2, b2를 키로 하는 기울기 딕셔너리 반환
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

def gradient_check(X: np.ndarray, y: np.ndarray, params: dict,
                   eps: float = 1e-5) -> dict:
    """Compare backward() against central finite differences.

    Get the analytic side by calling backward() itself — do not re-derive the
    gradients inside this function. A hidden test swaps backward() for a
    deliberately wrong version and expects gradient_check() to report a large
    error; a re-derived copy would keep reporting a small one.

    For every parameter array and every entry i in it, move that one entry by
    +eps and by -eps, recompute the loss with forward() and bce_loss(), and
    form the numerical derivative

        g_num[i] = (L(theta_i + eps) - L(theta_i - eps)) / (2 * eps).

    Restore the entry before moving on to the next one. `theta.ravel()` gives a
    flat VIEW of a parameter array (a view, not a copy, for the arrays
    init_params returns), so writing to it changes `params` in place, which is
    what makes the two forward passes see the perturbed value.

    Compare against the analytic gradient with the relative error

        rel[i] = |g_ana[i] - g_num[i]| / max(1e-8, |g_ana[i]| + |g_num[i]|)

    Returns:
        {"W1": float, "b1": float, "W2": float, "b2": float}
        — for each key, the LARGEST rel[i] over the entries of that array.
    A correct implementation stays below 1e-6; typical values run from 1e-13
    to 1e-7 (the W1 entries, whose gradients are smallest, sit at the top of
    that range). Anything above 1e-6 means the analytic gradient is wrong.
    """
    # raise NotImplementedError

    # forward로 원래 파라미터의 중간값 계산
    cache = forward(X, params)

    # backward를 호출하여 해석적 기울기 계산
    analytic_grads = backward(X, y, params, cache)

    # 파라미터별 최대 상대오차를 저장할 딕셔너리 생성
    errors = {}

    # 각 파라미터 배열의 모든 원소에 대해 아래 과정 반복
    for key, theta in params.items():
        numerical_grads = np.zeros_like(theta, dtype=np.float64)

        for index in np.ndindex(theta.shape):
            #   현재 원소의 원래 값 저장
            original = theta[index]

            try:
                #   원소를 원래 값 + eps로 바꾸고 forward와 bce_loss로 손실 계산
                theta[index] = original + eps
                plus_loss = bce_loss(y, forward(X, params)["a2"])

                #   원소를 원래 값 - eps로 바꾸고 forward와 bce_loss로 손실 계산
                theta[index] = original - eps
                minus_loss = bce_loss(y, forward(X, params)["a2"])
            finally:
                #   해당 원소를 원래 값으로 복원
                theta[index] = original

            #   수치 기울기 = (증가시킨 손실 - 감소시킨 손실) / (2 * eps)
            numerical_grads[index] = (plus_loss - minus_loss) / (2.0 * eps)

        #   분자 = abs(해석적 기울기 - 수치 기울기)
        numerator = np.abs(analytic_grads[key] - numerical_grads)

        #   분모 = max(1e-8, abs(해석적 기울기) + abs(수치 기울기))
        denominator = np.maximum(
            1e-8, np.abs(analytic_grads[key]) + np.abs(numerical_grads)
        )

        #   상대오차 = 분자 / 분모
        relative_errors = numerator / denominator

        # 각 파라미터 배열에서 가장 큰 상대오차 선택
        max_error = np.max(relative_errors)

        # 최대 상대오차를 Python float로 변환하여 딕셔너리에 저장
        errors[key] = float(max_error)

    # 파라미터별 최대 상대오차 딕셔너리 반환
    return errors

# Task 2 검증:
# python -m pytest -q tests/test_public.py::test_backward_shapes_and_gradient_check
# ========================== END TODO (Task 2) ==============================


# ================= TODO (Task 3): the mini-batch training loop =============
def train(X_train: np.ndarray, y_train: np.ndarray,
          X_val: np.ndarray, y_val: np.ndarray,
          n_hidden: int = N_HIDDEN, lr: float = 1.0, epochs: int = EPOCHS,
          batch_size: int = BATCH_SIZE, seed: int = SEED) -> tuple[dict, dict]:
    """Train the MLP with mini-batch gradient descent and record the history.

    Steps, in this order:
      1. params = init_params(X_train.shape[1], n_hidden, seed)
      2. rng = np.random.default_rng(seed)
      3. Repeat `epochs` times:
         a. perm = rng.permutation(len(X_train))          (reshuffle every epoch)
         b. walk perm in slices of `batch_size`; the last slice may be shorter.
            For each slice: forward() on that slice, backward() on that slice,
            then for every key:  params[key] -= lr * grads[key]
         c. AFTER the epoch, evaluate on the FULL sets and append one entry each:
              history["train_loss"]   : bce_loss on (X_train, y_train)
              history["val_loss"]     : bce_loss on (X_val,   y_val)
              history["val_accuracy"] : accuracy(y_val, predict(X_val, params))

    Only the training slices ever touch the update. The validation set is read
    for measurement and never for a gradient — a hidden test checks this by
    corrupting y_val and confirming the parameters do not change.

    Returns:
        (params, history) where history is
        {"train_loss": [...], "val_loss": [...], "val_accuracy": [...]},
        each list of length `epochs`, values plain Python floats.
    """
    # raise NotImplementedError

    # 입력 변수 수, 은닉 노드 수, seed를 이용해 초기 파라미터 생성
    params = init_params(X_train.shape[1], n_hidden, seed)

    # seed를 이용해 데이터 순서를 섞을 난수 생성기 생성
    rng = np.random.default_rng(seed)

    # train_loss, val_loss, val_accuracy를 기록할 빈 리스트 생성
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    # 지정된 epoch 수만큼 아래 과정 반복
    for _ in range(epochs):
        #   전체 학습 데이터의 인덱스를 새롭게 섞음
        perm = rng.permutation(len(X_train))

        #   섞인 인덱스를 batch_size 단위로 나눔
        #   마지막 배치의 크기가 작아도 학습에 포함
        #   각 배치에 대해 아래 과정 반복
        for start in range(0, len(X_train), batch_size):
            batch_indices = perm[start:start + batch_size]

            #     같은 인덱스로 입력과 정답 선택
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            #     forward로 배치의 예측과 중간값 계산
            cache = forward(X_batch, params)

            #     backward로 배치의 파라미터별 기울기 계산
            grads = backward(X_batch, y_batch, params, cache)

            #     각 파라미터에서 학습률 * 기울기를 빼서 업데이트
            for key in params:
                params[key] -= lr * grads[key]

        #   해당 epoch의 모든 배치 업데이트 종료 후 전체 데이터 평가
        #   전체 학습 데이터의 예측 확률과 평균 BCE 손실 계산
        train_prob = forward(X_train, params)["a2"]
        train_loss = bce_loss(y_train, train_prob)

        #   전체 검증 데이터의 예측 확률과 평균 BCE 손실 계산
        val_prob = forward(X_val, params)["a2"]
        val_loss = bce_loss(y_val, val_prob)

        #   predict와 accuracy를 이용해 검증 정확도 계산
        val_accuracy = accuracy(y_val, predict(X_val, params))

        #   학습 손실, 검증 손실, 검증 정확도를 각 리스트에 한 번씩 추가
        #   기록값은 Python float로 유지
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

    # 검증 데이터는 손실과 정확도 측정에만 사용
    # 각 기록 리스트의 길이는 지정된 epoch 수와 동일하게 유지
    # 최종 파라미터와 학습 기록을 튜플로 반환
    return params, history

# Task 3 검증:
# python -m pytest -q tests/test_public.py::test_train_history_and_convergence
# ========================== END TODO (Task 3) ==============================


# ============== TODO (Task 4): controlled learning-rate comparison =========
def run_experiment(X: np.ndarray, y: np.ndarray) -> dict:
    """Compare the learning rates in LEARNING_RATES under IDENTICAL conditions.

    Controlled experiment = one factor varies (the learning rate); everything
    else is fixed: the same split, the same seed, the same initial weights,
    the same number of epochs, the same batch size.

    Steps:
      1. X_train, X_val, y_train, y_val = train_val_split(X, y)   (defaults)
      2. For each lr in LEARNING_RATES, call train(..., lr=lr) and leave every
         other argument at its default.
      3. Store, under the key str(lr) — that is "0.01", "1.0", "100.0" —
             {"final_train_loss":   history["train_loss"][-1],
              "max_train_loss":     max(history["train_loss"]),
              "final_val_loss":     history["val_loss"][-1],
              "final_val_accuracy": history["val_accuracy"][-1],
              "best_val_accuracy":  max(history["val_accuracy"])}
         each value a plain Python float rounded to 4 decimals.

    `max_train_loss` is there because a diverged run oscillates: its last
    epoch is one arbitrary point on a curve that swings by an order of
    magnitude, so the final value alone does not show what happened.

    Returns:
        {"0.01": {...}, "1.0": {...}, "100.0": {...}}
    """
    # raise NotImplementedError

    # 전체 데이터를 학습 데이터와 검증 데이터로 한 번 분할
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    # 학습률별 결과를 저장할 딕셔너리 생성
    experiments = {}

    # LEARNING_RATES의 각 학습률에 대해 아래 과정 반복
    for lr in LEARNING_RATES:
        #   동일한 데이터 분할을 사용하여 train 호출
        #   학습률만 변경하고 나머지 인자는 기본값 유지
        #   각 학습은 동일한 seed와 초기 가중치에서 시작
        _, history = train(X_train, y_train, X_val, y_val, lr=lr)

        record = {
            #   final_train_loss = 마지막 epoch의 학습 손실
            "final_train_loss": history["train_loss"][-1],
            #   max_train_loss = 전체 epoch 중 가장 큰 학습 손실
            "max_train_loss": max(history["train_loss"]),
            #   final_val_loss = 마지막 epoch의 검증 손실
            "final_val_loss": history["val_loss"][-1],
            #   final_val_accuracy = 마지막 epoch의 검증 정확도
            "final_val_accuracy": history["val_accuracy"][-1],
            #   best_val_accuracy = 전체 epoch 중 가장 높은 검증 정확도
            "best_val_accuracy": max(history["val_accuracy"]),
        }

        #   각 지표를 소수점 4자리로 반올림하고 Python float로 변환
        record = {key: float(round(value, 4)) for key, value in record.items()}

        #   학습률을 문자열로 변환하여 결과 딕셔너리의 키로 사용
        lr_key = str(lr)

        #   해당 키에 다섯 지표를 담은 딕셔너리 저장
        experiments[lr_key] = record

    # "0.01", "1.0", "100.0"을 키로 하는 실험 결과 딕셔너리 반환
    return experiments

# Task 4 검증:
# python -m pytest -q tests/test_public.py::test_run_experiment_compares_learning_rates
# ========================== END TODO (Task 4) ==============================


def main() -> dict:
    """DO NOT MODIFY. Writes results.json with the course-wide schema."""
    set_seed()
    t0 = time.time()
    X, y = load_data(DATA_PATH)
    X_train, _, y_train, _ = train_val_split(X, y)
    check = gradient_check(X_train[:16], y_train[:16],
                           init_params(X.shape[1], N_HIDDEN, SEED))
    experiment = run_experiment(X, y)
    best = max(experiment, key=lambda k: experiment[k]["final_val_accuracy"])
    results = {
        "lab": "lab04_1",
        "student_id": STUDENT_ID,
        "name": STUDENT_NAME,
        "seed": SEED,
        "metrics": {
            "n_samples": int(len(X)),
            "positive_rate": float(round(y.mean(), 4)),
            "majority_baseline_accuracy": float(round(max(y.mean(), 1 - y.mean()), 4)),
            "n_hidden": N_HIDDEN,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_check_max_rel_error": float(f"{max(check.values()):.3g}"),
            **{f"lr_{name}_{k}": v for name, d in experiment.items() for k, v in d.items()},
            "best_lr": float(best),
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    out = Path(__file__).resolve().parent.parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
