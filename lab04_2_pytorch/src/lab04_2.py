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
STUDENT_ID = "20231879"   # TODO: your student id
STUDENT_NAME = "LeeJeongyeon"     # TODO: your name in roman letters

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
    # raise NotImplementedError

    # 모델 구성
    # 첫 번째 층으로 nn.Linear(input_dim, hidden_size) 생성
    layer1 = nn.Linear(input_dim, hidden_size)
    # 두 번째 층으로 nn.ReLU() 생성
    layer2 = nn.ReLU()
    # 세 번째 층으로 nn.Linear(hidden_size, hidden_size) 생성
    layer3 = nn.Linear(hidden_size, hidden_size)
    # 네 번째 층으로 nn.ReLU() 생성
    layer4 = nn.ReLU()
    # 다섯 번째 층으로 nn.Linear(hidden_size, 1) 생성
    layer5 = nn.Linear(hidden_size, 1)
    # 위 다섯 개 층을 순서대로 nn.Sequential에 묶기
    model = nn.Sequential(layer1, layer2, layer3, layer4, layer5)

    # 출력 및 반환
    # 마지막 출력은 확률 변환 전의 logit이며, 모양은 (batch, 1)
    # 확률 변환은 손실 함수에 맡기므로 모델은 Linear 층에서 끝남
    # 시드 초기화는 호출자인 Task 4에 맡김
    # 학습되지 않은 nn.Sequential 모델 반환
    return model

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
    # raise NotImplementedError

    # 학습 준비
    # model.train()으로 모델을 학습 모드로 전환
    model.train()
    # 전체 샘플의 손실을 누적할 total_loss를 0으로 초기화
    total_loss = 0.0
    # 전체 샘플 수를 len(loader.dataset)으로 구함
    n_samples = len(loader.dataset)

    # loader에서 x_batch와 y_batch를 꺼내며 모든 배치를 한 번씩 순회
    for x_batch, y_batch in loader:
        # 각 배치에서 아래 다섯 단계를 순서대로 수행

        # 1. optimizer.zero_grad()로 이전 배치의 기울기 초기화
        optimizer.zero_grad()
        # 2. model(x_batch)로 현재 배치의 logits 계산
        logits = model(x_batch)
        # 3. loss_fn(logits, y_batch)로 현재 배치의 평균 손실 계산
        loss = loss_fn(logits, y_batch)
        # 4. loss.backward()로 모델 파라미터에 대한 기울기 계산
        loss.backward()
        # 5. optimizer.step()으로 모델 파라미터 갱신
        optimizer.step()

        # 현재 배치의 샘플 수를 len(x_batch)로 구함
        batch_size = len(x_batch)
        # loss.item()으로 배치 평균 손실을 Python 숫자로 꺼냄
        batch_loss = loss.item()
        # 배치 평균 손실에 현재 배치의 샘플 수를 곱함
        batch_total_loss = batch_loss * batch_size
        # 계산한 배치 손실 합을 total_loss에 누적
        total_loss += batch_total_loss

    # 모든 배치의 학습이 끝난 뒤 평균 손실 계산
    # total_loss를 전체 샘플 수로 나누어 샘플당 평균 손실 계산
    mean_loss = total_loss / n_samples
    # 평균 손실을 소수점 6자리로 반올림
    mean_loss = round(mean_loss, 6)
    # 반올림한 값을 Python float로 반환
    return float(mean_loss)


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
    # raise NotImplementedError

    # 평가 준비
    # model.eval()로 모델을 평가 모드로 전환
    model.eval()
    # 전체 손실을 누적할 total_loss를 0으로 초기화
    total_loss = 0.0
    # 맞힌 샘플 수를 누적할 total_correct를 0으로 초기화
    total_correct = 0
    # 전체 샘플 수를 len(loader.dataset)으로 구함
    n_samples = len(loader.dataset)

    # torch.no_grad() 범위 안에서 전체 배치 평가
    with torch.no_grad():
        # loader에서 x_batch와 y_batch를 꺼내며 모든 배치를 순회
        for x_batch, y_batch in loader:

            # 현재 배치의 손실 계산
            # model(x_batch)로 logits 계산
            logits = model(x_batch)
            # loss_fn(logits, y_batch)로 배치 평균 손실 계산
            loss = loss_fn(logits, y_batch)
            # loss.item()에 len(x_batch)를 곱하여 배치 손실 합 계산
            batch_total_loss = loss.item() * len(x_batch)
            # 배치 손실 합을 total_loss에 누적
            total_loss += batch_total_loss

            # 현재 배치의 정답 수 계산
            # logits > 0인 샘플은 1, 나머지 샘플은 0으로 예측
            # 예측값과 y_batch의 모양을 동일한 (batch, 1)로 유지
            predictions = (logits > 0).to(y_batch.dtype)
            # 예측값과 y_batch를 비교하여 정답 여부 계산
            matches = predictions == y_batch
            # 정답 여부를 합산하고 item()으로 Python 숫자로 꺼냄
            batch_correct = matches.sum().item()
            # 현재 배치의 정답 수를 total_correct에 누적
            total_correct += batch_correct

    # 모든 배치의 평가가 끝난 뒤 지표 계산
    # 평균 손실 = total_loss / 전체 샘플 수
    mean_loss = total_loss / n_samples
    # 정확도 = total_correct / 전체 샘플 수
    accuracy = total_correct / n_samples
    # 평균 손실과 정확도를 각각 소수점 6자리로 반올림
    # 두 값을 각각 Python float로 변환
    mean_loss = float(round(mean_loss, 6))
    accuracy = float(round(accuracy, 6))

    # 반환 딕셔너리 구성
    # "loss"에 평균 손실 저장
    # "accuracy"에 정확도 저장
    # 두 키를 가진 딕셔너리 반환
    return {"loss": mean_loss, "accuracy": accuracy}
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
    # raise NotImplementedError

    # 손실 함수와 옵티마이저 준비
    # nn.BCEWithLogitsLoss()로 loss_fn 생성
    loss_fn = nn.BCEWithLogitsLoss()
    # model.parameters()와 전달받은 lr로 torch.optim.SGD 생성
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 학습 기록 준비
    # epoch별 학습 손실을 저장할 train_losses 리스트를 빈 리스트로 초기화
    train_losses = []
    # epoch별 검증 손실을 저장할 val_losses 리스트를 빈 리스트로 초기화
    val_losses = []
    # epoch별 검증 정확도를 저장할 val_accuracies 리스트를 빈 리스트로 초기화
    val_accuracies = []

    # 최적 epoch 추적 준비
    # best_val_loss를 양의 무한대로 초기화
    best_val_loss = float("inf")
    # best_epoch를 0으로 초기화
    best_epoch = 0
    # 최적 가중치를 저장할 best_weights를 None으로 초기화
    best_weights = None
    # 연속으로 개선되지 않은 epoch 수를 0으로 초기화
    epochs_without_improvement = 0

    # epoch를 1부터 max_epochs까지 증가시키며 반복
    # max_epochs와 patience는 함수로 전달받은 값을 사용
    for epoch in range(1, max_epochs + 1):

        # 현재 epoch 학습
        # train_one_epoch에 model, train_loader, loss_fn, optimizer 전달
        # 반환된 학습 손실을 현재 epoch의 train_loss 값으로 받음
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)

        # 현재 epoch 검증
        # evaluate에 model, val_loader, loss_fn 전달
        val = evaluate(model, val_loader, loss_fn)
        # 반환된 딕셔너리에서 검증 손실과 검증 정확도를 꺼냄
        val_loss = val["loss"]
        val_accuracy = val["accuracy"]

        # 현재 epoch의 기록 추가
        # 학습 손실을 train_losses 리스트에 추가
        train_losses.append(train_loss)
        # 검증 손실을 val_losses 리스트에 추가
        val_losses.append(val_loss)
        # 검증 정확도를 val_accuracies 리스트에 추가
        val_accuracies.append(val_accuracy)

        # 현재 검증 손실이 best_val_loss보다 엄격하게 작은지 확인
        # 같은 손실은 개선으로 처리하지 않음
        if val_loss < best_val_loss:

            # 검증 손실이 개선된 경우
            # best_val_loss를 현재 검증 손실로 갱신
            best_val_loss = val_loss
            # best_epoch를 현재 epoch 번호로 갱신
            best_epoch = epoch
            # copy.deepcopy(model.state_dict())로 현재 가중치의 독립적인 복사본 생성
            # 복사한 가중치를 best_weights에 저장
            best_weights = copy.deepcopy(model.state_dict())
            # 연속으로 개선되지 않은 epoch 수를 0으로 초기화
            epochs_without_improvement = 0

        # 검증 손실이 개선되지 않은 경우
        else:
            # 연속으로 개선되지 않은 epoch 수를 1 증가
            epochs_without_improvement += 1
            # 해당 횟수가 patience 이상이면 epoch 반복 종료
            if epochs_without_improvement >= patience:
                break

    # 반복 종료 후 최적 가중치 복원
    # model.load_state_dict(best_weights)로 최적 epoch의 가중치 복원
    # 조기 종료 여부와 관계없이 반복문이 끝난 뒤 복원 수행
    model.load_state_dict(best_weights)

    # 실제 실행한 epoch 수 계산
    # epochs_run을 train_losses 리스트의 길이로 계산
    epochs_run = len(train_losses)

    # 반환 딕셔너리 구성
    # "train_loss"에 epoch별 학습 손실 리스트 저장
    # "val_loss"에 epoch별 검증 손실 리스트 저장
    # "val_accuracy"에 epoch별 검증 정확도 리스트 저장
    # "best_epoch"에 최적 epoch 번호 저장
    # "best_val_loss"에 최저 검증 손실 저장
    # "epochs_run"에 실제 실행한 epoch 수 저장
    # 완성된 딕셔너리 반환
    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_accuracy": val_accuracies,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_run": epochs_run,
    }

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
    # raise NotImplementedError

    # 탐색 결과 보관 준비
    # 각 조합의 결과를 저장할 grid를 빈 리스트로 초기화
    grid = []
    # 각 조합의 학습된 모델을 저장할 models를 빈 리스트로 초기화
    models = []

    # 하이퍼파라미터 조합 순회
    # 바깥 반복문에서 HIDDEN_SIZES의 hidden_size를 순서대로 꺼냄
    for hidden_size in HIDDEN_SIZES:
        # 안쪽 반복문에서 LEARNING_RATES의 lr을 순서대로 꺼냄
        # 각 hidden_size와 lr 조합마다 아래 작업 수행
        for lr in LEARNING_RATES:

            # 현재 조합의 모델 생성
            # set_seed()로 난수 상태 초기화
            set_seed()
            # data["input_dim"]과 현재 hidden_size를 build_model에 전달
            # 반환된 새 모델을 model에 저장
            model = build_model(data["input_dim"], hidden_size)

            # 현재 조합의 모델 학습
            # train_model에 model, data["train_loader"], data["val_loader"], lr 전달
            # 반환된 학습 기록을 history에 저장
            # train_model이 model을 최적 epoch의 가중치로 복원한 상태임
            history = train_model(
                model, data["train_loader"], data["val_loader"], lr)

            # 최적 epoch의 지표 추출
            # history["best_epoch"]에서 최적 epoch 번호 꺼냄
            best_epoch = history["best_epoch"]
            # 최적 epoch 번호에서 1을 빼서 리스트 접근용 인덱스 계산
            best_epoch_index = best_epoch - 1
            # history["best_val_loss"]에서 최저 검증 손실 꺼냄
            val_loss = history["best_val_loss"]
            # history["val_accuracy"]에서 최적 epoch 인덱스의 정확도 꺼냄
            val_accuracy = history["val_accuracy"][best_epoch_index]
            # history["epochs_run"]에서 실제 실행한 epoch 수 꺼냄
            epochs_run = history["epochs_run"]

            # 현재 조합의 결과 딕셔너리 구성
            # "hidden_size"에 현재 hidden_size를 Python int로 저장
            # "lr"에 현재 lr을 Python float로 저장
            # "val_loss"에 최저 검증 손실 저장
            # "val_accuracy"에 최적 epoch의 검증 정확도 저장
            # "best_epoch"에 최적 epoch 번호 저장
            # "epochs_run"에 실제 실행한 epoch 수 저장
            record = {
                "hidden_size": int(hidden_size),
                "lr": float(lr),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "best_epoch": best_epoch,
                "epochs_run": epochs_run,
            }

            # 현재 조합의 결과와 모델 보관
            # 결과 딕셔너리를 grid에 추가
            grid.append(record)
            # 학습된 model을 models에 추가
            # grid와 models의 같은 인덱스가 같은 조합을 가리키도록 유지
            models.append(model)

    # 모든 조합의 탐색이 끝난 뒤 최적 조합 선택
    # grid에서 "val_loss"가 가장 작은 항목의 인덱스 구함
    # 최저 손실이 같은 항목이 여러 개면 먼저 등장한 항목 선택
    best_index = min(range(len(grid)), key=lambda i: grid[i]["val_loss"])
    # 선택한 인덱스로 grid에서 최적 결과 꺼냄
    best_record = grid[best_index]
    # 같은 인덱스로 models에서 최적 모델 꺼냄
    best_model = models[best_index]

    # 선택된 모델의 테스트 평가
    # nn.BCEWithLogitsLoss()로 테스트 평가용 손실 함수 생성
    loss_fn = nn.BCEWithLogitsLoss()
    # evaluate에 최적 모델, data["test_loader"], 손실 함수 전달
    # 테스트 평가를 이 위치에서 정확히 한 번 수행
    # 반환된 테스트 손실과 정확도를 test 결과로 보관
    test = evaluate(best_model, data["test_loader"], loss_fn)

    # best_config 딕셔너리 구성
    # "hidden_size"에 선택된 조합의 hidden_size 저장
    # "lr"에 선택된 조합의 lr 저장
    best_config = {
        "hidden_size": best_record["hidden_size"],
        "lr": best_record["lr"],
    }

    # val 딕셔너리 구성
    # "loss"에 선택된 조합의 최저 검증 손실 저장
    # "accuracy"에 선택된 조합의 최적 epoch 검증 정확도 저장
    val = {
        "loss": best_record["val_loss"],
        "accuracy": best_record["val_accuracy"],
    }

    # 최종 반환 딕셔너리 구성
    # "grid"에 모든 조합의 결과 리스트 저장
    # "best_config"에 선택된 조합의 설정 딕셔너리 저장
    # "val"에 선택된 조합의 검증 지표 딕셔너리 저장
    # "test"에 한 번 평가한 테스트 지표 딕셔너리 저장
    # 위 네 키를 가진 딕셔너리 반환
    return {
        "grid": grid,
        "best_config": best_config,
        "val": val,
        "test": test,
    }

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
