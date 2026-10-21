"""Lab 5 — CNNs and transfer learning: shape arithmetic, a small CNN, a frozen backbone.

Machine Learning Project (53744-01), Fall 2026.

Fill in every TODO block. Do NOT rename functions or change their
signatures/return types — automated (public + hidden) tests call them directly.
Run:      python src/lab05.py
Self-check: python -m pytest tests/ -q
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# ---- Fill in your information (used in results.json) ----
STUDENT_ID = "20231879"   # TODO: your student id
STUDENT_NAME = "LeeJeongyeon"     # TODO: your name in Korean or roman letters — "홍길동" / "HongGildong"

SEED = 42  # fixed for the whole course — DO NOT CHANGE
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_PATH = DATA_DIR / "parts_shapes.npz"
WEIGHTS_PATH = DATA_DIR / "backbone_shapes6.pt"

IMG_SIZE = 32          # images are 1 x 32 x 32, grayscale, values in [0, 1]
N_CLASSES = 4          # diamond, ellipse, ell, tee
FEATURE_DIM = 512      # 32 channels x 4 x 4 after the three conv/pool stages
N_LABELLED = 240       # labelled images available for training (the rest is validation)
EPOCHS = 8             # identical for every model in the comparison
BATCH_SIZE = 32
LR = 1e-3


def set_seed(seed: int = SEED) -> None:
    """DO NOT MODIFY. Seeds numpy and torch so weight init and shuffling repeat."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path: str | Path = DATA_PATH) -> tuple[torch.Tensor, torch.Tensor]:
    """DO NOT MODIFY. Returns (X, y).

    X: float32 tensor of shape (N, 1, 32, 32), pixel values divided by 255.
    y: int64 tensor of shape (N,), class indices 0..3.

    Dividing by 255 is a fixed constant, not a statistic learned from the data,
    so it is safe to apply before the split.
    """
    d = np.load(path, allow_pickle=False)
    X = torch.from_numpy(d["images"]).float().div(255.0).unsqueeze(1)
    y = torch.from_numpy(d["labels"]).long()
    return X, y


def split_data(X: torch.Tensor, y: torch.Tensor, n_train: int = N_LABELLED,
               seed: int = SEED) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """DO NOT MODIFY. Stratified split into (X_train, X_val, y_train, y_val).

    The training set holds exactly n_train images. Everything else is validation.
    The training set is small on purpose: that is the situation transfer learning
    is meant for.
    """
    idx = np.arange(len(y))
    tr, val = train_test_split(idx, train_size=n_train, random_state=seed,
                               shuffle=True, stratify=y.numpy())
    return X[tr], X[val], y[tr], y[val]


def make_optimizer(model: nn.Module, lr: float = LR) -> torch.optim.Optimizer:
    """DO NOT MODIFY. Adam over the parameters that still require gradients.

    A frozen parameter never reaches the optimizer, so freezing is what decides
    which weights can move.
    """
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)


def train_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
                lr: float = LR, seed: int = SEED) -> nn.Module:
    """DO NOT MODIFY. Mini-batch training with Adam and cross-entropy.

    You wrote this loop yourself in Lab 4-2. Here it is given so that every model in
    the comparison gets exactly the same training budget.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = make_optimizer(model, lr)
    loss_fn = nn.CrossEntropyLoss()
    n = len(y_train)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss_fn(model(X_train[batch]), y_train[batch]).backward()
            optimizer.step()
    return model


def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """DO NOT MODIFY. Returns int64 predicted class indices, shape (N,)."""
    model.eval()
    with torch.no_grad():
        return model(X).argmax(dim=1)


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """DO NOT MODIFY. Plain Python float rounded to 4 decimals."""
    return float(round((y_true == y_pred).float().mean().item(), 4))


def count_trainable(model: nn.Module) -> int:
    """DO NOT MODIFY. Number of scalar weights the optimizer is allowed to update."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ==================== TODO (Task 1): output-shape arithmetic ==================
def conv_output_shape(h: int, w: int, kernel_size: int, stride: int = 1,
                      padding: int = 0) -> tuple[int, int]:
    """Spatial size of the feature map a conv or pool layer produces.

    For one axis, with input length L:

        out = floor((L + 2*padding - kernel_size) / stride) + 1

    The same formula covers nn.Conv2d and nn.MaxPool2d with default dilation.
    Height and width are computed independently, so a non-square input stays
    non-square.

    Compute this arithmetically. Do NOT build a layer and read the shape off a
    dummy tensor: a hidden test calls this with inputs far too large to allocate.

    Returns:
        (out_h, out_w) as a tuple of two plain Python ints.
    """
    # raise NotImplementedError

    # 출력 높이 계산
    out_h = (h + 2 * padding - kernel_size) // stride + 1
    # 출력 너비 계산
    out_w = (w + 2 * padding - kernel_size) // stride + 1

    return out_h, out_w
# ============================ END TODO (Task 1) ==============================


# ================== TODO (Task 2): a small CNN from scratch ==================
class SmallCNN(nn.Module):
    """Conv, ReLU, pool three times, then flatten and one linear layer.

    Two attribute names are fixed, because the tests and Task 3 use them:

      self.features : nn.Sequential, in exactly this order
          0  nn.Conv2d(in_channels=1,  out_channels=8,  kernel_size=3, stride=1, padding=1)
          1  nn.ReLU()
          2  nn.MaxPool2d(kernel_size=2, stride=2)
          3  nn.Conv2d(in_channels=8,  out_channels=16, kernel_size=3, stride=1, padding=1)
          4  nn.ReLU()
          5  nn.MaxPool2d(kernel_size=2, stride=2)
          6  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
          7  nn.ReLU()
          8  nn.MaxPool2d(kernel_size=2, stride=2)

      self.head : nn.Linear(FEATURE_DIM, n_classes)

    Run the numbers through conv_output_shape before you write this. A 32x32
    input comes out of self.features as (N, 32, 4, 4), which flattens to
    FEATURE_DIM = 512. Getting that number wrong is the most common CNN bug and
    it shows up as a shape error in the linear layer.

    forward(x): x of shape (N, 1, 32, 32) goes through self.features, then
    torch.flatten(..., start_dim=1), then self.head. Return the raw logits of
    shape (N, n_classes). No softmax: nn.CrossEntropyLoss expects logits.
    """

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        # raise NotImplementedError
        # 특징 추출부를 지정된 9개 레이어로 생성
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 분류용 head 생성
        self.head = nn.Linear(FEATURE_DIM, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # raise NotImplementedError
        
        # 이미지에서 특징 추출
        x = self.features(x)

        # 배치 차원은 유지하면서 나머지 차원 펼치기
        x = torch.flatten(x, start_dim=1)

        # 분류 점수 계산 후 반환
        logits = self.head(x)
        return logits

def build_cnn(n_classes: int = N_CLASSES) -> SmallCNN:
    """Return an untrained SmallCNN with every parameter trainable."""
    # raise NotImplementedError

    # 전달받은 클래스 수로 학습 전 모델 생성
    model = SmallCNN(n_classes=n_classes)

    return model
# ============================ END TODO (Task 2) ==============================


# ============ TODO (Task 3): transfer from a frozen backbone =================
def freeze_backbone(model: SmallCNN) -> SmallCNN:
    """Set requires_grad = False on every parameter of model.features.

    model.head is left alone and stays trainable. Returns the same model object
    (modified in place), so that `model = freeze_backbone(model)` also works.
    """
    # raise NotImplementedError

    # 특징 추출부의 모든 파라미터를 동결
    for param in model.features.parameters():
        param.requires_grad = False

    return model

def build_transfer_model(n_classes: int = N_CLASSES,
                         weights_path: str | Path = WEIGHTS_PATH) -> SmallCNN:
    """Build a SmallCNN that reuses the shipped pretrained backbone.

    data/backbone_shapes6.pt is the state_dict of a SmallCNN trained on a
    DIFFERENT task: 6 other shape classes, 9000 labelled images. Its keys are
    "features.0.weight", ..., "head.weight", "head.bias".

    Steps:
      1. state = torch.load(weights_path, map_location="cpu")
      2. model = build_cnn(n_classes)
      3. Copy the BACKBONE weights only, into model.features:
             backbone = {k[len("features."):]: v for k, v in state.items()
                         if k.startswith("features.")}
             model.features.load_state_dict(backbone)
         The saved head has shape (6, 512) and does not fit a head for n_classes,
         which is why it is dropped. Keeping the fresh head from build_cnn is what
         "replacing the head" means here.
      4. freeze_backbone(model)

    Returns the model. After this call, count_trainable(model) must be exactly the
    number of parameters in the head, and make_optimizer(model) must see only the
    head parameters.
    """
    # raise NotImplementedError

    # 사전학습 state_dict를 CPU로 읽기
    state = torch.load(weights_path, map_location="cpu")

    # 목표 클래스 수에 맞는 새 모델 생성
    model = build_cnn(n_classes)

    # checkpoint에서 backbone 가중치만 추출
    backbone = {k[len("features."):]: v for k, v in state.items() if k.startswith("features.")}

    # 추출한 가중치를 특징 추출부에 적용
    model.features.load_state_dict(backbone)

    # 특징 추출부 동결
    freeze_backbone(model)

    return model

# ============================ END TODO (Task 3) ==============================


# =============== TODO (Task 4): scratch against transfer =====================
def run_experiment(X: torch.Tensor, y: torch.Tensor) -> dict:
    """Compare a scratch CNN with the frozen-backbone model under IDENTICAL conditions.

    One factor varies (where the backbone weights come from). The split, the seed,
    the number of epochs, the batch size, the learning rate, and the metric are
    all held fixed.

    Steps:
      1. X_train, X_val, y_train, y_val = split_data(X, y)   — call this ONCE and
         reuse the same split for every model.
      2. "baseline": predict the most frequent class of y_train for every
         validation image — build the prediction as a tensor shaped like y_val,
         e.g. torch.full_like(y_val, majority). Store
         {"accuracy": accuracy(y_val, that prediction), "trainable_params": 0}.
      3. For "scratch" and then "transfer", in that order:
             set_seed()                       # identical starting conditions
             model = build_cnn(N_CLASSES)     # scratch
             model = build_transfer_model(N_CLASSES, WEIGHTS_PATH)   # transfer
             train_model(model, X_train, y_train)
             y_pred = predict(model, X_val)
             store {"accuracy": accuracy(y_val, y_pred),
                    "trainable_params": count_trainable(model)}

    Calling set_seed() before building each model is what makes two runs of this
    function return identical numbers. A hidden test checks that.

    Returns:
        {"baseline": {"accuracy": float, "trainable_params": int},
         "scratch":  {"accuracy": float, "trainable_params": int},
         "transfer": {"accuracy": float, "trainable_params": int}}
    """
    # raise NotImplementedError

    # 데이터를 한 번 분할
    X_train, X_val, y_train, y_val = split_data(X, y)

    # 결과를 담을 딕셔너리 생성
    results = {}

    # 훈련 데이터에서 최빈 클래스 찾기
    class_counts = torch.bincount(y_train)
    majority_class = torch.argmax(class_counts)

    # baseline 예측과 결과 저장
    baseline_pred = torch.full_like(y_val, majority_class)
    results["baseline"] = {
        "accuracy": accuracy(y_val, baseline_pred),
        "trainable_params": 0
    }

    # scratch 모델 생성 및 학습
    set_seed()
    scratch_model = build_cnn(N_CLASSES)
    train_model(scratch_model, X_train, y_train)
    scratch_pred = predict(scratch_model, X_val)
    results["scratch"] = {
        "accuracy": accuracy(y_val, scratch_pred),
        "trainable_params": count_trainable(scratch_model)
    }

    # transfer 모델 생성 및 학습
    set_seed()
    transfer_model = build_transfer_model(N_CLASSES, WEIGHTS_PATH)
    train_model(transfer_model, X_train, y_train)
    transfer_pred = predict(transfer_model, X_val)
    results["transfer"] = {
        "accuracy": accuracy(y_val, transfer_pred),
        "trainable_params": count_trainable(transfer_model)
    }

    # transfer 모델 평가 및 결과 저장
    transfer_accuracy = accuracy(y_val, transfer_pred)
    transfer_trainable_params = count_trainable(transfer_model)
    results["transfer"] = {
        "accuracy": transfer_accuracy,
        "trainable_params": transfer_trainable_params
    }

    return results

# ============================ END TODO (Task 4) ==============================


def main() -> dict:
    """DO NOT MODIFY. Writes results.json with the course-wide schema."""
    set_seed()
    t0 = time.time()
    X, y = load_data(DATA_PATH)
    experiment = run_experiment(X, y)
    results = {
        "lab": "lab05",
        "student_id": STUDENT_ID,
        "name": STUDENT_NAME,
        "seed": SEED,
        "metrics": {
            "n_samples": int(len(X)),
            "n_classes": int(N_CLASSES),
            "n_labelled_train": int(N_LABELLED),
            "epochs": int(EPOCHS),
            **{f"{m}_{k}": v for m, d in experiment.items() for k, v in d.items()},
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    out = Path(__file__).resolve().parent.parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
