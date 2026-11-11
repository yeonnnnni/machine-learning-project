"""Lab 7 — Implementing a Transformer: attention from scratch.

Machine Learning Project (53744-01), Fall 2026.

Fill in every TODO block. Do NOT rename functions or change their
signatures/return types — automated (public + hidden) tests call them directly.
Run:      python src/lab07.py
Self-check: python -m pytest tests/ -q

You may NOT use torch.nn.MultiheadAttention, torch.nn.TransformerEncoderLayer,
torch.nn.TransformerEncoder, or torch.nn.functional.scaled_dot_product_attention
anywhere in this file. Building them is the lab.
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

# ---- Fill in your information (used in results.json) ----
STUDENT_ID = "20231879"   # TODO: your student id
STUDENT_NAME = "LeeJeongyeon"     # TODO: your name in Korean or roman letters — "홍길동" / "HongGildong"

SEED = 42  # fixed for the whole course — DO NOT CHANGE
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "machine_events.npz"

# ---- Fixed data and model settings. The tests read these; do not change them. ----
PAD_TOKEN = 0        # 0 = padding, 1 = heat alarm, 2 = pressure alarm, 3..11 = routine
VOCAB_SIZE = 12
MAX_LEN = 16
D_MODEL = 32
N_HEADS = 4          # D_MODEL must be divisible by N_HEADS
D_FF = 64

# ---- Fixed training settings. Identical for all three configurations. ----
VAL_FRACTION = 0.25
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 3e-3

# The three configurations compared in Task 4: (use_attention, use_positional).
CONFIGS = {
    "mean_pool": (False, False),      # permutation-invariant baseline
    "encoder_nopos": (True, False),   # attention, but no position information
    "encoder_pos": (True, True),      # the full encoder block
}


def set_seed(seed: int = SEED) -> None:
    """DO NOT MODIFY. Seeds python, numpy and torch in one call."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path: str | Path) -> dict:
    """DO NOT MODIFY. Loads the event logs and splits them train/validation.

    Returns a dict of tensors:
      "x_train"  (n_train, MAX_LEN) int64   token ids, right-padded with 0
      "mask_train" (n_train, MAX_LEN) bool  True on real events, False on padding
      "y_train"  (n_train,) float32         1.0 = heat alarm came first
      "x_val", "mask_val", "y_val"          same, for the held-out quarter
    The split is a fixed permutation with seed 42, so it never changes.
    """
    z = np.load(path)
    tokens = torch.from_numpy(z["tokens"].astype(np.int64))
    lengths = torch.from_numpy(z["lengths"].astype(np.int64))
    labels = torch.from_numpy(z["labels"].astype(np.float32))
    mask = torch.arange(tokens.shape[1])[None, :] < lengths[:, None]

    n = tokens.shape[0]
    n_val = int(round(VAL_FRACTION * n))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return {
        "x_train": tokens[train_idx], "mask_train": mask[train_idx],
        "y_train": labels[train_idx],
        "x_val": tokens[val_idx], "mask_val": mask[val_idx],
        "y_val": labels[val_idx],
    }


# ============= TODO (Task 1): scaled dot-product attention ==================
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                 mask: torch.Tensor | None = None
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """One attention head, written out in four steps.

    Shapes (`...` is any number of leading batch/head dimensions):
      Q    (..., L_q, d_k)
      K    (..., L_k, d_k)
      V    (..., L_k, d_v)
      mask (..., L_q, L_k) bool or broadcastable to it, or None.
        Use a broadcast-safe fill such as scores.masked_fill(~mask, -inf) —
        boolean indexing like scores[~mask] breaks on broadcast masks. You may
        assume every query row keeps at least one True key (no all-masked rows;
        the hidden tests do not probe that case).

    Steps, in this order:
      1. scores = Q @ K^T
      2. divide the scores by sqrt(d_k), where d_k = Q.shape[-1]
      3. if mask is not None, replace every score whose mask entry is False by
         float("-inf")  (mask semantics: True = keep, False = block)
      4. attn = softmax(scores, dim=-1);  output = attn @ V

    Returns:
        (output, attn) with output (..., L_q, d_v) and attn (..., L_q, L_k).

    Masked positions must receive EXACTLY 0.0 attention weight, not a small
    number. Dividing by sqrt(d_k) is not decoration: without it the scores grow
    with d_k, softmax saturates, and the gradients vanish.
    """
    # raise NotImplementedError
    # 1. 모든 query와 key 사이의 유사도 계산
    scores = Q @ K.transpose(-2, -1)

    # 2. query/key 차원의 제곱근으로 점수 크기 조절
    scores = scores / torch.sqrt(torch.tensor(Q.shape[-1], dtype=scores.dtype))

    # 3. mask가 있는 경우 차단할 위치의 점수를 음의 무한대로 변경
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # 4. key 방향으로 softmax 적용 후 value 가중합 계산
    attn = torch.softmax(scores, dim=-1)
    output = attn @ V

    return output, attn

# =========================== END TODO (Task 1) ==============================


# ================= TODO (Task 2): multi-head attention ======================
class MultiHeadAttention(nn.Module):
    """n_heads independent attention heads, computed in one batched call.

    __init__ is written for you. Implement split_heads, merge_heads and forward.
    """

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS) -> None:
        # ---- given; do not change (the tests read these attribute names) ----
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        # ---------------------------------------------------------------------

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, L, d_model) -> (batch, n_heads, L, d_k).

        Reshape the last axis into (n_heads, d_k) FIRST, then move the head axis
        in front of the length axis. A single view/reshape to (batch, n_heads, L,
        d_k) gives the right shape with the wrong contents: it would cut the
        sequence into blocks instead of cutting each token's vector into heads.
        """
        # raise NotImplementedError

        # 1. 입력의 배치 크기와 시퀀스 길이 확인
        batch_size, seq_len, d_model = x.shape

        # 2. 각 토큰의 feature 차원을 head 수와 head 내부 차원으로 분리
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)

        # 3. head 축을 length 축 앞으로 이동 후 반환
        x = x.transpose(1, 2)
        return x


    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_heads, L, d_k) -> (batch, L, d_model). Inverse of split_heads."""
        # raise NotImplementedError

        # 1. 입력의 배치 크기와 시퀀스 길이 확인
        batch_size, n_heads, seq_len, d_k = x.shape

        # 2. head 축을 length 축 뒤로 이동
        x = x.transpose(1, 2)

        # 3. head와 head 내부 차원을 하나의 feature 차원으로 합친 후 반환
        x = x.contiguous().view(batch_size, seq_len, self.d_model)
        return x


    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Self-attention over x.

        Args:
            x:    (batch, L, d_model)
            mask: None, or a bool tensor broadcastable to
                  (batch, n_heads, L, L). The models in this lab pass
                  (batch, 1, 1, L): True on real events, False on padding.

        Steps:
          1. project x with w_q, w_k, w_v, then split_heads on each
          2. one call to scaled_dot_product_attention on the split tensors
          3. merge_heads on the context, then project with w_o

        Returns:
            (output, attn) with output (batch, L, d_model)
            and attn (batch, n_heads, L, L).
        """
        # raise NotImplementedError

        # 1. 입력을 Q, K, V로 각각 투영한 후 head로 분리
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))

        # 2. 마스크를 전달하여 모든 head의 attention을 한 번에 계산
        context, attn = scaled_dot_product_attention(q, k, v, mask)

        # 3. attention 출력의 head 병합
        output = self.merge_heads(context)

        # 4. 병합된 결과에 출력 투영 적용
        output = self.w_o(output)

        return output, attn

# =========================== END TODO (Task 2) ==============================


# ========= TODO (Task 3): positional encoding and the encoder block =========
def positional_encoding(max_len: int = MAX_LEN, d_model: int = D_MODEL) -> torch.Tensor:
    """Sinusoidal positional encoding, exactly as in Vaswani et al. (2017).

    For position pos (0-based) and index i = 0, 1, ..., d_model/2 - 1:
        PE[pos, 2*i]     = sin(pos / 10000 ** (2*i / d_model))
        PE[pos, 2*i + 1] = cos(pos / 10000 ** (2*i / d_model))

    Returns:
        float32 tensor of shape (1, max_len, d_model). The leading 1 is there so
        it broadcasts over the batch when you add it to the embeddings.
        d_model is assumed even.
    """
    # raise NotImplementedError

    # 1. 위치 번호를 float32 세로 벡터로 생성
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

    # 2. sin/cos 쌍의 인덱스를 float32 벡터로 생성
    i = torch.arange(d_model // 2, dtype=torch.float32)

    # 3. 각 sin/cos 쌍에 사용할 분모 계산
    denom = 10000 ** (2 * i / d_model)

    # 4. 위치별·차원별 각도 계산
    angle = pos / denom

    # 5. float32 행렬의 짝수 열에 sin, 홀수 열에 cos 값 저장
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe.unsqueeze(0)

class EncoderBlock(nn.Module):
    """One transformer encoder block, post-norm ordering.

    __init__ is written for you. Implement forward.
    """

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 d_ff: int = D_FF) -> None:
        # ---- given; do not change (the tests read these attribute names) ----
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # ---------------------------------------------------------------------

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention, residual, norm, feed-forward, residual, norm.

        This lab fixes the POST-NORM order. The body is exactly these three
        lines plus the return, in this order (h is (batch, L, d_model), attn
        is the weight tensor):

            h, attn = self.attn(x, mask)
            h = self.norm1(x + h)
            h = self.norm2(h + self.ff(h))
            return h, attn

        Each residual adds a SUB-LAYER'S OWN INPUT to that sub-layer's output:
        x for the attention, h for the feed-forward. Normalising before the
        sub-layer instead of after the addition, or dropping a residual, is a
        different architecture, and the tests compare against this one.

        Returns:
            (output, attn) with output (batch, L, d_model)
            and attn (batch, n_heads, L, L).
        """
        # raise NotImplementedError

        # 1. 입력에 self-attention 적용
        h, attn = self.attn(x, mask)

        # 2. attention 입력과 출력을 더한 후 첫 번째 LayerNorm 적용
        h = self.norm1(x + h)

        # 3. FFN 입력과 출력을 더한 후 두 번째 LayerNorm 적용
        h = self.norm2(h + self.ff(h))

        # 4. encoder 출력과 attention 가중치 반환
        return h, attn


# =========================== END TODO (Task 3) ==============================


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """DO NOT MODIFY. Mean over real tokens only; padding contributes nothing."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class SeqClassifier(nn.Module):
    """DO NOT MODIFY. One class, the three configurations of Task 4.

    embedding -> (+ positional encoding) -> (encoder block) -> masked mean -> head.
    Only the two flags change between configurations; the embedding, the
    pooling, the head, and every training setting are identical.
    """

    def __init__(self, use_attention: bool, use_positional: bool) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.use_positional = use_positional
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_TOKEN)
        self.encoder = EncoderBlock(D_MODEL, N_HEADS, D_FF) if use_attention else None
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, D_FF), nn.ReLU(), nn.Linear(D_FF, 1))
        if use_positional:
            self.register_buffer("pe", positional_encoding(MAX_LEN, D_MODEL),
                                 persistent=False)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        if self.use_positional:
            h = h + self.pe[:, :tokens.shape[1]]
        if self.encoder is not None:
            h, _ = self.encoder(h, mask[:, None, None, :])
        return self.head(masked_mean(h, mask)).squeeze(-1)


def train_model(model: nn.Module, data: dict, seed: int = SEED) -> dict:
    """DO NOT MODIFY. Trains with Adam and reports validation accuracy and loss.

    Batch order comes from a torch.Generator seeded here, so two runs of the
    same configuration produce identical numbers.
    """
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    n = data["y_train"].shape[0]

    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(data["x_train"][idx], data["mask_train"][idx])
            loss = loss_fn(logits, data["y_train"][idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(data["x_val"], data["mask_val"])
        val_loss = loss_fn(logits, data["y_val"]).item()
        correct = ((logits > 0).float() == data["y_val"]).float().mean().item()
    return {"accuracy": round(correct, 4), "loss": round(val_loss, 4)}


# ============ TODO (Task 4): controlled comparison of three models ==========
def run_experiment(data: dict) -> dict:
    """Train the three configurations of CONFIGS under identical conditions.

    Controlled experiment = one factor varies, everything else is fixed. Here
    the factors are the two flags in CONFIGS; the data, the split, the seed, the
    optimiser, the learning rate, the batch size, and the number of epochs are
    the same for all three.

    Steps, for each (name, (use_attention, use_positional)) in CONFIGS.items():
      1. set_seed(SEED)  — call it again before EVERY model, so all three start
         from the same random state and the comparison is reproducible
      2. model = SeqClassifier(use_attention=..., use_positional=...)
      3. metrics = train_model(model, data)
      4. store metrics under name

    Returns:
        {"mean_pool": {"accuracy": float, "loss": float},
         "encoder_nopos": {...}, "encoder_pos": {...}}
        — one entry per key of CONFIGS, in the same order.
    """
    # raise NotImplementedError

    # 1. 모든 설정의 결과를 저장할 빈 딕셔너리 생성
    results = {}

    # 2. CONFIGS에 정의된 순서대로 설정 순회
    for name, (use_attention, use_positional) in CONFIGS.items():
        # 3. 각 모델 생성 직전에 시드 초기화
        set_seed(SEED)
        
        # 4. 현재 설정의 attention 및 positional encoding 사용 여부로 새 모델 생성
        model = SeqClassifier(use_attention=use_attention, use_positional=use_positional)

        # 5. 주어진 학습 함수로 모델 학습 및 평가
        metrics = train_model(model, data)
        # 6. 설정 이름을 키로 사용하여 평가 결과 저장
        results[name] = metrics

    return results

# =========================== END TODO (Task 4) ==============================


def main() -> dict:
    """DO NOT MODIFY. Writes results.json with the course-wide schema."""
    set_seed()
    t0 = time.time()
    data = load_data(DATA_PATH)
    experiment = run_experiment(data)
    metrics = {
        "n_train": int(data["y_train"].shape[0]),
        "n_val": int(data["y_val"].shape[0]),
        "max_len": MAX_LEN,
        "vocab_size": VOCAB_SIZE,
        "positive_rate": float(round(data["y_val"].mean().item(), 4)),
        **{f"{m}_{k}": v for m, d in experiment.items() for k, v in d.items()},
    }
    metrics["pe_gap"] = round(
        metrics["encoder_pos_accuracy"] - metrics["encoder_nopos_accuracy"], 4)
    results = {
        "lab": "lab07",
        "student_id": STUDENT_ID,
        "name": STUDENT_NAME,
        "seed": SEED,
        "metrics": metrics,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    out = Path(__file__).resolve().parent.parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
