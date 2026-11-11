"""Lab 7 public tests. DO NOT MODIFY.

Run: python -m pytest tests/ -q
Grading also runs hidden tests (different data, edge cases, determinism, and a
comparison against PyTorch's own attention modules).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lab07  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "machine_events.npz"


def test_attention_matches_the_definition():
    """Shapes, rows that sum to 1, and the 1/sqrt(d_k) factor."""
    torch.manual_seed(0)
    d_k = 4
    Q, K, V = torch.randn(2, 5, d_k), torch.randn(2, 7, d_k), torch.randn(2, 7, 3)
    out, attn = lab07.scaled_dot_product_attention(Q, K, V)

    assert out.shape == (2, 5, 3)
    assert attn.shape == (2, 5, 7)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 5), atol=1e-5)

    # independent reference, computed in numpy, including the scaling
    q, k, v = Q.numpy(), K.numpy(), V.numpy()
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_k)
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    ref_attn = e / e.sum(axis=-1, keepdims=True)
    assert np.allclose(attn.numpy(), ref_attn, atol=1e-5)   # fails without sqrt(d_k)
    assert np.allclose(out.numpy(), ref_attn @ v, atol=1e-5)


def test_attention_mask_gives_exactly_zero_weight():
    """Blocked keys must receive weight 0.0, so their values cannot leak in."""
    torch.manual_seed(1)
    Q, K, V = torch.randn(1, 3, 8), torch.randn(1, 6, 8), torch.randn(1, 6, 5)
    mask = torch.tensor([[True, True, False, True, False, False]])[:, None, :]
    out, attn = lab07.scaled_dot_product_attention(Q, K, V, mask)

    blocked = ~mask.expand(1, 3, 6)
    assert (attn[blocked] == 0).all()
    assert torch.allclose(attn.sum(-1), torch.ones(1, 3), atol=1e-5)

    V2 = V.clone()
    V2[0, 2] += 100.0
    V2[0, 4] -= 50.0
    out2, _ = lab07.scaled_dot_product_attention(Q, K, V2, mask)
    assert torch.allclose(out, out2, atol=1e-10)     # blocked values changed nothing


def test_multihead_splits_the_feature_axis_not_the_sequence():
    """The classic bug: a bare view() gives the right shape and wrong contents."""
    mha = lab07.MultiHeadAttention(lab07.D_MODEL, lab07.N_HEADS)
    x = torch.arange(2 * 5 * lab07.D_MODEL, dtype=torch.float32).reshape(2, 5, lab07.D_MODEL)

    heads = mha.split_heads(x)
    assert heads.shape == (2, lab07.N_HEADS, 5, mha.d_k)
    for h in range(lab07.N_HEADS):
        expected = x[:, :, h * mha.d_k:(h + 1) * mha.d_k]
        assert torch.equal(heads[:, h], expected)
    assert torch.equal(mha.merge_heads(heads), x)

    out, attn = mha(x)
    assert out.shape == (2, 5, lab07.D_MODEL)
    assert attn.shape == (2, lab07.N_HEADS, 5, 5)
    assert torch.allclose(attn.sum(-1), torch.ones(2, lab07.N_HEADS, 5), atol=1e-5)


def test_positional_encoding_values():
    pe = lab07.positional_encoding(lab07.MAX_LEN, lab07.D_MODEL)
    assert pe.shape == (1, lab07.MAX_LEN, lab07.D_MODEL)
    assert pe.dtype == torch.float32

    half = lab07.D_MODEL // 2
    assert torch.allclose(pe[0, 0, 0::2], torch.zeros(half), atol=1e-6)   # sin(0)
    assert torch.allclose(pe[0, 0, 1::2], torch.ones(half), atol=1e-6)    # cos(0)
    assert pe[0, 3, 0].item() == pytest.approx(math.sin(3.0), abs=1e-5)
    assert pe[0, 3, 1].item() == pytest.approx(math.cos(3.0), abs=1e-5)
    w = 10000.0 ** (4 / lab07.D_MODEL)
    assert pe[0, 5, 4].item() == pytest.approx(math.sin(5.0 / w), abs=1e-5)
    assert pe[0, 5, 5].item() == pytest.approx(math.cos(5.0 / w), abs=1e-5)

    dist = torch.cdist(pe[0], pe[0]) + torch.eye(lab07.MAX_LEN)
    assert dist.min().item() > 1e-3          # every position gets its own vector


def test_encoder_block_residual_and_norm_order():
    torch.manual_seed(2)
    block = lab07.EncoderBlock(lab07.D_MODEL, lab07.N_HEADS, lab07.D_FF).eval()
    x = torch.randn(3, 6, lab07.D_MODEL)

    with torch.no_grad():
        out, attn = block(x)
        assert out.shape == x.shape
        assert attn.shape == (3, lab07.N_HEADS, 6, 6)

        # rebuilt from the block's own submodules, in the order the spec fixes
        h, _ = block.attn(x)
        h = block.norm1(x + h)
        expected = block.norm2(h + block.ff(h))
        assert torch.allclose(out, expected, atol=1e-6)

    # the block ends with LayerNorm, so every position has mean 0 and std 1
    assert torch.allclose(out.mean(-1), torch.zeros(3, 6), atol=1e-5)
    assert torch.allclose(out.std(-1, unbiased=False), torch.ones(3, 6), atol=1e-3)


def test_experiment_positional_encoding_matters():
    data = lab07.load_data(DATA)
    r = lab07.run_experiment(data)

    assert set(r) == set(lab07.CONFIGS)
    for name, m in r.items():
        assert set(m) == {"accuracy", "loss"}, name

    # the two order-blind configurations cannot beat a coin toss on this task
    assert r["mean_pool"]["accuracy"] < 0.60
    assert r["encoder_nopos"]["accuracy"] < 0.60
    # the full encoder can
    assert r["encoder_pos"]["accuracy"] > 0.90
    assert r["encoder_pos"]["accuracy"] - r["encoder_nopos"]["accuracy"] > 0.30
