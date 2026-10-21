"""Lab 5 public tests. DO NOT MODIFY.

Run: python -m pytest tests/ -q
Grading also runs hidden tests (different data, edge cases, determinism).
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lab05  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "parts_shapes.npz"
WEIGHTS = Path(__file__).resolve().parent.parent / "data" / "backbone_shapes6.pt"


def pair(v):
    """nn layers report kernel_size/stride/padding as either an int or a tuple."""
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


@pytest.fixture(scope="module")
def data():
    return lab05.load_data(DATA)


def test_conv_output_shape():
    assert lab05.conv_output_shape(32, 32, 3, 1, 1) == (32, 32)   # "same" convolution
    assert lab05.conv_output_shape(32, 32, 2, 2, 0) == (16, 16)   # 2x2 pool
    assert lab05.conv_output_shape(16, 16, 2, 2, 0) == (8, 8)
    assert lab05.conv_output_shape(8, 8, 2, 2, 0) == (4, 4)
    assert lab05.conv_output_shape(28, 28, 5, 1, 0) == (24, 24)   # no padding shrinks
    assert lab05.conv_output_shape(32, 32, 3, 2, 0) == (15, 15)   # floor division
    assert lab05.conv_output_shape(32, 20, 3, 1, 1) == (32, 20)   # axes are independent
    out = lab05.conv_output_shape(32, 32, 3, 1, 1)
    assert isinstance(out, tuple) and all(type(v) is int for v in out)


def test_cnn_structure():
    model = lab05.build_cnn(lab05.N_CLASSES)
    assert isinstance(model, nn.Module)
    feats = model.features
    assert isinstance(feats, nn.Sequential) and len(feats) == 9

    for i, (cin, cout) in zip((0, 3, 6), ((1, 8), (8, 16), (16, 32))):
        conv = feats[i]
        assert isinstance(conv, nn.Conv2d), f"features[{i}] must be a Conv2d"
        assert (conv.in_channels, conv.out_channels) == (cin, cout)
        assert pair(conv.kernel_size) == (3, 3)
        assert pair(conv.stride) == (1, 1)
        assert pair(conv.padding) == (1, 1)
    for i in (1, 4, 7):
        assert isinstance(feats[i], nn.ReLU), f"features[{i}] must be a ReLU"
    for i in (2, 5, 8):
        pool = feats[i]
        assert isinstance(pool, nn.MaxPool2d), f"features[{i}] must be a MaxPool2d"
        assert pair(pool.kernel_size) == (2, 2) and pair(pool.stride) == (2, 2)

    assert isinstance(model.head, nn.Linear)
    assert model.head.in_features == lab05.FEATURE_DIM == 512
    assert model.head.out_features == lab05.N_CLASSES


def test_cnn_forward_shapes():
    model = lab05.build_cnn(lab05.N_CLASSES)
    x = torch.zeros(5, 1, lab05.IMG_SIZE, lab05.IMG_SIZE)
    with torch.no_grad():
        assert tuple(model.features(x).shape) == (5, 32, 4, 4)   # 32 -> 16 -> 8 -> 4
        out = model(x)
    assert tuple(out.shape) == (5, lab05.N_CLASSES)
    assert lab05.count_trainable(model) == 7940                  # every weight trainable


def test_freeze_backbone_and_optimizer():
    model = lab05.freeze_backbone(lab05.build_cnn(lab05.N_CLASSES))
    assert not any(p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    assert lab05.count_trainable(model) == 512 * 4 + 4           # head only

    opt = lab05.make_optimizer(model)
    in_opt = {id(p) for group in opt.param_groups for p in group["params"]}
    assert in_opt == {id(p) for p in model.head.parameters()}


def test_transfer_model_uses_the_shipped_weights():
    saved = torch.load(WEIGHTS, map_location="cpu")
    model = lab05.build_transfer_model(lab05.N_CLASSES, WEIGHTS)

    got = model.features.state_dict()
    for key, tensor in saved.items():
        if key.startswith("features."):
            assert torch.equal(got[key[len("features."):]], tensor), f"{key} was not loaded"
    assert not any(p.requires_grad for p in model.features.parameters())
    assert model.head.out_features == lab05.N_CLASSES            # not the source task's 6
    assert lab05.count_trainable(model) == 2052


def test_experiment_beats_baseline(data):
    X, y = data
    r = lab05.run_experiment(X, y)
    assert set(r) == {"baseline", "scratch", "transfer"}
    assert set(r["scratch"]) == {"accuracy", "trainable_params"}
    # 4 balanced classes: guessing the most frequent one lands near 0.25
    assert r["baseline"]["accuracy"] == pytest.approx(0.25, abs=0.05)
    assert r["baseline"]["trainable_params"] == 0
    # both models learn, and the frozen backbone wins on this small label budget
    assert r["scratch"]["accuracy"] > 0.6
    assert r["transfer"]["accuracy"] > r["scratch"]["accuracy"] + 0.03
    # and it wins while training far fewer weights
    assert r["transfer"]["trainable_params"] < r["scratch"]["trainable_params"]
