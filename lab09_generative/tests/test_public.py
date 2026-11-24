"""Lab 9 public tests. DO NOT MODIFY.

Run: python -m pytest tests/ -q
Grading also runs hidden tests (different data, edge cases, determinism).
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lab09  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "lab09_ring.npz"


@pytest.fixture(scope="module")
def data():
    return lab09.load_data(DATA)


def seeded_gan(seed: int):
    torch.manual_seed(seed)
    G, D = lab09.GenNet(), lab09.DiscNet()
    optG = torch.optim.Adam(G.parameters(), lr=lab09.LR_GAN)
    optD = torch.optim.Adam(D.parameters(), lr=lab09.LR_GAN)
    return G, D, optG, optD


def test_gan_step_updates_both_and_is_rng_driven(data):
    lab09.set_seed(0)
    real = torch.as_tensor(data["samples"][:64], dtype=torch.float32)
    G, D, optG, optD = seeded_gan(7)
    g_before = copy.deepcopy(G.state_dict())
    d_before = copy.deepcopy(D.state_dict())
    out = lab09.gan_step(G, D, optG, optD, real, torch.Generator().manual_seed(3))
    assert isinstance(out, tuple) and len(out) == 2
    d_loss, g_loss = out
    assert isinstance(d_loss, float) and isinstance(g_loss, float)
    assert d_loss > 0.0 and g_loss > 0.0
    # one alternating step must move BOTH networks
    assert any(not torch.equal(g_before[k], v) for k, v in G.state_dict().items())
    assert any(not torch.equal(d_before[k], v) for k, v in D.state_dict().items())
    # the injected generator is the only randomness: rebuild everything with
    # the same seeds and the two losses must reproduce exactly
    G2, D2, optG2, optD2 = seeded_gan(7)
    out2 = lab09.gan_step(G2, D2, optG2, optD2, real, torch.Generator().manual_seed(3))
    assert out == out2


def test_schedule_matches_definition():
    beta, alpha, alphabar = lab09.make_schedule(lab09.T_DIFF, lab09.BETA_MIN,
                                                lab09.BETA_MAX)
    # length exactly T — no prepended 1, no dropped step
    assert beta.shape == alpha.shape == alphabar.shape == (lab09.T_DIFF,)
    ref_beta = torch.linspace(lab09.BETA_MIN, lab09.BETA_MAX, lab09.T_DIFF)
    assert torch.allclose(beta, ref_beta, atol=1e-8)
    assert torch.allclose(alpha, 1.0 - ref_beta, atol=1e-8)
    assert torch.allclose(alphabar, torch.cumprod(1.0 - ref_beta, dim=0), atol=1e-7)
    # alphabar[0] is alpha[0] = 1 - beta_min, NOT 1.0
    assert abs(alphabar[0].item() - (1.0 - lab09.BETA_MIN)) < 5e-7
    # cumulative products strictly decrease
    assert (alphabar[1:] < alphabar[:-1]).all()


def test_q_sample_hand_values():
    # the worked example of the concept note: T=4, beta = 0.1, 0.2, 0.3, 0.4
    beta, alpha, alphabar = lab09.make_schedule(4, 0.1, 0.4)
    assert torch.allclose(alphabar,
                          torch.tensor([0.9, 0.72, 0.504, 0.3024]), atol=1e-6)
    x0 = torch.tensor([[2.0, 0.0]])
    noise = torch.tensor([[0.5, -1.0]])
    xt = lab09.q_sample(x0, torch.tensor([2]), noise, alphabar)
    # sqrt(0.504)*2 + sqrt(0.496)*0.5 = 1.7720..., sqrt(0.496)*(-1) = -0.7042...
    assert xt.shape == (1, 2)
    assert torch.allclose(xt, torch.tensor([[1.772026, -0.704273]]), atol=1e-4)
    # per-sample gather: two samples with different t get different alphabar
    x0b = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    nb = torch.zeros(2, 2)
    xtb = lab09.q_sample(x0b, torch.tensor([0, 3]), nb, alphabar)
    assert torch.allclose(xtb[0], torch.full((2,), 0.9 ** 0.5), atol=1e-6)
    assert torch.allclose(xtb[1], torch.full((2,), 0.3024 ** 0.5), atol=1e-6)


class EchoNoise(nn.Module):
    """Stub denoiser that returns a fixed tensor, whatever its input."""

    def __init__(self, out):
        super().__init__()
        self.out = out

    def forward(self, x, t):
        return self.out


def test_ddpm_train_loss_exact_and_differentiable():
    x0 = torch.tensor([[1.0, -1.0], [0.5, 2.0], [0.0, 0.0]])
    t = torch.tensor([0, 50, 99])
    noise = torch.tensor([[0.3, -0.2], [1.0, 0.0], [-0.5, 0.5]])
    # a model that predicts the mixed-in noise exactly has loss 0
    loss = lab09.ddpm_train_loss(EchoNoise(noise.clone()), x0, t, noise)
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0
    assert abs(loss.item()) < 1e-12
    # off by a constant 1 everywhere -> MSE exactly 1
    loss1 = lab09.ddpm_train_loss(EchoNoise(noise + 1.0), x0, t, noise)
    assert abs(loss1.item() - 1.0) < 1e-6
    # with a real network the loss must carry gradients back to its weights
    torch.manual_seed(0)
    model = lab09.DenoiseNet()
    loss2 = lab09.ddpm_train_loss(model, x0, t, noise)
    assert loss2.requires_grad


class ScaledEps(nn.Module):
    """Stub denoiser eps_hat = 0.1 * x_t: exercises every term of the
    reverse update (both the alphabar inside the mean and sigma_t)."""

    def forward(self, x, t):
        return 0.1 * x


def test_ddpm_sample_matches_reference_chain():
    n = 16
    out = lab09.ddpm_sample(ScaledEps(), n, torch.Generator().manual_seed(11))
    assert out.shape == (n, 2)
    assert torch.isfinite(out).all()
    # reference chain, written out independently of the implementation
    beta = torch.linspace(lab09.BETA_MIN, lab09.BETA_MAX, lab09.T_DIFF)
    alpha = 1.0 - beta
    alphabar = torch.cumprod(alpha, dim=0)
    g = torch.Generator().manual_seed(11)
    x = torch.randn(n, 2, generator=g)
    for t in range(lab09.T_DIFF - 1, -1, -1):
        eps = 0.1 * x
        x = (x - beta[t] / torch.sqrt(1.0 - alphabar[t]) * eps) / torch.sqrt(alpha[t])
        if t > 0:                       # NO noise at t == 0
            x = x + torch.sqrt(beta[t]) * torch.randn(n, 2, generator=g)
    assert torch.allclose(out, x, atol=1e-5)
    # rng is the only randomness
    out2 = lab09.ddpm_sample(ScaledEps(), n, torch.Generator().manual_seed(11))
    assert torch.equal(out, out2)


def test_run_experiment_schema_and_determinism(data):
    art = {}
    m1 = lab09.run_experiment(data, torch.Generator().manual_seed(lab09.SEED),
                              gan_steps=200, ddpm_steps=200, n_gen=200, out=art)
    m2 = lab09.run_experiment(data, torch.Generator().manual_seed(lab09.SEED),
                              gan_steps=200, ddpm_steps=200, n_gen=200)
    keys = {"n_train", "n_gen", "coverage_gan", "coverage_ddpm",
            "hq_gan", "hq_ddpm", "gan_mode_shares", "ddpm_mode_shares",
            "d_loss_final", "g_loss_final", "ddpm_loss_final",
            "gan_d_curve", "gan_g_curve", "ddpm_curve",
            "gan_sample_seconds", "ddpm_sample_seconds"}
    assert set(m1) == keys
    # wall-clock keys legitimately vary; everything else must reproduce exactly
    for k in keys:
        if not k.endswith("_seconds"):
            assert m1[k] == m2[k], f"{k} not deterministic"
    assert m1["n_train"] == len(data["samples"]) and m1["n_gen"] == 200
    assert isinstance(m1["coverage_gan"], int) and 0 <= m1["coverage_gan"] <= 8
    assert isinstance(m1["coverage_ddpm"], int) and 0 <= m1["coverage_ddpm"] <= 8
    for k in ("hq_gan", "hq_ddpm"):
        assert isinstance(m1[k], float) and 0.0 <= m1[k] <= 1.0
    for k in ("gan_mode_shares", "ddpm_mode_shares"):
        assert len(m1[k]) == 8 and abs(sum(m1[k]) - 1.0) < 1e-3
    # LOG_EVERY grid: 200 steps -> 2 recorded points per curve
    assert len(m1["gan_d_curve"]) == len(m1["gan_g_curve"]) == 2
    assert len(m1["ddpm_curve"]) == 2
    # LOG_EVERY divides both budgets, so each curve's last point IS the
    # corresponding final loss (same value, same rounding)
    assert m1["gan_d_curve"][-1] == m1["d_loss_final"]
    assert m1["gan_g_curve"][-1] == m1["g_loss_final"]
    assert m1["ddpm_curve"][-1] == m1["ddpm_loss_final"]
    assert m1["gan_sample_seconds"] >= 0.0 and m1["ddpm_sample_seconds"] >= 0.0
    # structural cost: DDPM sampling passes the network T_DIFF times, the GAN once
    assert m1["ddpm_sample_seconds"] > m1["gan_sample_seconds"]
    # the out dict serves the notebook
    assert art["samples_gan"].shape == (200, 2)
    assert art["samples_ddpm"].shape == (200, 2)

    # mode_stats (provided helper) semantics, including both threshold
    # directions at their exact boundaries -- the metrics above rely on them
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    # 100 samples: 90 at center 0, 3 at center 1 (share exactly 0.03 -> covered,
    # the threshold is >=), 7 near center 2 of which 2 sit at distance exactly
    # HQ_RADIUS (0.25 -> still high quality, the threshold is <=) and 5 at 0.30
    samples = np.concatenate([
        np.tile(centers[0], (90, 1)),
        np.tile(centers[1], (3, 1)),
        np.tile(centers[2] + [0.25, 0.0], (2, 1)),
        np.tile(centers[2] + [0.30, 0.0], (5, 1)),
    ])
    cov, hq, shares = lab09.mode_stats(samples, centers)
    assert cov == 3
    assert hq == pytest.approx(0.95)                 # 95 of 100 within 0.25
    assert shares == [0.9, 0.03, 0.07]
    # dropping below the share threshold uncovers the mode
    cov2, _, shares2 = lab09.mode_stats(samples[:95], centers)  # center-2 share 2/95
    assert shares2[2] == pytest.approx(round(2 / 95, 4))
    assert cov2 == 2

