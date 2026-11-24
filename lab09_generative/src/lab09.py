"""Lab 9 — GANs and Diffusion Models: learning to sample, not to classify.

Machine Learning Project (53744-01), Fall 2026.

Fill in every TODO block. Do NOT rename functions or change their
signatures/return types — automated (public + hidden) tests call them directly.
The three small MLPs (GenNet, DiscNet, DenoiseNet) are PROVIDED and fixed;
your job is the training steps, the diffusion math, and the experiment.
Run:      python src/lab09.py
Self-check: python -m pytest tests/ -q
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Fill in your information (used in results.json) ----
STUDENT_ID = "00000000"   # TODO: your student id, e.g. "20261234"
STUDENT_NAME = "None"     # TODO: your name in Korean or roman letters — "홍길동" / "HongGildong"

SEED = 42  # fixed for the whole course — DO NOT CHANGE
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "lab09_ring.npz"

# ---- Fixed experiment constants — DO NOT CHANGE (tests check them) ----
Z_DIM = 2          # GAN latent dimension
T_DIFF = 100       # diffusion steps, t in {0, ..., T_DIFF-1}
BETA_MIN = 1e-4    # linear beta schedule start (beta_0)
BETA_MAX = 0.2     # linear beta schedule end (beta_{T-1})
BATCH = 128        # minibatch size for BOTH models
GAN_STEPS = 5000   # alternating D/G updates
DDPM_STEPS = 6000  # denoiser updates
LR_GAN = 1e-3      # Adam lr for G and D (default betas)
LR_DDPM = 1e-3     # Adam lr for the denoiser (default betas)
N_GEN = 2000       # samples drawn from each trained model for evaluation
LOG_EVERY = 100    # loss-curve recording interval
HQ_RADIUS = 0.25       # a sample is "high quality" iff dist to nearest mode <= this
MODE_SHARE_MIN = 0.03  # a mode is "covered" iff its sample share >= this


def set_seed(seed: int = SEED) -> None:
    """DO NOT MODIFY. Full determinism on CPU: numpy + torch seeds, single
    thread, and deterministic kernels only."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def load_data(path: str | Path) -> dict:
    """DO NOT MODIFY. Loads the 8-Gaussian ring:
    samples (2000, 2) float32 — the training set, 250 points per mode;
    mode_centers (8, 2) float32 — the true mode centers, FOR EVALUATION ONLY
    (never use them in training — hidden tests catch that);
    radius, sigma — scalars describing the ring."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


class GenNet(nn.Module):
    """DO NOT MODIFY. GAN generator: latent z (B, Z_DIM) -> sample (B, 2)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_DIM, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DiscNet(nn.Module):
    """DO NOT MODIFY. GAN discriminator: sample (B, 2) -> logit (B,).
    Returns RAW logits — pair it with binary_cross_entropy_with_logits."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DenoiseNet(nn.Module):
    """DO NOT MODIFY. Diffusion denoiser: (x_t (B, 2), t (B,) int64) ->
    predicted noise epsilon_hat (B, 2). The timestep enters as five fixed
    Fourier features of t / (T_DIFF - 1)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 5, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s = t.to(torch.float32) / (T_DIFF - 1)
        emb = torch.stack([s,
                           torch.sin(2 * torch.pi * s), torch.cos(2 * torch.pi * s),
                           torch.sin(4 * torch.pi * s), torch.cos(4 * torch.pi * s)],
                          dim=1)
        return self.net(torch.cat([x, emb], dim=1))


def mode_stats(samples: np.ndarray, centers: np.ndarray) -> tuple[int, float, list]:
    """DO NOT MODIFY. Evaluation helper: nearest-mode assignment.

    Args:
        samples: (N, 2) generated (or real) points.
        centers: (K, 2) true mode centers (evaluation only).

    For each sample, find the nearest center (Euclidean). Then:
      shares[k]  = fraction of samples assigned to center k
      coverage   = number of modes with shares[k] >= MODE_SHARE_MIN
      hq_frac    = fraction of samples whose distance to their nearest
                   center is <= HQ_RADIUS

    Returns (coverage: int, hq_frac: float, shares: list of K floats),
    hq_frac and shares rounded to 4 decimals.
    """
    samples = np.asarray(samples, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    d = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    dmin = d.min(axis=1)
    shares = np.bincount(nearest, minlength=len(centers)) / len(samples)
    coverage = int((shares >= MODE_SHARE_MIN).sum())
    hq_frac = float(round((dmin <= HQ_RADIUS).mean(), 4))
    return coverage, hq_frac, [float(round(s, 4)) for s in shares]


# ================== TODO (Task 1): one alternating GAN step ==================
def gan_step(G: nn.Module, D: nn.Module,
             optG: torch.optim.Optimizer, optD: torch.optim.Optimizer,
             real_batch: torch.Tensor, rng_noise: torch.Generator
             ) -> tuple[float, float]:
    """One alternating GAN update: discriminator first, then generator.

    Args:
        G, D: GenNet and DiscNet (any modules with those interfaces).
        optG, optD: optimizers over G's and D's parameters respectively.
        real_batch: (B, 2) float32 real samples.
        rng_noise: an injected torch.Generator — the ONLY source of
            randomness here. Never call torch.randn without it.

    Steps (exactly — the tests reproduce them number for number):
        1. B = real_batch.shape[0]. Zero BOTH optimizers' gradients FIRST:
           optD.zero_grad(); optG.zero_grad().
        2. Discriminator step:
           z_d = torch.randn(B, Z_DIM, generator=rng_noise)
           fake = G(z_d).detach()   # DETACH: the D step must not build
                                    # gradients into G — without detach,
                                    # G's .grad is polluted and step 3
                                    # applies a corrupted update.
           d_loss = F.binary_cross_entropy_with_logits(D(real_batch), ones(B))
                  + F.binary_cross_entropy_with_logits(D(fake), zeros(B))
           (each term is a mean over the batch; real label 1, fake label 0)
           d_loss.backward(); optD.step().
        3. Generator step (NON-SATURATING loss):
           z_g = torch.randn(B, Z_DIM, generator=rng_noise)   # a FRESH draw
           g_loss = F.binary_cross_entropy_with_logits(D(G(z_g)), ones(B))
           i.e. maximize log D(G(z)) — NOT the saturating form
           -log(1 - D(G(z))), which has vanishing gradients when D wins.
           g_loss.backward(); optG.step().
        4. Return (float(d_loss.item()), float(g_loss.item())).
    """
    raise NotImplementedError
# ============================ END TODO (Task 1) ==============================


# ============ TODO (Task 2): forward noising (closed form) ===================
def make_schedule(T: int, beta_min: float, beta_max: float
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The linear beta schedule and its cumulative products.

    Indexing convention (fixed): t runs over {0, 1, ..., T-1}. All three
    returned tensors have length exactly T — do NOT prepend an extra 1.

        beta     = torch.linspace(beta_min, beta_max, T)   # beta[0]=beta_min,
                                                           # beta[T-1]=beta_max
        alpha    = 1 - beta
        alphabar = torch.cumprod(alpha, dim=0)             # alphabar[t] =
                                                           # alpha[0]*...*alpha[t]
    So alphabar[0] == alpha[0] == 1 - beta_min (NOT 1.0).

    Returns:
        (beta, alpha, alphabar) — three float32 tensors of shape (T,).
    """
    raise NotImplementedError


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
             alphabar: torch.Tensor) -> torch.Tensor:
    """Forward noising in closed form: jump from x_0 straight to x_t.

        x_t = sqrt(alphabar[t]) * x_0 + sqrt(1 - alphabar[t]) * noise

    Args:
        x0: (B, 2) clean samples.
        t: (B,) int64 timesteps, each in {0, ..., T-1}.
        noise: (B, 2) standard-normal noise (provided by the caller — this
            function draws nothing itself).
        alphabar: (T,) cumulative products from make_schedule. Use
            alphabar[t], the CUMULATIVE product, not alpha[t]; per-sample
            gather, e.g. alphabar[t].unsqueeze(1) to broadcast over xy.

    Returns:
        x_t: (B, 2) float32.
    """
    raise NotImplementedError
# ============================ END TODO (Task 2) ==============================


# ============ TODO (Task 3): DDPM training loss and sampling =================
def ddpm_train_loss(model: nn.Module, x0_batch: torch.Tensor,
                    t_batch: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """The epsilon-prediction objective (DDPM "simple" loss).

    Steps (exactly):
        1. (beta, alpha, alphabar) = make_schedule(T_DIFF, BETA_MIN, BETA_MAX).
        2. x_t = q_sample(x0_batch, t_batch, noise, alphabar).
        3. eps_hat = model(x_t, t_batch).
        4. Return F.mse_loss(eps_hat, noise) — the mean squared error between
           the PREDICTED noise and the noise that was actually mixed in.
           (Predicting x_0 is a different parameterization — not this spec.)

    Returns:
        A scalar tensor (keep the graph — the caller backpropagates it).
    """
    raise NotImplementedError


def ddpm_sample(model: nn.Module, n: int, rng: torch.Generator) -> torch.Tensor:
    """Ancestral sampling: run the learned reverse chain from pure noise.

    Steps (exactly):
        1. (beta, alpha, alphabar) = make_schedule(T_DIFF, BETA_MIN, BETA_MAX).
        2. model.eval(); everything below inside torch.no_grad().
        3. x = torch.randn(n, 2, generator=rng)          # x_{T-1} ~ N(0, I)
        4. For t = T_DIFF-1 down to 0:
             tt = torch.full((n,), t, dtype=torch.int64)
             eps_hat = model(x, tt)
             mean = (x - beta[t] / sqrt(1 - alphabar[t]) * eps_hat)
                    / sqrt(alpha[t])                      # posterior mean
             if t > 0:
                 x = mean + sqrt(beta[t]) * torch.randn(n, 2, generator=rng)
                 (sigma_t = sqrt(beta_t) — the fixed-variance choice)
             else:
                 x = mean                                 # NO noise at t == 0
        5. Return x, shape (n, 2).

    The rng is the only randomness: 1 initial draw + (T_DIFF - 1) step draws,
    in exactly that order (tests reproduce the trajectory).
    """
    raise NotImplementedError
# ============================ END TODO (Task 3) ==============================


# ==================== TODO (Task 4): experiment ==============================
def run_experiment(data: dict, rng: torch.Generator,
                   gan_steps: int = GAN_STEPS, ddpm_steps: int = DDPM_STEPS,
                   n_gen: int = N_GEN, out: dict | None = None) -> dict:
    """Train the GAN and the DDPM on the same ring, then compare them by
    mode coverage, sample quality, and sampling wall-clock. Feeds results.json.

    Args:
        data: dict from load_data — uses data["samples"] for TRAINING and
            data["mode_centers"] for EVALUATION ONLY (mode_stats). Training
            must never read mode_centers — hidden tests catch that.
        rng: an injected torch.Generator — the only source of randomness for
            batch selection, GAN noise, diffusion timesteps/noise, and
            sampling. Draw order is fixed by the steps below.
        gan_steps, ddpm_steps, n_gen: budget overrides (tests use small ones;
            main() uses the defaults).
        out: optional dict; if given, store the generated samples and curves
            in it for the notebook (keys "samples_gan", "samples_ddpm",
            "gan_d_curve", "gan_g_curve", "ddpm_curve").

    Steps (exactly):
        1. set_seed(SEED) FIRST — it fixes the three networks' initial
           weights. Then x = torch.as_tensor(data["samples"],
           dtype=torch.float32).
        2. Build, in this order: G = GenNet(); D = DiscNet();
           denoise = DenoiseNet();
           optG = torch.optim.Adam(G.parameters(), lr=LR_GAN);
           optD = torch.optim.Adam(D.parameters(), lr=LR_GAN);
           optE = torch.optim.Adam(denoise.parameters(), lr=LR_DDPM)
           (plain Adam, default betas, for all three).
        3. GAN training, gan_steps iterations. Per iteration:
           idx = torch.randint(0, len(x), (BATCH,), generator=rng);
           (d_loss, g_loss) = gan_step(G, D, optG, optD, x[idx], rng).
           After every LOG_EVERY-th iteration (i.e. when (step+1) % LOG_EVERY
           == 0) append round(d_loss, 4) / round(g_loss, 4) to the curves.
        4. DDPM training, ddpm_steps iterations. Per iteration, draw in this
           order: idx = torch.randint(0, len(x), (BATCH,), generator=rng);
           t_batch = torch.randint(0, T_DIFF, (BATCH,), generator=rng);
           noise = torch.randn(BATCH, 2, generator=rng);
           loss = ddpm_train_loss(denoise, x[idx], t_batch, noise);
           optE.zero_grad(); loss.backward(); optE.step().
           Record round(loss.item(), 4) on the same LOG_EVERY grid.
        5. Sampling, GAN first (time each with time.perf_counter):
           G.eval(); with torch.no_grad():
             z = torch.randn(n_gen, Z_DIM, generator=rng); s_gan = G(z).
           s_ddpm = ddpm_sample(denoise, n_gen, rng).
        6. Metrics: mode_stats(s.numpy(), data["mode_centers"]) for each.

    Returns dict (plain Python ints/floats, floats rounded to 4 decimals;
    the two *_sample_seconds are wall-clock and legitimately vary — every
    OTHER value must be byte-identical across runs):
        {"n_train": int, "n_gen": int,
         "coverage_gan": int, "coverage_ddpm": int,
         "hq_gan": float, "hq_ddpm": float,
         "gan_mode_shares": [8 floats], "ddpm_mode_shares": [8 floats],
         "d_loss_final": float, "g_loss_final": float,
         "ddpm_loss_final": float,
         "gan_d_curve": [...], "gan_g_curve": [...], "ddpm_curve": [...],
         "gan_sample_seconds": float, "ddpm_sample_seconds": float}
    """
    raise NotImplementedError
# ============================ END TODO (Task 4) ==============================


def main() -> dict:
    """DO NOT MODIFY. Writes results.json with the course-wide schema."""
    set_seed()
    t0 = time.time()
    data = load_data(DATA_PATH)
    rng = torch.Generator().manual_seed(SEED)
    metrics = run_experiment(data, rng)
    results = {
        "lab": "lab09",
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
