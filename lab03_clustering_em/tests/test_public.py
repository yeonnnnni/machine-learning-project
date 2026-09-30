"""Lab 3 public tests. DO NOT MODIFY.

Run: python -m pytest tests/ -q
Grading also runs hidden tests (different scenes, edge cases, determinism,
one E-step/M-step checked against your own responsibilities). Everything here
is numpy-only, like src/lab03.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lab03  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "lab03_scene.npz"


@pytest.fixture(scope="module")
def scene():
    return lab03.load_data(DATA)


def three_blobs(seed: int = 5):
    """Three well-separated isotropic blobs (k-means solves these exactly)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.15, (60, 3))
    B = rng.normal(0, 0.15, (50, 3)) + np.array([3.0, 0.0, 0.0])
    C = rng.normal(0, 0.15, (40, 3)) + np.array([0.0, 3.0, 2.0])
    X = np.vstack([A, B, C])
    lab = np.r_[np.zeros(60, int), np.ones(50, int), np.full(40, 2)]
    perm = rng.permutation(len(X))
    return X[perm], lab[perm]


def test_kmeans_recovers_separated_clusters():
    # the provided ARI helper is part of the graded contract — spot-check it
    # first (known values, verified against the contingency formula)
    a = np.array([0, 0, 1, 1, 2, 2])
    assert lab03.adjusted_rand_index(a, a) == pytest.approx(1.0)
    assert lab03.adjusted_rand_index(a, 2 - a) == pytest.approx(1.0)   # renamed
    assert lab03.adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(-0.5)
    assert lab03.adjusted_rand_index(a, np.zeros(6, int)) == pytest.approx(0.0)
    X, lab = three_blobs()
    centers, labels, inertia = lab03.kmeans(X, 3, np.random.default_rng(0))
    assert centers.shape == (3, 3) and labels.shape == (len(X),)
    assert np.issubdtype(labels.dtype, np.integer)
    assert isinstance(inertia, float)
    # perfect partition on well-separated blobs
    assert lab03.adjusted_rand_index(lab, labels) == pytest.approx(1.0)
    # returned inertia is consistent with the returned centers and labels
    recomputed = float(((X - centers[labels]) ** 2).sum())
    assert inertia == pytest.approx(recomputed, rel=1e-12)
    # centers sit on the construction centroids (order-free comparison)
    true_centers = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 2]], dtype=float)
    dists = np.linalg.norm(centers[:, None, :] - true_centers[None, :, :], axis=2)
    assert dists.min(axis=1).max() < 0.15
    # same generator seed -> identical result (all randomness comes from rng)
    _, labels2, inertia2 = lab03.kmeans(X, 3, np.random.default_rng(0))
    assert np.array_equal(labels, labels2) and inertia == inertia2


def test_kmeans_empty_cluster_rule():
    """k = 4 on only 3 distinct locations: one seeded center is always a
    duplicate, ties send its points to the lower index, and the cluster comes
    up empty on every Lloyd pass. The re-seeding rule in the docstring (take
    the farthest point, ascending cluster order) must keep all k clusters
    non-empty; skipping it leaves an empty cluster or NaN centers."""
    P = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 1.0], [0.0, 3.0, 1.0]])
    X = np.repeat(P, [30, 20, 10], axis=0)
    centers, labels, inertia = lab03.kmeans(X, 4, np.random.default_rng(3))
    assert np.isfinite(centers).all()
    assert np.bincount(labels, minlength=4).min() >= 1
    assert inertia == pytest.approx(0.0, abs=1e-12)


def test_gmm_em_basic_properties():
    X, lab = three_blobs(seed=11)
    means, covs, weights, resp, lls = lab03.gmm_em(X, 3, np.random.default_rng(1))
    assert means.shape == (3, 3) and covs.shape == (3, 3, 3)
    assert weights.shape == (3,) and resp.shape == (len(X), 3)
    # responsibilities: rows sum to 1; weights: a distribution
    np.testing.assert_allclose(resp.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-9)
    assert (weights > 0).all()
    # covariances: symmetric, and the reg * I floor keeps them non-singular
    for j in range(3):
        np.testing.assert_allclose(covs[j], covs[j].T, atol=1e-12)
        assert np.linalg.eigvalsh(covs[j]).min() >= 0.9 * lab03.REG
    # the trace has exactly GMM_MAX_ITER entries and never decreases
    # (beyond float rounding — EM's monotonicity guarantee)
    assert len(lls) == lab03.GMM_MAX_ITER
    assert all(isinstance(v, float) for v in lls)
    assert min(np.diff(lls)) > -1e-6
    # on separated blobs the k-means init is already near-optimal, so the
    # trace may be flat — it must simply never fall below its start
    assert lls[-1] >= lls[0] - 1e-9
    # separated blobs: the mixture recovers the construction exactly
    assert lab03.adjusted_rand_index(lab, resp.argmax(axis=1)) == pytest.approx(1.0)


def test_gmm_em_handles_flat_cluster():
    """A cluster that is exactly flat (zero variance along one axis) has a
    singular sample covariance. The reg * I term added to EVERY covariance
    estimate — at init and after each M-step — must keep the E-step running."""
    rng = np.random.default_rng(9)
    flat = np.column_stack([rng.uniform(-1, 1, 80), rng.uniform(-0.6, 0.6, 80),
                            np.zeros(80)])
    blob = rng.normal(0, 0.2, (70, 3)) + np.array([0.0, 0.0, 2.0])
    X = np.vstack([flat, blob])
    lab = np.r_[np.zeros(80, int), np.ones(70, int)]
    perm = rng.permutation(len(X))
    X, lab = X[perm], lab[perm]
    means, covs, weights, resp, lls = lab03.gmm_em(X, 2, np.random.default_rng(4))
    assert np.isfinite(lls).all() and np.isfinite(resp).all()
    for j in range(2):
        assert np.linalg.eigvalsh(covs[j]).min() >= 0.9 * lab03.REG
    assert lab03.adjusted_rand_index(lab, resp.argmax(axis=1)) == pytest.approx(1.0)


def test_bic_and_select_k_on_blobs():
    # hand-checkable BIC values: -2*ll + n_params * log(n)
    assert lab03.bic(-100.0, 10, 50) == pytest.approx(200.0 + 10 * np.log(50.0))
    assert lab03.bic(-10.0, 0, 123) == pytest.approx(20.0)
    X, _ = three_blobs(seed=21)
    bics, best_k = lab03.select_k(X, (2, 3, 4), np.random.default_rng(2))
    assert isinstance(bics, list) and len(bics) == 3
    assert all(isinstance(b, float) for b in bics)
    assert best_k == 3                      # the constructed number of blobs
    assert bics[1] == min(bics)


def test_run_experiment_schema_and_sanity(scene):
    m1 = lab03.run_experiment(scene, np.random.default_rng(lab03.SEED))
    m2 = lab03.run_experiment(scene, np.random.default_rng(lab03.SEED))
    assert m1 == m2                          # deterministic given the rng seed
    keys = {"n_points", "kmeans_inertia", "ari_kmeans", "ari_gmm",
            "ll_first", "ll_final", "ll_min_delta", "best_k"} \
        | {f"bic_k{k}" for k in lab03.K_RANGE}
    assert set(m1) == keys
    assert m1["n_points"] == len(scene["X"])
    # the lab's headline result: on this anisotropic scene the full-covariance
    # GMM beats k-means decisively
    assert m1["ari_gmm"] > 0.85
    assert m1["ari_kmeans"] < 0.75
    assert m1["ari_gmm"] > m1["ari_kmeans"] + 0.15
    # EM behaved: likelihood rose and never decreased (up to float rounding)
    assert m1["ll_final"] > m1["ll_first"]
    assert m1["ll_min_delta"] > -1e-6
    # BIC recovers the true number of objects
    assert m1["best_k"] == 4
    assert m1["bic_k4"] == min(m1[f"bic_k{k}"] for k in lab03.K_RANGE)
