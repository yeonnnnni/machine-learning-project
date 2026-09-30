"""Lab 3 — K-means, GMM, and EM: clustering a 3D scene without labels.

Machine Learning Project (53744-01), Fall 2026.

Fill in every TODO block. Do NOT rename functions or change their
signatures/return types — automated (public + hidden) tests call them directly.
Allowed imports: numpy and the standard library only (no scipy, no sklearn —
the point is to build K-means and EM yourself).
Run:      python src/lab03.py
Self-check: python -m pytest tests/ -q
"""
import json
import time
from pathlib import Path

import numpy as np

# ---- Fill in your information (used in results.json) ----
STUDENT_ID = "00000000"   # TODO: your student id, e.g. "20261234"
STUDENT_NAME = "None"     # TODO: your name in Korean or roman letters — "홍길동" / "HongGildong"

SEED = 42  # fixed for the whole course — DO NOT CHANGE
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "lab03_scene.npz"

# Constants used by the tasks — DO NOT CHANGE (hidden tests depend on them)
K_OBJECTS = 4          # number of objects in the shipped scene
K_RANGE = (2, 3, 4, 5, 6, 7, 8)   # candidate k values for model selection
N_INIT = 4             # k-means restarts (best of N_INIT by inertia)
GMM_MAX_ITER = 80      # EM iterations (fixed count, no early stopping)
REG = 1e-6             # covariance regularization added every M-step


def set_seed(seed: int = SEED) -> None:
    """DO NOT MODIFY."""
    np.random.seed(seed)


def load_data(path: str | Path) -> dict:
    """DO NOT MODIFY. Loads the scene file into a dict of arrays:
    X (N, 3) float64 merged surface points of several objects, shuffled;
    labels (N,) int64 ground-truth object index per point. The labels are
    shipped for EVALUATION ONLY: they may be used inside run_experiment's
    evaluation step (adjusted_rand_index) and nowhere else. They must never
    be passed to, or used around, kmeans / gmm_em / select_k — hidden tests
    detect fitting that depends on the labels.
    """
    with np.load(path) as f:
        return {"X": f["X"].astype(np.float64),
                "labels": f["labels"].astype(np.int64)}


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """DO NOT MODIFY. Adjusted Rand index between two labelings (provided).

    ARI = (RI - E[RI]) / (max RI - E[RI]) over point pairs; 1.0 = identical
    partitions (up to renaming), ~0.0 = as good as random assignment.
    """
    labels_a = np.asarray(labels_a).ravel()
    labels_b = np.asarray(labels_b).ravel()
    n = labels_a.size
    _, ia = np.unique(labels_a, return_inverse=True)
    _, ib = np.unique(labels_b, return_inverse=True)
    contingency = np.zeros((ia.max() + 1, ib.max() + 1), dtype=np.int64)
    np.add.at(contingency, (ia, ib), 1)

    def comb2(x):
        x = np.asarray(x, dtype=np.float64)
        return (x * (x - 1.0) / 2.0).sum()

    sum_ij = comb2(contingency)
    sum_a = comb2(contingency.sum(axis=1))
    sum_b = comb2(contingency.sum(axis=0))
    total = n * (n - 1.0) / 2.0
    expected = sum_a * sum_b / total
    max_index = 0.5 * (sum_a + sum_b)
    if max_index == expected:          # degenerate: single cluster on both sides
        return 1.0
    return float((sum_ij - expected) / (max_index - expected))


# =================== TODO (Task 1): K-means with k-means++ ===================
def kmeans(X: np.ndarray, k: int, rng: np.random.Generator,
           n_init: int = N_INIT, max_iter: int = 100
           ) -> tuple[np.ndarray, np.ndarray, float]:
    """K-means clustering: k-means++ seeding + Lloyd iterations, best of n_init.

    Args:
        X: (N, d) float64 points.
        k: number of clusters (2 <= k <= N).
        rng: np.random.Generator. ALL randomness comes from this generator
            (never np.random.*), so the same seed gives the same result.
            Follow the calls listed below, in this order, and your numbers
            will reproduce the reference values used in the concept note and
            the grading notes; the graded tests check what the algorithm
            does (clusters found, empty-cluster rule, restarts), not the
            exact sequence of draws.
        n_init: number of independent restarts; keep the best by inertia.
        max_iter: Lloyd iteration cap per restart.

    One restart (repeated n_init times, one after another on the same rng):
        A. k-means++ seeding:
           1. First center: idx = rng.integers(N); centers[0] = X[idx].
           2. For each next center j = 1 .. k-1:
              d2[i] = min over already chosen centers c of ||X[i] - c||^2.
              If d2.sum() == 0 (all points coincide with a chosen center):
                  idx = rng.integers(N).
              Else:
                  idx = rng.choice(N, p=d2 / d2.sum())   # D^2 sampling
              centers[j] = X[idx].
        B. Lloyd iterations (at most max_iter):
           1. Assign: labels[i] = argmin_j ||X[i] - centers[j]||^2
              (ties -> the smallest j, which is what np.argmin does).
           2. Empty-cluster fix: let d2a[i] = squared distance from X[i] to
              its assigned center (the centers used in step 1). For each
              empty cluster j in ASCENDING order:
                  sel = argmax_i d2a[i]   (ties -> smallest i)
                  labels[sel] = j; centers[j] = X[sel]; d2a[sel] = -1.0
              so the same point is never taken for two empty clusters.
           3. Converge: if this is not the first iteration and labels are
              identical to the previous iteration's labels, STOP (before
              updating the centers — they are already the means of these
              members from the previous update).
           4. Update: centers[j] = mean of the points with labels == j.
        C. Inertia: sum_i ||X[i] - centers[labels[i]]||^2 with the final
           centers and labels of this restart.

    Best restart: strictly smaller inertia wins; on a tie the earlier
    restart is kept.

    Returns:
        (centers, labels, inertia): centers (k, d) float64,
        labels (N,) int64, inertia plain Python float.
    """
    raise NotImplementedError
# ============================ END TODO (Task 1) ==============================


# ================== TODO (Task 2): Gaussian mixture via EM ===================
def gmm_em(X: np.ndarray, k: int, rng: np.random.Generator,
           max_iter: int = GMM_MAX_ITER, reg: float = REG
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Full-covariance Gaussian mixture fitted by EM, initialized from k-means.

    Args:
        X: (N, d) float64 points.
        k: number of components.
        rng: np.random.Generator, consumed ONLY by the internal kmeans call.
        max_iter: number of EM iterations. Run EXACTLY this many (no early
            stopping) so that the log-likelihood trace has max_iter entries.
        reg: covariance regularization. reg * I is added to EVERY estimated
            covariance — at initialization and after every M-step — so that
            a degenerate (e.g., perfectly planar) cluster still yields an
            invertible covariance.

    Initialization (from one kmeans run):
        centers, labels, _ = kmeans(X, k, rng)          # default n_init
        means = centers
        weights[j] = N_j / N          (N_j = members of cluster j)
        covs[j] = (1/N_j) * sum_{i in j} (X[i]-centers[j])(X[i]-centers[j])^T
                  + reg * I

    One EM iteration (repeated exactly max_iter times):
        E-step, IN LOG SPACE (this is not optional): for every i, j
            logp[i, j] = log weights[j] + log N(X[i] | means[j], covs[j])
        and the per-point normalizer via LOG-SUM-EXP:
            m[i] = max_j logp[i, j]
            lse[i] = m[i] + log( sum_j exp(logp[i, j] - m[i]) )
        resp[i, j] = exp(logp[i, j] - lse[i])           # rows sum to 1
        Append ll = sum_i lse[i] to the trace (the log-likelihood of the
        CURRENT parameters, i.e., before this iteration's M-step).
        Why log space: log-densities of far components reach the -10^2..-10^8
        range; exponentiating them before normalizing underflows to 0 for
        every component at once on wide-dynamic-range scenes, giving 0/0
        responsibilities and log(0) likelihoods. Subtracting the row maximum
        first makes the largest term exp(0) = 1, which never underflows.

        M-step (closed forms):
            Nk[j] = sum_i resp[i, j]
            weights[j] = Nk[j] / N
            means[j] = (sum_i resp[i, j] * X[i]) / Nk[j]
            covs[j] = (sum_i resp[i, j] * (X[i]-means[j])(X[i]-means[j])^T)
                      / Nk[j] + reg * I      # the NEW means[j]; reg added here

    Returns:
        (means, covs, weights, resp, log_likelihoods):
        means (k, d), covs (k, d, d), weights (k,), resp (N, k) — all float64,
        the parameters after the last M-step and the responsibilities of the
        last E-step; log_likelihoods = list of max_iter plain Python floats.
    """
    raise NotImplementedError
# ============================ END TODO (Task 2) ==============================


# ================ TODO (Task 3): model selection with BIC ====================
def bic(ll: float, n_params: int, n: int) -> float:
    """Bayesian information criterion (lower is better).

    BIC = -2 * ll + n_params * log(n)

    Args:
        ll: maximized log-likelihood of the model.
        n_params: number of free parameters of the model.
        n: number of data points.

    Returns:
        The BIC value as a plain Python float.
    """
    raise NotImplementedError


def select_k(X: np.ndarray, k_range: tuple[int, ...], rng: np.random.Generator
             ) -> tuple[list[float], int]:
    """Fit one GMM per candidate k and pick the k with the smallest BIC.

    Args:
        X: (N, d) float64 points.
        k_range: candidate k values, ascending (e.g., K_RANGE).
        rng: np.random.Generator. Fit the candidates IN ORDER with this SAME
            generator object (its state carries over from one fit to the
            next); do not create new generators inside.

    For each k in k_range:
        1. means, covs, weights, resp, lls = gmm_em(X, k, rng)  # defaults
        2. ll = lls[-1]
        3. Free-parameter count of a d-dimensional, k-component
           full-covariance GMM:
               n_params = k * d              (means)
                        + k * d * (d + 1) // 2   (symmetric covariances)
                        + (k - 1)           (weights sum to 1)
        4. bic_k = bic(ll, n_params, n)

    Returns:
        (bic_values, best_k): bic_values = list of plain Python floats, one
        per k in order; best_k = the k with the smallest BIC (ties -> the
        smaller k).
    """
    raise NotImplementedError
# ============================ END TODO (Task 3) ==============================


# ==================== TODO (Task 4): experiment ==============================
def run_experiment(data: dict, rng: np.random.Generator) -> dict:
    """K-means vs GMM on the 3D scene, EM monotonicity, and BIC model selection.

    Args:
        data: dict with X (N, 3) and labels (N,) — see load_data. The
            ground-truth labels are used ONLY as the reference argument of
            adjusted_rand_index in step 3 below. They must not influence
            kmeans, gmm_em, or select_k in any way (the hidden tests rerun
            this function with permuted labels and require every
            fit-derived metric to be unchanged).
        rng: np.random.Generator; consumed by the steps below in this order.

    Steps (in this order, all with K_OBJECTS / K_RANGE / defaults):
        1. centers, km_labels, inertia = kmeans(X, K_OBJECTS, rng)
        2. means, covs, weights, resp, lls = gmm_em(X, K_OBJECTS, rng)
           gmm_labels = resp.argmax(axis=1)
        3. ari_kmeans = adjusted_rand_index(labels, km_labels)
           ari_gmm    = adjusted_rand_index(labels, gmm_labels)
        4. EM sanity: ll_first = lls[0], ll_final = lls[-1],
           ll_min_delta = min over consecutive pairs of (lls[t+1] - lls[t])
           (an exact M-step never decreases the likelihood; the reg*I term
           makes the M-step slightly non-optimal, so tiny negative values
           on the order of -1e-8 are normal).
        5. bic_values, best_k = select_k(X, K_RANGE, rng)

    Returns dict (plain Python ints/floats; floats rounded to 4 decimals
    except ll_min_delta, rounded to 9):
        {"n_points": int, "kmeans_inertia": float,
         "ari_kmeans": float, "ari_gmm": float,
         "ll_first": float, "ll_final": float, "ll_min_delta": float,
         "bic_k2": float, ..., "bic_k8": float, "best_k": int}
    """
    raise NotImplementedError
# ============================ END TODO (Task 4) ==============================


def main() -> dict:
    """DO NOT MODIFY. Writes results.json with the course-wide schema."""
    set_seed()
    t0 = time.time()
    data = load_data(DATA_PATH)
    metrics = run_experiment(data, np.random.default_rng(SEED))
    results = {
        "lab": "lab03",
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
