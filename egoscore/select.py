"""Subset selectors. Each takes an embedding matrix and a budget K, returns K indices.

All selectors run *after* the quality gate, so what we measure here is the value of
selection, not of filtering. Filtering is priced separately by the `random_nogate`
condition in the experiment.
"""

from __future__ import annotations

import numpy as np


def _pairwise_sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.maximum(
        (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0
    )


def random_select(Z: np.ndarray, k: int, seed: int = 0, **_) -> np.ndarray:
    """The baseline that actually matters."""
    rng = np.random.default_rng(seed)
    return rng.choice(len(Z), size=min(k, len(Z)), replace=False)


def facility_location(Z: np.ndarray, k: int, seed: int = 0, **_) -> np.ndarray:
    """Greedy maximisation of sum_i max_{j in S} sim(i, j).

    Facility location is monotone submodular, so greedy carries the standard
    (1 - 1/e) approximation guarantee. That guarantee is the reason to prefer it over
    an ad-hoc diversity heuristic: we can say what the algorithm is approximating and
    how badly it can miss.

    Interpretation: pick the K episodes such that every episode in the pool has some
    similar representative in the chosen set. Coverage, not novelty.
    """
    n = len(Z)
    k = min(k, n)
    D = _pairwise_sq(Z, Z)
    S = -D  # similarity = negative squared distance
    best = np.full(n, -np.inf)
    chosen: list[int] = []
    # Seed with the medoid so the result is deterministic given Z.
    first = int(np.argmax(S.sum(axis=1)))
    chosen.append(first)
    best = np.maximum(best, S[:, first])
    while len(chosen) < k:
        # Gain of adding j = sum_i (max(best_i, S_ij) - best_i)
        gains = np.maximum(S - best[:, None], 0).sum(axis=0)
        gains[chosen] = -np.inf
        j = int(np.argmax(gains))
        chosen.append(j)
        best = np.maximum(best, S[:, j])
    return np.array(chosen)


def kcenter(Z: np.ndarray, k: int, seed: int = 0, **_) -> np.ndarray:
    """Greedy k-center: repeatedly take the point furthest from everything chosen."""
    n = len(Z)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(n))]
    d = _pairwise_sq(Z, Z[chosen[:1]]).ravel()
    while len(chosen) < k:
        j = int(np.argmax(d))
        chosen.append(j)
        d = np.minimum(d, _pairwise_sq(Z, Z[j:j + 1]).ravel())
    return np.array(chosen)


def dpp_logdet(Z: np.ndarray, k: int, seed: int = 0, gamma: float | None = None, **_) -> np.ndarray:
    """Greedy log-determinant maximisation over an RBF kernel — a diversity ablation.

    Where facility location rewards *coverage*, log-det rewards *spread*: it will
    happily pick outliers. Including both lets us show which notion actually helps.
    """
    n = len(Z)
    k = min(k, n)
    D = _pairwise_sq(Z, Z)
    if gamma is None:
        med = np.median(D[D > 0]) if (D > 0).any() else 1.0
        gamma = 1.0 / max(med, 1e-9)
    L = np.exp(-gamma * D) + 1e-6 * np.eye(n)

    chosen: list[int] = []
    # Standard greedy MAP for a DPP via incremental Cholesky.
    diag = np.diag(L).copy()
    c = np.zeros((k, n), dtype=np.float64)
    for it in range(k):
        j = int(np.argmax(np.where(np.isin(np.arange(n), chosen), -np.inf, diag)))
        chosen.append(j)
        if it == k - 1:
            break
        cj = (L[j] - c[:it].T @ c[:it, j]) / np.sqrt(max(diag[j], 1e-12))
        c[it] = cj
        diag = np.maximum(diag - cj ** 2, 1e-12)
    return np.array(chosen)


def degenerate(Z: np.ndarray, k: int, seed: int = 0, groups: np.ndarray | None = None, **_) -> np.ndarray:
    """POSITIVE CONTROL: concentrate the budget in as few operator/scene groups as possible.

    The EgoVerse paper established that demonstrator and scene diversity improve
    generalization. So this condition should lose. If it does not, our harness cannot
    detect an effect the dataset authors already found, and any null result elsewhere
    is uninterpretable. This is the load-bearing sanity check of the whole experiment.
    """
    n = len(Z)
    k = min(k, n)
    if groups is None:
        return random_select(Z, k, seed)
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(groups, return_counts=True)
    # Take whole groups, largest first, until the budget is met.
    order = uniq[np.argsort(-counts)]
    picked: list[int] = []
    for g in order:
        idx = np.flatnonzero(groups == g)
        rng.shuffle(idx)
        picked.extend(idx[: k - len(picked)].tolist())
        if len(picked) >= k:
            break
    return np.array(picked[:k])


SELECTORS = {
    "random": random_select,
    "curated": facility_location,
    "kcenter": kcenter,
    "dpp": dpp_logdet,
    "degenerate": degenerate,
}
