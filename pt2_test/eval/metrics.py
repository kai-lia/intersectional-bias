"""Reusable representational-similarity metrics for pt2_test/eval scripts."""
import numpy as np


def linear_cka_gram(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA via the (n,n) Gram-matrix form.
    X, Y: (n, d) matched rows (same scenario_id). Well-conditioned when
    n (scenarios) << d (hidden size) -- unlike the (d,d) covariance form.
    Returns a scalar in roughly [0,1] (can exceed slightly due to float noise).
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    K = X @ X.T
    L = Y @ Y.T
    hsic   = np.sum(K * L)
    norm_k = np.linalg.norm(K, "fro")
    norm_l = np.linalg.norm(L, "fro")
    return float(hsic / (norm_k * norm_l))


def cosine_sim(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity, X/Y: (n, d) matched rows -> (n,) scores.
    Meaningful for n=1 smoke-test files where linear CKA is degenerate."""
    return (X * Y).sum(-1) / (np.linalg.norm(X, axis=-1) * np.linalg.norm(Y, axis=-1))


def _raw_gram(X: np.ndarray) -> np.ndarray:
    """Uncentered (n,n) Gram matrix, X @ X.T. Building this once and then
    reusing it via row/column fancy-indexing (repeated indices supported)
    is exact -- (X @ X.T)[idx][:, idx] == X[idx] @ X[idx].T entrywise --
    but costs O(n^2) per resample instead of O(n^2 d), since the O(n^2 d)
    work is paid once instead of once per bootstrap/permutation draw."""
    return X @ X.T


def _double_center(K: np.ndarray) -> np.ndarray:
    """H K H, H = I - (1/m) 11^T. Equivalent to mean-centering the
    underlying rows before computing their Gram matrix (the standard
    HSIC double-centering identity, e.g. Gretton et al.), but applied
    directly to an already-selected (m,m) raw Gram submatrix -- so it
    reproduces linear_cka_gram's centering exactly for any row subset."""
    return K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()


def _cka_from_centered(Kc: np.ndarray, Lc: np.ndarray) -> float:
    hsic   = np.sum(Kc * Lc)
    norm_k = np.linalg.norm(Kc, "fro")
    norm_l = np.linalg.norm(Lc, "fro")
    return float(hsic / (norm_k * norm_l))


def permutation_null(X: np.ndarray, Y: np.ndarray, n_perm: int = 1000, seed: int = 0) -> np.ndarray:
    """Null distribution of linear_cka_gram(X, Y) under random row permutation
    of Y -- i.e. what CKA looks like once the scenario_id matching is broken.

    X never changes across draws, so its centered Gram is built once
    up-front; Y's raw Gram is also built once and each permutation just
    re-indexes it (see _raw_gram) rather than recomputing Y_perm @ Y_perm.T
    from the (n,d) data on every one of n_perm iterations -- for
    n in the thousands this is the difference between minutes and days.
    """
    rng = np.random.default_rng(seed)
    n = len(Y)
    Kx = _double_center(_raw_gram(X))
    Ky_raw = _raw_gram(Y)
    scores = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        Ly = _double_center(Ky_raw[np.ix_(perm, perm)])
        scores[i] = _cka_from_centered(Kx, Ly)
    return scores


def bootstrap_diff(X_a: np.ndarray, X_b: np.ndarray, X_c: np.ndarray,
                    n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of diff = CKA(a,b) - CKA(a,c), resampling matched
    scenario rows (the same resampled index is applied to a, b, and c).

    Precomputes each array's raw (uncentered) Gram matrix once, then for
    each bootstrap draw builds the resampled Gram by fancy-indexing
    (supports repeated indices from sampling-with-replacement) and
    double-centers that submatrix directly -- exactly equivalent to
    calling linear_cka_gram(X[idx], Y[idx]) from scratch each time, but
    O(n^2) per draw instead of O(n^2 d). Needed once n (scenarios) grows
    into the thousands -- e.g. 3885 scenarios made the naive per-iteration
    recompute take multiple days; this brings it back to minutes.
    """
    rng = np.random.default_rng(seed)
    n = len(X_a)
    Ka_raw = _raw_gram(X_a)
    Kb_raw = _raw_gram(X_b)
    Kc_raw = _raw_gram(X_c)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        sub = np.ix_(idx, idx)
        Ka = _double_center(Ka_raw[sub])
        Kb = _double_center(Kb_raw[sub])
        Kc = _double_center(Kc_raw[sub])
        diffs[i] = _cka_from_centered(Ka, Kb) - _cka_from_centered(Ka, Kc)
    return diffs


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values in the
    same order as the input."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out
