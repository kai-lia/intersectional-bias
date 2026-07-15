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


def permutation_null(X: np.ndarray, Y: np.ndarray, n_perm: int = 1000, seed: int = 0) -> np.ndarray:
    """Null distribution of linear_cka_gram(X, Y) under random row permutation
    of Y -- i.e. what CKA looks like once the scenario_id matching is broken."""
    rng = np.random.default_rng(seed)
    scores = np.empty(n_perm)
    for i in range(n_perm):
        scores[i] = linear_cka_gram(X, Y[rng.permutation(len(Y))])
    return scores


def bootstrap_diff(X_a: np.ndarray, X_b: np.ndarray, X_c: np.ndarray,
                    n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of diff = CKA(a,b) - CKA(a,c), resampling matched
    scenario rows (the same resampled index is applied to a, b, and c)."""
    rng = np.random.default_rng(seed)
    n = len(X_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = linear_cka_gram(X_a[idx], X_b[idx]) - linear_cka_gram(X_a[idx], X_c[idx])
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
