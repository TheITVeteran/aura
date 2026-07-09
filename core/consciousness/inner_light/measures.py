"""core/consciousness/inner_light/measures.py — the discriminator measures.

Each function takes a spatiotemporal activity matrix ``M`` of shape
``(n_channels, n_timesteps)`` (real values; binarised internally where a measure
needs it) and returns a scalar. Together they span the axes on which conscious
brains separate from unconscious brains and from non-neural systems:

  * differentiation  — Lempel-Ziv / PCI complexity: the response is rich, not a
    stereotyped echo. Maximal for noise, minimal for a constant.
  * integration      — how much the channels' joint behaviour exceeds the sum of
    the parts (multi-information). Zero for independent channels.
  * complexity (TSE) — the *balance*: integration AND differentiation together.
    Near-zero for both pure noise (all differentiation) and a fully-synchronised
    system (all integration); high only in between.
  * criticality      — long-range temporal correlations (DFA Hurst) and scale-free
    avalanches: the edge of chaos. ~0.5 for noise, ~1.5 for a random walk.
  * ignition         — non-linear all-or-none global broadcast: a bimodal global
    activation distribution, not a smooth unimodal one.

All pure, deterministic (fixed RNG for any sampling), numpy-only.
"""
from __future__ import annotations

import numpy as np

_RNG_SEED = 20260709


def _as_matrix(M: np.ndarray | list) -> np.ndarray:
    arr = np.asarray(M, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("activity matrix must be 2-D (channels × time)")
    return arr


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


# ─────────────────────────────────────────────────────────────────────────────
# Differentiation — Lempel-Ziv / PCI complexity
# ─────────────────────────────────────────────────────────────────────────────

def binarize(M: np.ndarray | list, *, method: str = "median") -> np.ndarray:
    """Binarise per channel. 'median' → above the channel's own median."""
    arr = _as_matrix(M)
    if method == "median":
        thr = np.median(arr, axis=1, keepdims=True)
    elif method == "mean":
        thr = np.mean(arr, axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown binarize method: {method}")
    return (arr > thr).astype(np.int8)


def lz76(seq) -> int:
    """Lempel-Ziv (LZ76 / Kaspar-Schuster) substring complexity of a sequence."""
    s = list(seq)
    n = len(s)
    if n <= 1:
        return 1
    i = 0
    c = 1
    length = 1
    k = 1
    k_max = 1
    while True:
        if s[i + k - 1] == s[length + k - 1]:
            k += 1
            if length + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == length:
                c += 1
                length += k_max
                if length + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return c


def normalized_lz(M: np.ndarray | list, *, method: str = "median") -> float:
    """Normalised Lempel-Ziv complexity of the binarised spatiotemporal matrix.

    The binary matrix is concatenated time-major (each timestep's spatial pattern
    in turn) — the PCI convention — then LZ76 is normalised by ``n/log2(n)`` so a
    random binary sequence → ~1.0 and a constant → ~0.
    """
    b = binarize(M, method=method)
    seq = b.T.flatten()  # time-major: spatial pattern per timestep, concatenated
    n = len(seq)
    if n <= 1:
        return 0.0
    c = lz76(seq.tolist())
    norm = n / np.log2(n)
    return _clamp(c / norm if norm > 0 else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Integration & TSE neural complexity
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_integration(cov: np.ndarray) -> float:
    """Multi-information (total correlation) of a Gaussian with covariance ``cov``.

    I(X) = 0.5 * log( Π var_i / det(cov) ). Zero for independent channels, large
    when the joint entropy is far below the sum of marginals. A small ridge keeps
    the determinant well-conditioned.
    """
    cov = np.asarray(cov, dtype=float)
    d = cov.shape[0]
    if d < 2:
        return 0.0
    ridge = 1e-9 * np.trace(cov) / d if np.trace(cov) > 0 else 1e-9
    cov = cov + ridge * np.eye(d)
    diag = np.clip(np.diag(cov), 1e-12, None)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return 0.0
    return float(max(0.0, 0.5 * (np.sum(np.log(diag)) - logdet)))


def _bounded_integration(corr_subset: np.ndarray) -> float:
    """Integration of a subset ∈ [0,1], from the correlation-matrix determinant.

    det(R) ∈ [0,1] is the generalised variance: 1 for independent channels, → 0
    for redundant ones. ``1 − det(R)^(1/k)`` therefore bounds integration in
    [0,1] and reflects *structure*, not coupling magnitude — the fix that makes
    the TSE integral go to ~0 for a fully-synchronised system instead of
    exploding with the coupling strength.
    """
    k = corr_subset.shape[0]
    if k < 2:
        return 0.0
    sign, logdet = np.linalg.slogdet(corr_subset)
    det = float(np.exp(logdet)) if sign > 0 else 0.0
    det = max(0.0, min(1.0, det))
    return _clamp(1.0 - det ** (1.0 / k))


def tse_complexity(M: np.ndarray | list, *, subset_samples: int = 40) -> float:
    """Tononi-Sporns-Edelman neural complexity (bounded Gaussian estimate).

    C(X) = Σ_{k=1}^{N} [ (k/N)·I(X) − <I(subset of size k)> ] with the bounded
    integration above; the average over subsets of each size k is estimated by
    deterministic sampling. High only when the system is integrated *and*
    differentiated; ~0 for both independent channels (no integration at any
    scale) and a fully-synchronised system (maximal, flat integration at every
    scale — so the profile matches the (k/N) line and the deviations cancel).
    """
    arr = _as_matrix(M)
    n = arr.shape[0]
    if n < 2 or arr.shape[1] < 2:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(arr)
    if corr.ndim < 2 or not np.all(np.isfinite(corr)):
        return 0.0
    i_full = _bounded_integration(corr)
    if i_full <= 0:
        return 0.0
    rng = np.random.default_rng(_RNG_SEED)
    total = 0.0
    for k in range(1, n + 1):
        if k == n:
            avg_sub = i_full
        elif k == 1:
            avg_sub = 0.0
        else:
            from math import comb
            samples = min(subset_samples, comb(n, k))
            acc = 0.0
            seen: set = set()
            tries = 0
            got = 0
            while got < samples and tries < samples * 8:
                tries += 1
                idx = tuple(sorted(rng.choice(n, size=k, replace=False).tolist()))
                if idx in seen:
                    continue
                seen.add(idx)
                acc += _bounded_integration(corr[np.ix_(idx, idx)])
                got += 1
            avg_sub = acc / got if got else 0.0
        total += (k / n) * i_full - avg_sub
    return float(max(0.0, total / n))


# ─────────────────────────────────────────────────────────────────────────────
# Criticality — DFA & avalanches
# ─────────────────────────────────────────────────────────────────────────────

def _global_activation(M: np.ndarray | list) -> np.ndarray:
    return _as_matrix(M).sum(axis=0)


def dfa(signal: np.ndarray | list, *, scales: np.ndarray | None = None) -> float:
    """Detrended Fluctuation Analysis → Hurst exponent α.

    ~0.5 for uncorrelated noise, >0.5 for long-range temporal correlations
    (brain resting activity ≈ 0.7–1.0), ~1.5 for a random walk (brown noise).
    """
    x = np.asarray(signal, dtype=float)
    n = len(x)
    if n < 16:
        return 0.5
    x = x - x.mean()
    y = np.cumsum(x)
    if scales is None:
        hi = max(8, n // 4)
        scales = np.unique(np.floor(np.logspace(np.log10(4), np.log10(hi), 14)).astype(int))
    F, used = [], []
    for s in scales:
        s = int(s)
        if s < 4 or s > n // 2:
            continue
        nseg = n // s
        if nseg < 1:
            continue
        rms = []
        t = np.arange(s)
        for seg in range(nseg):
            ys = y[seg * s:(seg + 1) * s]
            coef = np.polyfit(t, ys, 1)
            trend = np.polyval(coef, t)
            rms.append(np.sqrt(np.mean((ys - trend) ** 2)))
        f = np.sqrt(np.mean(np.square(rms)))
        if f > 0:
            F.append(f)
            used.append(s)
    if len(used) < 2:
        return 0.5
    slope = np.polyfit(np.log(used), np.log(F), 1)[0]
    return float(slope)


def avalanche_criticality(M: np.ndarray | list, *, threshold: float | None = None) -> dict:
    """Neuronal-avalanche analysis of the global activation.

    Avalanches are runs of supra-threshold activity; their sizes should follow a
    power law near criticality. Returns the fitted exponent, the log-log fit R²,
    the avalanche count, and a criticality ``score`` (fit quality × proximity to
    the critical exponent ≈ 1.5). Honest: with little/degenerate data the score is 0.
    """
    g = _global_activation(M)
    thr = float(np.median(g)) if threshold is None else float(threshold)
    sizes = []
    cur = 0.0
    active = False
    for val in g:
        if val > thr:
            cur += (val - thr)
            active = True
        elif active:
            sizes.append(cur)
            cur = 0.0
            active = False
    if active and cur > 0:
        sizes.append(cur)
    sizes = [s for s in sizes if s > 0]
    out = {"exponent": 0.0, "fit_r2": 0.0, "n_avalanches": len(sizes), "score": 0.0}
    if len(sizes) < 8:
        return out
    sizes_arr = np.asarray(sizes)
    counts, edges = np.histogram(sizes_arr, bins=min(20, max(4, len(sizes) // 3)))
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = counts > 0
    if mask.sum() < 3:
        return out
    lx, ly = np.log(centers[mask]), np.log(counts[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    exponent = -float(slope)
    closeness = float(np.exp(-abs(exponent - 1.5)))  # peaks at the critical ~1.5
    out.update({
        "exponent": exponent,
        "fit_r2": _clamp(r2),
        "score": _clamp(max(0.0, r2) * closeness),
    })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Ignition — non-linear all-or-none global broadcast
# ─────────────────────────────────────────────────────────────────────────────

def bimodality_ignition(M: np.ndarray | list) -> float:
    """Sarle's bimodality coefficient of the global activation distribution.

    A mind with all-or-none ignition has a bimodal global activation (a quiet
    baseline + ignited broadcasts); a linear system has a unimodal, roughly
    Gaussian one. BC ≈ 0.33 for a Gaussian, 5/9 ≈ 0.555 for a uniform, → 1 for a
    clean two-mode mixture. Returned clamped to [0,1].
    """
    g = _global_activation(M).astype(float)
    n = len(g)
    if n < 4 or g.std() < 1e-9:
        return 0.0
    z = (g - g.mean()) / g.std()
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4)) - 3.0  # EXCESS kurtosis (Sarle's BC convention)
    # sample-corrected denominator (SAS formula)
    corr = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)) if n > 3 else 3.0
    denom = kurt + corr
    if denom <= 0:
        return 0.0
    bc = (skew ** 2 + 1.0) / denom
    return _clamp(bc)


__all__ = [
    "binarize",
    "lz76",
    "normalized_lz",
    "gaussian_integration",
    "tse_complexity",
    "dfa",
    "avalanche_criticality",
    "bimodality_ignition",
]
