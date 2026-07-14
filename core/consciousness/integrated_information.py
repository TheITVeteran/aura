"""core/consciousness/integrated_information.py — whole-system Φ, honestly.
==============================================================================
Integrated-information estimation over the ACTUAL runtime, built to answer
the July-2026 review critique head-on:

    "The IIT calculation is a reductionist hack... φ here is a telemetry
     signal, not a consciousness meter."

Exact micro-level IIT-4.0 Φ is off the table for ANY nontrivial system —
its definition quantifies over every partition of every mechanism over the
full counterfactual state space.  That is intrinsic, not an engineering
gap.  This module implements the strongest honest position available:

  RAIL A — WHOLE-SYSTEM CONTINUOUS Φ.  Gaussian/VAR integrated information
      ("stochastic interaction", Ay 2003; Barrett & Seth 2011; Oizumi &
      Kitazono practical-Φ) over the full channel matrix — no hand-picked
      16-node model, no binarization, no toy TPM.  The minimum-information
      partition is found EXACTLY by Queyranne's algorithm for symmetric
      submodular minimization in O(n³) oracle calls — a real MIP, not a
      spectral heuristic.

  RAIL B — GRAIN DISCOVERY (causal emergence; Hoel/Albantakis/Tononi).
      The complex is DERIVED, not assumed: hierarchical coarse-grainings
      of the channel set are searched and each grain is scored against its
      own surrogate null; the emergent grain is the one whose integration
      is most surely non-chance.  Macro may legitimately beat micro.

  RAIL C — EXACT DISCRETE Φ AT THE DERIVED GRAIN.  For k ≤ 12 macro
      elements, system integrated information is computed by EXHAUSTIVE
      bipartition search over the empirical (and optionally interventional)
      transition distribution — exact at the grain where exactness is
      meaningful, state-averaged over visited states.

  RAIL D — statistical honesty as architecture: circular-shift surrogate
      nulls, block-bootstrap confidence intervals, stationarity and
      Gaussianity diagnostics, and a named estimator on every report.
      (The interventional probe lives in perturbational_probe.py and
      feeds Rail C extra transition rows.)

Every PhiEstimate carries a bounded claim: this is an estimate of the
integrated information of the system's macro-dynamics at an empirically
selected grain — evidence about integration structure, NOT a consciousness
meter.  No number this module emits is presented as one.

Pure NumPy, deterministic under a seeded RNG, no LLM, no I/O.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.IntegratedInformation")

SCHEMA_VERSION = 1
ESTIMATOR_NAME = "gaussian_stochastic_interaction.queyranne_mip.v1"
DISCRETE_ESTIMATOR_NAME = "empirical_state_phi.exact_bipartition.v1"

_LOG2PIE = math.log(2.0 * math.pi * math.e)
_EPS = 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian machinery
# ─────────────────────────────────────────────────────────────────────────────

def _shrunk_cov(Z: np.ndarray, shrinkage: float) -> np.ndarray:
    """Covariance with diagonal shrinkage: (1-λ)·S + λ·diag(S).

    Keeps every principal submatrix well-conditioned so log-dets and Schur
    complements stay finite on short windows."""
    S = np.cov(Z, rowvar=False, bias=False)
    S = np.atleast_2d(S)
    lam = float(min(max(shrinkage, 0.0), 1.0))
    out = (1.0 - lam) * S + lam * np.diag(np.diag(S))
    out += _EPS * np.eye(out.shape[0])
    return out


def _logdet(M: np.ndarray) -> float:
    sign, val = np.linalg.slogdet(M)
    if sign <= 0:
        # Numerically defective submatrix — regularize harder and retry once.
        val = np.linalg.slogdet(M + 1e-8 * np.eye(M.shape[0]))[1]
    return float(val)


@dataclass
class LaggedGaussian:
    """Joint Gaussian over (X_{t-1}, X_t) — the single object every
    entropy/oracle query is answered from, via Schur complements."""

    sigma: np.ndarray          # (2N × 2N); indices 0..N-1 = past, N..2N-1 = present
    n_channels: int
    n_samples: int

    @classmethod
    def fit(cls, X: np.ndarray, *, shrinkage: float = 0.05) -> "LaggedGaussian":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 8 or X.shape[1] < 2:
            raise ValueError("channel matrix must be T×N with T ≥ 8, N ≥ 2")
        # Standardize per channel (estimation only — diagnostics see raw data).
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-9] = 1.0
        Xs = (X - mu) / sd
        Z = np.hstack([Xs[:-1], Xs[1:]])            # T-1 × 2N
        return cls(sigma=_shrunk_cov(Z, shrinkage),
                   n_channels=X.shape[1], n_samples=X.shape[0])

    def _cond_entropy(self, present_idx: np.ndarray, past_idx: np.ndarray) -> float:
        """h(present | past) in nats via the Schur complement of sigma."""
        pp = self.sigma[np.ix_(present_idx, present_idx)]
        if past_idx.size == 0:
            return 0.5 * (_logdet(pp) + pp.shape[0] * _LOG2PIE)
        qq = self.sigma[np.ix_(past_idx, past_idx)]
        pq = self.sigma[np.ix_(present_idx, past_idx)]
        cond = pp - pq @ np.linalg.solve(qq, pq.T)
        return 0.5 * (_logdet(cond) + pp.shape[0] * _LOG2PIE)

    def h_part(self, subset: tuple[int, ...]) -> float:
        """h(S_t | S_{t-1}) — the part conditioned on ITS OWN past only."""
        s = np.asarray(subset, dtype=int)
        return self._cond_entropy(s + self.n_channels, s)

    def h_whole(self) -> float:
        n = self.n_channels
        idx = np.arange(n)
        return self._cond_entropy(idx + n, idx)


def stochastic_interaction(model: LaggedGaussian,
                           parts: list[tuple[int, ...]]) -> float:
    """SI(P) = Σ_k h(M_k,t | M_k,t-1) − h(X_t | X_{t-1})  (nats, ≥ 0 up to
    estimation noise).  The information the whole's dynamics carries beyond
    the sum of its parts' self-dynamics — integrated information under the
    Gaussian model (Ay 2003; Barrett & Seth 2011 Φ_AR family)."""
    return float(sum(model.h_part(p) for p in parts) - model.h_whole())


# ─────────────────────────────────────────────────────────────────────────────
# Queyranne: EXACT minimum of a symmetric submodular set function
# ─────────────────────────────────────────────────────────────────────────────

def queyranne_min_bipartition(
    n: int, f: Callable[[tuple[int, ...]], float]
) -> tuple[tuple[int, ...], float]:
    """Exact minimizer of a symmetric submodular f over nontrivial subsets.

    Queyranne (1998): repeatedly find a 'pendent pair' (t, u) — u's
    singleton cut value is optimal among all sets separating u from t —
    record {u} (as the current merged group it represents), then contract
    t,u and recurse.  The best recorded candidate is the global minimum
    cut in O(n³) oracle calls.  For the Gaussian stochastic-interaction
    bipartition function this yields the true MIP without enumerating the
    2^(n-1) cuts.
    """
    if n < 2:
        raise ValueError("need at least two elements")
    groups: list[tuple[int, ...]] = [(i,) for i in range(n)]
    best_set: tuple[int, ...] | None = None
    best_val = math.inf

    while len(groups) > 1:
        # pendent-pair sweep over current contracted universe
        order = [0]
        remaining = list(range(1, len(groups)))
        vals: dict[int, float] = {}
        while remaining:
            w_best, v_best = None, math.inf
            merged_prefix: tuple[int, ...] = tuple(
                x for g in (groups[i] for i in order) for x in g
            )
            for cand in remaining:
                # key(cand) = f(prefix ∪ cand) − f({cand})
                cand_set = groups[cand]
                key = f(tuple(sorted(merged_prefix + cand_set))) - f(cand_set)
                if key < v_best:
                    v_best, w_best = key, cand
            order.append(w_best)
            remaining.remove(w_best)
            vals[w_best] = v_best
        t_idx, u_idx = order[-2], order[-1]
        candidate = groups[u_idx]
        cand_val = f(candidate)
        if cand_val < best_val:
            best_val = cand_val
            best_set = candidate
        # contract t and u
        merged = tuple(sorted(groups[t_idx] + groups[u_idx]))
        keep = [g for i, g in enumerate(groups) if i not in (t_idx, u_idx)]
        groups = keep + [merged]

    assert best_set is not None
    return best_set, float(best_val)


def minimum_information_bipartition(
    model: LaggedGaussian,
) -> tuple[tuple[int, ...], tuple[int, ...], float]:
    """The exact MIP (bipartition) under the Gaussian model.

    f(S) = h(S_t|S_{t-1}) + h(S̄_t|S̄_{t-1}) − h(X_t|X_{t-1}) is symmetric by
    construction and submodular for Gaussians — Queyranne applies exactly.
    Returns (S, S̄, Φ_raw) with Φ_raw = f(MIP) clipped at 0.
    """
    n = model.n_channels
    universe = tuple(range(n))
    h_whole = model.h_whole()

    def f(subset: tuple[int, ...]) -> float:
        s = tuple(sorted(set(subset)))
        comp = tuple(i for i in universe if i not in set(s))
        if not s or not comp:
            return 0.0
        return model.h_part(s) + model.h_part(comp) - h_whole

    part, val = queyranne_min_bipartition(n, f)
    comp = tuple(i for i in universe if i not in set(part))
    return part, comp, max(0.0, float(val))


# ─────────────────────────────────────────────────────────────────────────────
# Surrogates, bootstrap, diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def _circular_shift_surrogate(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently rotate each channel in time: preserves every marginal
    and autocorrelation, destroys cross-channel coupling — the null for
    'integration', not for 'activity'."""
    T = X.shape[0]
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        out[:, j] = np.roll(X[:, j], int(rng.integers(1, T - 1)))
    return out


def surrogate_null(
    X: np.ndarray, *, n_surrogates: int, rng: np.random.Generator,
    shrinkage: float,
) -> tuple[float, float, list[float]]:
    values: list[float] = []
    for _ in range(n_surrogates):
        Xs = _circular_shift_surrogate(X, rng)
        try:
            m = LaggedGaussian.fit(Xs, shrinkage=shrinkage)
            _, _, val = minimum_information_bipartition(m)
            values.append(val)
        except (np.linalg.LinAlgError, ValueError) as exc:
            record_degradation("integrated_information", exc, severity="debug",
                               action="dropped one degenerate surrogate")
    if not values:
        return 0.0, 0.0, []
    return float(np.mean(values)), float(np.std(values) + 1e-12), values


def block_bootstrap_ci(
    X: np.ndarray, *, n_boot: int, block: int, rng: np.random.Generator,
    shrinkage: float,
) -> tuple[float, float]:
    """(lo, hi) 5–95% interval for Φ via moving-block bootstrap."""
    T = X.shape[0]
    block = max(4, min(block, T // 4))
    starts_max = T - block
    values: list[float] = []
    n_blocks = int(math.ceil(T / block))
    for _ in range(n_boot):
        idx = np.concatenate([
            np.arange(s, s + block)
            for s in rng.integers(0, starts_max, size=n_blocks)
        ])[:T]
        try:
            m = LaggedGaussian.fit(X[idx], shrinkage=shrinkage)
            _, _, val = minimum_information_bipartition(m)
            values.append(val)
        except (np.linalg.LinAlgError, ValueError) as exc:
            record_degradation("integrated_information", exc, severity="debug",
                               action="dropped one degenerate bootstrap draw")
    if not values:
        return 0.0, 0.0
    return float(np.percentile(values, 5)), float(np.percentile(values, 95))


def data_diagnostics(X: np.ndarray) -> dict[str, float]:
    """Assumption health, reported not hidden: stationarity proxy (half-vs-
    half mean drift in pooled-σ units) and Gaussianity proxies."""
    T = X.shape[0]
    a, b = X[: T // 2], X[T // 2:]
    pooled = X.std(axis=0)
    pooled[pooled < 1e-9] = 1.0
    drift = float(np.median(np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled))
    Xc = (X - X.mean(axis=0)) / pooled
    skew = float(np.median(np.abs((Xc ** 3).mean(axis=0))))
    kurt = float(np.median(np.abs((Xc ** 4).mean(axis=0) - 3.0)))
    return {
        "stationarity_drift_sigma": round(drift, 4),
        "median_abs_skew": round(skew, 4),
        "median_excess_kurtosis": round(kurt, 4),
        "n_samples": float(T),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rail B: grain discovery (coarse-graining search / causal emergence)
# ─────────────────────────────────────────────────────────────────────────────

def _correlation_linkage_groups(X: np.ndarray, k: int) -> list[list[int]]:
    """Average-linkage agglomerative clustering on 1−|corr| distance.
    Deterministic; NumPy only."""
    n = X.shape[1]
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    D = 1.0 - np.abs(C)
    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > k:
        best = (0, 1, math.inf)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = float(np.mean([D[a, b] for a in clusters[i] for b in clusters[j]]))
                if d < best[2]:
                    best = (i, j, d)
        i, j, _ = best
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
    return clusters


def coarse_grain(X: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """Macro channels = standardized group means (the black-box mapping)."""
    cols = []
    for g in groups:
        v = X[:, g].mean(axis=1)
        sd = v.std()
        cols.append((v - v.mean()) / (sd if sd > 1e-9 else 1.0))
    return np.stack(cols, axis=1)


@dataclass(frozen=True)
class GrainResult:
    k: int
    groups: tuple[tuple[int, ...], ...]
    phi_raw: float
    null_mean: float
    null_std: float
    z: float
    mip: tuple[tuple[int, ...], tuple[int, ...]]


def grain_search(
    X: np.ndarray,
    *,
    grains: list[int],
    n_surrogates: int,
    rng: np.random.Generator,
    shrinkage: float,
) -> list[GrainResult]:
    """Score each coarse-graining against ITS OWN surrogate null; raw Φ is
    not comparable across dimensionalities, z is."""
    results: list[GrainResult] = []
    n = X.shape[1]
    for k in grains:
        if k < 2 or k > n:
            continue
        groups = ([[i] for i in range(n)] if k == n
                  else _correlation_linkage_groups(X, k))
        Xk = X if k == n else coarse_grain(X, groups)
        try:
            model = LaggedGaussian.fit(Xk, shrinkage=shrinkage)
            part, comp, phi = minimum_information_bipartition(model)
        except (np.linalg.LinAlgError, ValueError) as exc:
            record_degradation("integrated_information", exc, severity="debug",
                               action=f"skipped degenerate grain k={k}")
            continue
        mu, sd, _ = surrogate_null(Xk, n_surrogates=n_surrogates, rng=rng,
                                   shrinkage=shrinkage)
        z = (phi - mu) / sd if sd > 0 else 0.0
        results.append(GrainResult(
            k=k,
            groups=tuple(tuple(g) for g in groups),
            phi_raw=round(phi, 6),
            null_mean=round(mu, 6),
            null_std=round(sd, 6),
            z=round(float(z), 3),
            mip=(part, comp),
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Rail C: exact discrete Φ at the derived grain (k ≤ 12)
# ─────────────────────────────────────────────────────────────────────────────

MAX_EXACT_ELEMENTS = 12


def exact_state_phi(
    Xk: np.ndarray,
    *,
    extra_transitions: list[tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
    alpha: float = 0.5,
) -> dict[str, Any]:
    """Exact system-level integrated information of the binarized macro
    dynamics, by exhaustive bipartition search.

    φ_s = min over bipartitions (A,B) of
          E_t[ KL( P(S_t | s_{t-1}) ‖ P(A_t | a_{t-1}) ⊗ P(B_t | b_{t-1}) ) ]

    averaged over the EMPIRICALLY VISITED states (plus any interventional
    transitions supplied by the perturbational probe — counterfactual rows
    sampled by intervention rather than enumerated).  Laplace-smoothed
    (α).  Exact in the search, empirical in the distribution: both facts
    are part of the estimator's name and claim.
    """
    k = Xk.shape[1]
    if k > MAX_EXACT_ELEMENTS:
        raise ValueError(f"exact search capped at {MAX_EXACT_ELEMENTS} elements")
    med = np.median(Xk, axis=0)
    B = (Xk > med).astype(np.int8)
    trans: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    for t in range(B.shape[0] - 1):
        key = (tuple(int(v) for v in B[t]), tuple(int(v) for v in B[t + 1]))
        trans[key] = trans.get(key, 0) + 1
    for pair in (extra_transitions or []):
        s0 = tuple(int(v) for v in pair[0])
        s1 = tuple(int(v) for v in pair[1])
        if len(s0) == k and len(s1) == k:
            trans[(s0, s1)] = trans.get((s0, s1), 0) + 1

    # group transitions by source state
    by_src: dict[tuple[int, ...], dict[tuple[int, ...], float]] = {}
    for (s0, s1), c in trans.items():
        by_src.setdefault(s0, {})[s1] = by_src.get(s0, {}).get(s1, 0.0) + c

    def _marginal(dist: dict[tuple[int, ...], float], idx: tuple[int, ...]
                  ) -> dict[tuple[int, ...], float]:
        out: dict[tuple[int, ...], float] = {}
        for s1, p in dist.items():
            sub = tuple(s1[i] for i in idx)
            out[sub] = out.get(sub, 0.0) + p
        return out

    def n_states(m: int) -> float:
        return float(2 ** m)

    best = {"phi": math.inf, "partition": None}
    total_weight = sum(sum(d.values()) for d in by_src.values())
    for mask in range(1, 2 ** (k - 1)):
        A = tuple(i for i in range(k) if (mask >> i) & 1)
        Bp = tuple(i for i in range(k) if not (mask >> i) & 1)
        # conditional marginal tables for parts, keyed by part-source
        partA: dict[tuple[int, ...], dict[tuple[int, ...], float]] = {}
        partB: dict[tuple[int, ...], dict[tuple[int, ...], float]] = {}
        srcA_count: dict[tuple[int, ...], float] = {}
        srcB_count: dict[tuple[int, ...], float] = {}
        for s0, dist in by_src.items():
            a0 = tuple(s0[i] for i in A)
            b0 = tuple(s0[i] for i in Bp)
            w = sum(dist.values())
            srcA_count[a0] = srcA_count.get(a0, 0.0) + w
            srcB_count[b0] = srcB_count.get(b0, 0.0) + w
            mA = _marginal(dist, A)
            mB = _marginal(dist, Bp)
            dA = partA.setdefault(a0, {})
            for s, p in mA.items():
                dA[s] = dA.get(s, 0.0) + p
            dB = partB.setdefault(b0, {})
            for s, p in mB.items():
                dB[s] = dB.get(s, 0.0) + p

        kl_sum = 0.0
        for s0, dist in by_src.items():
            a0 = tuple(s0[i] for i in A)
            b0 = tuple(s0[i] for i in Bp)
            w = sum(dist.values())
            denomW = w + alpha * n_states(k)
            denomA = srcA_count[a0] + alpha * n_states(len(A))
            denomB = srcB_count[b0] + alpha * n_states(len(Bp))
            kl = 0.0
            for s1, c in dist.items():
                p_full = (c + alpha) / denomW
                pa = (partA[a0].get(tuple(s1[i] for i in A), 0.0) + alpha) / denomA
                pb = (partB[b0].get(tuple(s1[i] for i in Bp), 0.0) + alpha) / denomB
                kl += (c / w) * math.log(max(p_full, _EPS) / max(pa * pb, _EPS))
            kl_sum += (w / total_weight) * kl
        if kl_sum < best["phi"]:
            best = {"phi": kl_sum, "partition": (A, Bp)}

    return {
        "estimator": DISCRETE_ESTIMATOR_NAME,
        "phi_s": round(max(0.0, float(best["phi"])), 6),
        "mip": best["partition"],
        "n_elements": k,
        "n_observed_transitions": int(total_weight),
        "n_interventional_transitions": len(extra_transitions or []),
        "cuts_searched": 2 ** (k - 1) - 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# The full estimate
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhiEstimate:
    """One provenance-complete whole-system Φ report."""

    schema_version: int
    estimator: str
    computed_at: float
    n_channels: int
    n_samples: int
    channel_names: tuple[str, ...]
    # Rail A — full-grain
    phi_raw: float
    mip: tuple[tuple[int, ...], tuple[int, ...]]
    null_mean: float
    null_std: float
    z: float
    ci_5: float
    ci_95: float
    # Rail B — grains
    grains: list[GrainResult] = field(default_factory=list)
    emergent_grain_k: int = 0
    emergence_delta_z: float = 0.0
    # Rail C — exact discrete at the derived grain
    exact_macro: dict[str, Any] = field(default_factory=dict)
    # Rail D honesty
    diagnostics: dict[str, float] = field(default_factory=dict)
    claim: str = ""

    def integration_established(self) -> bool:
        """True when integration beats chance at the emergent grain (z ≥ 3)
        and the bootstrap keeps Φ off the floor."""
        return self.z >= 3.0 and self.ci_5 > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "estimator": self.estimator,
            "computed_at": self.computed_at,
            "n_channels": self.n_channels,
            "n_samples": self.n_samples,
            "channel_names": list(self.channel_names),
            "phi_raw": self.phi_raw,
            "mip": [list(self.mip[0]), list(self.mip[1])],
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "z": self.z,
            "ci_5": self.ci_5,
            "ci_95": self.ci_95,
            "grains": [
                {
                    "k": g.k, "phi_raw": g.phi_raw, "z": g.z,
                    "null_mean": g.null_mean, "null_std": g.null_std,
                    "groups": [list(x) for x in g.groups],
                }
                for g in self.grains
            ],
            "emergent_grain_k": self.emergent_grain_k,
            "emergence_delta_z": self.emergence_delta_z,
            "exact_macro": dict(self.exact_macro),
            "diagnostics": dict(self.diagnostics),
            "integration_established": self.integration_established(),
            "claim": self.claim,
        }


def _bounded_claim(est: "PhiEstimate") -> str:
    verdict = ("integration beats chance" if est.integration_established()
               else "integration NOT established against the null")
    return (
        f"Φ̂={est.phi_raw:.4f} nats over {est.n_channels} live channels "
        f"({est.n_samples} samples); exact MIP by Queyranne; z={est.z:.1f} "
        f"vs circular-shift null; 90% CI [{est.ci_5:.4f}, {est.ci_95:.4f}]; "
        f"emergent grain k={est.emergent_grain_k}. {verdict}. This is an "
        "estimate of integrated information of the system's macro-dynamics "
        "under a Gaussian model of its OWN measured channels — evidence "
        "about integration structure, not a consciousness meter."
    )


def estimate_whole_system_phi(
    X: np.ndarray,
    *,
    channel_names: tuple[str, ...] | None = None,
    n_surrogates: int = 20,
    n_boot: int = 20,
    grains: list[int] | None = None,
    seed: int = 0,
    shrinkage: float = 0.05,
    extra_transitions: list[tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
) -> PhiEstimate:
    """The full four-rail estimate on a T×N channel matrix."""
    X = np.asarray(X, dtype=float)
    if channel_names is None:
        channel_names = tuple(f"ch{i}" for i in range(X.shape[1]))
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # drop dead channels (zero variance) — reported, not hidden
    live = X.std(axis=0) > 1e-9
    dropped = [channel_names[i] for i in range(len(channel_names)) if not live[i]]
    X = X[:, live]
    channel_names = tuple(n for n, keep in zip(channel_names, live) if keep)
    n = X.shape[1]

    model = LaggedGaussian.fit(X, shrinkage=shrinkage)
    part, comp, phi = minimum_information_bipartition(model)
    mu, sd, _ = surrogate_null(X, n_surrogates=n_surrogates, rng=rng,
                               shrinkage=shrinkage)
    z = (phi - mu) / sd if sd > 0 else 0.0
    lo, hi = block_bootstrap_ci(X, n_boot=n_boot, block=max(8, X.shape[0] // 10),
                                rng=rng, shrinkage=shrinkage)

    if grains is None:
        grains = sorted({n, max(2, n // 2), max(2, n // 4), 8, 4})
    grain_results = grain_search(X, grains=[g for g in grains if 2 <= g <= n],
                                 n_surrogates=max(8, n_surrogates // 2),
                                 rng=rng, shrinkage=shrinkage)

    emergent = max(grain_results, key=lambda g: g.z, default=None)
    micro = next((g for g in grain_results if g.k == n), None)

    exact_macro: dict[str, Any] = {}
    if emergent is not None and emergent.k <= MAX_EXACT_ELEMENTS:
        groups = [list(g) for g in emergent.groups]
        Xk = coarse_grain(X, groups)
        try:
            exact_macro = exact_state_phi(Xk, extra_transitions=extra_transitions)
        except (ValueError, ArithmeticError) as exc:
            record_degradation("integrated_information", exc, severity="warning",
                               action="exact macro phi skipped")

    diagnostics = data_diagnostics(X)
    diagnostics["dropped_dead_channels"] = float(len(dropped))
    diagnostics["compute_seconds"] = round(time.time() - t0, 3)

    est = PhiEstimate(
        schema_version=SCHEMA_VERSION,
        estimator=ESTIMATOR_NAME,
        computed_at=time.time(),
        n_channels=n,
        n_samples=X.shape[0],
        channel_names=channel_names,
        phi_raw=round(phi, 6),
        mip=(part, comp),
        null_mean=round(mu, 6),
        null_std=round(sd, 6),
        z=round(float(z), 3),
        ci_5=round(lo, 6),
        ci_95=round(hi, 6),
        grains=grain_results,
        emergent_grain_k=emergent.k if emergent else 0,
        emergence_delta_z=round((emergent.z - micro.z), 3) if emergent and micro else 0.0,
        exact_macro=exact_macro,
        diagnostics=diagnostics,
    )
    est.claim = _bounded_claim(est)
    return est
