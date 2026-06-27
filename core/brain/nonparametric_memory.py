"""Non-parametric memory — growable capacity for a fixed model (kNN-LM, made Aura's).

The honest way to increase a frozen model's capacity without enlarging its weights:
keep a datastore of (hidden-state key -> next token) pairs and, at generation time,
interpolate the nearest neighbors' tokens into the model's own next-token distribution.
A small model + a large datastore matches a much bigger model on knowledge, because the
datastore is *non-parametric* capacity that grows without retraining.

This is NOT prompt-level RAG — it operates on the next-token distribution itself, so the
information lives in a reachable store instead of in parameters. It is fail-open logit
reweighting (same risk class as the steering hooks, NOT weight mutation).

Aura-specific novelty:
  * the datastore is fed only TRUSTED knowledge (verifier-clean answers, beliefs) — sound,
    not a raw corpus that can teach errors;
  * the interpolation weight λ is Φ-ADAPTIVE — high free-energy (model surprised) + a
    confident neighbor ⇒ trust memory more; confident model ⇒ trust the weights;
  * entries carry affect/recency weight (the vault's "gravity") applied at the token level.

Bounded by construction (max entries, weight×recency eviction). Pure numpy; FAISS optional.
The datastore + kNN + adaptive-λ interpolation are built and tested here. Populating keys
from real model hidden states and applying per-token interpolation inside MLX generation is
the documented seam (`apply_to_logits`), flag-gated (AURA_NONPARAMETRIC_MEMORY).
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.NonParametricMemory")

# Φ thresholds mirror phi_consciousness so "is cognition integrated enough to trust
# non-parametric recall" matches the rest of the system.
PHI_DORMANT = 0.05
PHI_DELIBERATE = 0.55


def _flag_on(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "on", "yes", "enabled"}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class Neighbor:
    token_id: int
    token: str
    distance: float
    weight: float


class NonParametricMemory:
    """Datastore of (key, next-token) pairs + kNN + Φ-adaptive distribution interpolation."""

    _ERRORS = (OSError, ValueError, TypeError, json.JSONDecodeError)

    def __init__(
        self,
        dim: int,
        path: str | Path | None = None,
        *,
        max_entries: int = 200_000,
        base_lambda: float = 0.25,
        max_lambda: float = 0.7,
        dist_scale: float = 1.0,
    ) -> None:
        self._dim = int(dim)
        self._path = Path(path or os.path.expanduser("~/.aura/data/runtime/nonparametric_memory"))
        self._max = max(64, int(max_entries))
        self._base_lambda = float(base_lambda)
        self._max_lambda = float(max_lambda)
        self._dist_scale = max(1e-6, float(dist_scale))
        self._lock = threading.RLock()
        self._keys: np.ndarray = np.zeros((0, self._dim), dtype=np.float32)
        self._token_ids: list[int] = []
        self._tokens: list[str] = []
        self._weights: list[float] = []
        self._ts: list[float] = []
        self._stats = {"added": 0, "queried": 0, "evicted": 0, "interpolated": 0, "fallthrough": 0}
        self._load()

    # ── population ────────────────────────────────────────────────────────────
    def add(self, key: np.ndarray, token_id: int, token: str = "", *, weight: float = 1.0) -> bool:
        k = np.asarray(key, dtype=np.float32).reshape(-1)
        if k.shape[0] != self._dim or not np.all(np.isfinite(k)):
            return False
        with self._lock:
            self._keys = np.vstack([self._keys, k[None, :]]) if self._keys.size else k[None, :].copy()
            self._token_ids.append(int(token_id))
            self._tokens.append(str(token))
            self._weights.append(float(weight))
            self._ts.append(time.time())
            self._stats["added"] += 1
            self._evict_if_needed()
        return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._token_ids)

    # ── retrieval ─────────────────────────────────────────────────────────────
    def query(self, key: np.ndarray, k: int = 8) -> list[Neighbor]:
        q = np.asarray(key, dtype=np.float32).reshape(-1)
        with self._lock:
            n = len(self._token_ids)
            if n == 0 or q.shape[0] != self._dim:
                return []
            self._stats["queried"] += 1
            diffs = self._keys - q[None, :]
            dists = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
            kk = min(int(k), n)
            idx = np.argpartition(dists, kk - 1)[:kk]
            idx = idx[np.argsort(dists[idx])]
            return [
                Neighbor(self._token_ids[i], self._tokens[i], float(dists[i]), float(self._weights[i]))
                for i in idx
            ]

    def knn_probs(self, neighbors: list[Neighbor], *, temperature: float = 1.0) -> dict[int, float]:
        """kNN-LM distribution: softmax over -distance, weighted by entry gravity, per token."""
        if not neighbors:
            return {}
        t = max(1e-6, float(temperature)) * self._dist_scale
        logits = np.array([-(nb.distance / t) for nb in neighbors], dtype=np.float64)
        logits -= logits.max()
        w = np.array([max(0.0, nb.weight) for nb in neighbors], dtype=np.float64)
        ex = np.exp(logits) * (w if w.sum() > 0 else 1.0)
        total = ex.sum()
        if total <= 0:
            return {}
        probs: dict[int, float] = {}
        for nb, p in zip(neighbors, ex / total):
            probs[nb.token_id] = probs.get(nb.token_id, 0.0) + float(p)
        return probs

    # ── Φ-adaptive interpolation weight ─────────────────────────────────────────
    def adaptive_lambda(
        self,
        neighbors: list[Neighbor],
        *,
        phi: float | None = None,
        free_energy: float | None = None,
    ) -> float:
        """How much to trust non-parametric memory vs. the model's own weights → [0, max_lambda].

        Closer neighbors (confident recall) and higher free-energy (surprised model) raise λ;
        low Φ (fragmented cognition) caps it. No neighbors ⇒ 0 (pure model).
        """
        if not neighbors:
            return 0.0
        nearest = min(nb.distance for nb in neighbors)
        confidence = math.exp(-nearest / self._dist_scale)          # 1 at d=0 → 0 far away
        fe = _clamp(float(free_energy), 0.0, 1.0) if free_energy is not None else 0.5
        lam = self._base_lambda * (0.5 + 0.5 * confidence) * (0.6 + 0.8 * fe)
        if phi is not None:
            if phi < PHI_DORMANT:
                return 0.0                                          # too fragmented to trust recall
            cap = _clamp(0.3 + 0.7 * (phi / PHI_DELIBERATE), 0.0, 1.0)
            lam *= cap
        return _clamp(lam, 0.0, self._max_lambda)

    def interpolate(
        self,
        lm_probs: dict[int, float],
        query_key: np.ndarray,
        *,
        k: int = 8,
        temperature: float = 1.0,
        phi: float | None = None,
        free_energy: float | None = None,
        lam_override: float | None = None,
    ) -> dict[int, float]:
        """Blend the model's next-token probs with kNN recall: p = λ·p_kNN + (1-λ)·p_LM.

        Fail-open: returns ``lm_probs`` unchanged when there are no neighbors or λ≈0, so
        generation never degrades below the raw model.
        """
        neighbors = self.query(query_key, k=k)
        if not neighbors:
            with self._lock:
                self._stats["fallthrough"] += 1
            return dict(lm_probs)
        lam = lam_override if lam_override is not None else self.adaptive_lambda(
            neighbors, phi=phi, free_energy=free_energy
        )
        if lam <= 1e-6:
            with self._lock:
                self._stats["fallthrough"] += 1
            return dict(lm_probs)
        knn = self.knn_probs(neighbors, temperature=temperature)
        blended: dict[int, float] = {}
        for tok in set(lm_probs) | set(knn):
            blended[tok] = (1.0 - lam) * float(lm_probs.get(tok, 0.0)) + lam * float(knn.get(tok, 0.0))
        s = sum(blended.values())
        if s > 0:
            blended = {t: p / s for t, p in blended.items()}
        with self._lock:
            self._stats["interpolated"] += 1
        return blended

    def apply_to_logits(
        self, logits: np.ndarray, query_key: np.ndarray, **kw: Any
    ) -> np.ndarray:
        """Seam for the live MLX generation loop: reweight a full logit vector in-place-safe.

        Converts logits→probs over the top candidates, interpolates with recall, writes back
        log-probs. Flag-gated; fail-open to the original logits. (Live per-token wiring inside
        the MLX worker is the remaining, latency-validated step.)
        """
        if not _flag_on("AURA_NONPARAMETRIC_MEMORY"):
            return logits
        try:
            arr = np.asarray(logits, dtype=np.float64).reshape(-1)
            top = np.argpartition(arr, -min(64, arr.shape[0]))[-min(64, arr.shape[0]):]
            shift = arr[top] - arr[top].max()
            ex = np.exp(shift)
            lm_probs = {int(t): float(p) for t, p in zip(top, ex / ex.sum())}
            blended = self.interpolate(lm_probs, query_key, **kw)
            out = arr.copy()
            for t, p in blended.items():
                out[int(t)] = math.log(max(p, 1e-12))
            return out.astype(logits.dtype if hasattr(logits, "dtype") else np.float32)
        except (ValueError, TypeError, FloatingPointError, OverflowError) as exc:
            record_degradation("nonparametric_memory_logits", exc)
            return logits

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {**self._stats, "entries": len(self._token_ids), "dim": self._dim}

    # ── internals ───────────────────────────────────────────────────────────────
    def _evict_if_needed(self) -> None:
        n = len(self._token_ids)
        if n <= self._max:
            return
        now = time.time()
        # gravity = weight × recency; evict the lowest-gravity entries.
        gravity = np.array(
            [self._weights[i] * math.exp(-(now - self._ts[i]) / (14 * 24 * 3600.0)) for i in range(n)]
        )
        keep = np.argsort(gravity)[-(self._max):]
        keep.sort()
        self._keys = self._keys[keep]
        self._token_ids = [self._token_ids[i] for i in keep]
        self._tokens = [self._tokens[i] for i in keep]
        self._weights = [self._weights[i] for i in keep]
        self._ts = [self._ts[i] for i in keep]
        self._stats["evicted"] += n - len(keep)

    def _load(self) -> None:
        keys_p, meta_p = self._path.with_suffix(".keys.npy"), self._path.with_suffix(".meta.json")
        if not (keys_p.exists() and meta_p.exists()):
            return
        try:
            keys = np.load(keys_p)
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if keys.shape[1] != self._dim:
                return  # dim changed (e.g. new base model) → start fresh, never mix spaces
            with self._lock:
                self._keys = keys.astype(np.float32)
                self._token_ids = list(meta.get("token_ids", []))
                self._tokens = list(meta.get("tokens", []))
                self._weights = list(meta.get("weights", []))
                self._ts = list(meta.get("ts", []))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            record_degradation("nonparametric_memory_load", exc)

    def persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                np.save(self._path.with_suffix(".keys.npy"), self._keys)
                meta = {
                    "schema_version": 1, "dim": self._dim, "saved_at": time.time(),
                    "token_ids": self._token_ids, "tokens": self._tokens,
                    "weights": self._weights, "ts": self._ts,
                }
                self._path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
        except self._ERRORS as exc:
            record_degradation("nonparametric_memory_persist", exc)


_singleton: NonParametricMemory | None = None
_lock = threading.Lock()


def get_nonparametric_memory(dim: int = 0) -> NonParametricMemory | None:
    """Process-wide datastore (created on first call with a real model dim)."""
    global _singleton
    if _singleton is None:
        if dim <= 0:
            return None
        with _lock:
            if _singleton is None:
                _singleton = NonParametricMemory(dim)
    return _singleton


def reset_nonparametric_memory() -> None:
    global _singleton
    with _lock:
        _singleton = None
