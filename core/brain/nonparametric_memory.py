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
import tempfile
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
    # Anisotropy-corrected similarity in [-1, 1]: cosine between the query
    # and this key AFTER subtracting the running mean of the hidden space.
    # Raw last-token hidden states share a dominant common direction —
    # measured on the live 1.5B: UNRELATED prompts score raw cos 0.81-0.93
    # while identical prompts score 1.0000, so raw-cos gates cannot
    # separate them; mean-centered cosines put unrelated prompts at ≤0.36.
    # Until the mean estimate is ready, this falls back to raw cosine and
    # the gate compensates with a much higher threshold.
    similarity: float = -1.0
    # Position of this entry in the store — lets generation loops apply the
    # anti-stutter guard (the same entry may not fire on consecutive steps).
    index: int = -1


class NonParametricMemory:
    """Datastore of (key, next-token) pairs + kNN + Φ-adaptive distribution interpolation."""

    _ERRORS = (OSError, ValueError, TypeError, json.JSONDecodeError)

    def __init__(
        self,
        dim: int,
        path: str | Path | None = None,
        *,
        max_entries: int = 4_096,
        base_lambda: float = 0.25,
        max_lambda: float = 0.7,
        dist_scale: float = 1.0,
    ) -> None:
        self._dim = int(dim)
        if self._dim <= 0:
            raise ValueError("non-parametric memory dimension must be positive")
        default_path = f"~/.aura/data/runtime/nonparametric_memory_{self._dim}"
        self._path = Path(path or os.path.expanduser(default_path))
        self._max = max(64, int(max_entries))
        self._base_lambda = float(base_lambda)
        self._max_lambda = float(max_lambda)
        self._dist_scale = max(1e-6, float(dist_scale))
        self._lock = threading.RLock()
        self._capacity = min(64, self._max)
        self._size = 0
        self._keys = np.empty((self._capacity, self._dim), dtype=np.float32)
        self._key_norms = np.empty(self._capacity, dtype=np.float32)
        self._token_ids: list[int] = []
        self._tokens: list[str] = []
        self._weights: list[float] = []
        self._ts: list[float] = []
        self._stats = {"added": 0, "queried": 0, "evicted": 0, "interpolated": 0, "fallthrough": 0}
        # Running mean of QUERY keys — the estimator of the hidden space's
        # anisotropic common direction. Every query is a sample; persisted
        # with the store so the correction survives restarts.
        self._query_mu = np.zeros(self._dim, dtype=np.float32)
        self._query_mu_n = 0
        self._load()

    # How many query samples the mean needs before centered similarity is
    # trusted; below this the raw-cosine fallback gate applies.
    MU_READY_N = 16
    # Gate thresholds per mode (measured, see Neighbor.similarity).
    MIN_SIM_CENTERED = 0.60
    MIN_SIM_RAW = 0.98

    def similarity_ready(self) -> bool:
        return self._query_mu_n >= self.MU_READY_N

    def min_similarity(self) -> float:
        """The confident-recall gate matched to the active similarity mode."""
        return self.MIN_SIM_CENTERED if self.similarity_ready() else self.MIN_SIM_RAW

    # ── population ────────────────────────────────────────────────────────────
    def add(self, key: np.ndarray, token_id: int, token: str = "", *, weight: float = 1.0) -> bool:
        k = np.asarray(key, dtype=np.float32).reshape(-1)
        if k.shape[0] != self._dim or not np.all(np.isfinite(k)):
            return False
        with self._lock:
            if self._size >= self._max:
                self._evict_one_for_incoming()
            self._ensure_capacity(self._size + 1)
            self._keys[self._size] = k
            self._key_norms[self._size] = float(np.dot(k, k))
            self._token_ids.append(int(token_id))
            self._tokens.append(str(token))
            self._weights.append(max(0.0, float(weight)))
            self._ts.append(time.time())
            self._size += 1
            self._stats["added"] += 1
        return True

    def __len__(self) -> int:
        with self._lock:
            return self._size

    # ── retrieval ─────────────────────────────────────────────────────────────
    def query(self, key: np.ndarray, k: int = 8) -> list[Neighbor]:
        q = np.asarray(key, dtype=np.float32).reshape(-1)
        with self._lock:
            n = self._size
            if n == 0 or q.shape[0] != self._dim:
                return []
            self._stats["queried"] += 1
            # Every query is a sample of the hidden space: feed the running
            # mean that powers the anisotropy correction (cumulative mean up
            # to 256 samples, then a slow EMA that tracks drift).
            if np.all(np.isfinite(q)):
                if self._query_mu_n < 256:
                    self._query_mu_n += 1
                    self._query_mu += (q - self._query_mu) / float(self._query_mu_n)
                else:
                    self._query_mu += 0.005 * (q - self._query_mu)
            # ||x-q||^2 = ||x||^2 + ||q||^2 - 2x.q avoids allocating an
            # n-by-hidden-dimension difference matrix on every decode step.
            dists_sq = self._key_norms[:n] + float(np.dot(q, q))
            dists_sq = dists_sq - 2.0 * (self._keys[:n] @ q)
            dists = np.sqrt(np.maximum(dists_sq, 0.0))
            kk = min(max(1, int(k)), n)
            idx = np.argpartition(dists, kk - 1)[:kk]
            idx = idx[np.argsort(dists[idx])]
            similarities = self._similarities_for(q, idx)
            return [
                Neighbor(
                    self._token_ids[i],
                    self._tokens[i],
                    float(dists[i]),
                    float(self._weights[i]),
                    similarity=float(sim),
                    index=int(i),
                )
                for i, sim in zip(idx, similarities)
            ]

    def _similarities_for(self, q: np.ndarray, idx: Any) -> list[float]:
        """Gate-ready similarity per neighbor (caller holds the lock).

        Centered cosine once the mean is ready; raw cosine before that.
        O(k·dim) — computed only for the k returned neighbors.
        """
        centered = self._query_mu_n >= self.MU_READY_N
        mu = self._query_mu if centered else None
        sims: list[float] = []
        for i in idx:
            key_vec = self._keys[int(i)]
            if mu is not None:
                a = q - mu
                b = key_vec - mu
            else:
                a = q
                b = key_vec
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na <= 1e-8 or nb <= 1e-8:
                sims.append(-1.0)
                continue
            sims.append(float(np.dot(a, b)) / (na * nb))
        return sims

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
        # Anisotropy-corrected gate: below the confident-recall threshold the
        # blend contributes NOTHING (measured: raw hidden-state cosine cannot
        # separate unrelated prompts — the July proof caught the old
        # distance-only confidence corrupting even unrelated generations).
        best_sim = max(nb.similarity for nb in neighbors)
        min_sim = self.min_similarity()
        if best_sim < min_sim:
            return 0.0
        confidence = _clamp((best_sim - min_sim) / max(1e-6, 1.0 - min_sim), 0.0, 1.0)
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
        # Per-neighbor confidence filter: entries below the gate must not
        # leak probability mass into the kNN distribution. Measured failure
        # (July proof): with the soft filter, digits from OTHER facts at raw
        # cos ~0.93 outvoted the exact-match entry and corrupted recall.
        min_sim = self.min_similarity()
        neighbors = [nb for nb in neighbors if nb.similarity >= min_sim]
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
            original = np.asarray(logits)
            arr = np.asarray(logits, dtype=np.float64).reshape(-1)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                return logits
            neighbors = self.query(query_key, k=int(kw.pop("k", 8)))
            if not neighbors:
                return logits
            lam = kw.pop("lam_override", None)
            if lam is None:
                lam = self.adaptive_lambda(
                    neighbors,
                    phi=kw.pop("phi", None),
                    free_energy=kw.pop("free_energy", None),
                )
            lam = _clamp(float(lam), 0.0, self._max_lambda)
            if lam <= 1e-6:
                return logits
            knn = self.knn_probs(
                neighbors,
                temperature=float(kw.pop("temperature", 1.0)),
            )
            if not knn:
                return logits
            max_logit = float(np.max(arr))
            log_z = max_logit + math.log(float(np.exp(arr - max_logit).sum()))
            out = (arr - log_z) + math.log1p(-lam)
            log_lam = math.log(lam)
            for token_id, probability in knn.items():
                if 0 <= int(token_id) < out.size and probability > 0.0:
                    out[int(token_id)] = np.logaddexp(
                        out[int(token_id)],
                        log_lam + math.log(float(probability)),
                    )
            with self._lock:
                self._stats["interpolated"] += 1
            return out.reshape(original.shape).astype(
                original.dtype if hasattr(original, "dtype") else np.float32
            )
        except (ValueError, TypeError, FloatingPointError, OverflowError) as exc:
            record_degradation("nonparametric_memory_logits", exc)
            return logits

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "entries": self._size,
                "dim": self._dim,
                "max_entries": self._max,
                "allocated_bytes": int(self._keys.nbytes + self._key_norms.nbytes),
            }

    # ── internals ───────────────────────────────────────────────────────────────
    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        new_capacity = min(self._max, max(needed, self._capacity * 2))
        keys = np.empty((new_capacity, self._dim), dtype=np.float32)
        norms = np.empty(new_capacity, dtype=np.float32)
        keys[: self._size] = self._keys[: self._size]
        norms[: self._size] = self._key_norms[: self._size]
        self._keys = keys
        self._key_norms = norms
        self._capacity = new_capacity

    def _evict_one_for_incoming(self) -> None:
        n = self._size
        if n == 0:
            return
        now = time.time()
        gravity = np.array(
            [self._weights[i] * math.exp(-(now - self._ts[i]) / (14 * 24 * 3600.0)) for i in range(n)]
        )
        drop = int(np.argmin(gravity))
        if drop < n - 1:
            self._keys[drop : n - 1] = self._keys[drop + 1 : n]
            self._key_norms[drop : n - 1] = self._key_norms[drop + 1 : n]
        for values in (self._token_ids, self._tokens, self._weights, self._ts):
            values.pop(drop)
        self._size -= 1
        self._stats["evicted"] += 1

    def _load(self) -> None:
        keys_p, meta_p = self._path.with_suffix(".keys.npy"), self._path.with_suffix(".meta.json")
        if not (keys_p.exists() and meta_p.exists()):
            return
        try:
            keys = np.load(keys_p)
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if keys.ndim != 2 or keys.shape[1] != self._dim:
                return  # dim changed (e.g. new base model) → start fresh, never mix spaces
            token_ids = list(meta.get("token_ids", []))
            tokens = list(meta.get("tokens", []))
            weights = list(meta.get("weights", []))
            timestamps = list(meta.get("ts", []))
            count = min(len(keys), len(token_ids), len(tokens), len(weights), len(timestamps))
            if count <= 0 or len({len(keys), len(token_ids), len(tokens), len(weights), len(timestamps)}) != 1:
                raise ValueError("non-parametric memory persistence metadata is inconsistent")
            count = min(count, self._max)
            with self._lock:
                self._capacity = max(64, count)
                self._capacity = min(self._capacity, self._max)
                self._keys = np.empty((self._capacity, self._dim), dtype=np.float32)
                self._key_norms = np.empty(self._capacity, dtype=np.float32)
                self._keys[:count] = keys[-count:].astype(np.float32)
                self._key_norms[:count] = np.einsum(
                    "ij,ij->i", self._keys[:count], self._keys[:count]
                )
                self._token_ids = [int(value) for value in token_ids[-count:]]
                self._tokens = [str(value) for value in tokens[-count:]]
                self._weights = [max(0.0, float(value)) for value in weights[-count:]]
                self._ts = [float(value) for value in timestamps[-count:]]
                self._size = count
                saved_mu = meta.get("query_mu")
                if isinstance(saved_mu, list) and len(saved_mu) == self._dim:
                    self._query_mu = np.asarray(saved_mu, dtype=np.float32)
                    self._query_mu_n = max(0, int(meta.get("query_mu_n", 0) or 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            record_degradation("nonparametric_memory_load", exc)

    def persist(self) -> bool:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                keys = self._keys[: self._size].copy()
                meta = {
                    "schema_version": 2, "dim": self._dim, "saved_at": time.time(),
                    "token_ids": list(self._token_ids), "tokens": list(self._tokens),
                    "weights": list(self._weights), "ts": list(self._ts),
                    "query_mu": [float(v) for v in self._query_mu],
                    "query_mu_n": int(self._query_mu_n),
                }
            keys_path = self._path.with_suffix(".keys.npy")
            meta_path = self._path.with_suffix(".meta.json")
            with tempfile.NamedTemporaryFile(
                dir=self._path.parent, suffix=".npy", delete=False
            ) as handle:
                np.save(handle, keys)
                temporary_keys = Path(handle.name)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._path.parent,
                suffix=".json", delete=False
            ) as handle:
                json.dump(meta, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
                temporary_meta = Path(handle.name)
            os.replace(temporary_keys, keys_path)
            os.replace(temporary_meta, meta_path)
            return True
        except self._ERRORS as exc:
            record_degradation("nonparametric_memory_persist", exc)
            return False


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
    elif dim > 0 and _singleton._dim != int(dim):
        logger.warning(
            "Non-parametric memory dimension mismatch (%d requested, %d active); refusing cross-model reuse.",
            int(dim),
            _singleton._dim,
        )
        return None
    return _singleton


def reset_nonparametric_memory() -> None:
    global _singleton
    with _lock:
        _singleton = None
