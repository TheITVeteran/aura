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

import hashlib
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


# Below this, a centered vector has no usable direction: the cosine would be
# noise divided by ~zero. Queries and keys that degenerate this far are
# reported as unknown rather than ranked.
_DEGENERATE_NORM = 1e-9


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
        self._content_generation = 0
        self._identity_cache: tuple[int, dict[str, Any]] | None = None
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
            # Reject non-finite/negative weights: a NaN weight would poison the
            # kNN probability mass it later contributes to.
            try:
                w = float(weight)
            except (TypeError, ValueError):
                w = 1.0
            self._weights.append(w if math.isfinite(w) and w >= 0.0 else 0.0)
            self._ts.append(time.time())
            self._size += 1
            self._stats["added"] += 1
            self._content_generation += 1
            self._identity_cache = None
        return True

    def __len__(self) -> int:
        with self._lock:
            return self._size

    # ── retrieval ─────────────────────────────────────────────────────────────
    def query(self, key: np.ndarray, k: int = 8) -> list[Neighbor]:
        q = np.asarray(key, dtype=np.float32).reshape(-1)
        # Reject a non-finite query outright: searching with NaN/inf produces
        # NaN distances and garbage neighbor selection, and must never poison
        # the persisted running mean below.
        if q.shape[0] != self._dim or not np.all(np.isfinite(q)):
            return []
        with self._lock:
            n = self._size
            if n == 0:
                return []
            self._stats["queried"] += 1
            # Every query is a sample of the hidden space: feed the running
            # mean that powers the anisotropy correction (cumulative mean up
            # to 256 samples, then a slow EMA that tracks drift).
            if self._query_mu_n < 256:
                self._query_mu_n += 1
                self._query_mu += (q - self._query_mu) / float(self._query_mu_n)
            else:
                self._query_mu += 0.005 * (q - self._query_mu)
            # CP126 d5cc1faf. Candidates used to be selected by raw
            # Euclidean distance while the confidence gate judged them by
            # mean-centered cosine — two metrics that disagree precisely
            # where it matters. Raw hidden states share a dominant common
            # direction (measured on the live 1.5B: UNRELATED prompts at raw
            # cos 0.81-0.93), so ||x-q|| is dominated by vector norm and that
            # shared direction. The semantically strongest centered-cosine
            # neighbour could therefore be ranked outside the top k and never
            # reach the gate at all — and no gate threshold can repair a
            # candidate set that already excluded the right answer.
            #
            # Selection now uses the SAME metric as the gate, computed for
            # every entry without materialising an n-by-dim difference
            # matrix:
            #
            #   <q-mu, x-mu> = x.q - x.mu - q.mu + mu.mu
            #   ||x-mu||^2   = ||x||^2 - 2 x.mu + ||mu||^2
            #
            # so two matrix-vector products (keys@q, keys@mu) give exact
            # centered cosines for all n — the same asymptotic cost as the
            # distance scan it replaces. Before the mean estimate is ready mu
            # is zero and this reduces exactly to raw cosine.
            keys = self._keys[:n]
            centered = self._query_mu_n >= self.MU_READY_N
            mu = self._query_mu if centered else np.zeros_like(q)
            kq = keys @ q
            km = keys @ mu
            qm = float(np.dot(q, mu))
            mm = float(np.dot(mu, mu))
            numerator = kq - km - qm + mm
            key_centered_sq = np.maximum(self._key_norms[:n] - 2.0 * km + mm, 0.0)
            q_centered_norm = math.sqrt(max(float(np.dot(q, q)) - 2.0 * qm + mm, 0.0))
            if q_centered_norm <= _DEGENERATE_NORM:
                # The query sits on the common direction itself: after
                # centering it has no direction left to compare, so every
                # cosine here would be numerical noise amplified by a
                # near-zero denominator. Report unknown (-1) and let the
                # confidence gate refuse rather than inventing a ranking.
                self._stats["degenerate_query"] = (
                    int(self._stats.get("degenerate_query", 0)) + 1
                )
                return []
            denominator = np.sqrt(key_centered_sq) * q_centered_norm
            sims_all = np.where(
                denominator > _DEGENERATE_NORM,
                numerator / np.maximum(denominator, _DEGENERATE_NORM),
                -1.0,
            )
            kk = min(max(1, int(k)), n)
            # Highest similarity first — argpartition on the negated scores.
            idx = np.argpartition(-sims_all, kk - 1)[:kk]
            idx = idx[np.argsort(-sims_all[idx])]
            # Euclidean distance is still REPORTED (callers and telemetry use
            # it), computed only for the k that were selected.
            selected = keys[idx]
            diff_sq = (
                self._key_norms[:n][idx] + float(np.dot(q, q)) - 2.0 * (selected @ q)
            )
            dists = np.sqrt(np.maximum(diff_sq, 0.0))
            similarities = sims_all[idx]
            return [
                Neighbor(
                    self._token_ids[i],
                    self._tokens[i],
                    float(dist),
                    float(self._weights[i]),
                    similarity=float(sim),
                    index=int(i),
                )
                for i, dist, sim in zip(idx, dists, similarities, strict=True)
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
        for nb, p in zip(neighbors, ex / total, strict=True):
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
        generation never degrades below the raw model. An error anywhere in the
        recall/blend path also fails open to the raw model probabilities — the
        documented contract now has a real exception boundary behind it.
        """
        try:
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
            if not math.isfinite(lam) or lam <= 1e-6:
                with self._lock:
                    self._stats["fallthrough"] += 1
                return dict(lm_probs)
            lam = min(1.0, max(0.0, float(lam)))
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
        except (TypeError, ValueError, KeyError, AttributeError, ZeroDivisionError, FloatingPointError) as exc:
            # Honor the fail-open contract: any recall/blend error returns the
            # raw model distribution rather than breaking generation.
            with self._lock:
                self._stats["fallthrough"] = self._stats.get("fallthrough", 0) + 1
            logger.debug("Nonparametric interpolate failed open: %s", exc)
            return dict(lm_probs)

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
            # SAME CONFIDENCE GATE AS interpolate(). This path fed the raw
            # query result straight into knn_probs, so every neighbor BELOW
            # the active similarity threshold was still mixed back into the
            # recall distribution — the two recall paths enforced different
            # standards, and the one wired to logits was the permissive one.
            # A single weak neighbour was enough to obtain a nonzero kNN mass
            # and shift the token distribution.
            min_sim = self.min_similarity()
            gated = [nb for nb in neighbors if nb.similarity >= min_sim]
            if not gated:
                with self._lock:
                    self._stats["fallthrough"] += 1
                return logits
            neighbors = gated
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

    def identity_receipt_with_work(self) -> tuple[dict[str, Any], int]:
        """Content-address the active store and report bytes hashed this call."""

        with self._lock:
            cached = self._identity_cache
            if cached is not None and cached[0] == self._content_generation:
                return dict(cached[1]), 0
            hasher = hashlib.sha256()
            keys = self._keys[: self._size]
            dtype_bytes = str(keys.dtype).encode("ascii")
            shape_bytes = str(keys.shape).encode("ascii")
            hasher.update(dtype_bytes)
            hasher.update(shape_bytes)
            hasher.update(memoryview(keys).cast("B"))
            metadata = json.dumps(
                {
                    "token_ids": self._token_ids,
                    "tokens": self._tokens,
                    "weights": self._weights,
                    "timestamps": self._ts,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            hasher.update(metadata)
            payload = {
                "schema": "aura.nonparametric_memory.identity.v1",
                "dimension": self._dim,
                "entries": self._size,
                "source_bytes": int(
                    len(dtype_bytes) + len(shape_bytes) + keys.nbytes + len(metadata)
                ),
                "content_sha256": hasher.hexdigest(),
            }
            receipt = {
                **payload,
                "receipt_sha256": hashlib.sha256(
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
            self._identity_cache = (self._content_generation, receipt)
            return dict(receipt), int(receipt["source_bytes"])

    def identity_receipt(self) -> dict[str, Any]:
        """Content-address the active datastore without exposing learned keys."""

        receipt, _bytes_hashed = self.identity_receipt_with_work()
        return receipt

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
            # Build EVERY new array/list into local temporaries first. If any
            # conversion (a non-numeric weight, malformed timestamp, etc.)
            # raises, the live object is left untouched — a partial swap
            # previously desynchronized the parallel keys/tokens/weights arrays
            # and stranded a stale _size behind a swallowed exception.
            capacity = min(max(64, count), self._max)
            new_keys = np.empty((capacity, self._dim), dtype=np.float32)
            new_norms = np.empty(capacity, dtype=np.float32)
            new_keys[:count] = keys[-count:].astype(np.float32)
            if not np.all(np.isfinite(new_keys[:count])):
                raise ValueError("non-parametric memory persisted keys are non-finite")
            new_norms[:count] = np.einsum("ij,ij->i", new_keys[:count], new_keys[:count])
            new_token_ids = [int(value) for value in token_ids[-count:]]
            new_tokens = [str(value) for value in tokens[-count:]]
            new_weights = [max(0.0, float(value)) for value in weights[-count:]]
            if not all(math.isfinite(w) for w in new_weights):
                raise ValueError("non-parametric memory persisted weights are non-finite")
            new_ts = [float(value) for value in timestamps[-count:]]
            if not all(math.isfinite(timestamp) for timestamp in new_ts):
                raise ValueError("non-parametric memory persisted timestamps are non-finite")
            saved_mu = meta.get("query_mu")
            new_mu = None
            new_mu_n = 0
            if isinstance(saved_mu, list) and len(saved_mu) == self._dim:
                candidate_mu = np.asarray(saved_mu, dtype=np.float32)
                if np.all(np.isfinite(candidate_mu)):
                    new_mu = candidate_mu
                    new_mu_n = max(0, int(meta.get("query_mu_n", 0) or 0))
            # All conversions succeeded — commit atomically under the lock.
            with self._lock:
                self._capacity = capacity
                self._keys = new_keys
                self._key_norms = new_norms
                self._token_ids = new_token_ids
                self._tokens = new_tokens
                self._weights = new_weights
                self._ts = new_ts
                self._size = count
                if new_mu is not None:
                    self._query_mu = new_mu
                    self._query_mu_n = new_mu_n
                self._content_generation += 1
                self._identity_cache = None
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


def validate_nonparametric_memory_identity(value: Any) -> dict[str, Any]:
    required = {
        "schema",
        "dimension",
        "entries",
        "source_bytes",
        "content_sha256",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("non-parametric memory identity fields differ")
    payload = {key: value[key] for key in required - {"receipt_sha256"}}
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    digest = value.get("content_sha256")
    if (
        value.get("schema") != "aura.nonparametric_memory.identity.v1"
        or type(value.get("dimension")) is not int
        or value["dimension"] <= 0
        or type(value.get("entries")) is not int
        or value["entries"] <= 0
        or type(value.get("source_bytes")) is not int
        or value["source_bytes"] <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        or value.get("receipt_sha256") != expected
    ):
        raise ValueError("non-parametric memory identity is invalid")
    return dict(value)


def reset_nonparametric_memory() -> None:
    global _singleton
    with _lock:
        _singleton = None
