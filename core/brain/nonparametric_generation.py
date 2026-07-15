"""Generation with non-parametric memory — the real, KV-cached causal loop.

* ``MLXEncoder``               — text/ids → normalized hidden key + first continuation token.
* ``generate_with_memory``     — KV-CACHED generation: prefill once, then O(1) per token. At each
                                 step the model's hidden state is the query key; confident recall is
                                 interpolated into the next-token distribution. This is the
                                 production form (no O(n²) recompute).
* ``make_nonparametric_logits_processor`` — the same gating as an mlx_lm logits-processor.

Gating uses the datastore's **anisotropy-corrected similarity** (``Neighbor.similarity`` +
``memory.min_similarity()``). This matters: raw last-token hidden states share a dominant common
direction — measured, UNRELATED prompts score raw cosine 0.81–0.93 — so a naive raw-cosine gate
cannot separate related from unrelated. Mean-centred similarity does (unrelated ≤0.36).

Anti-stutter: the entry that fired last step is excluded from the next step's neighbors, so a
single stored entry can't lock generation into a repeat loop.

Fail-open everywhere: any memory error defers to the bare model.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.NonParametricGeneration")

_GEN_ERRORS = (RuntimeError, ValueError, TypeError, AttributeError, IndexError, KeyError)
# Φ floor mirrors phi_consciousness: fragmented cognition must not trust recall.
PHI_DORMANT = 0.05


def normalize(vec: np.ndarray) -> np.ndarray:
    """Unit-normalize a key. L2 distance on unit vectors encodes cosine: cos = 1 - d²/2."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def cosine_from_l2(distance: float) -> float:
    """Cosine similarity from the L2 distance between two UNIT vectors."""
    return 1.0 - (float(distance) ** 2) / 2.0


class MLXEncoder:
    """Real encoder over a loaded mlx_lm model: text -> (normalized hidden key, first token)."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tok = tokenizer
        self.dim = int(model.args.hidden_size)
        self._specials = set(getattr(tokenizer, "all_special_ids", []) or [])

    def encode_hidden(self, text: str) -> np.ndarray:
        return normalize(self._hidden_from_ids(self.tok.encode(text)))

    def encode_hidden_ids(self, ids: list[int]) -> np.ndarray:
        return normalize(self._hidden_from_ids(list(ids)))

    def _hidden_from_ids(self, ids: list[int]) -> np.ndarray:
        import mlx.core as mx

        h = self.model.model(mx.array([ids]))
        return np.array(h[0, -1], dtype=np.float32)

    def encode_tokens(self, text: str) -> list[int]:
        return list(self.tok.encode(text))

    def first_token(self, continuation: str) -> int:
        ids = [i for i in self.tok.encode(continuation) if i not in self._specials]
        return int(ids[0]) if ids else 0


def _lm_head(model: Any, hidden: Any) -> Any:
    """Project hidden states to logits (handles tied-embedding models)."""
    head = getattr(model, "lm_head", None)
    if callable(head):
        return head(hidden)
    return model.model.embed_tokens.as_linear(hidden)


def _topk_probs(logits: np.ndarray, k: int = 64) -> dict[int, float]:
    k = min(k, logits.shape[0])
    idx = np.argpartition(logits, -k)[-k:]
    sub = logits[idx] - logits[idx].max()
    ex = np.exp(sub)
    ex /= ex.sum()
    return {int(t): float(p) for t, p in zip(idx, ex)}


def _gated_lambda(similarity: float, min_sim: float, free_energy: float | None, base_lam: float) -> float:
    """λ scaled by how far the neighbor clears the confident-recall gate. 0 below the gate."""
    if similarity < min_sim:
        return 0.0
    fe = 0.5 if free_energy is None else float(free_energy)
    span = max(1e-6, 1.0 - min_sim)
    lam = base_lam * ((similarity - min_sim) / span) * (0.6 + 0.8 * fe)
    return max(0.0, min(lam, 0.9))


def _select_with_memory(
    memory: Any,
    key: np.ndarray,
    logits: np.ndarray,
    *,
    k: int,
    temperature: float,
    phi: float | None,
    free_energy: float | None,
    base_lam: float,
    exclude_index: int,
) -> tuple[int, int]:
    """Return (next_token_id, fired_entry_index). fired_index=-1 when memory didn't fire."""
    bare = int(np.argmax(logits))
    if phi is not None and float(phi) < PHI_DORMANT:
        return bare, -1
    neighbors = [nb for nb in memory.query(key, k=k) if int(getattr(nb, "index", -1)) != exclude_index]
    if not neighbors:
        return bare, -1
    top = neighbors[0]
    min_sim = memory.min_similarity() if hasattr(memory, "min_similarity") else 0.98
    lam = _gated_lambda(float(getattr(top, "similarity", -1.0)), float(min_sim), free_energy, base_lam)
    if lam <= 1e-6:
        return bare, -1
    knn = memory.knn_probs(neighbors, temperature=temperature)
    if not knn:
        return bare, -1
    lm_probs = _topk_probs(logits)
    blended = {
        t: (1.0 - lam) * lm_probs.get(t, 0.0) + lam * knn.get(t, 0.0)
        for t in set(lm_probs) | set(knn)
    }
    return int(max(blended, key=blended.get)), int(getattr(top, "index", -1))


def generate_with_memory(
    model: Any,
    tokenizer: Any,
    prompt: str,
    memory: Any,
    *,
    max_tokens: int = 40,
    k: int = 4,
    temperature: float = 2.0,
    phi: float | None = 0.5,
    free_energy: float | None = 0.7,
    use_memory: bool = True,
    base_lam: float = 0.75,
) -> str:
    """KV-cached greedy generation with confident non-parametric recall interpolated per token.

    Prefill once, then O(1) per token (no full recompute) — this is the production form.
    ``use_memory=False`` gives the bare-model baseline for A/B.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    ids = list(tokenizer.encode(prompt))
    eos = getattr(tokenizer, "eos_token_id", None)
    out: list[int] = []
    last_index = -1
    try:
        cache = make_prompt_cache(model)
        h = model.model(mx.array([ids]), cache=cache)
    except _GEN_ERRORS as exc:
        record_degradation("nonparametric_generation_prefill", exc)
        return ""
    for _ in range(max(1, int(max_tokens))):
        try:
            logits = np.array(_lm_head(model, h[:, -1:, :])[0, -1], dtype=np.float32)
        except _GEN_ERRORS as exc:
            record_degradation("nonparametric_generation_head", exc)
            break
        next_id = int(np.argmax(logits))
        if use_memory:
            try:
                key = normalize(np.array(h[0, -1], dtype=np.float32))
                next_id, last_index = _select_with_memory(
                    memory, key, logits, k=k, temperature=temperature, phi=phi,
                    free_energy=free_energy, base_lam=base_lam, exclude_index=last_index,
                )
            except _GEN_ERRORS as exc:
                record_degradation("nonparametric_generation_select", exc)
        out.append(next_id)
        if eos is not None and next_id == eos:
            break
        try:
            h = model.model(mx.array([[next_id]]), cache=cache)  # incremental, O(1)
        except _GEN_ERRORS as exc:
            record_degradation("nonparametric_generation_step", exc)
            break
    return tokenizer.decode(out).strip()


def make_nonparametric_logits_processor(
    model: Any,
    memory: Any,
    *,
    k: int = 4,
    temperature: float = 2.0,
    phi: float | None = 0.5,
    free_energy: float | None = 0.7,
    base_lam: float = 0.75,
) -> Any:
    """mlx_lm ``(tokens, logits) -> logits`` processor applying the same gated recall.

    Note: a logits-processor has no access to the hidden state, so this form recomputes it
    (a forward per token — O(n²) overall). Prefer ``generate_with_memory`` (KV-cached) for
    production; this exists for drop-in use inside an existing stream_generate call.
    Fail-open: any error returns the logits unchanged.
    """
    import mlx.core as mx

    state = {"last_index": -1}

    def _proc(tokens: Any, logits: Any) -> Any:
        try:
            seq = tokens.reshape(1, -1) if hasattr(tokens, "reshape") else mx.array([tokens])
            h = model.model(seq)
            key = normalize(np.array(h[0, -1], dtype=np.float32))
            lg = np.array(logits, dtype=np.float32).reshape(-1)
            chosen, fired = _select_with_memory(
                memory, key, lg, k=k, temperature=temperature, phi=phi,
                free_energy=free_energy, base_lam=base_lam, exclude_index=state["last_index"],
            )
            state["last_index"] = fired
            if fired < 0:
                return logits
            out = lg.copy()
            out[chosen] = float(np.max(lg)) + 1.0  # make the recalled token win the sampler
            return mx.array(out).reshape(logits.shape)
        except _GEN_ERRORS as exc:
            record_degradation("nonparametric_logits_processor", exc)
            return logits

    return _proc
