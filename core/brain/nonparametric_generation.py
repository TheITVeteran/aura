"""Generation with non-parametric memory — the real causal loop.

Two model-coupled pieces (imports mlx lazily so the rest of the package stays light):

* ``MLXEncoder``        — turns text into a datastore key (last-token hidden state) and the
                          first continuation token. This is the real ingestion encoder.
* ``generate_with_memory`` — a full causal generation loop that, at EACH step, extracts the
                          model's hidden state, queries the datastore, interpolates the recall
                          into the next-token distribution, and samples. This is the end-to-end
                          causal path that proves the mechanism *generates*, not just predicts
                          one token.

Recomputes the forward each step (no KV cache) — O(n²), fine for short completions and the
background/idle lane. A KV-cached version inside the MLX worker is the production foreground
graduation (documented; default-off). Fail-open: any memory error falls back to the bare model.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.NonParametricGeneration")


def normalize(vec: np.ndarray) -> np.ndarray:
    """Unit-normalize a key so L2 distance encodes cosine: cos = 1 - d²/2.

    This makes the confidence cutoff model-independent (cosine ∈ [-1,1]) instead of
    depending on raw hidden-state magnitudes that vary by model.
    """
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def cosine_from_l2(distance: float) -> float:
    """Cosine similarity from the L2 distance between two UNIT vectors."""
    return 1.0 - (float(distance) ** 2) / 2.0


class MLXEncoder:
    """Real encoder over a loaded mlx_lm model: text -> (hidden key, first token)."""

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


def _bare_logits(model: Any, ids: list[int]):
    import mlx.core as mx

    arr = mx.array([ids])
    h = model.model(arr)
    logits = model(arr)
    key = np.array(h[0, -1], dtype=np.float32)
    lg = np.array(logits[0, -1], dtype=np.float32)
    return key, lg


def _topk_probs(logits: np.ndarray, k: int = 64) -> dict[int, float]:
    k = min(k, logits.shape[0])
    idx = np.argpartition(logits, -k)[-k:]
    sub = logits[idx]
    sub = sub - sub.max()
    ex = np.exp(sub)
    ex /= ex.sum()
    return {int(t): float(p) for t, p in zip(idx, ex)}


def generate_with_memory(
    model: Any,
    tokenizer: Any,
    prompt: str,
    memory: Any,
    *,
    max_tokens: int = 40,
    k: int = 4,
    temperature: float = 0.1,
    phi: float | None = 0.5,
    free_energy: float | None = 0.7,
    use_memory: bool = True,
) -> str:
    """Greedy generation that interpolates non-parametric recall into each step.

    ``use_memory=False`` is the bare-model baseline (for A/B). Fail-open per token.
    """
    ids = list(tokenizer.encode(prompt))
    start = len(ids)
    eos = getattr(tokenizer, "eos_token_id", None)
    base_lam = 0.75            # at perfect recall (sim≈1) trust memory strongly
    min_cos = 0.55             # legacy fallback when the memory carries no gate
    last_fired_index = -1      # anti-stutter: an entry may not fire twice in a row
    for _ in range(max(1, int(max_tokens))):
        try:
            key, lg = _bare_logits(model, ids)
        except (RuntimeError, ValueError, TypeError) as exc:
            record_degradation("nonparametric_generation_forward", exc)
            break
        next_id = int(np.argmax(lg))
        if use_memory:
            try:
                qkey = normalize(key)
                neighbors = memory.query(qkey, k=k)
                if neighbors:
                    # Anisotropy-corrected gate (see Neighbor.similarity).
                    sim = float(getattr(neighbors[0], "similarity", -1.0))
                    gate = float(getattr(memory, "min_similarity", lambda: min_cos)())
                    nearest_index = int(getattr(neighbors[0], "index", -1))
                    if sim >= gate and nearest_index != last_fired_index:
                        # confidence-GATED λ: scales with similarity, zero below the gate.
                        fe = 0.5 if free_energy is None else float(free_energy)
                        lam = base_lam * max(0.0, (sim - gate) / max(1e-6, 1.0 - gate)) * (0.6 + 0.8 * fe)
                        lm_probs = _topk_probs(lg)
                        blended = memory.interpolate(
                            lm_probs, qkey, k=k, temperature=temperature, phi=phi,
                            free_energy=free_energy, lam_override=min(lam, 0.9),
                        )
                        memory_choice = int(max(blended, key=blended.get))
                        if memory_choice != next_id:
                            last_fired_index = nearest_index
                        next_id = memory_choice
            except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                record_degradation("nonparametric_generation_interpolate", exc)
        ids.append(next_id)
        if eos is not None and next_id == eos:
            break
    return tokenizer.decode(ids[start:]).strip()


def make_nonparametric_logits_processor(
    model: Any,
    memory: Any,
    *,
    k: int = 4,
    temperature: float = 0.1,
    phi: float | None = 0.5,
    free_energy: float | None = 0.7,
    min_cos: float = 0.55,
    base_lam: float = 0.75,
) -> Any:
    """A live mlx_lm logits-processor: interpolate non-parametric recall per token.

    Signature ``(tokens, logits) -> logits`` matches mlx_lm. Recomputes the hidden state
    from the running tokens to form the query key (so it works through the standard
    stream_generate seam) — that's an extra forward per token, so this is the *validation/
    opt-in* form; the KV-cached version inside the worker loop is the latency-optimized
    follow-up. Fail-open: any error returns the original logits unchanged.
    """
    import mlx.core as mx

    state = {"last_fired_index": -1}

    def _proc(tokens: Any, logits: Any) -> Any:
        try:
            seq = tokens.reshape(1, -1) if hasattr(tokens, "reshape") else mx.array([tokens])
            h = model.model(seq)
            key = normalize(np.array(h[0, -1], dtype=np.float32))
            neighbors = memory.query(key, k=k)
            if not neighbors:
                return logits
            # Anisotropy-corrected gate + anti-stutter (see Neighbor).
            sim = float(getattr(neighbors[0], "similarity", -1.0))
            gate = float(getattr(memory, "min_similarity", lambda: min_cos)())
            nearest_index = int(getattr(neighbors[0], "index", -1))
            if sim < gate or nearest_index == state["last_fired_index"]:
                return logits
            state["last_fired_index"] = nearest_index
            fe = 0.5 if free_energy is None else float(free_energy)
            lam = base_lam * ((sim - gate) / max(1e-6, 1.0 - gate)) * (0.6 + 0.8 * fe)
            lg = np.array(logits, dtype=np.float32).reshape(-1)
            ktop = min(64, lg.shape[0])
            idx = np.argpartition(lg, -ktop)[-ktop:]
            sub = lg[idx] - lg[idx].max()
            ex = np.exp(sub)
            ex /= ex.sum()
            lm_probs = {int(t): float(p) for t, p in zip(idx, ex)}
            blended = memory.interpolate(
                lm_probs, key, k=k, temperature=temperature, phi=phi,
                free_energy=free_energy, lam_override=min(lam, 0.9),
            )
            out = lg.copy()
            import math as _m

            for t, p in blended.items():
                out[int(t)] = _m.log(max(p, 1e-12))
            return mx.array(out).reshape(logits.shape)
        except (RuntimeError, ValueError, TypeError, AttributeError, IndexError) as exc:
            record_degradation("nonparametric_logits_processor", exc)
            return logits

    return _proc
