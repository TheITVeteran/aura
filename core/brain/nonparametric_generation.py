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
    temperature: float = 2.0,
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
    base_lam = 0.75            # at perfect recall (cos≈1) trust memory strongly
    min_cos = 0.55             # below this, the neighbor is unrelated → defer to the model
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
                    cos = cosine_from_l2(neighbors[0].distance)
                    if cos >= min_cos:
                        # confidence-GATED λ: scales with cosine, zero for far neighbors.
                        fe = 0.5 if free_energy is None else float(free_energy)
                        lam = base_lam * max(0.0, (cos - min_cos) / (1.0 - min_cos)) * (0.6 + 0.8 * fe)
                        lm_probs = _topk_probs(lg)
                        blended = memory.interpolate(
                            lm_probs, qkey, k=k, temperature=temperature, phi=phi,
                            free_energy=free_energy, lam_override=min(lam, 0.9),
                        )
                        next_id = int(max(blended, key=blended.get))
            except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                record_degradation("nonparametric_generation_interpolate", exc)
        ids.append(next_id)
        if eos is not None and next_id == eos:
            break
    return tokenizer.decode(ids[start:]).strip()
