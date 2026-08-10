"""Query-scoped associative control at the frozen checkpoint's output boundary.

The memory stores no text and changes no checkpoint parameter. A hidden state
must match the next committed key before a bounded logit margin is added to
its associated token. The cursor advances only on a match, making the write
both context- and order-sensitive. Erase drops every private tensor.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from core.brain.llm.latent_cortex.episodic_output_memory_contract import (
    OUTPUT_MEMORY_EXPERIMENT_SCHEMA,
    OUTPUT_MEMORY_GAIN_GRID,
    OUTPUT_MEMORY_SCHEMA,
    build_output_memory_experiment_receipt,
    validate_output_memory_experiment_receipt,
)
from core.brain.llm.latent_cortex.fast_weight_learning import token_sequence_sha256
from core.brain.llm.latent_cortex.verified_best import tensor_sha256


class EpisodicOutputMemory:
    """Finite ordered hidden-state to next-token associative memory."""

    def __init__(
        self,
        keys: Any,
        target_tokens: Sequence[int],
        *,
        similarity_floor: float = 0.995,
        margin: float = 8.0,
    ) -> None:
        import mlx.core as mx

        targets = [int(token) for token in target_tokens]
        if getattr(keys, "ndim", 0) != 2 or not targets:
            raise ValueError("episodic output memory requires key rows and targets")
        if int(keys.shape[0]) != len(targets):
            raise ValueError("episodic output key and target counts differ")
        if any(token < 0 for token in targets):
            raise ValueError("episodic output target token is invalid")
        if (
            isinstance(similarity_floor, bool)
            or not math.isfinite(float(similarity_floor))
            or not 0.0 < float(similarity_floor) <= 1.0
        ):
            raise ValueError("episodic output similarity floor is invalid")
        if (
            isinstance(margin, bool)
            or not math.isfinite(float(margin))
            or not 0.0 < float(margin) <= 64.0
        ):
            raise ValueError("episodic output margin is invalid")
        norms = mx.linalg.norm(keys.astype(mx.float32), axis=1, keepdims=True)
        if bool(mx.any(norms <= 1e-8)):
            raise ValueError("episodic output key contains a zero vector")
        self.keys = mx.stop_gradient(keys.astype(mx.float32) / norms)
        self.target_tokens = targets
        self.similarity_floor = float(similarity_floor)
        self.margin = float(margin)
        self.gain = 0.0
        self.cursor = 0
        self.matches = 0
        self.misses = 0
        self.minimum_similarity = 1.0
        self.erased = False
        mx.eval(self.keys)

    def reset(self, *, gain: float) -> None:
        if (
            isinstance(gain, bool)
            or not math.isfinite(float(gain))
            or not 0.0 <= float(gain) <= 2.0
        ):
            raise ValueError("episodic output gain is outside [0, 2]")
        if self.erased:
            raise RuntimeError("episodic output memory was erased")
        self.gain = float(gain)
        self.cursor = 0
        self.matches = 0
        self.misses = 0
        self.minimum_similarity = 1.0

    def apply(self, hidden: Any, logits: Any):
        """Apply one ordered association when the current hidden state matches."""

        import mlx.core as mx

        if self.erased or self.gain <= 0.0 or self.cursor >= len(self.target_tokens):
            return logits
        if getattr(hidden, "ndim", 0) != 3 or getattr(logits, "ndim", 0) != 3:
            raise ValueError("episodic output memory requires batched sequence tensors")
        query = hidden[0, -1].astype(mx.float32)
        query = query / mx.maximum(mx.linalg.norm(query), 1e-8)
        similarity = max(
            -1.0,
            min(1.0, float(mx.sum(query * self.keys[self.cursor]))),
        )
        self.minimum_similarity = min(self.minimum_similarity, similarity)
        if similarity < self.similarity_floor:
            self.misses += 1
            return logits
        token = self.target_tokens[self.cursor]
        if token >= int(logits.shape[-1]):
            raise ValueError("episodic output token exceeds vocabulary")
        row = logits[0, -1]
        required = mx.maximum(mx.max(row) - row[token] + self.margin, 0.0)
        delta = self.gain * required
        updated = logits.at[0, -1, token].add(delta.astype(logits.dtype))
        self.cursor += 1
        self.matches += 1
        return updated

    def erase(self) -> None:
        self.keys = None
        self.target_tokens = []
        self.gain = 0.0
        self.cursor = 0
        self.erased = True

    def receipt(self) -> dict[str, Any]:
        if self.erased:
            return {
                "schema": OUTPUT_MEMORY_SCHEMA,
                "erased": True,
                "key_count": 0,
                "target_count": 0,
            }
        return {
            "schema": OUTPUT_MEMORY_SCHEMA,
            "erased": False,
            "keys_sha256": tensor_sha256(self.keys),
            "targets_sha256": token_sequence_sha256(self.target_tokens),
            "key_count": int(self.keys.shape[0]),
            "target_count": len(self.target_tokens),
            "hidden_width": int(self.keys.shape[1]),
            "similarity_floor": self.similarity_floor,
            "margin": self.margin,
            "gain": self.gain,
            "matches": self.matches,
            "misses": self.misses,
            "minimum_similarity": round(float(self.minimum_similarity), 8),
        }


__all__ = [
    "EpisodicOutputMemory",
    "OUTPUT_MEMORY_EXPERIMENT_SCHEMA",
    "OUTPUT_MEMORY_GAIN_GRID",
    "OUTPUT_MEMORY_SCHEMA",
    "build_output_memory_experiment_receipt",
    "validate_output_memory_experiment_receipt",
]
