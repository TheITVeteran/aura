"""Query-scoped low-rank neural readout for verified corrections.

Unlike the ordered output memory, this operator has no cursor, token lookup,
or similarity threshold. It learns a bounded linear hidden-state to sparse
logit correction from a private teacher-forced trajectory, then generates
again after the teacher has been removed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from core.brain.llm.latent_cortex.episodic_neural_readout_contract import (
    NEURAL_READOUT_EXPERIMENT_SCHEMA,
    NEURAL_READOUT_GAIN_GRID,
    NEURAL_READOUT_SCHEMA,
    build_neural_readout_experiment_receipt,
    validate_neural_readout_experiment_receipt,
)
from core.brain.llm.latent_cortex.fast_weight_learning import token_sequence_sha256
from core.brain.llm.latent_cortex.verified_best import tensor_sha256


class EpisodicNeuralReadout:
    """A finite-rank linear readout fitted to one verified trajectory."""

    def __init__(
        self,
        keys: Any,
        target_tokens: Sequence[int],
        required_margins: Sequence[float],
        *,
        max_rank: int = 32,
        ridge: float = 1e-4,
        margin: float = 4.0,
    ) -> None:
        rows = np.asarray(keys, dtype=np.float32)
        targets = tuple(int(token) for token in target_tokens)
        required = np.asarray(required_margins, dtype=np.float32)
        if (
            rows.ndim != 2
            or rows.shape[0] == 0
            or len(targets) != rows.shape[0]
            or required.shape != (rows.shape[0],)
            or not np.isfinite(rows).all()
            or not np.isfinite(required).all()
            or np.any(required <= 0.0)
            or any(token < 0 for token in targets)
        ):
            raise ValueError("neural-readout training trajectory is invalid")
        if type(max_rank) is not int or max_rank <= 0:
            raise ValueError("neural-readout rank bound must be positive")
        if (
            isinstance(ridge, bool)
            or not math.isfinite(float(ridge))
            or not 1e-8 <= float(ridge) <= 1e4
            or isinstance(margin, bool)
            or not math.isfinite(float(margin))
            or not 0.0 < float(margin) <= 64.0
        ):
            raise ValueError("neural-readout fit parameters are invalid")
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        if np.any(norms <= 1e-8):
            raise ValueError("neural-readout key contains a zero vector")
        normalized = rows / norms
        token_ids = tuple(sorted(set(targets)))
        columns = {token: index for index, token in enumerate(token_ids)}
        desired = np.zeros((len(targets), len(token_ids)), dtype=np.float32)
        for index, token in enumerate(targets):
            desired[index, columns[token]] = required[index]

        gram = normalized @ normalized.T
        gram.flat[:: gram.shape[0] + 1] += float(ridge)
        try:
            weights = normalized.T @ np.linalg.solve(gram, desired)
        except np.linalg.LinAlgError as exc:
            raise ValueError("neural-readout ridge system is singular") from exc
        left, singular, right = np.linalg.svd(weights, full_matrices=False)
        numerical_rank = int(np.count_nonzero(singular > 1e-7))
        effective_rank = min(max_rank, numerical_rank)
        if effective_rank <= 0:
            raise ValueError("neural-readout fit collapsed")
        weights = (left[:, :effective_rank] * singular[np.newaxis, :effective_rank]) @ right[
            :effective_rank
        ]
        if not np.isfinite(weights).all() or not np.any(weights):
            raise FloatingPointError("neural-readout fit produced no finite correction")

        self.keys_sha256 = tensor_sha256(normalized)
        self.target_tokens = targets
        self.token_ids = token_ids
        self.weights = weights.astype(np.float32)
        self.sample_count = len(targets)
        self.hidden_width = int(rows.shape[1])
        self.effective_rank = effective_rank
        self.ridge = float(ridge)
        self.margin = float(margin)
        self.gain = 0.0
        self.applications = 0
        self.erased = False
        self._mlx_weights = None

    def reset(self, *, gain: float) -> None:
        if (
            isinstance(gain, bool)
            or not math.isfinite(float(gain))
            or not 0.0 <= float(gain) <= 2.0
        ):
            raise ValueError("neural-readout gain is outside [0, 2]")
        if self.erased:
            raise RuntimeError("neural readout was erased")
        self.gain = float(gain)
        self.applications = 0

    def apply(self, hidden: Any, logits: Any):
        if self.erased or self.gain <= 0.0:
            return logits
        if getattr(hidden, "ndim", 0) != 3 or getattr(logits, "ndim", 0) != 3:
            raise ValueError("neural readout requires batched sequence tensors")
        if int(hidden.shape[-1]) != self.hidden_width:
            raise ValueError("neural readout hidden width differs")
        if self.token_ids[-1] >= int(logits.shape[-1]):
            raise ValueError("neural readout token exceeds vocabulary")

        import mlx.core as mx

        if self._mlx_weights is None:
            self._mlx_weights = mx.array(self.weights)
            mx.eval(self._mlx_weights)
        query = hidden[0, -1].astype(mx.float32)
        query = query / mx.maximum(mx.linalg.norm(query), 1e-8)
        correction = query @ self._mlx_weights
        correction = mx.maximum(correction, 0.0) * self.gain
        updated = logits.at[0, -1, list(self.token_ids)].add(correction.astype(logits.dtype))
        self.applications += 1
        return updated

    def erase(self) -> None:
        self.weights = None
        self._mlx_weights = None
        self.target_tokens = ()
        self.token_ids = ()
        self.gain = 0.0
        self.erased = True

    def receipt(self) -> dict[str, Any]:
        if self.erased:
            return {
                "schema": NEURAL_READOUT_SCHEMA,
                "erased": True,
                "sample_count": 0,
                "token_count": 0,
            }
        return {
            "schema": NEURAL_READOUT_SCHEMA,
            "erased": False,
            "keys_sha256": self.keys_sha256,
            "weights_sha256": tensor_sha256(self.weights),
            "targets_sha256": token_sequence_sha256(self.target_tokens),
            "sample_count": self.sample_count,
            "hidden_width": self.hidden_width,
            "token_count": len(self.token_ids),
            "effective_rank": self.effective_rank,
            "ridge": self.ridge,
            "margin": self.margin,
            "gain": self.gain,
            "applications": self.applications,
        }


__all__ = [
    "EpisodicNeuralReadout",
    "NEURAL_READOUT_EXPERIMENT_SCHEMA",
    "NEURAL_READOUT_GAIN_GRID",
    "NEURAL_READOUT_SCHEMA",
    "build_neural_readout_experiment_receipt",
    "validate_neural_readout_experiment_receipt",
]
