"""Query-scoped associative control at the frozen checkpoint's output boundary.

The memory stores no text and changes no checkpoint parameter. A hidden state
must match the next committed key before a bounded logit margin is added to
its associated token. The cursor advances only on a match, making the write
both context- and order-sensitive. Erase drops every private tensor.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from core.brain.llm.latent_cortex.fast_weight_learning import token_sequence_sha256
from core.brain.llm.latent_cortex.verified_best import tensor_sha256

OUTPUT_MEMORY_SCHEMA = "aura.rlc.episodic_output_memory.v1"
OUTPUT_MEMORY_EXPERIMENT_SCHEMA = "aura.rlc.output_memory_experiment.v1"
OUTPUT_MEMORY_GAIN_GRID = (0.0, 0.5, 1.0)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema",
        "erased",
        "keys_sha256",
        "targets_sha256",
        "key_count",
        "target_count",
        "hidden_width",
        "similarity_floor",
        "margin",
        "gain",
        "matches",
        "misses",
        "minimum_similarity",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("output-memory identity fields do not match schema")
    if (
        value["schema"] != OUTPUT_MEMORY_SCHEMA
        or value["erased"] is not False
        or not _is_sha256(value["keys_sha256"])
        or not _is_sha256(value["targets_sha256"])
        or type(value["key_count"]) is not int
        or value["key_count"] <= 0
        or type(value["target_count"]) is not int
        or value["target_count"] != value["key_count"]
        or type(value["hidden_width"]) is not int
        or value["hidden_width"] <= 0
        or not math.isfinite(float(value["similarity_floor"]))
        or not 0.0 < float(value["similarity_floor"]) <= 1.0
        or not math.isfinite(float(value["margin"]))
        or not 0.0 < float(value["margin"]) <= 64.0
        or float(value["gain"]) != 0.0
        or value["matches"] != 0
        or value["misses"] != 0
        or float(value["minimum_similarity"]) != 1.0
    ):
        raise ValueError("output-memory identity is invalid")
    return dict(value)


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


def build_output_memory_experiment_receipt(
    *,
    baseline_score: float,
    treatment_identity: Mapping[str, Any],
    sham_identity: Mapping[str, Any],
    treatment_rows: Sequence[Mapping[str, Any]],
    sham_rows: Sequence[Mapping[str, Any]],
    erase_proven: bool,
) -> dict[str, Any]:
    """Select a treatment only when it beats baseline and matched sham."""

    if (
        not math.isfinite(float(baseline_score))
        or not 0.0 <= float(baseline_score) <= 1.0
    ):
        raise ValueError("output-memory baseline score is invalid")
    expected_gains = list(OUTPUT_MEMORY_GAIN_GRID)

    def normalize(arm: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        normalized = [dict(row) for row in rows]
        if [row.get("gain") for row in normalized] != expected_gains:
            raise ValueError("output-memory gain inventory differs")
        for index, row in enumerate(normalized):
            if (
                set(row)
                != {
                    "arm",
                    "gain",
                    "score",
                    "probe_tokens_sha256",
                    "probe_token_count",
                    "matches",
                    "misses",
                    "minimum_similarity",
                }
                or row["arm"] != arm
                or row["gain"] != expected_gains[index]
                or not math.isfinite(float(row["score"]))
                or not isinstance(row["probe_tokens_sha256"], str)
                or not _is_sha256(row["probe_tokens_sha256"])
                or type(row["probe_token_count"]) is not int
                or row["probe_token_count"] < 0
                or type(row["matches"]) is not int
                or type(row["misses"]) is not int
                or not 0.0 <= float(row["score"]) <= 1.0
                or not math.isfinite(float(row["minimum_similarity"]))
                or not -1.0 <= float(row["minimum_similarity"]) <= 1.0
            ):
                raise ValueError("output-memory experiment row is invalid")
        return normalized

    treatment_identity = _validate_identity(treatment_identity)
    sham_identity = _validate_identity(sham_identity)
    if (
        treatment_identity["targets_sha256"] == sham_identity["targets_sha256"]
        or treatment_identity["key_count"] != sham_identity["key_count"]
        or treatment_identity["hidden_width"] != sham_identity["hidden_width"]
        or treatment_identity["similarity_floor"] != sham_identity["similarity_floor"]
        or treatment_identity["margin"] != sham_identity["margin"]
    ):
        raise ValueError("output-memory treatment and sham are not matched")
    treatment = normalize("treatment", treatment_rows)
    sham = normalize("sham", sham_rows)

    def select(rows: list[dict[str, Any]]) -> dict[str, Any]:
        # Stable order makes zero the minimum-intervention equal-score winner.
        return max(rows, key=lambda row: float(row["score"]))

    selected_treatment = select(treatment)
    selected_sham = select(sham)
    accepted = bool(
        float(selected_treatment["gain"]) > 0.0
        and float(selected_treatment["score"]) > float(baseline_score) + 1e-6
        and float(selected_treatment["score"]) > float(selected_sham["score"]) + 1e-6
        and erase_proven is True
    )
    payload = {
        "schema": OUTPUT_MEMORY_EXPERIMENT_SCHEMA,
        "baseline_score": float(baseline_score),
        "gain_grid": expected_gains,
        "treatment_identity": dict(treatment_identity),
        "sham_identity": dict(sham_identity),
        "treatment": treatment,
        "sham": sham,
        "selected_treatment_gain": float(selected_treatment["gain"]),
        "selected_treatment_score": float(selected_treatment["score"]),
        "selected_sham_gain": float(selected_sham["gain"]),
        "selected_sham_score": float(selected_sham["score"]),
        "teacher_removed_from_probe_context": True,
        "matched_control": True,
        "erase_proven": bool(erase_proven),
        "accepted": accepted,
        "capability_claim_authority": False,
    }
    return {**payload, "receipt_sha256": _canonical_sha256(payload)}


def validate_output_memory_experiment_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the complete matched-control experiment receipt."""

    if not isinstance(value, Mapping):
        raise ValueError("output-memory experiment receipt must be a mapping")
    expected_fields = {
        "schema",
        "baseline_score",
        "gain_grid",
        "treatment_identity",
        "sham_identity",
        "treatment",
        "sham",
        "selected_treatment_gain",
        "selected_treatment_score",
        "selected_sham_gain",
        "selected_sham_score",
        "teacher_removed_from_probe_context",
        "matched_control",
        "erase_proven",
        "accepted",
        "capability_claim_authority",
        "receipt_sha256",
    }
    if set(value) != expected_fields:
        raise ValueError("output-memory experiment fields do not match schema")
    reconstructed = build_output_memory_experiment_receipt(
        baseline_score=value["baseline_score"],
        treatment_identity=value["treatment_identity"],
        sham_identity=value["sham_identity"],
        treatment_rows=value["treatment"],
        sham_rows=value["sham"],
        erase_proven=value["erase_proven"],
    )
    if dict(value) != reconstructed:
        raise ValueError("output-memory experiment receipt does not reconstruct")
    return dict(value)


__all__ = [
    "EpisodicOutputMemory",
    "OUTPUT_MEMORY_EXPERIMENT_SCHEMA",
    "OUTPUT_MEMORY_GAIN_GRID",
    "OUTPUT_MEMORY_SCHEMA",
    "build_output_memory_experiment_receipt",
    "validate_output_memory_experiment_receipt",
]
