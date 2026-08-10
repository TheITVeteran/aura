"""Verifier-bound evidence assimilation in Aura's recurrent latent workspace.

The executable objective lane may provide a private, independently verified
derivation.  This module never promotes that lane's answer.  It binds the
derivation to hidden evidence rows, constructs a norm-matched semantic sham,
and receipts the actual task-verifier comparison after both arms recur and
decode through the same neural path.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

WORKSPACE_EVIDENCE_SCHEMA = "aura.rlc.verified_workspace_evidence.v1"


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


def deterministic_semantic_sham(state: Any, *, salt: str):
    """Destroy coordinate semantics while preserving every row's exact L2 norm."""

    import mlx.core as mx

    if not isinstance(salt, str) or not salt:
        raise ValueError("workspace evidence sham salt is empty")
    if not hasattr(state, "shape") or len(state.shape) != 3:
        raise ValueError("workspace evidence state must be rank three")
    hidden = int(state.shape[-1])
    if hidden < 2:
        raise ValueError("workspace evidence hidden width is too small")
    digest = hashlib.sha256(salt.encode("utf-8")).digest()
    shift = 1 + int.from_bytes(digest[:4], "big") % (hidden - 1)
    rolled = mx.roll(state, shift=shift, axis=-1)
    # A deterministic orthogonal sign matrix prevents a rotationally symmetric
    # downstream projection from making the sham accidentally equivalent.
    signs = mx.array(
        [
            1.0 if digest[index % len(digest)] & (1 << (index % 8)) else -1.0
            for index in range(hidden)
        ]
    ).astype(state.dtype)
    sham = rolled * signs.reshape(1, 1, hidden)
    mx.eval(sham)
    return sham


def replace_workspace_slots(
    state: Any,
    evidence: Any,
    *,
    slot_indices: Sequence[int],
):
    """Replace an exact ordered slot inventory without mutating other rows."""

    import mlx.core as mx

    slots = tuple(slot_indices)
    if (
        not hasattr(state, "shape")
        or len(state.shape) != 3
        or not hasattr(evidence, "shape")
        or len(evidence.shape) != 3
        or int(state.shape[0]) != 1
        or int(evidence.shape[0]) != 1
        or int(state.shape[2]) != int(evidence.shape[2])
        or int(evidence.shape[1]) != len(slots)
        or not slots
        or any(type(index) is not int for index in slots)
        or len(set(slots)) != len(slots)
        or any(not 0 <= index < int(state.shape[1]) for index in slots)
    ):
        raise ValueError("workspace evidence slot replacement is invalid")
    by_slot = {slot: offset for offset, slot in enumerate(slots)}
    result = mx.concatenate(
        [
            evidence[:, by_slot[index] : by_slot[index] + 1, :]
            if index in by_slot
            else state[:, index : index + 1, :]
            for index in range(int(state.shape[1]))
        ],
        axis=1,
    )
    mx.eval(result)
    return result


def build_workspace_evidence_receipt(
    *,
    objective_sha256: str,
    teaching_event_sha256: str,
    private_witness_sha256: str,
    private_witness_token_count: int,
    target_slots: Sequence[int],
    assimilation_steps: int,
    source_state_sha256: str,
    treatment_seed_sha256: str,
    sham_seed_sha256: str,
    treatment_state_sha256: str,
    sham_state_sha256: str,
    baseline_score: float,
    treatment_score: float,
    sham_score: float,
    treatment_tokens_sha256: str,
    sham_tokens_sha256: str,
) -> dict[str, Any]:
    """Build the public commitment and strict causal disposition."""

    digests = (
        objective_sha256,
        teaching_event_sha256,
        private_witness_sha256,
        source_state_sha256,
        treatment_seed_sha256,
        sham_seed_sha256,
        treatment_state_sha256,
        sham_state_sha256,
        treatment_tokens_sha256,
        sham_tokens_sha256,
    )
    slots = list(target_slots)
    scores = (baseline_score, treatment_score, sham_score)
    if (
        any(not _is_sha256(value) for value in digests)
        or type(private_witness_token_count) is not int
        or not 0 < private_witness_token_count <= 512
        or type(assimilation_steps) is not int
        or not 1 <= assimilation_steps <= 8
        or not slots
        or any(type(index) is not int or index <= 0 for index in slots)
        or len(set(slots)) != len(slots)
        or any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            for score in scores
        )
    ):
        raise ValueError("workspace evidence receipt inputs are invalid")
    strict_improvement = float(treatment_score) > float(baseline_score)
    semantic_specificity = float(treatment_score) > float(sham_score)
    accepted = strict_improvement and semantic_specificity
    payload = {
        "schema": WORKSPACE_EVIDENCE_SCHEMA,
        "objective_sha256": objective_sha256,
        "teaching_event_sha256": teaching_event_sha256,
        "private_witness_sha256": private_witness_sha256,
        "private_witness_token_count": private_witness_token_count,
        "private_witness_publicly_disclosed": False,
        "target_slots": slots,
        "assimilation_steps": assimilation_steps,
        "source_state_sha256": source_state_sha256,
        "treatment_seed_sha256": treatment_seed_sha256,
        "sham_seed_sha256": sham_seed_sha256,
        "treatment_state_sha256": treatment_state_sha256,
        "sham_state_sha256": sham_state_sha256,
        "baseline_score": float(baseline_score),
        "treatment_score": float(treatment_score),
        "sham_score": float(sham_score),
        "treatment_tokens_sha256": treatment_tokens_sha256,
        "sham_tokens_sha256": sham_tokens_sha256,
        "strict_improvement": strict_improvement,
        "semantic_specificity": semantic_specificity,
        "accepted": accepted,
        "disposition": "accepted" if accepted else "rejected_non_improvement",
        "answer_authority": "neural_decode_only",
        "producer_answer_promoted": False,
        "claim_boundary": "verified_evidence_conditioned_neural_mechanism",
        "sham_policy": "hidden_dimension_signed_permutation_norm_preserving_v1",
    }
    receipt = {**payload, "receipt_sha256": _canonical_sha256(payload)}
    return validate_workspace_evidence_receipt(receipt)


def validate_workspace_evidence_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema",
        "objective_sha256",
        "teaching_event_sha256",
        "private_witness_sha256",
        "private_witness_token_count",
        "private_witness_publicly_disclosed",
        "target_slots",
        "assimilation_steps",
        "source_state_sha256",
        "treatment_seed_sha256",
        "sham_seed_sha256",
        "treatment_state_sha256",
        "sham_state_sha256",
        "baseline_score",
        "treatment_score",
        "sham_score",
        "treatment_tokens_sha256",
        "sham_tokens_sha256",
        "strict_improvement",
        "semantic_specificity",
        "accepted",
        "disposition",
        "answer_authority",
        "producer_answer_promoted",
        "claim_boundary",
        "sham_policy",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("workspace evidence receipt fields differ")
    payload = {field: value[field] for field in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != _canonical_sha256(payload):
        raise ValueError("workspace evidence receipt commitment mismatch")
    token_count = value["private_witness_token_count"]
    steps = value["assimilation_steps"]
    slots = value["target_slots"]
    raw_scores = tuple(value[name] for name in ("baseline_score", "treatment_score", "sham_score"))
    if (
        type(token_count) is not int
        or not 0 < token_count <= 512
        or type(steps) is not int
        or not 1 <= steps <= 8
        or not isinstance(slots, list)
        or not slots
        or any(type(index) is not int or index <= 0 for index in slots)
        or len(set(slots)) != len(slots)
        or any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            for score in raw_scores
        )
    ):
        raise ValueError("workspace evidence receipt values are invalid")
    scores = tuple(float(score) for score in raw_scores)
    strict = scores[1] > scores[0]
    specific = scores[1] > scores[2]
    accepted = strict and specific
    if (
        value["schema"] != WORKSPACE_EVIDENCE_SCHEMA
        or any(
            not _is_sha256(value[name])
            for name in (
                "objective_sha256",
                "teaching_event_sha256",
                "private_witness_sha256",
                "source_state_sha256",
                "treatment_seed_sha256",
                "sham_seed_sha256",
                "treatment_state_sha256",
                "sham_state_sha256",
                "treatment_tokens_sha256",
                "sham_tokens_sha256",
            )
        )
        or value["private_witness_publicly_disclosed"] is not False
        or value["strict_improvement"] is not strict
        or value["semantic_specificity"] is not specific
        or value["accepted"] is not accepted
        or value["disposition"] != ("accepted" if accepted else "rejected_non_improvement")
        or value["answer_authority"] != "neural_decode_only"
        or value["producer_answer_promoted"] is not False
        or value["claim_boundary"] != "verified_evidence_conditioned_neural_mechanism"
        or value["sham_policy"] != "hidden_dimension_signed_permutation_norm_preserving_v1"
    ):
        raise ValueError("workspace evidence receipt policy differs")
    return dict(value)


__all__ = [
    "WORKSPACE_EVIDENCE_SCHEMA",
    "build_workspace_evidence_receipt",
    "deterministic_semantic_sham",
    "replace_workspace_slots",
    "validate_workspace_evidence_receipt",
]
