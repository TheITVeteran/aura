"""Runtime learned accept/discard gate for recurrent latent proposals."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.loop_core import (
    assert_finite_state,
    canonical_sha256,
)
from core.brain.llm.latent_cortex.workspace import per_position_rms
from core.learning.update_acceptance import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_SHA256,
    UPDATE_ACCEPTANCE_FEATURE_SCHEMA,
    UpdateAcceptanceHead,
)

UPDATE_GATE_SCHEMA = "aura.rlc.update_gate.v1"
PASSTHROUGH = "passthrough"
LEARNED = "learned"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _pooled(value: Any) -> Any:
    import mlx.core as mx

    axes = tuple(range(value.ndim - 1))
    return mx.mean(value.astype(mx.float32), axis=axes)


def _cosine(left: Any, right: Any) -> float:
    import mlx.core as mx

    left_pool = _pooled(left)
    right_pool = _pooled(right)
    denominator = mx.maximum(
        mx.linalg.norm(left_pool) * mx.linalg.norm(right_pool),
        1e-9,
    )
    value = float(mx.sum(left_pool * right_pool) / denominator)
    return max(-1.0, min(1.0, value))


def _mean_rms(value: Any) -> float:
    import mlx.core as mx

    return float(mx.mean(per_position_rms(value)))


def _relative_distance(value: Any, reference: Any) -> float:
    import mlx.core as mx

    if tuple(value.shape) == tuple(reference.shape):
        numerator = mx.mean(per_position_rms(value - reference))
        denominator = mx.maximum(mx.mean(per_position_rms(reference)), 1e-6)
    else:
        value_pool = _pooled(value)
        reference_pool = _pooled(reference)
        numerator = mx.linalg.norm(value_pool - reference_pool)
        denominator = mx.maximum(mx.linalg.norm(reference_pool), 1e-6)
    return float(numerator / denominator)


def _bounded(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("update-gate feature is non-finite")
    return round(max(-32.0, min(32.0, value)), 10)


def extract_update_features(
    previous_state: Any,
    proposal_state: Any,
    anchor_state: Any,
    *,
    evidence_state: Any | None,
    previous_residual: float | None,
    previous_delta: Any | None,
) -> tuple[dict[str, float], Any]:
    """Measure one proposal against its prior, fixed anchor, and evidence."""

    import mlx.core as mx

    shape = tuple(previous_state.shape)
    if tuple(proposal_state.shape) != shape or tuple(anchor_state.shape) != shape:
        raise ValueError("update-gate reasoning-state shapes differ")
    for stage, value in (
        ("update_gate_previous", previous_state),
        ("update_gate_proposal", proposal_state),
        ("update_gate_anchor", anchor_state),
    ):
        assert_finite_state(value, stage=stage)
    evidence_available = evidence_state is not None
    evidence = anchor_state if evidence_state is None else evidence_state
    if evidence.shape[-1] != previous_state.shape[-1] or evidence.size < 1:
        raise ValueError("update-gate evidence state is incompatible")
    assert_finite_state(evidence, stage="update_gate_evidence")

    delta = proposal_state - previous_state
    residual = _relative_distance(proposal_state, previous_state)
    previous_anchor_alignment = _cosine(previous_state, anchor_state)
    proposal_anchor_alignment = _cosine(proposal_state, anchor_state)
    previous_evidence_alignment = _cosine(previous_state, evidence)
    proposal_evidence_alignment = _cosine(proposal_state, evidence)
    anchor_distance_improvement = _relative_distance(
        previous_state, anchor_state
    ) - _relative_distance(proposal_state, anchor_state)
    evidence_distance_improvement = _relative_distance(
        previous_state, evidence
    ) - _relative_distance(proposal_state, evidence)
    anchor_rms = max(_mean_rms(anchor_state), 1e-6)
    proposal_anchor_log_rms_error = abs(
        math.log(max(_mean_rms(proposal_state), 1e-6) / anchor_rms)
    )
    previous_anchor_log_rms_error = abs(
        math.log(max(_mean_rms(previous_state), 1e-6) / anchor_rms)
    )
    contraction = 1.0
    if previous_residual is not None:
        if not _finite(previous_residual) or float(previous_residual) < 0.0:
            raise ValueError("update-gate previous residual is invalid")
        contraction = residual / max(float(previous_residual), 1e-9)
    delta_cosine = 0.0
    if previous_delta is not None:
        if tuple(previous_delta.shape) != shape:
            raise ValueError("update-gate previous delta shape differs")
        assert_finite_state(previous_delta, stage="update_gate_previous_delta")
        denominator = mx.maximum(
            mx.linalg.norm(previous_delta) * mx.linalg.norm(delta),
            1e-9,
        )
        delta_cosine = float(mx.sum(previous_delta * delta) / denominator)
        delta_cosine = max(-1.0, min(1.0, delta_cosine))
    features = {
        "proposal_residual": _bounded(residual),
        "anchor_alignment_delta": _bounded(
            proposal_anchor_alignment - previous_anchor_alignment
        ),
        "evidence_alignment_delta": _bounded(
            proposal_evidence_alignment - previous_evidence_alignment
        ),
        "anchor_distance_improvement": _bounded(anchor_distance_improvement),
        "evidence_distance_improvement": _bounded(
            evidence_distance_improvement
        ),
        "proposal_previous_cosine": _bounded(
            _cosine(proposal_state, previous_state)
        ),
        "delta_anchor_cosine": _bounded(_cosine(delta, anchor_state)),
        "delta_evidence_cosine": _bounded(_cosine(delta, evidence)),
        "proposal_anchor_log_rms_error": _bounded(
            proposal_anchor_log_rms_error
        ),
        "previous_anchor_log_rms_error": _bounded(
            previous_anchor_log_rms_error
        ),
        "residual_contraction_ratio": _bounded(contraction),
        "delta_cosine_previous": _bounded(delta_cosine),
        "evidence_available": 1.0 if evidence_available else 0.0,
    }
    if tuple(features) != FEATURE_NAMES:
        raise RuntimeError("update-gate feature construction order drifted")
    mx.eval(delta)
    return features, delta


@dataclass(frozen=True)
class UpdateGateDecision:
    accepted: bool
    probability: float
    threshold: float
    reason: str
    features: dict[str, float]
    features_sha256: str
    delta: Any


class UpdateGateRuntime:
    """Loaded runtime policy; passthrough remains explicit and receipted."""

    def __init__(
        self,
        *,
        mode: str,
        head: UpdateAcceptanceHead | None = None,
        head_sha256: str = "",
    ) -> None:
        if mode not in {PASSTHROUGH, LEARNED}:
            raise ValueError("update-gate mode is invalid")
        if mode == PASSTHROUGH and (head is not None or head_sha256):
            raise ValueError("passthrough update gate cannot carry a head")
        if mode == LEARNED and (
            head is None or not head.calibrated or not _is_sha256(head_sha256)
        ):
            raise ValueError("learned update gate requires a calibrated pinned head")
        self.mode = mode
        self.head = head
        self.head_sha256 = head_sha256

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> UpdateGateRuntime:
        config = dict(value or {})
        mode = str(config.get("mode", PASSTHROUGH))
        if mode == PASSTHROUGH:
            return cls(mode=PASSTHROUGH)
        if mode != LEARNED:
            raise ValueError("update-gate mode is invalid")
        path = config.get("head_path")
        expected_sha256 = config.get("head_sha256")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("learned update gate requires head_path")
        if not _is_sha256(expected_sha256):
            raise ValueError("learned update gate requires head_sha256")
        try:
            head = UpdateAcceptanceHead.load(
                path,
                expected_sha256=expected_sha256,
            )
        except OSError as exc:
            raise ValueError("learned update-gate artifact is unreadable") from exc
        return cls(mode=LEARNED, head=head, head_sha256=expected_sha256)

    @property
    def threshold(self) -> float:
        return 0.0 if self.head is None else float(self.head.threshold)

    @property
    def manifest(self) -> dict[str, Any]:
        return {} if self.head is None else self.head.to_manifest()

    def probability_for_features(self, features: Mapping[str, Any]) -> float:
        return 1.0 if self.head is None else self.head.probability(features)

    def evaluate(
        self,
        previous_state: Any,
        proposal_state: Any,
        anchor_state: Any,
        *,
        evidence_state: Any | None,
        previous_residual: float | None,
        previous_delta: Any | None,
    ) -> UpdateGateDecision:
        features, delta = extract_update_features(
            previous_state,
            proposal_state,
            anchor_state,
            evidence_state=evidence_state,
            previous_residual=previous_residual,
            previous_delta=previous_delta,
        )
        probability = self.probability_for_features(features)
        accepted = bool(self.mode == PASSTHROUGH or probability >= self.threshold)
        reason = (
            "passthrough"
            if self.mode == PASSTHROUGH
            else "learned_probability_admitted"
            if accepted
            else "learned_probability_rejected"
        )
        return UpdateGateDecision(
            accepted=accepted,
            probability=round(probability, 10),
            threshold=round(self.threshold, 10),
            reason=reason,
            features=features,
            features_sha256=canonical_sha256(
                {
                    "schema": UPDATE_ACCEPTANCE_FEATURE_SCHEMA,
                    "values": features,
                }
            ),
            delta=delta,
        )


def _branch_summary(branch: Any) -> dict[str, Any]:
    transitions = [dict(row) for row in branch.update_acceptance_trace]
    return {
        "branch_index": int(branch.index),
        "role": str(branch.role),
        "transition_count": len(transitions),
        "accepted": sum(row["accepted"] is True for row in transitions),
        "rejected": sum(row["accepted"] is False for row in transitions),
        "transitions": transitions,
    }


def _causal_rejection(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("accepted") is False
        and (
            row.get("proposal_hypothesis_sha256")
            != row.get("admitted_hypothesis_sha256")
            or row.get("proposal_reasoning_sha256")
            != row.get("admitted_reasoning_sha256")
        )
    )


def build_update_gate_receipt(
    *,
    branches: list[Any],
    selected_branch: int,
    gate: UpdateGateRuntime,
    recurrent_grounding: dict[str, Any],
    loop_stability: dict[str, Any],
) -> dict[str, Any]:
    branch_rows = [_branch_summary(branch) for branch in branches]
    rejected = sum(row["rejected"] for row in branch_rows)
    causal_rejections = sum(
        _causal_rejection(transition)
        for branch in branch_rows
        for transition in branch["transitions"]
    )
    payload = {
        "schema": UPDATE_GATE_SCHEMA,
        "mode": gate.mode,
        "head_sha256": gate.head_sha256,
        "head_manifest": gate.manifest,
        "feature_schema": UPDATE_ACCEPTANCE_FEATURE_SCHEMA,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "threshold": round(gate.threshold, 10),
        "selected_branch": selected_branch,
        "branches": branch_rows,
        "decision_count": sum(row["transition_count"] for row in branch_rows),
        "accepted": sum(row["accepted"] for row in branch_rows),
        "rejected": rejected,
        "causal_rejections": causal_rejections,
        "all_proposals_decided": all(
            row["transition_count"] > 0 for row in branch_rows
        ),
        "head_was_causal": bool(gate.mode == LEARNED and causal_rejections > 0),
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return validate_update_gate_receipt(
        receipt,
        expected_gate=gate,
        recurrent_grounding=recurrent_grounding,
        loop_stability=loop_stability,
    )


def validate_update_gate_receipt(
    value: Any,
    *,
    expected_gate: UpdateGateRuntime,
    recurrent_grounding: dict[str, Any],
    loop_stability: dict[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema",
        "mode",
        "head_sha256",
        "head_manifest",
        "feature_schema",
        "feature_schema_sha256",
        "threshold",
        "selected_branch",
        "branches",
        "decision_count",
        "accepted",
        "rejected",
        "causal_rejections",
        "all_proposals_decided",
        "head_was_causal",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("update-gate receipt fields do not match schema")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != canonical_sha256(payload):
        raise ValueError("update-gate receipt commitment mismatch")
    grounding_branches = recurrent_grounding.get("branches")
    loop_branches = loop_stability.get("branches")
    branches = value["branches"]
    if (
        value["schema"] != UPDATE_GATE_SCHEMA
        or value["mode"] != expected_gate.mode
        or value["head_sha256"] != expected_gate.head_sha256
        or value["head_manifest"] != expected_gate.manifest
        or value["feature_schema"] != UPDATE_ACCEPTANCE_FEATURE_SCHEMA
        or value["feature_schema_sha256"] != FEATURE_SCHEMA_SHA256
        or not _finite(value["threshold"])
        or not math.isclose(
            float(value["threshold"]),
            expected_gate.threshold,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        or type(value["selected_branch"]) is not int
        or type(value["decision_count"]) is not int
        or type(value["accepted"]) is not int
        or type(value["rejected"]) is not int
        or type(value["causal_rejections"]) is not int
        or value["selected_branch"] != recurrent_grounding.get("selected_branch")
        or value["selected_branch"] != loop_stability.get("selected_branch")
        or not isinstance(branches, list)
        or not branches
        or not isinstance(grounding_branches, list)
        or not isinstance(loop_branches, list)
        or not (
            len(branches) == len(grounding_branches) == len(loop_branches)
        )
    ):
        raise ValueError("update-gate topology or identity is invalid")
    branch_fields = {
        "branch_index",
        "role",
        "transition_count",
        "accepted",
        "rejected",
        "transitions",
    }
    transition_fields = {
        "ordinal",
        "branch_step",
        "prior_hypothesis_sha256",
        "proposal_hypothesis_sha256",
        "admitted_hypothesis_sha256",
        "prior_reasoning_sha256",
        "proposal_reasoning_sha256",
        "admitted_reasoning_sha256",
        "probability",
        "threshold",
        "accepted",
        "reason",
        "features",
        "features_sha256",
    }
    total_accepted = 0
    total_rejected = 0
    total_causal_rejections = 0
    total_decisions = 0
    for index, (branch, grounding, loop_branch) in enumerate(
        zip(branches, grounding_branches, loop_branches, strict=True)
    ):
        transitions = branch.get("transitions") if isinstance(branch, dict) else None
        grounded = grounding.get("transitions") if isinstance(grounding, dict) else None
        loop_transitions = (
            loop_branch.get("transitions") if isinstance(loop_branch, dict) else None
        )
        if (
            not isinstance(branch, dict)
            or set(branch) != branch_fields
            or type(branch.get("branch_index")) is not int
            or branch["branch_index"] != index
            or grounding.get("branch_index") != index
            or loop_branch.get("branch_index") != index
            or branch["role"] != grounding.get("role")
            or branch["role"] != loop_branch.get("role")
            or not isinstance(transitions, list)
            or not transitions
            or not isinstance(grounded, list)
            or not isinstance(loop_transitions, list)
            or not (
                len(transitions) == len(grounded) == len(loop_transitions)
            )
            or type(branch["transition_count"]) is not int
            or type(branch["accepted"]) is not int
            or type(branch["rejected"]) is not int
            or branch["transition_count"] != len(transitions)
        ):
            raise ValueError("update-gate branch evidence is invalid")
        accepted_count = 0
        rejected_count = 0
        for ordinal, (row, ground_row, loop_row) in enumerate(
            zip(transitions, grounded, loop_transitions, strict=True)
        ):
            if (
                not isinstance(row, dict)
                or set(row) != transition_fields
                or type(row.get("ordinal")) is not int
                or type(row.get("branch_step")) is not int
                or row["ordinal"] != ordinal
                or row["branch_step"] != ground_row.get("branch_step")
                or row["branch_step"] != loop_row.get("branch_step")
                or row["prior_hypothesis_sha256"]
                != ground_row.get("hypothesis_pre_sha256")
                or row["prior_hypothesis_sha256"]
                != loop_row.get("hypothesis_pre_sha256")
                or row["proposal_hypothesis_sha256"]
                != loop_row.get("hypothesis_post_sha256")
                or row["admitted_hypothesis_sha256"]
                != ground_row.get("hypothesis_post_sha256")
                or row["prior_reasoning_sha256"]
                != loop_row.get("reasoning_pre_sha256")
                or row["proposal_reasoning_sha256"]
                != loop_row.get("reasoning_post_sha256")
                or any(
                    not _is_sha256(row[name])
                    for name in (
                        "prior_hypothesis_sha256",
                        "proposal_hypothesis_sha256",
                        "admitted_hypothesis_sha256",
                        "prior_reasoning_sha256",
                        "proposal_reasoning_sha256",
                        "admitted_reasoning_sha256",
                        "features_sha256",
                    )
                )
                or type(row["accepted"]) is not bool
                or not _finite(row["probability"])
                or not 0.0 <= float(row["probability"]) <= 1.0
                or not _finite(row["threshold"])
                or not math.isclose(
                    float(row["threshold"]),
                    expected_gate.threshold,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                )
                or not isinstance(row["features"], dict)
                or set(row["features"]) != set(FEATURE_NAMES)
                or any(
                    not _finite(feature) or abs(float(feature)) > 32.0
                    for feature in row["features"].values()
                )
                or row["features_sha256"]
                != canonical_sha256(
                    {
                        "schema": UPDATE_ACCEPTANCE_FEATURE_SCHEMA,
                        "values": row["features"],
                    }
                )
            ):
                raise ValueError("update-gate transition evidence is invalid")
            probability = expected_gate.probability_for_features(row["features"])
            expected_accepted = bool(
                expected_gate.mode == PASSTHROUGH
                or probability >= expected_gate.threshold
            )
            expected_reason = (
                "passthrough"
                if expected_gate.mode == PASSTHROUGH
                else "learned_probability_admitted"
                if expected_accepted
                else "learned_probability_rejected"
            )
            if (
                not math.isclose(
                    float(row["probability"]),
                    probability,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or row["accepted"] is not expected_accepted
                or row["reason"] != expected_reason
                or (
                    expected_accepted
                    and (
                        row["admitted_hypothesis_sha256"]
                        != row["proposal_hypothesis_sha256"]
                        or row["admitted_reasoning_sha256"]
                        != row["proposal_reasoning_sha256"]
                    )
                )
                or (
                    not expected_accepted
                    and (
                        row["admitted_hypothesis_sha256"]
                        != row["prior_hypothesis_sha256"]
                        or row["admitted_reasoning_sha256"]
                        != row["prior_reasoning_sha256"]
                    )
                )
                or (
                    not expected_accepted
                    and loop_row.get("disposition")
                    not in {"quality_rejected", "contained_divergence"}
                )
            ):
                raise ValueError("update-gate decision does not match its head")
            accepted_count += expected_accepted
            rejected_count += not expected_accepted
            total_causal_rejections += _causal_rejection(row)
        if (
            branch["accepted"] != accepted_count
            or branch["rejected"] != rejected_count
        ):
            raise ValueError("update-gate branch summary mismatch")
        total_accepted += accepted_count
        total_rejected += rejected_count
        total_decisions += len(transitions)
    if (
        value["decision_count"] != total_decisions
        or value["accepted"] != total_accepted
        or value["rejected"] != total_rejected
        or value["causal_rejections"] != total_causal_rejections
        or value["all_proposals_decided"] is not True
        or value["head_was_causal"]
        is not (
            expected_gate.mode == LEARNED and total_causal_rejections > 0
        )
        or (expected_gate.mode == PASSTHROUGH and total_rejected != 0)
    ):
        raise ValueError("update-gate aggregate verdict is invalid")
    return dict(value)


__all__ = [
    "LEARNED",
    "PASSTHROUGH",
    "UPDATE_GATE_SCHEMA",
    "UpdateGateDecision",
    "UpdateGateRuntime",
    "build_update_gate_receipt",
    "extract_update_features",
    "validate_update_gate_receipt",
]
