"""Calibrated stop/convergence gate for recurrent latent execution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.learning.stop_policy import (
    STOP_FEATURE_NAMES,
    STOP_FEATURE_SCHEMA_SHA256,
    STOP_POLICY_FEATURE_SCHEMA,
    StopPolicyHead,
)

STOP_GATE_SCHEMA = "aura.rlc.stop_gate.v1"
RESIDUAL = "residual"
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


def _bounded(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("stop-gate feature is non-finite")
    return round(max(-4.0, min(4.0, value)), 10)


@dataclass(frozen=True, slots=True)
class StopContext:
    """Public engine-level evidence available at one recurrent transition."""

    action_step: int
    max_steps: int
    policy_uncertainty: float
    verifier_score: float | None
    verifier_delta: float | None
    expected_gain_lcb: float
    expected_cost_ucb: float
    quality_measured: bool
    evoc_measured: bool
    budget_remaining_fraction: float

    def __post_init__(self) -> None:
        if type(self.action_step) is not int or self.action_step < 0:
            raise ValueError("stop context action_step is invalid")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("stop context max_steps is invalid")
        for name, value, minimum, maximum in (
            ("policy_uncertainty", self.policy_uncertainty, 0.0, 1.0),
            ("expected_gain_lcb", self.expected_gain_lcb, -4.0, 4.0),
            ("expected_cost_ucb", self.expected_cost_ucb, 0.0, 1.0),
            (
                "budget_remaining_fraction",
                self.budget_remaining_fraction,
                0.0,
                1.0,
            ),
        ):
            if not _finite(value) or not minimum <= float(value) <= maximum:
                raise ValueError(f"stop context {name} is invalid")
        for name in ("verifier_score", "verifier_delta"):
            value = getattr(self, name)
            if value is not None and (
                not _finite(value) or not -1.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"stop context {name} is invalid")
        if type(self.quality_measured) is not bool:
            raise ValueError("stop context quality_measured must be boolean")
        if type(self.evoc_measured) is not bool:
            raise ValueError("stop context evoc_measured must be boolean")


@dataclass(frozen=True, slots=True)
class StopGateDecision:
    halt: bool
    reason: str
    probability: float
    threshold: float
    evidence_ready: bool
    features: dict[str, float]
    features_sha256: str


class StopGateRuntime:
    """Loaded stop policy; residual mode is explicit and receipted."""

    def __init__(
        self,
        *,
        mode: str,
        head: StopPolicyHead | None = None,
        head_sha256: str = "",
    ) -> None:
        if mode not in {RESIDUAL, LEARNED}:
            raise ValueError("stop-gate mode is invalid")
        if mode == RESIDUAL and (head is not None or head_sha256):
            raise ValueError("residual stop gate cannot carry a learned head")
        if mode == LEARNED and (
            head is None or not _is_sha256(head_sha256)
        ):
            raise ValueError("learned stop gate requires a calibrated pinned head")
        self.mode = mode
        self.head = head
        self.head_sha256 = head_sha256

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> StopGateRuntime:
        config = dict(value or {})
        mode = str(config.get("mode", RESIDUAL))
        if mode == RESIDUAL:
            return cls(mode=RESIDUAL)
        if mode != LEARNED:
            raise ValueError("stop-gate mode is invalid")
        path = config.get("head_path")
        expected_sha256 = config.get("head_sha256")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("learned stop gate requires head_path")
        if not _is_sha256(expected_sha256):
            raise ValueError("learned stop gate requires head_sha256")
        try:
            head = StopPolicyHead.load(path, expected_sha256=expected_sha256)
        except OSError as exc:
            raise ValueError("learned stop-gate artifact is unreadable") from exc
        return cls(mode=LEARNED, head=head, head_sha256=expected_sha256)

    @property
    def threshold(self) -> float:
        return 1.0 if self.head is None else float(self.head.threshold)

    @property
    def manifest(self) -> dict[str, Any]:
        return {} if self.head is None else self.head.manifest()

    def probability_for_features(self, features: Mapping[str, Any]) -> float:
        return 0.0 if self.head is None else self.head.probability(features)

    def evaluate(
        self,
        *,
        step: int,
        residual: float,
        previous_residual: float | None,
        update_decision: Any,
        context: StopContext,
    ) -> StopGateDecision:
        if type(step) is not int or step < 1 or step > context.max_steps:
            raise ValueError("stop-gate step is invalid")
        if not _finite(residual) or float(residual) < 0.0:
            raise ValueError("stop-gate residual is invalid")
        probability = float(getattr(update_decision, "probability", 1.0))
        accepted = getattr(update_decision, "accepted", None)
        features_source = getattr(update_decision, "features", {})
        if (
            not 0.0 <= probability <= 1.0
            or type(accepted) is not bool
            or not isinstance(features_source, Mapping)
        ):
            raise ValueError("stop-gate update evidence is invalid")
        quality_measured = context.quality_measured
        reported_residual = round(float(residual), 8)
        reported_previous_residual = (
            None if previous_residual is None else round(float(previous_residual), 8)
        )
        contraction = (
            1.0
            if reported_previous_residual is None
            else reported_residual / max(reported_previous_residual, 1e-9)
        )
        evidence_improvement = 0.5 * (
            float(features_source.get("anchor_distance_improvement", 0.0))
            + float(features_source.get("evidence_distance_improvement", 0.0))
        )
        verifier_score = (
            0.0 if context.verifier_score is None else float(context.verifier_score)
        )
        verifier_delta = (
            0.0 if context.verifier_delta is None else float(context.verifier_delta)
        )
        features = {
            "step_fraction": _bounded(step / context.max_steps),
            "residual": _bounded(reported_residual),
            "residual_contraction_ratio": _bounded(contraction),
            "quality_probability": _bounded(probability),
            "quality_uncertainty": _bounded(1.0 - abs(2.0 * probability - 1.0)),
            "evidence_improvement": _bounded(evidence_improvement),
            "verifier_score": _bounded(verifier_score),
            "verifier_delta": _bounded(verifier_delta),
            "policy_uncertainty": _bounded(context.policy_uncertainty),
            "expected_gain_lcb": _bounded(context.expected_gain_lcb),
            "expected_cost_ucb": _bounded(context.expected_cost_ucb),
            "expected_net_value": _bounded(
                context.expected_gain_lcb - context.expected_cost_ucb
            ),
            "budget_remaining_fraction": _bounded(
                context.budget_remaining_fraction
            ),
            "proposal_accepted": 1.0 if accepted else 0.0,
            "quality_measured": 1.0 if quality_measured else 0.0,
            "evoc_measured": 1.0 if context.evoc_measured else 0.0,
            "verifier_available": (
                1.0 if context.verifier_score is not None else 0.0
            ),
        }
        if tuple(features) != STOP_FEATURE_NAMES:
            raise RuntimeError("stop-gate feature construction order drifted")
        predicted = self.probability_for_features(features)
        evidence_ready = bool(quality_measured and context.evoc_measured)
        halt = bool(
            self.mode == LEARNED
            and evidence_ready
            and predicted >= self.threshold
        )
        reason = (
            "residual_policy"
            if self.mode == RESIDUAL
            else "learned_stop"
            if halt
            else "continue_unmeasured_evidence"
            if not evidence_ready
            else "continue_expected_value_positive"
        )
        feature_payload = {
            "schema": STOP_POLICY_FEATURE_SCHEMA,
            "values": features,
        }
        return StopGateDecision(
            halt=halt,
            reason=reason,
            probability=round(predicted, 10),
            threshold=round(self.threshold, 10),
            evidence_ready=evidence_ready,
            features=features,
            features_sha256=canonical_sha256(feature_payload),
        )


def build_stop_gate_receipt(
    *,
    branches: list[Any],
    gate: StopGateRuntime,
    update_acceptance: dict[str, Any],
    loop_stability: dict[str, Any],
    cognitive_action_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    branch_rows = []
    for branch in branches:
        traces = [dict(row) for row in branch.halting.stop_trace]
        branch_rows.append(
            {
                "branch_index": int(branch.index),
                "role": str(branch.role),
                "halt_reason": str(branch.halt_reason),
                "steps_taken": int(branch.steps),
                "decision_count": len(traces),
                "learned_halts": sum(row["halt"] is True for row in traces),
                "decisions": traces,
            }
        )
    learned_halts = sum(row["learned_halts"] for row in branch_rows)
    payload = {
        "schema": STOP_GATE_SCHEMA,
        "mode": gate.mode,
        "head_sha256": gate.head_sha256,
        "head_manifest": gate.manifest,
        "feature_schema": STOP_POLICY_FEATURE_SCHEMA,
        "feature_schema_sha256": STOP_FEATURE_SCHEMA_SHA256,
        "threshold": round(gate.threshold, 10),
        "update_acceptance_sha256": update_acceptance.get("receipt_sha256"),
        "loop_stability_sha256": loop_stability.get("receipt_sha256"),
        "cognitive_action_trace_sha256": canonical_sha256(
            cognitive_action_trace
        ),
        "branches": branch_rows,
        "decision_count": sum(row["decision_count"] for row in branch_rows),
        "learned_halts": learned_halts,
        "head_was_causal": bool(gate.mode == LEARNED and learned_halts > 0),
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    return validate_stop_gate_receipt(
        receipt,
        expected_gate=gate,
        expected_n_branches=len(branches),
        update_acceptance=update_acceptance,
        loop_stability=loop_stability,
        cognitive_action_trace=cognitive_action_trace,
    )


def validate_stop_gate_receipt(
    value: Any,
    *,
    expected_gate: StopGateRuntime,
    expected_n_branches: int | None = None,
    update_acceptance: dict[str, Any],
    loop_stability: dict[str, Any],
    cognitive_action_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    fields = {
        "schema",
        "mode",
        "head_sha256",
        "head_manifest",
        "feature_schema",
        "feature_schema_sha256",
        "threshold",
        "update_acceptance_sha256",
        "loop_stability_sha256",
        "cognitive_action_trace_sha256",
        "branches",
        "decision_count",
        "learned_halts",
        "head_was_causal",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("stop-gate receipt fields differ")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != canonical_sha256(payload):
        raise ValueError("stop-gate receipt commitment mismatch")
    if (
        value["schema"] != STOP_GATE_SCHEMA
        or value["mode"] != expected_gate.mode
        or value["head_sha256"] != expected_gate.head_sha256
        or value["head_manifest"] != expected_gate.manifest
        or value["feature_schema"] != STOP_POLICY_FEATURE_SCHEMA
        or value["feature_schema_sha256"] != STOP_FEATURE_SCHEMA_SHA256
        or not _finite(value["threshold"])
        or not math.isclose(
            float(value["threshold"]),
            expected_gate.threshold,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        or value["update_acceptance_sha256"]
        != update_acceptance.get("receipt_sha256")
        or value["loop_stability_sha256"]
        != loop_stability.get("receipt_sha256")
        or value["cognitive_action_trace_sha256"]
        != canonical_sha256(cognitive_action_trace)
        or not isinstance(value["branches"], list)
        or not value["branches"]
        or (
            expected_n_branches is not None
            and (
                type(expected_n_branches) is not int
                or expected_n_branches < 1
                or len(value["branches"]) != expected_n_branches
            )
        )
        or type(value["decision_count"]) is not int
        or type(value["learned_halts"]) is not int
        or type(value["head_was_causal"]) is not bool
    ):
        raise ValueError("stop-gate identity is invalid")
    branch_fields = {
        "branch_index",
        "role",
        "halt_reason",
        "steps_taken",
        "decision_count",
        "learned_halts",
        "decisions",
    }
    decision_fields = {
        "ordinal",
        "action_step",
        "step",
        "halt",
        "reason",
        "probability",
        "threshold",
        "evidence_ready",
        "features",
        "features_sha256",
    }
    total_decisions = 0
    total_halts = 0
    update_branches = update_acceptance.get("branches")
    loop_branches = loop_stability.get("branches")
    loop_core = loop_stability.get("loop_core")
    if (
        not isinstance(update_branches, list)
        or not isinstance(loop_branches, list)
        or len(update_branches) != len(value["branches"])
        or len(loop_branches) != len(value["branches"])
        or not isinstance(loop_core, dict)
        or type(loop_core.get("max_steps")) is not int
    ):
        raise ValueError("stop-gate causal source topology is invalid")
    if not isinstance(cognitive_action_trace, list):
        raise ValueError("stop-gate cognitive action trace must be a list")
    action_rows: dict[int, dict[str, Any]] = {}
    for action_row in cognitive_action_trace:
        decision = (
            action_row.get("decision") if isinstance(action_row, dict) else None
        )
        action_step = (
            decision.get("step_index") if isinstance(decision, dict) else None
        )
        if (
            type(action_step) is not int
            or action_step in action_rows
        ):
            raise ValueError("stop-gate cognitive action trace is invalid")
        action_rows[action_step] = action_row
    for index, branch in enumerate(value["branches"]):
        decisions = branch.get("decisions") if isinstance(branch, dict) else None
        if (
            not isinstance(branch, dict)
            or set(branch) != branch_fields
            or type(branch["branch_index"]) is not int
            or branch["branch_index"] != index
            or not isinstance(branch["role"], str)
            or not isinstance(branch["halt_reason"], str)
            or type(branch["steps_taken"]) is not int
            or branch["steps_taken"] < 0
            or type(branch["decision_count"]) is not int
            or type(branch["learned_halts"]) is not int
            or not isinstance(decisions, list)
            or branch["decision_count"] != len(decisions)
        ):
            raise ValueError("stop-gate branch evidence is invalid")
        branch_halts = 0
        prior_step = 0
        update_rows = update_branches[index].get("transitions")
        loop_rows = loop_branches[index].get("transitions")
        if not isinstance(update_rows, list) or not isinstance(loop_rows, list):
            raise ValueError("stop-gate causal transition sources are invalid")
        for ordinal, row in enumerate(decisions):
            if (
                not isinstance(row, dict)
                or set(row) != decision_fields
                or type(row["ordinal"]) is not int
                or row["ordinal"] != ordinal
                or type(row["action_step"]) is not int
                or row["action_step"] < 0
                or type(row["step"]) is not int
                or row["step"] < 1
                or row["step"] <= prior_step
                or row["step"] > branch["steps_taken"]
                or type(row["halt"]) is not bool
                or type(row["evidence_ready"]) is not bool
                or not isinstance(row["reason"], str)
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
                or set(row["features"]) != set(STOP_FEATURE_NAMES)
                or any(
                    not _finite(feature) or abs(float(feature)) > 4.0
                    for feature in row["features"].values()
                )
                or not _is_sha256(row["features_sha256"])
                or row["features_sha256"]
                != canonical_sha256(
                    {
                        "schema": STOP_POLICY_FEATURE_SCHEMA,
                        "values": row["features"],
                    }
                )
            ):
                raise ValueError("stop-gate decision evidence is invalid")
            prior_step = row["step"]
            update_row = next(
                (
                    item
                    for item in update_rows
                    if item.get("branch_step") == row["step"] - 1
                ),
                None,
            )
            loop_row = next(
                (
                    item
                    for item in loop_rows
                    if item.get("branch_step") == row["step"] - 1
                ),
                None,
            )
            action_row = action_rows.get(row["action_step"])
            if (
                not isinstance(update_row, dict)
                or not isinstance(loop_row, dict)
                or not isinstance(action_row, dict)
            ):
                raise ValueError("stop-gate causal source row is absent")
            state_signal = action_row.get("state_signal")
            action_decision = action_row.get("decision")
            evidence = (
                action_decision.get("evidence")
                if isinstance(action_decision, dict)
                else None
            )
            if not isinstance(state_signal, dict) or not isinstance(evidence, dict):
                raise ValueError("stop-gate action evidence is invalid")
            update_features = update_row.get("features")
            if not isinstance(update_features, dict):
                raise ValueError("stop-gate update features are invalid")
            required_source_values = (
                update_features.get("anchor_distance_improvement"),
                update_features.get("evidence_distance_improvement"),
                update_row.get("probability"),
                state_signal.get("uncertainty"),
                state_signal.get("budget_remaining_fraction"),
                evidence.get("gain_used"),
                evidence.get("cost_used"),
            )
            if (
                any(not _finite(item) for item in required_source_values)
                or type(update_row.get("accepted")) is not bool
                or type(evidence.get("measured")) is not bool
            ):
                raise ValueError("stop-gate causal source values are invalid")
            previous_loop = next(
                (
                    item
                    for item in loop_rows
                    if item.get("branch_step") == row["step"] - 2
                ),
                None,
            )
            residual_value = loop_row.get("residual")
            previous_residual_value = (
                None if previous_loop is None else previous_loop.get("residual")
            )
            if (
                not _finite(residual_value)
                or float(residual_value) < 0.0
                or (
                    previous_loop is not None
                    and (
                        not _finite(previous_residual_value)
                        or float(previous_residual_value) < 0.0
                    )
                )
            ):
                raise ValueError("stop-gate loop source values are invalid")
            residual = float(residual_value)
            contraction = (
                1.0
                if previous_loop is None
                else residual / max(float(previous_residual_value), 1e-9)
            )
            verifier_score = state_signal.get("verifier_score")
            verifier_delta = state_signal.get("verifier_delta")
            if (
                verifier_score is not None and not _finite(verifier_score)
            ) or (
                verifier_delta is not None and not _finite(verifier_delta)
            ):
                raise ValueError("stop-gate verifier source values are invalid")
            expected_features = {
                "step_fraction": row["step"] / int(loop_core["max_steps"]),
                "residual": residual,
                "residual_contraction_ratio": contraction,
                "quality_probability": float(update_row.get("probability")),
                "quality_uncertainty": (
                    1.0 - abs(2.0 * float(update_row.get("probability")) - 1.0)
                ),
                "evidence_improvement": 0.5
                * (
                    float(update_features["anchor_distance_improvement"])
                    + float(update_features["evidence_distance_improvement"])
                ),
                "verifier_score": (
                    0.0 if verifier_score is None else float(verifier_score)
                ),
                "verifier_delta": (
                    0.0 if verifier_delta is None else float(verifier_delta)
                ),
                "policy_uncertainty": float(state_signal["uncertainty"]),
                "expected_gain_lcb": float(evidence["gain_used"]),
                "expected_cost_ucb": float(evidence["cost_used"]),
                "expected_net_value": (
                    float(evidence["gain_used"]) - float(evidence["cost_used"])
                ),
                "budget_remaining_fraction": float(
                    state_signal["budget_remaining_fraction"]
                ),
                "proposal_accepted": 1.0 if update_row.get("accepted") else 0.0,
                "quality_measured": (
                    1.0 if update_acceptance.get("mode") == LEARNED else 0.0
                ),
                "evoc_measured": 1.0 if evidence.get("measured") else 0.0,
                "verifier_available": 1.0 if verifier_score is not None else 0.0,
            }
            for name in STOP_FEATURE_NAMES:
                observed = float(row["features"][name])
                expected = _bounded(float(expected_features[name]))
                if not math.isclose(
                    observed,
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        "stop-gate feature differs from causal source: "
                        f"{name} observed={observed} expected={expected}"
                    )
            probability = expected_gate.probability_for_features(row["features"])
            evidence_ready = bool(
                row["features"]["quality_measured"] >= 0.5
                and row["features"]["evoc_measured"] >= 0.5
            )
            expected_halt = bool(
                expected_gate.mode == LEARNED
                and evidence_ready
                and probability >= expected_gate.threshold
            )
            expected_reason = (
                "residual_policy"
                if expected_gate.mode == RESIDUAL
                else "learned_stop"
                if expected_halt
                else "continue_unmeasured_evidence"
                if not evidence_ready
                else "continue_expected_value_positive"
            )
            if (
                not math.isclose(
                    float(row["probability"]),
                    probability,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or row["evidence_ready"] is not evidence_ready
                or row["halt"] is not expected_halt
                or row["reason"] != expected_reason
            ):
                raise ValueError("stop-gate decision differs from its head")
            branch_halts += expected_halt
        if (
            branch["learned_halts"] != branch_halts
            or (
                branch_halts > 0
                and (
                    not branch["halt_reason"].startswith("learned_stop")
                    or not decisions[-1]["halt"]
                    or branch["steps_taken"] != decisions[-1]["step"]
                )
            )
            or branch_halts > 1
        ):
            raise ValueError("stop-gate branch halt summary is invalid")
        total_decisions += len(decisions)
        total_halts += branch_halts
    if (
        value["decision_count"] != total_decisions
        or value["learned_halts"] != total_halts
        or value["head_was_causal"]
        is not (expected_gate.mode == LEARNED and total_halts > 0)
        or (expected_gate.mode == RESIDUAL and total_decisions != 0)
    ):
        raise ValueError("stop-gate aggregate verdict is invalid")
    return dict(value)


__all__ = [
    "LEARNED",
    "RESIDUAL",
    "STOP_GATE_SCHEMA",
    "StopContext",
    "StopGateDecision",
    "StopGateRuntime",
    "build_stop_gate_receipt",
    "validate_stop_gate_receipt",
]
