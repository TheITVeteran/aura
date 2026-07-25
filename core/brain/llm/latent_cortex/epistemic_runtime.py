"""Durable operation admission for live Recursive Latent Cortex episodes.

The execution controller used to influence an in-memory config and later emit a
separate telemetry row.  That left the live recurrent computation outside the
transactional epistemic state.  This module binds one controller decision to
the exact objective, config, budget, and worker request, persists an in-flight
attempt before compute, and completes it through the existing retry lineage.

The pre-execution record deliberately has outcome UNKNOWN and zero cost.  If
the process disappears, recovery therefore says exactly what is known: the
operation was admitted but no terminal receipt was durably observed.  A normal
completion is an explicit retry carrying the measured cost and terminal
outcome; history is never rewritten in place.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.epistemic_journal import (
    EpistemicStateJournal,
)
from core.brain.llm.latent_cortex.epistemic_state import (
    EpistemicState,
    EpistemicStateError,
    EpistemicTransaction,
    OperationKind,
    OperationOutcome,
    OperationRecord,
    canonical_sha256,
)
from core.governance_context import local_internal_governed_scope

RUNTIME_OPERATION_SCHEMA = "aura.rlc.runtime_operation.v2"
RUNTIME_OPERATION_OPERATOR_ID = "latent_execution_controller"
RUNTIME_OPERATION_OPERATOR_VERSION = "v2"

_AUTHORITY_FIELDS = {
    "schema",
    "episode_id",
    "objective_sha256",
    "input_state_sha256",
    "admitted_state_sha256",
    "admitted_state_version",
    "admitted_journal_head_sha256",
    "admitted_journal_entry_count",
    "operation_id",
    "operation_kind",
    "operator_id",
    "operator_version",
    "attempt_sha256",
    "input_payload_sha256",
    "decision_sha256",
    "config_sha256",
    "budget_sha256",
    "action_policy_sha256",
    "controller_schema",
    "controller_bucket",
    "controller_arm",
    "controller_mode",
    "controller_evidence",
    "input_claim_ids",
    "input_hypothesis_ids",
    "input_evidence_ids",
    "admission_reason",
    "retry_of_operation_id",
}
_OPTIONAL_AUTHORITY_FIELDS = {"external_execution_offer_sha256"}


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _bounded_text(value: Any, *, name: str, limit: int = 160) -> str:
    if not isinstance(value, str):
        raise EpistemicStateError(f"{name} must be a string")
    rendered = value.strip()
    if not rendered or len(rendered) > limit or any(ord(char) < 32 for char in rendered):
        raise EpistemicStateError(f"{name} is not a bounded printable string")
    return rendered


def _normalized_external_execution_offer(value: Any) -> dict[str, Any]:
    try:
        from core.brain.llm.latent_cortex.external_execution import (
            validate_external_execution_offer,
        )

        return validate_external_execution_offer(value)
    except (ImportError, TypeError, ValueError) as exc:
        raise EpistemicStateError(
            "external execution offer is invalid"
        ) from exc


def _visible_objective(prompt: str | None, messages: list | None) -> str:
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    raise EpistemicStateError("runtime operation has no visible objective")


def operation_kind_for_decision(
    decision: Mapping[str, Any],
    config: Mapping[str, Any],
) -> OperationKind:
    """Name the cognitive action the selected configuration actually runs."""

    arm = _bounded_text(decision.get("arm"), name="controller arm", limit=64)
    if arm == "probe_guided_bytecode":
        return OperationKind.CHECK_ASSUMPTION
    branches = config.get("n_branches", 1)
    if type(branches) is not int or branches < 1:
        raise EpistemicStateError("runtime operation n_branches is invalid")
    if branches > 1:
        return OperationKind.BRANCH
    return OperationKind.BLIND_RESOLVE


def _normalized_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(decision, Mapping):
        raise EpistemicStateError("controller decision must be an object")
    normalized = {
        "schema": _bounded_text(
            decision.get("schema"), name="controller schema", limit=96
        ),
        "bucket": _bounded_text(
            decision.get("bucket"), name="controller bucket", limit=160
        ),
        "arm": _bounded_text(decision.get("arm"), name="controller arm", limit=64),
        "mode": _bounded_text(decision.get("mode"), name="controller mode", limit=32),
        "evidence": dict(decision.get("evidence") or {}),
    }
    canonical_sha256(normalized)
    return normalized


def runtime_operation_payload(
    *,
    state: EpistemicState,
    decision: Mapping[str, Any],
    config: Mapping[str, Any],
    budget: Mapping[str, Any],
    action_policy_evidence: Mapping[str, Any],
    external_execution_offer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(state, EpistemicState):
        raise TypeError("state must be an EpistemicState")
    if (
        not isinstance(config, Mapping)
        or not isinstance(budget, Mapping)
        or not isinstance(action_policy_evidence, Mapping)
    ):
        raise EpistemicStateError(
            "runtime operation config, budget, and action policy must be objects"
        )
    from core.brain.llm.latent_cortex.value_of_computation import (
        validate_evidence_snapshot,
    )

    normalized_action_policy = validate_evidence_snapshot(action_policy_evidence)
    normalized_decision = _normalized_decision(decision)
    normalized_config = dict(config)
    normalized_budget = dict(budget)
    payload = {
        "objective_sha256": state.problem.objective_sha256,
        "decision_sha256": canonical_sha256(normalized_decision),
        "config_sha256": canonical_sha256(normalized_config),
        "budget_sha256": canonical_sha256(normalized_budget),
        "action_policy_sha256": normalized_action_policy["snapshot_sha256"],
        "controller": normalized_decision,
    }
    if external_execution_offer is not None:
        offer = _normalized_external_execution_offer(external_execution_offer)
        if offer["objective_sha256"] != state.problem.objective_sha256:
            raise EpistemicStateError(
                "external execution offer objective differs from operation state"
            )
        payload["external_execution_offer_sha256"] = offer["offer_sha256"]
    return payload


def validate_runtime_operation_authority(
    authority: Any,
    *,
    prompt: str | None,
    messages: list | None,
    config: Mapping[str, Any],
    budget: Mapping[str, Any],
    cognitive_context: list | None,
    action_policy_evidence: Mapping[str, Any] | None = None,
    external_execution_offer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the worker-wire authority without trusting service objects."""

    authority_fields = set(authority) if isinstance(authority, Mapping) else set()
    if (
        not isinstance(authority, Mapping)
        or not _AUTHORITY_FIELDS.issubset(authority_fields)
        or authority_fields - _AUTHORITY_FIELDS != (
            _OPTIONAL_AUTHORITY_FIELDS if external_execution_offer is not None else set()
        )
    ):
        raise EpistemicStateError("runtime operation authority fields differ")
    normalized = dict(authority)
    if normalized.get("schema") != RUNTIME_OPERATION_SCHEMA:
        raise EpistemicStateError("runtime operation authority schema is invalid")
    for name in (
        "objective_sha256",
        "input_state_sha256",
        "admitted_state_sha256",
        "admitted_journal_head_sha256",
        "attempt_sha256",
        "input_payload_sha256",
        "decision_sha256",
        "config_sha256",
        "budget_sha256",
        "action_policy_sha256",
    ):
        if not _is_digest(normalized.get(name)):
            raise EpistemicStateError(f"runtime operation authority {name} is invalid")
    if external_execution_offer is not None:
        normalized_offer = _normalized_external_execution_offer(
            external_execution_offer
        )
        if (
            normalized.get("external_execution_offer_sha256")
            != normalized_offer["offer_sha256"]
        ):
            raise EpistemicStateError(
                "runtime operation external execution offer differs"
            )
    for name, limit in (
        ("episode_id", 96),
        ("operation_id", 96),
        ("operation_kind", 64),
        ("operator_id", 96),
        ("operator_version", 96),
        ("controller_schema", 96),
        ("controller_bucket", 160),
        ("controller_arm", 64),
        ("controller_mode", 32),
        ("admission_reason", 96),
    ):
        normalized[name] = _bounded_text(normalized.get(name), name=name, limit=limit)
    controller_evidence = normalized.get("controller_evidence")
    if not isinstance(controller_evidence, Mapping):
        raise EpistemicStateError("runtime operation controller evidence is invalid")
    normalized["controller_evidence"] = dict(controller_evidence)
    canonical_sha256(normalized["controller_evidence"])
    input_refs: dict[str, tuple[str, ...]] = {}
    for name in ("input_claim_ids", "input_hypothesis_ids", "input_evidence_ids"):
        values = normalized.get(name)
        if (
            not isinstance(values, list)
            or len(values) > 128
            or any(
                not isinstance(value, str)
                or not value
                or len(value) > 96
                or any(ord(char) < 33 for char in value)
                for value in values
            )
            or len(set(values)) != len(values)
        ):
            raise EpistemicStateError(f"runtime operation {name} is invalid")
        input_refs[name] = tuple(sorted(values))
        normalized[name] = list(input_refs[name])
    retry = normalized.get("retry_of_operation_id")
    if not isinstance(retry, str) or len(retry) > 96:
        raise EpistemicStateError("runtime operation retry identifier is invalid")
    version = normalized.get("admitted_state_version")
    journal_entry_count = normalized.get("admitted_journal_entry_count")
    if (
        type(version) is not int
        or version < 1
        or type(journal_entry_count) is not int
        or journal_entry_count != version + 1
    ):
        raise EpistemicStateError(
            "runtime operation admitted state or journal version is invalid"
        )
    try:
        operation_kind = OperationKind(normalized["operation_kind"])
    except ValueError as exc:
        raise EpistemicStateError("runtime operation kind is unsupported") from exc
    if (
        normalized["operator_id"] != RUNTIME_OPERATION_OPERATOR_ID
        or normalized["operator_version"] != RUNTIME_OPERATION_OPERATOR_VERSION
    ):
        raise EpistemicStateError("runtime operation producer identity is invalid")
    objective = _visible_objective(prompt, messages)
    # ProblemFrame hashes raw objective text, not its JSON string encoding.
    from core.brain.llm.latent_cortex.epistemic_state import text_sha256

    if text_sha256(objective) != normalized["objective_sha256"]:
        raise EpistemicStateError("runtime operation objective digest mismatches")
    if canonical_sha256(dict(config)) != normalized["config_sha256"]:
        raise EpistemicStateError("runtime operation config digest mismatches")
    if canonical_sha256(dict(budget)) != normalized["budget_sha256"]:
        raise EpistemicStateError("runtime operation budget digest mismatches")
    from core.brain.llm.latent_cortex.value_of_computation import (
        build_evidence_snapshot,
        validate_evidence_snapshot,
    )

    normalized_action_policy = validate_evidence_snapshot(
        action_policy_evidence
        if action_policy_evidence is not None
        else build_evidence_snapshot(
            bucket=normalized["controller_bucket"],
            cells={},
        )
    )
    if (
        normalized_action_policy["snapshot_sha256"]
        != normalized["action_policy_sha256"]
    ):
        raise EpistemicStateError("runtime operation action policy digest mismatches")
    decision = {
        "schema": normalized["controller_schema"],
        "bucket": normalized["controller_bucket"],
        "arm": normalized["controller_arm"],
        "mode": normalized["controller_mode"],
        "evidence": normalized["controller_evidence"],
    }
    if canonical_sha256(decision) != normalized["decision_sha256"]:
        raise EpistemicStateError("runtime operation decision digest mismatches")
    if operation_kind_for_decision(decision, config) is not operation_kind:
        raise EpistemicStateError("runtime operation kind mismatches selected execution")
    payload = {
        "objective_sha256": normalized["objective_sha256"],
        "decision_sha256": normalized["decision_sha256"],
        "config_sha256": normalized["config_sha256"],
        "budget_sha256": normalized["budget_sha256"],
        "action_policy_sha256": normalized["action_policy_sha256"],
        "controller": decision,
    }
    if external_execution_offer is not None:
        payload["external_execution_offer_sha256"] = normalized[
            "external_execution_offer_sha256"
        ]
    if canonical_sha256(payload) != normalized["input_payload_sha256"]:
        raise EpistemicStateError("runtime operation payload digest mismatches")
    expected_attempt = OperationRecord.compute_attempt_sha256(
        kind=operation_kind,
        operator_id=normalized["operator_id"],
        operator_version=normalized["operator_version"],
        input_payload_sha256=normalized["input_payload_sha256"],
        input_claim_ids=input_refs["input_claim_ids"],
        input_hypothesis_ids=input_refs["input_hypothesis_ids"],
        input_evidence_ids=input_refs["input_evidence_ids"],
    )
    if expected_attempt != normalized["attempt_sha256"]:
        raise EpistemicStateError("runtime operation attempt digest mismatches")
    if normalized["operation_id"] != f"rlc-op-{expected_attempt[:20]}-a1":
        raise EpistemicStateError("runtime operation identifier mismatches attempt")
    memory_state_hashes = {
        item.get("epistemic_state_sha256")
        for item in (cognitive_context or [])
        if isinstance(item, Mapping) and item.get("context_role") == "memory_observation"
    }
    if memory_state_hashes and memory_state_hashes != {
        normalized["admitted_state_sha256"]
    }:
        raise EpistemicStateError("runtime operation and memory state authority differ")
    return normalized


def validate_completed_runtime_operation_receipt(
    receipt: Any,
    *,
    external_execution_offer: Mapping[str, Any],
    action_policy_evidence: Mapping[str, Any],
    action_policy_receipt: Mapping[str, Any],
    cognitive_action_trace: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the durable host completion that authorized one worker trace."""

    expected_fields = {
        "schema",
        "authority",
        "intent",
        "terminal",
        "action_operations",
        "completed",
        "admitted_state",
        "current_state_sha256",
        "current_state_version",
        "current_state",
        "journal",
        "compute",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_fields:
        raise EpistemicStateError("runtime operation completion fields differ")
    if receipt.get("schema") != RUNTIME_OPERATION_SCHEMA:
        raise EpistemicStateError("runtime operation completion schema is invalid")
    authority = receipt.get("authority")
    authority_fields = set(authority) if isinstance(authority, Mapping) else set()
    if (
        not isinstance(authority, Mapping)
        or authority_fields != _AUTHORITY_FIELDS | _OPTIONAL_AUTHORITY_FIELDS
    ):
        raise EpistemicStateError("runtime operation completion authority differs")
    normalized_authority = dict(authority)
    if normalized_authority.get("schema") != RUNTIME_OPERATION_SCHEMA:
        raise EpistemicStateError("runtime operation authority schema is invalid")
    for name in (
        "objective_sha256",
        "input_state_sha256",
        "admitted_state_sha256",
        "admitted_journal_head_sha256",
        "attempt_sha256",
        "input_payload_sha256",
        "decision_sha256",
        "config_sha256",
        "budget_sha256",
        "action_policy_sha256",
        "external_execution_offer_sha256",
    ):
        if not _is_digest(normalized_authority.get(name)):
            raise EpistemicStateError(
                f"runtime operation completion authority {name} is invalid"
            )
    from core.brain.llm.latent_cortex.external_execution import (
        validate_external_execution_offer,
    )
    from core.brain.llm.latent_cortex.value_of_computation import (
        validate_action_trace,
        validate_evidence_snapshot,
    )

    offer = validate_external_execution_offer(external_execution_offer)
    evidence = validate_evidence_snapshot(action_policy_evidence)
    if (
        normalized_authority["external_execution_offer_sha256"]
        != offer["offer_sha256"]
        or normalized_authority["objective_sha256"] != offer["objective_sha256"]
        or normalized_authority["action_policy_sha256"]
        != evidence["snapshot_sha256"]
    ):
        raise EpistemicStateError("runtime operation completion authority differs")
    for name, limit in (
        ("episode_id", 96),
        ("operation_id", 96),
        ("operation_kind", 64),
        ("operator_id", 96),
        ("operator_version", 96),
        ("controller_schema", 96),
        ("controller_bucket", 160),
        ("controller_arm", 64),
        ("controller_mode", 32),
        ("admission_reason", 96),
    ):
        normalized_authority[name] = _bounded_text(
            normalized_authority.get(name),
            name=name,
            limit=limit,
        )
    controller_evidence = normalized_authority.get("controller_evidence")
    if not isinstance(controller_evidence, Mapping):
        raise EpistemicStateError(
            "runtime operation completion controller evidence is invalid"
        )
    normalized_authority["controller_evidence"] = dict(controller_evidence)
    canonical_sha256(normalized_authority["controller_evidence"])
    input_refs: dict[str, tuple[str, ...]] = {}
    for name in ("input_claim_ids", "input_hypothesis_ids", "input_evidence_ids"):
        values = normalized_authority.get(name)
        if (
            not isinstance(values, list)
            or len(values) > 128
            or any(
                not isinstance(value, str)
                or not value
                or len(value) > 96
                or any(ord(char) < 33 for char in value)
                for value in values
            )
            or len(set(values)) != len(values)
        ):
            raise EpistemicStateError(
                f"runtime operation completion {name} is invalid"
            )
        input_refs[name] = tuple(sorted(values))
        normalized_authority[name] = list(input_refs[name])
    retry = normalized_authority.get("retry_of_operation_id")
    admitted_version = normalized_authority.get("admitted_state_version")
    admitted_journal_entry_count = normalized_authority.get(
        "admitted_journal_entry_count"
    )
    if (
        not isinstance(retry, str)
        or len(retry) > 96
        or type(admitted_version) is not int
        or admitted_version < 1
        or type(admitted_journal_entry_count) is not int
        or admitted_journal_entry_count != admitted_version + 1
    ):
        raise EpistemicStateError(
            "runtime operation completion retry, state, or journal version is invalid"
        )
    try:
        operation_kind = OperationKind(normalized_authority["operation_kind"])
    except ValueError as exc:
        raise EpistemicStateError(
            "runtime operation completion kind is unsupported"
        ) from exc
    if (
        normalized_authority["operator_id"] != RUNTIME_OPERATION_OPERATOR_ID
        or normalized_authority["operator_version"]
        != RUNTIME_OPERATION_OPERATOR_VERSION
    ):
        raise EpistemicStateError(
            "runtime operation completion producer identity is invalid"
        )
    decision = {
        "schema": normalized_authority["controller_schema"],
        "bucket": normalized_authority["controller_bucket"],
        "arm": normalized_authority["controller_arm"],
        "mode": normalized_authority["controller_mode"],
        "evidence": normalized_authority["controller_evidence"],
    }
    if canonical_sha256(decision) != normalized_authority["decision_sha256"]:
        raise EpistemicStateError(
            "runtime operation completion decision digest differs"
        )
    operation_payload = {
        "objective_sha256": normalized_authority["objective_sha256"],
        "decision_sha256": normalized_authority["decision_sha256"],
        "config_sha256": normalized_authority["config_sha256"],
        "budget_sha256": normalized_authority["budget_sha256"],
        "action_policy_sha256": normalized_authority["action_policy_sha256"],
        "controller": decision,
        "external_execution_offer_sha256": normalized_authority[
            "external_execution_offer_sha256"
        ],
    }
    if (
        canonical_sha256(operation_payload)
        != normalized_authority["input_payload_sha256"]
    ):
        raise EpistemicStateError(
            "runtime operation completion payload digest differs"
        )
    expected_attempt = OperationRecord.compute_attempt_sha256(
        kind=operation_kind,
        operator_id=normalized_authority["operator_id"],
        operator_version=normalized_authority["operator_version"],
        input_payload_sha256=normalized_authority["input_payload_sha256"],
        input_claim_ids=input_refs["input_claim_ids"],
        input_hypothesis_ids=input_refs["input_hypothesis_ids"],
        input_evidence_ids=input_refs["input_evidence_ids"],
    )
    if (
        expected_attempt != normalized_authority["attempt_sha256"]
        or normalized_authority["operation_id"]
        != f"rlc-op-{expected_attempt[:20]}-a1"
    ):
        raise EpistemicStateError(
            "runtime operation completion attempt lineage differs"
        )
    raw_executors = action_policy_receipt.get("executors")
    try:
        executors = tuple(OperationKind(row) for row in raw_executors)
    except (TypeError, ValueError) as exc:
        raise EpistemicStateError(
            "runtime operation completion executor inventory is invalid"
        ) from exc
    if not executors or len(set(executors)) != len(executors):
        raise EpistemicStateError(
            "runtime operation completion executor inventory is invalid"
        )
    trace = validate_action_trace(
        cognitive_action_trace,
        evidence_snapshot=evidence,
        executors=executors,
    )
    intent = OperationRecord.from_dict(receipt.get("intent"))
    terminal = OperationRecord.from_dict(receipt.get("terminal"))
    admitted_state = EpistemicState.from_dict(receipt.get("admitted_state"))
    current_state = EpistemicState.from_dict(receipt.get("current_state"))
    if (
        receipt.get("completed") is not True
        or terminal.outcome is not OperationOutcome.SUCCEEDED
        or terminal.failure_code
        or intent.outcome is not OperationOutcome.UNKNOWN
        or intent.failure_code != "execution_pending"
        or intent.operation_id != normalized_authority.get("operation_id")
        or intent.kind.value != normalized_authority.get("operation_kind")
        or intent.operator_id != normalized_authority.get("operator_id")
        or intent.operator_version != normalized_authority.get("operator_version")
        or intent.attempt_sha256 != normalized_authority.get("attempt_sha256")
        or intent.input_payload_sha256
        != normalized_authority.get("input_payload_sha256")
        or intent.input_state_sha256 != normalized_authority.get("input_state_sha256")
        or terminal.retry_of_operation_id != intent.operation_id
        or terminal.kind is not intent.kind
        or terminal.operator_id != intent.operator_id
        or terminal.operator_version != intent.operator_version
        or terminal.input_payload_sha256 != intent.input_payload_sha256
        or admitted_state.episode_id != normalized_authority["episode_id"]
        or admitted_state.problem.objective_sha256
        != normalized_authority["objective_sha256"]
        or admitted_state.state_sha256
        != normalized_authority["admitted_state_sha256"]
        or admitted_state.version
        != normalized_authority["admitted_state_version"]
        or intent not in admitted_state.operations
    ):
        raise EpistemicStateError("runtime operation completion lineage differs")
    raw_action_operations = receipt.get("action_operations")
    if (
        not isinstance(raw_action_operations, list)
        or len(raw_action_operations) != len(trace["rows"])
    ):
        raise EpistemicStateError("runtime operation action lineage differs")
    action_operations = [
        OperationRecord.from_dict(row) for row in raw_action_operations
    ]
    compute = receipt.get("compute")
    if not isinstance(compute, Mapping):
        raise EpistemicStateError("runtime operation compute receipt is invalid")
    expected_compute_fields = {
        "basis",
        "spent_layer_apps",
        "max_layer_apps",
        "fraction",
        "state_cost",
        "action_state_cost",
        "action_operation_count",
    }
    if set(compute) != expected_compute_fields:
        raise EpistemicStateError("runtime operation compute receipt fields differ")
    spent = compute.get("spent_layer_apps")
    maximum = compute.get("max_layer_apps")
    fraction = compute.get("fraction")
    state_cost = compute.get("state_cost")
    action_state_cost = compute.get("action_state_cost")
    if (
        compute.get("basis") != "token_layer_fraction_of_remaining_episode_budget"
        or type(spent) is not int
        or type(maximum) is not int
        or spent < 0
        or maximum <= 0
        or spent > maximum
        or not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not math.isfinite(float(fraction))
        or not math.isclose(float(fraction), spent / maximum, rel_tol=0.0, abs_tol=1e-12)
        or not isinstance(state_cost, (int, float))
        or isinstance(state_cost, bool)
        or not math.isfinite(float(state_cost))
        or float(state_cost) < 0.0
        or not isinstance(action_state_cost, (int, float))
        or isinstance(action_state_cost, bool)
        or not math.isfinite(float(action_state_cost))
        or float(action_state_cost) < 0.0
        or compute.get("action_operation_count") != len(action_operations)
    ):
        raise EpistemicStateError("runtime operation compute receipt is invalid")
    action_cost_total = math.fsum(operation.cost for operation in action_operations)
    if (
        not math.isclose(
            action_cost_total,
            float(action_state_cost),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            terminal.cost + action_cost_total,
            float(state_cost),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise EpistemicStateError("runtime operation compute cost lineage differs")
    failed_outcome_markers = ("refused", "unavailable", "nonfinite")
    for index, (operation, trace_row) in enumerate(
        zip(action_operations, trace["rows"], strict=True)
    ):
        decision = trace_row["decision"]
        transition = trace_row["transition"]
        action_failed = any(
            marker in transition["outcome"] for marker in failed_outcome_markers
        )
        expected_outcome = (
            OperationOutcome.FAILED if action_failed else OperationOutcome.SUCCEEDED
        )
        expected_failure = "action_execution_failed" if action_failed else ""
        expected_detail = (
            f"step={transition['step_index']}; mode={transition['mode']}; "
            f"outcome={transition['outcome']}; checked={transition['checked']}"
        )
        if (
            operation.kind.value != decision["action"]
            or operation.input_payload_sha256 != decision["decision_sha256"]
            or operation.operator_id != "value_of_computation"
            or operation.operator_version != "v1"
            or operation.operation_id
            != f"rlc-action-{decision['decision_sha256'][:20]}-{index:03d}"
            or operation.outcome is not expected_outcome
            or operation.failure_code != expected_failure
            or operation.input_state_sha256
            != normalized_authority["admitted_state_sha256"]
            or operation.input_claim_ids != intent.input_claim_ids
            or operation.input_hypothesis_ids != intent.input_hypothesis_ids
            or operation.input_evidence_ids != intent.input_evidence_ids
            or operation.affected_claim_ids
            or operation.affected_hypothesis_ids
            or operation.evidence_gained
            or operation.retry_of_operation_id
            or operation.started_at != intent.started_at
            or operation.completed_at != terminal.completed_at
            or operation.detail != expected_detail
        ):
            raise EpistemicStateError("runtime operation action lineage differs")
    current_version = receipt.get("current_state_version")
    expected_current = EpistemicTransaction(admitted_state).add_operation(
        terminal
    )
    for operation in action_operations:
        expected_current.add_operation(operation)
    reconstructed_current = expected_current.commit()
    journal = receipt.get("journal")
    journal_fields = {
        "state_sha256",
        "state_version",
        "entry_count",
        "previous_head_sha256",
        "head_sha256",
        "size_bytes",
        "repaired_torn_tail_bytes",
    }
    journal_entry = {
        "schema": "aura.rlc.epistemic_journal.v1",
        "sequence": current_state.version,
        "previous_entry_sha256": journal.get("previous_head_sha256"),
        "state_sha256": current_state.state_sha256,
        "state": current_state.to_dict(),
    }
    if (
        current_state != reconstructed_current
        or receipt.get("current_state_sha256") != current_state.state_sha256
        or not _is_digest(receipt.get("current_state_sha256"))
        or type(current_version) is not int
        or type(admitted_version) is not int
        or current_version != admitted_version + 1
        or current_version != current_state.version
        or not isinstance(journal, Mapping)
        or set(journal) != journal_fields
        or journal.get("state_sha256") != current_state.state_sha256
        or journal.get("state_version") != current_state.version
        or journal.get("entry_count") != admitted_journal_entry_count + 1
        or not _is_digest(journal.get("previous_head_sha256"))
        or journal.get("previous_head_sha256")
        != normalized_authority["admitted_journal_head_sha256"]
        or not _is_digest(journal.get("head_sha256"))
        or journal.get("head_sha256") != canonical_sha256(journal_entry)
        or type(journal.get("size_bytes")) is not int
        or journal["size_bytes"] <= 0
        or type(journal.get("repaired_torn_tail_bytes")) is not int
        or not 0
        <= journal["repaired_torn_tail_bytes"]
        <= journal["size_bytes"]
        or terminal.operation_id
        != f"rlc-op-{normalized_authority['attempt_sha256'][:20]}-a2"
        or terminal.input_state_sha256
        != normalized_authority["admitted_state_sha256"]
        or terminal.attempt_sha256 != intent.attempt_sha256
        or terminal.input_claim_ids != intent.input_claim_ids
        or terminal.input_hypothesis_ids != intent.input_hypothesis_ids
        or terminal.input_evidence_ids != intent.input_evidence_ids
        or terminal.affected_claim_ids
        or terminal.affected_hypothesis_ids
        or terminal.evidence_gained
        or terminal.started_at != intent.started_at
        or terminal.completed_at < terminal.started_at
        or terminal.detail
        != (
            "worker outcome=succeeded; "
            f"cost basis={compute['basis']}"
        )
    ):
        raise EpistemicStateError("runtime operation completion state is invalid")
    return {
        **dict(receipt),
        "authority": normalized_authority,
        "intent": intent.to_dict(),
        "terminal": terminal.to_dict(),
        "action_operations": [operation.to_dict() for operation in action_operations],
    }


def _history_extends(supplied: EpistemicState, recovered: EpistemicState) -> bool:
    """Whether recovered is a monotonic extension of the caller's state."""

    if supplied.episode_id != recovered.episode_id or supplied.problem != recovered.problem:
        return False
    if supplied.version > recovered.version:
        return False
    if supplied.evidence != recovered.evidence:
        return False
    if supplied.claims != recovered.claims or supplied.hypotheses != recovered.hypotheses:
        return False
    if supplied.calibrations != recovered.calibrations:
        return False
    if recovered.operations[: len(supplied.operations)] != supplied.operations:
        return False
    return supplied.budget.total == recovered.budget.total


@dataclass(slots=True)
class RuntimeOperationLease:
    """One admitted, durably pending live cognitive operation."""

    journal: EpistemicStateJournal
    state: EpistemicState
    decision: dict[str, Any]
    kind: OperationKind
    payload: dict[str, Any]
    intent: OperationRecord
    authority: dict[str, Any]
    admitted_state: EpistemicState
    journal_parent_sha256: str
    _completed: bool = False
    _terminal: OperationRecord | None = None
    _action_operations: tuple[OperationRecord, ...] = ()

    @classmethod
    def begin(
        cls,
        *,
        genesis: EpistemicState,
        state: EpistemicState,
        decision: Mapping[str, Any],
        config: Mapping[str, Any],
        budget: Mapping[str, Any],
        action_policy_evidence: Mapping[str, Any] | None = None,
        external_execution_offer: Mapping[str, Any] | None = None,
        root: str | Path,
        started_at: float | None = None,
    ) -> RuntimeOperationLease:
        if not isinstance(genesis, EpistemicState) or genesis.version != 0:
            raise EpistemicStateError("runtime operation requires episode genesis")
        direct_state = (
            isinstance(state, EpistemicState)
            and state.version == 1
            and state.parent_sha256 == genesis.state_sha256
        )
        genesis_state = (
            isinstance(state, EpistemicState)
            and state.version == 0
            and state.state_sha256 == genesis.state_sha256
        )
        if (
            not (direct_state or genesis_state)
            or state.episode_id != genesis.episode_id
            or state.problem != genesis.problem
            or state.budget.total != genesis.budget.total
            or state.budget.tool_calls_total != genesis.budget.tool_calls_total
        ):
            raise EpistemicStateError("runtime operation state does not extend genesis")
        normalized_decision = _normalized_decision(decision)
        from core.brain.llm.latent_cortex.value_of_computation import (
            build_evidence_snapshot,
            validate_evidence_snapshot,
        )

        normalized_action_policy = validate_evidence_snapshot(
            action_policy_evidence
            if action_policy_evidence is not None
            else build_evidence_snapshot(
                bucket=normalized_decision["bucket"],
                cells={},
            )
        )
        if normalized_action_policy["bucket"] != normalized_decision["bucket"]:
            raise EpistemicStateError(
                "runtime operation action policy belongs to another context bucket"
            )
        kind = operation_kind_for_decision(normalized_decision, config)
        payload = runtime_operation_payload(
            state=state,
            decision=normalized_decision,
            config=config,
            budget=budget,
            action_policy_evidence=normalized_action_policy,
            external_execution_offer=external_execution_offer,
        )
        input_payload_sha256 = canonical_sha256(payload)
        input_claim_ids = tuple(claim.claim_id for claim in state.claims)
        input_hypothesis_ids = tuple(
            hypothesis.hypothesis_id for hypothesis in state.hypotheses
        )
        input_evidence_ids = tuple(evidence.evidence_id for evidence in state.evidence)
        attempt_sha256 = OperationRecord.compute_attempt_sha256(
            kind=kind,
            operator_id=RUNTIME_OPERATION_OPERATOR_ID,
            operator_version=RUNTIME_OPERATION_OPERATOR_VERSION,
            input_payload_sha256=input_payload_sha256,
            input_claim_ids=input_claim_ids,
            input_hypothesis_ids=input_hypothesis_ids,
            input_evidence_ids=input_evidence_ids,
        )
        journal = EpistemicStateJournal(Path(root) / f"{state.episode_id}.jsonl")
        with local_internal_governed_scope(
            "rlc_runtime_operation", domain="state_mutation"
        ):
            recovered = journal.bootstrap(genesis)
            if (
                recovered.state_sha256 == genesis.state_sha256
                and state.state_sha256 != genesis.state_sha256
            ):
                journal.append(expected_base=genesis, candidate=state)
                recovered = state
        if not _history_extends(state, recovered):
            raise EpistemicStateError("runtime operation journal diverges from supplied state")

        attempts = recovered.operation_attempts(attempt_sha256)
        if attempts:
            latest = attempts[-1]
            if not (
                latest.outcome is OperationOutcome.UNKNOWN
                and latest.failure_code == "execution_pending"
            ):
                admission = recovered.operation_admission(attempt_sha256)
                raise EpistemicStateError(
                    f"runtime operation cannot resume: {admission.reason}"
                )
            intent = latest
            admitted_state = recovered
            admission_reason = "recovered_pending_operation"
        else:
            admission = recovered.operation_admission(attempt_sha256)
            if not admission.allowed:
                raise EpistemicStateError(
                    f"runtime operation admission refused: {admission.reason}"
                )
            observed = time.time() if started_at is None else float(started_at)
            if not math.isfinite(observed) or observed < 0.0:
                raise EpistemicStateError("runtime operation start time is invalid")
            intent = OperationRecord.create(
                operation_id=f"rlc-op-{attempt_sha256[:20]}-a1",
                kind=kind,
                outcome=OperationOutcome.UNKNOWN,
                input_state_sha256=recovered.state_sha256,
                cost=0.0,
                operator_id=RUNTIME_OPERATION_OPERATOR_ID,
                operator_version=RUNTIME_OPERATION_OPERATOR_VERSION,
                input_payload_sha256=input_payload_sha256,
                started_at=observed,
                completed_at=observed,
                input_claim_ids=input_claim_ids,
                input_hypothesis_ids=input_hypothesis_ids,
                input_evidence_ids=input_evidence_ids,
                failure_code="execution_pending",
                detail="operation admitted before worker execution; terminal receipt pending",
            )
            transaction = EpistemicTransaction(recovered).add_operation(intent)
            admitted_state = transaction.commit()
            with local_internal_governed_scope(
                "rlc_runtime_operation", domain="state_mutation"
            ):
                journal.append(expected_base=recovered, candidate=admitted_state)
            admission_reason = admission.reason

        authority = {
            "schema": RUNTIME_OPERATION_SCHEMA,
            "episode_id": admitted_state.episode_id,
            "objective_sha256": admitted_state.problem.objective_sha256,
            "input_state_sha256": intent.input_state_sha256,
            "admitted_state_sha256": admitted_state.state_sha256,
            "admitted_state_version": admitted_state.version,
            "operation_id": intent.operation_id,
            "operation_kind": kind.value,
            "operator_id": intent.operator_id,
            "operator_version": intent.operator_version,
            "attempt_sha256": intent.attempt_sha256,
            "input_payload_sha256": intent.input_payload_sha256,
            "decision_sha256": payload["decision_sha256"],
            "config_sha256": payload["config_sha256"],
            "budget_sha256": payload["budget_sha256"],
            "action_policy_sha256": payload["action_policy_sha256"],
            "controller_schema": normalized_decision["schema"],
            "controller_bucket": normalized_decision["bucket"],
            "controller_arm": normalized_decision["arm"],
            "controller_mode": normalized_decision["mode"],
            "controller_evidence": dict(normalized_decision["evidence"]),
            "input_claim_ids": list(intent.input_claim_ids),
            "input_hypothesis_ids": list(intent.input_hypothesis_ids),
            "input_evidence_ids": list(intent.input_evidence_ids),
            "admission_reason": admission_reason,
            "retry_of_operation_id": "",
        }
        if external_execution_offer is not None:
            authority["external_execution_offer_sha256"] = payload[
                "external_execution_offer_sha256"
            ]
        recovery = journal.last_recovery
        if recovery is None:
            raise EpistemicStateError("runtime operation journal head is unavailable")
        authority["admitted_journal_head_sha256"] = recovery.head_sha256
        authority["admitted_journal_entry_count"] = recovery.entry_count
        return cls(
            journal=journal,
            state=admitted_state,
            decision=normalized_decision,
            kind=kind,
            payload=payload,
            intent=intent,
            authority=authority,
            admitted_state=admitted_state,
            journal_parent_sha256=recovery.head_sha256,
        )

    def complete(
        self,
        *,
        outcome: OperationOutcome,
        cost: float,
        action_transitions: tuple[Mapping[str, Any], ...] = (),
        action_costs: tuple[float, ...] = (),
        failure_code: str = "",
        detail: str = "",
        completed_at: float | None = None,
    ) -> EpistemicState:
        if self._completed:
            raise EpistemicStateError("runtime operation lease is already complete")
        if outcome is OperationOutcome.UNKNOWN:
            raise EpistemicStateError("runtime operation completion cannot remain unknown")
        measured_cost = float(cost)
        if not math.isfinite(measured_cost) or measured_cost < 0.0:
            raise EpistemicStateError("runtime operation cost is invalid")
        if measured_cost > self.state.budget.total - self.state.budget.used + 1e-12:
            raise EpistemicStateError("runtime operation cost exceeds remaining budget")
        if not isinstance(action_transitions, tuple) or not isinstance(action_costs, tuple):
            raise EpistemicStateError("runtime action transitions and costs must be tuples")
        if len(action_transitions) != len(action_costs):
            raise EpistemicStateError("runtime action transition cost count differs")
        from core.brain.llm.latent_cortex.value_of_computation import (
            validate_action_transition,
        )

        try:
            normalized_transitions = tuple(
                validate_action_transition(row, require_checked=False)
                for row in action_transitions
            )
        except (TypeError, ValueError) as exc:
            raise EpistemicStateError("runtime action transition is invalid") from exc
        normalized_action_costs = tuple(float(item) for item in action_costs)
        if any(
            not math.isfinite(item) or item < 0.0
            for item in normalized_action_costs
        ):
            raise EpistemicStateError("runtime action cost is invalid")
        action_cost_total = math.fsum(normalized_action_costs)
        if action_cost_total > measured_cost + 1e-12:
            raise EpistemicStateError("runtime action costs exceed measured operation cost")
        wrapper_cost = max(0.0, measured_cost - action_cost_total)
        finished = time.time() if completed_at is None else float(completed_at)
        if not math.isfinite(finished) or finished < self.intent.started_at:
            raise EpistemicStateError("runtime operation completion time is invalid")
        admission = self.state.operation_admission(
            self.intent.attempt_sha256,
            retry_of_operation_id=self.intent.operation_id,
        )
        if not admission.allowed:
            raise EpistemicStateError(
                f"runtime operation completion refused: {admission.reason}"
            )
        terminal = OperationRecord.create(
            operation_id=f"rlc-op-{self.intent.attempt_sha256[:20]}-a2",
            kind=self.intent.kind,
            outcome=outcome,
            input_state_sha256=self.state.state_sha256,
            cost=wrapper_cost,
            operator_id=self.intent.operator_id,
            operator_version=self.intent.operator_version,
            input_payload_sha256=self.intent.input_payload_sha256,
            started_at=self.intent.started_at,
            completed_at=finished,
            input_claim_ids=self.intent.input_claim_ids,
            input_hypothesis_ids=self.intent.input_hypothesis_ids,
            input_evidence_ids=self.intent.input_evidence_ids,
            retry_of_operation_id=self.intent.operation_id,
            failure_code=failure_code,
            detail=detail,
        )
        action_operations: list[OperationRecord] = []
        failed_outcome_markers = ("refused", "unavailable", "nonfinite")
        for index, (transition, action_cost) in enumerate(
            zip(normalized_transitions, normalized_action_costs, strict=True)
        ):
            transition_outcome = transition["outcome"]
            action_failed = any(
                marker in transition_outcome for marker in failed_outcome_markers
            )
            action_operations.append(
                OperationRecord.create(
                    operation_id=(
                        f"rlc-action-{transition['decision_sha256'][:20]}-{index:03d}"
                    ),
                    kind=OperationKind(transition["action"]),
                    outcome=(
                        OperationOutcome.FAILED
                        if action_failed
                        else OperationOutcome.SUCCEEDED
                    ),
                    input_state_sha256=self.state.state_sha256,
                    cost=action_cost,
                    operator_id="value_of_computation",
                    operator_version="v1",
                    input_payload_sha256=transition["decision_sha256"],
                    started_at=self.intent.started_at,
                    completed_at=finished,
                    input_claim_ids=self.intent.input_claim_ids,
                    input_hypothesis_ids=self.intent.input_hypothesis_ids,
                    input_evidence_ids=self.intent.input_evidence_ids,
                    failure_code="action_execution_failed" if action_failed else "",
                    detail=(
                        f"step={transition['step_index']}; mode={transition['mode']}; "
                        f"outcome={transition_outcome}; checked={transition['checked']}"
                    ),
                )
            )
        transaction = EpistemicTransaction(self.state).add_operation(terminal)
        for action_operation in action_operations:
            transaction.add_operation(action_operation)
        candidate = transaction.commit()
        with local_internal_governed_scope(
            "rlc_runtime_operation", domain="state_mutation"
        ):
            self.journal.append(expected_base=self.state, candidate=candidate)
        self.state = candidate
        self._completed = True
        self._terminal = terminal
        self._action_operations = tuple(action_operations)
        return candidate

    def to_receipt(self) -> dict[str, Any]:
        recovery = self.journal.last_recovery
        journal_receipt = recovery.to_dict() if recovery is not None else {}
        if journal_receipt:
            journal_receipt["previous_head_sha256"] = (
                self.journal_parent_sha256
            )
        return {
            "schema": RUNTIME_OPERATION_SCHEMA,
            "authority": dict(self.authority),
            "intent": self.intent.to_dict(),
            "terminal": self._terminal.to_dict() if self._terminal is not None else None,
            "action_operations": [
                operation.to_dict() for operation in self._action_operations
            ],
            "completed": self._completed,
            "admitted_state": self.admitted_state.to_dict(),
            "current_state_sha256": self.state.state_sha256,
            "current_state_version": self.state.version,
            "current_state": self.state.to_dict(),
            "journal": journal_receipt,
        }


def measured_operation_cost(
    receipt: Any,
    *,
    requested_budget: Mapping[str, Any],
    state: EpistemicState,
) -> tuple[float, dict[str, Any]]:
    """Translate measured token-layer work into the state's unit budget."""

    remaining = max(0.0, state.budget.total - state.budget.used)
    requested_max = requested_budget.get("max_layer_apps")
    worker_budget = receipt.get("budget") if isinstance(receipt, Mapping) else None
    if type(requested_max) is not int or requested_max <= 0 or not isinstance(
        worker_budget, Mapping
    ):
        return 0.0, {
            "basis": "unmeasured",
            "spent_layer_apps": None,
            "max_layer_apps": requested_max if type(requested_max) is int else None,
        }
    spent = worker_budget.get("spent_layer_apps")
    worker_max = worker_budget.get("max_layer_apps")
    if (
        type(spent) is not int
        or type(worker_max) is not int
        or spent < 0
        or worker_max != requested_max
        or spent > worker_max
    ):
        raise EpistemicStateError("worker compute receipt does not match admitted budget")
    fraction = spent / worker_max
    cost = min(remaining, max(0.0, fraction * remaining))
    return cost, {
        "basis": "token_layer_fraction_of_remaining_episode_budget",
        "spent_layer_apps": spent,
        "max_layer_apps": worker_max,
        "fraction": round(fraction, 12),
        "state_cost": round(cost, 12),
    }


__all__ = [
    "RUNTIME_OPERATION_OPERATOR_ID",
    "RUNTIME_OPERATION_OPERATOR_VERSION",
    "RUNTIME_OPERATION_SCHEMA",
    "RuntimeOperationLease",
    "measured_operation_cost",
    "operation_kind_for_decision",
    "runtime_operation_payload",
    "validate_completed_runtime_operation_receipt",
    "validate_runtime_operation_authority",
]
