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

RUNTIME_OPERATION_SCHEMA = "aura.rlc.runtime_operation.v1"
RUNTIME_OPERATION_OPERATOR_ID = "latent_execution_controller"
RUNTIME_OPERATION_OPERATOR_VERSION = "v2"

_AUTHORITY_FIELDS = {
    "schema",
    "episode_id",
    "objective_sha256",
    "input_state_sha256",
    "admitted_state_sha256",
    "admitted_state_version",
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
    return {
        "objective_sha256": state.problem.objective_sha256,
        "decision_sha256": canonical_sha256(normalized_decision),
        "config_sha256": canonical_sha256(normalized_config),
        "budget_sha256": canonical_sha256(normalized_budget),
        "action_policy_sha256": normalized_action_policy["snapshot_sha256"],
        "controller": normalized_decision,
    }


def validate_runtime_operation_authority(
    authority: Any,
    *,
    prompt: str | None,
    messages: list | None,
    config: Mapping[str, Any],
    budget: Mapping[str, Any],
    cognitive_context: list | None,
    action_policy_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the worker-wire authority without trusting service objects."""

    if not isinstance(authority, Mapping) or set(authority) != _AUTHORITY_FIELDS:
        raise EpistemicStateError("runtime operation authority fields differ")
    normalized = dict(authority)
    if normalized.get("schema") != RUNTIME_OPERATION_SCHEMA:
        raise EpistemicStateError("runtime operation authority schema is invalid")
    for name in (
        "objective_sha256",
        "input_state_sha256",
        "admitted_state_sha256",
        "attempt_sha256",
        "input_payload_sha256",
        "decision_sha256",
        "config_sha256",
        "budget_sha256",
        "action_policy_sha256",
    ):
        if not _is_digest(normalized.get(name)):
            raise EpistemicStateError(f"runtime operation authority {name} is invalid")
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
    if type(version) is not int or version < 1:
        raise EpistemicStateError("runtime operation admitted state version is invalid")
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
        return cls(
            journal=journal,
            state=admitted_state,
            decision=normalized_decision,
            kind=kind,
            payload=payload,
            intent=intent,
            authority=authority,
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
        return {
            "schema": RUNTIME_OPERATION_SCHEMA,
            "authority": dict(self.authority),
            "intent": self.intent.to_dict(),
            "terminal": self._terminal.to_dict() if self._terminal is not None else None,
            "action_operations": [
                operation.to_dict() for operation in self._action_operations
            ],
            "completed": self._completed,
            "current_state_sha256": self.state.state_sha256,
            "current_state_version": self.state.version,
            "journal": recovery.to_dict() if recovery is not None else {},
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
    "validate_runtime_operation_authority",
]
