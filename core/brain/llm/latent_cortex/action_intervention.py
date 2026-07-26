"""Externally authorized interventions for causal cognitive-action campaigns.

The ordinary value-of-computation policy is observational: it chooses the
action it currently predicts will help.  A paired causal campaign needs a
different primitive.  It must force one preregistered action, or omit that
same action at the matched opportunity, without allowing an unsigned caller
to turn the research mechanism into a runtime policy override.

This module owns that narrow authority boundary.  It authenticates a
plan-bound intervention against the campaign runner key under an independently
configured trust root and validates the public receipt emitted by the resident
worker.  The worker never receives a private key.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Final

from core.brain.llm.latent_cortex.campaign_journal import (
    ACTION_INTERVENTION_CLAIMED,
    ARM_RESULT,
    COMMITTED,
    EVENT_SCHEMA,
    FAILED,
    PLAN_EVENT,
    STARTED,
    VERIFIED,
    CampaignJournal,
    CampaignPlan,
)
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    TASK_ISSUER,
    VerifiedCampaignTrustPolicy,
    build_role_attestation,
    validate_campaign_trust_policy,
    verify_role_attestation,
)
from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.runtime.file_read_gateway import read_stable_bytes

ACTION_INTERVENTION_SCHEMA: Final = "aura.rlc.action_intervention.v3"
ACTION_INTERVENTION_AUTHORITY_SCHEMA: Final = "aura.rlc.action_intervention.authority.v3"
ACTION_INTERVENTION_RECEIPT_SCHEMA: Final = "aura.rlc.action_intervention.receipt.v3"
ACTION_INTERVENTION_REPLAY_SCHEMA: Final = "aura.rlc.action_intervention.replay.v3"
INTERVENTION_CONSUMED: Final = "CONSUMED"
INTERVENTION_EXECUTION_CLAIMED: Final = "EXECUTION_CLAIMED"
TREATMENT_ARM: Final = "forced_action"
CONTROL_ARM: Final = "matched_no_action"
INTERVENTION_ARMS: Final = (TREATMENT_ARM, CONTROL_ARM)
_TRUST_ROOT_ENV: Final = "AURA_RLC_ACTION_CALIBRATION_TRUST_ROOT"
_CURRENT_POLICY_ENV: Final = "AURA_RLC_ACTION_CALIBRATION_POLICY"
_REPLAY_LEDGER_ENV: Final = "AURA_RLC_ACTION_CALIBRATION_REPLAY_LEDGER"
_CAMPAIGN_JOURNAL_ENV: Final = "AURA_RLC_ACTION_CALIBRATION_JOURNAL"
_MAX_REPLAY_LEDGER_BYTES: Final = 32 * 1024 * 1024
_STARTING_STATE_COMPONENTS: Final = {
    "latent_slots_sha256",
    "branch_state_sha256",
    "kv_cache_sha256",
    "evidence_state_sha256",
    "memory_state_sha256",
    "public_action_state_sha256",
    "durable_state_sha256",
    "rng_state_sha256",
}

_AUTHORITY_FIELDS = {
    "schema",
    "campaign_name",
    "campaign_plan_sha256",
    "campaign_protocol_sha256",
    "policy_sha256",
    "policy_revision",
    "cell_id",
    "definition_sha256",
    "pair_id",
    "task_id",
    "task_payload_sha256",
    "starting_state_sha256",
    "starting_state_components",
    "expected_pre_state_sha256",
    "expected_pre_kv_sha256",
    "action",
    "arm",
    "intervention_ordinal",
    "execution_ordinal",
    "attempt_number",
    "attempt_id",
    "campaign_journal_path_sha256",
    "journal_head_sha256",
    "journal_event_count",
    "request_payload_sha256",
    "engine_request_sha256",
    "task_prompt_sha256",
}
_INTERVENTION_FIELDS = {
    "schema",
    "authority_payload",
    "campaign_plan",
    "campaign_journal_prefix",
    "policy_document",
    "task_issuer_attestation",
    "runner_attestation",
    "intervention_sha256",
}
_RECEIPT_FIELDS = {
    "schema",
    "intervention_sha256",
    "campaign_plan_sha256",
    "cell_id",
    "attempt_id",
    "request_payload_sha256",
    "arm",
    "action",
    "intervention_ordinal",
    "execution_ordinal",
    "attempt_number",
    "execution_claim",
    "execution_claim_sha256",
    "consumed",
    "selection_mode",
    "selected_action",
    "selected_action_occurrences",
    "action_excluded_after_intervention",
    "starting_state_components",
    "component_observation_owners",
    "pre_state_components",
    "post_state_components",
    "pre_state_sha256",
    "pre_kv_sha256",
    "post_state_sha256",
    "post_kv_sha256",
    "decision_sha256",
    "cognitive_action_trace_sha256",
    "receipt_sha256",
}
_CELL_FIELDS = {
    "action",
    "arm",
    "execution_ordinal",
    "pair_arm_ordinal",
    "pair_id",
    "starting_state",
    "starting_state_sha256",
    "task_id",
    "task_payload_sha256",
    "task_sampling_identity_sha256",
    "task_sampling_stratum_sha256",
}
_STARTING_STATE_FIELDS = {
    "schema",
    "capture_mode",
    "capture_id",
    "captured_at_unix",
    "campaign_name",
    "action",
    "task_id",
    "task_sampling_identity_sha256",
    "calibration_bucket",
    "bucket_classifier_sha256",
    "bucket_evidence_sha256",
    *_STARTING_STATE_COMPONENTS,
    "model_identity_sha256",
    "continuation_policy_sha256",
    "budget_policy_sha256",
    "state_sha256",
    "capture_attestation",
}
_REPLAY_EVENT_FIELDS = {
    "schema",
    "sequence",
    "previous_event_sha256",
    "event",
    "intervention_sha256",
    "attempt_id",
    "request_payload_sha256",
    "consumption_event_sha256",
    "campaign_claim_event_sha256",
    "recorded_at_unix",
    "event_sha256",
}


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise ValueError("action intervention is not canonical JSON") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identifier(value: Any, *, name: str, limit: int = 240) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > limit
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"action intervention {name} is invalid")
    return value


def action_intervention_attempt_id(
    *,
    campaign_plan_sha256: str,
    cell_id: str,
    attempt_number: int = 1,
) -> str:
    """Return the campaign journal's deterministic first-attempt identity."""

    if not _is_sha256(campaign_plan_sha256):
        raise ValueError("action intervention campaign plan digest is invalid")
    normalized_cell = _identifier(cell_id, name="cell_id")
    if type(attempt_number) is not int or not 1 <= attempt_number <= 1_000_000:
        raise ValueError("action intervention attempt number is invalid")
    return "attempt-" + _sha256(
        {
            "attempt_number": attempt_number,
            "cell_id": normalized_cell,
            "plan_sha256": campaign_plan_sha256,
            "schema": "aura.latent_cortex.campaign_attempt.v1",
        }
    )


def action_intervention_campaign_journal_sha256(path: str | Path) -> str:
    """Commit the configured canonical journal path without disclosing it."""

    candidate = Path(path).expanduser().absolute()
    return hashlib.sha256(str(candidate).encode("utf-8")).hexdigest()


def _state_components(value: Any) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _STARTING_STATE_COMPONENTS
        or any(not _is_sha256(value.get(name)) for name in _STARTING_STATE_COMPONENTS)
    ):
        raise ValueError("action intervention starting-state components are invalid")
    return {name: str(value[name]) for name in sorted(_STARTING_STATE_COMPONENTS)}


def _json_contract_value(value: Any, *, name: str) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    try:
        return json.loads(_canonical_json_bytes(value))
    except ValueError as exc:
        raise ValueError(f"action intervention {name} is not canonical JSON") from exc


def action_intervention_engine_request_sha256(
    *,
    prompt: str,
    domain: str,
    config: Any,
    budget: Any,
    cognitive_context: Sequence[Mapping[str, Any]],
    action_policy_evidence: Mapping[str, Any],
    external_execution_offer: Mapping[str, Any] | None,
    verifier_present: bool,
    ablate_slot: int | None = None,
    ablate_mode: str = "zero",
) -> str:
    """Commit the complete engine-visible semantics of one campaign request."""

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("action intervention engine prompt is invalid")
    if not isinstance(domain, str) or not domain:
        raise ValueError("action intervention engine domain is invalid")
    if type(verifier_present) is not bool:
        raise ValueError("action intervention verifier presence is invalid")
    if ablate_slot is not None or ablate_mode != "zero":
        raise ValueError("action intervention does not permit latent ablations")
    if not isinstance(cognitive_context, Sequence) or isinstance(
        cognitive_context, (str, bytes)
    ):
        raise ValueError("action intervention cognitive context is invalid")
    if not isinstance(action_policy_evidence, Mapping):
        raise ValueError("action intervention action policy evidence is invalid")
    if external_execution_offer is not None and not isinstance(
        external_execution_offer, Mapping
    ):
        raise ValueError("action intervention external execution offer is invalid")
    budget_contract = {
        "max_layer_apps": getattr(budget, "max_layer_apps", None),
        "wall_clock_s": getattr(budget, "wall_clock_s", None),
        "spent_layer_apps": getattr(budget, "spent_layer_apps", None),
    }
    if (
        type(budget_contract["max_layer_apps"]) is not int
        or budget_contract["max_layer_apps"] <= 0
        or not isinstance(budget_contract["wall_clock_s"], float)
        or budget_contract["wall_clock_s"] <= 0.0
        or budget_contract["spent_layer_apps"] != 0
    ):
        raise ValueError("action intervention engine budget is invalid")
    payload = {
        "schema": "aura.rlc.action_intervention.engine_request.v1",
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "domain": domain,
        "config": _json_contract_value(config, name="engine config"),
        "budget": budget_contract,
        "cognitive_context": _json_contract_value(
            list(cognitive_context),
            name="cognitive context",
        ),
        "action_policy_evidence": _json_contract_value(
            dict(action_policy_evidence),
            name="action policy evidence",
        ),
        "external_execution_offer": (
            None
            if external_execution_offer is None
            else _json_contract_value(
                dict(external_execution_offer),
                name="external execution offer",
            )
        ),
        "verifier_present": verifier_present,
        "ablate_slot": ablate_slot,
        "ablate_mode": ablate_mode,
    }
    return _sha256(payload)


def action_intervention_authority_payload(
    *,
    campaign_name: str,
    campaign_plan_sha256: str,
    campaign_protocol_sha256: str,
    policy_sha256: str,
    policy_revision: int,
    cell_id: str,
    definition_sha256: str,
    pair_id: str,
    task_id: str,
    task_payload_sha256: str,
    starting_state_sha256: str,
    starting_state_components: Mapping[str, str],
    expected_pre_state_sha256: str,
    expected_pre_kv_sha256: str,
    action: OperationKind | str,
    arm: str,
    execution_ordinal: int,
    attempt_number: int,
    attempt_id: str,
    campaign_journal_path_sha256: str,
    journal_head_sha256: str,
    journal_event_count: int,
    request_payload_sha256: str,
    engine_request_sha256: str,
    task_prompt_sha256: str,
) -> dict[str, Any]:
    """Build the exact payload the external campaign runner must sign."""

    try:
        normalized_action = action if isinstance(action, OperationKind) else OperationKind(action)
    except (TypeError, ValueError) as exc:
        raise ValueError("action intervention action is invalid") from exc
    normalized_arm = _identifier(arm, name="arm", limit=32)
    if normalized_arm not in INTERVENTION_ARMS:
        raise ValueError("action intervention arm is invalid")
    for name, digest in (
        ("campaign_plan_sha256", campaign_plan_sha256),
        ("campaign_protocol_sha256", campaign_protocol_sha256),
        ("policy_sha256", policy_sha256),
        ("definition_sha256", definition_sha256),
        ("task_payload_sha256", task_payload_sha256),
        ("starting_state_sha256", starting_state_sha256),
        ("expected_pre_state_sha256", expected_pre_state_sha256),
        ("expected_pre_kv_sha256", expected_pre_kv_sha256),
        ("request_payload_sha256", request_payload_sha256),
        ("engine_request_sha256", engine_request_sha256),
        ("campaign_journal_path_sha256", campaign_journal_path_sha256),
        ("journal_head_sha256", journal_head_sha256),
        ("task_prompt_sha256", task_prompt_sha256),
    ):
        if not _is_sha256(digest):
            raise ValueError(f"action intervention {name} is invalid")
    if type(policy_revision) is not int or policy_revision <= 0:
        raise ValueError("action intervention policy revision is invalid")
    if type(execution_ordinal) is not int or execution_ordinal < 0:
        raise ValueError("action intervention execution ordinal is invalid")
    if type(attempt_number) is not int or not 1 <= attempt_number <= 1_000_000:
        raise ValueError("action intervention attempt number is invalid")
    if type(journal_event_count) is not int or journal_event_count < 2:
        raise ValueError("action intervention journal event count is invalid")
    normalized_cell_id = _identifier(cell_id, name="cell_id")
    normalized_attempt_id = _identifier(attempt_id, name="attempt_id")
    expected_attempt_id = action_intervention_attempt_id(
        campaign_plan_sha256=campaign_plan_sha256,
        cell_id=normalized_cell_id,
        attempt_number=attempt_number,
    )
    if normalized_attempt_id != expected_attempt_id:
        raise ValueError("action intervention attempt identity differs")
    return {
        "schema": ACTION_INTERVENTION_AUTHORITY_SCHEMA,
        "campaign_name": _identifier(campaign_name, name="campaign_name"),
        "campaign_plan_sha256": campaign_plan_sha256,
        "campaign_protocol_sha256": campaign_protocol_sha256,
        "policy_sha256": policy_sha256,
        "policy_revision": policy_revision,
        "cell_id": normalized_cell_id,
        "definition_sha256": definition_sha256,
        "pair_id": _identifier(pair_id, name="pair_id"),
        "task_id": _identifier(task_id, name="task_id"),
        "task_payload_sha256": task_payload_sha256,
        "starting_state_sha256": starting_state_sha256,
        "starting_state_components": _state_components(starting_state_components),
        "expected_pre_state_sha256": expected_pre_state_sha256,
        "expected_pre_kv_sha256": expected_pre_kv_sha256,
        "action": normalized_action.value,
        "arm": normalized_arm,
        "intervention_ordinal": 0,
        "execution_ordinal": execution_ordinal,
        "attempt_number": attempt_number,
        "attempt_id": normalized_attempt_id,
        "campaign_journal_path_sha256": campaign_journal_path_sha256,
        "journal_head_sha256": journal_head_sha256,
        "journal_event_count": journal_event_count,
        "request_payload_sha256": request_payload_sha256,
        "engine_request_sha256": engine_request_sha256,
        "task_prompt_sha256": task_prompt_sha256,
    }


def _validate_plan_lineage(
    value: Any,
    *,
    authority: Mapping[str, Any],
    policy: VerifiedCampaignTrustPolicy,
    task_issuer_attestation: Mapping[str, Any],
) -> CampaignPlan:
    try:
        plan = CampaignPlan.from_dict(value)
        definition = plan.cell_definition(str(authority["cell_id"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("action intervention campaign plan is invalid") from exc
    document = plan.to_dict()
    metadata = document.get("metadata")
    try:
        from core.brain.llm.latent_cortex.action_calibration import (
            _validate_plan_sampling_frame,
            action_calibration_issuer_payload,
        )

        _validate_plan_sampling_frame(plan)
        verify_role_attestation(
            policy,
            task_issuer_attestation,
            role=TASK_ISSUER,
            expected_payload=action_calibration_issuer_payload(plan),
        )
    except (ImportError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "action intervention independent preregistration is invalid"
        ) from exc
    if (
        plan.plan_sha256 != authority["campaign_plan_sha256"]
        or plan.campaign_name != authority["campaign_name"]
        or not isinstance(metadata, Mapping)
        or metadata.get("schema") != "aura.rlc.action_calibration.protocol.v1"
        or not isinstance(metadata.get("campaign_trust"), Mapping)
        or metadata["campaign_trust"].get("policy_sha256") != policy.policy_sha256
        or set(definition) != _CELL_FIELDS
        or _sha256(definition) != authority["definition_sha256"]
    ):
        raise ValueError("action intervention campaign lineage differs")
    task_manifest = metadata.get("task_manifest")
    task_rows = (
        task_manifest.get("tasks") if isinstance(task_manifest, Mapping) else None
    )
    matching_tasks = [
        row
        for row in (task_rows or [])
        if isinstance(row, Mapping) and row.get("task_id") == authority["task_id"]
    ]
    if len(matching_tasks) != 1:
        raise ValueError("action intervention preregistered task is unavailable")
    task = matching_tasks[0]
    prompt = task.get("prompt")
    if (
        not isinstance(prompt, str)
        or not prompt
        or task.get("task_payload_sha256") != authority["task_payload_sha256"]
        or hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        != authority["task_prompt_sha256"]
    ):
        raise ValueError("action intervention preregistered task differs")
    starting_state = definition.get("starting_state")
    if not isinstance(starting_state, Mapping) or set(starting_state) != _STARTING_STATE_FIELDS:
        raise ValueError("action intervention starting-state receipt is invalid")
    capture_attestation = starting_state.get("capture_attestation")
    state_payload = {
        name: starting_state[name] for name in _STARTING_STATE_FIELDS - {"capture_attestation"}
    }
    state_body = {name: state_payload[name] for name in set(state_payload) - {"state_sha256"}}
    if (
        state_payload.get("schema") != "aura.rlc.action_calibration.state_capture.v1"
        or state_payload.get("capture_mode") != "externally_captured_runtime_state_v1"
        or state_payload.get("campaign_name") != plan.campaign_name
        or state_payload.get("action") != definition.get("action")
        or state_payload.get("task_id") != definition.get("task_id")
        or state_payload.get("state_sha256") != _sha256(state_body)
    ):
        raise ValueError("action intervention starting-state receipt differs")
    try:
        verify_role_attestation(
            policy,
            capture_attestation,
            role=CAMPAIGN_RUNNER,
            expected_payload=state_payload,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("action intervention starting-state attestation is invalid") from exc
    components = _state_components(
        {name: starting_state.get(name) for name in _STARTING_STATE_COMPONENTS}
    )
    expected_pre_state_sha256 = _sha256(components)
    if (
        definition.get("action") != authority["action"]
        or definition.get("arm") != authority["arm"]
        or definition.get("execution_ordinal") != authority["execution_ordinal"]
        or definition.get("pair_id") != authority["pair_id"]
        or definition.get("task_id") != authority["task_id"]
        or definition.get("task_payload_sha256") != authority["task_payload_sha256"]
        or definition.get("starting_state_sha256") != authority["starting_state_sha256"]
        or starting_state.get("state_sha256") != authority["starting_state_sha256"]
        or components != authority["starting_state_components"]
        or expected_pre_state_sha256 != authority["expected_pre_state_sha256"]
        or components["kv_cache_sha256"] != authority["expected_pre_kv_sha256"]
    ):
        raise ValueError("action intervention campaign cell differs")
    return plan


def _validate_journal_prefix(
    value: Any,
    *,
    plan: CampaignPlan,
    authority: Mapping[str, Any],
) -> list[dict[str, Any]]:
    event_fields = {
        "schema",
        "sequence",
        "plan_sha256",
        "previous_event_sha256",
        "event",
        "cell_id",
        "attempt_id",
        "payload",
        "event_sha256",
    }
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != authority["journal_event_count"]
    ):
        raise ValueError("action intervention campaign journal prefix is invalid")
    previous = "0" * 64
    attempts: dict[str, dict[str, Any]] = {}
    active_by_cell: dict[str, str] = {}
    start_counts: dict[str, int] = {}
    committed_cells: set[str] = set()
    transcript: list[dict[str, Any]] = []
    for sequence, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != event_fields:
            raise ValueError("action intervention campaign journal event differs")
        event = dict(raw)
        body = {name: event[name] for name in event_fields - {"event_sha256"}}
        if (
            event.get("schema") != EVENT_SCHEMA
            or event.get("sequence") != sequence
            or event.get("plan_sha256") != plan.plan_sha256
            or event.get("previous_event_sha256") != previous
            or event.get("event_sha256") != _sha256(body)
        ):
            raise ValueError("action intervention campaign journal chain differs")
        event_name = event.get("event")
        cell_id = event.get("cell_id")
        attempt_id = event.get("attempt_id")
        payload = event.get("payload")
        if sequence == 0:
            if (
                event_name != PLAN_EVENT
                or cell_id is not None
                or attempt_id is not None
                or payload != {"plan": plan.to_dict()}
            ):
                raise ValueError(
                    "action intervention campaign journal genesis differs"
                )
        else:
            if (
                not isinstance(cell_id, str)
                or cell_id not in plan.cell_ids
                or not isinstance(attempt_id, str)
                or not isinstance(payload, Mapping)
            ):
                raise ValueError(
                    "action intervention campaign journal identity differs"
                )
            if event_name == STARTED:
                attempt_number = start_counts.get(cell_id, 0) + 1
                if (
                    payload != {"attempt_number": attempt_number}
                    or cell_id in active_by_cell
                    or cell_id in committed_cells
                    or attempt_id in attempts
                    or attempt_id
                    != action_intervention_attempt_id(
                        campaign_plan_sha256=plan.plan_sha256,
                        cell_id=cell_id,
                        attempt_number=attempt_number,
                    )
                ):
                    raise ValueError(
                        "action intervention campaign journal attempt differs"
                    )
                start_counts[cell_id] = attempt_number
                attempts[attempt_id] = {
                    "cell_id": cell_id,
                    "attempt_number": attempt_number,
                    "state": STARTED,
                }
                active_by_cell[cell_id] = attempt_id
            else:
                attempt = attempts.get(attempt_id)
                if (
                    attempt is None
                    or attempt["cell_id"] != cell_id
                    or active_by_cell.get(cell_id) != attempt_id
                ):
                    raise ValueError(
                        "action intervention campaign journal attempt is inactive"
                    )
                state = attempt["state"]
                if event_name == ACTION_INTERVENTION_CLAIMED:
                    valid = (
                        state == STARTED
                        and set(payload)
                        == {
                            "intervention_sha256",
                            "request_payload_sha256",
                            "signed_journal_head_sha256",
                            "signed_journal_event_count",
                        }
                        and _is_sha256(payload.get("intervention_sha256"))
                        and _is_sha256(payload.get("request_payload_sha256"))
                        and payload.get("signed_journal_head_sha256")
                        == event.get("previous_event_sha256")
                        and payload.get("signed_journal_event_count") == sequence
                    )
                elif event_name == ARM_RESULT:
                    valid = state in {STARTED, ACTION_INTERVENTION_CLAIMED} and set(
                        payload
                    ) == {"result"}
                elif event_name == VERIFIED:
                    valid = state == ARM_RESULT and set(payload) == {"verification"}
                elif event_name == COMMITTED:
                    valid = state == VERIFIED and set(payload) == {"commit"}
                elif event_name == FAILED:
                    valid = (
                        set(payload) == {"details", "reason"}
                        and isinstance(payload.get("details"), Mapping)
                        and isinstance(payload.get("reason"), str)
                        and bool(payload["reason"].strip())
                    )
                else:
                    valid = False
                if not valid:
                    raise ValueError(
                        "action intervention campaign journal transition differs"
                    )
                attempt["state"] = event_name
                if event_name in {COMMITTED, FAILED}:
                    del active_by_cell[cell_id]
                if event_name == COMMITTED:
                    committed_cells.add(cell_id)
        previous = str(event["event_sha256"])
        transcript.append(event)
    target_attempt = attempts.get(str(authority["attempt_id"]))
    if (
        previous != authority["journal_head_sha256"]
        or active_by_cell.get(authority["cell_id"]) != authority["attempt_id"]
        or target_attempt is None
        or target_attempt["state"] != STARTED
        or target_attempt["attempt_number"] != authority["attempt_number"]
    ):
        raise ValueError("action intervention active journal attempt differs")
    return transcript


def build_action_intervention(
    *,
    policy: VerifiedCampaignTrustPolicy,
    runner_private_key: Any,
    signed_at_unix: int,
    authority_payload: Mapping[str, Any],
    campaign_plan: CampaignPlan | Mapping[str, Any],
    campaign_journal_prefix: Sequence[Mapping[str, Any]],
    task_issuer_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a runner-signed intervention for one preregistered cell."""

    normalized_authority = _validate_authority_payload(
        authority_payload,
        policy=policy,
    )
    plan = (
        campaign_plan
        if isinstance(campaign_plan, CampaignPlan)
        else CampaignPlan.from_dict(campaign_plan)
    )
    plan_document = plan.to_dict()
    _validate_plan_lineage(
        plan_document,
        authority=normalized_authority,
        policy=policy,
        task_issuer_attestation=task_issuer_attestation,
    )
    journal_prefix = _validate_journal_prefix(
        campaign_journal_prefix,
        plan=plan,
        authority=normalized_authority,
    )
    attestation = build_role_attestation(
        policy,
        role=CAMPAIGN_RUNNER,
        payload=normalized_authority,
        signed_at_unix=signed_at_unix,
        private_key=runner_private_key,
    )
    body = {
        "schema": ACTION_INTERVENTION_SCHEMA,
        "authority_payload": normalized_authority,
        "campaign_plan": plan_document,
        "campaign_journal_prefix": journal_prefix,
        "policy_document": dict(policy.document),
        "task_issuer_attestation": dict(task_issuer_attestation),
        "runner_attestation": attestation,
    }
    return {**body, "intervention_sha256": _sha256(body)}


def _validate_authority_payload(
    value: Any,
    *,
    policy: VerifiedCampaignTrustPolicy | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _AUTHORITY_FIELDS:
        raise ValueError("action intervention authority fields differ")
    if value.get("schema") != ACTION_INTERVENTION_AUTHORITY_SCHEMA:
        raise ValueError("action intervention authority schema is invalid")
    normalized = action_intervention_authority_payload(
        campaign_name=value.get("campaign_name"),
        campaign_plan_sha256=value.get("campaign_plan_sha256"),
        campaign_protocol_sha256=value.get("campaign_protocol_sha256"),
        policy_sha256=value.get("policy_sha256"),
        policy_revision=value.get("policy_revision"),
        cell_id=value.get("cell_id"),
        definition_sha256=value.get("definition_sha256"),
        pair_id=value.get("pair_id"),
        task_id=value.get("task_id"),
        task_payload_sha256=value.get("task_payload_sha256"),
        starting_state_sha256=value.get("starting_state_sha256"),
        starting_state_components=value.get("starting_state_components"),
        expected_pre_state_sha256=value.get("expected_pre_state_sha256"),
        expected_pre_kv_sha256=value.get("expected_pre_kv_sha256"),
        action=value.get("action"),
        arm=value.get("arm"),
        execution_ordinal=value.get("execution_ordinal"),
        attempt_number=value.get("attempt_number"),
        attempt_id=value.get("attempt_id"),
        campaign_journal_path_sha256=value.get("campaign_journal_path_sha256"),
        journal_head_sha256=value.get("journal_head_sha256"),
        journal_event_count=value.get("journal_event_count"),
        request_payload_sha256=value.get("request_payload_sha256"),
        engine_request_sha256=value.get("engine_request_sha256"),
        task_prompt_sha256=value.get("task_prompt_sha256"),
    )
    if value.get("intervention_ordinal") != 0 or dict(value) != normalized:
        raise ValueError("action intervention authority payload differs")
    if policy is not None and (
        normalized["campaign_name"] != policy.document["campaign_name"]
        or normalized["campaign_protocol_sha256"] != policy.document["protocol_sha256"]
        or normalized["policy_sha256"] != policy.policy_sha256
        or normalized["policy_revision"] != policy.document["policy_revision"]
    ):
        raise ValueError("action intervention campaign authority differs")
    return normalized


def validate_action_intervention(
    value: Any,
    *,
    require_current_policy: bool,
    now_unix: int | None = None,
) -> dict[str, Any]:
    """Authenticate an intervention against the separately configured root."""

    if not isinstance(value, Mapping) or set(value) != _INTERVENTION_FIELDS:
        raise ValueError("action intervention fields differ")
    if value.get("schema") != ACTION_INTERVENTION_SCHEMA:
        raise ValueError("action intervention schema is invalid")
    body = {name: value[name] for name in _INTERVENTION_FIELDS - {"intervention_sha256"}}
    if value.get("intervention_sha256") != _sha256(body):
        raise ValueError("action intervention digest differs")
    authority = _validate_authority_payload(value.get("authority_payload"))
    attestation = value.get("runner_attestation")
    signed_payload = attestation.get("signed_payload") if isinstance(attestation, Mapping) else None
    signed_at_unix = (
        signed_payload.get("signed_at_unix") if isinstance(signed_payload, Mapping) else None
    )
    if type(signed_at_unix) is not int or signed_at_unix <= 0:
        raise ValueError("action intervention signature time is invalid")
    root_path = os.environ.get(_TRUST_ROOT_ENV)
    if not isinstance(root_path, str) or not root_path.strip():
        raise ValueError("action intervention trust root is not configured")
    validation_time = int(time.time()) if require_current_policy and now_unix is None else now_unix
    if not require_current_policy:
        validation_time = signed_at_unix
    if type(validation_time) is not int or validation_time <= 0:
        raise ValueError("action intervention validation time is invalid")
    try:
        root_pem = read_stable_bytes(
            Path(root_path).expanduser(),
            max_bytes=64 * 1024,
        )
        policy_document = value.get("policy_document")
        if require_current_policy:
            current_policy_path = os.environ.get(_CURRENT_POLICY_ENV)
            if not isinstance(current_policy_path, str) or not current_policy_path.strip():
                raise ValueError("action intervention current policy is not configured")
            current_policy_bytes = read_stable_bytes(
                Path(current_policy_path).expanduser(),
                max_bytes=1024 * 1024,
            )
            current_policy_document = json.loads(current_policy_bytes)
            if current_policy_document != policy_document:
                raise ValueError("action intervention embeds a superseded policy")
            policy_document = current_policy_document
        policy = validate_campaign_trust_policy(
            policy_document,
            trusted_root_public_key_pem=root_pem,
            expected_campaign_name=authority["campaign_name"],
            expected_policy_sha256=authority["policy_sha256"],
            expected_protocol_sha256=authority["campaign_protocol_sha256"],
            minimum_policy_revision=authority["policy_revision"],
            now_unix=validation_time,
        )
        authority = _validate_authority_payload(authority, policy=policy)
        plan = _validate_plan_lineage(
            value.get("campaign_plan"),
            authority=authority,
            policy=policy,
            task_issuer_attestation=value.get("task_issuer_attestation"),
        )
        _validate_journal_prefix(
            value.get("campaign_journal_prefix"),
            plan=plan,
            authority=authority,
        )
        verify_role_attestation(
            policy,
            attestation,
            role=CAMPAIGN_RUNNER,
            expected_payload=authority,
            not_after_unix=validation_time if require_current_policy else None,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("action intervention trust admission failed") from exc
    return json.loads(_canonical_json_bytes(value))


def validate_action_intervention_objective(
    intervention: Mapping[str, Any],
    *,
    prompt: Any,
    messages: Any = None,
    token_ids: Any = None,
) -> str:
    """Bind a direct engine call to the issuer's exact preregistered prompt."""

    normalized = validate_action_intervention(intervention, require_current_policy=True)
    if messages is not None or token_ids is not None:
        raise ValueError("action intervention requires the prompt-only laboratory lane")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("action intervention preregistered prompt is unavailable")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != normalized[
        "authority_payload"
    ]["task_prompt_sha256"]:
        raise ValueError("action intervention prompt differs from preregistration")
    return prompt


def _replay_ledger_path() -> Path:
    ledger_path_raw = os.environ.get(_REPLAY_LEDGER_ENV)
    if not isinstance(ledger_path_raw, str) or not ledger_path_raw.strip():
        raise ValueError("action intervention replay ledger is not configured")
    return Path(ledger_path_raw).expanduser()


def _claim_canonical_campaign_attempt(
    intervention: Mapping[str, Any],
) -> str:
    authority = intervention["authority_payload"]
    journal_path_raw = os.environ.get(_CAMPAIGN_JOURNAL_ENV)
    if not isinstance(journal_path_raw, str) or not journal_path_raw.strip():
        raise ValueError("action intervention canonical journal is not configured")
    journal_path = Path(journal_path_raw).expanduser().absolute()
    if (
        not journal_path.exists()
        or action_intervention_campaign_journal_sha256(journal_path)
        != authority["campaign_journal_path_sha256"]
    ):
        raise ValueError("action intervention canonical journal identity differs")
    try:
        plan = CampaignPlan.from_dict(intervention["campaign_plan"])
        with CampaignJournal(journal_path, plan) as journal:
            return journal.claim_action_intervention(
                authority["cell_id"],
                authority["attempt_id"],
                intervention_sha256=intervention["intervention_sha256"],
                request_payload_sha256=authority["request_payload_sha256"],
                expected_journal_head_sha256=authority["journal_head_sha256"],
                expected_journal_event_count=authority["journal_event_count"],
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("action intervention canonical journal claim failed") from exc


def _validate_replay_row(
    value: Any,
    *,
    sequence: int,
    previous_event_sha256: str | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _REPLAY_EVENT_FIELDS:
        raise ValueError(f"action intervention replay row {sequence + 1} differs")
    row = dict(value)
    body = {name: row[name] for name in _REPLAY_EVENT_FIELDS - {"event_sha256"}}
    event = row.get("event")
    consumption_sha256 = row.get("consumption_event_sha256")
    campaign_claim_sha256 = row.get("campaign_claim_event_sha256")
    if (
        row.get("schema") != ACTION_INTERVENTION_REPLAY_SCHEMA
        or row.get("sequence") != sequence
        or row.get("previous_event_sha256") != previous_event_sha256
        or row.get("event_sha256") != _sha256(body)
        or event not in {INTERVENTION_CONSUMED, INTERVENTION_EXECUTION_CLAIMED}
        or not _is_sha256(row.get("intervention_sha256"))
        or not _is_sha256(row.get("request_payload_sha256"))
        or not _is_sha256(campaign_claim_sha256)
        or not isinstance(row.get("attempt_id"), str)
        or not row["attempt_id"]
        or type(row.get("recorded_at_unix")) is not int
        or row["recorded_at_unix"] <= 0
        or (
            event == INTERVENTION_CONSUMED
            and consumption_sha256 is not None
        )
        or (
            event == INTERVENTION_EXECUTION_CLAIMED
            and not _is_sha256(consumption_sha256)
        )
    ):
        raise ValueError(f"action intervention replay row {sequence + 1} differs")
    return row


def _load_replay_ledger(ledger_path: Path) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    raw = read_stable_bytes(ledger_path, max_bytes=_MAX_REPLAY_LEDGER_BYTES)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            candidate = json.loads(line)
        except (TypeError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(
                f"action intervention replay row {line_number} is invalid"
            ) from exc
        rows.append(
            _validate_replay_row(
                candidate,
                sequence=len(rows),
                previous_event_sha256=(rows[-1]["event_sha256"] if rows else None),
            )
        )
    consumed_by_sha = {
        row["event_sha256"]: row
        for row in rows
        if row["event"] == INTERVENTION_CONSUMED
    }
    seen_consumed_attempts: set[str] = set()
    seen_consumed_interventions: set[str] = set()
    seen_claims: set[str] = set()
    for row in rows:
        if row["event"] == INTERVENTION_CONSUMED:
            if (
                row["attempt_id"] in seen_consumed_attempts
                or row["intervention_sha256"] in seen_consumed_interventions
            ):
                raise ValueError("action intervention replay ledger contains duplicate consumption")
            seen_consumed_attempts.add(row["attempt_id"])
            seen_consumed_interventions.add(row["intervention_sha256"])
            continue
        consumed = consumed_by_sha.get(row["consumption_event_sha256"])
        if (
            consumed is None
            or consumed["intervention_sha256"] != row["intervention_sha256"]
            or consumed["attempt_id"] != row["attempt_id"]
            or consumed["request_payload_sha256"] != row["request_payload_sha256"]
            or consumed["campaign_claim_event_sha256"]
            != row["campaign_claim_event_sha256"]
            or row["consumption_event_sha256"] in seen_claims
        ):
            raise ValueError("action intervention execution claim lineage differs")
        seen_claims.add(row["consumption_event_sha256"])
    return rows


def _save_replay_ledger(ledger_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    from core.brain.llm.latent_cortex.persistence import LatentCortexPersistence

    payload = b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)
    if len(payload) > _MAX_REPLAY_LEDGER_BYTES:
        raise ValueError("action intervention replay ledger exceeded its byte budget")
    LatentCortexPersistence().save_action_intervention_replay_ledger(
        ledger_path,
        payload,
    )


def consume_action_intervention_once(
    intervention: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically reserve one request-bound attempt before worker dispatch."""

    normalized = validate_action_intervention(intervention, require_current_policy=True)
    campaign_claim_event_sha256 = _claim_canonical_campaign_attempt(normalized)
    ledger_path = _replay_ledger_path()
    authority = normalized["authority_payload"]
    try:
        from core.runtime.atomic_writer import interprocess_file_lock

        lock_path = ledger_path.with_name(f".{ledger_path.name}.lock")
        with interprocess_file_lock(lock_path):
            rows = _load_replay_ledger(ledger_path)
            if any(
                (
                    row["event"] == INTERVENTION_CONSUMED
                    and (
                        row["intervention_sha256"] == normalized["intervention_sha256"]
                        or row["attempt_id"] == authority["attempt_id"]
                    )
                )
                for row in rows
            ):
                raise ValueError("action intervention attempt was already consumed")
            body = {
                "schema": ACTION_INTERVENTION_REPLAY_SCHEMA,
                "sequence": len(rows),
                "previous_event_sha256": (rows[-1]["event_sha256"] if rows else None),
                "event": INTERVENTION_CONSUMED,
                "intervention_sha256": normalized["intervention_sha256"],
                "attempt_id": authority["attempt_id"],
                "request_payload_sha256": authority["request_payload_sha256"],
                "consumption_event_sha256": None,
                "campaign_claim_event_sha256": campaign_claim_event_sha256,
                "recorded_at_unix": int(time.time()),
            }
            event = {**body, "event_sha256": _sha256(body)}
            _save_replay_ledger(ledger_path, [*rows, event])
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("action intervention replay admission failed") from exc
    return event


def claim_action_intervention_execution(
    intervention: Mapping[str, Any],
    consumption_event: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically claim actual engine execution for a consumed intervention."""

    normalized = validate_action_intervention(intervention, require_current_policy=True)
    campaign_claim_event_sha256 = _claim_canonical_campaign_attempt(normalized)
    ledger_path = _replay_ledger_path()
    authority = normalized["authority_payload"]
    try:
        from core.runtime.atomic_writer import interprocess_file_lock

        lock_path = ledger_path.with_name(f".{ledger_path.name}.lock")
        with interprocess_file_lock(lock_path):
            rows = _load_replay_ledger(ledger_path)
            if not isinstance(consumption_event, Mapping):
                raise ValueError("action intervention consumption event is invalid")
            normalized_consumption = _validate_replay_row(
                consumption_event,
                sequence=int(consumption_event.get("sequence", -1)),
                previous_event_sha256=consumption_event.get("previous_event_sha256"),
            )
            persisted_consumption = next(
                (
                    row
                    for row in rows
                    if row["event_sha256"] == normalized_consumption["event_sha256"]
                ),
                None,
            )
            if (
                persisted_consumption != normalized_consumption
                or normalized_consumption["event"] != INTERVENTION_CONSUMED
                or normalized_consumption["intervention_sha256"]
                != normalized["intervention_sha256"]
                or normalized_consumption["attempt_id"] != authority["attempt_id"]
                or normalized_consumption["request_payload_sha256"]
                != authority["request_payload_sha256"]
                or normalized_consumption["campaign_claim_event_sha256"]
                != campaign_claim_event_sha256
            ):
                raise ValueError("action intervention consumption lineage differs")
            if any(
                row["event"] == INTERVENTION_EXECUTION_CLAIMED
                and row["consumption_event_sha256"]
                == normalized_consumption["event_sha256"]
                for row in rows
            ):
                raise ValueError("action intervention execution was already claimed")
            body = {
                "schema": ACTION_INTERVENTION_REPLAY_SCHEMA,
                "sequence": len(rows),
                "previous_event_sha256": (rows[-1]["event_sha256"] if rows else None),
                "event": INTERVENTION_EXECUTION_CLAIMED,
                "intervention_sha256": normalized["intervention_sha256"],
                "attempt_id": authority["attempt_id"],
                "request_payload_sha256": authority["request_payload_sha256"],
                "consumption_event_sha256": normalized_consumption["event_sha256"],
                "campaign_claim_event_sha256": campaign_claim_event_sha256,
                "recorded_at_unix": int(time.time()),
            }
            claim = {**body, "event_sha256": _sha256(body)}
            _save_replay_ledger(ledger_path, [*rows, claim])
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("action intervention execution claim failed") from exc
    return claim


def validate_action_intervention_execution_claim(
    intervention: Mapping[str, Any],
    execution_claim: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that an execution claim is present in the durable replay ledger."""

    normalized = validate_action_intervention(intervention, require_current_policy=False)
    ledger_path = _replay_ledger_path()
    rows = _load_replay_ledger(ledger_path)
    candidate = next(
        (
            row
            for row in rows
            if isinstance(execution_claim, Mapping)
            and row["event_sha256"] == execution_claim.get("event_sha256")
        ),
        None,
    )
    authority = normalized["authority_payload"]
    if (
        candidate is None
        or candidate != dict(execution_claim)
        or candidate["event"] != INTERVENTION_EXECUTION_CLAIMED
        or candidate["intervention_sha256"] != normalized["intervention_sha256"]
        or candidate["attempt_id"] != authority["attempt_id"]
        or candidate["request_payload_sha256"] != authority["request_payload_sha256"]
    ):
        raise ValueError("action intervention execution claim differs")
    return candidate


def build_action_intervention_receipt(
    *,
    intervention: Mapping[str, Any],
    execution_claim: Mapping[str, Any],
    pre_state_components: Mapping[str, str],
    post_state_components: Mapping[str, str],
    pre_state_sha256: str,
    pre_kv_sha256: str,
    post_state_sha256: str,
    post_kv_sha256: str,
    decision_sha256: str,
    cognitive_action_trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the public worker receipt after the complete trace is known."""

    normalized = validate_action_intervention(
        intervention,
        require_current_policy=False,
    )
    normalized_execution_claim = validate_action_intervention_execution_claim(
        normalized,
        execution_claim,
    )
    authority = normalized["authority_payload"]
    normalized_pre_components = _state_components(pre_state_components)
    normalized_post_components = _state_components(post_state_components)
    arm = authority["arm"]
    action = authority["action"]
    decisions = [
        row.get("decision")
        for row in cognitive_action_trace
        if isinstance(row, Mapping) and isinstance(row.get("decision"), Mapping)
    ]
    occurrences = sum(decision.get("action") == action for decision in decisions)
    if arm == TREATMENT_ARM:
        selected_action: str | None = action
        selection_mode = "campaign_forced"
        if (
            occurrences != 1
            or not decisions
            or decisions[0].get("action") != action
            or decisions[0].get("mode") != selection_mode
            or decisions[0].get("decision_sha256") != decision_sha256
        ):
            raise ValueError("forced action intervention trace differs")
    else:
        selected_action = None
        selection_mode = "matched_no_action_control"
        if occurrences or decision_sha256:
            raise ValueError("matched no-action intervention trace differs")
        unchanged_control_components = _STARTING_STATE_COMPONENTS - {
            "public_action_state_sha256"
        }
        if (
            any(
                normalized_pre_components[name] != normalized_post_components[name]
                for name in unchanged_control_components
            )
            or normalized_pre_components["public_action_state_sha256"]
            == normalized_post_components["public_action_state_sha256"]
            or pre_state_sha256 == post_state_sha256
            or pre_kv_sha256 != post_kv_sha256
        ):
            raise ValueError("matched no-action intervention state transition differs")
    for name, digest in (
        ("pre_state_sha256", pre_state_sha256),
        ("pre_kv_sha256", pre_kv_sha256),
        ("post_state_sha256", post_state_sha256),
        ("post_kv_sha256", post_kv_sha256),
    ):
        if not _is_sha256(digest):
            raise ValueError(f"action intervention receipt {name} is invalid")
    if (
        normalized_pre_components != authority["starting_state_components"]
        or pre_state_sha256 != _sha256(normalized_pre_components)
        or post_state_sha256 != _sha256(normalized_post_components)
        or pre_kv_sha256 != normalized_pre_components["kv_cache_sha256"]
        or post_kv_sha256 != normalized_post_components["kv_cache_sha256"]
        or pre_state_sha256 != authority["expected_pre_state_sha256"]
        or pre_kv_sha256 != authority["expected_pre_kv_sha256"]
    ):
        raise ValueError("action intervention started from an unexpected resident state")
    component_observation_owners = {
        name: (
            "runner_attested_precondition_carried_forward_not_worker_observed"
            if name in {"durable_state_sha256", "rng_state_sha256"}
            else "resident_worker_measured_pre_and_post"
        )
        for name in sorted(_STARTING_STATE_COMPONENTS)
    }
    body = {
        "schema": ACTION_INTERVENTION_RECEIPT_SCHEMA,
        "intervention_sha256": normalized["intervention_sha256"],
        "campaign_plan_sha256": authority["campaign_plan_sha256"],
        "cell_id": authority["cell_id"],
        "attempt_id": authority["attempt_id"],
        "request_payload_sha256": authority["request_payload_sha256"],
        "arm": arm,
        "action": action,
        "intervention_ordinal": 0,
        "execution_ordinal": authority["execution_ordinal"],
        "attempt_number": authority["attempt_number"],
        "execution_claim": normalized_execution_claim,
        "execution_claim_sha256": normalized_execution_claim["event_sha256"],
        "consumed": True,
        "selection_mode": selection_mode,
        "selected_action": selected_action,
        "selected_action_occurrences": occurrences,
        "action_excluded_after_intervention": True,
        "starting_state_components": dict(authority["starting_state_components"]),
        "component_observation_owners": component_observation_owners,
        "pre_state_components": normalized_pre_components,
        "post_state_components": normalized_post_components,
        "pre_state_sha256": pre_state_sha256,
        "pre_kv_sha256": pre_kv_sha256,
        "post_state_sha256": post_state_sha256,
        "post_kv_sha256": post_kv_sha256,
        "decision_sha256": decision_sha256,
        "cognitive_action_trace_sha256": _sha256(list(cognitive_action_trace)),
    }
    return {**body, "receipt_sha256": _sha256(body)}


def validate_action_intervention_receipt(
    value: Any,
    *,
    intervention: Mapping[str, Any],
    cognitive_action_trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconstruct a worker intervention receipt from its signed authority."""

    if not isinstance(value, Mapping) or set(value) != _RECEIPT_FIELDS:
        raise ValueError("action intervention receipt fields differ")
    if value.get("schema") != ACTION_INTERVENTION_RECEIPT_SCHEMA:
        raise ValueError("action intervention receipt schema is invalid")
    rebuilt = build_action_intervention_receipt(
        intervention=intervention,
        execution_claim=value.get("execution_claim"),
        pre_state_components=value.get("pre_state_components"),
        post_state_components=value.get("post_state_components"),
        pre_state_sha256=value.get("pre_state_sha256"),
        pre_kv_sha256=value.get("pre_kv_sha256"),
        post_state_sha256=value.get("post_state_sha256"),
        post_kv_sha256=value.get("post_kv_sha256"),
        decision_sha256=value.get("decision_sha256"),
        cognitive_action_trace=cognitive_action_trace,
    )
    if dict(value) != rebuilt:
        raise ValueError("action intervention receipt differs")
    return rebuilt


__all__ = [
    "ACTION_INTERVENTION_AUTHORITY_SCHEMA",
    "ACTION_INTERVENTION_RECEIPT_SCHEMA",
    "ACTION_INTERVENTION_REPLAY_SCHEMA",
    "ACTION_INTERVENTION_SCHEMA",
    "CONTROL_ARM",
    "INTERVENTION_CONSUMED",
    "INTERVENTION_EXECUTION_CLAIMED",
    "INTERVENTION_ARMS",
    "TREATMENT_ARM",
    "action_intervention_campaign_journal_sha256",
    "action_intervention_authority_payload",
    "action_intervention_attempt_id",
    "action_intervention_engine_request_sha256",
    "build_action_intervention",
    "build_action_intervention_receipt",
    "claim_action_intervention_execution",
    "consume_action_intervention_once",
    "validate_action_intervention",
    "validate_action_intervention_execution_claim",
    "validate_action_intervention_objective",
    "validate_action_intervention_receipt",
]
