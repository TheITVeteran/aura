"""Digest-bound handoff between latent cognition and governed world effects.

The resident worker may decide that a concrete, already-Will-admitted action
should execute, but it never receives effect authority or raw parameters. The
host offers only bounded public identity, the worker emits a request bound to
its value-of-computation decision, and ActionExecutor independently
reconstructs that request before dispatching the real effect.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from core.brain.llm.latent_cortex.epistemic_state import OperationKind

EXTERNAL_EXECUTION_OFFER_SCHEMA = "aura.rlc.external_execution_offer.v1"
EXTERNAL_EXECUTION_HANDOFF_SCHEMA = "aura.rlc.external_execution_handoff.v2"
EXTERNAL_EXECUTION_READINESS_SCHEMA = "aura.rlc.external_execution_readiness.v1"

_OFFER_FIELDS = frozenset(
    {
        "schema",
        "action_id",
        "domain",
        "action_name",
        "request_digest",
        "will_receipt_id",
        "objective_sha256",
        "expectation_sha256",
        "offer_sha256",
    }
)
_HANDOFF_FIELDS = frozenset(
    {
        "schema",
        "offer_sha256",
        "requested",
        "decision_sha256",
        "step_index",
        "mode",
        "outcome",
        "trace_sha256",
        "handoff_sha256",
    }
)
_READINESS_MODEL_FIELDS = frozenset(
    {
        "action_ready",
        "preconditions_met",
        "risk_acceptable",
        "expected_effect",
        "reason",
    }
)
_READINESS_FIELDS = frozenset(
    {
        "schema",
        "offer_sha256",
        *_READINESS_MODEL_FIELDS,
        "model_output_sha256",
        "readiness_sha256",
    }
)


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"external execution payload is not canonical: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _bounded_text(value: Any, *, name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > limit
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError(f"{name} must be bounded printable text")
    return normalized


def _bounded_objective(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("objective must be a string")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 1000
        or any(
            ord(character) < 32 and character not in {"\n", "\r", "\t"}
            for character in normalized
        )
    ):
        raise ValueError("objective must be bounded text")
    return normalized


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _request_digest(value: Any) -> str:
    normalized = _bounded_text(value, name="request_digest", limit=80)
    if not normalized.startswith("sha256:"):
        raise ValueError("request_digest must use the sha256: scheme")
    _sha256(normalized.removeprefix("sha256:"), name="request_digest")
    return normalized


def build_external_execution_offer(
    *,
    action_id: str,
    domain: str,
    action_name: str,
    request_digest: str,
    will_receipt_id: str,
    objective: str,
    expectation: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the parameter-free action identity admitted to the worker."""

    normalized_expectation = dict(expectation)
    payload = {
        "schema": EXTERNAL_EXECUTION_OFFER_SCHEMA,
        "action_id": _bounded_text(action_id, name="action_id", limit=160),
        "domain": _bounded_text(domain, name="domain", limit=64),
        "action_name": _bounded_text(action_name, name="action_name", limit=160),
        "request_digest": _request_digest(request_digest),
        "will_receipt_id": _bounded_text(
            will_receipt_id,
            name="will_receipt_id",
            limit=192,
        ),
        "objective_sha256": hashlib.sha256(
            _bounded_objective(objective).encode("utf-8")
        ).hexdigest(),
        "expectation_sha256": _canonical_sha256(normalized_expectation),
    }
    return {**payload, "offer_sha256": _canonical_sha256(payload)}


def validate_external_execution_offer(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _OFFER_FIELDS:
        raise ValueError("external execution offer fields differ")
    normalized = {
        "schema": value.get("schema"),
        "action_id": _bounded_text(value.get("action_id"), name="action_id", limit=160),
        "domain": _bounded_text(value.get("domain"), name="domain", limit=64),
        "action_name": _bounded_text(
            value.get("action_name"),
            name="action_name",
            limit=160,
        ),
        "request_digest": _request_digest(value.get("request_digest")),
        "will_receipt_id": _bounded_text(
            value.get("will_receipt_id"),
            name="will_receipt_id",
            limit=192,
        ),
        "objective_sha256": _sha256(
            value.get("objective_sha256"),
            name="objective_sha256",
        ),
        "expectation_sha256": _sha256(
            value.get("expectation_sha256"),
            name="expectation_sha256",
        ),
    }
    if normalized["schema"] != EXTERNAL_EXECUTION_OFFER_SCHEMA:
        raise ValueError("external execution offer schema is invalid")
    expected = _canonical_sha256(normalized)
    if value.get("offer_sha256") != expected:
        raise ValueError("external execution offer digest does not match")
    return {**normalized, "offer_sha256": expected}


def build_external_execution_handoff(
    offer: Mapping[str, Any],
    cognitive_action_trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive the worker's one-shot EXECUTE request from its public trace."""

    normalized_offer = validate_external_execution_offer(offer)
    if not isinstance(cognitive_action_trace, Sequence) or isinstance(
        cognitive_action_trace,
        (str, bytes, bytearray),
    ):
        raise ValueError("cognitive action trace must be a sequence")
    execute_rows: list[Mapping[str, Any]] = []
    for row in cognitive_action_trace:
        if not isinstance(row, Mapping):
            raise ValueError("cognitive action trace row must be a mapping")
        decision = row.get("decision")
        if isinstance(decision, Mapping) and decision.get("action") == OperationKind.EXECUTE.value:
            execute_rows.append(row)
    if len(execute_rows) > 1:
        raise ValueError("external execution may be requested at most once")
    if execute_rows:
        row = execute_rows[0]
        decision = row.get("decision")
        transition = row.get("transition")
        state_signal = row.get("state_signal")
        if (
            not isinstance(decision, Mapping)
            or not isinstance(transition, Mapping)
            or not isinstance(state_signal, Mapping)
            or transition.get("action") != OperationKind.EXECUTE.value
            or transition.get("decision_sha256") != decision.get("decision_sha256")
            or transition.get("outcome") != "external_execute_requested"
            or transition.get("checked") is not False
            or state_signal.get("can_execute") is not True
        ):
            raise ValueError("external execution trace is not a valid request")
        requested = True
        decision_sha256 = _sha256(
            decision.get("decision_sha256"),
            name="decision_sha256",
        )
        step_index = decision.get("step_index")
        if type(step_index) is not int or step_index < 0:
            raise ValueError("external execution step index is invalid")
        mode = _bounded_text(decision.get("mode"), name="mode", limit=32)
        outcome = "external_execute_requested"
    else:
        requested = False
        decision_sha256 = ""
        step_index = -1
        mode = "not_selected"
        outcome = "not_selected"
    payload = {
        "schema": EXTERNAL_EXECUTION_HANDOFF_SCHEMA,
        "offer_sha256": normalized_offer["offer_sha256"],
        "requested": requested,
        "decision_sha256": decision_sha256,
        "step_index": step_index,
        "mode": mode,
        "outcome": outcome,
        "trace_sha256": _canonical_sha256(list(cognitive_action_trace)),
    }
    return {**payload, "handoff_sha256": _canonical_sha256(payload)}


def validate_external_execution_handoff(
    value: Any,
    *,
    offer: Mapping[str, Any],
    cognitive_action_trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _HANDOFF_FIELDS:
        raise ValueError("external execution handoff fields differ")
    expected = build_external_execution_handoff(offer, cognitive_action_trace)
    if dict(value) != expected:
        raise ValueError("external execution handoff differs from worker trace")
    return expected


def _readiness_model_payload(model_output: Any) -> dict[str, Any]:
    if not isinstance(model_output, str):
        raise ValueError("external execution readiness output must be text")
    rendered = model_output.strip()
    if rendered.startswith("```") and rendered.endswith("```"):
        lines = rendered.splitlines()
        if len(lines) < 3 or lines[0].strip() not in {"```", "```json"}:
            raise ValueError("external execution readiness fence is invalid")
        rendered = "\n".join(lines[1:-1]).strip()
    if not rendered or len(rendered) > 4000:
        raise ValueError("external execution readiness output is not bounded")
    try:
        raw = json.loads(rendered)
    except json.JSONDecodeError as exc:
        raise ValueError("external execution readiness output is not JSON") from exc
    if not isinstance(raw, Mapping) or frozenset(raw) != _READINESS_MODEL_FIELDS:
        raise ValueError("external execution readiness fields differ")
    for name in ("action_ready", "preconditions_met", "risk_acceptable"):
        if type(raw.get(name)) is not bool:
            raise ValueError(f"external execution readiness {name} must be boolean")
    expected_effect = _bounded_text(
        raw.get("expected_effect"),
        name="expected_effect",
        limit=800,
    )
    reason = _bounded_text(raw.get("reason"), name="reason", limit=800)
    return {
        "action_ready": raw["action_ready"],
        "preconditions_met": raw["preconditions_met"],
        "risk_acceptable": raw["risk_acceptable"],
        "expected_effect": expected_effect,
        "reason": reason,
    }


def build_external_execution_readiness(
    offer: Mapping[str, Any],
    model_output: str,
) -> dict[str, Any]:
    normalized_offer = validate_external_execution_offer(offer)
    model_payload = _readiness_model_payload(model_output)
    payload = {
        "schema": EXTERNAL_EXECUTION_READINESS_SCHEMA,
        "offer_sha256": normalized_offer["offer_sha256"],
        **model_payload,
        "model_output_sha256": hashlib.sha256(
            model_output.strip().encode("utf-8")
        ).hexdigest(),
    }
    return {**payload, "readiness_sha256": _canonical_sha256(payload)}


def validate_external_execution_readiness(
    value: Any,
    *,
    offer: Mapping[str, Any],
    model_output: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _READINESS_FIELDS:
        raise ValueError("external execution readiness receipt fields differ")
    normalized_offer = validate_external_execution_offer(offer)
    if model_output is not None:
        expected = build_external_execution_readiness(normalized_offer, model_output)
        if dict(value) != expected:
            raise ValueError("external execution readiness differs from model output")
        return expected
    payload = {
        "schema": value.get("schema"),
        "offer_sha256": value.get("offer_sha256"),
        "action_ready": value.get("action_ready"),
        "preconditions_met": value.get("preconditions_met"),
        "risk_acceptable": value.get("risk_acceptable"),
        "expected_effect": value.get("expected_effect"),
        "reason": value.get("reason"),
        "model_output_sha256": value.get("model_output_sha256"),
    }
    if payload["schema"] != EXTERNAL_EXECUTION_READINESS_SCHEMA:
        raise ValueError("external execution readiness schema is invalid")
    if payload["offer_sha256"] != normalized_offer["offer_sha256"]:
        raise ValueError("external execution readiness offer differs")
    _readiness_model_payload(
        json.dumps(
            {
                key: payload[key]
                for key in _READINESS_MODEL_FIELDS
            },
            sort_keys=True,
        )
    )
    _sha256(payload["model_output_sha256"], name="model_output_sha256")
    expected_digest = _canonical_sha256(payload)
    if value.get("readiness_sha256") != expected_digest:
        raise ValueError("external execution readiness digest does not match")
    return {**payload, "readiness_sha256": expected_digest}


__all__ = [
    "EXTERNAL_EXECUTION_HANDOFF_SCHEMA",
    "EXTERNAL_EXECUTION_OFFER_SCHEMA",
    "EXTERNAL_EXECUTION_READINESS_SCHEMA",
    "build_external_execution_handoff",
    "build_external_execution_offer",
    "build_external_execution_readiness",
    "validate_external_execution_handoff",
    "validate_external_execution_offer",
    "validate_external_execution_readiness",
]
