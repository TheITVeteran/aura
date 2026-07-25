"""Bounded service-side acquisition for recurrent cognitive actions.

The MLX worker may decide that admitted memory or evidence deserves focus, but
it never performs I/O. This contract converts one validated worker decision
into one content-addressed acquisition request, distinguishes repeated context
from genuinely new observations, and binds an optional second episode to the
first. It does not grant instruction authority to retrieved text.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from core.brain.llm.latent_cortex.context_focus import source_matches_action
from core.brain.llm.latent_cortex.epistemic_state import (
    OperationKind,
    canonical_sha256,
)

COGNITIVE_ACQUISITION_SCHEMA = "aura.rlc.cognitive_acquisition.v1"
COGNITIVE_ACQUISITION_ACTIONS = {
    OperationKind.SEARCH_MEMORY,
    OperationKind.RETRIEVE_EVIDENCE,
}
MAX_RETRIEVAL_QUERY_CHARS = 1_200
MAX_CONTINUATION_ROUNDS = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _text_sha256(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _context_digest(item: Mapping[str, Any]) -> str:
    for field in ("content_sha256", "text_sha256"):
        supplied = item.get(field)
        if isinstance(supplied, str) and _SHA256_RE.fullmatch(supplied):
            return supplied
    return _text_sha256(item.get("text"))


def _source_inventory(
    context: Sequence[Mapping[str, Any]] | None,
    action: OperationKind,
    *,
    acquired_only: bool = False,
) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for raw in context or ():
        if not isinstance(raw, Mapping):
            raise ValueError("cognitive acquisition context row is invalid")
        source = str(raw.get("source") or "")
        text = raw.get("text")
        has_committed_text = any(
            isinstance(raw.get(field), str)
            and _SHA256_RE.fullmatch(str(raw.get(field)))
            for field in ("content_sha256", "text_sha256")
        )
        if (
            not source
            or (
                not has_committed_text
                and (not isinstance(text, str) or not text.strip())
            )
        ):
            raise ValueError("cognitive acquisition context row is invalid")
        matches = source_matches_action(source, action)
        if acquired_only:
            matches = (
                source == "memory" or source.startswith("memory.")
                if action is OperationKind.SEARCH_MEMORY
                else source == "reference"
            )
        if matches:
            rows.append((source, _context_digest(raw)))
    return tuple(sorted(set(rows)))


def _selected_transition(
    receipt: Mapping[str, Any],
) -> tuple[int, OperationKind, dict[str, Any]] | None:
    trace = receipt.get("cognitive_action_trace")
    if not isinstance(trace, list):
        return None
    selected: list[tuple[int, OperationKind, dict[str, Any]]] = []
    for step, row in enumerate(trace):
        if not isinstance(row, Mapping):
            raise ValueError("cognitive action trace row is invalid")
        transition = row.get("transition")
        if not isinstance(transition, Mapping):
            raise ValueError("cognitive action transition is missing")
        try:
            action = OperationKind(transition.get("action"))
        except (TypeError, ValueError):
            continue
        if (
            action in COGNITIVE_ACQUISITION_ACTIONS
            and transition.get("outcome") == "succeeded"
        ):
            selected.append((step, action, dict(transition)))
    return selected[-1] if selected else None


def build_acquisition_request(
    *,
    objective: str,
    first_text: str,
    first_receipt: Mapping[str, Any],
    cognitive_context: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any] | None:
    """Create one query from the latest successful retrieval-class action."""

    if not isinstance(first_receipt, Mapping):
        raise ValueError("first episode receipt is invalid")
    objective = str(objective or "").strip()
    first_text = str(first_text or "").strip()
    if not objective or not first_text:
        raise ValueError("acquisition objective and first answer must be non-empty")
    selected = _selected_transition(first_receipt)
    if selected is None:
        return None
    step, action, transition = selected
    transition_sha256 = canonical_sha256(transition)
    directive = (
        "Recall prior context that could confirm, contradict, or refine this "
        "tentative answer:"
        if action is OperationKind.SEARCH_MEMORY
        else "Find reference evidence that could confirm, contradict, or refine "
        "this tentative answer:"
    )
    retrieval_query = (
        f"{objective}\n{directive}\n{first_text[:800]}"
    )[:MAX_RETRIEVAL_QUERY_CHARS].strip()
    inventory = _source_inventory(cognitive_context, action)
    payload = {
        "schema": COGNITIVE_ACQUISITION_SCHEMA,
        "action": action.value,
        "action_step": step,
        "objective_sha256": _text_sha256(objective),
        "first_answer_sha256": _text_sha256(first_text),
        "transition_sha256": transition_sha256,
        "retrieval_query": retrieval_query,
        "retrieval_query_sha256": _text_sha256(retrieval_query),
        "before_inventory": [list(row) for row in inventory],
        "before_inventory_sha256": canonical_sha256(inventory),
        "max_acquisitions": 1,
        "max_continuation_rounds": MAX_CONTINUATION_ROUNDS,
        "worker_performed_io": False,
    }
    return {**payload, "request_sha256": canonical_sha256(payload)}


def validate_acquisition_request(
    value: Any,
    *,
    objective: str,
    first_text: str,
    first_receipt: Mapping[str, Any],
    cognitive_context: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    expected = build_acquisition_request(
        objective=objective,
        first_text=first_text,
        first_receipt=first_receipt,
        cognitive_context=cognitive_context,
    )
    if expected is None or value != expected:
        raise ValueError("cognitive acquisition request differs")
    return dict(expected)


def build_acquisition_receipt(
    request: Mapping[str, Any],
    *,
    acquired_context: Sequence[Mapping[str, Any]] | None,
    ingress_receipt: Mapping[str, Any],
    elapsed_s: float,
    error_code: str = "",
) -> dict[str, Any]:
    """Bind source-class deduplication and continuation eligibility."""

    if not isinstance(request, Mapping):
        raise ValueError("cognitive acquisition request is invalid")
    action = OperationKind(request.get("action"))
    if action not in COGNITIVE_ACQUISITION_ACTIONS:
        raise ValueError("cognitive acquisition action is unsupported")
    if request.get("request_sha256") != canonical_sha256(
        {key: request[key] for key in request if key != "request_sha256"}
    ):
        raise ValueError("cognitive acquisition request commitment differs")
    if not isinstance(ingress_receipt, Mapping):
        raise ValueError("cognitive acquisition ingress receipt is invalid")
    if isinstance(elapsed_s, bool) or not isinstance(elapsed_s, (int, float)):
        raise ValueError("cognitive acquisition elapsed time is invalid")
    elapsed_s = float(elapsed_s)
    if not 0.0 <= elapsed_s <= 30.0:
        raise ValueError("cognitive acquisition elapsed time is outside bounds")
    target_source = (
        "memory" if action is OperationKind.SEARCH_MEMORY else "reference"
    )
    absent_sources = ingress_receipt.get("absent_sources")
    if (
        not error_code
        and isinstance(absent_sources, list)
        and target_source in absent_sources
    ):
        error_code = f"{target_source}_source_unavailable"
    before = {
        tuple(row)
        for row in request.get("before_inventory", [])
        if isinstance(row, list) and len(row) == 2
    }
    before_content = {digest for _source, digest in before}
    after = _source_inventory(
        acquired_context,
        action,
        acquired_only=True,
    )
    new_rows = tuple(row for row in after if row[1] not in before_content)
    error_code = str(error_code or "").strip()
    if error_code and not re.fullmatch(r"[a-z0-9_.:-]{1,96}", error_code):
        raise ValueError("cognitive acquisition error code is invalid")
    status = (
        "failed"
        if error_code
        else "completed_new_context"
        if new_rows
        else "completed_no_new_context"
    )
    payload = {
        "schema": COGNITIVE_ACQUISITION_SCHEMA,
        "request_sha256": request["request_sha256"],
        "action": action.value,
        "status": status,
        "error_code": error_code,
        "acquisition_attempts": 1,
        "continuation_rounds_authorized": (
            1 if status == "completed_new_context" else 0
        ),
        "after_inventory": [list(row) for row in after],
        "after_inventory_sha256": canonical_sha256(after),
        "new_inventory": [list(row) for row in new_rows],
        "new_context_count": len(new_rows),
        "ingress_receipt_sha256": canonical_sha256(ingress_receipt),
        "elapsed_ms": round(elapsed_s * 1000.0, 3),
        "worker_performed_io": False,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def validate_acquisition_receipt(
    value: Any,
    *,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate caps, inventory commitments, status, and receipt integrity."""

    fields = {
        "schema",
        "request_sha256",
        "action",
        "status",
        "error_code",
        "acquisition_attempts",
        "continuation_rounds_authorized",
        "after_inventory",
        "after_inventory_sha256",
        "new_inventory",
        "new_context_count",
        "ingress_receipt_sha256",
        "elapsed_ms",
        "worker_performed_io",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("cognitive acquisition receipt fields differ")
    if (
        value["schema"] != COGNITIVE_ACQUISITION_SCHEMA
        or value["request_sha256"] != request.get("request_sha256")
        or value["action"] != request.get("action")
        or value["acquisition_attempts"] != 1
        or value["worker_performed_io"] is not False
        or value["status"]
        not in {"failed", "completed_new_context", "completed_no_new_context"}
    ):
        raise ValueError("cognitive acquisition receipt metadata differs")
    after = value["after_inventory"]
    new = value["new_inventory"]
    if (
        not isinstance(after, list)
        or not isinstance(new, list)
        or any(not isinstance(row, list) or len(row) != 2 for row in [*after, *new])
        or value["after_inventory_sha256"]
        != canonical_sha256(tuple(tuple(row) for row in after))
        or value["new_context_count"] != len(new)
    ):
        raise ValueError("cognitive acquisition inventory differs")
    before_content = {
        row[1]
        for row in request.get("before_inventory", [])
        if isinstance(row, list) and len(row) == 2
    }
    expected_new = [row for row in after if row[1] not in before_content]
    if new != expected_new:
        raise ValueError("cognitive acquisition deduplication differs")
    expected_status = (
        "failed"
        if value["error_code"]
        else "completed_new_context"
        if new
        else "completed_no_new_context"
    )
    if (
        value["status"] != expected_status
        or value["continuation_rounds_authorized"]
        != (1 if expected_status == "completed_new_context" else 0)
        or isinstance(value["elapsed_ms"], bool)
        or not isinstance(value["elapsed_ms"], (int, float))
        or not 0.0 <= float(value["elapsed_ms"]) <= 30_000.0
    ):
        raise ValueError("cognitive acquisition outcome differs")
    for name in (
        "request_sha256",
        "after_inventory_sha256",
        "ingress_receipt_sha256",
        "receipt_sha256",
    ):
        if not isinstance(value[name], str) or not _SHA256_RE.fullmatch(value[name]):
            raise ValueError("cognitive acquisition commitment is invalid")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != canonical_sha256(payload):
        raise ValueError("cognitive acquisition receipt commitment differs")
    return dict(value)


def build_continuation_receipt(
    request: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    *,
    first_result: Mapping[str, Any],
    second_result: Mapping[str, Any] | None,
    returned_round: int,
    continuation_reason: str,
) -> dict[str, Any]:
    """Commit both episode results and the exact result returned to the caller."""

    if returned_round not in {1, 2}:
        raise ValueError("returned continuation round is invalid")
    if returned_round == 2 and second_result is None:
        raise ValueError("round two cannot be returned without a second result")
    continuation_reason = str(continuation_reason or "").strip()
    if not re.fullmatch(r"[a-z0-9_.:-]{1,96}", continuation_reason):
        raise ValueError("continuation reason is invalid")
    status = str(acquisition.get("status") or "")
    second_attempted = second_result is not None
    payload = {
        "schema": COGNITIVE_ACQUISITION_SCHEMA,
        "request": dict(request),
        "acquisition": dict(acquisition),
        "request_sha256": request.get("request_sha256"),
        "acquisition_receipt_sha256": acquisition.get("receipt_sha256"),
        "first_result_sha256": canonical_sha256(first_result),
        "second_result_sha256": (
            canonical_sha256(second_result) if second_result is not None else None
        ),
        "second_attempted": second_attempted,
        "second_succeeded": (
            bool(second_result.get("ok") is True)
            if second_result is not None
            else False
        ),
        "returned_round": returned_round,
        "continuation_reason": continuation_reason,
        "status": (
            "continued"
            if returned_round == 2
            else "retained_first_after_second_failure"
            if second_attempted
            else status
        ),
        "acquisition_cap_exhausted": True,
        "continuation_cap_exhausted": second_attempted,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def validate_continuation_receipt(value: Any) -> dict[str, Any]:
    """Validate the closed two-round envelope without re-running either model."""

    fields = {
        "schema",
        "request",
        "acquisition",
        "request_sha256",
        "acquisition_receipt_sha256",
        "first_result_sha256",
        "second_result_sha256",
        "second_attempted",
        "second_succeeded",
        "returned_round",
        "continuation_reason",
        "status",
        "acquisition_cap_exhausted",
        "continuation_cap_exhausted",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("cognitive continuation receipt fields differ")
    request = value["request"]
    acquisition = value["acquisition"]
    if not isinstance(request, Mapping) or not isinstance(acquisition, Mapping):
        raise ValueError("cognitive continuation nested receipts are invalid")
    validate_acquisition_receipt(acquisition, request=request)
    if (
        value["schema"] != COGNITIVE_ACQUISITION_SCHEMA
        or value["request_sha256"] != request.get("request_sha256")
        or value["acquisition_receipt_sha256"] != acquisition.get("receipt_sha256")
        or value["second_attempted"] is not (value["second_result_sha256"] is not None)
        or value["returned_round"] not in {1, 2}
        or value["returned_round"] == 2
        and (
            value["second_attempted"] is not True
            or value["second_succeeded"] is not True
        )
        or value["acquisition_cap_exhausted"] is not True
        or value["continuation_cap_exhausted"] is not value["second_attempted"]
        or not re.fullmatch(
            r"[a-z0-9_.:-]{1,96}",
            str(value["continuation_reason"] or ""),
        )
    ):
        raise ValueError("cognitive continuation outcome differs")
    for name in (
        "request_sha256",
        "acquisition_receipt_sha256",
        "first_result_sha256",
        "receipt_sha256",
    ):
        if not isinstance(value[name], str) or not _SHA256_RE.fullmatch(value[name]):
            raise ValueError("cognitive continuation commitment is invalid")
    if value["second_result_sha256"] is not None and (
        not isinstance(value["second_result_sha256"], str)
        or not _SHA256_RE.fullmatch(value["second_result_sha256"])
    ):
        raise ValueError("cognitive continuation second-result commitment is invalid")
    expected_status = (
        "continued"
        if value["returned_round"] == 2
        else "retained_first_after_second_failure"
        if value["second_attempted"]
        else acquisition["status"]
    )
    if value["status"] != expected_status:
        raise ValueError("cognitive continuation status differs")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != canonical_sha256(payload):
        raise ValueError("cognitive continuation receipt commitment differs")
    return dict(value)


__all__ = [
    "COGNITIVE_ACQUISITION_ACTIONS",
    "COGNITIVE_ACQUISITION_SCHEMA",
    "MAX_CONTINUATION_ROUNDS",
    "MAX_RETRIEVAL_QUERY_CHARS",
    "build_acquisition_receipt",
    "build_acquisition_request",
    "build_continuation_receipt",
    "validate_acquisition_receipt",
    "validate_acquisition_request",
    "validate_continuation_receipt",
]
