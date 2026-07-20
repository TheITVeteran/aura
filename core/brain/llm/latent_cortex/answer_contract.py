"""Deterministic public answer-contract surface (CP180).

The CP179 pilot diagnosed `decode_response_contract_failure` and
`shared_vanilla_decode_budget_truncates_contract_answers`: every arm was
being graded on whether a 256-token decode happened to end with a terminal
``FINAL_ANSWER: {…}`` line, and almost none did — the measurement was
strangling the models, not measuring them.

This module makes the contract a first-class, deterministic surface:

- ``contract_answer_state(text)`` — full incremental analysis: marker
  count, whether the post-marker JSON object is COMPLETE (strict-JSON
  brace scan, string/escape aware), whether the whole text would satisfy
  ``parse_final_answer``, and the parsed object when it would.
- ``is_contract_complete(text)`` — the decode early-stop predicate: the
  moment a single marker's JSON object closes and parses, decoding may
  stop. Applied identically to every arm (vanilla, RLC, adapter, control),
  it is a uniform serving-side stop rule — the standard stop-sequence
  discipline every production inference stack applies — never an edit of
  model text.

Semantics match ``frontier_tasks.parse_final_answer`` exactly: one marker,
terminal line, strict JSON object payload. Anything the parser would
reject, this module reports as incomplete/invalid for the same reason.
"""
from __future__ import annotations

import json
from typing import Any

from core.brain.llm.latent_cortex.frontier_tasks import (
    FINAL_ANSWER_MARKER,
    MAX_RESPONSE_BYTES,
    FrontierTaskError,
    parse_final_answer,
)

ANSWER_CONTRACT_SCHEMA = "aura.latent_cortex.answer_contract.v1"

# A contract answer object is small (task payloads are bounded); a scan
# bound keeps adversarial no-close-brace tails from consuming the decode.
_MAX_PAYLOAD_SCAN_CHARS = 8_192


def _complete_json_object_span(tail: str) -> tuple[int, int] | None:
    """[start, end) of the first COMPLETE top-level JSON object in ``tail``.

    Strict-JSON aware brace scan: strings and escapes do not move depth.
    Returns None while the object is still open (or absent).
    """
    started = False
    depth = 0
    in_string = False
    escaped = False
    start_index = 0
    for index, char in enumerate(tail[:_MAX_PAYLOAD_SCAN_CHARS]):
        if not started:
            if char in " \t":
                continue
            if char != "{":
                return None  # payload must be a JSON object
            started = True
            start_index = index
            depth = 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return (start_index, index + 1)
    return None


def contract_answer_state(text: Any) -> dict[str, Any]:
    """Incremental contract analysis of a (possibly still streaming) text."""
    state: dict[str, Any] = {
        "schema": ANSWER_CONTRACT_SCHEMA,
        "marker_count": 0,
        "complete": False,
        "valid": False,
        "parsed": None,
        "reason": "no_marker",
    }
    if not isinstance(text, str) or not text:
        return state
    if len(text.encode("utf-8", errors="replace")) > MAX_RESPONSE_BYTES:
        state["reason"] = "response_too_large"
        return state
    marker_count = text.count(FINAL_ANSWER_MARKER)
    state["marker_count"] = marker_count
    if marker_count == 0:
        return state
    if marker_count > 1:
        state["reason"] = "multiple_markers"
        return state
    tail = text.split(FINAL_ANSWER_MARKER, 1)[1]
    if "\n" in tail.split("{", 1)[0]:
        # Marker line ended before any object began — parser would reject.
        state["reason"] = "marker_line_has_no_object"
        return state
    span = _complete_json_object_span(tail)
    if span is None:
        state["reason"] = "object_open"
        return state
    payload = tail[span[0] : span[1]]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        state["reason"] = "object_invalid_json"
        return state
    if not isinstance(parsed, dict):
        state["reason"] = "object_not_dict"
        return state
    state["complete"] = True
    state["reason"] = "complete"
    # Validity is the FULL strict parser's judgment of the text AS IT
    # STANDS — trailing content after the object (the model kept talking)
    # breaks terminality exactly as parse_final_answer rules it.
    try:
        state["parsed"] = parse_final_answer(text)
        state["valid"] = True
    except FrontierTaskError as exc:
        state["valid"] = False
        state["parsed"] = None
        state["reason"] = f"parser_rejected:{exc}"
    return state


def is_contract_complete(text: Any) -> bool:
    """Decode early-stop predicate: one marker, its JSON object just closed.

    Stopping here makes the answer terminal by construction; identical for
    every arm, receipted by the caller as termination ``contract_complete``.
    """
    return bool(contract_answer_state(text)["complete"])


__all__ = [
    "ANSWER_CONTRACT_SCHEMA",
    "contract_answer_state",
    "is_contract_complete",
]
