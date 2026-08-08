"""Research-only output arbitration against a hidden benchmark oracle.

This module is intentionally not a serving policy.  It lets a preregistered
oracle arm expose whether the recurrent system generated a correct candidate
that the deployable verifier stack failed to promote.  The public receipt
commits to the task, scorer, candidate texts, exact verdicts, and selected
output without exposing the answer key or candidate prose.

The rule is monotonic with respect to the measured benchmark: replace the
current output only when it is oracle-refuted and the already-selected
recurrent candidate is oracle-verified.  All other cases retain the current
output.  Receipts explicitly carry no serving or capability-claim authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

RESEARCH_ORACLE_ASSESSMENT_SCHEMA = "aura.rlc.research_oracle_assessment.v1"
RESEARCH_ORACLE_ARBITRATION_SCHEMA = "aura.rlc.research_oracle_arbitration.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _token_sha(tokens: Sequence[int]) -> str:
    return _sha(list(tokens))


def _require_sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def build_research_oracle_assessment(
    *,
    candidate: str,
    task_id: str,
    task_payload_sha256: str,
    answer_commitment_sha256: str,
    scorer_id: str,
    scorer_version: str,
    scorer_source_sha256: str,
    parsed: bool,
    correct: bool,
    reason: str,
    normalized_answer_sha256: str | None,
) -> dict[str, Any]:
    """Bind one hidden-ground-truth verdict without exposing answer material."""

    if not isinstance(candidate, str):
        raise ValueError("research oracle candidate must be text")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("research oracle task identity is missing")
    for field, value in (
        ("task_payload_sha256", task_payload_sha256),
        ("answer_commitment_sha256", answer_commitment_sha256),
        ("scorer_source_sha256", scorer_source_sha256),
    ):
        _require_sha(value, field=field)
    if (
        not isinstance(scorer_id, str)
        or not scorer_id
        or not isinstance(scorer_version, str)
        or not scorer_version
        or type(parsed) is not bool
        or type(correct) is not bool
        or not isinstance(reason, str)
        or not reason
    ):
        raise ValueError("research oracle scorer verdict is invalid")
    if normalized_answer_sha256 is not None:
        _require_sha(normalized_answer_sha256, field="normalized_answer_sha256")
    if correct and (not parsed or normalized_answer_sha256 is None):
        raise ValueError("a correct oracle verdict requires a parsed bound answer")
    payload = {
        "schema": RESEARCH_ORACLE_ASSESSMENT_SCHEMA,
        "scope": "research_oracle_only",
        "task_id": task_id,
        "task_payload_sha256": task_payload_sha256,
        "answer_commitment_sha256": answer_commitment_sha256,
        "scorer_id": scorer_id,
        "scorer_version": scorer_version,
        "scorer_source_sha256": scorer_source_sha256,
        "candidate_sha256": _text_sha(candidate),
        "parsed": parsed,
        "correct": correct,
        "reason": reason,
        "normalized_answer_sha256": normalized_answer_sha256,
        "full_span_ground_truth": True,
        "quality_interval": {
            "lower_bound": 1.0 if correct else 0.0,
            "upper_bound": 1.0 if correct else 0.0,
        },
        "answer_key_exposed": False,
        "serving_authority": False,
        "capability_claim_authority": False,
        "research_measurement_authority": True,
    }
    return {**payload, "receipt_sha256": _sha(payload)}


def validate_research_oracle_assessment(
    value: Any,
    *,
    candidate: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("research oracle assessment is missing")
    required = {
        "schema",
        "scope",
        "task_id",
        "task_payload_sha256",
        "answer_commitment_sha256",
        "scorer_id",
        "scorer_version",
        "scorer_source_sha256",
        "candidate_sha256",
        "parsed",
        "correct",
        "reason",
        "normalized_answer_sha256",
        "full_span_ground_truth",
        "quality_interval",
        "answer_key_exposed",
        "serving_authority",
        "capability_claim_authority",
        "research_measurement_authority",
        "receipt_sha256",
    }
    if set(value) != required:
        raise ValueError("research oracle assessment fields differ")
    payload = {key: value[key] for key in required - {"receipt_sha256"}}
    correct = value.get("correct")
    expected_interval = {
        "lower_bound": 1.0 if correct is True else 0.0,
        "upper_bound": 1.0 if correct is True else 0.0,
    }
    if (
        value.get("schema") != RESEARCH_ORACLE_ASSESSMENT_SCHEMA
        or value.get("scope") != "research_oracle_only"
        or value.get("candidate_sha256") != _text_sha(candidate)
        or type(value.get("parsed")) is not bool
        or type(correct) is not bool
        or not isinstance(value.get("task_id"), str)
        or not value["task_id"]
        or not isinstance(value.get("scorer_id"), str)
        or not value["scorer_id"]
        or not isinstance(value.get("scorer_version"), str)
        or not value["scorer_version"]
        or not isinstance(value.get("reason"), str)
        or not value["reason"]
        or value.get("full_span_ground_truth") is not True
        or value.get("quality_interval") != expected_interval
        or value.get("answer_key_exposed") is not False
        or value.get("serving_authority") is not False
        or value.get("capability_claim_authority") is not False
        or value.get("research_measurement_authority") is not True
        or value.get("receipt_sha256") != _sha(payload)
    ):
        raise ValueError("research oracle assessment is invalid")
    for field in (
        "task_payload_sha256",
        "answer_commitment_sha256",
        "scorer_source_sha256",
    ):
        _require_sha(value.get(field), field=field)
    normalized = value.get("normalized_answer_sha256")
    if normalized is not None:
        _require_sha(normalized, field="normalized_answer_sha256")
    if correct and (value.get("parsed") is not True or normalized is None):
        raise ValueError("correct research oracle assessment is not bound")
    return dict(value)


def build_research_oracle_arbitration(
    *,
    current_text: str,
    current_tokens: Sequence[int],
    recurrent_text: str,
    recurrent_tokens: Sequence[int],
    selected_branch: int,
    assess: Callable[[str], Mapping[str, Any]],
) -> tuple[dict[str, Any], list[int]]:
    """Promote only an oracle-correct recurrent answer over an oracle-wrong one."""

    if (
        not isinstance(current_text, str)
        or not isinstance(recurrent_text, str)
        or type(selected_branch) is not int
        or selected_branch < 0
        or not callable(assess)
    ):
        raise ValueError("research oracle arbitration inputs are invalid")
    current = list(current_tokens)
    recurrent = list(recurrent_tokens)
    if any(type(token) is not int or token < 0 for token in current + recurrent):
        raise ValueError("research oracle arbitration tokens are invalid")
    if bool(current_text) is not bool(current) or bool(recurrent_text) is not bool(recurrent):
        raise ValueError("research oracle text/token presence differs")
    current_assessment = validate_research_oracle_assessment(
        assess(current_text),
        candidate=current_text,
    )
    recurrent_assessment = validate_research_oracle_assessment(
        assess(recurrent_text),
        candidate=recurrent_text,
    )
    identity_fields = (
        "task_id",
        "task_payload_sha256",
        "answer_commitment_sha256",
        "scorer_id",
        "scorer_version",
        "scorer_source_sha256",
    )
    if any(current_assessment[field] != recurrent_assessment[field] for field in identity_fields):
        raise ValueError("research oracle candidate assessments bind different tasks")
    replace = bool(
        current_assessment["correct"] is False
        and recurrent_assessment["correct"] is True
    )
    accepted_text = recurrent_text if replace else current_text
    accepted_tokens = recurrent if replace else current
    decision = "replace" if replace else "retain"
    reason = (
        "oracle_correct_recurrent_candidate_replaces_oracle_wrong_current_output"
        if replace
        else "current_output_or_recurrent_candidate_not_strictly_oracle_dominated"
    )
    payload = {
        "schema": RESEARCH_ORACLE_ARBITRATION_SCHEMA,
        "scope": "research_oracle_only",
        "selected_branch": selected_branch,
        "task_binding": {field: current_assessment[field] for field in identity_fields},
        "current_assessment_sha256": current_assessment["receipt_sha256"],
        "recurrent_assessment_sha256": recurrent_assessment["receipt_sha256"],
        "current_output": {
            "text_sha256": _text_sha(current_text),
            "tokens_sha256": _token_sha(current),
            "token_count": len(current),
            "correct": current_assessment["correct"],
        },
        "recurrent_output": {
            "text_sha256": _text_sha(recurrent_text),
            "tokens_sha256": _token_sha(recurrent),
            "token_count": len(recurrent),
            "correct": recurrent_assessment["correct"],
        },
        "decision": decision,
        "reason": reason,
        "accepted_output": {
            "source": "recurrent_candidate" if replace else "current_output",
            "text_sha256": _text_sha(accepted_text),
            "tokens_sha256": _token_sha(accepted_tokens),
            "token_count": len(accepted_tokens),
        },
        "selection_rule": "replace_iff_current_wrong_and_selected_recurrent_correct",
        "answer_key_exposed": False,
        "serving_authority": False,
        "capability_claim_authority": False,
        "research_measurement_authority": True,
    }
    receipt = {**payload, "receipt_sha256": _sha(payload)}
    validate_research_oracle_arbitration(
        receipt,
        current_text=current_text,
        current_tokens=current,
        recurrent_text=recurrent_text,
        recurrent_tokens=recurrent,
        selected_branch=selected_branch,
        current_assessment=current_assessment,
        recurrent_assessment=recurrent_assessment,
        expected_output_text=accepted_text,
        expected_output_tokens=accepted_tokens,
    )
    return receipt, accepted_tokens


def validate_research_oracle_arbitration(
    value: Any,
    *,
    current_text: str,
    current_tokens: Sequence[int],
    recurrent_text: str,
    recurrent_tokens: Sequence[int],
    selected_branch: int,
    current_assessment: Any,
    recurrent_assessment: Any,
    expected_output_text: str | None = None,
    expected_output_tokens: Sequence[int] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("research oracle arbitration is missing")
    current = list(current_tokens)
    recurrent = list(recurrent_tokens)
    left = validate_research_oracle_assessment(current_assessment, candidate=current_text)
    right = validate_research_oracle_assessment(recurrent_assessment, candidate=recurrent_text)
    replace = left["correct"] is False and right["correct"] is True
    accepted_text = recurrent_text if replace else current_text
    accepted_tokens = recurrent if replace else current
    # Reconstruct directly to avoid recursive validation through the builder.
    identity_fields = (
        "task_id",
        "task_payload_sha256",
        "answer_commitment_sha256",
        "scorer_id",
        "scorer_version",
        "scorer_source_sha256",
    )
    if any(left[field] != right[field] for field in identity_fields):
        raise ValueError("research oracle candidate assessments bind different tasks")
    payload = {
        "schema": RESEARCH_ORACLE_ARBITRATION_SCHEMA,
        "scope": "research_oracle_only",
        "selected_branch": selected_branch,
        "task_binding": {field: left[field] for field in identity_fields},
        "current_assessment_sha256": left["receipt_sha256"],
        "recurrent_assessment_sha256": right["receipt_sha256"],
        "current_output": {
            "text_sha256": _text_sha(current_text),
            "tokens_sha256": _token_sha(current),
            "token_count": len(current),
            "correct": left["correct"],
        },
        "recurrent_output": {
            "text_sha256": _text_sha(recurrent_text),
            "tokens_sha256": _token_sha(recurrent),
            "token_count": len(recurrent),
            "correct": right["correct"],
        },
        "decision": "replace" if replace else "retain",
        "reason": (
            "oracle_correct_recurrent_candidate_replaces_oracle_wrong_current_output"
            if replace
            else "current_output_or_recurrent_candidate_not_strictly_oracle_dominated"
        ),
        "accepted_output": {
            "source": "recurrent_candidate" if replace else "current_output",
            "text_sha256": _text_sha(accepted_text),
            "tokens_sha256": _token_sha(accepted_tokens),
            "token_count": len(accepted_tokens),
        },
        "selection_rule": "replace_iff_current_wrong_and_selected_recurrent_correct",
        "answer_key_exposed": False,
        "serving_authority": False,
        "capability_claim_authority": False,
        "research_measurement_authority": True,
    }
    reconstructed = {**payload, "receipt_sha256": _sha(payload)}
    if dict(value) != reconstructed:
        raise ValueError("research oracle arbitration reconstruction differs")
    if expected_output_text is not None and expected_output_text != accepted_text:
        raise ValueError("research oracle arbitration output text differs")
    if expected_output_tokens is not None and list(expected_output_tokens) != accepted_tokens:
        raise ValueError("research oracle arbitration output tokens differ")
    return dict(value)


__all__ = [
    "RESEARCH_ORACLE_ARBITRATION_SCHEMA",
    "RESEARCH_ORACLE_ASSESSMENT_SCHEMA",
    "build_research_oracle_arbitration",
    "build_research_oracle_assessment",
    "validate_research_oracle_arbitration",
    "validate_research_oracle_assessment",
]
