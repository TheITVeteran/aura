"""Blind, origin-free branch review with a mechanically provable boundary."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import Any

BLIND_REVIEW_SCHEMA = "aura.rlc.blind_branch_review.v1"
_VISIBLE_FIELDS = ("candidate_text",)
_FORBIDDEN_FIELDS = (
    "anonymous_id",
    "branch",
    "role",
    "operator",
    "selected_branch",
    "first_answer",
    "ownership",
    "doubt_prompt",
)
_EMPTY_REVIEW_TEXT = "[no substantive candidate content]"
_ORIGIN_PATTERNS = (
    re.compile(r"\b(?:my|our)\s+(?:first|previous|earlier)\s+answer\b", re.I),
    re.compile(r"\bas\s+aura\b", re.I),
    re.compile(r"\b(?:branch|candidate)\s*#?\s*\d+\b", re.I),
    re.compile(r"\byou\s+asked\s+me\s+to\s+(?:review|critique)\b", re.I),
    re.compile(r"\bare\s+you\s+sure\b", re.I),
)


def _sha(value: Any) -> str:
    payload = (
        value.encode("utf-8")
        if isinstance(value, str)
        else json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def _blind_text(text: str) -> tuple[str, int]:
    if not isinstance(text, str):
        raise ValueError("blind-review candidate must be text")
    blinded = text
    redactions = 0
    for pattern in _ORIGIN_PATTERNS:
        blinded, count = pattern.subn("", blinded)
        redactions += count
    blinded = re.sub(r"[ \t]{2,}", " ", blinded).strip()
    if not blinded:
        blinded = _EMPTY_REVIEW_TEXT
    return blinded, redactions


def _isolation_sha256(value: Any) -> str:
    if not isinstance(value, dict):
        raise ValueError("blind-review isolation receipt is invalid")
    return _sha(
        {
            key: value.get(key)
            for key in (
                "schema",
                "n_branches",
                "required_steps",
                "sealed",
                "certified",
                "configured_role_lesion",
                "seed_alias_free",
                "seed_states_unique",
                "rng_streams_unique",
                "candidates",
            )
        }
    )


def _deranged_order(indices: list[int], seed: str) -> list[int]:
    if len(indices) <= 1:
        return list(indices)
    ranked = sorted(indices, key=lambda index: _sha(f"{seed}:{index}"))
    for offset in range(1, len(ranked)):
        rotated = ranked[offset:] + ranked[:offset]
        if all(position != branch for position, branch in enumerate(rotated)):
            return rotated
    # Deterministic fallback for unusual non-contiguous inputs.
    return indices[1:] + indices[:1]


def run_blind_review(
    candidates: dict[int, str],
    reviewer: Callable[[str], float],
    *,
    episode_id: str,
    objective_sha256: str,
    isolation_receipt: dict[str, Any],
) -> tuple[dict[int, float], dict[str, Any]]:
    """Review a deranged anonymized batch and return private score mapping."""

    if not candidates or sorted(candidates) != list(range(len(candidates))):
        raise ValueError("blind-review candidates must use contiguous branch indices")
    if not callable(reviewer):
        raise ValueError("blind reviewer is unavailable")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("blind-review episode identity is missing")
    if not isinstance(objective_sha256, str) or len(objective_sha256) != 64:
        raise ValueError("blind-review objective commitment is invalid")
    if not isinstance(isolation_receipt, dict) or isolation_receipt.get("certified") is not True:
        raise ValueError("blind review requires certified fresh-context candidates")

    indices = sorted(candidates)
    seed = _sha(f"blind-review:{episode_id}:{objective_sha256}")
    order = _deranged_order(indices, seed)
    scores: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    for position, branch in enumerate(order):
        original = candidates[branch]
        blinded, redactions = _blind_text(original)
        commitment = _sha(original)
        anonymous_id = _sha(f"{seed}:review-position:{position}:{commitment}")[:20]
        reviewer_view = {"candidate_text": blinded}
        score = float(reviewer(reviewer_view["candidate_text"]))
        if score != score or score in {float("inf"), float("-inf")}:
            raise ValueError("blind reviewer returned a non-finite score")
        scores[branch] = score
        rows.append(
            {
                "review_position": position,
                "branch": branch,
                "anonymous_id": anonymous_id,
                "candidate_sha256": commitment,
                "review_text_sha256": _sha(blinded),
                "origin_redactions": redactions,
                "score": round(score, 6),
            }
        )
    deranged = len(indices) <= 1 or all(
        row["review_position"] != row["branch"] for row in rows
    )
    payload = {
        "schema": BLIND_REVIEW_SCHEMA,
        "objective_sha256": objective_sha256,
        "isolation_receipt_sha256": _isolation_sha256(isolation_receipt),
        "reviewer_visible_fields": list(_VISIBLE_FIELDS),
        "reviewer_forbidden_fields": list(_FORBIDDEN_FIELDS),
        "candidate_count": len(indices),
        "deranged_order": deranged,
        "first_answer_designated": False,
        "ownership_framing_supplied": False,
        "doubt_framing_supplied": False,
        "rows": rows,
    }
    return scores, {**payload, "receipt_sha256": _sha(payload)}


def validate_blind_review_receipt(
    value: Any,
    *,
    n_branches: int,
    branch_scores: Any,
    isolation_receipt: Any,
    objective_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("blind-review receipt is missing")
    required = {
        "schema",
        "objective_sha256",
        "isolation_receipt_sha256",
        "reviewer_visible_fields",
        "reviewer_forbidden_fields",
        "candidate_count",
        "deranged_order",
        "first_answer_designated",
        "ownership_framing_supplied",
        "doubt_framing_supplied",
        "rows",
        "receipt_sha256",
    }
    if set(value) != required or value.get("schema") != BLIND_REVIEW_SCHEMA:
        raise ValueError("blind-review receipt schema is invalid")
    if (
        type(n_branches) is not int
        or n_branches <= 0
        or value.get("candidate_count") != n_branches
        or value.get("objective_sha256") != objective_sha256
        or value.get("reviewer_visible_fields") != list(_VISIBLE_FIELDS)
        or value.get("reviewer_forbidden_fields") != list(_FORBIDDEN_FIELDS)
        or value.get("deranged_order") is not True
        or value.get("first_answer_designated") is not False
        or value.get("ownership_framing_supplied") is not False
        or value.get("doubt_framing_supplied") is not False
        or value.get("isolation_receipt_sha256") != _isolation_sha256(isolation_receipt)
    ):
        raise ValueError("blind-review boundary is invalid")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != n_branches:
        raise ValueError("blind-review row coverage is invalid")
    if not isinstance(branch_scores, list) or len(branch_scores) != n_branches:
        raise ValueError("blind-review score coverage is invalid")
    seen_branches: set[int] = set()
    seen_ids: set[str] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "review_position", "branch", "anonymous_id", "candidate_sha256",
            "review_text_sha256", "origin_redactions", "score",
        }:
            raise ValueError("blind-review row is invalid")
        branch = row.get("branch")
        if (
            row.get("review_position") != position
            or type(branch) is not int
            or not 0 <= branch < n_branches
            or branch in seen_branches
            or (n_branches > 1 and branch == position)
            or not isinstance(row.get("anonymous_id"), str)
            or len(row["anonymous_id"]) != 20
            or row["anonymous_id"] in seen_ids
            or not isinstance(row.get("candidate_sha256"), str)
            or len(row["candidate_sha256"]) != 64
            or not isinstance(row.get("review_text_sha256"), str)
            or len(row["review_text_sha256"]) != 64
            or type(row.get("origin_redactions")) is not int
            or row["origin_redactions"] < 0
            or row.get("score") != round(float(branch_scores[branch]), 6)
        ):
            raise ValueError("blind-review mapping is invalid")
        seen_branches.add(branch)
        seen_ids.add(row["anonymous_id"])
    payload = {key: value[key] for key in required - {"receipt_sha256"}}
    if value.get("receipt_sha256") != _sha(payload):
        raise ValueError("blind-review receipt digest differs")
    return dict(value)


__all__ = [
    "BLIND_REVIEW_SCHEMA",
    "run_blind_review",
    "validate_blind_review_receipt",
]
