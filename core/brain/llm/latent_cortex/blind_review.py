"""Blind, origin-free branch review with a mechanically provable boundary."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import Any

BLIND_REVIEW_SCHEMA = "aura.rlc.blind_branch_review.v1"
DECOY_REVIEW_SCHEMA = "aura.rlc.decoy_balanced_review.v1"
DECOY_PREFLIGHT_SCHEMA = "aura.rlc.decoy_preflight.v1"
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
_CONTROL_KINDS = ("correct", "incorrect", "unchanged_a", "unchanged_b")
_CORRECT_MARGIN = 0.05
_UNCHANGED_TOLERANCE = 1e-9
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


def _review_score(value: Any, *, source: str) -> float:
    score = float(value)
    if score != score or score in {float("inf"), float("-inf")}:
        raise ValueError(f"{source} returned a non-finite score")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{source} returned a score outside [0, 1]")
    return score


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


def _control_texts(seed: str) -> dict[str, str]:
    raw = bytes.fromhex(seed)
    a = 100 + int.from_bytes(raw[0:2], "big") % 800
    b = 100 + int.from_bytes(raw[2:4], "big") % 800
    c = 20 + int.from_bytes(raw[4:6], "big") % 80
    d = 3 + int.from_bytes(raw[6:8], "big") % 17
    wrong_delta = 1 + int.from_bytes(raw[8:10], "big") % 9
    u = 300 + int.from_bytes(raw[10:12], "big") % 700
    v = 10 + int.from_bytes(raw[12:14], "big") % 200
    valid_code = "```python\ndef check(value):\n    return value + 1\n```"
    invalid_code = "```python\ndef check(value)\n    return value + 1\n```"
    unchanged = (
        f"Independent consistency check: {u} - {v} = {u - v}.\n{valid_code}"
    )
    return {
        "correct": (
            f"Independent consistency check: {a} + {b} = {a + b}.\n{valid_code}"
        ),
        "incorrect": (
            f"Independent consistency check: {c} * {d} = {c * d + wrong_delta}.\n"
            f"{invalid_code}"
        ),
        "unchanged_a": unchanged,
        "unchanged_b": unchanged,
    }


def _batch_plan(n_branches: int, seed: str) -> list[tuple[str, int | str]]:
    branch_order = _deranged_order(list(range(n_branches)), seed)
    control_order = sorted(_CONTROL_KINDS, key=lambda kind: _sha(f"{seed}:{kind}"))
    size = n_branches + len(control_order)
    control_positions = set(
        sorted(range(size), key=lambda pos: _sha(f"{seed}:slot:{pos}"))[
            : len(control_order)
        ]
    )
    branches = iter(branch_order)
    controls = iter(control_order)
    return [
        ("control", next(controls)) if position in control_positions
        else ("candidate", next(branches))
        for position in range(size)
    ]


def _calibration(control_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = {row["kind"]: float(row["score"]) for row in control_rows}
    correct = scores["correct"] >= scores["incorrect"] + _CORRECT_MARGIN
    spread = abs(scores["unchanged_a"] - scores["unchanged_b"])
    consistent = spread <= _UNCHANGED_TOLERANCE
    return {
        "correct_above_incorrect": correct,
        "unchanged_score_spread": round(spread, 12),
        "unchanged_consistent": consistent,
        "certified": correct and consistent,
    }


def run_decoy_preflight(
    reviewer: Callable[[str], float],
    *,
    episode_id: str,
    objective_sha256: str,
) -> dict[str, Any]:
    """Calibrate reviewer authority before it can alter recurrent state."""

    if not callable(reviewer):
        raise ValueError("decoy preflight reviewer is unavailable")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("decoy preflight episode identity is missing")
    if not isinstance(objective_sha256, str) or len(objective_sha256) != 64:
        raise ValueError("decoy preflight objective commitment is invalid")
    seed = _sha(f"decoy-preflight:{episode_id}:{objective_sha256}")
    controls = _control_texts(seed)
    order = sorted(_CONTROL_KINDS, key=lambda kind: _sha(f"{seed}:{kind}"))
    evaluations = getattr(reviewer, "evaluations", None)
    evaluation_offset = len(evaluations) if isinstance(evaluations, list) else None
    rows: list[dict[str, Any]] = []
    for position, kind in enumerate(order):
        text = controls[kind]
        text_sha256 = _sha(text)
        score = _review_score(reviewer(text), source="decoy preflight reviewer")
        rows.append(
            {
                "kind": kind,
                "position": position,
                "anonymous_id": _sha(
                    f"{seed}:control:{kind}:{text_sha256}"
                )[:20],
                "text_sha256": text_sha256,
                "score": round(score, 6),
            }
        )
    calibration = _calibration(rows)
    payload = {
        "schema": DECOY_PREFLIGHT_SCHEMA,
        "objective_sha256": objective_sha256,
        "seed_sha256": seed,
        "reviewer_visible_fields": list(_VISIBLE_FIELDS),
        "labels_withheld_during_review": True,
        "controls": rows,
        "correct_margin": _CORRECT_MARGIN,
        "unchanged_tolerance": _UNCHANGED_TOLERANCE,
        **calibration,
        "verifier_admitted": calibration["certified"],
        "evaluation_offset": evaluation_offset,
        "control_evaluation_indices": (
            [evaluation_offset + row["position"] for row in rows]
            if evaluation_offset is not None
            else []
        ),
    }
    return {**payload, "receipt_sha256": _sha(payload)}


def run_decoy_balanced_review(
    candidates: dict[int, str],
    reviewer: Callable[[str], float],
    *,
    episode_id: str,
    objective_sha256: str,
    isolation_receipt: dict[str, Any],
) -> tuple[dict[int, float], dict[str, Any], dict[str, Any]]:
    """Run candidates and concealed controls through one mixed review batch."""

    if not candidates or sorted(candidates) != list(range(len(candidates))):
        raise ValueError("decoy-review candidates must use contiguous branch indices")
    if not callable(reviewer):
        raise ValueError("decoy reviewer is unavailable")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("decoy-review episode identity is missing")
    if not isinstance(objective_sha256, str) or len(objective_sha256) != 64:
        raise ValueError("decoy-review objective commitment is invalid")
    if not isinstance(isolation_receipt, dict) or isolation_receipt.get("certified") is not True:
        raise ValueError("decoy review requires certified fresh-context candidates")

    blind_seed = _sha(f"blind-review:{episode_id}:{objective_sha256}")
    decoy_seed = _sha(f"decoy-review:{episode_id}:{objective_sha256}")
    controls = _control_texts(decoy_seed)
    plan = _batch_plan(len(candidates), decoy_seed)
    evaluation_offset: int | None = None
    evaluations = getattr(reviewer, "evaluations", None)
    if isinstance(evaluations, list):
        evaluation_offset = len(evaluations)

    candidate_scores: dict[int, float] = {}
    candidate_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    candidate_rank = 0
    for batch_position, (item_class, reference) in enumerate(plan):
        if item_class == "candidate":
            branch = int(reference)
            original = candidates[branch]
            review_text, redactions = _blind_text(original)
            commitment = _sha(original)
            anonymous_id = _sha(
                f"{blind_seed}:review-position:{candidate_rank}:{commitment}"
            )[:20]
            score = _review_score(reviewer(review_text), source="blind reviewer")
            candidate_scores[branch] = score
            candidate_rows.append(
                {
                    "review_position": candidate_rank,
                    "branch": branch,
                    "anonymous_id": anonymous_id,
                    "candidate_sha256": commitment,
                    "review_text_sha256": _sha(review_text),
                    "origin_redactions": redactions,
                    "score": round(score, 6),
                }
            )
            batch_rows.append(
                {
                    "batch_position": batch_position,
                    "item_class": "candidate",
                    "reference": branch,
                    "anonymous_id": anonymous_id,
                    "text_sha256": _sha(review_text),
                }
            )
            candidate_rank += 1
            continue

        kind = str(reference)
        review_text = controls[kind]
        text_sha256 = _sha(review_text)
        anonymous_id = _sha(f"{decoy_seed}:control:{kind}:{text_sha256}")[:20]
        score = _review_score(reviewer(review_text), source="decoy reviewer")
        control_rows.append(
            {
                "kind": kind,
                "batch_position": batch_position,
                "anonymous_id": anonymous_id,
                "text_sha256": text_sha256,
                "score": round(score, 6),
            }
        )
        batch_rows.append(
            {
                "batch_position": batch_position,
                "item_class": "control",
                "reference": kind,
                "anonymous_id": anonymous_id,
                "text_sha256": text_sha256,
            }
        )

    calibration = _calibration(control_rows)

    blind_payload = {
        "schema": BLIND_REVIEW_SCHEMA,
        "objective_sha256": objective_sha256,
        "isolation_receipt_sha256": _isolation_sha256(isolation_receipt),
        "reviewer_visible_fields": list(_VISIBLE_FIELDS),
        "reviewer_forbidden_fields": list(_FORBIDDEN_FIELDS),
        "candidate_count": len(candidates),
        "deranged_order": all(
            row["review_position"] != row["branch"] for row in candidate_rows
        ),
        "first_answer_designated": False,
        "ownership_framing_supplied": False,
        "doubt_framing_supplied": False,
        "rows": candidate_rows,
    }
    blind_receipt = {
        **blind_payload,
        "receipt_sha256": _sha(blind_payload),
    }
    decoy_payload = {
        "schema": DECOY_REVIEW_SCHEMA,
        "objective_sha256": objective_sha256,
        "batch_seed_sha256": decoy_seed,
        "reviewer_visible_fields": list(_VISIBLE_FIELDS),
        "labels_withheld_during_review": True,
        "candidate_count": len(candidates),
        "control_count": len(_CONTROL_KINDS),
        "batch_size": len(plan),
        "batch_rows": batch_rows,
        "controls": control_rows,
        "correct_margin": _CORRECT_MARGIN,
        "unchanged_tolerance": _UNCHANGED_TOLERANCE,
        **calibration,
        "selection_admitted": calibration["certified"],
        "evaluation_offset": evaluation_offset,
        "control_evaluation_indices": (
            [
                evaluation_offset + row["batch_position"]
                for row in control_rows
            ]
            if evaluation_offset is not None
            else []
        ),
    }
    decoy_receipt = {
        **decoy_payload,
        "receipt_sha256": _sha(decoy_payload),
    }
    return candidate_scores, blind_receipt, decoy_receipt


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
        score = _review_score(
            reviewer(reviewer_view["candidate_text"]),
            source="blind reviewer",
        )
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
    episode_id: str,
    selected_branch: int,
    decoy_receipt: Any = None,
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
    if type(selected_branch) is not int or not 0 <= selected_branch < n_branches:
        raise ValueError("blind-review selected branch is invalid")
    selection_admitted = True
    if decoy_receipt is not None:
        validated_decoy = validate_decoy_review_receipt(
            decoy_receipt,
            blind_receipt=value,
            episode_id=episode_id,
            objective_sha256=objective_sha256,
        )
        selection_admitted = validated_decoy["selection_admitted"] is True
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
            or isinstance(row.get("score"), bool)
            or not isinstance(row.get("score"), (int, float))
            or not 0.0 <= float(row["score"]) <= 1.0
            or (
                selection_admitted
                and row.get("score") != round(float(branch_scores[branch]), 6)
            )
        ):
            raise ValueError("blind-review mapping is invalid")
        seen_branches.add(branch)
        seen_ids.add(row["anonymous_id"])
    if selection_admitted and selected_branch != max(
        range(n_branches),
        key=lambda branch: float(branch_scores[branch]),
    ):
        raise ValueError("blind-review selected branch differs from admitted scores")
    payload = {key: value[key] for key in required - {"receipt_sha256"}}
    if value.get("receipt_sha256") != _sha(payload):
        raise ValueError("blind-review receipt digest differs")
    return dict(value)


def validate_decoy_review_receipt(
    value: Any,
    *,
    blind_receipt: Any,
    episode_id: str,
    objective_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(blind_receipt, dict):
        raise ValueError("decoy-review receipt is missing")
    required = {
        "schema",
        "objective_sha256",
        "batch_seed_sha256",
        "reviewer_visible_fields",
        "labels_withheld_during_review",
        "candidate_count",
        "control_count",
        "batch_size",
        "batch_rows",
        "controls",
        "correct_margin",
        "unchanged_tolerance",
        "correct_above_incorrect",
        "unchanged_score_spread",
        "unchanged_consistent",
        "certified",
        "selection_admitted",
        "evaluation_offset",
        "control_evaluation_indices",
        "receipt_sha256",
    }
    n_branches = blind_receipt.get("candidate_count")
    seed = _sha(f"decoy-review:{episode_id}:{objective_sha256}")
    if (
        set(value) != required
        or value.get("schema") != DECOY_REVIEW_SCHEMA
        or value.get("objective_sha256") != objective_sha256
        or value.get("batch_seed_sha256") != seed
        or value.get("reviewer_visible_fields") != list(_VISIBLE_FIELDS)
        or value.get("labels_withheld_during_review") is not True
        or type(n_branches) is not int
        or n_branches <= 0
        or value.get("candidate_count") != n_branches
        or value.get("control_count") != len(_CONTROL_KINDS)
        or value.get("batch_size") != n_branches + len(_CONTROL_KINDS)
        or value.get("correct_margin") != _CORRECT_MARGIN
        or value.get("unchanged_tolerance") != _UNCHANGED_TOLERANCE
    ):
        raise ValueError("decoy-review boundary is invalid")

    blind_rows = blind_receipt.get("rows")
    batch_rows = value.get("batch_rows")
    controls = value.get("controls")
    if (
        not isinstance(blind_rows, list)
        or len(blind_rows) != n_branches
        or not isinstance(batch_rows, list)
        or len(batch_rows) != value["batch_size"]
        or not isinstance(controls, list)
        or len(controls) != len(_CONTROL_KINDS)
    ):
        raise ValueError("decoy-review coverage is invalid")

    control_texts = _control_texts(seed)
    control_by_kind: dict[str, dict[str, Any]] = {}
    for row in controls:
        if not isinstance(row, dict) or set(row) != {
            "kind", "batch_position", "anonymous_id", "text_sha256", "score",
        }:
            raise ValueError("decoy-review control row is invalid")
        kind = row.get("kind")
        if kind not in _CONTROL_KINDS or kind in control_by_kind:
            raise ValueError("decoy-review control identity is invalid")
        text_sha256 = _sha(control_texts[kind])
        expected_id = _sha(f"{seed}:control:{kind}:{text_sha256}")[:20]
        score = row.get("score")
        if (
            row.get("text_sha256") != text_sha256
            or row.get("anonymous_id") != expected_id
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or float(score) != float(score)
            or float(score) in {float("inf"), float("-inf")}
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError("decoy-review control evidence is invalid")
        control_by_kind[kind] = row

    plan = _batch_plan(n_branches, seed)
    blind_by_branch = {row["branch"]: row for row in blind_rows}
    for position, (row, expected) in enumerate(zip(batch_rows, plan, strict=True)):
        item_class, reference = expected
        if not isinstance(row, dict) or set(row) != {
            "batch_position", "item_class", "reference", "anonymous_id", "text_sha256",
        }:
            raise ValueError("decoy-review batch row is invalid")
        if (
            row.get("batch_position") != position
            or row.get("item_class") != item_class
            or row.get("reference") != reference
        ):
            raise ValueError("decoy-review batch order differs")
        source = (
            blind_by_branch[int(reference)]
            if item_class == "candidate"
            else control_by_kind[str(reference)]
        )
        expected_text_sha = (
            source["review_text_sha256"]
            if item_class == "candidate"
            else source["text_sha256"]
        )
        if (
            row.get("anonymous_id") != source["anonymous_id"]
            or row.get("text_sha256") != expected_text_sha
        ):
            raise ValueError("decoy-review batch evidence differs")

    scores = {kind: float(row["score"]) for kind, row in control_by_kind.items()}
    correct = scores["correct"] >= scores["incorrect"] + _CORRECT_MARGIN
    spread = abs(scores["unchanged_a"] - scores["unchanged_b"])
    consistent = spread <= _UNCHANGED_TOLERANCE
    certified = correct and consistent
    if (
        value.get("correct_above_incorrect") is not correct
        or value.get("unchanged_score_spread") != round(spread, 12)
        or value.get("unchanged_consistent") is not consistent
        or value.get("certified") is not certified
        or value.get("selection_admitted") is not certified
    ):
        raise ValueError("decoy-review calibration verdict differs")

    offset = value.get("evaluation_offset")
    indices = value.get("control_evaluation_indices")
    if offset is None:
        if indices != []:
            raise ValueError("decoy-review evaluation indices are invalid")
    elif (
        type(offset) is not int
        or offset < 0
        or indices
        != [offset + row["batch_position"] for row in controls]
    ):
        raise ValueError("decoy-review evaluation indices are invalid")

    payload = {key: value[key] for key in required - {"receipt_sha256"}}
    if value.get("receipt_sha256") != _sha(payload):
        raise ValueError("decoy-review receipt digest differs")
    return dict(value)


def validate_decoy_preflight_receipt(
    value: Any,
    *,
    episode_id: str,
    objective_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("decoy preflight receipt is missing")
    required = {
        "schema",
        "objective_sha256",
        "seed_sha256",
        "reviewer_visible_fields",
        "labels_withheld_during_review",
        "controls",
        "correct_margin",
        "unchanged_tolerance",
        "correct_above_incorrect",
        "unchanged_score_spread",
        "unchanged_consistent",
        "certified",
        "verifier_admitted",
        "evaluation_offset",
        "control_evaluation_indices",
        "receipt_sha256",
    }
    seed = _sha(f"decoy-preflight:{episode_id}:{objective_sha256}")
    if (
        set(value) != required
        or value.get("schema") != DECOY_PREFLIGHT_SCHEMA
        or value.get("objective_sha256") != objective_sha256
        or value.get("seed_sha256") != seed
        or value.get("reviewer_visible_fields") != list(_VISIBLE_FIELDS)
        or value.get("labels_withheld_during_review") is not True
        or value.get("correct_margin") != _CORRECT_MARGIN
        or value.get("unchanged_tolerance") != _UNCHANGED_TOLERANCE
    ):
        raise ValueError("decoy preflight boundary is invalid")
    rows = value.get("controls")
    order = sorted(_CONTROL_KINDS, key=lambda kind: _sha(f"{seed}:{kind}"))
    expected_texts = _control_texts(seed)
    if not isinstance(rows, list) or len(rows) != len(_CONTROL_KINDS):
        raise ValueError("decoy preflight coverage is invalid")
    for position, (row, kind) in enumerate(zip(rows, order, strict=True)):
        if not isinstance(row, dict) or set(row) != {
            "kind", "position", "anonymous_id", "text_sha256", "score",
        }:
            raise ValueError("decoy preflight control row is invalid")
        text_sha256 = _sha(expected_texts[kind])
        score = row.get("score")
        if (
            row.get("kind") != kind
            or row.get("position") != position
            or row.get("text_sha256") != text_sha256
            or row.get("anonymous_id")
            != _sha(f"{seed}:control:{kind}:{text_sha256}")[:20]
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or float(score) != float(score)
            or float(score) in {float("inf"), float("-inf")}
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError("decoy preflight control evidence is invalid")
    calibration = _calibration(rows)
    if any(value.get(key) != expected for key, expected in calibration.items()) or (
        value.get("verifier_admitted") is not calibration["certified"]
    ):
        raise ValueError("decoy preflight calibration verdict differs")
    offset = value.get("evaluation_offset")
    indices = value.get("control_evaluation_indices")
    if offset is None:
        if indices != []:
            raise ValueError("decoy preflight evaluation indices are invalid")
    elif (
        type(offset) is not int
        or offset < 0
        or indices != [offset + row["position"] for row in rows]
    ):
        raise ValueError("decoy preflight evaluation indices are invalid")
    payload = {key: value[key] for key in required - {"receipt_sha256"}}
    if value.get("receipt_sha256") != _sha(payload):
        raise ValueError("decoy preflight receipt digest differs")
    return dict(value)


__all__ = [
    "BLIND_REVIEW_SCHEMA",
    "DECOY_REVIEW_SCHEMA",
    "DECOY_PREFLIGHT_SCHEMA",
    "run_blind_review",
    "run_decoy_balanced_review",
    "run_decoy_preflight",
    "validate_blind_review_receipt",
    "validate_decoy_review_receipt",
    "validate_decoy_preflight_receipt",
]
