"""Receipts for equal-compute verifier search over episodic write strength."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

VERIFIER_GAIN_SEARCH_SCHEMA = "aura.rlc.verifier_gain_search.v1"
# Zero is a first-class candidate.  Therefore the search cannot select a
# verifier-worse write, even before the independent strict-improvement and
# matched-sham gates run.  Increasing absolute magnitudes explore the learned
# direction; signed points test whether credit assignment inverted it.
VERIFIER_GAIN_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_rows(
    rows: Any,
    *,
    arm: str,
    gain_grid: Sequence[float],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(gain_grid):
        raise ValueError("verifier gain-search row coverage differs")
    normalized = []
    for index, (row, gain) in enumerate(zip(rows, gain_grid, strict=True)):
        if not isinstance(row, Mapping) or set(row) != {
            "arm",
            "index",
            "gain",
            "probe_tokens_sha256",
            "probe_token_count",
            "score",
            "layer_apps",
        }:
            raise ValueError("verifier gain-search row fields differ")
        if (
            row["arm"] != arm
            or row["index"] != index
            or float(row["gain"]) != float(gain)
            or not _is_sha(row["probe_tokens_sha256"])
            or type(row["probe_token_count"]) is not int
            or row["probe_token_count"] <= 0
            or isinstance(row["score"], bool)
            or not isinstance(row["score"], (int, float))
            or not math.isfinite(float(row["score"]))
            or type(row["layer_apps"]) is not int
            or row["layer_apps"] <= 0
        ):
            raise ValueError("verifier gain-search row is invalid")
        normalized.append(dict(row))
    return normalized


def _selected(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # Grid order is the deterministic tie-break.  Zero comes first, so equal
    # evidence selects the least invasive point instead of changing cognition
    # without demonstrated benefit.
    return dict(max(rows, key=lambda row: float(row["score"])))


def build_verifier_gain_search_receipt(
    *,
    treatment_rows: list[dict[str, Any]],
    sham_rows: list[dict[str, Any]],
    baseline_score: float,
) -> dict[str, Any]:
    gains = list(VERIFIER_GAIN_GRID)
    treatment = _validate_rows(treatment_rows, arm="treatment", gain_grid=gains)
    sham = _validate_rows(sham_rows, arm="sham", gain_grid=gains)
    if isinstance(baseline_score, bool) or not isinstance(baseline_score, (int, float)):
        raise ValueError("verifier gain-search baseline is invalid")
    if not math.isfinite(float(baseline_score)):
        raise ValueError("verifier gain-search baseline is non-finite")
    compute_matched = all(
        left["probe_token_count"] == right["probe_token_count"]
        and left["layer_apps"] == right["layer_apps"]
        for left, right in zip(treatment, sham, strict=True)
    )
    treatment_selected = _selected(treatment)
    sham_selected = _selected(sham)
    payload = {
        "schema": VERIFIER_GAIN_SEARCH_SCHEMA,
        "gain_grid": gains,
        "selection_policy": "highest_public_verifier_score_grid_order_tiebreak",
        "teacher_removed_from_probe_context": True,
        "capability_claim_authority": False,
        "baseline_score": float(baseline_score),
        "evaluations_per_arm": len(gains),
        "compute_matched": compute_matched,
        "treatment": treatment,
        "sham": sham,
        "selected_treatment_gain": float(treatment_selected["gain"]),
        "selected_treatment_score": float(treatment_selected["score"]),
        "selected_treatment_tokens_sha256": treatment_selected[
            "probe_tokens_sha256"
        ],
        "selected_sham_gain": float(sham_selected["gain"]),
        "selected_sham_score": float(sham_selected["score"]),
        "selected_sham_tokens_sha256": sham_selected["probe_tokens_sha256"],
    }
    receipt = {**payload, "receipt_sha256": _sha(payload)}
    return validate_verifier_gain_search_receipt(receipt)


def validate_verifier_gain_search_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("verifier gain-search receipt is missing")
    fields = {
        "schema",
        "gain_grid",
        "selection_policy",
        "teacher_removed_from_probe_context",
        "capability_claim_authority",
        "baseline_score",
        "evaluations_per_arm",
        "compute_matched",
        "treatment",
        "sham",
        "selected_treatment_gain",
        "selected_treatment_score",
        "selected_treatment_tokens_sha256",
        "selected_sham_gain",
        "selected_sham_score",
        "selected_sham_tokens_sha256",
        "receipt_sha256",
    }
    if set(value) != fields:
        raise ValueError("verifier gain-search fields differ")
    payload = {field: value[field] for field in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != _sha(payload):
        raise ValueError("verifier gain-search commitment differs")
    if (
        isinstance(value["baseline_score"], bool)
        or not isinstance(value["baseline_score"], (int, float))
        or not math.isfinite(float(value["baseline_score"]))
        or type(value["evaluations_per_arm"]) is not int
        or type(value["compute_matched"]) is not bool
        or type(value["teacher_removed_from_probe_context"]) is not bool
        or type(value["capability_claim_authority"]) is not bool
        or not _is_sha(value["selected_treatment_tokens_sha256"])
        or not _is_sha(value["selected_sham_tokens_sha256"])
    ):
        raise ValueError("verifier gain-search scalar fields are invalid")
    gains = list(VERIFIER_GAIN_GRID)
    treatment = _validate_rows(value["treatment"], arm="treatment", gain_grid=gains)
    sham = _validate_rows(value["sham"], arm="sham", gain_grid=gains)
    selected_treatment = _selected(treatment)
    selected_sham = _selected(sham)
    compute_matched = all(
        left["probe_token_count"] == right["probe_token_count"]
        and left["layer_apps"] == right["layer_apps"]
        for left, right in zip(treatment, sham, strict=True)
    )
    if (
        value["schema"] != VERIFIER_GAIN_SEARCH_SCHEMA
        or value["gain_grid"] != gains
        or value["selection_policy"]
        != "highest_public_verifier_score_grid_order_tiebreak"
        or value["teacher_removed_from_probe_context"] is not True
        or value["capability_claim_authority"] is not False
        or value["evaluations_per_arm"] != len(gains)
        or value["compute_matched"] is not compute_matched
        or compute_matched is not True
        or float(value["selected_treatment_gain"])
        != float(selected_treatment["gain"])
        or float(value["selected_treatment_score"])
        != float(selected_treatment["score"])
        or value["selected_treatment_tokens_sha256"]
        != selected_treatment["probe_tokens_sha256"]
        or float(value["selected_sham_gain"]) != float(selected_sham["gain"])
        or float(value["selected_sham_score"]) != float(selected_sham["score"])
        or value["selected_sham_tokens_sha256"]
        != selected_sham["probe_tokens_sha256"]
    ):
        raise ValueError("verifier gain-search verdict does not reconstruct")
    return dict(value)


__all__ = [
    "VERIFIER_GAIN_GRID",
    "VERIFIER_GAIN_SEARCH_SCHEMA",
    "build_verifier_gain_search_receipt",
    "validate_verifier_gain_search_receipt",
]
