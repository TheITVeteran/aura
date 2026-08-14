"""Pure evidence contract for a query-scoped neural output readout."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

NEURAL_READOUT_SCHEMA = "aura.rlc.episodic_neural_readout.v1"
NEURAL_READOUT_EXPERIMENT_SCHEMA = "aura.rlc.neural_readout_experiment.v1"
NEURAL_READOUT_GAIN_GRID = (0.0, 0.5, 1.0, 2.0)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema",
        "erased",
        "keys_sha256",
        "weights_sha256",
        "targets_sha256",
        "sample_count",
        "hidden_width",
        "token_count",
        "effective_rank",
        "ridge",
        "margin",
        "gain",
        "applications",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("neural-readout identity fields do not match schema")
    if (
        value["schema"] != NEURAL_READOUT_SCHEMA
        or value["erased"] is not False
        or any(
            not _is_sha256(value[name])
            for name in ("keys_sha256", "weights_sha256", "targets_sha256")
        )
        or type(value["sample_count"]) is not int
        or value["sample_count"] <= 0
        or type(value["hidden_width"]) is not int
        or value["hidden_width"] <= 0
        or type(value["token_count"]) is not int
        or value["token_count"] <= 0
        or type(value["effective_rank"]) is not int
        or not 1 <= value["effective_rank"] <= min(value["sample_count"], value["token_count"])
        or not math.isfinite(float(value["ridge"]))
        or not 1e-8 <= float(value["ridge"]) <= 1e4
        or not math.isfinite(float(value["margin"]))
        or not 0.0 < float(value["margin"]) <= 64.0
        or float(value["gain"]) != 0.0
        or value["applications"] != 0
    ):
        raise ValueError("neural-readout identity is invalid")
    return dict(value)


def build_neural_readout_experiment_receipt(
    *,
    treatment_identity: Mapping[str, Any],
    sham_identity: Mapping[str, Any],
    treatment_rows: Sequence[Mapping[str, Any]],
    sham_rows: Sequence[Mapping[str, Any]],
    erase_proven: bool,
) -> dict[str, Any]:
    """Admit only exact teacher-free treatment replay over a matched sham."""

    treatment_identity = _validate_identity(treatment_identity)
    sham_identity = _validate_identity(sham_identity)
    if (
        treatment_identity["targets_sha256"] == sham_identity["targets_sha256"]
        or treatment_identity["sample_count"] != sham_identity["sample_count"]
        or treatment_identity["hidden_width"] != sham_identity["hidden_width"]
        or treatment_identity["effective_rank"] != sham_identity["effective_rank"]
        or treatment_identity["ridge"] != sham_identity["ridge"]
        or treatment_identity["margin"] != sham_identity["margin"]
    ):
        raise ValueError("neural-readout treatment and sham are not matched")

    def normalize(arm: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        normalized = [dict(row) for row in rows]
        if [row.get("gain") for row in normalized] != list(NEURAL_READOUT_GAIN_GRID):
            raise ValueError("neural-readout gain inventory differs")
        fields = {
            "arm",
            "gain",
            "score",
            "probe_tokens_sha256",
            "probe_token_count",
            "target_replayed_exactly",
            "task_verified",
            "applications",
        }
        for row in normalized:
            if (
                set(row) != fields
                or row["arm"] != arm
                or not math.isfinite(float(row["score"]))
                or not 0.0 <= float(row["score"]) <= 1.0
                or not _is_sha256(row["probe_tokens_sha256"])
                or type(row["probe_token_count"]) is not int
                or row["probe_token_count"] < 0
                or type(row["target_replayed_exactly"]) is not bool
                or type(row["task_verified"]) is not bool
                or type(row["applications"]) is not int
                or row["applications"] < 0
            ):
                raise ValueError("neural-readout experiment row is invalid")
        return normalized

    treatment = normalize("treatment", treatment_rows)
    sham = normalize("sham", sham_rows)
    selected_treatment = max(
        treatment,
        key=lambda row: (
            row["task_verified"],
            row["target_replayed_exactly"],
            float(row["score"]),
            -float(row["gain"]),
        ),
    )
    selected_sham = max(
        sham,
        key=lambda row: (
            row["task_verified"],
            row["target_replayed_exactly"],
            float(row["score"]),
            -float(row["gain"]),
        ),
    )
    accepted = bool(
        float(selected_treatment["gain"]) > 0.0
        and selected_treatment["target_replayed_exactly"] is True
        and selected_treatment["task_verified"] is True
        and selected_sham["task_verified"] is False
        and erase_proven is True
    )
    payload = {
        "schema": NEURAL_READOUT_EXPERIMENT_SCHEMA,
        "gain_grid": list(NEURAL_READOUT_GAIN_GRID),
        "treatment_identity": treatment_identity,
        "sham_identity": sham_identity,
        "treatment": treatment,
        "sham": sham,
        "selected_treatment_gain": float(selected_treatment["gain"]),
        "selected_treatment_score": float(selected_treatment["score"]),
        "selected_sham_gain": float(selected_sham["gain"]),
        "selected_sham_score": float(selected_sham["score"]),
        "teacher_removed_from_probe_context": True,
        "matched_control": True,
        "erase_proven": bool(erase_proven),
        "accepted": accepted,
        "capability_claim_authority": False,
        "claim_boundary": "single_query_verified_correction_internalization_only",
    }
    return {**payload, "receipt_sha256": _canonical_sha256(payload)}


def validate_neural_readout_experiment_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("neural-readout experiment receipt must be a mapping")
    expected = {
        "schema",
        "gain_grid",
        "treatment_identity",
        "sham_identity",
        "treatment",
        "sham",
        "selected_treatment_gain",
        "selected_treatment_score",
        "selected_sham_gain",
        "selected_sham_score",
        "teacher_removed_from_probe_context",
        "matched_control",
        "erase_proven",
        "accepted",
        "capability_claim_authority",
        "claim_boundary",
        "receipt_sha256",
    }
    if set(value) != expected:
        raise ValueError("neural-readout experiment fields do not match schema")
    reconstructed = build_neural_readout_experiment_receipt(
        treatment_identity=value["treatment_identity"],
        sham_identity=value["sham_identity"],
        treatment_rows=value["treatment"],
        sham_rows=value["sham"],
        erase_proven=value["erase_proven"],
    )
    if dict(value) != reconstructed:
        raise ValueError("neural-readout experiment receipt does not reconstruct")
    return dict(value)


__all__ = [
    "NEURAL_READOUT_EXPERIMENT_SCHEMA",
    "NEURAL_READOUT_GAIN_GRID",
    "NEURAL_READOUT_SCHEMA",
    "build_neural_readout_experiment_receipt",
    "validate_neural_readout_experiment_receipt",
]
