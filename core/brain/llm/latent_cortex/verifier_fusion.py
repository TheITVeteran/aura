"""Correlation-aware fusion of historically calibrated verifier evidence.

The mesh deliberately separates measurement from authority.  Individual
verifiers retain their existing narrow contracts; this module normalizes their
signals, calibrates them against independently checked outcomes, discounts
shared errors, and emits a diagnostic receipt.  It never selects a branch,
certifies correctness, or lets one probabilistic source stand in for consensus.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

CHECKED_OUTCOME_SCHEMA = "aura.rlc.checked_verifier_outcome.v1"
VERIFIER_EVIDENCE_SCHEMA = "aura.rlc.verifier_fusion_evidence.v1"
VERIFIER_FUSION_SCHEMA = "aura.rlc.verifier_fusion.v1"

VERIFIER_IDS = (
    "blind_task_verifier",
    "generative_refutation",
    "counterfactual_robustness",
    "prefix_recurrence",
    "neural_uncertainty",
    "process_verifier",
)
MIN_RELIABILITY_OUTCOMES = 12
MIN_CALIBRATION_BIN_OUTCOMES = 8
MIN_EFFECTIVE_SOURCES = 1.5
_SHRINKAGE_PRIOR = 24.0
_MAX_OUTCOME_ROWS = 5000
_Z_95 = 1.959963984540054
_BIN_COUNT = 10


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _probability(value: Any, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field} must be a finite probability")
    return float(value)


def _wilson(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    z2 = _Z_95 * _Z_95
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    radius = (
        _Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return round(max(0.0, center - radius), 10), round(min(1.0, center + radius), 10)


def _phi(n11: int, n10: int, n01: int, n00: int) -> float:
    numerator = n11 * n00 - n10 * n01
    denominator = math.sqrt(
        (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    )
    if denominator <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, numerator / denominator))


def _pair_key(left: str, right: str) -> str:
    return "|".join(sorted((left, right)))


def _bin_index(probability: float) -> int:
    return min(_BIN_COUNT - 1, int(probability * _BIN_COUNT))


def _validate_checked_outcome(value: Any) -> dict[str, Any]:
    fields = {
        "schema",
        "bucket",
        "task_sha256",
        "grade_receipt_sha256",
        "checked",
        "outcome_correct",
        "signals",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema") != CHECKED_OUTCOME_SCHEMA
        or value.get("checked") is not True
        or type(value.get("outcome_correct")) is not bool
    ):
        raise ValueError("checked verifier outcome fields are invalid")
    bucket = value["bucket"]
    if not isinstance(bucket, str) or not bucket or len(bucket) > 160:
        raise ValueError("checked verifier outcome bucket is invalid")
    if not _is_sha256(value["task_sha256"]):
        raise ValueError("checked verifier task commitment is invalid")
    if not _is_sha256(value["grade_receipt_sha256"]):
        raise ValueError("checked verifier grade commitment is invalid")
    signals = value["signals"]
    if (
        not isinstance(signals, Mapping)
        or not signals
        or not set(signals).issubset(VERIFIER_IDS)
    ):
        raise ValueError("checked verifier signal inventory is invalid")
    normalized_signals: dict[str, dict[str, Any]] = {}
    for verifier_id in sorted(signals):
        signal = signals[verifier_id]
        if not isinstance(signal, Mapping) or set(signal) != {
            "probability_correct",
            "source_receipt_sha256",
        }:
            raise ValueError("checked verifier signal fields are invalid")
        normalized_signals[verifier_id] = {
            "probability_correct": round(
                _probability(
                    signal["probability_correct"],
                    field="checked verifier probability",
                ),
                10,
            ),
            "source_receipt_sha256": signal["source_receipt_sha256"],
        }
        if not _is_sha256(normalized_signals[verifier_id]["source_receipt_sha256"]):
            raise ValueError("checked verifier source commitment is invalid")
    return {
        "schema": CHECKED_OUTCOME_SCHEMA,
        "bucket": bucket,
        "task_sha256": value["task_sha256"],
        "grade_receipt_sha256": value["grade_receipt_sha256"],
        "checked": True,
        "outcome_correct": value["outcome_correct"],
        "signals": normalized_signals,
    }


def _empty_bin(index: int) -> dict[str, Any]:
    return {
        "index": index,
        "lower": round(index / _BIN_COUNT, 10),
        "upper": round((index + 1) / _BIN_COUNT, 10),
        "upper_inclusive": index == _BIN_COUNT - 1,
        "n": 0,
        "positives": 0,
        "probability_sum": 0.0,
        "squared_error_sum": 0.0,
        "mean_probability": None,
        "empirical_probability": None,
        "brier_score": None,
        "wilson_95": {"lower": None, "upper": None},
        "calibration_admitted": False,
    }


def _verifier_statistics(
    verifier_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    observations = [
        (
            float(row["signals"][verifier_id]["probability_correct"]),
            bool(row["outcome_correct"]),
        )
        for row in rows
        if verifier_id in row["signals"]
    ]
    bins = [_empty_bin(index) for index in range(_BIN_COUNT)]
    directional_correct = 0
    for probability, outcome in observations:
        predicted = probability >= 0.5
        directional_correct += predicted is outcome
        cell = bins[_bin_index(probability)]
        cell["n"] += 1
        cell["positives"] += int(outcome)
        cell["probability_sum"] += probability
        cell["squared_error_sum"] += (probability - float(outcome)) ** 2
    for cell in bins:
        n = int(cell["n"])
        cell["probability_sum"] = round(float(cell["probability_sum"]), 10)
        cell["squared_error_sum"] = round(float(cell["squared_error_sum"]), 10)
        if n:
            cell["mean_probability"] = round(cell["probability_sum"] / n, 10)
            cell["empirical_probability"] = round(cell["positives"] / n, 10)
            cell["brier_score"] = round(cell["squared_error_sum"] / n, 10)
            lower, upper = _wilson(cell["positives"], n)
            cell["wilson_95"] = {"lower": lower, "upper": upper}
            cell["calibration_admitted"] = n >= MIN_CALIBRATION_BIN_OUTCOMES
    n = len(observations)
    reliability_lower, reliability_upper = _wilson(directional_correct, n)
    brier = (
        round(sum(float(cell["squared_error_sum"]) for cell in bins) / n, 10)
        if n
        else None
    )
    ece = (
        round(
            sum(
                int(cell["n"])
                * abs(
                    float(cell["empirical_probability"])
                    - float(cell["mean_probability"])
                )
                for cell in bins
                if cell["n"]
            )
            / n,
            10,
        )
        if n
        else None
    )
    return {
        "verifier_id": verifier_id,
        "n": n,
        "directionally_correct": directional_correct,
        "directional_accuracy": round(directional_correct / n, 10) if n else None,
        "wilson_95": {
            "lower": reliability_lower,
            "upper": reliability_upper,
        },
        "brier_score": brier,
        "expected_calibration_error": ece,
        "calibration_bins": bins,
        "reliability_admitted": n >= MIN_RELIABILITY_OUTCOMES,
    }


def _pair_statistics(
    left: str,
    right: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    n11 = n10 = n01 = n00 = 0
    for row in rows:
        if left not in row["signals"] or right not in row["signals"]:
            continue
        outcome = bool(row["outcome_correct"])
        left_error = (
            float(row["signals"][left]["probability_correct"]) >= 0.5
        ) is not outcome
        right_error = (
            float(row["signals"][right]["probability_correct"]) >= 0.5
        ) is not outcome
        if left_error and right_error:
            n11 += 1
        elif left_error:
            n10 += 1
        elif right_error:
            n01 += 1
        else:
            n00 += 1
    n = n11 + n10 + n01 + n00
    raw_phi = _phi(n11, n10, n01, n00)
    enough = n >= MIN_RELIABILITY_OUTCOMES
    positive_shrunk = (
        max(0.0, raw_phi) * n / (n + _SHRINKAGE_PRIOR) if enough else 0.0
    )
    return {
        "pair": _pair_key(left, right),
        "left": left,
        "right": right,
        "n": n,
        "error_table": {
            "both": n11,
            "left_only": n10,
            "right_only": n01,
            "neither": n00,
        },
        "phi": round(raw_phi, 10),
        "positive_shrunk_dependence": round(positive_shrunk, 10),
        "dependence_admitted": enough,
    }


def _scope_summary(
    *,
    scope: str,
    bucket: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "scope": scope,
        "bucket": bucket,
        "checked_tasks": len(rows),
        "verifiers": [
            _verifier_statistics(verifier_id, rows) for verifier_id in VERIFIER_IDS
        ],
        "pairwise_dependence": [
            _pair_statistics(left, right, rows)
            for left, right in combinations(VERIFIER_IDS, 2)
        ],
    }


def build_verifier_fusion_evidence(
    *,
    bucket: str,
    checked_outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate checked outcomes into domain/global calibration evidence."""

    if not isinstance(bucket, str) or not bucket or len(bucket) > 160:
        raise ValueError("verifier fusion bucket is invalid")
    if not isinstance(checked_outcomes, list) or len(checked_outcomes) > _MAX_OUTCOME_ROWS:
        raise ValueError("checked verifier outcome inventory is invalid")
    rows = [_validate_checked_outcome(row) for row in checked_outcomes]
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["bucket"], row["task_sha256"])
        if key in seen:
            raise ValueError("duplicate checked verifier outcome")
        seen.add(key)
    rows.sort(key=lambda row: (row["bucket"], row["task_sha256"]))
    domain_rows = [row for row in rows if row["bucket"] == bucket]
    domain = _scope_summary(scope="domain", bucket=bucket, rows=domain_rows)
    global_scope = _scope_summary(scope="global", bucket="*", rows=rows)
    domain_measured = any(
        row["reliability_admitted"] for row in domain["verifiers"]
    )
    global_measured = any(
        row["reliability_admitted"] for row in global_scope["verifiers"]
    )
    payload = {
        "schema": VERIFIER_EVIDENCE_SCHEMA,
        "bucket": bucket,
        "verifier_ids": list(VERIFIER_IDS),
        "minimum_reliability_outcomes": MIN_RELIABILITY_OUTCOMES,
        "minimum_calibration_bin_outcomes": MIN_CALIBRATION_BIN_OUTCOMES,
        "shrinkage_prior": _SHRINKAGE_PRIOR,
        "checked_outcomes_sha256": _sha(rows),
        "checked_tasks_total": len(rows),
        "scopes": {"domain": domain, "global": global_scope},
        "evidence_state": (
            "domain_measured"
            if domain_measured
            else "global_measured"
            if global_measured
            else "bootstrap_unmeasured"
        ),
    }
    return {**payload, "snapshot_sha256": _sha(payload)}


def _validate_bin(cell: Any, *, index: int) -> None:
    expected_fields = set(_empty_bin(index))
    if not isinstance(cell, Mapping) or set(cell) != expected_fields:
        raise ValueError("verifier calibration-bin fields differ")
    n = cell["n"]
    positives = cell["positives"]
    probability_sum = cell["probability_sum"]
    squared_error_sum = cell["squared_error_sum"]
    if (
        cell["index"] != index
        or cell["lower"] != round(index / _BIN_COUNT, 10)
        or cell["upper"] != round((index + 1) / _BIN_COUNT, 10)
        or cell["upper_inclusive"] is not (index == _BIN_COUNT - 1)
        or type(n) is not int
        or n < 0
        or type(positives) is not int
        or not 0 <= positives <= n
    ):
        raise ValueError("verifier calibration-bin identity is invalid")
    probability_sum = _probability_sum(probability_sum, n, "probability sum")
    squared_error_sum = _probability_sum(
        squared_error_sum, n, "squared-error sum"
    )
    if n == 0:
        if dict(cell) != _empty_bin(index):
            raise ValueError("empty verifier calibration bin is not canonical")
        return
    mean = round(probability_sum / n, 10)
    empirical = round(positives / n, 10)
    brier = round(squared_error_sum / n, 10)
    lower, upper = _wilson(positives, n)
    if (
        cell["mean_probability"] != mean
        or cell["empirical_probability"] != empirical
        or cell["brier_score"] != brier
        or cell["wilson_95"] != {"lower": lower, "upper": upper}
        or cell["calibration_admitted"]
        is not (n >= MIN_CALIBRATION_BIN_OUTCOMES)
    ):
        raise ValueError("verifier calibration-bin statistics differ")


def _probability_sum(value: Any, n: int, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= float(n)
    ):
        raise ValueError(f"verifier {field} is invalid")
    return float(value)


def _validate_scope(value: Any, *, scope: str, bucket: str) -> None:
    fields = {
        "scope",
        "bucket",
        "checked_tasks",
        "verifiers",
        "pairwise_dependence",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value["scope"] != scope
        or value["bucket"] != bucket
        or type(value["checked_tasks"]) is not int
        or value["checked_tasks"] < 0
    ):
        raise ValueError("verifier evidence scope identity is invalid")
    verifiers = value["verifiers"]
    if (
        not isinstance(verifiers, list)
        or [row.get("verifier_id") for row in verifiers if isinstance(row, Mapping)]
        != list(VERIFIER_IDS)
    ):
        raise ValueError("verifier evidence coverage differs")
    verifier_fields = {
        "verifier_id",
        "n",
        "directionally_correct",
        "directional_accuracy",
        "wilson_95",
        "brier_score",
        "expected_calibration_error",
        "calibration_bins",
        "reliability_admitted",
    }
    for row in verifiers:
        if not isinstance(row, Mapping) or set(row) != verifier_fields:
            raise ValueError("verifier reliability fields differ")
        n = row["n"]
        correct = row["directionally_correct"]
        bins = row["calibration_bins"]
        if (
            type(n) is not int
            or not 0 <= n <= value["checked_tasks"]
            or type(correct) is not int
            or not 0 <= correct <= n
            or not isinstance(bins, list)
            or len(bins) != _BIN_COUNT
        ):
            raise ValueError("verifier reliability counts are invalid")
        for index, cell in enumerate(bins):
            _validate_bin(cell, index=index)
        bin_n = sum(int(cell["n"]) for cell in bins)
        directional_from_bins = sum(
            (
                int(cell["positives"])
                if index >= _BIN_COUNT // 2
                else int(cell["n"]) - int(cell["positives"])
            )
            for index, cell in enumerate(bins)
        )
        lower, upper = _wilson(correct, n)
        expected_brier = (
            round(
                sum(float(cell["squared_error_sum"]) for cell in bins) / n,
                10,
            )
            if n
            else None
        )
        expected_ece = (
            round(
                sum(
                    int(cell["n"])
                    * abs(
                        float(cell["empirical_probability"])
                        - float(cell["mean_probability"])
                    )
                    for cell in bins
                    if cell["n"]
                )
                / n,
                10,
            )
            if n
            else None
        )
        if (
            bin_n != n
            or directional_from_bins != correct
            or row["directional_accuracy"]
            != (round(correct / n, 10) if n else None)
            or row["wilson_95"] != {"lower": lower, "upper": upper}
            or row["brier_score"] != expected_brier
            or row["expected_calibration_error"] != expected_ece
            or row["reliability_admitted"]
            is not (n >= MIN_RELIABILITY_OUTCOMES)
        ):
            raise ValueError("verifier reliability statistics differ")
    pairs = value["pairwise_dependence"]
    expected_pairs = {
        _pair_key(left, right) for left, right in combinations(VERIFIER_IDS, 2)
    }
    if (
        not isinstance(pairs, list)
        or {row.get("pair") for row in pairs if isinstance(row, Mapping)}
        != expected_pairs
    ):
        raise ValueError("verifier dependence coverage differs")
    pair_fields = {
        "pair",
        "left",
        "right",
        "n",
        "error_table",
        "phi",
        "positive_shrunk_dependence",
        "dependence_admitted",
    }
    for row in pairs:
        if not isinstance(row, Mapping) or set(row) != pair_fields:
            raise ValueError("verifier dependence fields differ")
        expected_order = {
            _pair_key(left, right): (left, right)
            for left, right in combinations(VERIFIER_IDS, 2)
        }
        if (
            row["pair"] != _pair_key(row["left"], row["right"])
            or (row["left"], row["right"]) != expected_order.get(row["pair"])
            or row["left"] not in VERIFIER_IDS
            or row["right"] not in VERIFIER_IDS
            or row["left"] == row["right"]
        ):
            raise ValueError("verifier dependence identity is invalid")
        table = row["error_table"]
        if (
            not isinstance(table, Mapping)
            or set(table) != {"both", "left_only", "right_only", "neither"}
            or any(type(count) is not int or count < 0 for count in table.values())
        ):
            raise ValueError("verifier dependence table is invalid")
        n = sum(int(count) for count in table.values())
        phi = round(
            _phi(
                table["both"],
                table["left_only"],
                table["right_only"],
                table["neither"],
            ),
            10,
        )
        admitted = n >= MIN_RELIABILITY_OUTCOMES
        shrunk = round(
            max(0.0, phi) * n / (n + _SHRINKAGE_PRIOR)
            if admitted
            else 0.0,
            10,
        )
        if (
            row["n"] != n
            or n > value["checked_tasks"]
            or row["phi"] != phi
            or row["positive_shrunk_dependence"] != shrunk
            or row["dependence_admitted"] is not admitted
        ):
            raise ValueError("verifier dependence statistics differ")


def validate_verifier_fusion_evidence(value: Any) -> dict[str, Any]:
    if value is None:
        return build_verifier_fusion_evidence(
            bucket="runtime|unmeasured",
            checked_outcomes=[],
        )
    fields = {
        "schema",
        "bucket",
        "verifier_ids",
        "minimum_reliability_outcomes",
        "minimum_calibration_bin_outcomes",
        "shrinkage_prior",
        "checked_outcomes_sha256",
        "checked_tasks_total",
        "scopes",
        "evidence_state",
        "snapshot_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("verifier fusion evidence fields differ")
    payload = {key: value[key] for key in fields - {"snapshot_sha256"}}
    bucket = value["bucket"]
    if (
        value["schema"] != VERIFIER_EVIDENCE_SCHEMA
        or not isinstance(bucket, str)
        or not bucket
        or len(bucket) > 160
        or value["verifier_ids"] != list(VERIFIER_IDS)
        or value["minimum_reliability_outcomes"] != MIN_RELIABILITY_OUTCOMES
        or value["minimum_calibration_bin_outcomes"]
        != MIN_CALIBRATION_BIN_OUTCOMES
        or value["shrinkage_prior"] != _SHRINKAGE_PRIOR
        or not _is_sha256(value["checked_outcomes_sha256"])
        or type(value["checked_tasks_total"]) is not int
        or value["checked_tasks_total"] < 0
        or value["snapshot_sha256"] != _sha(payload)
    ):
        raise ValueError("verifier fusion evidence identity is invalid")
    scopes = value["scopes"]
    if not isinstance(scopes, Mapping) or set(scopes) != {"domain", "global"}:
        raise ValueError("verifier fusion evidence scopes differ")
    _validate_scope(scopes["domain"], scope="domain", bucket=bucket)
    _validate_scope(scopes["global"], scope="global", bucket="*")
    if (
        scopes["global"]["checked_tasks"] != value["checked_tasks_total"]
        or scopes["domain"]["checked_tasks"] > value["checked_tasks_total"]
    ):
        raise ValueError("verifier fusion checked-task totals differ")
    domain_measured = any(
        row["reliability_admitted"] for row in scopes["domain"]["verifiers"]
    )
    global_measured = any(
        row["reliability_admitted"] for row in scopes["global"]["verifiers"]
    )
    state = (
        "domain_measured"
        if domain_measured
        else "global_measured"
        if global_measured
        else "bootstrap_unmeasured"
    )
    if value["evidence_state"] != state:
        raise ValueError("verifier fusion evidence state differs")
    return dict(value)


def _receipt_hash(receipt: Any) -> str:
    if isinstance(receipt, Mapping) and _is_sha256(receipt.get("receipt_sha256")):
        return str(receipt["receipt_sha256"])
    return ""


def _raw_signals(
    *,
    blind_review: Any,
    decoy_verification: Any,
    generative_verifier: Any,
    counterfactual_verifier: Any,
    prefix_stability: Any,
    neural_uncertainty: Any,
    mistake_locator: Any,
    selected_branch: int,
) -> list[dict[str, Any]]:
    raw: dict[str, tuple[float | None, str, str]] = {
        verifier_id: (None, "", "source_unavailable") for verifier_id in VERIFIER_IDS
    }
    if (
        isinstance(blind_review, Mapping)
        and blind_review.get("schema") == "aura.rlc.blind_branch_review.v1"
        and isinstance(decoy_verification, Mapping)
        and decoy_verification.get("selection_admitted") is True
    ):
        row = next(
            (
                item
                for item in blind_review.get("rows", [])
                if isinstance(item, Mapping) and item.get("branch") == selected_branch
            ),
            None,
        )
        if isinstance(row, Mapping):
            raw["blind_task_verifier"] = (
                _probability(row.get("score"), field="blind verifier score"),
                _receipt_hash(blind_review),
                "",
            )
    if (
        isinstance(generative_verifier, Mapping)
        and generative_verifier.get("schema")
        == "aura.rlc.generative_verifier.v1"
    ):
        raw["generative_refutation"] = (
            None,
            _receipt_hash(generative_verifier),
            (
                "refutation_applied_to_prior_selection"
                if generative_verifier.get("causal_refutation") is True
                else "absence_of_refutation_is_not_positive_evidence"
            ),
        )
    if (
        isinstance(counterfactual_verifier, Mapping)
        and counterfactual_verifier.get("schema")
        == "aura.rlc.counterfactual_verifier.v1"
        and counterfactual_verifier.get("selection_authority_admitted") is True
        and counterfactual_verifier.get("selected_branch") == selected_branch
    ):
        row = next(
            (
                item
                for item in counterfactual_verifier.get("branches", [])
                if isinstance(item, Mapping) and item.get("branch") == selected_branch
            ),
            None,
        )
        if isinstance(row, Mapping) and row.get("robustness_score") is not None:
            raw["counterfactual_robustness"] = (
                _probability(
                    row["robustness_score"],
                    field="counterfactual robustness",
                ),
                _receipt_hash(counterfactual_verifier),
                "",
            )
    if (
        isinstance(prefix_stability, Mapping)
        and prefix_stability.get("schema")
        == "aura.rlc.prefix_stability_verifier.v1"
        and prefix_stability.get("measurement_admitted") is True
    ):
        metrics = prefix_stability.get("metrics")
        if isinstance(metrics, Mapping) and metrics.get("raw_stability") is not None:
            raw["prefix_recurrence"] = (
                _probability(
                    metrics["raw_stability"],
                    field="prefix recurrence",
                ),
                _receipt_hash(prefix_stability),
                "",
            )
    if (
        isinstance(neural_uncertainty, Mapping)
        and neural_uncertainty.get("schema")
        == "aura.rlc.neural_uncertainty_receipt.v1"
        and neural_uncertainty.get("selected_branch") == selected_branch
    ):
        scores = neural_uncertainty.get("latest_supported_scores")
        score = scores.get(str(selected_branch)) if isinstance(scores, Mapping) else None
        if score is not None:
            raw["neural_uncertainty"] = (
                _probability(score, field="neural uncertainty"),
                _receipt_hash(neural_uncertainty),
                "",
            )
    if (
        isinstance(mistake_locator, Mapping)
        and mistake_locator.get("schema") == "aura.rlc.mistake_locator_receipt.v2"
        and mistake_locator.get("selected_branch") == selected_branch
    ):
        row = next(
            (
                item
                for item in mistake_locator.get("branches", [])
                if isinstance(item, Mapping)
                and item.get("branch_index") == selected_branch
            ),
            None,
        )
        process = row.get("process") if isinstance(row, Mapping) else None
        if (
            isinstance(process, Mapping)
            and process.get("selection_authority_admitted") is True
            and process.get("process_score") is not None
        ):
            raw["process_verifier"] = (
                _probability(process["process_score"], field="process score"),
                _receipt_hash(mistake_locator),
                "",
            )
    targets = {
        "blind_task_verifier": "candidate_quality",
        "generative_refutation": "deterministic_candidate_refutation",
        "counterfactual_robustness": "counterfactual_robustness",
        "prefix_recurrence": "conclusion_recurrence",
        "neural_uncertainty": "hidden_state_correctness",
        "process_verifier": "accepted_transition_integrity",
    }
    return [
        {
            "verifier_id": verifier_id,
            "semantic_target": targets[verifier_id],
            "raw_probability_correct": raw[verifier_id][0],
            "source_receipt_sha256": raw[verifier_id][1],
            "observation_available": raw[verifier_id][0] is not None,
            "observation_reason": raw[verifier_id][2],
        }
        for verifier_id in VERIFIER_IDS
    ]


def checked_signals_from_receipt(receipt: Any) -> dict[str, dict[str, Any]]:
    """Extract gradeable final-selection signals from a verified RLC receipt.

    Generative refutation is intentionally absent: it evaluates the vetoed
    provisional candidate, not the final selected branch. A benchmark that
    independently grades that prior candidate may record it explicitly.
    """

    if not isinstance(receipt, Mapping):
        raise ValueError("checked verifier receipt must be a mapping")
    selected_branch = receipt.get("selected_branch")
    if type(selected_branch) is not int or selected_branch < 0:
        raise ValueError("checked verifier receipt selected branch is invalid")
    rows = _raw_signals(
        blind_review=receipt.get("blind_review"),
        decoy_verification=receipt.get("decoy_verification"),
        generative_verifier=receipt.get("generative_verifier"),
        counterfactual_verifier=receipt.get("counterfactual_verifier"),
        prefix_stability=receipt.get("prefix_stability"),
        neural_uncertainty=receipt.get("neural_uncertainty"),
        mistake_locator=receipt.get("mistake_locator"),
        selected_branch=selected_branch,
    )
    signals = {
        row["verifier_id"]: {
            "probability_correct": row["raw_probability_correct"],
            "source_receipt_sha256": row["source_receipt_sha256"],
        }
        for row in rows
        if row["observation_available"] is True
        and _is_sha256(row["source_receipt_sha256"])
    }
    if not signals:
        raise ValueError("verified RLC receipt has no gradeable verifier signals")
    return signals


def _scope_verifier(
    evidence: Mapping[str, Any],
    *,
    scope: str,
    verifier_id: str,
) -> Mapping[str, Any]:
    return next(
        row
        for row in evidence["scopes"][scope]["verifiers"]
        if row["verifier_id"] == verifier_id
    )


def _calibrate_signal(
    signal: dict[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    probability = signal["raw_probability_correct"]
    base = {
        **signal,
        "calibration_scope": "none",
        "calibration_method": "none",
        "historical_n": 0,
        "historical_brier_score": None,
        "historical_expected_calibration_error": None,
        "calibrated_probability_correct": None,
        "confidence_interval_95": {"lower": None, "upper": None},
        "admitted_to_fusion": False,
        "abstention_reason": signal["observation_reason"],
    }
    if probability is None:
        return base
    index = _bin_index(float(probability))
    for scope in ("domain", "global"):
        stats = _scope_verifier(
            evidence,
            scope=scope,
            verifier_id=signal["verifier_id"],
        )
        cell = stats["calibration_bins"][index]
        if cell["calibration_admitted"] is True:
            return {
                **base,
                "calibration_scope": scope,
                "calibration_method": "fixed_probability_bin",
                "historical_n": cell["n"],
                "historical_brier_score": cell["brier_score"],
                "historical_expected_calibration_error": abs(
                    float(cell["empirical_probability"])
                    - float(cell["mean_probability"])
                ),
                "calibrated_probability_correct": cell["empirical_probability"],
                "confidence_interval_95": dict(cell["wilson_95"]),
                "admitted_to_fusion": True,
                "abstention_reason": "",
            }
    for scope in ("domain", "global"):
        stats = _scope_verifier(
            evidence,
            scope=scope,
            verifier_id=signal["verifier_id"],
        )
        if stats["reliability_admitted"] is True:
            supports = float(probability) >= 0.5
            accuracy = float(stats["directional_accuracy"])
            lower = float(stats["wilson_95"]["lower"])
            upper = float(stats["wilson_95"]["upper"])
            return {
                **base,
                "calibration_scope": scope,
                "calibration_method": "directional_reliability",
                "historical_n": stats["n"],
                "historical_brier_score": stats["brier_score"],
                "historical_expected_calibration_error": (
                    stats["expected_calibration_error"]
                ),
                "calibrated_probability_correct": round(
                    accuracy if supports else 1.0 - accuracy,
                    10,
                ),
                "confidence_interval_95": {
                    "lower": round(lower if supports else 1.0 - upper, 10),
                    "upper": round(upper if supports else 1.0 - lower, 10),
                },
                "admitted_to_fusion": True,
                "abstention_reason": "",
            }
    return {
        **base,
        "abstention_reason": "historical_calibration_insufficient",
    }


def _dependence(
    evidence: Mapping[str, Any],
    *,
    left: str,
    right: str,
) -> dict[str, Any]:
    pair = _pair_key(left, right)
    for scope in ("domain", "global"):
        row = next(
            item
            for item in evidence["scopes"][scope]["pairwise_dependence"]
            if item["pair"] == pair
        )
        if row["dependence_admitted"] is True:
            return {
                "pair": pair,
                "scope": scope,
                "n": row["n"],
                "dependence": row["positive_shrunk_dependence"],
                "conservative_dependence_upper_bound": row[
                    "positive_shrunk_dependence"
                ],
                "measured": True,
            }
    return {
        "pair": pair,
        "scope": "none",
        "n": 0,
        "dependence": None,
        "conservative_dependence_upper_bound": 1.0,
        "measured": False,
    }


def _capped_weights(raw: dict[str, float]) -> dict[str, float]:
    if len(raw) < 2:
        return {}
    remaining = set(raw)
    result: dict[str, float] = {}
    budget = 1.0
    while remaining:
        total = sum(raw[key] for key in remaining)
        if total <= 0.0:
            share = budget / len(remaining)
            result.update({key: share for key in remaining})
            break
        tentative = {key: budget * raw[key] / total for key in remaining}
        over = [key for key, value in tentative.items() if value > 0.5]
        if not over:
            result.update(tentative)
            break
        for key in over:
            result[key] = 0.5
            budget -= 0.5
            remaining.remove(key)
    return {key: round(value, 10) for key, value in sorted(result.items())}


def build_verifier_fusion_receipt(
    *,
    blind_review: Any,
    decoy_verification: Any,
    generative_verifier: Any,
    counterfactual_verifier: Any,
    prefix_stability: Any,
    neural_uncertainty: Any,
    mistake_locator: Any,
    selected_branch: int,
    evidence: Any,
) -> dict[str, Any]:
    """Fuse current signals without granting selection/correctness authority."""

    if type(selected_branch) is not int or selected_branch < 0:
        raise ValueError("verifier fusion selected branch is invalid")
    checked = validate_verifier_fusion_evidence(evidence)
    signals = [
        _calibrate_signal(signal, checked)
        for signal in _raw_signals(
            blind_review=blind_review,
            decoy_verification=decoy_verification,
            generative_verifier=generative_verifier,
            counterfactual_verifier=counterfactual_verifier,
            prefix_stability=prefix_stability,
            neural_uncertainty=neural_uncertainty,
            mistake_locator=mistake_locator,
            selected_branch=selected_branch,
        )
    ]
    admitted = [row for row in signals if row["admitted_to_fusion"] is True]
    dependencies = [
        _dependence(
            checked,
            left=left["verifier_id"],
            right=right["verifier_id"],
        )
        for left, right in combinations(admitted, 2)
    ]
    dependence_by_pair = {
        row["pair"]: float(row["conservative_dependence_upper_bound"])
        for row in dependencies
    }
    raw_weights: dict[str, float] = {}
    for signal in admitted:
        verifier_id = signal["verifier_id"]
        brier = signal["historical_brier_score"]
        quality = max(0.05, 1.0 - float(brier if brier is not None else 0.5))
        sample_factor = signal["historical_n"] / (
            signal["historical_n"] + _SHRINKAGE_PRIOR
        )
        burden = sum(
            value
            for pair, value in dependence_by_pair.items()
            if verifier_id in pair.split("|")
        )
        raw_weights[verifier_id] = quality * sample_factor / (1.0 + burden)
    weights = _capped_weights(raw_weights)
    n_sources = len(admitted)
    dependence_sum = sum(
        float(row["conservative_dependence_upper_bound"])
        for row in dependencies
    )
    dependence_coverage_complete = all(
        row["measured"] is True for row in dependencies
    )
    effective = (
        n_sources * n_sources / (n_sources + 2.0 * dependence_sum)
        if n_sources
        else 0.0
    )
    measurement_admitted = (
        n_sources >= 2
        and dependence_coverage_complete
        and effective >= MIN_EFFECTIVE_SOURCES
        and len(weights) == n_sources
        and max(weights.values(), default=1.0) <= 0.5
    )
    fused: float | None = None
    interval = {"lower": None, "upper": None}
    verdict = "insufficient_independent_evidence"
    if measurement_admitted:
        fused = round(
            sum(
                weights[row["verifier_id"]]
                * float(row["calibrated_probability_correct"])
                for row in admitted
            ),
            10,
        )
        lower = sum(
            weights[row["verifier_id"]]
            * float(row["confidence_interval_95"]["lower"])
            for row in admitted
        )
        upper = sum(
            weights[row["verifier_id"]]
            * float(row["confidence_interval_95"]["upper"])
            for row in admitted
        )
        inflation = max(0.0, 1.0 - effective / n_sources)
        lower = max(0.0, lower - 0.5 * inflation)
        upper = min(1.0, upper + 0.5 * inflation)
        interval = {"lower": round(lower, 10), "upper": round(upper, 10)}
        verdict = (
            "historically_supported"
            if lower > 0.5
            else "historically_opposed"
            if upper < 0.5
            else "historically_inconclusive"
        )
    payload = {
        "schema": VERIFIER_FUSION_SCHEMA,
        "evidence_snapshot_sha256": checked["snapshot_sha256"],
        "evidence_bucket": checked["bucket"],
        "evidence_state": checked["evidence_state"],
        "selected_branch": selected_branch,
        "signals": signals,
        "probabilistic_sources_observed": sum(
            row["observation_available"] is True for row in signals
        ),
        "probabilistic_sources_admitted": n_sources,
        "pairwise_dependence": dependencies,
        "dependence_coverage_complete": dependence_coverage_complete,
        "source_weights": weights,
        "effective_independent_sources": round(effective, 10),
        "minimum_effective_sources": MIN_EFFECTIVE_SOURCES,
        "single_source_weight_cap": 0.5,
        "fusion_measurement_admitted": measurement_admitted,
        "fused_probability_correct": fused,
        "confidence_interval_95": interval,
        "verdict": verdict,
        "authority_mode": "diagnostic_fusion_no_single_probabilistic_authority",
        "selection_authority_admitted": False,
        "correctness_authority_admitted": False,
        "selection_effect": "none",
        "correctness_effect": "none",
    }
    return {**payload, "receipt_sha256": _sha(payload)}


def validate_verifier_fusion_receipt(
    value: Any,
    *,
    blind_review: Any,
    decoy_verification: Any,
    generative_verifier: Any,
    counterfactual_verifier: Any,
    prefix_stability: Any,
    neural_uncertainty: Any,
    mistake_locator: Any,
    selected_branch: int,
    evidence: Any,
) -> dict[str, Any]:
    expected = build_verifier_fusion_receipt(
        blind_review=blind_review,
        decoy_verification=decoy_verification,
        generative_verifier=generative_verifier,
        counterfactual_verifier=counterfactual_verifier,
        prefix_stability=prefix_stability,
        neural_uncertainty=neural_uncertainty,
        mistake_locator=mistake_locator,
        selected_branch=selected_branch,
        evidence=evidence,
    )
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError("verifier fusion receipt differs from reconstruction")
    return dict(value)


class VerifierFusionLedger:
    """Governed durable source of independently checked verifier outcomes."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            try:
                from core.config import DATA_DIR

                path = (
                    Path(DATA_DIR)
                    / "latent_cortex"
                    / "verifier_fusion"
                    / "checked_outcomes.jsonl"
                )
            except (ImportError, AttributeError, RuntimeError, TypeError):
                path = Path(
                    "data/latent_cortex/verifier_fusion/checked_outcomes.jsonl"
                )
        self.path = Path(path)
        self._rows: list[dict[str, Any]] = []
        self._task_keys: set[tuple[str, str]] = set()
        self.restore_errors = 0
        self._restore()

    def _restore(self) -> None:
        self._rows = []
        self._task_keys = set()
        try:
            if not self.path.exists():
                return
            with open(self.path, encoding="utf-8") as handle:
                for raw in handle:
                    try:
                        row = _validate_checked_outcome(json.loads(raw))
                        key = (row["bucket"], row["task_sha256"])
                        if key in self._task_keys:
                            raise ValueError("duplicate checked verifier task")
                        self._task_keys.add(key)
                        self._rows.append(row)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        self.restore_errors += 1
        except OSError:
            self.restore_errors += 1
        if len(self._rows) > _MAX_OUTCOME_ROWS:
            self._rows = self._rows[-_MAX_OUTCOME_ROWS:]
            self._task_keys = {
                (row["bucket"], row["task_sha256"]) for row in self._rows
            }

    def record_checked(
        self,
        *,
        bucket: str,
        task_sha256: str,
        grade_receipt_sha256: str,
        outcome_correct: bool,
        signals: dict[str, dict[str, Any]],
    ) -> bool:
        row = _validate_checked_outcome(
            {
                "schema": CHECKED_OUTCOME_SCHEMA,
                "bucket": bucket,
                "task_sha256": task_sha256,
                "grade_receipt_sha256": grade_receipt_sha256,
                "checked": True,
                "outcome_correct": outcome_correct,
                "signals": signals,
            }
        )
        key = (row["bucket"], row["task_sha256"])
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.atomic_writer import interprocess_file_lock
            from core.runtime.file_write_gateway import get_file_write_gateway

            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            lock_path = self.path.with_name(f".{self.path.name}.lock")
            with interprocess_file_lock(lock_path):
                self._restore()
                if key in self._task_keys:
                    raise ValueError("checked verifier task is already recorded")
                with local_internal_governed_scope(
                    "latent_verifier_fusion", domain="state_mutation"
                ):
                    gateway = get_file_write_gateway()
                    gateway.append_text(
                        self.path,
                        line,
                        source="latent_verifier_fusion",
                    )
                    rows = [*self._rows, row]
                    if len(rows) > _MAX_OUTCOME_ROWS:
                        rows = rows[-_MAX_OUTCOME_ROWS:]
                        gateway.write_text(
                            self.path,
                            "".join(
                                json.dumps(item, sort_keys=True, separators=(",", ":"))
                                + "\n"
                                for item in rows
                            ),
                            source="latent_verifier_fusion.compact",
                        )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            if key in self._task_keys:
                raise ValueError("checked verifier task is already recorded") from exc
            return False
        self._task_keys.add(key)
        self._rows.append(row)
        if len(self._rows) > _MAX_OUTCOME_ROWS:
            self._rows = self._rows[-_MAX_OUTCOME_ROWS:]
        return True

    def record_receipt_checked(
        self,
        *,
        bucket: str,
        task_sha256: str,
        grade_receipt_sha256: str,
        outcome_correct: bool,
        verified_receipt: dict[str, Any],
    ) -> bool:
        """Record the final-branch signals from a service-verified receipt."""

        return self.record_checked(
            bucket=bucket,
            task_sha256=task_sha256,
            grade_receipt_sha256=grade_receipt_sha256,
            outcome_correct=outcome_correct,
            signals=checked_signals_from_receipt(verified_receipt),
        )

    def evidence(self, *, bucket: str) -> dict[str, Any]:
        self._restore()
        return build_verifier_fusion_evidence(
            bucket=bucket,
            checked_outcomes=list(self._rows),
        )

    def status(self) -> dict[str, Any]:
        self._restore()
        return {
            "path": str(self.path),
            "checked_outcomes": len(self._rows),
            "restore_errors": self.restore_errors,
        }


_LEDGER: VerifierFusionLedger | None = None


def get_verifier_fusion_ledger() -> VerifierFusionLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = VerifierFusionLedger()
    return _LEDGER


__all__ = [
    "CHECKED_OUTCOME_SCHEMA",
    "MIN_CALIBRATION_BIN_OUTCOMES",
    "MIN_EFFECTIVE_SOURCES",
    "MIN_RELIABILITY_OUTCOMES",
    "VERIFIER_EVIDENCE_SCHEMA",
    "VERIFIER_FUSION_SCHEMA",
    "VERIFIER_IDS",
    "VerifierFusionLedger",
    "build_verifier_fusion_evidence",
    "build_verifier_fusion_receipt",
    "checked_signals_from_receipt",
    "get_verifier_fusion_ledger",
    "validate_verifier_fusion_evidence",
    "validate_verifier_fusion_receipt",
]
