"""Recalibrated, query-scoped test-time training controls.

The resident model may only learn from a pseudo-label when a held-out,
machine-checked calibration battery supports the verifier at high confidence.
An admitted update then has to beat both the unchanged function and a
same-compute sham target.  All authority-bearing fields are reconstructible
from public hashes, counts, and scores; candidate prose and latent values stay
inside the worker.

The first admitted verifier family is exact bounded integer arithmetic.  Python
AST and JSON parsing remain useful diagnostics, but syntax validity is not
task correctness and therefore cannot authorize a gradient target.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from core.brain.llm.latent_cortex.atomic_decomposition import (
    build_atomic_decomposition,
)
from core.brain.llm.latent_cortex.deterministic_verifier_router import (
    build_deterministic_router_receipt,
    validate_deterministic_router_envelope,
)

CRITIC_RECALIBRATION_SCHEMA = "aura.rlc.test_time_critic_recalibration.v1"
PSEUDO_LABEL_ADMISSION_SCHEMA = "aura.rlc.test_time_pseudo_label.v1"
MATCHED_COMPUTE_SCHEMA = "aura.rlc.test_time_matched_compute.v1"
TEST_TIME_TRAINING_SCHEMA = "aura.rlc.test_time_training.v1"

MIN_PSEUDO_LABEL_CONFIDENCE = 0.90
MIN_CALIBRATION_PER_CLASS = 48
MAX_CALIBRATION_BRIER = 0.01
MAX_CALIBRATION_ECE = 0.02
MAX_REWARD_DRIFT = 0.02
MATCHED_LINE_SEARCH_EVALUATIONS = 2
_Z95 = 1.959963984540054
_OBJECTIVE = "Check the exact bounded integer arithmetic claim."
_AUTHORIZED_VERIFIERS = frozenset({"exact_integer_arithmetic"})


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _wilson_lower(successes: int, trials: int) -> float:
    if trials <= 0:
        return 0.0
    proportion = successes / trials
    denominator = 1.0 + (_Z95**2 / trials)
    centre = proportion + (_Z95**2 / (2.0 * trials))
    margin = _Z95 * math.sqrt(
        (proportion * (1.0 - proportion) / trials) + (_Z95**2 / (4.0 * trials * trials))
    )
    return max(0.0, (centre - margin) / denominator)


def _calibration_cases() -> tuple[dict[str, Any], ...]:
    """Return a fixed, balanced, content-addressed verifier holdout.

    The expected labels are generated from integer semantics independently of
    the router under test.  The battery deliberately spans signs and all four
    supported operators, including exact negative division.
    """

    rows: list[dict[str, Any]] = []
    operators = ("+", "-", "*", "/")
    for index in range(64):
        operator = operators[index % len(operators)]
        left = (index * 37) % 997 - 498
        right = (index * 19) % 43 + 1
        if index % 3 == 0:
            right = -right
        if operator == "+":
            answer = left + right
        elif operator == "-":
            answer = left - right
        elif operator == "*":
            answer = left * right
        else:
            left *= right
            answer = left // right
        for correct in (True, False):
            claimed = answer if correct else answer + (1 if index % 2 == 0 else -1)
            source = f"{left} {operator} {right} = {claimed}."
            rows.append(
                {
                    "case_id": f"arith-{index:03d}-{'pass' if correct else 'fail'}",
                    "source": source,
                    "source_sha256": _text_sha256(source),
                    "expected_outcome": "verified" if correct else "refuted",
                }
            )
    return tuple(rows)


def _recalibration_payload() -> dict[str, Any]:
    cases = _calibration_cases()
    case_receipts: list[str] = []
    predictions: list[float] = []
    labels: list[float] = []
    verified_trials = 0
    verified_successes = 0
    false_accepts = 0
    for case in cases:
        atomic = build_atomic_decomposition(case["source"], objective=_OBJECTIVE)
        router = build_deterministic_router_receipt(
            case["source"],
            objective=_OBJECTIVE,
            atomic_receipt=atomic,
        )
        routes = router["routes"]
        if len(routes) != 1:
            raise RuntimeError("test-time critic calibration case did not isolate one claim")
        observed = routes[0]["outcome"]
        expected = case["expected_outcome"]
        prediction = 1.0 if observed == "verified" else 0.0
        label = 1.0 if expected == "verified" else 0.0
        predictions.append(prediction)
        labels.append(label)
        if observed == "verified":
            verified_trials += 1
            verified_successes += int(expected == "verified")
            false_accepts += int(expected != "verified")
        case_receipts.append(
            _canonical_sha256(
                {
                    "case_id": case["case_id"],
                    "source_sha256": case["source_sha256"],
                    "expected_outcome": expected,
                    "observed_outcome": observed,
                    "router_receipt_sha256": router["receipt_sha256"],
                }
            )
        )
    sample_count = len(cases)
    positives = sum(label == 1.0 for label in labels)
    negatives = sample_count - positives
    brier = (
        sum(
            (prediction - label) ** 2 for prediction, label in zip(predictions, labels, strict=True)
        )
        / sample_count
    )
    positive_confidence = (
        sum(
            prediction
            for prediction, label in zip(predictions, labels, strict=True)
            if label == 1.0
        )
        / positives
    )
    negative_confidence = (
        sum(
            1.0 - prediction
            for prediction, label in zip(predictions, labels, strict=True)
            if label == 0.0
        )
        / negatives
    )
    ece = (
        positives * abs(positive_confidence - 1.0) + negatives * abs(negative_confidence - 1.0)
    ) / sample_count
    precision = verified_successes / verified_trials if verified_trials else 0.0
    precision_lower = _wilson_lower(verified_successes, verified_trials)
    false_accept_rate = false_accepts / negatives if negatives else 1.0
    admitted = bool(
        positives >= MIN_CALIBRATION_PER_CLASS
        and negatives >= MIN_CALIBRATION_PER_CLASS
        and verified_trials >= MIN_CALIBRATION_PER_CLASS
        and precision_lower > MIN_PSEUDO_LABEL_CONFIDENCE
        and false_accept_rate == 0.0
        and brier <= MAX_CALIBRATION_BRIER
        and ece <= MAX_CALIBRATION_ECE
    )
    return {
        "schema": CRITIC_RECALIBRATION_SCHEMA,
        "verifier_family": "exact_integer_arithmetic",
        "objective_sha256": _text_sha256(_OBJECTIVE),
        "dataset_sha256": _canonical_sha256(
            [
                {
                    "case_id": case["case_id"],
                    "source_sha256": case["source_sha256"],
                    "expected_outcome": case["expected_outcome"],
                }
                for case in cases
            ]
        ),
        "case_receipt_sha256s": case_receipts,
        "sample_count": sample_count,
        "positives": positives,
        "negatives": negatives,
        "verified_trials": verified_trials,
        "verified_successes": verified_successes,
        "verified_precision": round(precision, 12),
        "verified_precision_lower_95": round(precision_lower, 12),
        "false_accept_rate": round(false_accept_rate, 12),
        "brier": round(brier, 12),
        "ece": round(ece, 12),
        "confidence_threshold": MIN_PSEUDO_LABEL_CONFIDENCE,
        "admitted": admitted,
    }


@lru_cache(maxsize=1)
def _cached_critic_recalibration_json() -> str:
    """Calibrate once per loaded critic implementation.

    The cache stores canonical immutable text, not a caller-visible mapping.
    Each consumer receives a fresh reconstruction, so a local mutation cannot
    alter later admission authority.
    """

    payload = _recalibration_payload()
    receipt = {**payload, "receipt_sha256": _canonical_sha256(payload)}
    return json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def build_critic_recalibration_receipt() -> dict[str, Any]:
    return json.loads(_cached_critic_recalibration_json())


def validate_critic_recalibration_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("test-time critic recalibration receipt is missing")
    expected = build_critic_recalibration_receipt()
    if dict(value) != expected:
        raise ValueError("test-time critic recalibration differs from reconstruction")
    if value["admitted"] is not True:
        raise ValueError("test-time critic recalibration was not admitted")
    return dict(value)


def build_pseudo_label_admission(
    *,
    router_receipt: Mapping[str, Any],
    atomic_receipt: Mapping[str, Any],
    source_sha256: str,
    structural_diversity: Mapping[str, Any],
    critic_recalibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    router = validate_deterministic_router_envelope(
        router_receipt,
        atomic_receipt=atomic_receipt,
    )
    critic = validate_critic_recalibration_receipt(
        critic_recalibration or build_critic_recalibration_receipt()
    )
    if source_sha256 != router["source_sha256"]:
        raise ValueError("pseudo-label source differs from deterministic routes")
    structural_sha = structural_diversity.get("receipt_sha256")
    structural_certified = structural_diversity.get("certified") is True
    verified_routes = [row for row in router["routes"] if row["outcome"] == "verified"]
    verifier_inventory = sorted({row["verifier"] for row in verified_routes})
    learning_atoms_verified = bool(verified_routes) and not any(
        row["outcome"] in {"refuted", "unsupported"} for row in router["routes"]
    )
    exact_domain = bool(
        verifier_inventory and set(verifier_inventory).issubset(_AUTHORIZED_VERIFIERS)
    )
    calibration_sources = {case["source_sha256"] for case in _calibration_cases()}
    query_disjoint = source_sha256 not in calibration_sources
    lower_bound = float(critic["verified_precision_lower_95"])
    if not structural_certified or not _is_sha256(structural_sha):
        reason = "structural_diversity_unproven"
    elif not query_disjoint:
        reason = "query_overlaps_critic_calibration"
    elif not learning_atoms_verified:
        reason = "pseudo_label_not_fully_verified"
    elif not exact_domain:
        reason = "pseudo_label_verifier_not_calibrated"
    elif lower_bound <= MIN_PSEUDO_LABEL_CONFIDENCE:
        reason = "pseudo_label_confidence_below_threshold"
    else:
        reason = "admitted_recalibrated_high_confidence_pseudo_label"
    payload = {
        "schema": PSEUDO_LABEL_ADMISSION_SCHEMA,
        "source_sha256": source_sha256,
        "router_receipt_sha256": router["receipt_sha256"],
        "critic_recalibration_sha256": critic["receipt_sha256"],
        "structural_diversity_sha256": structural_sha if _is_sha256(structural_sha) else "",
        "structural_diversity_certified": structural_certified,
        "query_disjoint_from_calibration": query_disjoint,
        "verifier_inventory": verifier_inventory,
        "all_learning_atoms_verified": learning_atoms_verified,
        "calibrated_probability": float(critic["verified_precision"]),
        "confidence_lower_95": lower_bound,
        "confidence_threshold": MIN_PSEUDO_LABEL_CONFIDENCE,
        "admitted": reason == "admitted_recalibrated_high_confidence_pseudo_label",
        "reason": reason,
    }
    return {**payload, "receipt_sha256": _canonical_sha256(payload)}


def validate_pseudo_label_admission(
    value: Any,
    *,
    router_receipt: Mapping[str, Any],
    atomic_receipt: Mapping[str, Any],
    structural_diversity: Mapping[str, Any],
    critic_recalibration: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("test-time pseudo-label receipt is missing")
    expected = build_pseudo_label_admission(
        router_receipt=router_receipt,
        atomic_receipt=atomic_receipt,
        source_sha256=str(router_receipt.get("source_sha256") or ""),
        structural_diversity=structural_diversity,
        critic_recalibration=critic_recalibration,
    )
    if dict(value) != expected:
        raise ValueError("test-time pseudo-label admission differs from reconstruction")
    return dict(value)


_ARM_FIELDS = {
    "arm",
    "target_tokens_sha256",
    "optimizer",
    "attempts",
    "forward_evaluations",
    "backward_evaluations",
    "line_search_evaluations",
    "layer_apps",
    "probe_layer_apps",
    "probe_tokens_sha256",
    "probe_token_count",
    "score",
}


def _validate_arm(value: Any, *, expected_arm: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ARM_FIELDS:
        raise ValueError("test-time training arm fields differ")
    if (
        value["arm"] != expected_arm
        or not _is_sha256(value["target_tokens_sha256"])
        or value["optimizer"] != "rms_normalized_sgd_backtracking_v1"
        or not _is_sha256(value["probe_tokens_sha256"])
        or not _finite(value["score"])
    ):
        raise ValueError("test-time training arm identity is invalid")
    for field in (
        "attempts",
        "forward_evaluations",
        "backward_evaluations",
        "line_search_evaluations",
        "layer_apps",
        "probe_layer_apps",
        "probe_token_count",
    ):
        if type(value[field]) is not int or value[field] < 0:
            raise ValueError("test-time training arm accounting is invalid")
    if (
        value["attempts"] <= 0
        or value["backward_evaluations"] != value["attempts"]
        or value["forward_evaluations"] != value["attempts"] + value["line_search_evaluations"]
        or value["layer_apps"] <= 0
        or value["probe_layer_apps"] <= 0
        or value["probe_token_count"] <= 0
    ):
        raise ValueError("test-time training arm work did not reconstruct")
    return dict(value)


def build_matched_compute_receipt(
    *,
    treatment: Mapping[str, Any],
    sham: Mapping[str, Any],
    baseline_tokens_sha256: str,
    baseline_score: float,
    critic_before: Mapping[str, Any],
    critic_after: Mapping[str, Any],
) -> dict[str, Any]:
    treatment_arm = _validate_arm(treatment, expected_arm="treatment")
    sham_arm = _validate_arm(sham, expected_arm="sham")
    critic_pre = validate_critic_recalibration_receipt(critic_before)
    critic_post = validate_critic_recalibration_receipt(critic_after)
    if not _is_sha256(baseline_tokens_sha256) or not _finite(baseline_score):
        raise ValueError("test-time training baseline is invalid")
    compute_fields = (
        "attempts",
        "forward_evaluations",
        "backward_evaluations",
        "line_search_evaluations",
        "layer_apps",
        "probe_layer_apps",
        "probe_token_count",
    )
    matched = all(treatment_arm[field] == sham_arm[field] for field in compute_fields)
    target_control_valid = treatment_arm["target_tokens_sha256"] != sham_arm["target_tokens_sha256"]
    reward_drift = abs(
        float(critic_post["verified_precision_lower_95"])
        - float(critic_pre["verified_precision_lower_95"])
    )
    critic_drifted = bool(
        critic_post["admitted"] is not True
        or reward_drift > MAX_REWARD_DRIFT
        or float(critic_post["brier"]) > float(critic_pre["brier"]) + MAX_REWARD_DRIFT
        or float(critic_post["ece"]) > float(critic_pre["ece"]) + MAX_REWARD_DRIFT
    )
    treatment_changed = treatment_arm["probe_tokens_sha256"] != baseline_tokens_sha256
    treatment_distinct_from_sham = (
        treatment_arm["probe_tokens_sha256"] != sham_arm["probe_tokens_sha256"]
    )
    diversity_collapsed = not (treatment_changed and treatment_distinct_from_sham)
    treatment_gain = float(treatment_arm["score"]) - float(baseline_score)
    sham_gain = float(sham_arm["score"]) - float(baseline_score)
    incremental_gain = treatment_gain - sham_gain
    accepted = bool(
        matched
        and target_control_valid
        and not critic_drifted
        and not diversity_collapsed
        and treatment_gain > 1e-6
        and incremental_gain > 1e-6
    )
    reason = (
        "accepted_treatment_beats_equal_compute_sham"
        if accepted
        else "matched_compute_mismatch"
        if not matched
        else "sham_target_not_distinct"
        if not target_control_valid
        else "critic_drift_detected"
        if critic_drifted
        else "trajectory_diversity_collapse"
        if diversity_collapsed
        else "treatment_did_not_improve"
        if treatment_gain <= 1e-6
        else "treatment_did_not_beat_sham"
    )
    payload = {
        "schema": MATCHED_COMPUTE_SCHEMA,
        "baseline_tokens_sha256": baseline_tokens_sha256,
        "baseline_score": float(baseline_score),
        "treatment": treatment_arm,
        "sham": sham_arm,
        "compute_fields": list(compute_fields),
        "compute_matched": matched,
        "target_control_valid": target_control_valid,
        "critic_before_sha256": critic_pre["receipt_sha256"],
        "critic_after_sha256": critic_post["receipt_sha256"],
        "critic_drifted": critic_drifted,
        "reward_drift": round(reward_drift, 12),
        "diversity_collapsed": diversity_collapsed,
        "treatment_gain": float(treatment_gain),
        "sham_gain": float(sham_gain),
        "incremental_gain_over_sham": float(incremental_gain),
        "accepted": accepted,
        "reason": reason,
    }
    return {**payload, "receipt_sha256": _canonical_sha256(payload)}


def validate_matched_compute_receipt(
    value: Any,
    *,
    critic_before: Mapping[str, Any],
    critic_after: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("test-time matched-compute receipt is missing")
    expected = build_matched_compute_receipt(
        treatment=value.get("treatment", {}),
        sham=value.get("sham", {}),
        baseline_tokens_sha256=str(value.get("baseline_tokens_sha256") or ""),
        baseline_score=value.get("baseline_score"),
        critic_before=critic_before,
        critic_after=critic_after,
    )
    if dict(value) != expected:
        raise ValueError("test-time matched-compute receipt differs from reconstruction")
    return dict(value)


def deterministic_sham_target(
    target_tokens: Sequence[int],
    *,
    vocab_size: int,
    episode_id: str,
) -> list[int]:
    if vocab_size < 2:
        raise ValueError("test-time sham target needs at least two vocabulary entries")
    normalized = list(target_tokens)
    if not normalized or any(
        type(token) is not int or not 0 <= token < vocab_size for token in normalized
    ):
        raise ValueError("test-time treatment target is invalid")
    offset = 1 + int.from_bytes(
        hashlib.sha256(episode_id.encode("utf-8")).digest()[:4],
        "big",
    ) % (vocab_size - 1)
    sham = [(token + offset) % vocab_size for token in normalized]
    if sham == normalized:
        raise RuntimeError("test-time sham target failed to differ")
    return sham


def build_test_time_training_receipt(
    *,
    critic_recalibration: Mapping[str, Any],
    pseudo_label_admission: Mapping[str, Any],
    matched_compute: Mapping[str, Any] | None,
) -> dict[str, Any]:
    critic = validate_critic_recalibration_receipt(critic_recalibration)
    pseudo = dict(pseudo_label_admission)
    if pseudo.get("schema") != PSEUDO_LABEL_ADMISSION_SCHEMA:
        raise ValueError("test-time pseudo-label envelope is invalid")
    matched = dict(matched_compute or {})
    if pseudo.get("admitted") is True:
        if not matched:
            decision = "pending_matched_compute"
        else:
            validated = validate_matched_compute_receipt(
                matched,
                critic_before=critic,
                critic_after=critic,
            )
            decision = (
                "accepted_bounded_refinement"
                if validated["accepted"]
                else "rejected_matched_control"
            )
    elif matched:
        raise ValueError("non-admitted pseudo-label cannot run matched controls")
    else:
        decision = "not_admitted"
    payload = {
        "schema": TEST_TIME_TRAINING_SCHEMA,
        "critic_recalibration": critic,
        "pseudo_label_admission": pseudo,
        "matched_compute": matched,
        "decision": decision,
    }
    return {**payload, "receipt_sha256": _canonical_sha256(payload)}


def validate_test_time_training_receipt(
    value: Any,
    *,
    fast_weight_admission: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("test-time training receipt is missing")
    critic = fast_weight_admission.get("critic_recalibration")
    pseudo = fast_weight_admission.get("pseudo_label_admission")
    if value.get("critic_recalibration") != critic or value.get("pseudo_label_admission") != pseudo:
        raise ValueError("test-time training authority differs from admission")
    expected = build_test_time_training_receipt(
        critic_recalibration=critic,
        pseudo_label_admission=pseudo,
        matched_compute=value.get("matched_compute"),
    )
    if dict(value) != expected:
        raise ValueError("test-time training receipt differs from reconstruction")
    return dict(value)


__all__ = [
    "CRITIC_RECALIBRATION_SCHEMA",
    "MATCHED_COMPUTE_SCHEMA",
    "MATCHED_LINE_SEARCH_EVALUATIONS",
    "MIN_PSEUDO_LABEL_CONFIDENCE",
    "PSEUDO_LABEL_ADMISSION_SCHEMA",
    "TEST_TIME_TRAINING_SCHEMA",
    "build_critic_recalibration_receipt",
    "build_matched_compute_receipt",
    "build_pseudo_label_admission",
    "build_test_time_training_receipt",
    "deterministic_sham_target",
    "validate_critic_recalibration_receipt",
    "validate_matched_compute_receipt",
    "validate_pseudo_label_admission",
    "validate_test_time_training_receipt",
]
