#!/usr/bin/env python
"""Implementation-independent scoring kernel for resident RLC campaigns.

This module intentionally does not import the production campaign grader,
experiment statistics, response parser, or task scorer.  It consumes the raw
committed campaign rows and independently reconstructs correctness, paired
effects, multiplicity correction, compute matching, and the 2x2 interaction.

The kernel is a second implementation, not an alternate source of truth.  A
claim is eligible only when this result and the production result agree.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Never, cast

SCHEMA = "aura.latent_cortex.independent_campaign_grade.v1"
PROVEN = "PROVEN"
SUPPORTED = "SUPPORTED"
CONJECTURE = "CONJECTURE"
REFUTED = "REFUTED"

BASE_VANILLA = "base_vanilla"
BASE_RLC = "base_rlc"
ADAPTER_VANILLA = "adapter_vanilla"
ADAPTER_RLC = "adapter_rlc"
BASE_EQUAL_COMPUTE = "base_equal_compute"
ADAPTER_EQUAL_COMPUTE = "adapter_equal_compute"
PRIMARY_ARMS = (BASE_VANILLA, BASE_RLC, ADAPTER_VANILLA, ADAPTER_RLC)
FULL_ARMS = (*PRIMARY_ARMS, BASE_EQUAL_COMPUTE, ADAPTER_EQUAL_COMPUTE)

_MIN_DOMAIN_TRIALS = 20
_BOOTSTRAP_RESAMPLES = 20_000
_MAX_RESPONSE_BYTES = 256 * 1024
_FINAL_MARKER = "FINAL_ANSWER:"


class IndependentScoringError(ValueError):
    """The raw evidence cannot be scored without trusting producer output."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    error = IndependentScoringError(code)
    raise error


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise IndependentScoringError("independent_noncanonical_value") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_json_object(payload: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("independent_answer_duplicate_key")
            result[key] = value
        return result

    def parse_int(raw: str) -> int:
        value = int(raw)
        if not -(1 << 63) <= value <= (1 << 63) - 1:
            _fail("independent_answer_integer_out_of_bounds")
        return value

    def reject_number(_raw: str) -> Never:
        _fail("independent_answer_noninteger_number")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs_hook,
            parse_int=parse_int,
            parse_float=reject_number,
            parse_constant=reject_number,
        )
    except IndependentScoringError:
        raise
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
        raise IndependentScoringError("independent_answer_invalid_json") from exc
    if not isinstance(value, dict):
        _fail("independent_answer_not_object")
    return cast(dict[str, Any], value)


def _parse_terminal_answer(response: Any) -> dict[str, Any]:
    if (
        not isinstance(response, str)
        or not response.strip()
        or len(response.encode("utf-8")) > _MAX_RESPONSE_BYTES
        or "\x00" in response
    ):
        _fail("independent_response_invalid")
    if response.count(_FINAL_MARKER) != 1:
        _fail("independent_answer_marker_count")
    lines = response.rstrip().splitlines()
    if not lines or not lines[-1].startswith(_FINAL_MARKER):
        _fail("independent_answer_not_terminal")
    encoded = lines[-1][len(_FINAL_MARKER) :].strip()
    if not encoded:
        _fail("independent_answer_missing")
    return _strict_json_object(encoded)


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _score_response(task: Any, response: Any) -> dict[str, Any]:
    try:
        answer = _parse_terminal_answer(response)
    except IndependentScoringError as exc:
        return {
            "parsed": False,
            "correct": False,
            "reason": exc.code,
            "normalized_answer_sha256": None,
        }
    try:
        blinded = task.reveal_for_verifier()
    except Exception as exc:  # noqa: BLE001 - issuer evidence is untrusted here
        raise IndependentScoringError("independent_answer_reveal_failed") from exc
    expected = blinded.get("expected") if isinstance(blinded, Mapping) else None
    if not isinstance(expected, dict):
        _fail("independent_expected_answer_invalid")
    correct = _strict_equal(answer, expected)
    return {
        "parsed": True,
        "correct": correct,
        "reason": "correct" if correct else "incorrect_or_schema_mismatch",
        "normalized_answer_sha256": _sha256(answer),
    }


def _extract_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    plan: Any,
    issuer_tasks: Sequence[Any],
) -> tuple[dict[str, dict[str, tuple[str, bool, int]]], tuple[str, ...]]:
    document = plan.to_dict()
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        _fail("independent_plan_metadata_invalid")
    raw_arms = metadata.get("arms")
    task_manifest = metadata.get("task_manifest")
    if not isinstance(raw_arms, list) or not isinstance(task_manifest, Mapping):
        _fail("independent_plan_metadata_invalid")
    arms = tuple(raw_arms)
    if (
        len(set(arms)) != len(arms)
        or not set(PRIMARY_ARMS).issubset(arms)
        or any(arm not in FULL_ARMS for arm in arms)
    ):
        _fail("independent_plan_arms_invalid")
    public_tasks = task_manifest.get("tasks")
    if not isinstance(public_tasks, list) or not public_tasks:
        _fail("independent_plan_tasks_invalid")
    task_records: dict[str, Mapping[str, Any]] = {}
    for task in public_tasks:
        if not isinstance(task, Mapping) or not isinstance(task.get("task_id"), str):
            _fail("independent_plan_tasks_invalid")
        task_id = cast(str, task["task_id"])
        if task_id in task_records:
            _fail("independent_plan_task_duplicate")
        task_records[task_id] = task
    issuer_by_id: dict[str, Any] = {}
    for task in issuer_tasks:
        task_id = getattr(task, "task_id", None)
        if not isinstance(task_id, str) or task_id in issuer_by_id:
            _fail("independent_issuer_tasks_invalid")
        public = getattr(task, "public", None)
        if public is None or public.to_dict() != dict(task_records.get(task_id, {})):
            _fail("independent_issuer_task_mismatch")
        issuer_by_id[task_id] = task
    if set(issuer_by_id) != set(task_records):
        _fail("independent_issuer_task_mismatch")

    expected_pairs = {
        (task_id, arm) for task_id in task_records for arm in arms
    }
    planned_pairs: set[tuple[str, str]] = set()
    for cell_id in plan.cell_ids:
        definition = plan.cell_definition(cell_id)
        pair = (definition.get("task_id"), definition.get("arm"))
        if (
            not isinstance(pair[0], str)
            or pair[0] not in task_records
            or pair[1] not in arms
            or pair in planned_pairs
        ):
            _fail("independent_plan_cell_invalid")
        planned_pairs.add(cast(tuple[str, str], pair))
    if planned_pairs != expected_pairs:
        _fail("independent_plan_coverage_invalid")

    rows: dict[str, dict[str, tuple[str, bool, int]]] = defaultdict(dict)
    seen_cells: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            _fail("independent_record_invalid")
        cell_id = record.get("cell_id")
        definition = record.get("definition")
        result = record.get("result")
        verification = record.get("verification")
        commit = record.get("commit")
        if (
            not isinstance(cell_id, str)
            or cell_id not in plan.cell_ids
            or cell_id in seen_cells
            or not isinstance(definition, Mapping)
            or not isinstance(result, Mapping)
            or not isinstance(verification, Mapping)
            or not isinstance(commit, Mapping)
        ):
            _fail("independent_record_shape_invalid")
        seen_cells.add(cell_id)
        expected_definition = plan.cell_definition(cell_id)
        if dict(definition) != expected_definition:
            _fail("independent_record_definition_mismatch")
        task_id = expected_definition.get("task_id")
        domain = expected_definition.get("domain")
        arm = expected_definition.get("arm")
        if (
            not isinstance(task_id, str)
            or not isinstance(domain, str)
            or arm not in arms
            or result.get("arm") != arm
        ):
            _fail("independent_record_identity_invalid")
        text = result.get("text")
        output_sha256 = result.get("output_sha256")
        layer_apps = result.get("layer_apps")
        if (
            not isinstance(text, str)
            or not isinstance(output_sha256, str)
            or hashlib.sha256(text.encode("utf-8")).hexdigest() != output_sha256
            or type(layer_apps) is not int
            or layer_apps <= 0
        ):
            _fail("independent_result_invalid")
        independent = _score_response(issuer_by_id[task_id], text)
        score_receipt = verification.get("score_receipt")
        if (
            type(verification.get("correct")) is not bool
            or verification.get("correct") is not independent["correct"]
            or not isinstance(score_receipt, Mapping)
            or score_receipt.get("parsed") is not independent["parsed"]
            or score_receipt.get("correct") is not independent["correct"]
            or score_receipt.get("normalized_answer_sha256")
            != independent["normalized_answer_sha256"]
            or verification.get("answer_commitment_sha256")
            != task_records[task_id].get("answer_commitment_sha256")
        ):
            _fail("independent_score_disagrees_with_receipt")
        if (
            commit.get("result_sha256") != _sha256(dict(result))
            or commit.get("verification_sha256")
            != _sha256(dict(verification))
        ):
            _fail("independent_commitment_mismatch")
        if arm in rows[task_id]:
            _fail("independent_task_arm_duplicate")
        rows[task_id][cast(str, arm)] = (
            domain,
            cast(bool, independent["correct"]),
            layer_apps,
        )
    if seen_cells != set(plan.cell_ids) or any(
        set(task_arms) != set(arms) for task_arms in rows.values()
    ):
        _fail("independent_campaign_incomplete")
    return dict(rows), arms


def _exact_greater_pvalue(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant <= 0:
        return 1.0
    numerator = sum(
        math.comb(discordant, k) for k in range(wins, discordant + 1)
    )
    return min(1.0, numerator / (2**discordant))


def _holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * pvalue))
        adjusted[name] = running
    return adjusted


def _bootstrap_interval(
    values: Sequence[int], *, alpha: float, seed: int
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(set(values)) == 1:
        value = float(values[0])
        return value, value
    import numpy as np

    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty((_BOOTSTRAP_RESAMPLES,), dtype=np.float64)
    for start in range(0, _BOOTSTRAP_RESAMPLES, 250):
        count = min(250, _BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, len(data), size=(count, len(data)))
        means[start : start + count] = data[indices].mean(axis=1)
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _comparison(
    rows: Mapping[str, Mapping[str, tuple[str, bool, int]]],
    *,
    treatment: str,
    control: str,
    require_compute: bool,
    compute_tolerance: float,
) -> dict[str, Any]:
    by_domain: dict[str, list[tuple[str, bool, bool, int, int]]] = defaultdict(list)
    for task_id in sorted(rows):
        arms = rows[task_id]
        if treatment not in arms or control not in arms:
            _fail("independent_comparison_incomplete")
        treatment_domain, treatment_ok, treatment_cost = arms[treatment]
        control_domain, control_ok, control_cost = arms[control]
        if treatment_domain != control_domain:
            _fail("independent_domain_drift")
        by_domain[treatment_domain].append(
            (task_id, treatment_ok, control_ok, treatment_cost, control_cost)
        )

    alpha = 0.05
    minimum_effect = 0.02
    family_alpha = alpha / max(1, len(by_domain))
    raw_pvalues: dict[str, float] = {}
    family_stats: dict[str, dict[str, Any]] = {}
    invalid_compute: list[str] = []
    underpowered: list[str] = []
    pooled: list[int] = []
    for ordinal, domain in enumerate(sorted(by_domain)):
        observations = by_domain[domain]
        differences = [int(row[1]) - int(row[2]) for row in observations]
        wins = differences.count(1)
        losses = differences.count(-1)
        mismatches = [
            row[0]
            for row in observations
            if abs(row[3] - row[4]) / max(1, row[4]) > compute_tolerance
        ]
        if require_compute and mismatches:
            invalid_compute.append(domain)
        low, high = _bootstrap_interval(
            differences,
            alpha=family_alpha,
            seed=0xA17A0000 + ordinal,
        )
        pvalue = _exact_greater_pvalue(wins, losses)
        if len(observations) < _MIN_DOMAIN_TRIALS:
            underpowered.append(domain)
        else:
            raw_pvalues[domain] = pvalue
        family_stats[domain] = {
            "n": len(observations),
            "treatment_wins": wins,
            "control_wins": losses,
            "ties": len(observations) - wins - losses,
            "paired_effect": sum(differences) / len(differences),
            "interval": [low, high],
            "one_sided_exact_p": pvalue,
            "compute_mismatch_task_ids": mismatches,
        }
        pooled.extend(differences)
    adjusted = _holm(raw_pvalues)
    positive = [
        domain
        for domain, stats in family_stats.items()
        if domain in adjusted
        and adjusted[domain] < alpha
        and stats["interval"][0] > minimum_effect
        and domain not in invalid_compute
    ]
    regressed = [
        domain
        for domain, stats in family_stats.items()
        if stats["interval"][1] < -minimum_effect
    ]
    pooled_wins = pooled.count(1)
    pooled_losses = pooled.count(-1)
    pooled_low, pooled_high = _bootstrap_interval(
        pooled,
        alpha=alpha,
        seed=0xA17AFFFF,
    )
    pooled_p = _exact_greater_pvalue(pooled_wins, pooled_losses)
    required_positive = max(2, math.ceil(len(family_stats) * 2 / 3))
    pooled_positive = (
        len(pooled) >= _MIN_DOMAIN_TRIALS
        and pooled_p < alpha
        and pooled_low > minimum_effect
    )
    if invalid_compute or underpowered:
        tier = CONJECTURE
    elif len(positive) >= required_positive and pooled_positive and not regressed:
        tier = PROVEN
    elif positive and pooled_positive and not regressed:
        tier = SUPPORTED
    elif regressed or (pooled and pooled_high <= 0.0):
        tier = REFUTED
    else:
        tier = CONJECTURE
    return {
        "tier": tier,
        "families": family_stats,
        "holm_adjusted_p": adjusted,
        "positive_families": positive,
        "regressed_families": regressed,
        "underpowered_families": underpowered,
        "invalid_compute_families": invalid_compute,
        "required_positive_families": required_positive,
        "pooled": {
            "n": len(pooled),
            "treatment_wins": pooled_wins,
            "control_wins": pooled_losses,
            "paired_effect": sum(pooled) / len(pooled) if pooled else 0.0,
            "interval": [pooled_low, pooled_high],
            "one_sided_exact_p": pooled_p,
        },
    }


def _exact_sign_flip_pvalue(values: Sequence[int]) -> float:
    magnitudes = [abs(value) for value in values if value]
    if not magnitudes:
        return 1.0
    distribution: dict[int, int] = {0: 1}
    for magnitude in magnitudes:
        updated: dict[int, int] = defaultdict(int)
        for total, count in distribution.items():
            updated[total + magnitude] += count
            updated[total - magnitude] += count
        distribution = dict(updated)
    observed = sum(values)
    tail = sum(count for total, count in distribution.items() if total >= observed)
    return tail / (2 ** len(magnitudes))


def independent_grade_campaign(
    records: Iterable[Mapping[str, Any]],
    *,
    plan: Any,
    issuer_tasks: Sequence[Any],
) -> dict[str, Any]:
    """Recompute a complete campaign without production grading code."""

    rows, arms = _extract_rows(records, plan=plan, issuer_tasks=issuer_tasks)
    comparisons = {
        "base_rlc_gain": _comparison(
            rows,
            treatment=BASE_RLC,
            control=BASE_VANILLA,
            require_compute=False,
            compute_tolerance=1.0,
        ),
        "adapter_rlc_gain": _comparison(
            rows,
            treatment=ADAPTER_RLC,
            control=ADAPTER_VANILLA,
            require_compute=False,
            compute_tolerance=1.0,
        ),
        "adapter_effect_under_rlc": _comparison(
            rows,
            treatment=ADAPTER_RLC,
            control=BASE_RLC,
            require_compute=False,
            compute_tolerance=1.0,
        ),
        "adapter_effect_under_vanilla": _comparison(
            rows,
            treatment=ADAPTER_VANILLA,
            control=BASE_VANILLA,
            require_compute=False,
            compute_tolerance=1.0,
        ),
    }
    if BASE_EQUAL_COMPUTE in arms:
        comparisons["base_equal_compute"] = _comparison(
            rows,
            treatment=BASE_RLC,
            control=BASE_EQUAL_COMPUTE,
            require_compute=True,
            compute_tolerance=0.20,
        )
    if ADAPTER_EQUAL_COMPUTE in arms:
        comparisons["adapter_equal_compute"] = _comparison(
            rows,
            treatment=ADAPTER_RLC,
            control=ADAPTER_EQUAL_COMPUTE,
            require_compute=True,
            compute_tolerance=0.20,
        )

    domain_counts: dict[str, int] = defaultdict(int)
    interactions: list[int] = []
    for task_arms in rows.values():
        domain_counts[task_arms[BASE_VANILLA][0]] += 1
        interactions.append(
            (int(task_arms[ADAPTER_RLC][1]) - int(task_arms[ADAPTER_VANILLA][1]))
            - (int(task_arms[BASE_RLC][1]) - int(task_arms[BASE_VANILLA][1]))
        )
    interaction_low, interaction_high = _bootstrap_interval(
        interactions,
        alpha=0.025,
        seed=0xA17A2A2A,
    )
    interaction_p = _exact_sign_flip_pvalue(interactions)
    underpowered = sorted(
        domain for domain, count in domain_counts.items()
        if count < _MIN_DOMAIN_TRIALS
    )
    required = ["adapter_rlc_gain", "adapter_effect_under_rlc"]
    if ADAPTER_EQUAL_COMPUTE in arms:
        required.append("adapter_equal_compute")
    statistically_proven = (
        not underpowered
        and all(comparisons[name]["tier"] == PROVEN for name in required)
        and interaction_low > 0.02
        and interaction_p < 0.05
        and not comparisons["adapter_effect_under_vanilla"]["regressed_families"]
    )
    refuted = (
        comparisons["adapter_rlc_gain"]["tier"] == REFUTED
        or interaction_high <= 0.0
    )
    claim_eligible = plan.to_dict().get("metadata", {}).get("claim_eligible") is True
    if statistically_proven and claim_eligible:
        verdict, tier = "gain_proven", PROVEN
    elif statistically_proven:
        verdict, tier = "gain_observed_preflight", CONJECTURE
    elif underpowered:
        verdict, tier = "incomplete_underpowered", CONJECTURE
    elif refuted:
        verdict, tier = "gain_refuted", REFUTED
    else:
        verdict, tier = "inconclusive", CONJECTURE
    return {
        "schema": SCHEMA,
        "verdict": verdict,
        "claim_tier": tier,
        "observed_task_count": len(rows),
        "observed_cell_count": sum(len(task_arms) for task_arms in rows.values()),
        "domain_counts": dict(sorted(domain_counts.items())),
        "comparisons": comparisons,
        "interaction": {
            "mean": sum(interactions) / len(interactions),
            "interval_95": [interaction_low, interaction_high],
            "one_sided_exact_sign_flip_p": interaction_p,
        },
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


__all__ = [
    "IndependentScoringError",
    "independent_grade_campaign",
]
