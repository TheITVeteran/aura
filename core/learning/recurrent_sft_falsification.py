"""Deterministic controls and paired scoring for recurrent-SFT falsification.

This module is intentionally free of MLX and model-loading concerns. Training
and evaluation processes can therefore bind the same control transformations
and statistical decision rules without importing one another or sharing
evaluator-only data.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Never

SCHEMA_VERSION: Final = "aura.rlc.recurrent_sft_falsification.v1"
CONTROL_ARMS: Final = (
    "sham_labels",
    "shuffled_traces",
    "syntax_only",
)
BASE_ARM: Final = "base_recurrent"
TRAINED_ARM: Final = "trained_recurrent"
ALL_ARMS: Final = (BASE_ARM, TRAINED_ARM, *CONTROL_ARMS)

_ROW_FIELDS: Final = frozenset(
    {
        "example_id",
        "family",
        "target_kind",
        "prompt_tokens",
        "answer_tokens",
        "full_token_count",
    }
)
_OBSERVATION_FIELDS: Final = frozenset(
    {
        "example_id",
        "family",
        "loss",
        "target_top1",
        "generated_correct",
    }
)


class RecurrentSFTFalsificationError(ValueError):
    """A control or observation violated the preregistered contract."""


def _fail(code: str) -> Never:
    raise RecurrentSFTFalsificationError(
        str(code or "recurrent_sft_falsification_invalid")
    )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise RecurrentSFTFalsificationError(
            "recurrent_sft_falsification_json_invalid"
        ) from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _tokens(value: Any, *, role: str) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
        or any(type(token) is not int or token < 0 for token in value)
    ):
        _fail(f"recurrent_sft_{role}_tokens_invalid")
    return tuple(value)


def _projected_rows(rows: Any) -> tuple[dict[str, Any], ...]:
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
        or not rows
    ):
        _fail("recurrent_sft_control_rows_invalid")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            _fail("recurrent_sft_control_row_schema_invalid")
        example_id = row.get("example_id")
        family = row.get("family")
        target_kind = row.get("target_kind")
        full_token_count = row.get("full_token_count")
        prompt = _tokens(row.get("prompt_tokens"), role="prompt")
        answer = _tokens(row.get("answer_tokens"), role="answer")
        if (
            not isinstance(example_id, str)
            or len(example_id) != 64
            or any(character not in "0123456789abcdef" for character in example_id)
            or example_id in seen_ids
            or not isinstance(family, str)
            or not family
            or not isinstance(target_kind, str)
            or not target_kind
            or type(full_token_count) is not int
            or full_token_count != len(prompt) + len(answer)
        ):
            _fail("recurrent_sft_control_row_identity_invalid")
        seen_ids.add(example_id)
        normalized.append(
            {
                "example_id": example_id,
                "family": family,
                "target_kind": target_kind,
                "prompt_tokens": list(prompt),
                "answer_tokens": list(answer),
                "full_token_count": full_token_count,
            }
        )
    return tuple(normalized)


def _repeat_to_length(source: Sequence[int], length: int) -> list[int]:
    if length <= 0:
        return []
    if not source:
        _fail("recurrent_sft_sham_donor_empty")
    quotient, remainder = divmod(length, len(source))
    return list(source) * quotient + list(source[:remainder])


def _sham_answers(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[int]]:
    ordered = sorted(rows, key=lambda row: str(row["example_id"]))
    if len(ordered) < 2:
        _fail("recurrent_sft_sham_requires_multiple_rows")
    result: dict[str, list[int]] = {}
    for index, row in enumerate(ordered):
        donor = ordered[(index + 1) % len(ordered)]
        answer = list(row["answer_tokens"])
        donor_answer = list(donor["answer_tokens"])
        content_length = len(answer) - 1
        donor_content = donor_answer[:-1] or donor_answer
        result[str(row["example_id"])] = [
            *_repeat_to_length(donor_content, content_length),
            answer[-1],
        ]
    return result


def _shuffled_answer(
    answer: Sequence[int],
    *,
    example_id: str,
    seed: int,
) -> list[int]:
    if len(answer) <= 2:
        return list(answer)
    content = list(answer[:-1])
    order = sorted(
        range(len(content)),
        key=lambda index: hashlib.sha256(
            f"{seed}:{example_id}:{index}".encode("ascii")
        ).digest(),
    )
    if order == list(range(len(content))):
        order = order[1:] + order[:1]
    return [*(content[index] for index in order), answer[-1]]


def _syntax_only_answer(
    answer: Sequence[int],
    *,
    token_surfaces: Mapping[int, str],
    neutral_token_id: int,
) -> list[int]:
    if type(neutral_token_id) is not int or neutral_token_id < 0:
        _fail("recurrent_sft_syntax_neutral_token_invalid")
    transformed: list[int] = []
    for index, token in enumerate(answer):
        if index == len(answer) - 1:
            transformed.append(token)
            continue
        surface = token_surfaces.get(token)
        if not isinstance(surface, str):
            _fail("recurrent_sft_syntax_token_surface_missing")
        transformed.append(
            neutral_token_id if any(character.isalnum() for character in surface) else token
        )
    return transformed


@dataclass(frozen=True, slots=True)
class ControlTransformation:
    """Exact transformed rows plus a hash-bound workload-equivalence receipt."""

    rows: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def transform_control_rows(
    rows: Any,
    *,
    arm: str,
    seed: int,
    token_surfaces: Mapping[int, str] | None = None,
    neutral_token_id: int | None = None,
) -> ControlTransformation:
    """Build one negative-control label arm without changing model workload."""

    if arm not in CONTROL_ARMS:
        _fail("recurrent_sft_control_arm_invalid")
    if type(seed) is not int or not 0 <= seed < 2**63:
        _fail("recurrent_sft_control_seed_invalid")
    original = _projected_rows(rows)
    sham = _sham_answers(original) if arm == "sham_labels" else {}
    transformed: list[dict[str, Any]] = []
    changed_tokens = 0
    for row in original:
        answer = list(row["answer_tokens"])
        if arm == "sham_labels":
            replacement = sham[str(row["example_id"])]
        elif arm == "shuffled_traces":
            replacement = _shuffled_answer(
                answer,
                example_id=str(row["example_id"]),
                seed=seed,
            )
        else:
            if token_surfaces is None or neutral_token_id is None:
                _fail("recurrent_sft_syntax_token_contract_missing")
            replacement = _syntax_only_answer(
                answer,
                token_surfaces=token_surfaces,
                neutral_token_id=neutral_token_id,
            )
        changed_tokens += sum(
            left != right for left, right in zip(answer, replacement, strict=True)
        )
        candidate = {**row, "answer_tokens": replacement}
        if (
            candidate["prompt_tokens"] != row["prompt_tokens"]
            or len(replacement) != len(answer)
            or candidate["full_token_count"]
            != len(candidate["prompt_tokens"]) + len(replacement)
            or replacement[-1] != answer[-1]
        ):
            _fail("recurrent_sft_control_workload_drift")
        transformed.append(candidate)
    if changed_tokens == 0:
        _fail("recurrent_sft_control_no_effect")
    original_token_budget = sum(row["full_token_count"] for row in original)
    transformed_token_budget = sum(row["full_token_count"] for row in transformed)
    if transformed_token_budget != original_token_budget:
        _fail("recurrent_sft_control_token_budget_drift")
    body = {
        "schema": f"{SCHEMA_VERSION}.control_transform",
        "arm": arm,
        "seed": seed,
        "row_count": len(original),
        "original_rows_sha256": sha256_json(original),
        "transformed_rows_sha256": sha256_json(transformed),
        "prompt_tokens_unchanged": True,
        "answer_lengths_unchanged": True,
        "terminal_tokens_unchanged": True,
        "row_order_unchanged": True,
        "full_token_budget": original_token_budget,
        "changed_answer_tokens": changed_tokens,
        "changed_answer_fraction": round(
            changed_tokens
            / sum(len(row["answer_tokens"]) for row in original),
            12,
        ),
    }
    manifest = {**body, "manifest_sha256": sha256_json(body)}
    return ControlTransformation(tuple(transformed), manifest)


def _probability_mass_at_or_below(successes: int, trials: int) -> float:
    return sum(math.comb(trials, index) for index in range(successes + 1)) / (
        2**trials
    )


def exact_two_sided_sign_test(deltas: Sequence[float]) -> dict[str, Any]:
    """Return an exact paired sign test, excluding exact ties."""

    finite = [float(delta) for delta in deltas]
    if any(not math.isfinite(delta) for delta in finite):
        _fail("recurrent_sft_sign_test_nonfinite")
    improved = sum(delta < 0.0 for delta in finite)
    regressed = sum(delta > 0.0 for delta in finite)
    trials = improved + regressed
    p_value = (
        1.0
        if trials == 0
        else min(
            1.0,
            2.0 * _probability_mass_at_or_below(min(improved, regressed), trials),
        )
    )
    return {
        "non_ties": trials,
        "improved": improved,
        "regressed": regressed,
        "ties": len(finite) - trials,
        "two_sided_p_value": round(p_value, 12),
    }


def _observation_rows(rows: Any, *, arm: str) -> tuple[dict[str, Any], ...]:
    if arm not in ALL_ARMS:
        _fail("recurrent_sft_observation_arm_invalid")
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
        or not rows
    ):
        _fail("recurrent_sft_observations_invalid")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _OBSERVATION_FIELDS:
            _fail("recurrent_sft_observation_schema_invalid")
        example_id = row.get("example_id")
        family = row.get("family")
        loss = row.get("loss")
        target_top1 = row.get("target_top1")
        generated_correct = row.get("generated_correct")
        if (
            not isinstance(example_id, str)
            or len(example_id) != 64
            or example_id in seen_ids
            or not isinstance(family, str)
            or not family
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or float(loss) < 0.0
            or not isinstance(target_top1, Sequence)
            or isinstance(target_top1, (str, bytes, bytearray))
            or not target_top1
            or any(type(value) is not bool for value in target_top1)
            or (
                generated_correct is not None
                and type(generated_correct) is not bool
            )
        ):
            _fail("recurrent_sft_observation_invalid")
        seen_ids.add(example_id)
        normalized.append(
            {
                "example_id": example_id,
                "family": family,
                "loss": float(loss),
                "target_top1": list(target_top1),
                "generated_correct": generated_correct,
            }
        )
    return tuple(sorted(normalized, key=lambda row: row["example_id"]))


def _paired_summary(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(reference) != len(candidate):
        _fail("recurrent_sft_pair_count_mismatch")
    loss_deltas: list[float] = []
    wrong_to_right = 0
    right_to_wrong = 0
    token_count = 0
    reference_correct = 0
    candidate_correct = 0
    generated_pairs = 0
    generated_wrong_to_right = 0
    generated_right_to_wrong = 0
    pair_rows: list[dict[str, Any]] = []
    for baseline, observed in zip(reference, candidate, strict=True):
        if (
            baseline["example_id"] != observed["example_id"]
            or baseline["family"] != observed["family"]
            or len(baseline["target_top1"]) != len(observed["target_top1"])
        ):
            _fail("recurrent_sft_pair_alignment_invalid")
        delta = float(observed["loss"]) - float(baseline["loss"])
        loss_deltas.append(delta)
        transitions: Counter[str] = Counter()
        for before, after in zip(
            baseline["target_top1"],
            observed["target_top1"],
            strict=True,
        ):
            token_count += 1
            reference_correct += int(before)
            candidate_correct += int(after)
            if not before and after:
                wrong_to_right += 1
                transitions["wrong_to_right"] += 1
            elif before and not after:
                right_to_wrong += 1
                transitions["right_to_wrong"] += 1
        before_generation = baseline["generated_correct"]
        after_generation = observed["generated_correct"]
        if before_generation is not None or after_generation is not None:
            if type(before_generation) is not bool or type(after_generation) is not bool:
                _fail("recurrent_sft_generation_pair_incomplete")
            generated_pairs += 1
            if not before_generation and after_generation:
                generated_wrong_to_right += 1
            elif before_generation and not after_generation:
                generated_right_to_wrong += 1
        pair_rows.append(
            {
                "example_id": baseline["example_id"],
                "family": baseline["family"],
                "loss_delta": round(delta, 12),
                "wrong_to_right": transitions["wrong_to_right"],
                "right_to_wrong": transitions["right_to_wrong"],
            }
        )
    sign_test = exact_two_sided_sign_test(loss_deltas)
    return {
        "example_count": len(reference),
        "mean_reference_loss": round(
            statistics.fmean(float(row["loss"]) for row in reference),
            12,
        ),
        "mean_candidate_loss": round(
            statistics.fmean(float(row["loss"]) for row in candidate),
            12,
        ),
        "mean_loss_delta": round(statistics.fmean(loss_deltas), 12),
        "median_loss_delta": round(statistics.median(loss_deltas), 12),
        "sign_test": sign_test,
        "target_token_count": token_count,
        "reference_target_top1": reference_correct,
        "candidate_target_top1": candidate_correct,
        "target_top1_delta": candidate_correct - reference_correct,
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "net_error_corrections": wrong_to_right - right_to_wrong,
        "generated_pairs": generated_pairs,
        "generated_wrong_to_right": generated_wrong_to_right,
        "generated_right_to_wrong": generated_right_to_wrong,
        "pairing_sha256": sha256_json(pair_rows),
    }


def compare_observation_arms(
    reference_rows: Any,
    candidate_rows: Any,
    *,
    reference_arm: str,
    candidate_arm: str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Build exact paired overall and per-family transfer measurements."""

    if (
        reference_arm == candidate_arm
        or reference_arm not in ALL_ARMS
        or candidate_arm not in ALL_ARMS
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0.0 < float(alpha) < 1.0
    ):
        _fail("recurrent_sft_comparison_contract_invalid")
    reference = _observation_rows(reference_rows, arm=reference_arm)
    candidate = _observation_rows(candidate_rows, arm=candidate_arm)
    if [row["example_id"] for row in reference] != [
        row["example_id"] for row in candidate
    ]:
        _fail("recurrent_sft_comparison_identity_mismatch")
    overall = _paired_summary(reference, candidate)
    families = sorted({str(row["family"]) for row in reference})
    by_family: dict[str, Any] = {}
    for family in families:
        family_reference = [row for row in reference if row["family"] == family]
        family_candidate = [row for row in candidate if row["family"] == family]
        by_family[family] = _paired_summary(family_reference, family_candidate)
    passed = (
        overall["mean_loss_delta"] < 0.0
        and overall["sign_test"]["two_sided_p_value"] <= float(alpha)
        and overall["net_error_corrections"] > 0
        and all(summary["mean_loss_delta"] <= 0.0 for summary in by_family.values())
    )
    body = {
        "schema": f"{SCHEMA_VERSION}.paired_comparison",
        "reference_arm": reference_arm,
        "candidate_arm": candidate_arm,
        "alpha": float(alpha),
        "overall": overall,
        "by_family": by_family,
        "negative_transfer_families": [
            family
            for family, summary in by_family.items()
            if summary["mean_loss_delta"] > 0.0
        ],
        "passed": passed,
    }
    return {**body, "comparison_sha256": sha256_json(body)}


def build_falsification_verdict(
    observations_by_arm: Mapping[str, Any],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Require trained recurrence to beat base and every negative control."""

    if not isinstance(observations_by_arm, Mapping) or set(
        observations_by_arm
    ) != set(ALL_ARMS):
        _fail("recurrent_sft_verdict_arm_set_invalid")
    trained_vs_base = compare_observation_arms(
        observations_by_arm[BASE_ARM],
        observations_by_arm[TRAINED_ARM],
        reference_arm=BASE_ARM,
        candidate_arm=TRAINED_ARM,
        alpha=alpha,
    )
    trained_vs_controls = {
        control: compare_observation_arms(
            observations_by_arm[control],
            observations_by_arm[TRAINED_ARM],
            reference_arm=control,
            candidate_arm=TRAINED_ARM,
            alpha=alpha,
        )
        for control in CONTROL_ARMS
    }
    passed = trained_vs_base["passed"] and all(
        comparison["passed"] for comparison in trained_vs_controls.values()
    )
    body = {
        "schema": f"{SCHEMA_VERSION}.verdict",
        "alpha": float(alpha),
        "trained_vs_base": trained_vs_base,
        "trained_vs_controls": trained_vs_controls,
        "heldout_transfer_proven": passed,
        "reasoning_gain_proven": False,
        "frontier_performance_proven": False,
        "resident_32b_result": False,
        "wow_signal": False,
        "status": (
            "small_checkpoint_heldout_transfer_passed"
            if passed
            else "small_checkpoint_heldout_transfer_not_proven"
        ),
    }
    return {**body, "verdict_sha256": sha256_json(body)}


__all__ = [
    "ALL_ARMS",
    "BASE_ARM",
    "CONTROL_ARMS",
    "ControlTransformation",
    "RecurrentSFTFalsificationError",
    "SCHEMA_VERSION",
    "TRAINED_ARM",
    "build_falsification_verdict",
    "compare_observation_arms",
    "exact_two_sided_sign_test",
    "sha256_json",
    "transform_control_rows",
]
