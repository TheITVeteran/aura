from __future__ import annotations

import copy

import pytest

from tools.adjudicate_unified_intrinsic_resident_transfer import (
    INCONCLUSIVE_CEILING,
    INCONCLUSIVE_INSTRUMENT,
    REFUTED,
    SUPPORTED,
    ResidentTransferAdjudicationError,
    adjudicate_report,
)
from tools.unified_intrinsic_resident_identity import canonical_sha256


def _report(
    *,
    control: int = 1,
    treatment: int = 9,
    trained_t1: int = 3,
    base: int = 0,
    grammar: int = 0,
    pointer: int = 1,
    compiled: int = 9,
    wrong_to_right: int = 8,
    right_to_wrong: int = 0,
) -> dict:
    tasks = 9

    def arm(correct: int) -> dict:
        return {
            "correct": correct,
            "tasks": tasks,
            "accuracy": correct / tasks,
            "eos_stops": tasks,
        }

    body = {
        "schema": "aura.unified_intrinsic_decode_evaluation.v1",
        "checkpoint_sha256": "a" * 64,
        "evaluation_seed": 23,
        "task_count": tasks,
        "task_depths": [1, 2, 4],
        "recurrence_depths": [4],
        "arm_results": {
            "base_t1": arm(base),
            "trained_t1": arm(trained_t1),
            "untrained_t4": arm(control),
            "trained_t4": arm(treatment),
            "grammar_lesion_t4": arm(grammar),
            "pointer_lesion_t4": arm(pointer),
            "compiled_t4": arm(compiled),
        },
        "paired_training_effects": {
            "4": {
                "tasks": tasks,
                "control_arm": "untrained_t4",
                "trained_arm": "trained_t4",
                "untrained_correct": control,
                "trained_correct": treatment,
                "net_correct_gain": wrong_to_right - right_to_wrong,
                "wrong_to_right": wrong_to_right,
                "right_to_wrong": right_to_wrong,
            }
        },
    }
    return {**body, "report_sha256": canonical_sha256(body)}


def test_support_requires_matched_gain_recurrence_and_both_lesions() -> None:
    verdict = adjudicate_report(_report())
    assert verdict["verdict"] == SUPPORTED
    assert verdict["supported"] is True
    assert all(verdict["checks"].values())


def test_control_ceiling_is_inconclusive_not_positive() -> None:
    verdict = adjudicate_report(
        _report(
            control=9,
            treatment=9,
            trained_t1=8,
            wrong_to_right=0,
        )
    )
    assert verdict["verdict"] == INCONCLUSIVE_CEILING


def test_nonexact_compiled_arm_is_an_instrument_failure() -> None:
    verdict = adjudicate_report(_report(compiled=8))
    assert verdict["verdict"] == INCONCLUSIVE_INSTRUMENT


@pytest.mark.parametrize(
    "overrides",
    (
        {"control": 3, "treatment": 3, "wrong_to_right": 0},
        {
            "control": 2,
            "treatment": 2,
            "wrong_to_right": 1,
            "right_to_wrong": 1,
        },
        {"grammar": 9},
        {"pointer": 9},
        {"trained_t1": 9},
    ),
)
def test_missing_causal_requirement_refutes_transfer(overrides: dict) -> None:
    verdict = adjudicate_report(_report(**overrides))
    assert verdict["verdict"] == REFUTED
    assert verdict["supported"] is False


def test_report_commitment_and_transition_accounting_are_recomputed() -> None:
    report = _report()
    report["arm_results"]["trained_t4"]["correct"] = 8
    with pytest.raises(ResidentTransferAdjudicationError, match="commitment"):
        adjudicate_report(report)

    report = _report()
    changed = copy.deepcopy(report)
    changed["paired_training_effects"]["4"]["wrong_to_right"] = 7
    body = {key: value for key, value in changed.items() if key != "report_sha256"}
    changed["report_sha256"] = canonical_sha256(body)
    with pytest.raises(ResidentTransferAdjudicationError, match="accounting"):
        adjudicate_report(changed)
