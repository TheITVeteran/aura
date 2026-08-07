from __future__ import annotations

import copy

import pytest

from core.brain.llm.latent_cortex.paired_campaign import (
    ADAPTER_RLC,
    ADAPTER_VANILLA,
    BASE_RLC,
    BASE_VANILLA,
)
from tools import verify_latent_cortex_directional_gate as gate


def _arm_summary() -> dict[str, dict[str, int]]:
    return {
        BASE_VANILLA: {"correct": 10, "total": 28},
        BASE_RLC: {"correct": 12, "total": 28},
        ADAPTER_VANILLA: {"correct": 10, "total": 28},
        ADAPTER_RLC: {"correct": 16, "total": 28},
    }


def _mechanics() -> dict[str, object]:
    return {
        "ordinary_generation_exact_match": True,
        "raw_terminal_outputs_retained": True,
        "base_recurrence_adapter_activation": {
            "calls": 0,
            "adapted_positions": 0,
            "observed_positions": 0,
        },
        "adapter_recurrence_adapter_activation": {
            "calls": 112,
            "adapted_positions": 896,
            "observed_positions": 1792,
        },
        "causal_first_logit_digest_changes": 28,
    }


def test_directional_rules_require_positive_interaction() -> None:
    rules = gate._evaluate_rules(
        arm_summary=_arm_summary(),
        mechanics=_mechanics(),
        expected_cells=112,
        committed_cells=112,
        replayed_cells=112,
        independent_valid=True,
    )
    assert all(rules.values())

    weak = _arm_summary()
    weak[BASE_RLC]["correct"] = 16
    weak[ADAPTER_RLC]["correct"] = 16
    rules = gate._evaluate_rules(
        arm_summary=weak,
        mechanics=_mechanics(),
        expected_cells=112,
        committed_cells=112,
        replayed_cells=112,
        independent_valid=True,
    )
    assert rules[
        "adapter_rlc_interaction_strictly_exceeds_base_rlc_interaction"
    ] is False


@pytest.mark.parametrize(
    ("mutation", "rule"),
    [
        (
            lambda value: value.update(raw_terminal_outputs_retained=False),
            "raw_terminal_output_policy_is_symmetric_and_unedited",
        ),
        (
            lambda value: value["adapter_recurrence_adapter_activation"].update(calls=0),
            "adapter_rlc_has_positive_scoped_recurrence_adapter_activity",
        ),
        (
            lambda value: value["base_recurrence_adapter_activation"].update(calls=1),
            "base_rlc_has_zero_recurrence_adapter_activity",
        ),
    ],
)
def test_directional_rules_fail_closed_on_mechanics(mutation, rule: str) -> None:
    mechanics = copy.deepcopy(_mechanics())
    mutation(mechanics)
    rules = gate._evaluate_rules(
        arm_summary=_arm_summary(),
        mechanics=mechanics,
        expected_cells=112,
        committed_cells=112,
        replayed_cells=112,
        independent_valid=True,
    )
    assert rules[rule] is False


def test_replacement_receipt_must_retain_the_raw_baseline() -> None:
    receipt = {
        "answer_replacement": {
            "answer_selection_effect": "retained",
            "accepted_output": {
                "source": "baseline_decode",
                "text_sha256": "a" * 64,
                "tokens_sha256": "b" * 64,
            },
            "baseline_decode": {
                "text_sha256": "a" * 64,
                "tokens_sha256": "b" * 64,
            },
        }
    }
    assert gate._replacement_retained(receipt) is True
    receipt["answer_replacement"]["answer_selection_effect"] = "replaced"
    assert gate._replacement_retained(receipt) is False


def test_independent_evidence_is_a_required_rule() -> None:
    rules = gate._evaluate_rules(
        arm_summary=_arm_summary(),
        mechanics=_mechanics(),
        expected_cells=112,
        committed_cells=112,
        replayed_cells=112,
        independent_valid=False,
    )
    assert rules["independent_evidence_and_receipts_validate"] is False


def test_directional_verdict_write_is_create_or_verify(tmp_path) -> None:
    path = tmp_path / "verdict.json"
    document = {"schema": "fixture", "value": 1}
    gate._write_once(path, document)
    gate._write_once(path, document)
    with pytest.raises(gate.DirectionalGateError, match="output_conflict"):
        gate._write_once(path, {"schema": "fixture", "value": 2})
