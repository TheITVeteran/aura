from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from tools import verify_resident_pilot_result as verifier


def _inputs() -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    arms: dict[str, dict[str, object]] = {
        "base_vanilla": {"correct": 2},
        "base_rlc": {"correct": 3},
        "adapter_vanilla": {"correct": 2},
        "adapter_rlc": {"correct": 4},
    }
    mechanics: dict[str, object] = {
        "ordinary_generation_exact_match": True,
        "base_recurrence_adapter_activation": {
            "calls": 0,
            "adapted_positions": 0,
            "observed_positions": 0,
        },
        "adapter_recurrence_adapter_activation": {
            "calls": 2,
            "adapted_positions": 32,
            "observed_positions": 32,
        },
        "causal_first_logit_digest_changes": 1,
    }
    return arms, mechanics


def _rules(
    arms: dict[str, dict[str, object]],
    mechanics: dict[str, object],
    *,
    committed: int = 56,
    replayed: int = 56,
    receipts: bool = True,
) -> dict[str, bool]:
    return verifier._evaluate_advance_rules(
        arm_summary=arms,
        mechanics=mechanics,
        committed_cells=committed,
        replayed_cells=replayed,
        receipts_valid=receipts,
    )


def test_all_preregistered_advance_rules_can_pass() -> None:
    arms, mechanics = _inputs()

    assert _rules(arms, mechanics) == {rule: True for rule in verifier.EXPECTED_RULES}


@pytest.mark.parametrize("failed_rule", verifier.EXPECTED_RULES)
def test_each_preregistered_advance_rule_fails_independently(
    failed_rule: str,
) -> None:
    arms, mechanics = _inputs()
    committed = replayed = 56
    receipts = True
    if failed_rule == verifier.EXPECTED_RULES[0]:
        replayed = 55
    elif failed_rule == verifier.EXPECTED_RULES[1]:
        arms["base_rlc"]["correct"] = 1
        arms["adapter_vanilla"]["correct"] = 2
        arms["adapter_rlc"]["correct"] = 2
    elif failed_rule == verifier.EXPECTED_RULES[2]:
        arms["base_vanilla"]["correct"] = 1
        arms["base_rlc"]["correct"] = 2
        arms["adapter_vanilla"]["correct"] = 1
        arms["adapter_rlc"]["correct"] = 2
    elif failed_rule == verifier.EXPECTED_RULES[3]:
        mechanics["ordinary_generation_exact_match"] = False
    elif failed_rule == verifier.EXPECTED_RULES[4]:
        arms["adapter_vanilla"]["correct"] = 1
    elif failed_rule == verifier.EXPECTED_RULES[5]:
        mechanics["base_recurrence_adapter_activation"] = {
            "calls": 1,
            "adapted_positions": 16,
            "observed_positions": 16,
        }
    elif failed_rule == verifier.EXPECTED_RULES[6]:
        mechanics["adapter_recurrence_adapter_activation"] = {
            "calls": 0,
            "adapted_positions": 0,
            "observed_positions": 0,
        }
    elif failed_rule == verifier.EXPECTED_RULES[7]:
        mechanics["causal_first_logit_digest_changes"] = 0
    else:
        receipts = False

    observed = _rules(
        arms,
        mechanics,
        committed=committed,
        replayed=replayed,
        receipts=receipts,
    )

    assert observed[failed_rule] is False
    assert all(value for rule, value in observed.items() if rule != failed_rule)


def test_independent_evidence_rejects_rehashed_semantic_substitution() -> None:
    evidence = {
        "passed": True,
        "failures": [],
        "committed_records": 56,
        "task_count": 14,
        "production_semantic_grade_sha256": "a" * 64,
        "independent_semantic_grade_sha256": "b" * 64,
        "published_verdict": "incomplete_underpowered",
        "recomputed_verdict": "incomplete_underpowered",
        "independent_verdict": "incomplete_underpowered",
    }

    with pytest.raises(
        verifier.ResidentPilotResultError,
        match="independent_campaign_evidence_invalid",
    ):
        verifier._validate_independent_evidence(evidence)


def test_mechanics_gate_rejects_rehashed_reasoning_claim(tmp_path: Path) -> None:
    material: dict[str, object] = {
        "passed": True,
        "ready_for_fresh_hidden_task_pilot": True,
        "reasoning_gain_proven": True,
        "frontier_gain_proven": False,
    }
    document = {**material, "verdict_sha256": verifier._sha(material)}
    path = tmp_path / "mechanics.json"
    path.write_bytes(canonical_json_bytes(document) + b"\n")
    contract = {
        "mechanics_gate": {
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "verdict_sha256": document["verdict_sha256"],
        }
    }

    with pytest.raises(
        verifier.ResidentPilotResultError,
        match="pilot_mechanics_gate_invalid",
    ):
        verifier._verified_mechanics(path, contract)


def test_diagnosis_detects_contract_failure_and_branch_collapse() -> None:
    arms = {
        "base_vanilla": {
            "correct": 1,
            "score_reasons": {"final_answer_marker_count_invalid": 13},
        },
        "base_rlc": {
            "correct": 2,
            "score_reasons": {"final_answer_marker_count_invalid": 12},
        },
        "adapter_vanilla": {
            "correct": 1,
            "score_reasons": {"final_answer_marker_count_invalid": 13},
        },
        "adapter_rlc": {
            "correct": 1,
            "total": 14,
            "score_reasons": {"final_answer_invalid_json": 5},
            "branch_score_ties": 14,
            "selected_branches": {"0": 14},
            "branch_score_spread_minmax": [0.0, 0.0],
        },
    }
    rules = {rule: True for rule in verifier.EXPECTED_RULES}
    rules[verifier.EXPECTED_RULES[1]] = False
    rules[verifier.EXPECTED_RULES[2]] = False

    diagnoses = verifier._diagnoses(
        deepcopy(arms),
        {"ordinary_generation_exact_match": True},
        rules,
    )

    assert diagnoses == [
        "decode_response_contract_failure",
        "adapter_virtual_width_functionally_collapsed",
        "recurrence_training_failed_directional_gain_gate",
        "shared_vanilla_decode_budget_truncates_contract_answers",
    ]
