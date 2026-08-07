from __future__ import annotations

import copy

import pytest

from core.brain.llm.latent_cortex.paired_campaign import (
    ADAPTER_EQUAL_COMPUTE,
    BASE_EQUAL_COMPUTE,
)
from tools import materialize_powered_latent_cortex_handoff as handoff
from tools import verify_latent_cortex_directional_gate as directional


def _directional() -> dict:
    material = {
        "schema": directional.SCHEMA,
        "passed": True,
        "evidence_valid": True,
        "directional_gate_passed": True,
        "decision": "advance_to_powered_external_campaign",
        "required_next_gate": "powered_external_campaign",
        "reasoning_gain_proven": False,
        "frontier_gain_proven": False,
        "production_activation_authorized": False,
        "static_weight_fusion_authorized": False,
        "advance_rules": {rule: True for rule in directional.EXPECTED_RULES},
    }
    return {**material, "verdict_sha256": directional._sha(material)}


def test_powered_design_is_exact_six_arm_floor() -> None:
    power = handoff._powered_design()
    assert power["minimum_observations"] == 411
    assert power["planned_total_tasks"] == 2_877
    assert power["planned_total_cells"] == 17_262
    assert power["powered_for_zero_loss_noninferiority"] is True


def test_positive_directional_certificate_is_required() -> None:
    verdict = _directional()
    assert handoff._verified_directional_verdict(verdict) == verdict
    verdict = copy.deepcopy(verdict)
    verdict["advance_rules"][directional.EXPECTED_RULES[3]] = False
    material = {key: value for key, value in verdict.items() if key != "verdict_sha256"}
    verdict["verdict_sha256"] = directional._sha(material)
    with pytest.raises(handoff.PoweredHandoffError, match="positive_directional"):
        handoff._verified_directional_verdict(verdict)


def test_inherited_execution_contract_preserves_rlc_shape() -> None:
    execution = {
        "n_slots": 16,
        "branches": 2,
        "rlc_steps": 4,
        "rlc_profile": "recurrence_attribution",
        "decode_max_tokens": 320,
        "difficulty": 2,
        "task_registry_version": "2026.08.06.1",
        "equal_compute_max_samples": 8,
        "requested_rlc_shape": {"n_slots": 16, "branches": 2, "rlc_steps": 4},
        "response_contract_policy": {
            "causal_attribution_rule": "raw_terminal_decode_all_arms"
        },
        "effective_rlc_config": {"answer_replacement_enabled": False},
        "adapter_execution_spec": {"recurrent_steps": 4},
    }
    inherited = handoff._inherited_execution_contract(execution)
    assert inherited["n_slots"] == 16
    assert inherited["branches"] == 2
    assert inherited["rlc_steps"] == 4


def test_powered_arms_include_equal_compute_controls() -> None:
    assert BASE_EQUAL_COMPUTE in handoff.POWERED_ARMS
    assert ADAPTER_EQUAL_COMPUTE in handoff.POWERED_ARMS
    assert len(handoff.POWERED_ARMS) == 6
