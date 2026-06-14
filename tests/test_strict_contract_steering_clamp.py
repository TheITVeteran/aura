"""tests/test_strict_contract_steering_clamp.py
================================================
Strict/structured proof generations must run with affective steering driven
near-off. Full steering (alpha 5.0) on a constrained strict-contract generation
corrupts the first-token logits → zero-token generation that hangs to the 90s
first-token timeout (the intermittent DNU cortex wedge: R011/R040/R022).

These pin that strict contracts are covered by the surface-steering clamp and
get a near-off alpha, while normal conversational turns keep full steering.
"""
from __future__ import annotations

from core.brain.llm.mlx_worker import (
    _surface_control_alpha,
    _surface_generation_contract_enabled,
)


def test_strict_contracts_enable_surface_clamp():
    assert _surface_generation_contract_enabled({"strict_answer_contract": True})
    assert _surface_generation_contract_enabled({"strict_value_contract": True})
    assert _surface_generation_contract_enabled({"proof_evaluation_contract": True})


def test_existing_contracts_still_clamped():
    assert _surface_generation_contract_enabled({"clean_user_surface_contract": True})
    assert _surface_generation_contract_enabled({"health_probe": True})
    assert _surface_generation_contract_enabled({"operator_evidence_contract": True})


def test_ordinary_turn_not_clamped():
    assert _surface_generation_contract_enabled({}) is False
    assert _surface_generation_contract_enabled({"foo": True}) is False


def test_strict_contract_steering_is_near_off():
    # current_alpha = 5.0 (full bootstrap steering); strict contract must clamp
    # it to near-off (~0.08), well below the value that corrupts proof logits.
    alpha = _surface_control_alpha({"strict_answer_contract": True}, 5.0)
    assert alpha <= 0.1
    alpha_v = _surface_control_alpha({"strict_value_contract": True}, 5.0)
    assert alpha_v <= 0.1


def test_operator_evidence_and_prose_alphas_unchanged():
    assert _surface_control_alpha({"operator_evidence_contract": True}, 5.0) <= 0.12 + 1e-9
    # Ordinary user-visible prose keeps the moderate 0.35 clamp.
    prose = _surface_control_alpha({"clean_user_surface_contract": True}, 5.0)
    assert 0.3 <= prose <= 0.35 + 1e-9
