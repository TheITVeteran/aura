"""Live-mind contract: proofs that passed without proving anything."""
from __future__ import annotations

import hashlib

import pytest

from core.brain.live_mind_contract import (
    live_mind_generation_controls_present,
    verify_text_mutation_chain,
)

pytestmark = pytest.mark.unit


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _valid_controls(**overrides):
    controls = {
        "temperature": 0.7,
        "top_p": 0.95,
        "clean_user_surface_recurrent_loops": 2,
        "clean_user_surface_steering_alpha": 0.22,
    }
    controls.update(overrides)
    return controls


# ── no-change proof must consult the ledger ────────────────────────────────


def test_equal_hashes_with_an_empty_ledger_pass():
    digest = _sha("delivered")

    result = verify_text_mutation_chain([], before_sha256=digest, after_sha256=digest)

    assert result["passed"] is True
    assert result["chain_length"] == 0


def test_equal_hashes_cannot_override_a_contradicting_ledger():
    """'Nothing changed' is a claim ABOUT the ledger, so it must be checked
    against it. This returned passed without examining the entries at all, so a
    recorded post-certification mutation still produced a clean no-change
    proof — exactly what the proof exists to catch."""
    certified = _sha("certified output")
    mutated = _sha("mutated output")
    ledger = [{
        "event_id": "evt-1",
        "sequence": 1,
        "stage": "post_certification",
        "method": "surface_rewrite",
        "before_sha256": certified,
        "after_sha256": mutated,
    }]

    result = verify_text_mutation_chain(
        ledger, before_sha256=certified, after_sha256=certified
    )

    assert result["passed"] is False
    assert "no_change_claim_contradicted_by_ledger" in result["reasons"]


def test_equal_hashes_still_allow_legitimate_prefix_provenance():
    """Entries BEFORE the certified hash are legitimate worker-side history and
    must not invalidate a genuine no-change delivery."""
    earlier = _sha("draft")
    certified = _sha("certified output")
    ledger = [{
        "event_id": "evt-0",
        "sequence": 1,
        "stage": "pre_certification",
        "method": "worker_cleanup",
        "before_sha256": earlier,
        "after_sha256": certified,
    }]

    result = verify_text_mutation_chain(
        ledger, before_sha256=certified, after_sha256=certified
    )

    assert result["passed"] is True


# ── controls must be usable, not merely present ────────────────────────────


def test_valid_controls_are_accepted():
    assert live_mind_generation_controls_present(_valid_controls()) is True


@pytest.mark.parametrize("override", [
    {"temperature": None},
    {"temperature": float("nan")},
    {"temperature": "0.7"},
    {"temperature": 99.0},
    {"top_p": 0.0},
    {"top_p": float("inf")},
    {"clean_user_surface_steering_alpha": -0.5},
    {"clean_user_surface_steering_alpha": 1.5},
    {"clean_user_surface_recurrent_loops": 0},
    {"clean_user_surface_recurrent_loops": 2.5},
    {"clean_user_surface_recurrent_loops": True},
    {"clean_user_surface_recurrent_loops": 10_000},
])
def test_unusable_control_values_are_not_counted_as_present(override):
    """Key presence was the whole test, so nulls, NaN, strings and
    out-of-range numbers counted as 'controls structurally present' — a proof
    of configuration that proved nothing about whether generation could
    actually be steered."""
    assert live_mind_generation_controls_present(_valid_controls(**override)) is False


def test_missing_keys_are_still_rejected():
    partial = _valid_controls()
    partial.pop("top_p")

    assert live_mind_generation_controls_present(partial) is False


def test_non_mapping_is_rejected():
    assert live_mind_generation_controls_present(None) is False
    assert live_mind_generation_controls_present(["temperature"]) is False
