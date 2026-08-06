"""A φ measured at a post-hoc width may not present itself as evidence.

The 2026-08-04 sweep ran three encoder widths on ONE campaign, 12 scored best,
and 12 became `GRASSMANN_ANCHORS_DEFAULT` — described in the source as "the
measured optimum, not the smallest number that works". Best-of-three is also
the expected shape of noise. The artifact then said, in one section, that the
0.307 fraction was "a real, live, activation-grounded, null-corrected result",
and in another that the same measurement is "not evidence of integration".

Nothing computed the rule, so both could stand. These tests compute it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.consciousness.phi_core import (
    GRASSMANN_ANCHORS_DEFAULT,
    GRASSMANN_ANCHORS_EXACT,
    GRASSMANN_WIDTH_PREREGISTERED,
    GRASSMANN_WIDTH_SELECTION,
    PhiResult,
    grassmann_fold_collision_rate,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parent.parent
PREREGISTRATION = REPO / "artifacts" / "phi" / "PREREGISTRATION_phi_width_replication.json"
ARTIFACT = REPO / "artifacts" / "phi" / "LIVE_NULL_CORRECTED_PHI_2026-08-04.md"


def _result(**overrides) -> PhiResult:
    fields = dict(
        phi_s=0.22373,
        phi_null_mean=0.15499,
        null_p_value=0.01,
        integration_fraction=0.307,
        null_surrogates=8,
        encoder_width=GRASSMANN_ANCHORS_DEFAULT,
        encoder_width_selection=GRASSMANN_WIDTH_SELECTION,
    )
    fields.update(overrides)
    result = PhiResult.__new__(PhiResult)
    for key, value in fields.items():
        object.__setattr__(result, key, value)
    # Defaults the dataclass would otherwise supply.
    for key, value in {
        "grounding": "activation_geometry",
        "node_count": 8,
        "population_size": 12,
        "coverage": None,
        "sampling": "",
        "tpm_n_samples": 507,
        "phi_lower": None,
        "phi_upper": None,
        "interval_method": "",
        "is_subsampled": False,
        "phi_null_sd": 0.01,
    }.items():
        if not hasattr(result, key):
            object.__setattr__(result, key, value)
    return result


class TestTheFoldsCostIsANumber:
    """"12 modes folded into 8 bits" answered with a measurement, not a claim."""

    def test_eight_modes_lose_nothing(self):
        assert grassmann_fold_collision_rate(8) == 0.0

    def test_twelve_modes_collide_almost_entirely(self):
        rate = grassmann_fold_collision_rate(12)
        # 4096 states into 256 buckets.
        assert rate == pytest.approx(1 - 256 / 4096, abs=1e-9)
        assert rate > 0.9

    def test_sixteen_modes_collide_more(self):
        assert grassmann_fold_collision_rate(16) > grassmann_fold_collision_rate(12)

    def test_the_fold_still_uses_every_mode(self):
        """Lossy like a hash, not like a truncation.

        `state & 0xFF` kept modes 0-7 and discarded the rest, so a WIDER
        encoder measured less. Every high mode must still change the byte.
        """
        from core.consciousness.phi_core import _fold_modes_to_byte

        for bit in range(8, 16):
            assert _fold_modes_to_byte(1 << bit) != 0, f"mode {bit} vanished"


class TestWhatAPhiIsAllowedToClaim:
    def test_the_current_default_is_not_preregistered(self):
        assert GRASSMANN_ANCHORS_DEFAULT != GRASSMANN_WIDTH_PREREGISTERED
        assert GRASSMANN_WIDTH_PREREGISTERED == GRASSMANN_ANCHORS_EXACT
        assert GRASSMANN_WIDTH_SELECTION == "exploratory_post_hoc"

    def test_the_headline_result_is_not_citable(self):
        """0.307 with p=0.01 and 8 surrogates — and still exploratory."""
        result = _result()
        assert result.integration_is_significant is True
        assert result.null_surrogates >= 2
        assert result.citable_as_evidence is False, (
            "a width chosen after the sweep that justifies it is not evidence"
        )

    def test_the_same_numbers_at_a_preregistered_width_are_citable(self):
        assert _result(encoder_width_selection="preregistered").citable_as_evidence

    def test_a_result_below_the_floor_is_not_citable_either(self):
        result = _result(
            integration_fraction=0.007, encoder_width_selection="preregistered"
        )
        assert result.integration_is_significant is False
        assert result.citable_as_evidence is False

    def test_too_few_surrogates_is_not_citable(self):
        result = _result(null_surrogates=1, encoder_width_selection="preregistered")
        assert result.citable_as_evidence is False

    def test_the_provenance_carries_the_whole_story(self):
        provenance = _result().provenance()

        assert provenance["encoder_width"] == 12
        assert provenance["encoder_width_selection"] == "exploratory_post_hoc"
        assert provenance["encoder_fold_collision_rate"] == pytest.approx(0.9375)
        assert provenance["citable_as_evidence"] is False

    def test_an_unstamped_result_says_unrecorded_not_nothing(self):
        provenance = _result(
            encoder_width=None, encoder_width_selection=""
        ).provenance()
        assert provenance["encoder_width_selection"] == "unrecorded"
        assert provenance["citable_as_evidence"] is False


class TestTheReplicationIsRegisteredInAdvance:
    def test_the_plan_exists_and_is_unedited(self):
        from core.evaluation.preregistration import load_preregistration

        plan = load_preregistration(PREREGISTRATION)
        assert plan.parameters["encoder_width"] == 12
        assert plan.parameters["encoder_width_selection"] == "preregistered"
        assert plan.metrics["integration_fraction"] >= 0.10

    def test_the_plan_names_both_controls(self):
        plan = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
        arms = set(plan["arms"])
        assert "coupled_ring_positive_control" in arms
        assert "independent_halves_negative_control" in arms

    def test_the_2026_08_04_run_would_not_confirm_it(self):
        """Judged against the plan, the campaign that produced 0.307 is exploratory."""
        from core.evaluation.preregistration import load_preregistration

        plan = load_preregistration(PREREGISTRATION)
        verdict = plan.verify_result(
            {"integration_fraction": 0.307, "null_p_value_inverted": 0.99},
            parameters_used={
                "encoder_width": 12,
                "encoder_width_selection": "exploratory_post_hoc",
                "independent_boots": 1,
            },
        )
        assert verdict["confirms_hypothesis"] is False
        assert verdict["parameter_drift"], "the run's selection differed from the plan"


class TestTheArtifactSaysTheSameThingTwice:
    def test_it_no_longer_calls_the_result_established(self):
        body = ARTIFACT.read_text(encoding="utf-8")
        assert "That is a real, live, activation-grounded" not in body
        assert "exploratory" in body.lower()

    def test_it_still_records_the_retraction(self):
        body = ARTIFACT.read_text(encoding="utf-8")
        assert "RETRACTED" in body

    def test_it_publishes_the_fold_collision_rate(self):
        body = ARTIFACT.read_text(encoding="utf-8")
        assert "0.938" in body
