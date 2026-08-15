"""CP126 596c9811 + 9e810491: the sample has to be able to see the effect.

Two gaps sat under the same certificate. Adequacy was decided by a trial
COUNT fixed before any trial was excluded, so a bundle that discarded half its
trials for contamination kept its claim to being adequately powered. And run
order was checked only for global balance over ALL trials, including rejected
ones — which answers "did both orders run?" and never "did order matter?".

These tests pin the replacements: achieved power recomputed from the trials
that survived admission, per-domain balance, and a measured order effect.
"""
from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.frontier_certification import (
    _binomial_sf,
    _exact_mcnemar_power,
)
from tests.fixtures.latent_frontier import _bundle, _certify, _refresh_task_commitment


class TestExactPower:
    def test_survival_function_matches_hand_computed_values(self):
        assert _binomial_sf(0, 5, 0.5) == pytest.approx(1.0)
        assert _binomial_sf(6, 5, 0.5) == pytest.approx(0.0)
        assert _binomial_sf(5, 5, 0.5) == pytest.approx(1 / 32)
        assert _binomial_sf(4, 5, 0.5) == pytest.approx(6 / 32)

    def test_no_discordant_pairs_is_no_power(self):
        """Concordant pairs carry no information for McNemar."""
        assert _exact_mcnemar_power(0, 0.05, 0.9) == 0.0

    def test_a_sample_too_small_to_ever_reject_has_zero_power(self):
        """With 4 discordant pairs the smallest attainable p-value is 1/16."""
        assert _binomial_sf(4, 4, 0.5) == pytest.approx(1 / 16)
        assert _exact_mcnemar_power(4, 0.05, 1.0) == 0.0

    def test_power_rises_with_the_preregistered_alternative(self):
        weak = _exact_mcnemar_power(30, 0.05, 0.6)
        strong = _exact_mcnemar_power(30, 0.05, 0.9)
        assert 0.0 < weak < strong <= 1.0

    def test_power_rises_with_surviving_sample_size(self):
        small = _exact_mcnemar_power(12, 0.05, 0.75)
        large = _exact_mcnemar_power(40, 0.05, 0.75)
        assert small < large

    def test_the_fixture_is_powered_for_what_it_preregistered(self):
        assert _exact_mcnemar_power(30, 0.05, 0.75) >= 0.8


class TestPreregisteredPowerParameters:
    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("target_power", None, "invalid_target_power"),
            ("target_power", 0.5, "invalid_target_power"),
            ("target_power", 1.0, "invalid_target_power"),
            ("preregistered_discordant_win_share", None, "invalid_discordant_win_share"),
            ("preregistered_discordant_win_share", 0.5, "invalid_discordant_win_share"),
            ("preregistered_discordant_win_share", 1.1, "invalid_discordant_win_share"),
            ("max_order_effect", None, "invalid_max_order_effect"),
            ("max_order_effect", 0.0, "invalid_max_order_effect"),
            ("max_order_effect", 0.9, "invalid_max_order_effect"),
        ],
    )
    def test_each_parameter_is_required_and_bounded(self, field, value, expected):
        bundle = _bundle()
        if value is None:
            del bundle["preregistration"][field]
        else:
            bundle["preregistration"][field] = value
        certificate = _certify(bundle)
        assert expected in certificate["reasons"]
        assert certificate["accepted"] is False


class TestAchievedPower:
    def test_the_certificate_reports_power_per_domain(self):
        certificate = _certify(_bundle())
        assert certificate["accepted"] is True, certificate["reasons"]
        power = certificate["achieved_power_by_domain"]
        assert set(power) == {"math", "coding", "science"}
        assert all(value >= certificate["target_power"] for value in power.values())

    def test_exclusions_can_drop_a_domain_below_its_target(self):
        """The count is preregistered; what survives admission is not.

        Every math trial here keeps its preregistered trial count in the
        bundle, but all but a handful are excluded, so the surviving sample
        can no longer see the effect the study was sized for.
        """
        bundle = _bundle()
        excluded = 0
        for trial in bundle["trials"]:
            if trial["domain"] != "math":
                continue
            excluded += 1
            if excluded > 36:
                break
            trial["verifier_blinded"] = False
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "math:achieved_power_below_target" in certificate["reasons"]
        assert certificate["achieved_power_by_domain"]["math"] < 0.8
        # The other domains kept theirs: the verdict is per-domain.
        assert certificate["achieved_power_by_domain"]["coding"] >= 0.8

    def test_a_domain_of_pure_agreement_has_no_power_at_any_count(self):
        """40 trials where the arms never disagree measure nothing."""
        bundle = _bundle()
        for trial in bundle["trials"]:
            if trial["domain"] == "science":
                trial["control_success"] = True
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["achieved_power_by_domain"]["science"] == 0.0
        assert "science:achieved_power_below_target" in certificate["reasons"]


class TestRunOrder:
    def test_a_single_lopsided_domain_is_caught(self):
        """Balanced overall, rigged in one domain."""
        bundle = _bundle()
        for trial in bundle["trials"]:
            if trial["domain"] == "coding":
                trial["run_order"] = "treatment_first"
            elif trial["domain"] == "science":
                trial["run_order"] = "control_first"
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert "coding:run_order_imbalanced" in certificate["reasons"]
        assert "science:run_order_imbalanced" in certificate["reasons"]
        # Overall the two rigged domains cancel, which is exactly what the
        # old global-only check would have called balanced.
        assert "run_order_imbalanced" not in certificate["reasons"]

    def test_balance_is_counted_over_admitted_trials_only(self):
        """Rejected trials must not be able to balance a run.

        Every control-first trial here is excluded for an unrelated defect,
        so the admitted sample is entirely treatment-first even though the
        bundle's own trial list looks balanced.
        """
        bundle = _bundle()
        for trial in bundle["trials"]:
            if trial["run_order"] == "control_first":
                trial["verifier_blinded"] = False
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "run_order_imbalanced" in certificate["reasons"]
        for domain, counts in certificate["order_balance_by_domain"].items():
            assert counts["control_first"] == 0, domain

    def test_an_order_effect_larger_than_preregistered_is_refused(self):
        """The gain lives in one order and vanishes in the other.

        Balance is satisfied — both orders ran, equally often. The paired
        difference is 1.0 when the treatment went first and 0.0 when it went
        second, which makes order a rival explanation for the entire result.
        """
        bundle = _bundle()
        for trial in bundle["trials"]:
            trial["control_success"] = trial["run_order"] == "control_first"
            trial["treatment_success"] = True
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "order_effect_exceeds_preregistered_maximum" in certificate["reasons"]
        assert certificate["measured_order_effect"] == pytest.approx(1.0)

    def test_the_fixture_design_is_orthogonal_to_order(self):
        certificate = _certify(_bundle())
        assert certificate["measured_order_effect"] == pytest.approx(0.0)
        assert "order_effect_unmeasurable" not in certificate["reasons"]
