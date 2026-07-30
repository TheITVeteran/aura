"""Training must not optimise a component that cannot fix the failure.

Seven consecutive resident recurrent-GRPO campaigns (cp259, 271, 273, 285,
291, 294, 305) ran with ``adapter_scope = latent_slots_only`` against
failures that were entirely decode-path output-contract failures — the model
burning its whole token budget without ever emitting the answer marker. Each
spent ~86 minutes optimising parameters with no causal path to the measured
failure, so every reward was zero, every group degenerate, every advantage a
vector of zeros, and no gradient existed.

No learning rate, group size or reward shaping fixes that, and the optimizer
cannot see it: from inside, a futile objective and a hard one look the same.
"""
from __future__ import annotations

from core.learning import scope_reachability as sr


class TestTheRecordedCampaignShapeIsRefused:
    def _cp305_reasons(self):
        # Verbatim from the cp305 receipt.
        return sr.merge_reason_counts(
            {"unparseable": 36},
            {"no_marker": 32, "marker_line_has_no_object": 4},
        )

    def test_the_campaign_that_burned_seven_runs_is_refused(self):
        verdict = sr.assess(self._cp305_reasons(), adapter_scope="latent_slots_only")
        assert verdict.verdict == sr.UNREACHABLE
        assert verdict.should_refuse is True

    def test_the_refusal_names_the_site_and_the_scope(self):
        verdict = sr.assess(self._cp305_reasons(), adapter_scope="latent_slots_only")
        assert "decode_path" in verdict.detail
        assert "latent_slots_only" in verdict.detail
        assert verdict.out_of_scope == 72
        assert verdict.in_scope == 0

    def test_a_scope_that_reaches_the_decode_path_is_permitted(self):
        """The same failures are learnable if the weights can reach them."""
        for scope in ("full_model", "coda_and_late_layers", "decode_path"):
            verdict = sr.assess(self._cp305_reasons(), adapter_scope=scope)
            assert verdict.should_refuse is False, scope
            assert verdict.verdict == sr.REACHABLE, scope


class TestReasoningFailuresRemainLearnable:
    def test_wrong_but_well_formed_answers_are_in_scope(self):
        """A wrong answer in the right shape is what the slots can fix."""
        verdict = sr.assess(
            {"incorrect": 30, "wrong_answer": 10}, adapter_scope="latent_slots_only",
        )
        assert verdict.verdict == sr.REACHABLE
        assert verdict.should_refuse is False

    def test_a_mixed_failure_profile_is_not_refused(self):
        """Any in-scope signal is enough to learn from; do not block it."""
        verdict = sr.assess(
            {"incorrect": 8, "no_marker": 40}, adapter_scope="latent_slots_only",
        )
        assert verdict.verdict == sr.REACHABLE
        assert verdict.in_scope == 8


class TestTheGuardIsConservative:
    def test_a_small_sample_is_unknown_not_refused(self):
        verdict = sr.assess({"no_marker": 3}, adapter_scope="latent_slots_only")
        assert verdict.verdict == sr.UNKNOWN
        assert verdict.should_refuse is False

    def test_unrecognized_reasons_do_not_refuse(self):
        """A reason nobody mapped is not evidence of anything."""
        verdict = sr.assess(
            {"some_new_reason": 50}, adapter_scope="latent_slots_only",
        )
        assert verdict.verdict == sr.UNKNOWN
        assert verdict.should_refuse is False
        assert verdict.unrecognized == {"some_new_reason": 50}

    def test_an_undeclared_scope_is_unknown_not_omnipotent(self):
        """Assuming reach is how this defect arises; never assume it."""
        verdict = sr.assess(
            {"no_marker": 50}, adapter_scope="some_future_scope",
        )
        assert verdict.verdict == sr.UNKNOWN
        assert verdict.should_refuse is False

    def test_no_observations_is_unknown(self):
        assert sr.assess({}, adapter_scope="latent_slots_only").verdict == sr.UNKNOWN
        assert sr.assess(None, adapter_scope="latent_slots_only").verdict == sr.UNKNOWN

    def test_every_failure_being_unfixable_by_this_scope_does_refuse(self):
        """Decode-path AND task-side failures are both beyond the slots."""
        verdict = sr.assess(
            {"no_marker": 10, "task_malformed": 10}, adapter_scope="latent_slots_only",
        )
        assert verdict.should_refuse is True

    def test_a_mostly_unclassifiable_profile_abstains(self):
        """The threshold is measured against ALL observations.

        Dividing by the attributable subset would make the ratio 1.0
        whenever nothing is in scope, so the threshold would never bind and
        a half-unrecognized profile would be refused on the strength of the
        half we happen to understand.
        """
        verdict = sr.assess(
            {"no_marker": 20, "brand_new_reason": 40},
            adapter_scope="latent_slots_only",
        )
        assert verdict.verdict == sr.UNKNOWN
        assert verdict.should_refuse is False
        assert verdict.unrecognized == {"brand_new_reason": 40}


class TestCountHygiene:
    def test_non_positive_and_non_integer_counts_are_ignored(self):
        verdict = sr.assess(
            {"no_marker": 0, "unparseable": -5, "incorrect": True, "token_limit": 20},
            adapter_scope="latent_slots_only",
        )
        assert verdict.observations == 20

    def test_merge_combines_and_drops_junk(self):
        merged = sr.merge_reason_counts(
            {"a": 1}, {"a": 2, "b": 3}, None, {"c": 0}, {"d": True},
        )
        assert merged == {"a": 3, "b": 3}

    def test_the_verdict_is_serializable(self):
        payload = sr.assess(
            {"no_marker": 40}, adapter_scope="latent_slots_only",
        ).to_dict()
        assert payload["schema"] == sr.SCHEMA
        assert payload["verdict"] == sr.UNREACHABLE
        assert payload["by_site"]["decode_path"] == 40


class TestTheMapsAreCoherent:
    def test_every_failure_site_is_a_known_site(self):
        known = {
            sr.SITE_RECURRENT_SLOTS, sr.SITE_DECODE, sr.SITE_PROMPT, sr.SITE_UNKNOWN,
        }
        for reason, site in sr.FAILURE_SITES.items():
            assert site in known, reason

    def test_every_scope_reaches_known_sites(self):
        known = {sr.SITE_RECURRENT_SLOTS, sr.SITE_DECODE, sr.SITE_PROMPT}
        for scope, sites in sr.SCOPE_REACHES.items():
            assert sites, scope
            assert set(sites) <= known, scope

    def test_the_contract_reasons_the_runtime_emits_are_mapped(self):
        """These identifiers come from answer_contract.py and
        verifiable_tasks.py, not from this module's imagination."""
        for reason in ("no_marker", "marker_line_has_no_object", "unparseable"):
            assert reason in sr.FAILURE_SITES

    def test_latent_slots_only_cannot_reach_decode(self):
        """The specific fact that made seven campaigns futile."""
        assert sr.SITE_DECODE not in sr.SCOPE_REACHES["latent_slots_only"]


class TestTheTrainerConsultsTheGuard:
    def _trainer_source(self) -> str:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        return (root / "tools" / "train_grpo.py").read_text(encoding="utf-8")

    def test_the_guard_runs_in_the_trainer(self):
        source = self._trainer_source()
        assert "scope_reachability" in source
        assert "scope_unreachable" in source

    def test_it_runs_before_calibration(self):
        """Calibration alone costs over an hour; the check must precede it."""
        source = self._trainer_source()
        assert source.index("scope_unreachable") < source.index(
            "if args.calibrate and calibration is None and training_allowed:"
        )

    def test_calibration_cannot_override_a_scope_refusal(self):
        source = self._trainer_source()
        assert "if args.calibrate and calibration is None and training_allowed:" in source
        assert (
            "training_allowed = training_allowed and bool(" in source
        )

    def test_the_verdict_is_recorded_on_the_baseline(self):
        source = self._trainer_source()
        assert 'baseline_eval["scope_reachability"]' in source


class TestARefusalNamesTheRemedy:
    """A refusal that does not say what would work is only half a finding."""

    def test_a_decode_path_refusal_points_at_distillation(self):
        verdict = sr.assess(
            {"no_marker": 40}, adapter_scope="latent_slots_only",
        )
        assert verdict.should_refuse is True
        assert "decode path" in verdict.remedy
        assert "latent_adapter_distillation" in verdict.remedy

    def test_a_task_side_refusal_says_no_weight_update_helps(self):
        verdict = sr.assess(
            {"task_malformed": 40}, adapter_scope="latent_slots_only",
        )
        assert verdict.should_refuse is True
        assert "no weight update" in verdict.remedy

    def test_a_permitted_run_carries_no_remedy(self):
        verdict = sr.assess({"incorrect": 40}, adapter_scope="latent_slots_only")
        assert verdict.should_refuse is False
        assert verdict.remedy == ""

    def test_the_remedy_is_serialized(self):
        payload = sr.assess(
            {"no_marker": 40}, adapter_scope="latent_slots_only",
        ).to_dict()
        assert payload["remedy"]

    def test_the_trainer_prints_the_remedy(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        source = (root / "tools" / "train_grpo.py").read_text(encoding="utf-8")
        assert "[halt] remedy:" in source
