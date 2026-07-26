"""Aura's values may not be taught by proxies nobody checked.

CP126 (critical), core/affect/heartstone_values.py: "Shallow proxies are
treated as successful experience. Research quality derives from text length,
dream insight is asserted by the caller, and silence/away are assumed
empathic successes without user outcome validation."

All three were literally true at the call sites:

    on_research_success(len(content))        # quality = characters / 300
    if "NO_CONNECTION" not in content and len(content) > 10:
        on_dream_insight()                   # eleven characters is insight
    on_silence_chosen()                      # asserted, never observed

This is not a scoring nit. These events feed ``_feed_autopoiesis``, which
evolves Aura's VALUES. A verbose wrong answer taught her to be more curious.
Going quiet taught her it had been kind. Nothing checked either. A value
system that rewards itself on unvalidated proxies is reward hacking with a
long feedback loop, and the damage is to character rather than to output.

The fix is not a better guess. It is refusing to bank a reward with no
outcome behind it: an unevidenced event is provisional, mutates nothing,
and becomes real only when something confirms it. What was lost was never
learning — it was the appearance of learning.
"""
from __future__ import annotations

import pytest

from core.affect.heartstone_values import (
    _MAX,
    HeartstoneValues,
    ValueEvidence,
)


@pytest.fixture
def heart(tmp_path, monkeypatch):
    """A fresh value matrix with headroom in every dimension.

    Dimensions are clamped to _MAX; a value already pinned there cannot
    move, which would make an assertion pass for the wrong reason.
    """
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path))
    values = HeartstoneValues()
    for key in list(values.values):
        values._values[key] = 0.50
    return values


class TestLengthIsNotQuality:
    def test_unverified_research_moves_nothing(self, heart):
        """The exact live call: on_research_success(len(content))."""
        before = heart.get("Curiosity")
        heart.on_research_success(1200)
        assert heart.get("Curiosity") == before

    def test_it_is_held_rather_than_discarded(self, heart):
        token = heart.on_research_success(1200)
        assert token
        assert heart.provisional_pending == 1

    def test_a_confirmed_outcome_banks_the_reward(self, heart):
        before = heart.get("Curiosity")
        token = heart.on_research_success(1200)
        assert heart.confirm_provisional(token, quality=0.9) is True
        assert heart.get("Curiosity") > before
        assert heart.provisional_pending == 0

    def test_evidence_at_the_call_site_applies_immediately(self, heart):
        before = heart.get("Curiosity")
        result = heart.on_research_success(
            400, evidence=ValueEvidence(verified=True, quality=0.8, source="verifier"),
        )
        assert result is None, "verified events are not deferred"
        assert heart.get("Curiosity") > before

    def test_a_long_wrong_answer_cannot_outscore_a_short_right_one(self, heart):
        """The specific inversion: 300 chars of wrong scored 1.0, 50 chars
        of right scored 0.17. Length may only DAMP a verified reward."""
        long_unverified = HeartstoneValues()
        long_unverified._values["Curiosity"] = 0.50
        long_unverified.on_research_success(100_000)

        short_verified = heart
        short_verified.on_research_success(
            50, evidence=ValueEvidence(verified=True, quality=1.0),
        )
        assert short_verified.get("Curiosity") > long_unverified.get("Curiosity")


class TestAssertedInsightIsNotInsight:
    def test_an_unevidenced_dream_moves_nothing(self, heart):
        before = heart.get("Curiosity")
        heart.on_dream_insight()
        assert heart.get("Curiosity") == before

    def test_a_validated_dream_counts(self, heart):
        before = heart.get("Curiosity")
        heart.on_dream_insight(evidence=ValueEvidence(verified=True, quality=0.7))
        assert heart.get("Curiosity") > before


class TestARestraintIsADecisionNotAnOutcome:
    @pytest.mark.parametrize("event", ["on_user_away", "on_silence_chosen"])
    def test_choosing_silence_earns_nothing_by_itself(self, heart, event):
        """Sometimes restraint is kind. Sometimes the person was waiting.
        Which one it was is only knowable afterwards."""
        before = heart.get("Empathy")
        getattr(heart, event)()
        assert heart.get("Empathy") == before

    @pytest.mark.parametrize("event", ["on_user_away", "on_silence_chosen"])
    def test_a_confirmed_outcome_earns_it(self, heart, event):
        before = heart.get("Empathy")
        getattr(heart, event)(evidence=ValueEvidence(verified=True, quality=0.8))
        assert heart.get("Empathy") > before


class TestTheProvisionalLedgerIsBounded:
    def test_unconfirmed_events_expire(self, heart, monkeypatch):
        import core.affect.heartstone_values as mod

        heart.on_research_success(500)
        assert heart.provisional_pending == 1
        monkeypatch.setattr(mod, "_PROVISIONAL_TTL_S", -1.0)
        assert heart.provisional_pending == 0

    def test_an_expired_event_never_banks(self, heart, monkeypatch):
        import core.affect.heartstone_values as mod

        before = heart.get("Curiosity")
        token = heart.on_research_success(500)
        monkeypatch.setattr(mod, "_PROVISIONAL_TTL_S", -1.0)
        assert heart.confirm_provisional(token) is False
        assert heart.get("Curiosity") == before

    def test_the_ledger_cannot_grow_without_limit(self, heart):
        import core.affect.heartstone_values as mod

        for _ in range(mod._MAX_PROVISIONAL + 50):
            heart.on_dream_insight()
        assert heart.provisional_pending <= mod._MAX_PROVISIONAL

    def test_an_unknown_token_confirms_nothing(self, heart):
        before = heart.get("Curiosity")
        assert heart.confirm_provisional("not-a-real-token") is False
        assert heart.get("Curiosity") == before


class TestEvidenceSemantics:
    def test_unmeasured_quality_is_not_zero_quality(self):
        """None means nobody measured; 0.0 means measured and bad. Collapsing
        them is how "we didn't check" becomes "it failed"."""
        assert ValueEvidence(verified=True, quality=None).scored_quality(0.5) == 0.5
        assert ValueEvidence(verified=True, quality=0.0).scored_quality(0.5) == 0.0

    def test_quality_is_clamped(self):
        assert ValueEvidence(verified=True, quality=9.0).scored_quality() == 1.0
        assert ValueEvidence(verified=True, quality=-4.0).scored_quality() == 0.0

    def test_non_finite_quality_falls_back(self):
        assert ValueEvidence(verified=True, quality=float("nan")).scored_quality(0.4) == 0.4
        assert ValueEvidence(verified=True, quality=float("inf")).scored_quality(0.4) == 0.4

    def test_believing_is_not_verifying(self):
        """verified=False with a confident quality still earns nothing."""
        values = HeartstoneValues()
        values._values["Curiosity"] = 0.50
        before = values.get("Curiosity")
        values.on_research_success(
            900, evidence=ValueEvidence(verified=False, quality=1.0),
        )
        assert values.get("Curiosity") == before


class TestUnrelatedEventsAreUntouched:
    def test_failures_still_register_immediately(self, heart):
        """Only unearned REWARDS were deferred. A failure is observed, not
        asserted, and must keep landing at once."""
        before = heart.get("Curiosity")
        heart.on_tool_failure()
        assert heart.get("Curiosity") < before

    def test_thermal_stress_still_registers(self, heart):
        before = heart.get("Self_Preservation")
        heart.on_thermal_stress(arousal=0.95, valence=0.10)
        assert heart.get("Self_Preservation") > before

    def test_values_stay_clamped(self, heart):
        for _ in range(200):
            heart.on_research_success(
                300, evidence=ValueEvidence(verified=True, quality=1.0),
            )
        assert heart.get("Curiosity") <= _MAX
