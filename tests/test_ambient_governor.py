"""What an unprompted mind is allowed to do.

The three constraints under test are the ones that make an always-on
companion liveable: ACT and ESCALATE can never be auto-approved, an
unconfigured interruption budget means silence rather than a guessed
number, and every ambient claim can be graded against what happened.
"""
from __future__ import annotations

import time

import pytest

from core.agency.ambient_governor import (
    AmbientGovernor,
    Outcome,
    ProposalStatus,
    Strength,
)


@pytest.fixture
def gov() -> AmbientGovernor:
    return AmbientGovernor()


# ---------------------------------------------------------------- the ladder


@pytest.mark.parametrize("strength", [Strength.ACT, Strength.ESCALATE])
def test_act_and_escalate_can_never_be_auto_approved(gov, strength):
    gov.configure(daily_silence_cap=10)
    assert gov.may_auto_approve(strength) is False
    proposal = gov.propose(strength=strength, summary="do the thing")
    result = gov.auto_approve(proposal.proposal_id)
    assert result.status is ProposalStatus.WITHHELD
    assert "never be auto-approved" in result.reason


@pytest.mark.parametrize("strength", [Strength.OBSERVE, Strength.NUDGE])
def test_observe_and_nudge_may_be_auto_approved(gov, strength):
    gov.configure(daily_silence_cap=10)
    assert gov.may_auto_approve(strength) is True
    proposal = gov.propose(strength=strength, summary="mention it")
    assert gov.auto_approve(proposal.proposal_id).status is ProposalStatus.APPROVED


def test_an_owner_can_still_approve_an_act(gov):
    """The prohibition is on AUTO-approval, not on the action existing."""
    gov.configure(daily_silence_cap=10)
    proposal = gov.propose(strength=Strength.ACT, summary="send it")
    decided = gov.decide(proposal.proposal_id, approved=True)
    assert decided.status is ProposalStatus.APPROVED


def test_unknown_strength_degrades_to_observation(gov):
    gov.configure(daily_silence_cap=10)
    proposal = gov.propose(strength="shout", summary="x")
    assert proposal.strength is Strength.OBSERVE


# ---------------------------------------------------------------- the budget


def test_an_unconfigured_governor_stays_silent(gov):
    """No agreed cap means silence, not a reasonable-sounding default."""
    assert gov.is_configured is False
    proposal = gov.propose(strength=Strength.NUDGE, summary="hey")
    assert proposal.status is ProposalStatus.WITHHELD
    assert "no interruption budget configured" in proposal.reason


def test_observation_is_always_allowed_even_unconfigured(gov):
    proposal = gov.propose(strength=Strength.OBSERVE, summary="noted")
    assert proposal.status is ProposalStatus.PROPOSED


def test_zero_is_a_real_answer_distinct_from_unconfigured(gov):
    """Zero is an owner saying never; None is nobody having said anything."""
    gov.configure(daily_silence_cap=0)
    assert gov.is_configured is True
    proposal = gov.propose(strength=Strength.NUDGE, summary="hey")
    assert proposal.status is ProposalStatus.WITHHELD
    assert "budget spent" in proposal.reason


def test_budget_is_spent_and_then_exhausted(gov):
    gov.configure(daily_silence_cap=2)
    assert gov.remaining_budget() == 2
    assert gov.propose(strength=Strength.NUDGE, summary="1").status is ProposalStatus.PROPOSED
    assert gov.propose(strength=Strength.NUDGE, summary="2").status is ProposalStatus.PROPOSED
    assert gov.remaining_budget() == 0
    third = gov.propose(strength=Strength.NUDGE, summary="3")
    assert third.status is ProposalStatus.WITHHELD


def test_observations_do_not_spend_budget(gov):
    gov.configure(daily_silence_cap=1)
    for _ in range(20):
        gov.propose(strength=Strength.OBSERVE, summary="watching")
    assert gov.remaining_budget() == 1


def test_silence_protecting_proposals_are_exempt(gov):
    """Charging a quiet-hours request would make the mechanism eat itself."""
    gov.configure(daily_silence_cap=1)
    gov.propose(strength=Strength.NUDGE, summary="spends it")
    assert gov.remaining_budget() == 0
    exempt = gov.propose(
        strength=Strength.NUDGE, summary="hold quiet hours?", silence_exempt=True
    )
    assert exempt.status is ProposalStatus.PROPOSED


def test_budget_resets_on_a_new_calendar_day(gov):
    gov.configure(daily_silence_cap=1)
    yesterday = time.time() - 26 * 3600
    gov.propose(strength=Strength.NUDGE, summary="old", now=yesterday)
    assert gov.remaining_budget() == 1
    assert gov.propose(strength=Strength.NUDGE, summary="new").status is ProposalStatus.PROPOSED


def test_withheld_proposals_do_not_themselves_spend_budget(gov):
    gov.configure(daily_silence_cap=1)
    gov.propose(strength=Strength.NUDGE, summary="1")
    for _ in range(5):
        gov.propose(strength=Strength.NUDGE, summary="refused")
    # Still exactly one spent; refusals are recorded, not charged.
    assert gov.status()["spent_today"] == 1


def test_a_refused_proposal_is_recorded_not_discarded(gov):
    """The mind should be able to see what it wanted to say and could not."""
    gov.configure(daily_silence_cap=0)
    proposal = gov.propose(strength=Strength.NUDGE, summary="wanted to say this")
    assert proposal.summary == "wanted to say this"
    assert gov.status()["withheld"] == 1


def test_restraint_rate_reports_silence_as_the_success_it_is(gov):
    gov.configure(daily_silence_cap=1)
    gov.propose(strength=Strength.NUDGE, summary="allowed")
    gov.propose(strength=Strength.NUDGE, summary="refused")
    gov.propose(strength=Strength.NUDGE, summary="refused")
    assert gov.status()["restraint_rate"] == pytest.approx(2 / 3, abs=1e-4)


# ------------------------------------------------------------- predictions


def test_a_prediction_can_be_graded(gov):
    p = gov.predict(claim_type="leave_by", claim="leaves by 9", p_hat=0.7, resolve_by=time.time() + 60)
    resolved = gov.resolve(p.prediction_id, correct=True)
    assert resolved.outcome is Outcome.CORRECT


def test_an_unobservable_prediction_expires_unknown(gov):
    gov.predict(claim_type="leave_by", claim="x", p_hat=0.7, resolve_by=time.time() - 1)
    expired = gov.expire_due()
    assert [p.outcome for p in expired] == [Outcome.UNKNOWN]


def test_unknown_outcomes_stay_out_of_the_calibration(gov):
    """An unobservable prediction must not improve the record by expiring."""
    gov.predict(claim_type="t", claim="x", p_hat=0.9, resolve_by=time.time() - 1)
    gov.expire_due()
    cal = gov.calibration("t")
    assert cal["graded"] == 0
    assert cal["unknown"] == 1
    assert cal["base_rate"] is None


def test_history_does_not_blend_below_the_sample_floor(gov):
    now = time.time()
    for _ in range(3):
        p = gov.predict(claim_type="t", claim="x", p_hat=0.9, resolve_by=now + 60)
        gov.resolve(p.prediction_id, correct=False)
    fresh = gov.predict(claim_type="t", claim="x", p_hat=0.9, resolve_by=now + 60)
    assert fresh.p_hat == pytest.approx(0.9)


def test_history_corrects_confidence_once_there_is_enough(gov):
    now = time.time()
    for _ in range(6):
        p = gov.predict(claim_type="t", claim="x", p_hat=0.9, resolve_by=now + 60)
        gov.resolve(p.prediction_id, correct=False)
    fresh = gov.predict(claim_type="t", claim="x", p_hat=0.9, resolve_by=now + 60)
    # Base rate is 0.0; the rule keeps most of its say but is pulled down.
    assert fresh.p_hat < 0.9
    assert fresh.p_hat == pytest.approx(0.9 * 0.6, abs=1e-4)


def test_calibration_reports_overconfidence(gov):
    now = time.time()
    for i in range(10):
        p = gov.predict(claim_type="t", claim="x", p_hat=0.5, resolve_by=now + 60)
        gov.resolve(p.prediction_id, correct=i < 2)
    cal = gov.calibration("t")
    assert cal["graded"] == 10
    assert cal["base_rate"] == pytest.approx(0.2)
    assert cal["overconfident_by"] > 0
    assert cal["brier"] is not None


def test_calibration_says_so_when_nothing_is_graded(gov):
    assert "unvalidated" in gov.calibration()["verdict"]


def test_negative_cap_is_rejected(gov):
    with pytest.raises(ValueError):
        gov.configure(daily_silence_cap=-1)
