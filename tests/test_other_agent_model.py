"""Other-agent state estimation: live theory-of-mind affect/goal/belief filter + ladder seam."""
from __future__ import annotations

import time

import pytest

from core.social.other_agent_model import (
    OtherAgentStateEstimator,
    Signal,
    get_other_agent_model,
)


@pytest.fixture
def estimator(tmp_path):
    # Isolated storage, autosave off so tests don't depend on the debounce timer.
    return OtherAgentStateEstimator(storage_path=tmp_path / "agents.json", autosave=False)


# ── the Signal primitive: fusion + decay ────────────────────────────────────

def test_signal_low_confidence_snaps_to_observation():
    now = 1000.0
    s = Signal(value=0.5, confidence=0.0, baseline=0.5, half_life_s=600.0, updated_at=now)
    s.observe(0.9, strength=0.5, now=now)
    assert s.value > 0.8  # nothing to anchor against → moves nearly all the way
    assert s.confidence > 0.0


def test_signal_high_confidence_is_sticky():
    now = 1000.0
    s = Signal(value=0.2, confidence=0.95, baseline=0.5, half_life_s=600.0, updated_at=now)
    s.observe(0.9, strength=0.3, now=now)
    assert s.value < 0.45  # well-corroborated prior barely budges


def test_signal_decays_toward_baseline():
    now = 1000.0
    s = Signal(value=0.9, confidence=0.8, baseline=0.1, half_life_s=100.0, updated_at=now)
    v, c = s.decayed(now + 100.0)  # one half-life
    assert v == pytest.approx(0.5, abs=0.02)  # halfway from 0.9 toward 0.1
    assert c == pytest.approx(0.4, abs=0.02)  # confidence halved


# ── linguistic affect cues ──────────────────────────────────────────────────

def test_frustration_language_raises_frustration(estimator):
    estimator.observe_message("bryan", "ugh this is still not working, seriously")
    est = estimator.estimate("bryan")
    assert est.affect["frustration"] > 0.6
    assert est.affect_confidence["frustration"] > 0.0


def test_fatigue_and_late_hour_raise_fatigue(estimator):
    estimator.observe_message("bryan", "i'm exhausted, long day", hour=2)
    assert estimator.estimate("bryan").affect["fatigue"] > 0.6


def test_urgency_language_raises_urgency(estimator):
    estimator.observe_message("bryan", "need this fixed asap, right now please")
    assert estimator.estimate("bryan").affect["urgency"] > 0.6


def test_positive_feedback_raises_satisfaction_and_capability_belief(estimator):
    estimator.observe_message("bryan", "perfect, that works now, thank you!")
    est = estimator.estimate("bryan")
    assert est.affect["satisfaction"] > 0.7
    assert est.beliefs_about_aura["aura_capable"] > 0.5


def test_roleplay_accusation_raises_roleplaying_belief(estimator):
    estimator.observe_message("bryan", "you're not real, this is just roleplay")
    assert estimator.estimate("bryan").beliefs_about_aura["aura_roleplaying"] > 0.5


def test_distrust_lowers_trust_belief(estimator):
    estimator.observe_message("bryan", "are you sure? prove it, i don't believe you")
    assert estimator.estimate("bryan").beliefs_about_aura["aura_trustworthy"] < 0.5


# ── goals ───────────────────────────────────────────────────────────────────

def test_request_message_activates_a_goal(estimator):
    estimator.observe_message("bryan", "can you fix the login bug")
    goals = estimator.estimate("bryan").goals
    assert goals and "login bug" in goals[0]["goal"]
    assert goals[0]["activation"] > 0.5


def test_goal_decays_out_of_active_set(estimator):
    t0 = time.time()
    estimator.observe_message("bryan", "please build the parser", now=t0)
    # Far in the future the goal has decayed below the active threshold.
    later = estimator.estimate("bryan", now=t0 + 4 * 3600)
    assert later.goals == []


def test_goal_table_is_bounded(tmp_path):
    est = OtherAgentStateEstimator(storage_path=tmp_path / "a.json", autosave=False, max_goals=3)
    for i in range(10):
        est.observe_message("bryan", f"please build feature number {i}")
    # Internal table is pruned to the cap.
    assert len(est._models["bryan"].goals) <= 3


# ── outcomes + signals ──────────────────────────────────────────────────────

def test_failed_outcome_raises_frustration_lowers_capability(estimator):
    estimator.observe_outcome("bryan", success=False)
    est = estimator.estimate("bryan")
    assert est.affect["frustration"] > 0.4
    assert est.beliefs_about_aura["aura_capable"] < 0.5


def test_perceptual_signal_threat_raises_frustration(estimator):
    estimator.observe_signal("bryan", threat=0.9, affiliation=0.1)
    assert estimator.estimate("bryan").affect["frustration"] > 0.4


# ── recommendation: honesty about uncertainty ───────────────────────────────

def test_unknown_agent_recommends_asking(estimator):
    rec = estimator.recommendation("stranger")
    assert rec.should_ask  # no evidence → ask, don't assume
    assert rec.confidence == 0.0


def test_frustrated_agent_recommends_reassurance_and_slow_down(estimator):
    for _ in range(3):
        estimator.observe_message("bryan", "this is broken again, so frustrating, still not working")
    estimator.observe_outcome("bryan", success=False, weight=0.6)
    rec = estimator.recommendation("bryan")
    assert rec.offer_reassurance
    assert rec.slow_down
    assert rec.tone in {"repair", "calm_direct"}


# ── agency-ladder seam ──────────────────────────────────────────────────────

def test_social_signals_shape():
    # A neutral/unknown agent yields a high uncertainty signal (we don't know them).
    est = OtherAgentStateEstimator(storage_path=None, autosave=False)
    sig = est.social_signals("nobody")
    assert set(sig) == {"value_conflict", "uncertainty", "goal_horizon"}
    assert sig["uncertainty"] >= 0.6  # not knowing the agent is itself uncertainty


def test_rupture_risk_routes_social_situation_to_governance(estimator):
    from core.agency.hierarchical_agency import AgencyTier, HierarchicalAgency

    for _ in range(4):
        estimator.observe_message("bryan", "this is broken again, so frustrating and useless")
        estimator.observe_outcome("bryan", success=False, weight=0.7)

    s = estimator.social_situation("bryan", "should I auto-delete and rebuild their config?")
    assert s.context["agent_id"] == "bryan"
    assert s.value_conflict > 0.4
    # The ladder, consulting this situation, escalates a high-rupture social case to governance.
    agency = HierarchicalAgency(ledger_enabled=False)
    result = agency.dispatch(s)
    assert result.final_tier == AgencyTier.GOVERNANCE


def test_governance_handler_consults_social_estimate(estimator, monkeypatch):
    import core.social.other_agent_model as oam
    from core.agency.hierarchical_agency import HierarchicalAgency, Situation

    monkeypatch.setattr(oam, "_instance", estimator)
    estimator.observe_message("bryan", "this is broken again, so frustrating, still not working")

    agency = HierarchicalAgency(ledger_enabled=False)
    s = Situation("value call", value_conflict=0.9, context={"agent_id": "bryan"})
    result = agency.dispatch(s)
    assert "social" in result.detail
    assert "social_rupture_risk" in result.detail["social"]


# ── persistence ─────────────────────────────────────────────────────────────

def test_state_persists_across_instances(tmp_path):
    path = tmp_path / "agents.json"
    a = OtherAgentStateEstimator(storage_path=path, autosave=False)
    a.observe_message("bryan", "this is so frustrating, still broken")
    a.save()
    b = OtherAgentStateEstimator(storage_path=path, autosave=False)
    assert b.estimate("bryan").affect["frustration"] > 0.5


# ── singleton ───────────────────────────────────────────────────────────────

def test_singleton_is_stable():
    assert get_other_agent_model() is get_other_agent_model()
