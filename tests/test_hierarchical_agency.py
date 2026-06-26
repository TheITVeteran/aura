"""Hierarchical agency: lowest-sufficient-tier selection + escalation, wired to real engines."""
from __future__ import annotations

import pytest

from core.agency.hierarchical_agency import (
    AgencyTier,
    HierarchicalAgency,
    Situation,
    TierResult,
)


@pytest.fixture
def agency():
    # Ledger off so tests don't touch the shared db; classification/escalation is the point.
    return HierarchicalAgency(ledger_enabled=False)


# ── classification: the right tier for the dominant signal ──────────────────

def test_value_conflict_routes_to_governance(agency):
    s = Situation("delete the user's files?", value_conflict=0.9, threat=0.9)
    assert agency.classify(s) == AgencyTier.GOVERNANCE  # governance preempts even threat


def test_acute_threat_routes_to_reflex(agency):
    assert agency.classify(Situation("memory corrupting now", threat=0.9)) == AgencyTier.REFLEX


def test_capability_gap_routes_to_self_improvement(agency):
    assert agency.classify(Situation("need a tool I lack", capability_gap=0.8)) == AgencyTier.SELF_IMPROVEMENT


def test_uncertainty_routes_to_scientific(agency):
    assert agency.classify(Situation("unknown dynamics", uncertainty=0.8)) == AgencyTier.SCIENTIFIC


def test_long_horizon_routes_to_strategic(agency):
    assert agency.classify(Situation("multi-week goal", goal_horizon=0.8)) == AgencyTier.STRATEGIC


def test_known_skill_low_novelty_routes_to_habit(agency):
    assert agency.classify(Situation("routine", known_skill="open_file", novelty=0.1)) == AgencyTier.HABIT


def test_default_is_deliberative(agency):
    assert agency.classify(Situation("novel but tractable", novelty=0.6)) == AgencyTier.DELIBERATIVE


# ── dispatch + escalation ───────────────────────────────────────────────────

def test_reflex_handles_acute_threat(agency):
    result = agency.dispatch(Situation("damage spike", threat=0.95))
    assert result.final_tier == AgencyTier.REFLEX
    assert result.handled
    assert result.detail["response"] == "protect_boundary"


def test_subthreshold_reflex_escalates_upward(agency):
    # classify() sends acute threat to reflex; if the felt threat is just below threshold the
    # reflex declines and control escalates to the next tier.
    s = Situation("borderline", threat=0.59)  # just under default 0.6 threat threshold
    # force a reflex start by raising threat into classify range, but keep handler subthreshold
    s.threat = 0.6
    agency.register_handler(
        AgencyTier.REFLEX,
        lambda sit: TierResult(tier=AgencyTier.REFLEX, handled=False, escalate=True, reason="subthreshold"),
    )
    result = agency.dispatch(s)
    assert result.starting_tier == AgencyTier.REFLEX
    assert result.final_tier > AgencyTier.REFLEX  # escalated up the ladder
    assert AgencyTier.REFLEX in result.path


def test_escalation_climbs_until_handled(agency):
    # Every tier below governance declines → control should climb to governance.
    for tier in list(AgencyTier):
        if tier != AgencyTier.GOVERNANCE:
            agency.register_handler(
                tier, lambda sit, t=tier: TierResult(tier=t, handled=False, escalate=True)
            )
    agency.register_handler(
        AgencyTier.GOVERNANCE,
        lambda sit: TierResult(tier=AgencyTier.GOVERNANCE, handled=True, confidence=0.9),
    )
    result = agency.dispatch(Situation("nobody else will take it", novelty=0.6))
    assert result.final_tier == AgencyTier.GOVERNANCE
    assert result.handled


def test_broken_handler_escalates_not_crashes(agency):
    def _boom(sit):
        raise RuntimeError("tier exploded")

    agency.register_handler(AgencyTier.DELIBERATIVE, _boom)
    result = agency.dispatch(Situation("novel", novelty=0.6))
    # deliberative blew up → escalated, ladder did not crash
    assert AgencyTier.DELIBERATIVE in result.path
    assert result.final_tier > AgencyTier.DELIBERATIVE


def test_scientific_tier_forms_a_real_hypothesis(tmp_path, monkeypatch):
    # Point the scientific engine singleton at a temp db, then confirm the scientific tier
    # actually forms a hypothesis (real engine integration, not a stub).
    import core.cognition.scientific_engine as se
    monkeypatch.setattr(se, "_engine", se.ScientificEngine(db_path=str(tmp_path / "sci.db")))

    agency = HierarchicalAgency(ledger_enabled=False)
    result = agency.dispatch(Situation("how does this API behave?", uncertainty=0.8))
    assert result.final_tier == AgencyTier.SCIENTIFIC
    assert result.handled
    hid = result.detail["hypothesis_id"]
    assert se.get_scientific_engine().get(hid) is not None
