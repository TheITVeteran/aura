"""Zero-shot transfer: numbers describing a rejected action, and rules that
could never be falsified."""
from __future__ import annotations

import pytest

from core.advanced_cognition.schemas import (
    ActionCandidate,
    Episode,
    Observation,
    Outcome,
    Principle,
)
from core.advanced_cognition.zero_shot_transfer import (
    ZeroShotTransferEngine,
    _risk_score,
)

pytestmark = pytest.mark.unit


def _engine(tmp_path):
    engine = ZeroShotTransferEngine()
    engine.state_path = tmp_path / "zst.json"
    return engine


def _episode(*, success=True, reward=0.8, harm=0.0, terminal=False, kind="do"):
    return Episode(
        Observation(domain="d", state={"x": 1}, timestamp=1000.0),
        ActionCandidate("a1", kind),
        {},
        Outcome(success=success, reward=reward, harm=harm, terminal=terminal),
        created_at=1000.0,
    )


# ── the decision must describe the action it selected ──────────────────────


def test_reported_risk_belongs_to_the_selected_action(tmp_path):
    """`selected` came from acceptable[0] but risk/confidence from ranking[0],
    so whenever the top-ranked candidate exceeded tolerance the caller got
    numbers describing a REJECTED action — and a risk above the tolerance it
    had just been told was satisfied."""
    engine = _engine(tmp_path)
    obs = Observation(domain="d", state={"x": 1})
    risky = ActionCandidate("risky", "wipe", reversible=False, authority_tier=6)
    safe = ActionCandidate("safe", "read", reversible=True, authority_tier=0)

    decision = engine.rank_actions(obs, [risky, safe], risk_tolerance=0.3)

    assert decision.selected is not None
    assert decision.risk <= 0.3, "reported risk must satisfy the stated tolerance"
    chosen = next(r for r in decision.ranking
                  if r["action"]["action_id"] == decision.selected.action_id)
    assert decision.risk == pytest.approx(chosen["risk"])
    assert decision.confidence == pytest.approx(chosen["confidence"])


def test_no_acceptable_action_reports_maximum_risk(tmp_path):
    engine = _engine(tmp_path)
    obs = Observation(domain="d", state={"x": 1})
    risky = ActionCandidate("risky", "wipe", reversible=False, authority_tier=9)

    decision = engine.rank_actions(obs, [risky], risk_tolerance=0.01)

    assert decision.selected is None
    assert decision.risk == 1.0
    assert decision.confidence == 0.0


# ── risk fails closed; confidence fails low ────────────────────────────────


def test_unusable_risk_is_maximal_not_minimal():
    """clamp() maps non-finite to its LOW bound — right for confidence, exactly
    backwards for risk, where 0.0 passes every tolerance."""
    assert _risk_score(float("nan")) == 1.0
    assert _risk_score(float("inf")) == 1.0
    assert _risk_score("not a number") == 1.0


def test_risk_stays_bounded():
    assert _risk_score(-5.0) == 0.0
    assert _risk_score(17.0) == 1.0
    assert _risk_score(0.42) == pytest.approx(0.42)


# ── induced rules must be falsifiable ──────────────────────────────────────


def test_a_prefer_rule_is_contradicted_by_a_harmful_outcome(tmp_path):
    """Every upsert passed matched=True and nothing ever passed False, so
    confidence rose monotonically and no outcome could falsify a rule — an
    induced principle became permanent the moment it was created."""
    engine = _engine(tmp_path)
    principle = Principle(name="p", condition_features={"a"},
                          action_features={"b"}, effect="positive_affordance")

    assert engine._episode_confirms(principle, _episode(success=True, reward=0.9))
    assert not engine._episode_confirms(
        principle, _episode(success=False, reward=0.0, harm=0.9)
    )


def test_an_avoid_rule_is_contradicted_by_a_clean_run(tmp_path):
    engine = _engine(tmp_path)
    principle = Principle(name="p", condition_features={"a"},
                          action_features={"b"}, effect="terminal_hazard")

    assert engine._episode_confirms(principle, _episode(success=False, harm=0.9))
    assert not engine._episode_confirms(principle, _episode(success=True, reward=0.8))


def test_contradiction_actually_lowers_confidence(tmp_path):
    """The end-to-end consequence: a rule can now lose standing."""
    principle = Principle(name="p", condition_features={"a"},
                          action_features={"b"}, effect="positive_affordance")
    principle.update(_episode(success=True, reward=0.9), True)
    confident = principle.confidence

    for i in range(4):
        ep = _episode(success=False, reward=0.0, harm=0.9)
        object.__setattr__(ep, "episode_id", f"contradiction_{i}")
        principle.update(ep, False)

    assert principle.confidence < confident
    assert principle.contradictions == 4
