from __future__ import annotations

import time

import numpy as np
import pytest

from core.consciousness.recursive_tom import (
    ACTION_CATEGORIES,
    BACKGROUND_PRIVATE_ACTIONS,
    BIAS_DIM,
    FOREGROUND_ACTIONS,
    MAX_DEPTH,
    ObserverContextModel,
    RecursiveTheoryOfMind,
)


def _digest(char: str) -> str:
    return char * 64


def _snapshot(
    agent_id: str = "bryan",
    *,
    digest: str = "a" * 64,
    confidence: object = 0.8,
) -> dict:
    return {
        "agent_id": agent_id,
        "confidence": confidence,
        "observations": 5,
        "social_rupture_risk": 0.6,
        "evidence_digest": digest,
        "at": time.time(),
        "affect_hypotheses": {
            "frustration": {"value": 0.7, "confidence": 0.8},
            "urgency": {"value": 0.6, "confidence": 0.7},
            "satisfaction": {"value": 0.1, "confidence": 0.9},
        },
        "likely_goals": [{"goal": "raw goal must not copy"}],
        "beliefs_about_aura": {"aura_trustworthy": 0.2},
    }


def test_initial_state_is_explicitly_non_recursive_and_background():
    model = ObserverContextModel()

    status = model.get_status()

    assert MAX_DEPTH == 0
    assert status["recursive_mind_claims"] is False
    assert status["snapshot_count"] == 0
    assert status["privacy_posture"] == "background"


def test_nonzero_recursive_depth_and_unscoped_identity_are_rejected():
    with pytest.raises(ValueError, match="recursive depth"):
        ObserverContextModel(max_depth=3)

    model = ObserverContextModel()
    with pytest.raises(ValueError, match="exact non-empty agent_id"):
        model.observe_agent("", evidence_digest=_digest("a"))
    with pytest.raises(ValueError, match="SHA-256"):
        model.observe_agent("bryan", evidence_digest="raw text")


def test_calibrated_snapshot_creates_only_depth_zero_hypothesis():
    model = ObserverContextModel()

    root = model.register_interaction("bryan", _snapshot())

    assert root.agent_id == "bryan"
    assert root.depth == 0
    assert root.hypothesis is True
    assert root.nested is None
    assert model.depth_reached("bryan") == 0
    assert model.get_mind_at_depth("bryan", 1) is None
    assert "satisfaction" not in root.affect_hypotheses


def test_snapshot_must_match_exact_agent_and_have_evidence():
    model = ObserverContextModel()

    with pytest.raises(ValueError, match="match the exact agent_id"):
        model.register_interaction("bryan", _snapshot("alice"))
    malformed = _snapshot()
    malformed["evidence_digest"] = "none"
    with pytest.raises(ValueError, match="SHA-256"):
        model.register_interaction("bryan", malformed)


def test_snapshot_projection_is_bounded_and_does_not_copy_raw_goals_or_beliefs():
    model = ObserverContextModel()
    snapshot = _snapshot(confidence=float("nan"))
    snapshot["observations"] = "not-an-int"
    snapshot["social_rupture_risk"] = float("inf")

    projected = model.register_interaction("bryan", snapshot).to_dict()

    assert projected["confidence"] == 0.0
    assert projected["observations"] == 0
    assert projected["social_rupture_risk"] == 0.0
    assert "raw goal" not in str(projected)
    assert "aura_trustworthy" not in str(projected)


def test_same_snapshot_evidence_is_idempotent():
    model = ObserverContextModel()
    first = model.register_interaction("bryan", _snapshot(confidence=0.8))
    second = model.register_interaction("bryan", _snapshot(confidence=0.1))

    assert second is first
    assert second.confidence == pytest.approx(0.8)


def test_observer_bias_is_zero_without_digest_backed_presence():
    model = ObserverContextModel()

    profile = model.get_observer_bias()

    assert np.allclose(profile.bias, 0.0)
    assert profile.total_observer_presence == 0.0
    assert profile.privacy_posture == "background"


def test_presence_prioritizes_foreground_and_suppresses_private_background_work():
    model = ObserverContextModel()
    assert model.observe_agent(
        "bryan",
        evidence_digest=_digest("b"),
        strength=0.9,
    )

    profile = model.get_observer_bias()

    assert profile.privacy_posture == "interactive"
    assert profile.active_observers == ["bryan"]
    for name in FOREGROUND_ACTIONS:
        assert profile.bias[ACTION_CATEGORIES.index(name)] > 0.0
    for name in BACKGROUND_PRIVATE_ACTIONS:
        assert profile.bias[ACTION_CATEGORIES.index(name)] < 0.0
    assert profile.bias[ACTION_CATEGORIES.index("tool_use")] == 0.0
    assert model.should_defer_background_action("dream") is True
    assert model.should_defer_background_action("tool_use") is False


def test_duplicate_presence_evidence_does_not_inflate_presence():
    model = ObserverContextModel()
    assert model.observe_agent("bryan", evidence_digest=_digest("c"), strength=0.7)
    initial = model.total_observer_presence()

    assert model.observe_agent(
        "bryan",
        evidence_digest=_digest("c"),
        strength=1.0,
    ) is False
    assert model.total_observer_presence() == pytest.approx(initial, abs=0.01)


def test_old_presence_evidence_is_inactive():
    model = ObserverContextModel()
    model.observe_agent(
        "bryan",
        evidence_digest=_digest("d"),
        strength=1.0,
        observed_at=time.time() - 120.0,
    )

    assert model.total_observer_presence() == 0.0
    assert model.active_observers() == []


def test_future_presence_timestamp_is_rejected():
    model = ObserverContextModel()

    with pytest.raises(ValueError, match="current"):
        model.observe_agent(
            "bryan",
            evidence_digest=_digest("7"),
            observed_at=time.time() + 60.0,
        )


def test_multiple_exact_observers_aggregate_without_crossing_identity():
    model = ObserverContextModel()
    model.observe_agent("alice", evidence_digest=_digest("e"), strength=0.6)
    model.observe_agent("bryan", evidence_digest=_digest("f"), strength=0.6)

    profile = model.get_observer_bias()

    assert profile.total_observer_presence > 0.6
    assert set(profile.active_observers) == {"alice", "bryan"}


def test_forget_agent_removes_presence_and_social_projection():
    model = ObserverContextModel()
    model.register_interaction("bryan", _snapshot())
    model.observe_agent("bryan", evidence_digest=_digest("9"), strength=0.9)

    model.forget_agent("bryan")

    assert model.get_mind("bryan") is None
    assert model.active_observers() == []
    assert model.get_status()["snapshot_count"] == 0


def test_compatibility_class_has_stable_bias_shape():
    model = RecursiveTheoryOfMind()
    model.observe_agent("bryan", evidence_digest=_digest("8"), strength=0.9)

    assert model.get_observer_bias().bias.shape == (BIAS_DIM,)
    assert FOREGROUND_ACTIONS.isdisjoint(BACKGROUND_PRIVATE_ACTIONS)
