from __future__ import annotations

import pytest

from core.consciousness.theory_of_mind import AgentModel, TheoryOfMindEngine
from core.container import ServiceContainer


@pytest.fixture()
def tom(monkeypatch, tmp_path):
    monkeypatch.setattr(
        TheoryOfMindEngine,
        "_resolve_data_path",
        lambda self: tmp_path / "theory_of_mind.json",
    )
    return TheoryOfMindEngine(cognitive_engine=None)


def test_theory_of_mind_records_attachment_rupture_as_causal_guidance(tom):
    tom.known_selves["bryan"] = AgentModel(identifier="bryan")
    for message in (
        "That was wrong and bad.",
        "I hate when you ignore what I asked.",
        "That was rude and not what I meant.",
        "Wrong again. Bad answer.",
    ):
        result = tom._fast_heuristic_update(
            "bryan",
            message,
            response_feedback_context=True,
        )

    effects = result["attachment_effects"]
    guidance = tom.get_response_guidance("bryan")

    assert effects["relational_state"] in {"guarded", "injured"}
    assert effects["relational_rupture"] > 0.0
    assert guidance["attachment_effects"]["restricted_skill_classes"]
    assert guidance["tone_hint"] in {
        "clear, honest, and repair-oriented",
        "careful, boundaried, and specific",
    }


def test_theory_of_mind_repair_reduces_rupture_and_restores_trust(tom):
    tom.known_selves["bryan"] = AgentModel(identifier="bryan")
    tom._fast_heuristic_update(
        "bryan",
        "That was wrong and bad.",
        response_feedback_context=True,
    )
    before = tom.get_response_guidance("bryan")["attachment_effects"]

    tom.update_from_response("bryan", "previous", "Thanks, that was exactly helpful and correct.")
    after = tom.get_response_guidance("bryan")["attachment_effects"]

    assert after["relational_trust"] >= before["relational_trust"]
    assert after["relational_rupture"] <= before["relational_rupture"]


@pytest.mark.asyncio
async def test_short_continue_turn_is_not_misclassified_as_terse(tom):
    result = await tom.understand_user("bryan", "continue", {"user_id": "bryan"})

    assert tom.active_user_id == "bryan"
    assert result["emotional_state"] == "neutral"
    assert tom.known_selves["bryan"].trust_level == 0.5
    assert "message" not in tom.known_selves["bryan"].interaction_history[-1]


@pytest.mark.asyncio
async def test_relationship_persistence_redacts_messages_and_transient_goals(tom):
    await tom.understand_user(
        "bryan",
        "can you fix private-project-codename login bug?",
        {"user_id": "bryan"},
    )

    tom.save()
    payload = tom._data_path.read_text()

    assert "private-project-codename" not in payload
    assert "message_digest" in payload
    assert tom.known_selves["bryan"].goals


def test_default_social_context_uses_estimator_active_user_not_stale_tom_user(tom):
    class ActiveEstimator:
        active_agent_id = "bryan"

        @staticmethod
        def cognitive_snapshot(user_id):
            return {
                "agent_id": user_id,
                "confidence": 0.7,
                "observations": 2,
                "affect_hypotheses": {
                    "urgency": {"value": 0.8, "confidence": 0.6},
                },
                "recommendation": {},
            }

    tom.known_selves["alice"] = AgentModel(identifier="alice")
    tom.known_selves["bryan"] = AgentModel(identifier="bryan")
    tom.active_user_id = "alice"
    ServiceContainer.clear()
    ServiceContainer.register_instance(
        "other_agent_model",
        ActiveEstimator(),
        required=False,
    )

    try:
        context = tom.get_context_block()
        guidance = tom.get_response_guidance()
    finally:
        ServiceContainer.clear()

    assert "agent=bryan" in context
    assert "agent=alice" not in context
    assert "urgency~0.80@0.60" in context
    assert guidance["social_confidence"] == pytest.approx(0.7)
    assert guidance["tone_hint"] == "neutral and respectful"
