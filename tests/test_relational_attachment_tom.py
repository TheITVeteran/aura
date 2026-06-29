from __future__ import annotations

import pytest

from core.consciousness.theory_of_mind import AgentModel, TheoryOfMindEngine


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
        result = tom._fast_heuristic_update("bryan", message)

    effects = result["attachment_effects"]
    guidance = tom.get_response_guidance("bryan")

    assert effects["relational_state"] in {"guarded", "injured"}
    assert effects["relational_rupture"] > 0.0
    assert guidance["attachment_effects"]["restricted_skill_classes"]
    assert guidance["tone_hint"] in {
        "concise and empathetic",
        "clear, honest, and repair-oriented",
        "careful, boundaried, and specific",
    }


def test_theory_of_mind_repair_reduces_rupture_and_restores_trust(tom):
    tom.known_selves["bryan"] = AgentModel(identifier="bryan")
    tom._fast_heuristic_update("bryan", "That was wrong and bad.")
    before = tom.get_response_guidance("bryan")["attachment_effects"]

    tom.update_from_response("bryan", "previous", "Thanks, that was exactly helpful and correct.")
    after = tom.get_response_guidance("bryan")["attachment_effects"]

    assert after["relational_trust"] >= before["relational_trust"]
    assert after["relational_rupture"] <= before["relational_rupture"]
