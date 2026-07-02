from __future__ import annotations

from types import SimpleNamespace

from core.agency.ambient_life_director import AmbientLifeDirector
from core.agency.initiative_arbiter import ScoredInitiative


def _scored(goal: str, score: float, metadata: dict | None = None) -> ScoredInitiative:
    return ScoredInitiative(
        initiative={"goal": goal, "metadata": dict(metadata or {})},
        scores={},
        final_score=score,
        rationale="",
    )


def test_ambient_life_director_buckets_and_lod_protect_host_under_pressure(tmp_path):
    director = AmbientLifeDirector(state_path=tmp_path / "ambient.json")
    state = SimpleNamespace(runtime=SimpleNamespace(resource_pressure=0.9))
    scored = [
        _scored("Play with a whimsical sandbox idea", 0.80),
        _scored("Repair memory pressure before it crashes Aura", 0.65),
    ]

    ranked = director.prioritize_scored(scored, state)

    assert ranked[0].initiative["metadata"]["ambient_bucket"] == "repair"
    assert ranked[0].initiative["metadata"]["ambient_lod_mode"] == "deferred"
    assert ranked[1].initiative["metadata"]["ambient_bucket"] == "play"


def test_ambient_life_director_allows_curiosity_when_pressure_is_low(tmp_path):
    director = AmbientLifeDirector(state_path=tmp_path / "ambient.json")
    state = SimpleNamespace(runtime=SimpleNamespace(resource_pressure=0.0))
    scored = [
        _scored("Run routine maintenance", 0.70),
        _scored("Research a novel question for future conversation", 0.68),
    ]

    ranked = director.prioritize_scored(scored, state)

    assert ranked[0].initiative["metadata"]["ambient_bucket"] == "curiosity"
    assert ranked[0].initiative["goal"].startswith("Research")


def test_ambient_life_director_persists_encounter_memory(tmp_path):
    path = tmp_path / "ambient.json"
    director = AmbientLifeDirector(state_path=path)
    memory = director.record_encounter(
        "chatgpt",
        outcome="Held a useful conversation about sentience criteria.",
        valence=0.75,
        traits={"kind": "ai_interlocutor"},
    )

    assert memory.seen_count == 1
    reloaded = AmbientLifeDirector(state_path=path)
    recalled = reloaded.recall_encounter("chatgpt")
    assert recalled is not None
    assert recalled.valence == 0.75
    assert recalled.traits["kind"] == "ai_interlocutor"

