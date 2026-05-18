import pytest

from core.dual_memory import DualMemorySystem


def test_dual_memory_stores_and_decodes_current_episode_model(tmp_path):
    memory = DualMemorySystem(base_dir=str(tmp_path / "memory"))

    episode_id = memory.store_experience(
        "User strongly prefers live-verifiable memory behavior.",
        emotional_valence=0.4,
        importance=0.8,
        tags=["preference"],
    )

    recent = memory.episodic.retrieve_recent(limit=1)

    assert recent
    assert recent[0].episode_id == episode_id
    assert "live-verifiable memory behavior" in recent[0].full_description


@pytest.mark.asyncio
async def test_dual_memory_retrieves_blended_context_without_horcrux_boot(tmp_path):
    memory = DualMemorySystem(base_dir=str(tmp_path / "memory"))
    memory.store_experience(
        "User asked Aura to harden every memory subsystem before launch.",
        importance=0.9,
    )
    memory.learn_fact(
        "Aura memory",
        "must be",
        "durable, encrypted, and semantically retrievable",
        0.91,
    )

    context = await memory.retrieve_context("Aura memory durable launch", max_episodes=2, max_facts=2)

    assert "Personal Memory" in context
    assert "Known Facts" in context
    assert "durable, encrypted, and semantically retrievable" in context
