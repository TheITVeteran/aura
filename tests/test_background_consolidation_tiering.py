from types import SimpleNamespace

import pytest

from core.memory.sovereign_pruner import MemoryRecord, SovereignPruner
from core.phases.memory_consolidation import MemoryConsolidationPhase
from core.runtime.errors import get_degradation_tracker
from core.state.aura_state import AuraState


class AsyncCallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


@pytest.mark.asyncio
async def test_sovereign_pruner_uses_background_think_lane(monkeypatch):
    monkeypatch.setenv("AURA_PRUNER_LLM_CONSOLIDATION", "1")
    think = AsyncCallRecorder(SimpleNamespace(content="The exchange revealed a stable preference."))
    brain = SimpleNamespace(
        think=think,
    )
    orchestrator = SimpleNamespace(cognitive_engine=brain)
    pruner = SovereignPruner(orchestrator=orchestrator)

    result = await pruner._consolidate(
        MemoryRecord(
            id="mem-1",
            content="Bryan prefers concise architecture updates.",
            timestamp=0.0,
            source="conversation",
            emotional_weight=0.2,
            identity_relevance=0.8,
        )
    )

    assert result == "The exchange revealed a stable preference."
    assert len(think.calls) == 1
    _, kwargs = think.calls[0]
    assert kwargs["origin"] == "sovereign_pruner"
    assert kwargs["is_background"] is True


@pytest.mark.asyncio
async def test_memory_consolidation_skips_ephemeral_fallback_messages():
    container = SimpleNamespace(get=lambda name, default=None: default)
    phase = MemoryConsolidationPhase(container)

    state = AuraState.default()
    state.affect.arousal = 0.95
    state.cognition.working_memory.append(
        {
            "role": "assistant",
            "content": "Give me a moment — I'm thinking through something.",
            "timestamp": 1.0,
            "origin": "mind_tick_fallback",
            "ephemeral": True,
        }
    )

    new_state = await phase.execute(state)

    assert new_state.cold.evolution_log == []
    assert new_state.cognition.working_memory[-1]["ephemeral"] is True


@pytest.mark.asyncio
async def test_memory_consolidation_loop_detection_preserves_latest_live_answer():
    container = SimpleNamespace(get=lambda name, default=None: default)
    phase = MemoryConsolidationPhase(container)

    state = AuraState.default()
    repeated = (
        "I would use browser research to compare sources, then draft the findings "
        "into a document and preserve the receipt trail."
    )
    state.cognition.working_memory.extend(
        [
            {"role": "user", "content": "What tools can you use?", "timestamp": 1.0},
            {"role": "assistant", "content": repeated, "timestamp": 2.0},
            {"role": "user", "content": "How would you use browser research and a document?", "timestamp": 3.0},
            {"role": "assistant", "content": repeated, "timestamp": 4.0},
        ]
    )

    new_state = await phase.execute(
        state,
        objective="How would you use browser research and a document?",
    )

    assert new_state.cognition.working_memory[-2]["role"] == "user"
    assert new_state.cognition.working_memory[-1]["role"] == "assistant"
    assert new_state.cognition.working_memory[-1]["content"] == repeated
    assert sum(
        1
        for message in new_state.cognition.working_memory
        if message.get("role") == "assistant" and message.get("content") == repeated
    ) == 1
    assert (
        new_state.response_modifiers["memory_consolidation_loop_signal"]["latest_answer_preserved"]
        is True
    )


@pytest.mark.asyncio
async def test_memory_consolidation_commits_completed_turn_to_memory_facade():
    commit_interaction = AsyncCallRecorder()
    memory_facade = SimpleNamespace(commit_interaction=commit_interaction)
    container = SimpleNamespace(
        get=lambda name, default=None: memory_facade if name == "memory_facade" else default
    )
    phase = MemoryConsolidationPhase(container)

    state = AuraState.default()
    state.cognition.current_origin = "api"
    state.cognition.working_memory.extend(
        [
            {"role": "user", "content": "What do you think this song means?", "timestamp": 1.0},
            {"role": "assistant", "content": "It feels like a song about clarity under pressure.", "timestamp": 2.0},
        ]
    )

    new_state = await phase.execute(state, objective="Discuss the song")

    assert len(commit_interaction.calls) == 1
    _, kwargs = commit_interaction.calls[0]
    assert kwargs["context"] == "What do you think this song means?"
    assert kwargs["action"] == "conversation_reply"
    assert kwargs["outcome"] == "It feels like a song about clarity under pressure."
    assert kwargs["success"] is True
    assert kwargs["metadata"]["origin"] == "api"
    assert "dominant_emotion" in kwargs["metadata"]
    assert "memory_salience" in kwargs["metadata"]
    assert "affective_complexity" in kwargs["metadata"]
    assert new_state.cold.evolution_log


@pytest.mark.asyncio
async def test_memory_consolidation_includes_imagination_memory_pressure():
    commit_interaction = AsyncCallRecorder()
    memory_facade = SimpleNamespace(commit_interaction=commit_interaction)
    container = SimpleNamespace(
        get=lambda name, default=None: memory_facade if name == "memory_facade" else default
    )
    phase = MemoryConsolidationPhase(container)

    state = AuraState.default()
    state.response_modifiers["imagination_memory_pressure"] = 0.82
    state.response_modifiers["imagination_verification_pressure"] = 0.67
    state.cognition.current_origin = "desktop"
    state.cognition.working_memory.extend(
        [
            {"role": "user", "content": "Imagine the workflow, then verify it.", "timestamp": 1.0},
            {"role": "assistant", "content": "I modeled it and separated imagined steps from verified ones.", "timestamp": 2.0},
        ]
    )

    await phase.execute(state, objective="Imagine the workflow, then verify it.")

    assert len(commit_interaction.calls) == 1
    _, kwargs = commit_interaction.calls[0]
    metadata = kwargs["metadata"]
    assert metadata["memory_salience"] == pytest.approx(0.82)
    assert metadata["imagination_memory_pressure"] == pytest.approx(0.82)
    assert metadata["imagination_verification_pressure"] == pytest.approx(0.67)


@pytest.mark.asyncio
async def test_memory_consolidation_includes_bicameral_memory_pressure():
    commit_interaction = AsyncCallRecorder()
    memory_facade = SimpleNamespace(commit_interaction=commit_interaction)
    container = SimpleNamespace(
        get=lambda name, default=None: memory_facade if name == "memory_facade" else default
    )
    phase = MemoryConsolidationPhase(container)

    state = AuraState.default()
    state.response_modifiers["bicameral_memory_priority"] = 0.79
    state.response_modifiers["bicameral_verification_pressure"] = 0.66
    state.cognition.current_origin = "desktop"
    state.cognition.working_memory.extend(
        [
            {"role": "user", "content": "Reflect on what you can remember and verify.", "timestamp": 1.0},
            {"role": "assistant", "content": "I checked the assumption and marked what should be remembered.", "timestamp": 2.0},
        ]
    )

    await phase.execute(state, objective="Reflect on verified memory.")

    assert len(commit_interaction.calls) == 1
    _, kwargs = commit_interaction.calls[0]
    metadata = kwargs["metadata"]
    assert metadata["memory_salience"] == pytest.approx(0.79)
    assert metadata["bicameral_memory_priority"] == pytest.approx(0.79)
    assert metadata["bicameral_verification_pressure"] == pytest.approx(0.66)


@pytest.mark.asyncio
async def test_memory_consolidation_queues_failed_facade_commit_for_retry():
    class _MemoryFacade:
        async def commit_interaction(self, **kwargs):
            assert kwargs["action"] == "conversation_reply"
            raise OSError("database unavailable")

    class _DeadLetterQueue:
        def __init__(self):
            self.entries = []

        def push(self, skill_name, params, error):
            self.entries.append((skill_name, params, error))
            return "entry-1"

    dlq = _DeadLetterQueue()

    def service_get(name, default=None):
        if name == "memory_facade":
            return _MemoryFacade()
        if name == "dead_letter_queue":
            return dlq
        return default

    get_degradation_tracker().reset()
    phase = MemoryConsolidationPhase(SimpleNamespace(get=service_get))
    state = AuraState.default()
    state.cognition.working_memory.extend(
        [
            {"role": "user", "content": "Remember this preference.", "timestamp": 1.0},
            {"role": "assistant", "content": "I will preserve it.", "timestamp": 2.0},
        ]
    )

    new_state = await phase.execute(state, objective="memory retry")

    assert new_state.cold.evolution_log
    assert new_state.cognition.modifiers["memory_consolidation_status"]["status"] == "degraded"
    assert dlq.entries[0][0] == "memory_consolidation.commit_interaction"
    assert "Remember this preference" in dlq.entries[0][1]["context"]
    recent = get_degradation_tracker().recent(subsystem="memory_consolidation")
    assert "queued failed memory facade commit" in recent[-1].action
