import asyncio
from types import SimpleNamespace

import pytest

from core.phases.learning_phase import LearningPhase
from core.state.aura_state import AuraState


class AsyncCallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


@pytest.mark.asyncio
async def test_learning_phase_schedules_enrichment_and_flags_distillation(monkeypatch):
    phase = LearningPhase(SimpleNamespace())
    phase._learner = SimpleNamespace(record_tick=lambda **_kwargs: SimpleNamespace(raw_score=0.4))

    state = AuraState.default()
    state.cognition.current_origin = "api"
    state.cognition.last_response = "How can I help you today?"
    state.cognition.working_memory.extend(
        [
            {"role": "user", "content": 'Tell me who wrote "Beautiful Mind".'},
            {"role": "assistant", "content": "How can I help you today?"},
        ]
    )
    state.response_modifiers["response_contract"] = {
        "requires_search": True,
        "tool_evidence_available": False,
    }

    enrich_from_conversation = AsyncCallRecorder({"facts": 1})
    flag_for_distillation = AsyncCallRecorder()
    enricher = SimpleNamespace(enrich_from_conversation=enrich_from_conversation)
    distill = SimpleNamespace(flag_for_distillation=flag_for_distillation)
    tasks = []
    original_create_task = asyncio.create_task

    def _patched_create_task(coro, name=None):
        task = original_create_task(coro, name=name)
        tasks.append(task)
        return task

    monkeypatch.setattr("core.cognition.knowledge_enrichment.get_enricher", lambda **_kwargs: enricher)
    monkeypatch.setattr("core.adaptation.distillation_pipe.get_distillation_pipe", lambda: distill)
    tracker = SimpleNamespace(create_task=_patched_create_task, track_task=lambda _task: None)
    monkeypatch.setattr("core.utils.task_tracker.get_task_tracker", lambda: tracker)
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda _name, default=None: default),
    )

    state = await phase._perform_standard_learning(state, 'Tell me who wrote "Beautiful Mind".')
    await phase._wire_conversation_learning(state, 'Tell me who wrote "Beautiful Mind".')
    if tasks:
        await asyncio.gather(*tasks)

    assert len(enrich_from_conversation.calls) == 1
    assert len(flag_for_distillation.calls) == 1
    _, kwargs = flag_for_distillation.calls[0]
    assert "affect_signature" in kwargs["context"]


@pytest.mark.asyncio
async def test_learning_phase_flags_prompt_fishing_dialogue_failures(monkeypatch):
    phase = LearningPhase(SimpleNamespace())
    phase._learner = SimpleNamespace(record_tick=lambda **_kwargs: SimpleNamespace(raw_score=0.9))

    state = AuraState.default()
    state.cognition.current_origin = "api"
    state.cognition.last_response = "Blue is a great color. What about you?"
    state.cognition.working_memory.extend(
        [
            {"role": "user", "content": "Why do you like blue?"},
            {"role": "assistant", "content": "Blue is a great color. What about you?"},
        ]
    )
    state.response_modifiers["response_contract"] = {
        "requires_aura_stance": True,
        "prefers_dialogue_participation": True,
        "avoid_question_fishing": True,
    }

    enrich_from_conversation = AsyncCallRecorder({"facts": 1})
    flag_for_distillation = AsyncCallRecorder()
    enricher = SimpleNamespace(enrich_from_conversation=enrich_from_conversation)
    distill = SimpleNamespace(flag_for_distillation=flag_for_distillation)
    tasks = []
    original_create_task = asyncio.create_task

    def _patched_create_task(coro, name=None):
        task = original_create_task(coro, name=name)
        tasks.append(task)
        return task

    monkeypatch.setattr("core.cognition.knowledge_enrichment.get_enricher", lambda **_kwargs: enricher)
    monkeypatch.setattr("core.adaptation.distillation_pipe.get_distillation_pipe", lambda: distill)
    tracker = SimpleNamespace(create_task=_patched_create_task, track_task=lambda _task: None)
    monkeypatch.setattr("core.utils.task_tracker.get_task_tracker", lambda: tracker)
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda _name, default=None: default),
    )

    await phase._wire_conversation_learning(state, "Why do you like blue?")
    if tasks:
        await asyncio.gather(*tasks)

    assert len(flag_for_distillation.calls) == 1
    _, kwargs = flag_for_distillation.calls[0]
    assert "dialogue_validation" in kwargs["context"]
    assert "affect_signature" in kwargs["context"]
