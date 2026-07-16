from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.cognition.knowledge_enrichment import KnowledgeEnricher
from core.runtime import runtime_settings


@pytest.fixture(autouse=True)
def _isolated_runtime_settings(tmp_path, monkeypatch):
    path = tmp_path / "runtime.json"
    monkeypatch.setenv("AURA_SETTINGS_PATH", str(path))
    runtime_settings.clear_runtime_settings_cache()
    yield path
    runtime_settings.clear_runtime_settings_cache()


def _write_settings(path, values):
    path.write_text(json.dumps(values), encoding="utf-8")
    runtime_settings.clear_runtime_settings_cache()


class _KnowledgeGraph:
    def __init__(self):
        self.knowledge = []
        self.relationships = []

    def add_knowledge(self, **payload):
        self.knowledge.append(payload)
        return f"node-{len(self.knowledge)}"

    def add_relationship(self, *args, **kwargs):
        self.relationships.append((args, kwargs))

    def upsert_relationship(self, *args, **kwargs):
        self.relationships.append((args, kwargs))


class _Brain:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def generate(self, *_args, **_kwargs):
        self.calls += 1
        return self.response


def _messages():
    return [
        {"role": "user", "content": "Bryan prefers detailed release evidence."},
        {"role": "assistant", "content": "I will retain exact verification receipts."},
    ]


@pytest.mark.asyncio
async def test_enrichment_setting_blocks_model_and_storage(_isolated_runtime_settings):
    _write_settings(
        _isolated_runtime_settings,
        {"learning.auto_enrichment_enabled": False},
    )
    graph = _KnowledgeGraph()
    brain = _Brain('[{"type":"fact","content":"must not store"}]')
    enricher = KnowledgeEnricher(graph, brain)

    result = await enricher.enrich_from_conversation(_messages(), force=True)

    assert result == {"facts": 0, "entities": 0, "preferences": 0, "beliefs": 0}
    assert brain.calls == 0
    assert graph.knowledge == []
    assert enricher.get_stats()["last_outcome"] == "disabled_by_runtime_setting"


@pytest.mark.asyncio
async def test_none_model_response_is_quality_rejection_not_runtime_error(
    monkeypatch,
):
    degradations = []
    monkeypatch.setattr(
        "core.cognition.knowledge_enrichment.record_degradation",
        lambda *args, **kwargs: degradations.append((args, kwargs)),
    )
    enricher = KnowledgeEnricher(_KnowledgeGraph(), _Brain(None))

    result = await enricher.enrich_from_conversation(_messages(), force=True)

    assert result["facts"] == 0
    assert degradations == []
    assert enricher.get_stats()["last_outcome"] == "empty_model_response"
    assert enricher.get_stats()["rejected_model_outputs"] == 1


@pytest.mark.asyncio
async def test_invalid_message_container_is_rejected_before_model_call():
    brain = _Brain('[{"type":"fact","content":"must not run"}]')
    enricher = KnowledgeEnricher(_KnowledgeGraph(), brain)

    result = await enricher.enrich_from_conversation(None, force=True)

    assert result == {"facts": 0, "entities": 0, "preferences": 0, "beliefs": 0}
    assert brain.calls == 0
    assert enricher.get_stats()["last_outcome"] == "invalid_messages_container"
    assert enricher.get_stats()["last_attempt"] == 0.0


@pytest.mark.asyncio
async def test_non_mapping_metadata_does_not_break_enrichment():
    graph = _KnowledgeGraph()
    brain = _Brain('[{"type":"fact","content":"Metadata is optional."}]')
    enricher = KnowledgeEnricher(graph, brain)
    messages = [
        {"role": "user", "content": "A sufficiently detailed message.", "metadata": []},
        {"role": "assistant", "content": "A grounded response.", "metadata": "bad"},
    ]

    result = await enricher.enrich_from_conversation(messages, force=True)

    assert result["facts"] == 1
    assert enricher.get_stats()["last_outcome"] == "completed"


@pytest.mark.asyncio
async def test_enrichment_continues_after_independent_storage_failure(monkeypatch):
    class _FailFirstGraph(_KnowledgeGraph):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def add_knowledge(self, **payload):
            self.calls += 1
            if self.calls == 1:
                raise OSError("temporary database failure")
            return super().add_knowledge(**payload)

    graph = _FailFirstGraph()
    brain = _Brain(
        '[{"type":"fact","content":"First valid fact."},'
        '{"type":"fact","content":"Second valid fact."}]'
    )
    degradations = []
    monkeypatch.setattr(
        "core.cognition.knowledge_enrichment.record_degradation",
        lambda *args, **kwargs: degradations.append((args, kwargs)),
    )
    enricher = KnowledgeEnricher(graph, brain)

    result = await enricher.enrich_from_conversation(_messages(), force=True)

    assert result["facts"] == 1
    assert [item["content"] for item in graph.knowledge] == ["Second valid fact."]
    assert enricher.get_stats()["last_outcome"] == "completed_with_storage_errors"
    assert len(degradations) == 1
    assert degradations[0][0][0] == "knowledge_enrichment.storage"
    assert degradations[0][1]["extra"]["error_count"] == 1


@pytest.mark.asyncio
async def test_structured_enrichment_stores_relationships_without_content_field():
    response = {
        "items": [
            {
                "type": "fact",
                "content": "Checksums detect accidental changes.",
                "confidence": 0.9,
            },
            {
                "type": "relationship",
                "entity_a": "checksums",
                "relation": "detect",
                "entity_b": "changes",
                "strength": 0.8,
            },
        ]
    }
    graph = _KnowledgeGraph()
    enricher = KnowledgeEnricher(graph, _Brain(response))

    result = await enricher.enrich_from_conversation(_messages(), force=True)

    assert result["facts"] == 2
    assert len(graph.knowledge) == 1
    assert graph.relationships == [
        (("checksums", "detect", "changes"), {"weight": 0.8})
    ]


@pytest.mark.asyncio
async def test_enrichment_parser_skips_prose_and_decodes_first_json_array():
    brain = _Brain(
        "Result follows:\n"
        '[{"type":"entity","content":"Aura runtime","related_to":[]} ]\n'
        "Unrelated trailing [not-json]."
    )
    graph = _KnowledgeGraph()
    enricher = KnowledgeEnricher(graph, brain)

    result = await enricher.enrich_from_conversation(_messages(), force=True)

    assert result["entities"] == 1
    assert graph.knowledge[0]["content"] == "Aura runtime"


@pytest.mark.asyncio
async def test_enrichment_rejects_structures_that_only_stringify_cleanly():
    brain = _Brain(
        {
            "items": [
                {"type": "fact", "content": {"claim": "not text"}},
                {
                    "type": "relationship",
                    "entity_a": ["Aura"],
                    "entity_b": "runtime",
                    "relation": "uses",
                },
                {"type": 7, "content": "numeric types are not accepted"},
            ]
        }
    )
    graph = _KnowledgeGraph()
    enricher = KnowledgeEnricher(graph, brain)

    result = await enricher.enrich_from_conversation(_messages(), force=True)

    assert result == {"facts": 0, "entities": 0, "preferences": 0, "beliefs": 0}
    assert graph.knowledge == []
    assert graph.relationships == []
    assert enricher.get_stats()["last_outcome"] == "invalid_extraction_items"


@pytest.mark.asyncio
async def test_enrichment_rejects_null_item_fields_without_runtime_degradation(
    monkeypatch,
):
    degradations = []
    monkeypatch.setattr(
        "core.cognition.knowledge_enrichment.record_degradation",
        lambda *args, **kwargs: degradations.append((args, kwargs)),
    )
    brain = _Brain(
        {
            "items": [
                {"type": "fact", "content": None},
                {"type": "entity", "content": None, "related_to": [None]},
                {
                    "type": "relationship",
                    "entity_a": None,
                    "relation": "uses",
                    "entity_b": "runtime",
                },
            ]
        }
    )
    graph = _KnowledgeGraph()
    enricher = KnowledgeEnricher(graph, brain)

    result = await enricher.enrich_from_conversation(_messages(), force=True)

    assert result == {"facts": 0, "entities": 0, "preferences": 0, "beliefs": 0}
    assert graph.knowledge == []
    assert graph.relationships == []
    assert degradations == []
    assert enricher.get_stats()["last_outcome"] == "invalid_extraction_items"


@pytest.mark.asyncio
async def test_conversation_reflection_setting_blocks_generation(
    _isolated_runtime_settings,
):
    from core.conversation_reflection import ConversationReflector

    _write_settings(
        _isolated_runtime_settings,
        {"learning.reflection_enabled": False},
    )

    class _ReflectionBrain:
        def __init__(self):
            self.calls = 0

        async def think(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(content="A durable reflection with enough text.")

    brain = _ReflectionBrain()
    reflector = ConversationReflector()
    result = await reflector.maybe_reflect(
        [*_messages(), *_messages()],
        brain,
    )

    assert result is None
    assert brain.calls == 0
    assert list(reflector.reflections) == []


@pytest.mark.asyncio
async def test_setting_changed_during_reflection_prevents_persistence(monkeypatch):
    from core import conversation_reflection

    checks = iter((True, False))
    monkeypatch.setattr(
        conversation_reflection,
        "_reflection_learning_enabled",
        lambda: next(checks),
    )

    class _ReflectionBrain:
        async def think(self, *_args, **_kwargs):
            return SimpleNamespace(content="A durable reflection with enough text.")

    reflector = conversation_reflection.ConversationReflector()
    result = await reflector.maybe_reflect(
        [*_messages(), *_messages()],
        _ReflectionBrain(),
    )

    assert result is None
    assert list(reflector.reflections) == []


@pytest.mark.asyncio
async def test_pattern_reflector_honors_reflection_learning_setting(
    _isolated_runtime_settings,
):
    from core.conversation.conversation_reflector import ConversationReflector

    _write_settings(
        _isolated_runtime_settings,
        {"learning.reflection_enabled": False},
    )
    opened = []
    reflector = ConversationReflector()
    reflector._inquiry_engine = SimpleNamespace(
        open_question=lambda **payload: opened.append(payload)
    )

    await reflector.reflect_on_history(
        [{"role": "user", "content": "What remains unaddressed?"}]
    )

    assert opened == []
    assert reflector._last_reflection == 0.0


@pytest.mark.asyncio
async def test_learning_phase_does_not_schedule_disabled_enrichment(
    _isolated_runtime_settings,
    monkeypatch,
):
    from core.phases.learning_phase import LearningPhase
    from core.state.aura_state import AuraState

    _write_settings(
        _isolated_runtime_settings,
        {"learning.auto_enrichment_enabled": False},
    )
    monkeypatch.setattr(
        "core.cognition.knowledge_enrichment.get_enricher",
        lambda **_kwargs: pytest.fail("disabled enrichment owner was resolved"),
    )
    state = AuraState.default()
    state.cognition.last_response = "Checksums reveal accidental data changes."
    state.cognition.working_memory.extend(_messages())
    phase = LearningPhase(SimpleNamespace())

    await phase._wire_conversation_learning(
        state,
        "Why do checksums matter?",
    )

    assert state.response_modifiers["knowledge_enrichment"] == {
        "status": "disabled_by_runtime_setting",
        "setting": "learning.auto_enrichment_enabled",
    }
