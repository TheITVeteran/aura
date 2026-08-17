from __future__ import annotations

import pytest


def test_foreground_chat_dependency_materializer_resolves_complete_read_path(
    monkeypatch,
) -> None:
    import core.agency.intention_loop as intention_module
    import core.consciousness.temporal_finitude as finitude_module
    import core.conversation.unified_transcript as transcript_module
    import core.social.social_imagination as social_module
    from interface import chat_dependencies

    class _CognitiveEngine:
        async def think(self, _message):
            return "ok"

    class _CapabilityEngine:
        def get_available_skills(self):
            return ["web_search", "desktop_task"]

    services = {
        "cognitive_engine": _CognitiveEngine(),
        "capability_engine": _CapabilityEngine(),
    }
    monkeypatch.setattr(
        chat_dependencies.ServiceContainer,
        "get",
        classmethod(lambda _cls, key, default=None: services.get(key, default)),
    )
    monkeypatch.setattr(
        transcript_module.UnifiedTranscript,
        "get_instance",
        classmethod(lambda _cls: object()),
    )
    monkeypatch.setattr(finitude_module, "get_temporal_finitude_model", object)
    monkeypatch.setattr(social_module, "get_social_imagination", object)
    monkeypatch.setattr(intention_module, "get_intention_loop", object)
    monkeypatch.setattr(
        chat_dependencies,
        "_materialize_expression_path",
        lambda: {
            "elapsed_ms": 12.5,
            "contract_type": "ResponseContract",
            "requires_live_grounding": True,
        },
    )

    receipt = chat_dependencies.materialize_foreground_chat_dependencies()

    assert receipt["skill_count"] == 2
    assert receipt["cognitive_engine"] == "_CognitiveEngine"
    assert receipt["capability_engine"] == "_CapabilityEngine"
    assert receipt["expression_path"]["contract_type"] == "ResponseContract"


def test_foreground_chat_dependency_materializer_refuses_empty_catalog(
    monkeypatch,
) -> None:
    from interface import chat_dependencies

    class _CognitiveEngine:
        async def think(self, _message):
            return "ok"

    class _CapabilityEngine:
        def get_available_skills(self):
            return []

    services = {
        "cognitive_engine": _CognitiveEngine(),
        "capability_engine": _CapabilityEngine(),
    }
    monkeypatch.setattr(
        chat_dependencies.ServiceContainer,
        "get",
        classmethod(lambda _cls, key, default=None: services.get(key, default)),
    )

    with pytest.raises(RuntimeError, match="catalog is empty"):
        chat_dependencies.materialize_foreground_chat_dependencies()
