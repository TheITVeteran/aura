import asyncio
import threading
from types import SimpleNamespace

import pytest

from core.runtime import conversation_support
from core.social.relationship_graph import RelationshipConsentRequiredError
from core.state.aura_state import AuraState


@pytest.mark.asyncio
async def test_dialogue_cognition_cold_start_does_not_block_event_loop(monkeypatch):
    release = threading.Event()

    def _blocking_dialogue_resolution(*, default=None):
        release.wait(1.0)
        return default

    monkeypatch.setattr(
        conversation_support.service_access,
        "optional_service",
        lambda *_args, **kwargs: kwargs.get("default"),
    )
    monkeypatch.setattr(
        conversation_support.service_access,
        "resolve_dialogue_cognition",
        _blocking_dialogue_resolution,
    )

    loop = asyncio.get_running_loop()
    loop.call_later(0.05, release.set)
    started_at = loop.time()

    await conversation_support.update_conversational_intelligence(
        "Still with me?",
        "I am here.",
        None,
        agent_id="owner",
    )

    assert loop.time() - started_at < 0.4


@pytest.mark.asyncio
async def test_shared_ground_callback_abstains_without_exact_partner(monkeypatch):
    from core.memory import shared_ground as shared_ground_module

    class IdentitylessSharedGround:
        MAX_ENTRIES = 100
        active_agent_id = ""

        def get_top_entries(self, *_args, **_kwargs):
            raise AssertionError("identity-less callback queried relational memory")

    monkeypatch.setattr(
        shared_ground_module,
        "get_shared_ground",
        lambda: IdentitylessSharedGround(),
    )

    await conversation_support.record_shared_ground_callbacks("a normal response")


@pytest.mark.asyncio
async def test_shared_ground_callback_keeps_scheduled_partner_identity(monkeypatch):
    from core.memory import shared_ground as shared_ground_module

    calls = []

    class SharedGround:
        MAX_ENTRIES = 100
        active_agent_id = "someone-else"

        def get_top_entries(self, limit, *, agent_id):
            calls.append(("query", limit, agent_id))
            return [SimpleNamespace(reference="shared project milestone")]

        def record_callback(self, reference, *, agent_id):
            calls.append(("callback", reference, agent_id))
            return True

    monkeypatch.setattr(shared_ground_module, "get_shared_ground", lambda: SharedGround())

    await conversation_support.record_shared_ground_callbacks(
        "That shared project milestone still matters.",
        agent_id="bryan",
    )

    assert calls == [
        ("query", 100, "bryan"),
        ("callback", "shared project milestone", "bryan"),
    ]


@pytest.mark.asyncio
async def test_identityless_conversation_update_only_runs_global_model(monkeypatch):
    global_updates = []

    class GlobalConversationIntelligence:
        async def update(self, *args):
            global_updates.append(args)

    person_specific = SimpleNamespace(
        update_from_interaction=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity-less work reached a person-specific model")
        )
    )

    def optional_service(*names, default=None):
        if "conversation_intelligence" in names:
            return GlobalConversationIntelligence()
        if any(
            name in names
            for name in (
                "conversational_profiler",
                "humor_engine",
                "relational_intelligence",
            )
        ):
            return person_specific
        return default

    monkeypatch.setattr(conversation_support.service_access, "optional_service", optional_service)
    monkeypatch.setattr(
        conversation_support.service_access,
        "resolve_dialogue_cognition",
        lambda default=None: person_specific,
    )
    monkeypatch.setattr(
        conversation_support.service_access,
        "resolve_social_imagination",
        lambda default=None: person_specific,
    )
    monkeypatch.setattr(
        conversation_support.service_access,
        "resolve_conversational_dynamics",
        lambda default=None: default,
    )
    monkeypatch.setattr(conversation_support, "resolve_exact_partner_id", lambda _state: None)
    monkeypatch.setattr(
        conversation_support,
        "relational_memory_allows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity-less work performed a consent lookup")
        ),
    )

    await conversation_support.update_conversational_intelligence(
        "hello",
        "hi",
        AuraState.default(),
    )

    assert len(global_updates) == 1


@pytest.mark.asyncio
async def test_scheduler_creates_one_named_bounded_owner_with_stable_partner(monkeypatch):
    calls = []
    scheduling = {}

    async def update(*_args, agent_id=None, **_kwargs):
        calls.append(("update", agent_id))

    async def callback(*_args, agent_id=None, **_kwargs):
        calls.append(("callback", agent_id))

    def create(awaitable, **kwargs):
        scheduling.update(kwargs)
        return asyncio.create_task(awaitable)

    monkeypatch.setattr(
        conversation_support,
        "resolve_exact_partner_id",
        lambda _state: "bryan",
    )
    monkeypatch.setattr(conversation_support, "update_conversational_intelligence", update)
    monkeypatch.setattr(conversation_support, "record_shared_ground_callbacks", callback)
    monkeypatch.setattr(conversation_support, "create_tracked_task", create)

    task = conversation_support.schedule_conversation_support_updates(
        "hello",
        "hi",
        AuraState.default(),
    )
    assert isinstance(task, asyncio.Task)
    await task

    assert scheduling == {
        "name": "conversation_support.turn_updates",
        "owner": "response_generation",
        "bounded": True,
    }
    assert calls == [("update", "bryan"), ("callback", "bryan")]


@pytest.mark.asyncio
async def test_record_conversation_experience_prefers_memory_facade_commit(monkeypatch):
    state = AuraState.default()
    captured = {}

    class DummyFacade:
        async def commit_interaction(self, **kwargs):
            captured.update(kwargs)
            return "episode-1"

    def fake_optional_service(*names, default=None):
        if "memory_facade" in names:
            return DummyFacade()
        if "episodic_memory" in names:
            raise AssertionError("record_conversation_experience should not bypass memory_facade when it exists")
        return default

    monkeypatch.setattr(conversation_support.service_access, "optional_service", fake_optional_service)
    monkeypatch.setattr(
        conversation_support,
        "update_conversational_intelligence",
        lambda *args, **kwargs: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        conversation_support,
        "record_shared_ground_callbacks",
        lambda *args, **kwargs: asyncio.sleep(0),
    )

    await conversation_support.record_conversation_experience(
        "Please explain this architecture in detail.",
        (
            "The architecture routes each request through one governed cognition path, "
            "then verifies effects before committing the result to durable state."
        ),
        state,
    )

    assert captured["action"] == "conversation_reply"
    assert captured["success"] is True
    assert captured["metadata"]["origin"] == "api"
    assert captured["metadata"]["domain"] == "conversation"
    assert captured["metadata"]["objective"] == "Please explain this architecture in detail."
    assert captured["metadata"]["semantic_mode"] == "technical"
    assert captured["metadata"]["learning_admission"] == "verified"


@pytest.mark.asyncio
async def test_record_conversation_experience_adds_searchable_continuity_memory(monkeypatch):
    state = AuraState.default()
    captured = {"commit": None, "continuity": None}

    class DummyFacade:
        async def commit_interaction(self, **kwargs):
            captured["commit"] = kwargs
            return "episode-1"

        async def add_memory(self, text, metadata=None):
            captured["continuity"] = {"text": text, "metadata": dict(metadata or {})}
            return True

    def fake_optional_service(*names, default=None):
        if "memory_facade" in names:
            return DummyFacade()
        return default

    monkeypatch.setattr(conversation_support.service_access, "optional_service", fake_optional_service)
    monkeypatch.setattr(
        conversation_support,
        "update_conversational_intelligence",
        lambda *args, **kwargs: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        conversation_support,
        "record_shared_ground_callbacks",
        lambda *args, **kwargs: asyncio.sleep(0),
    )

    await conversation_support.record_conversation_experience(
        "My laptop crashed when Aura used over 100GB of RAM.",
        "I will treat that as a live desktop reliability fault and preserve the context.",
        state,
    )

    assert captured["commit"] is not None
    continuity = captured["continuity"]
    assert continuity is not None
    assert continuity["text"].startswith("Conversation continuity memory.")
    assert "100GB of RAM" in continuity["text"]
    assert continuity["metadata"]["memory_type"] == "conversation_continuity"
    assert continuity["metadata"]["searchable_conversation_context"] is True
    assert continuity["metadata"]["preserve_for_continuity"] is True
    assert continuity["metadata"]["provenance_source"] == "live_conversation_turn"
    assert continuity["metadata"]["learning_admission"] == "verified"


@pytest.mark.asyncio
async def test_record_conversation_experience_rejects_misgrounded_self_condition(
    monkeypatch,
):
    calls = []

    class DummyFacade:
        async def commit_interaction(self, **kwargs):
            calls.append(("commit", kwargs))
            return "episode-should-not-exist"

        async def add_memory(self, text, metadata=None):
            calls.append(("memory", text, metadata))
            return True

    monkeypatch.setattr(
        conversation_support.service_access,
        "optional_service",
        lambda *names, default=None: (
            DummyFacade() if "memory_facade" in names else default
        ),
    )
    monkeypatch.setattr(
        conversation_support,
        "update_conversational_intelligence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("misgrounded reply reached conversational intelligence")
        ),
    )

    await conversation_support.record_conversation_experience(
        "Are you okay though? Feeling fine?",
        (
            "I am with you. RAM pressure is 75.6% with 15.6 GB available; "
            "CPU load is 25.8% on this host."
        ),
        AuraState.default(),
    )

    assert calls == []


class _TopologyAuthority:
    def __init__(self, operations=()):
        self.operations = set(operations)
        self.calls = []

    def allows(self, agent_id, kind, operation):
        self.calls.append((agent_id, kind, operation))
        return operation in self.operations


class _TopologyGraph:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def register_interaction(self, *args):
        if self.error is not None:
            raise self.error
        self.calls.append(args)


def _configure_topology_recording(monkeypatch, authority, graph, degradations):
    def optional_service(*names, default=None):
        if "relational_memory" in names:
            return authority
        if "entity_graph" in names or "relationship_graph" in names:
            return graph
        return default

    class _CodingMemory:
        def record_conversation_turn(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(conversation_support.service_access, "optional_service", optional_service)
    monkeypatch.setattr(conversation_support, "get_coding_session_memory", _CodingMemory)
    monkeypatch.setattr(
        conversation_support,
        "update_conversational_intelligence",
        lambda *args, **kwargs: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        conversation_support,
        "record_shared_ground_callbacks",
        lambda *args, **kwargs: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        conversation_support,
        "record_degradation",
        lambda *args, **kwargs: degradations.append((args, kwargs)),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operations", "expected_calls"),
    [
        ((), 0),
        (("recall",), 0),
        (("prompt",), 0),
        (("recall", "prompt"), 1),
        (("persist", "recall", "prompt"), 1),
    ],
)
async def test_relationship_topology_requires_exact_recall_and_prompt_consent(
    monkeypatch,
    operations,
    expected_calls,
):
    authority = _TopologyAuthority(operations)
    graph = _TopologyGraph()
    degradations = []
    _configure_topology_recording(monkeypatch, authority, graph, degradations)

    await conversation_support.record_conversation_experience(
        "Please keep this architecture coherent.",
        "I will preserve the verified causal path and report its evidence.",
        AuraState.default(),
        principal_id="bryan",
    )

    assert len(graph.calls) == expected_calls
    if graph.calls:
        assert graph.calls == [
            ("aura_self", "bryan", "conversation", "self", "person")
        ]
    assert degradations == []


@pytest.mark.asyncio
async def test_relationship_topology_abstains_without_exact_principal(monkeypatch):
    authority = _TopologyAuthority(("recall", "prompt"))
    graph = _TopologyGraph()
    degradations = []
    _configure_topology_recording(monkeypatch, authority, graph, degradations)

    await conversation_support.record_conversation_experience(
        "Please keep this architecture coherent.",
        "I will preserve the verified causal path and report its evidence.",
        AuraState.default(),
        principal_id="",
    )

    assert authority.calls == []
    assert graph.calls == []
    assert degradations == []


@pytest.mark.asyncio
async def test_relationship_consent_race_abstains_without_health_degradation(monkeypatch):
    authority = _TopologyAuthority(("recall", "prompt"))
    graph = _TopologyGraph(RelationshipConsentRequiredError("consent revoked"))
    degradations = []
    _configure_topology_recording(monkeypatch, authority, graph, degradations)

    await conversation_support.record_conversation_experience(
        "Please keep this architecture coherent.",
        "I will preserve the verified causal path and report its evidence.",
        AuraState.default(),
        principal_id="bryan",
    )

    assert degradations == []


@pytest.mark.asyncio
async def test_relationship_filesystem_permission_error_remains_operational_failure(monkeypatch):
    authority = _TopologyAuthority(("recall", "prompt"))
    graph = _TopologyGraph(PermissionError("disk denied"))
    degradations = []
    _configure_topology_recording(monkeypatch, authority, graph, degradations)

    with pytest.raises(PermissionError, match="disk denied"):
        await conversation_support.record_conversation_experience(
            "Please keep this architecture coherent.",
            "I will preserve the verified causal path and report its evidence.",
            AuraState.default(),
            principal_id="bryan",
        )

    assert degradations == []
