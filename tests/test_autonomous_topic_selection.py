from __future__ import annotations

from types import SimpleNamespace


def test_topic_selection_prefers_shared_context_over_static_interests(monkeypatch):
    from core.autonomy.topic_selection import select_autonomous_topic
    from core.container import ServiceContainer

    services = {
        "initiative_synthesizer": None,
        "goal_engine": None,
        "identity": None,
        "knowledge_graph": None,
        "drive_engine": SimpleNamespace(latent_interests=["fallback latent interest"]),
    }
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None, **kwargs: services.get(name, default)),
    )
    state = SimpleNamespace(
        cognition=SimpleNamespace(
            working_memory=[
                {
                    "role": "user",
                    "content": "I keep wondering how octopuses coordinate their distributed nervous systems.",
                }
            ],
            pending_initiatives=[],
        ),
        motivation=SimpleNamespace(latent_interests=["fallback latent interest"]),
    )

    selected = select_autonomous_topic(SimpleNamespace(), state)

    assert selected is not None
    assert selected.source == "conversation"
    assert "octopuses" in selected.text


def test_topic_selection_uses_unresolved_tension_without_scripted_fallback(monkeypatch):
    from core.autonomy.topic_selection import select_autonomous_topic
    from core.container import ServiceContainer

    synthesizer = SimpleNamespace(
        get_tensions=lambda: [
            {
                "content": "Why did the last desktop action lose visual focus?",
                "urgency": 0.7,
            }
        ]
    )
    services = {
        "initiative_synthesizer": synthesizer,
        "goal_engine": None,
        "identity": None,
        "knowledge_graph": None,
        "drive_engine": None,
    }
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None, **kwargs: services.get(name, default)),
    )

    selected = select_autonomous_topic(SimpleNamespace(), SimpleNamespace(cognition=None))

    assert selected is not None
    assert selected.source == "unresolved_tension"
    assert "visual focus" in selected.text


def test_topic_selection_returns_none_without_grounded_sources(monkeypatch):
    from core.autonomy.topic_selection import select_autonomous_topic
    from core.container import ServiceContainer

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None, **kwargs: default),
    )

    assert select_autonomous_topic(SimpleNamespace(), SimpleNamespace(cognition=None)) is None


def test_topic_selection_does_not_recycle_stale_desktop_action_objectives(monkeypatch):
    from core.autonomy.topic_selection import select_autonomous_topic
    from core.container import ServiceContainer

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None, **kwargs: default),
    )
    state = SimpleNamespace(
        cognition=SimpleNamespace(
            working_memory=[],
            pending_initiatives=[
                {
                    "goal": (
                        "Stalled goal: Please create a folder named 'Aura Live Proof' "
                        "in my Documents folder and write a file named live_test.txt in it."
                    )
                }
            ],
        ),
        motivation=SimpleNamespace(latent_interests=[]),
    )

    assert select_autonomous_topic(SimpleNamespace(), state) is None


def test_failed_curiosity_search_remains_retryable():
    import asyncio

    from core.curiosity_engine import CuriosityEngine, CuriosityTopic

    orchestrator = SimpleNamespace(
        is_busy=False,
        execute_tool=lambda *args, **kwargs: _failed_search(),
        knowledge_graph=None,
    )
    engine = CuriosityEngine(orchestrator=orchestrator)
    topic = CuriosityTopic("distributed cognition", "test", 0.8, explored=True)

    asyncio.run(engine._explore(topic))

    assert topic.explored is False
    assert "distributed cognition" not in engine.explored_topics


def test_curiosity_explore_skips_prompt_shaped_self_prediction_topic():
    import asyncio

    from core.curiosity_engine import CuriosityEngine, CuriosityTopic

    calls = []

    async def execute_tool(*args, **kwargs):
        calls.append((args, kwargs))
        return {"ok": True, "answer": "should not run"}

    orchestrator = SimpleNamespace(
        is_busy=False,
        execute_tool=execute_tool,
        knowledge_graph=None,
    )
    engine = CuriosityEngine(orchestrator=orchestrator)
    topic = CuriosityTopic(
        "Predict how self will react if I take this action: {'type': 'autonomous_goal', "
        "'description': 'Mastery of: User asked about: I am about to give you a smarter brain'}",
        "test",
        0.9,
        explored=True,
    )

    asyncio.run(engine._explore(topic))

    assert calls == []
    assert topic.explored is True


async def _failed_search():
    return {"ok": False, "error": "offline"}
