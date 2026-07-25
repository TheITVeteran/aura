"""Curiosity must be curious ABOUT something.

The live 2026-07-25 idle hour ran nine identical web searches for
"What do I not know about something new?" and came back with Windows release
notes, which the pipeline then tried to file as learning. The cause was a
caller substituting the literal string ``"something new"`` whenever there was
no current objective — a placeholder that reads like a topic and is not one.

Feeling curious with no object is boredom. It is not a research question, and
asking the same non-question repeatedly is a stuck loop wearing curiosity's
clothes.
"""
from __future__ import annotations

import pytest

from core.agi.curiosity_explorer import CuriosityExplorer

pytestmark = pytest.mark.unit


@pytest.fixture()
def explorer(monkeypatch):
    monkeypatch.setattr(
        "core.agi.curiosity_explorer._background_learning_allowed", lambda o: True
    )
    return CuriosityExplorer()


class TestPlaceholderTopicsAreRefused:
    @pytest.mark.parametrize(
        "topic", ["something new", "current interests", "unknown", "N/A", "  ", None]
    )
    def test_a_placeholder_produces_no_exploration(self, explorer, topic):
        explorer.tick(curiosity=0.9, active_topic=topic)
        assert explorer._queue == []

    def test_a_real_topic_is_explored(self, explorer):
        explorer.tick(curiosity=0.9, active_topic="MLX unified memory limits")
        assert len(explorer._queue) == 1
        assert "MLX unified memory limits" in explorer._queue[0].question

    def test_explicit_gaps_still_work_with_a_real_topic(self, explorer):
        explorer.tick(
            curiosity=0.9,
            active_topic="lease reaping",
            knowledge_gaps=["why does a lease outlive its holder?"],
        )
        assert explorer._queue[0].question == "why does a lease outlive its holder?"


class TestTheSameQuestionIsNotReasked:
    def test_a_queued_question_is_not_queued_twice(self, explorer):
        for _ in range(5):
            explorer._last_exploration = 0.0
            explorer.tick(curiosity=0.9, active_topic="lease reaping")
        assert len(explorer._queue) == 1

    def test_case_and_spacing_do_not_defeat_the_guard(self, explorer):
        explorer.tick(curiosity=0.9, active_topic="x", knowledge_gaps=["Why  IS it Slow?"])
        explorer._last_exploration = 0.0
        explorer.tick(curiosity=0.9, active_topic="x", knowledge_gaps=["why is it slow?"])
        assert len(explorer._queue) == 1

    def test_an_already_answered_question_is_not_re_explored(self, explorer):
        explorer._findings.append({"question": "why is it slow?", "finding": "cache"})
        explorer.tick(curiosity=0.9, active_topic="x", knowledge_gaps=["why is it slow?"])
        assert explorer._queue == []

    def test_a_genuinely_new_question_still_gets_through(self, explorer):
        explorer._findings.append({"question": "why is it slow?", "finding": "cache"})
        explorer.tick(curiosity=0.9, active_topic="x", knowledge_gaps=["why is it fast?"])
        assert len(explorer._queue) == 1


class TestAffectLoopDoesNotInventTopics:
    """Drives the real loop; the positive control proves the path is reached."""

    @staticmethod
    def _install(monkeypatch, objective: str) -> list[dict]:
        from core.evolution import singularity_loops as sl

        ticks: list[dict] = []

        class FakeCuriosity:
            def tick(self, **kwargs):
                ticks.append(kwargs)

        class FakeRepo:
            async def get_current(self):
                cognition = type("C", (), {"current_objective": objective})()
                return type("S", (), {"cognition": cognition})()

        def fake_get(name, default=None):
            if name == "curiosity_explorer":
                return FakeCuriosity()
            if name == "state_repository":
                return FakeRepo()
            if name == "affect_engine":
                return type(
                    "A", (), {"get_state_sync": lambda self: {"valence": 0.1, "arousal": 0.1}}
                )()
            return default

        monkeypatch.setattr(sl.ServiceContainer, "get", staticmethod(fake_get))
        return ticks

    @pytest.mark.asyncio
    async def test_boredom_without_an_objective_explores_nothing(self, monkeypatch):
        """The exact live shape: bored, no objective, must not fabricate a topic."""
        from core.evolution.singularity_loops import SingularityLoops

        ticks = self._install(monkeypatch, "")
        await SingularityLoops()._loop_affect_to_exploration()

        assert ticks == [], "boredom with no objective must not invent a research topic"

    @pytest.mark.asyncio
    async def test_boredom_with_a_real_objective_still_explores(self, monkeypatch):
        from core.evolution.singularity_loops import SingularityLoops

        ticks = self._install(monkeypatch, "why the lane wedges after warmup")
        await SingularityLoops()._loop_affect_to_exploration()

        assert len(ticks) == 1, "the guard must not disable genuine curiosity"
        assert ticks[0]["active_topic"] == "why the lane wedges after warmup"
