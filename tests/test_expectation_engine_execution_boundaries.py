from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.coordinators.cognitive_coordinator import CognitiveCoordinator
from core.orchestrator.mixins.message_pipeline import MessagePipelineMixin
from core.world_model.expectation_engine import (
    ExpectationEngine,
    result_supports_learning,
)


@pytest.mark.parametrize(
    "result",
    [
        {"ok": False, "status": "deferred", "error": "orchestrator_busy"},
        "{'ok': False, 'status': 'deferred', 'error': 'orchestrator_busy'}",
        {"success": False, "reason": "temporal_obligation_active:repair"},
        {"status": "admission-denied"},
        "WILL REFUSED: standing authority lease missing",
    ],
)
def test_non_executed_results_do_not_support_learning(result) -> None:
    assert result_supports_learning(result) is False


@pytest.mark.asyncio
async def test_deferred_result_never_enters_cognitive_extraction() -> None:
    class Brain:
        async def generate(self, *_args, **_kwargs):
            raise AssertionError("deferred result entered direct generation")

        async def think(self, *_args, **_kwargs):
            raise AssertionError("deferred result entered task cognition")

    engine = ExpectationEngine(Brain())

    await engine.update_beliefs_from_result(
        "web_search",
        {"ok": False, "status": "deferred", "error": "orchestrator_busy"},
    )


@pytest.mark.asyncio
async def test_successful_unknown_result_uses_tool_disabled_generation(monkeypatch) -> None:
    import core.world_model.belief_graph as belief_module

    calls: list[tuple[str, dict]] = []
    updates: list[tuple[str, str, str, float]] = []

    class Brain:
        async def generate(self, prompt, **kwargs):
            calls.append((prompt, dict(kwargs)))
            return "Europa | evidence_state | observed"

        async def think(self, *_args, **_kwargs):
            raise AssertionError("utility extraction entered the task pipeline")

    class BeliefGraph:
        @staticmethod
        def detect_contradiction(_entity, _relation, _target):
            return None

        @staticmethod
        def update_belief(entity, relation, target, *, confidence_score):
            updates.append((entity, relation, target, confidence_score))

    monkeypatch.setattr(belief_module, "belief_graph", BeliefGraph())
    engine = ExpectationEngine(Brain())

    await engine.update_beliefs_from_result(
        "web_search",
        {"ok": True, "results": [{"title": "Europa ocean evidence"}]},
    )

    assert len(calls) == 1
    _prompt, kwargs = calls[0]
    assert kwargs["origin"] == "expectation_engine"
    assert kwargs["purpose"] == "internal_analysis"
    assert kwargs["allow_tools"] is False
    assert kwargs["use_strategies"] is False
    assert updates == [("Europa", "evidence_state", "observed", 0.8)]


@pytest.mark.asyncio
async def test_message_pipeline_skips_surprise_for_deferred_result() -> None:
    class Brain:
        async def generate(self, *_args, **_kwargs):
            raise AssertionError("deferred result entered surprise generation")

        async def think(self, *_args, **_kwargs):
            raise AssertionError("deferred result entered surprise cognition")

    pipeline = object.__new__(MessagePipelineMixin)
    pipeline.cognitive_engine = Brain()

    should_rethink = await pipeline._check_surprise_and_learn(
        SimpleNamespace(expectation="search executes"),
        {"ok": False, "status": "deferred", "error": "orchestrator_busy"},
        "web_search",
    )

    assert should_rethink is False


@pytest.mark.asyncio
async def test_cognitive_coordinator_skips_surprise_for_deferred_result() -> None:
    class Brain:
        async def generate(self, *_args, **_kwargs):
            raise AssertionError("deferred result entered surprise generation")

        async def think(self, *_args, **_kwargs):
            raise AssertionError("deferred result entered surprise cognition")

    coordinator = object.__new__(CognitiveCoordinator)
    coordinator.orch = SimpleNamespace(cognitive_engine=Brain())

    should_rethink = await coordinator.check_surprise_and_learn(
        SimpleNamespace(expectation="search executes"),
        {"ok": False, "status": "deferred", "error": "orchestrator_busy"},
        "web_search",
    )

    assert should_rethink is False
