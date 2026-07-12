from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.orchestrator.mixins.incoming_logic import IncomingLogicMixin


class FakeEstimator:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def observe_message(self, user_id, message, **kwargs):
        self.events.append(("observe", (user_id, message, kwargs.get("persist"))))

    def cognitive_snapshot(self, user_id, _now=None):
        self.events.append(("snapshot", user_id))
        return {
            "agent_id": user_id,
            "confidence": 0.8,
            "observations": 1,
            "affect_hypotheses": {},
        }

    def save_if_due(self):
        self.events.append(("persist", True))
        return True

    def record_response(self, user_id, response_text):
        self.events.append(("response", (user_id, response_text)))


class Harness(IncomingLogicMixin):
    def __init__(self) -> None:
        self.user_identity = {"name": "alice"}
        self.tasks: list[asyncio.Task] = []

    def _fire_and_forget(self, awaitable, *, name):
        task = asyncio.create_task(awaitable, name=name)
        self.tasks.append(task)
        return task


class FakeOutputGate:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.emitted: list[tuple[str, str, str, dict[str, object]]] = []

    async def emit(self, response_text, *, origin, target, **kwargs):
        if self.fail:
            raise RuntimeError("delivery failed")
        self.emitted.append((response_text, origin, target, kwargs))


@pytest.mark.asyncio
async def test_incoming_social_state_is_exact_current_and_ready_before_cognition() -> None:
    estimator = FakeEstimator()
    ServiceContainer.clear()
    ServiceContainer.register_instance("other_agent_model", estimator, required=False)
    harness = Harness()
    context = {"user_id": "bryan"}
    live_state = SimpleNamespace(cognition=SimpleNamespace(current_partner=""))

    try:
        user_id = harness._observe_social_turn(
            context,
            "this is still broken and urgent",
            live_state,
        )
        assert user_id == "bryan"
        assert context["user_id"] == "bryan"
        assert context["social_situation"]["agent_id"] == "bryan"
        assert live_state.cognition.current_partner == "bryan"
        assert estimator.events[:2] == [
            ("observe", ("bryan", "this is still broken and urgent", False)),
            ("snapshot", "bryan"),
        ]
        assert harness.tasks
        await asyncio.gather(*harness.tasks)
        assert estimator.events[-1] == ("persist", True)
    finally:
        ServiceContainer.clear()


def test_only_delivered_response_is_paired_to_exact_user() -> None:
    estimator = FakeEstimator()
    ServiceContainer.clear()
    ServiceContainer.register_instance("other_agent_model", estimator, required=False)
    harness = Harness()

    try:
        harness._record_delivered_social_response(
            {"user_id": "bryan"},
            "verified response",
        )
        assert estimator.events == [("response", ("bryan", "verified response"))]
    finally:
        ServiceContainer.clear()


@pytest.mark.asyncio
async def test_emit_records_feedback_context_only_after_successful_delivery() -> None:
    estimator = FakeEstimator()
    ServiceContainer.clear()
    ServiceContainer.register_instance("other_agent_model", estimator, required=False)
    harness = Harness()
    harness.output_gate = FakeOutputGate()

    try:
        await harness._emit_user_response(
            {"user_id": "bryan"},
            "delivered",
            origin="user",
            metadata={"voice": True},
        )
        assert harness.output_gate.emitted == [
            ("delivered", "user", "primary", {"metadata": {"voice": True}})
        ]
        assert estimator.events == [("response", ("bryan", "delivered"))]

        harness.output_gate = FakeOutputGate(fail=True)
        with pytest.raises(RuntimeError, match="delivery failed"):
            await harness._emit_user_response(
                {"user_id": "bryan"},
                "not delivered",
                origin="user",
            )
        assert estimator.events == [("response", ("bryan", "delivered"))]
    finally:
        ServiceContainer.clear()


def test_social_user_id_resolution_never_selects_an_unrelated_agent() -> None:
    harness = Harness()

    assert harness._resolve_social_user_id({"user_id": "bryan"}) == "bryan"
    assert harness._resolve_social_user_id({}) == "alice"
    assert harness._resolve_social_user_id({"user_id": " "}) == "alice"
    assert len(harness._resolve_social_user_id({"user_id": "x" * 500})) == 160

    harness.user_identity = {}
    assert harness._resolve_social_user_id({}) == "local_user"
