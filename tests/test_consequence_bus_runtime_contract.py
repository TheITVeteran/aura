from __future__ import annotations

import pytest

from core.being.body_state_service import BodyStateService
from core.being.welfare_state import WelfareState
from core.being.welfare_transaction import WelfareTransaction
from core.runtime.consequence_bus import ConsequenceBus, ConsequenceEvent
from core.runtime.errors import get_degradation_tracker


def _event() -> ConsequenceEvent:
    return ConsequenceEvent(
        event_id="evt-test",
        timestamp=1.0,
        source="test",
        domain="tool_execution",
        action_content="exercise consequence bus",
    )


def test_consequence_bus_records_expected_subscriber_failures_and_continues():
    tracker = get_degradation_tracker()
    tracker.reset()
    bus = ConsequenceBus()
    delivered: list[str] = []

    def failing_subscriber(_event: ConsequenceEvent) -> None:
        delivered.append("failing")
        raise RuntimeError("subscriber unavailable")

    def healthy_subscriber(_event: ConsequenceEvent) -> None:
        delivered.append("healthy")

    bus.subscribe("tool_execution", failing_subscriber)
    bus.subscribe("tool_execution", healthy_subscriber)

    bus.publish(_event())

    assert delivered == ["failing", "healthy"]
    recent = tracker.recent(subsystem="consequence_bus", limit=1)
    assert recent
    assert recent[0].action == "continued consequence broadcast after subscriber delivery failed"


def test_consequence_bus_does_not_swallow_unexpected_subscriber_exception_classes():
    bus = ConsequenceBus()
    delivered: list[str] = []

    class UnexpectedSubscriberFailure(Exception):
        def __init__(self) -> None:
            super().__init__("unexpected subscriber failure")

    def failing_subscriber(_event: ConsequenceEvent) -> None:
        delivered.append("unexpected")
        raise UnexpectedSubscriberFailure()

    bus.subscribe("tool_execution", failing_subscriber)

    with pytest.raises(UnexpectedSubscriberFailure):
        bus.publish(_event())
    assert delivered == ["unexpected"]


def test_body_state_service_records_consequence_subscription_failure(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()
    attempts: list[str] = []

    def unavailable_bus():
        attempts.append("get")
        raise RuntimeError("bus unavailable")

    monkeypatch.setattr(
        "core.being.body_state_service.ConsequenceBus.get",
        unavailable_bus,
    )
    service = BodyStateService()

    service._subscribe_consequences()

    assert attempts == ["get"]
    assert service._consequence_subscribed is False
    recent = tracker.recent(subsystem="body_state_service", limit=1)
    assert recent
    assert recent[0].action == "continued without consequence-bus body feedback subscription"


def test_welfare_state_records_consequence_subscription_failure(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()
    attempts: list[str] = []

    def unavailable_bus():
        attempts.append("get")
        raise RuntimeError("bus unavailable")

    monkeypatch.setattr(
        "core.being.welfare_state.ConsequenceBus.get",
        unavailable_bus,
    )
    welfare = WelfareState()

    welfare._subscribe_consequences()

    assert attempts == ["get"]
    assert welfare._consequence_subscribed is False
    recent = tracker.recent(subsystem="welfare_state", limit=1)
    assert recent
    assert recent[0].action == "continued without consequence-bus welfare feedback subscription"


def test_welfare_transaction_records_consequence_publication_failure(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()
    attempts: list[str] = []

    def unavailable_bus():
        attempts.append("get")
        raise RuntimeError("bus unavailable")

    monkeypatch.setattr(
        "core.being.welfare_transaction.ConsequenceBus.get",
        unavailable_bus,
    )
    tx = WelfareTransaction.begin(domain="tool_execution", action="exercise transaction")

    record = tx.complete(outcome="success")

    assert record.outcome == "success"
    assert attempts == ["get"]
    recent = tracker.recent(subsystem="welfare_transaction", limit=1)
    assert recent
    assert recent[0].action == "continued after consequence-bus transaction publication failed"
