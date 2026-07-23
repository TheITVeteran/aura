"""CP126 contract tests for the input bus."""
from __future__ import annotations

import math
import threading
import time

import pytest

from core.bus.events import (
    EVENT_SCHEMA_VERSION,
    MAX_DELIVERY_ATTEMPTS,
    MAX_PAYLOAD_BYTES,
    Event,
    EventPriority,
    EventType,
    InputBus,
)


@pytest.fixture()
def bus():
    instance = InputBus()
    yield instance
    instance.shutdown(timeout=2.0)


# --- 03d556a3: envelopes carry a trusted schema ----------------------------


def test_every_event_gets_an_id_and_schema_version(bus):
    bus.publish(Event(type=EventType.SYSTEM, payload={"a": 1}))
    event = bus.next(timeout=1.0)

    assert event.event_id
    assert event.schema_version == EVENT_SCHEMA_VERSION
    assert event.to_dict()["event_id"] == event.event_id


def test_caller_built_events_are_normalized_not_trusted(bus):
    hostile = Event(
        type=EventType.SYSTEM,
        ts=float("inf"),
        topic="t" * 5000,
        source="s" * 500,
        payload={"ok": True},
    )

    bus.publish(hostile)
    event = bus.next(timeout=1.0)

    assert math.isfinite(event.ts)
    assert len(event.topic) <= 256
    assert len(event.source) <= 128
    assert event.envelope_faults


def test_implausible_timestamps_are_replaced(bus):
    bus.publish(Event(ts=1.0, payload={"x": 1}))
    event = bus.next(timeout=1.0)

    assert abs(event.ts - time.time()) < 60
    assert any("implausible" in fault for fault in event.envelope_faults)


def test_oversized_payloads_are_bounded(bus):
    bus.publish(Event(payload={"blob": "x" * (MAX_PAYLOAD_BYTES + 1000)}))
    event = bus.next(timeout=1.0)

    assert event.payload["_truncated"] is True
    assert event.payload["_original_bytes"] > MAX_PAYLOAD_BYTES


def test_unknown_dict_type_is_demoted_with_a_fault(bus):
    bus.publish({"type": "NOT_A_REAL_TYPE", "topic": "x"})
    event = bus.next(timeout=1.0)

    assert event.type is EventType.SYSTEM
    assert any("unknown event type" in fault for fault in event.envelope_faults)


def test_dict_envelope_is_not_swallowed_into_the_payload(bus):
    bus.publish({"type": "system", "topic": "t", "source": "s", "data": 1})
    event = bus.next(timeout=1.0)

    assert event.topic == "t" and event.source == "s"
    assert "topic" not in event.payload
    assert event.payload["data"] == 1


def test_explicit_payload_key_is_honoured(bus):
    bus.publish({"type": "system", "payload": {"real": True}, "topic": "t"})
    event = bus.next(timeout=1.0)

    assert event.payload == {"real": True}


def test_publisher_cannot_buy_queue_position_with_a_timestamp(bus):
    bus.publish(Event(payload={"n": 1}, ts=time.time()))
    bus.publish(Event(payload={"n": 2}, ts=0.0))  # would sort first on ts

    first = bus.next(timeout=1.0)
    second = bus.next(timeout=1.0)

    assert first.payload["n"] == 1
    assert second.payload["n"] == 2


def test_priority_still_outranks_arrival_order(bus):
    bus.publish(Event(payload={"n": 1}, priority=EventPriority.LOW))
    bus.publish(Event(payload={"n": 2}, priority=EventPriority.CRITICAL))

    assert bus.next(timeout=1.0).payload["n"] == 2


def test_idempotency_key_suppresses_duplicates(bus):
    first = bus.publish(Event(payload={"n": 1}, idempotency_key="k1"))
    second = bus.publish(Event(payload={"n": 1}, idempotency_key="k1"))

    assert first.queued is True
    assert second.queued is False and second.duplicate is True
    assert bus.queue_depth == 1


# --- 1ae03583: admission precedes effect -----------------------------------


def test_a_rejected_event_fires_no_side_effects():
    small = InputBus(maxsize=1)
    seen = []
    small.subscribe(EventType.SYSTEM, lambda e: seen.append(e))
    try:
        first = small.emit(EventType.SYSTEM, {"n": 1})
        second = small.emit(EventType.SYSTEM, {"n": 2})

        assert first.queued is True
        assert second.queued is False
        assert second.dropped_reason == "queue_full"
        assert len(seen) == 1
    finally:
        small.shutdown(timeout=2.0)


def test_publish_returns_a_delivery_receipt(bus):
    bus.subscribe(EventType.SYSTEM, lambda e: None)

    receipt = bus.emit(EventType.SYSTEM, {"n": 1})

    assert receipt.queued is True
    assert receipt.notified == 1
    assert receipt.ok is True
    assert receipt.to_dict()["event_id"]


def test_async_publish_enqueues_before_notifying(bus):
    done = threading.Event()
    depth_at_callback = []

    def watcher(event):
        depth_at_callback.append(bus.queue_depth)
        done.set()

    bus.subscribe(EventType.SYSTEM, watcher)
    bus.publish_async(Event(type=EventType.SYSTEM, payload={"n": 1}))

    assert done.wait(timeout=3.0)
    assert depth_at_callback == [1]


def test_async_notification_preserves_submission_order(bus):
    order = []
    ready = threading.Event()

    def collect(event):
        order.append(event.payload["n"])
        if len(order) == 5:
            ready.set()

    bus.subscribe(EventType.SYSTEM, collect)
    for index in range(5):
        bus.publish_async(Event(type=EventType.SYSTEM, payload={"n": index}))

    assert ready.wait(timeout=5.0)
    assert order == [0, 1, 2, 3, 4]


# --- 086d721e: subscribers cannot rewrite the shared event ----------------


def test_event_is_immutable(bus):
    event = Event(type=EventType.SYSTEM, payload={"n": 1})

    with pytest.raises((AttributeError, TypeError)):
        event.priority = EventPriority.CRITICAL
    with pytest.raises((AttributeError, TypeError)):
        event.retry_count = 99


def test_a_subscriber_cannot_change_what_the_next_one_sees(bus):
    observed = []

    def rewriter(event):
        event.payload["injected"] = True
        event.payload["n"] = 999

    def observer(event):
        observed.append(dict(event.payload))

    bus.subscribe(EventType.SYSTEM, rewriter)
    bus.subscribe(EventType.SYSTEM, observer)
    bus.emit(EventType.SYSTEM, {"n": 1})

    assert observed == [{"n": 1}]


def test_a_subscriber_cannot_change_what_the_consumer_sees(bus):
    bus.subscribe(EventType.SYSTEM, lambda e: e.payload.update({"tampered": True}))
    bus.emit(EventType.SYSTEM, {"n": 1})

    queued = bus.next(timeout=1.0)

    assert queued.payload == {"n": 1}


# --- c280c642: DLQ receipts must be true ----------------------------------


def test_three_failing_subscribers_are_one_delivery_not_three_retries(bus):
    def boom(event):
        raise ValueError("nope")

    for index in range(3):
        failing = lambda e, i=index: boom(e)  # noqa: E731 - distinct callables
        failing.__name__ = f"failing_{index}"
        bus.subscribe(EventType.SYSTEM, failing)

    receipt = bus.emit(EventType.SYSTEM, {"n": 1})

    stats = bus.get_dlq_stats()
    assert stats["count"] == 1
    assert len(receipt.failed) == 3
    entry = bus.dlq_events()[0]
    assert entry["delivery_attempts"] == 1
    assert len(entry["failed_subscribers"]) == 3
    assert "retries are disabled" in entry["reason"]


def test_the_same_event_is_dead_lettered_only_once(bus):
    def boom(event):
        raise RuntimeError("nope")

    bus.subscribe(EventType.SYSTEM, boom)
    event = Event(type=EventType.SYSTEM, payload={"n": 1})
    bus.publish(event)
    bus._notify_subscribers(event)
    bus._notify_subscribers(event)

    assert bus.get_dlq_stats()["count"] == 1


def test_nack_requeues_until_the_attempt_budget_is_spent(bus):
    bus.emit(EventType.SYSTEM, {"n": 1})

    attempts = 0
    while True:
        event = bus.next(timeout=1.0)
        if event is None:
            break
        attempts += 1
        bus.nack(event, reason="handler failed")

    assert attempts == MAX_DELIVERY_ATTEMPTS
    assert bus.get_dlq_stats()["count"] == 1
    assert "3 delivery attempt" in bus.dlq_events()[0]["reason"]


def test_opt_in_retries_actually_re_run_the_failing_subscriber():
    retrying = InputBus(retry_failed_subscribers=True)
    calls = []
    done = threading.Event()

    def flaky(event):
        calls.append(1)
        if len(calls) >= MAX_DELIVERY_ATTEMPTS:
            done.set()
        raise ValueError("still failing")

    try:
        retrying.subscribe(EventType.SYSTEM, flaky)
        retrying.emit(EventType.SYSTEM, {"n": 1})

        assert done.wait(timeout=5.0)
        assert len(calls) >= 2
    finally:
        retrying.shutdown(timeout=3.0)


def test_replay_dlq_re_admits_events(bus):
    bus.subscribe(EventType.SYSTEM, lambda e: (_ for _ in ()).throw(ValueError("x")))
    bus.emit(EventType.SYSTEM, {"n": 1})
    bus.next(timeout=1.0)  # drain the original admission
    assert bus.get_dlq_stats()["count"] == 1

    receipts = bus.replay_dlq()

    assert len(receipts) == 1 and receipts[0].queued
    assert bus.get_dlq_stats()["count"] == 0


# --- 2ea4046f: callbacks are fully isolated -------------------------------


@pytest.mark.parametrize(
    "exc", [KeyboardInterrupt, SystemExit, ZeroDivisionError, OSError, IndexError]
)
def test_any_subscriber_exception_is_contained(bus, exc):
    survivors = []

    def exploder(event):
        raise exc("boom")

    bus.subscribe(EventType.SYSTEM, exploder)
    bus.subscribe(EventType.SYSTEM, lambda e: survivors.append(e))

    if issubclass(exc, BaseException) and not issubclass(exc, Exception):
        with pytest.raises(exc):
            bus.emit(EventType.SYSTEM, {"n": 1})
        return

    receipt = bus.emit(EventType.SYSTEM, {"n": 1})

    assert len(survivors) == 1
    assert receipt.failed == ("exploder",)


def test_slow_callbacks_are_reported(bus, caplog):
    bus.callback_timeout_s = 0.01
    bus.subscribe(EventType.SYSTEM, lambda e: time.sleep(0.05))

    with caplog.at_level("WARNING"):
        bus.emit(EventType.SYSTEM, {"n": 1})

    assert any("budget" in record.message for record in caplog.records)


def test_shutdown_is_bounded_and_reports_drainage():
    slow = InputBus()
    started = threading.Event()

    def blocker(event):
        started.set()
        time.sleep(2.0)

    slow.subscribe(EventType.SYSTEM, blocker)
    slow.publish_async(Event(type=EventType.SYSTEM, payload={"n": 1}))
    assert started.wait(timeout=3.0)

    begin = time.monotonic()
    drained = slow.shutdown(timeout=0.2)
    elapsed = time.monotonic() - begin

    assert elapsed < 1.5
    assert drained is False


def test_publishing_after_shutdown_is_refused():
    closed = InputBus()
    closed.shutdown(timeout=1.0)

    receipt = closed.emit(EventType.SYSTEM, {"n": 1})

    assert receipt.queued is False
    assert receipt.dropped_reason == "bus_closed"


# --- f93e3dfd: consumers have an acknowledgement contract -----------------


def test_events_are_held_in_flight_until_acked(bus):
    bus.emit(EventType.SYSTEM, {"n": 1})
    event = bus.next(timeout=1.0)

    assert bus.in_flight_count == 1
    assert bus.ack(event) is True
    assert bus.in_flight_count == 0
    assert bus.ack(event) is False


def test_nack_without_requeue_dead_letters_immediately(bus):
    bus.emit(EventType.SYSTEM, {"n": 1})
    event = bus.next(timeout=1.0)

    receipt = bus.nack(event, requeue=False, reason="poison message")

    assert receipt.dead_lettered is True
    assert bus.in_flight_count == 0
    assert bus.get_dlq_stats()["count"] == 1
    assert "poison message" in bus.dlq_events()[0]["reason"]


def test_unsubscribe_removes_a_callback(bus):
    seen = []

    def watcher(event):
        seen.append(event)

    bus.subscribe(EventType.SYSTEM, watcher)
    assert bus.unsubscribe(EventType.SYSTEM, watcher) is True
    bus.emit(EventType.SYSTEM, {"n": 1})

    assert seen == []
    assert bus.unsubscribe(EventType.SYSTEM, watcher) is False
