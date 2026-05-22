from __future__ import annotations

import time
from types import SimpleNamespace

import pytest


def _state(*, phi: float = 0.2):
    return SimpleNamespace(
        phi=phi,
        affect=SimpleNamespace(
            emotions={"calm": 0.8, "curiosity": 0.5, "worry": 0.1},
            valence=0.25,
            arousal=0.2,
            curiosity=0.5,
            dominant_emotion="calm",
        ),
        cognition=SimpleNamespace(
            current_mode=SimpleNamespace(value="attentive"),
            phenomenal_state=None,
        ),
    )


def test_feedback_observer_circuit_breaks_repeated_callback_failures(monkeypatch):
    from core.kernel import feedback_observer as module
    from core.runtime.errors import FallbackClassification

    records = []

    def fake_record_degradation(subsystem, error, **kwargs):
        records.append((subsystem, error, kwargs))

    monkeypatch.setattr(module, "record_degradation", fake_record_degradation)

    observer = module.FeedbackObserver()
    failing_calls = 0
    good_calls = 0

    def failing_callback(_entry):
        nonlocal failing_calls
        failing_calls += 1
        raise RuntimeError("callback storage unavailable")

    def good_callback(_entry):
        nonlocal good_calls
        good_calls += 1

    observer.on_tick(failing_callback)
    observer.on_tick(good_callback)

    for index in range(4):
        entry = observer.begin_tick(_state(phi=0.1 + index), f"tick {index}")
        observer.end_tick(entry, "ok", _state(phi=0.2 + index), time.time())

    assert failing_calls == 3
    assert good_calls == 4
    assert len(records) == 3
    assert records[-1][2]["classification"] == FallbackClassification.AUDIT_GAP
    assert "disabled failing feedback callback" in records[-1][2]["action"]
    assert records[-1][2]["extra"]["consecutive_failures"] == 3

    status = observer.get_callback_status()
    assert status[0]["disabled"] is True
    assert status[0]["consecutive_failures"] == 3
    assert status[1]["disabled"] is False


def test_feedback_observer_resets_callback_failure_count_after_success(monkeypatch):
    from core.kernel import feedback_observer as module

    monkeypatch.setattr(module, "record_degradation", lambda *_args, **_kwargs: None)
    observer = module.FeedbackObserver()
    attempts = 0

    def flaky_callback(_entry):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient dashboard outage")

    observer.on_tick(flaky_callback)

    entry = observer.begin_tick(_state(), "first")
    observer.end_tick(entry, "first", _state(phi=0.3), time.time())
    assert observer.get_callback_status()[0]["consecutive_failures"] == 1

    entry = observer.begin_tick(_state(), "second")
    observer.end_tick(entry, "second", _state(phi=0.4), time.time())
    assert observer.get_callback_status()[0]["consecutive_failures"] == 0
    assert observer.get_callback_status()[0]["disabled"] is False


def test_feedback_observer_rejects_non_callable_callbacks():
    from core.kernel.feedback_observer import FeedbackObserver

    observer = FeedbackObserver()

    with pytest.raises(TypeError, match="callback must be callable"):
        observer.on_tick("not a callback")
