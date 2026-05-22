from __future__ import annotations


def test_degraded_event_async_forward_failure_records_structured_audit_gap(monkeypatch):
    from core.health import degraded_events as de_mod
    from core.runtime.errors import FallbackClassification

    calls = []

    def fake_record_degradation(subsystem, error, **kwargs):
        calls.append((subsystem, error, kwargs))

    class ImmediateThread:
        def __init__(self, target, name=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    class FailingForward:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self):
            self.calls += 1
            raise RuntimeError("forward bus unavailable")

    monkeypatch.setattr(de_mod, "record_degradation", fake_record_degradation)
    monkeypatch.setattr(de_mod.threading, "Thread", ImmediateThread)

    forward = FailingForward()
    de_mod._schedule_awaitable(forward.run(), label="degraded_event_forward")

    assert calls
    assert forward.calls == 1
    subsystem, error, kwargs = calls[-1]
    assert subsystem == "degraded_events"
    assert isinstance(error, RuntimeError)
    assert kwargs["classification"] == FallbackClassification.AUDIT_GAP
    assert "retained degraded event locally" in kwargs["action"]
    assert kwargs["extra"]["label"] == "degraded_event_forward"
