from __future__ import annotations

from types import SimpleNamespace

from interface import helpers


class _FailingPresence:
    def __init__(self) -> None:
        self.called = False

    def mark_user_spoke_with_message(self, _message: str) -> None:
        self.called = True
        raise RuntimeError("presence hook failed")


class _RecordingComm:
    def __init__(self) -> None:
        self.called = False

    def record_user_interaction(self) -> None:
        self.called = True


class _RecordingInitiative:
    def __init__(self) -> None:
        self.called = False

    def register_user_interaction(self) -> None:
        self.called = True


def test_notify_user_spoke_continues_after_optional_hook_failure(monkeypatch):
    comm = _RecordingComm()
    initiative = _RecordingInitiative()
    presence = _FailingPresence()
    orchestrator = SimpleNamespace(
        proactive_presence=presence,
        proactive_comm=comm,
        proactive_initiative_engine=initiative,
    )

    monkeypatch.setattr(
        helpers,
        "get_runtime_service",
        lambda _name, default=None: orchestrator,
    )

    helpers._notify_user_spoke("hello")

    assert presence.called is True
    assert comm.called is True
    assert initiative.called is True
    assert orchestrator._last_user_interaction_time > 0
