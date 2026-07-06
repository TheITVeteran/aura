"""Audit finding B pinned: when the Unified Will engine itself throws, the
output gate must FAIL CLOSED — autonomous output is kept off the primary
channel, and a user-awaited reply is still delivered but honestly marked
governance_degraded (never silently claimed as governed)."""
from __future__ import annotations

import asyncio

import pytest

from core.utils.output_gate import AutonomousOutputGate


class _RecordingGate(AutonomousOutputGate):
    def __init__(self):
        super().__init__(orchestrator=None)
        self.primary_calls = []
        self.secondary_calls = []

    async def _send_to_primary(self, content, origin, metadata, timeout=5.0):
        self.primary_calls.append({"content": content, "origin": origin, "metadata": metadata})

    async def _send_to_secondary(self, content, origin, metadata, timeout=5.0):
        self.secondary_calls.append({"content": content, "origin": origin, "metadata": metadata})


def _break_will(monkeypatch):
    import core.will as will_mod

    def _raise(*_a, **_k):
        raise RuntimeError("will engine unavailable (test)")

    monkeypatch.setattr(will_mod, "get_will", _raise, raising=False)


def test_autonomous_output_fails_closed_when_will_throws(monkeypatch):
    _break_will(monkeypatch)
    gate = _RecordingGate()
    # autonomous origin, target primary: must NOT reach primary
    asyncio.run(gate.emit("A spontaneous musing.", origin="curiosity", target="primary",
                           metadata={"spontaneous": True, "force_user": True}))
    assert gate.primary_calls == [], "autonomous output leaked to primary during Will outage"


def test_autonomous_both_target_reroutes_to_secondary(monkeypatch):
    _break_will(monkeypatch)
    gate = _RecordingGate()
    asyncio.run(gate.emit("Autonomous note.", origin="reflection", target="both"))
    assert gate.primary_calls == []
    assert len(gate.secondary_calls) == 1


def test_user_reply_delivered_but_marked_degraded_when_will_throws(monkeypatch):
    _break_will(monkeypatch)
    gate = _RecordingGate()
    # a direct user-awaited reply is never silently dropped
    asyncio.run(gate.emit("I hear you.", origin="user", target="primary"))
    assert len(gate.primary_calls) == 1, "user reply was dropped during Will outage"
    meta = gate.primary_calls[0]["metadata"]
    assert meta.get("governance_degraded") is True
    assert meta.get("will_receipt_id") is None


def test_healthy_will_still_attaches_a_receipt(monkeypatch):
    import core.will as will_mod

    class _Decision:
        receipt_id = "will-ok-123"

        def is_approved(self):
            return True

    class _Will:
        def decide(self, **_k):
            return _Decision()

    monkeypatch.setattr(will_mod, "get_will", lambda: _Will(), raising=False)
    gate = _RecordingGate()
    asyncio.run(gate.emit("Governed reply.", origin="user", target="primary"))
    assert len(gate.primary_calls) == 1
    assert gate.primary_calls[0]["metadata"].get("will_receipt_id") == "will-ok-123"
    assert gate.primary_calls[0]["metadata"].get("governance_degraded") is not True
