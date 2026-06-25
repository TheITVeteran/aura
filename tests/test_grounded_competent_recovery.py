"""Tests for the grounded competent-recovery helper (recover, don't fail closed)."""
from __future__ import annotations

import pytest

from interface.routes.chat import _grounded_competent_recovery


class _FakeGate:
    def __init__(self, reply):
        self._reply = reply
        self.calls = []

    async def generate(self, message, context=None, timeout=None):
        self.calls.append((message, dict(context or {})))
        return self._reply


@pytest.mark.asyncio
async def test_recovers_with_competent_grounded_reply():
    gate = _FakeGate("Not much — just here and ready. What's actually on your mind?")
    out = await _grounded_competent_recovery("Huh?", gate=gate)
    assert out and "what's actually on your mind" in out.lower()
    # the regeneration carried the anti-confabulation grounding brief
    brief = gate.calls[0][1].get("brief", "")
    assert "do not invent" in brief.lower() and "grounded" in brief.lower()
    assert gate.calls[0][1].get("grounded_recovery") is True


@pytest.mark.asyncio
async def test_empty_reply_returns_none():
    gate = _FakeGate("")
    assert await _grounded_competent_recovery("Huh?", gate=gate) is None


@pytest.mark.asyncio
async def test_too_short_reply_returns_none():
    gate = _FakeGate("ok")
    assert await _grounded_competent_recovery("Huh?", gate=gate) is None


@pytest.mark.asyncio
async def test_no_usable_gate_returns_none():
    class _NoGen:
        pass

    assert await _grounded_competent_recovery("Huh?", gate=_NoGen()) is None


@pytest.mark.asyncio
async def test_degraded_recovery_is_rejected(monkeypatch):
    # if the regeneration is itself assessed degraded, don't serve it
    import interface.routes.chat as chat_mod
    from core.conversation import response_reliability as rr

    gate = _FakeGate("As an AI language model, I cannot help with that request.")

    class _HardFail:
        reasons = ("generic_assistant_language",)   # a genuinely-unservable defect

    monkeypatch.setattr(rr, "assess_user_facing_reply", lambda *a, **k: _HardFail())
    assert await _grounded_competent_recovery("Huh?", gate=gate) is None


@pytest.mark.asyncio
async def test_soft_assessment_flag_still_serves_competent_reply(monkeypatch):
    # the false-positive that caused the original fail-closed (foreign_name_intrusion on a
    # confusion-repair reply) must NOT block a competent recovery
    from core.conversation import response_reliability as rr

    gate = _FakeGate("No worries — I think I misread you. What did you mean?")

    class _SoftFlag:
        reasons = ("foreign_name_intrusion",)

    monkeypatch.setattr(rr, "assess_user_facing_reply", lambda *a, **k: _SoftFlag())
    out = await _grounded_competent_recovery("Huh?", gate=gate)
    assert out and "what did you mean" in out.lower()
