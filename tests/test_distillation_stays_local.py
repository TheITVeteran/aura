"""Learning from the owner's conversations does not require a third party.

The distillation pipe builds its teacher prompt around the owner's ORIGINAL
PROMPT, verbatim. It used to pass ``allow_cloud_fallback=True``
unconditionally, which made a cloud provider the FIRST reader of those turns
and the local lane the fallback.

Redaction does not answer this. The egress privacy boundary strips
credentials and personal identifiers from whatever reaches a provider, but a
redacted transcript of someone's work is still their work — sitting in
somebody else's log so that a LoRA could be trained on it. Where it goes is
a separate decision from what is scrubbed out of it, and this pins the
default.
"""
from __future__ import annotations

import asyncio

import pytest

from core.adaptation import distillation_pipe
from core.adaptation.distillation_pipe import DistillationPipe, _cloud_teacher_allowed


class _RecordingBrain:
    """Captures what the teacher call was actually authorised to do."""

    def __init__(self):
        self.contexts: list[dict] = []

    async def think(self, *, objective, context, **kwargs):
        self.contexts.append(dict(context or {}))
        raise ConnectionError("teacher unreachable; exercise the fallback")


def test_cloud_teacher_is_off_by_default():
    assert _cloud_teacher_allowed() is False


def test_a_broken_config_does_not_open_the_cloud_leg(monkeypatch):
    def _explode():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr("core.config.get_config", _explode)

    # Fail-closed in the direction that keeps transcripts on the machine.
    assert _cloud_teacher_allowed() is False


def test_the_teacher_call_is_not_authorised_for_cloud_by_default(monkeypatch):
    monkeypatch.setattr(distillation_pipe, "_cloud_teacher_allowed", lambda: False)
    brain = _RecordingBrain()

    asyncio.run(
        DistillationPipe()._get_teacher_response(brain, "teach me about this prompt")
    )

    assert brain.contexts, "the teacher path was never exercised"
    assert brain.contexts[0]["allow_cloud_fallback"] is False


def test_the_owner_can_opt_in(monkeypatch):
    monkeypatch.setattr(distillation_pipe, "_cloud_teacher_allowed", lambda: True)
    brain = _RecordingBrain()

    asyncio.run(DistillationPipe()._get_teacher_response(brain, "teach me"))

    assert brain.contexts[0]["allow_cloud_fallback"] is True


@pytest.mark.parametrize("opted_in", [False, True])
def test_the_local_secondary_lane_never_falls_back_to_cloud(monkeypatch, opted_in):
    """Even opted in, the FALLBACK leg stays local — it is the safety net."""
    monkeypatch.setattr(distillation_pipe, "_cloud_teacher_allowed", lambda: opted_in)
    captured: list[dict] = []

    class _Router:
        async def think(self, **kwargs):
            captured.append(kwargs)
            return None

    from core.container import ServiceContainer

    monkeypatch.setattr(
        ServiceContainer, "get", staticmethod(lambda name, default=None: _Router() if name == "llm_router" else default)
    )

    asyncio.run(DistillationPipe()._get_teacher_response(_RecordingBrain(), "teach me"))

    assert captured, "the local secondary lane was never reached"
    assert captured[0]["allow_cloud_fallback"] is False
