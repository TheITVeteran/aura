"""Distillation remains inside Aura's managed local model boundary."""

from __future__ import annotations

import asyncio

from core.adaptation.distillation_pipe import DistillationPipe


class _RecordingBrain:
    def __init__(self):
        self.contexts: list[dict] = []

    async def think(self, *, objective, context, **kwargs):
        self.contexts.append(dict(context or {}))
        raise ConnectionError("deep teacher unavailable")


def test_deep_teacher_call_is_unconditionally_local():
    brain = _RecordingBrain()

    asyncio.run(DistillationPipe()._get_teacher_response(brain, "teach me"))

    assert brain.contexts
    assert brain.contexts[0]["allow_cloud_fallback"] is False
    assert brain.contexts[0]["teacher_target"] == "local_deep"


def test_resident_teacher_fallback_is_unconditionally_local(monkeypatch):
    captured: list[dict] = []

    class _Router:
        async def think(self, **kwargs):
            captured.append(kwargs)
            return None

    from core.container import ServiceContainer

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: _Router()
            if name == "llm_router"
            else default
        ),
    )

    asyncio.run(DistillationPipe()._get_teacher_response(_RecordingBrain(), "teach me"))

    assert captured
    assert captured[0]["prefer_tier"] == "primary"
    assert captured[0]["allow_cloud_fallback"] is False


def test_distillation_has_no_remote_teacher_switch():
    import inspect
    from core.adaptation import distillation_pipe

    source = inspect.getsource(distillation_pipe)
    assert "_cloud_teacher_allowed" not in source
    assert "allow_cloud_teacher_distillation" not in source
    assert "teacher_model" not in source
