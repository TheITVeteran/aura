from types import SimpleNamespace

import pytest

from core.skills import visual_context_skill as visual_context_module
from core.brain.types import ThinkingMode
from core.senses.continuous_vision import ContinuousSensoryBuffer
from core.skills.visual_context_skill import VisualContextSkill


class VisionHarness:
    def __init__(self, result: str):
        self.result = result
        self.calls = []

    async def query_visual_context(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return self.result


def test_sensory_buffer_exposes_frame_parts(tmp_path):
    buffer = ContinuousSensoryBuffer(tmp_path)
    try:
        buffer.frame_buffer.append(("image/png", b"frame-1"))
        buffer.frame_buffer.append(("image/jpeg", b"frame-2"))

        assert buffer.get_visual_context_parts() == [
            {"mime_type": "image/png", "data": b"frame-1"},
            {"mime_type": "image/jpeg", "data": b"frame-2"},
        ]
    finally:
        buffer._vision_executor.shutdown(wait=False, cancel_futures=True)


@pytest.mark.asyncio
async def test_visual_context_skill_routes_prompt_to_buffer(monkeypatch):
    vision = VisionHarness("The active window contains a terminal.")
    brain = SimpleNamespace()

    def get_service(name, default=None):
        if name == "continuous_vision":
            return vision
        if name == "cognitive_engine":
            return brain
        return default

    monkeypatch.setattr(visual_context_module.ServiceContainer, "get", staticmethod(get_service))

    result = await VisualContextSkill().execute({"prompt": "What is on screen?"}, {})

    assert result["ok"] is True
    assert result["analysis"] == "The active window contains a terminal."
    assert len(vision.calls) == 1
    assert vision.calls[0].kwargs["prompt"] == "What is on screen?"
    assert vision.calls[0].kwargs["brain"] is brain
    assert vision.calls[0].kwargs["mode"] is ThinkingMode.FAST
