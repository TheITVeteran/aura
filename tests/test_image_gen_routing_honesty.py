"""Regression contracts from the July 8 live misroute.

The soak's memory-retention plant — "Keep this in mind for later: the paint
color I chose..." — was dispatched to the diffusion skill because the trigger
"paint (?:me )?(?:an? )?" had an all-optional tail (bare-verb match), and the
skill then crashed CRITICAL because the generic dispatcher passes ``query``
while the input model demands ``prompt``. Both halves are contracts now.
"""
from __future__ import annotations

import pytest

from core.capability_engine import CapabilityEngine
from core.skills.image_gen import ImageGenInput

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def engine():
    return CapabilityEngine()


class TestImageGenTriggers:
    @pytest.mark.parametrize("message", [
        "Keep this in mind for later: the paint color I chose for the bedroom is sage.",
        "we should paint the fence this weekend",
        "I need to draw a conclusion from these numbers",
        "let me withdraw the earlier point",
        "can you imagine how tired I am",
        "help me visualize my calendar for next week",
        "the painting in the hallway is crooked",
    ])
    def test_conversational_mentions_do_not_dispatch(self, engine, message):
        assert "image_gen" not in engine.detect_intent(message)

    @pytest.mark.parametrize("message", [
        "generate an image of a lighthouse at dusk",
        "create a picture of my dream studio",
        "draw me a dragon curled around a teapot",
        "paint me a sunset over the marina",
        "can you make a logo for my podcast",
        "edit this image to remove the background",
        "run img2img on this with strength 0.6",
        "text to image: a fox in the snow",
    ])
    def test_real_requests_still_dispatch(self, engine, message):
        assert "image_gen" in engine.detect_intent(message)


class TestImageGenInputShape:
    def test_generic_dispatch_query_maps_to_prompt(self):
        params = ImageGenInput(**{"query": "a lighthouse at dusk", "strength": 0.75})
        assert params.prompt == "a lighthouse at dusk"

    def test_text_key_also_accepted(self):
        assert ImageGenInput(**{"text": "a red bicycle"}).prompt == "a red bicycle"

    def test_explicit_prompt_wins_over_query(self):
        params = ImageGenInput(**{"prompt": "the real prompt", "query": "ignored"})
        assert params.prompt == "the real prompt"

    def test_truly_empty_input_still_fails_validation(self):
        with pytest.raises(Exception):
            ImageGenInput(**{"strength": 0.75})


@pytest.mark.asyncio
async def test_save_and_respond_writes_inside_governed_scope(tmp_path, monkeypatch):
    """Live RENDER THIS failure (Jul 2026): every authorization gate passed,
    the image generated, and the SAVE failed with 'called outside governed
    context'. The artifact write must run inside a governed scope."""
    from types import SimpleNamespace

    from core.skills.image_gen import ImageGenSkill

    skill = ImageGenSkill.__new__(ImageGenSkill)
    skill._output_dir = tmp_path

    class FakeImage:
        def save(self, output, format="PNG"):
            output.write(b"\x89PNG-fake-payload")

    monkeypatch.setattr(
        ImageGenSkill, "_is_degenerate", staticmethod(lambda image: False)
    )

    captured: dict = {}

    class FakeGateway:
        async def write_bytes_async(self, path, payload, *, source=""):
            from core.governance_context import get_active_governance

            captured["governance"] = get_active_governance()
            captured["path"] = str(path)
            captured["source"] = source

    import core.skills.image_gen as image_gen_mod

    monkeypatch.setattr(
        image_gen_mod, "get_file_write_gateway", lambda: FakeGateway()
    )

    result = await skill._save_and_respond(FakeImage(), "a test prompt")
    assert result["ok"], result
    assert captured["source"] == "skills.image_gen.output"
    assert captured["governance"] is not None, (
        "artifact write must carry an active governed scope"
    )
