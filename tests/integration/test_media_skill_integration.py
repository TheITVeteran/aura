from types import SimpleNamespace

import pytest

from core.skills.image_gen import ImageGenSkill


class SavedImageProbe:
    """Mirrors PIL's save contract: the skill encodes into a BytesIO with
    an explicit format, then routes the BYTES through the governed file
    write gateway (image writes are consequential writes now)."""

    def __init__(self):
        self.save_calls = []

    def save(self, target, format=None, **kwargs):
        self.save_calls.append({"format": format})
        target.write(b"png-probe")


class RecordingTextToImagePipeline:
    def __init__(self, image: SavedImageProbe):
        self.image = image
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        return SimpleNamespace(images=[self.image])


class GeneratorProbe:
    def __init__(self, device=None):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


@pytest.mark.asyncio
async def test_image_generation_skill_saves_recorded_pipeline_output(monkeypatch, tmp_path):
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        SimpleNamespace(Generator=GeneratorProbe),
    )
    skill = ImageGenSkill()
    # A non-turbo checkpoint honours the caller's sampler settings; turbo
    # models intentionally clamp to 1-4 steps (covered separately below).
    skill._model_id = "runwayml/stable-diffusion-v1-5"
    image = SavedImageProbe()
    pipeline = RecordingTextToImagePipeline(image)
    skill._pipeline = pipeline
    skill._output_dir = tmp_path
    skill._load_pipeline = lambda img2img=False: True

    result = await skill.execute(
        {
            "prompt": "Generate a futuristic city",
            "quality": "standard",
            "width": 512,
            "height": 512,
            "steps": 12,
            "seed": 123,
        },
        {},
    )

    assert result["ok"] is True
    assert result["type"] == "image"
    assert result["mode"] == "txt2img"
    assert result["url"].startswith("/data/generated_images/gen_txt2img_")
    assert image.save_calls and image.save_calls[0]["format"] == "PNG"
    written = sorted(tmp_path.glob("gen_txt2img_*.png"))
    assert written and written[0].read_bytes() == b"png-probe"
    assert pipeline.calls[0]["width"] == 512
    assert pipeline.calls[0]["height"] == 512
    assert pipeline.calls[0]["num_inference_steps"] == 12
    assert "Generate a futuristic city" in pipeline.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_image_generation_skill_rejects_missing_prompt():
    skill = ImageGenSkill()

    result = await skill.execute({"prompt": " "}, {})

    assert result["ok"] is False
    assert "prompt" in result["error"].lower()


@pytest.mark.asyncio
async def test_turbo_checkpoint_clamps_sampler_settings(monkeypatch, tmp_path):
    """Turbo/distilled checkpoints are adversarially trained for 1-4 steps
    with guidance off; a 12-step request is clamped by design, not honoured."""
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        SimpleNamespace(Generator=GeneratorProbe),
    )
    skill = ImageGenSkill()
    skill._model_id = "stabilityai/sdxl-turbo"
    image = SavedImageProbe()
    pipeline = RecordingTextToImagePipeline(image)
    skill._pipeline = pipeline
    skill._output_dir = tmp_path
    skill._load_pipeline = lambda img2img=False: True

    result = await skill.execute(
        {"prompt": "Generate a futuristic city", "steps": 12, "seed": 123},
        {},
    )

    assert result["ok"] is True
    assert pipeline.calls[0]["num_inference_steps"] <= 4
