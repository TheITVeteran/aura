import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.skills.image_gen import ImageGenSkill


class SavedImageProbe:
    def __init__(self):
        self.saved_paths = []

    def save(self, target, *, format=None):
        self.saved_paths.append(target)
        if hasattr(target, "write"):
            target.write(b"png-probe")
        else:
            target.write_bytes(b"png-probe")


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
    assert image.saved_paths and await asyncio.to_thread(Path(result["path"]).exists)
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
