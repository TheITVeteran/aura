"""Sovereign image-generation facade over the canonical diffusers backend."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.skills.image_gen import ImageGenInput, ImageGenSkill


class ImageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=20_000)
    negative_prompt: str = Field(
        default="blurry, low quality, deformed, ugly, text, watermark",
        max_length=10_000,
    )
    steps: int = Field(default=30, ge=10, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: int | None = None


class SovereignImaginationSkill(ImageGenSkill):
    name = "sovereign_imagination"
    description = (
        "Generate a high-quality local image through Aura's canonical governed image backend."
    )
    input_model = ImageInput
    effect_scope = "read_write_artifacts"

    async def execute(
        self,
        params: ImageInput | dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        validated = params if isinstance(params, ImageInput) else ImageInput.model_validate(params)
        return await super().execute(
            ImageGenInput(
                prompt=validated.prompt,
                negative_prompt=validated.negative_prompt,
                steps=validated.steps,
                guidance_scale=validated.guidance_scale,
                seed=validated.seed,
            ),
            context or {},
        )
