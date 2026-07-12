
import hashlib
import logging
import math
import struct
import time
import zlib
from pathlib import Path
from typing import Any

from core.config import config
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from infrastructure import BaseSkill

logger = logging.getLogger("Skills.LocalMedia")


class LocalMediaGenerationSkill(BaseSkill):
    name = "local_media_generation"
    description = "Generate images locally using Stable Diffusion (Offline)."
    inputs = {
        "prompt": "Description of the image to generate.",
        "negative_prompt": "Optional. What to avoid in the image.",
        "style": "Optional style guidance.",
    }
    
    def __init__(self):
        super().__init__()
        self._canonical_skill = None
        
        self.output_dir = Path(config.paths.data_dir) / "generated_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write_png(path: Path, width: int, height: int, rows: list[bytes]) -> None:
        """Write a simple RGB PNG without optional imaging dependencies."""

        def chunk(kind: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
            )

        raw = b"".join(b"\x00" + row for row in rows)
        get_file_write_gateway().write_bytes(
            path,
            b"".join(
                [
                    b"\x89PNG\r\n\x1a\n",
                    chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
                    chunk(b"IDAT", zlib.compress(raw, level=6)),
                    chunk(b"IEND", b""),
                ]
            ),
            source="local_media_generation.procedural_png",
        )

    def _generate_procedural_image(self, prompt: str) -> dict[str, Any]:
        """Generate a deterministic local image when diffusion weights are unavailable."""
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        width, height = 768, 512
        rows: list[bytes] = []
        for y in range(height):
            row = bytearray()
            fy = y / max(height - 1, 1)
            for x in range(width):
                fx = x / max(width - 1, 1)
                wave = math.sin((fx * digest[0] + fy * digest[1]) * math.pi * 4.0)
                swirl = math.cos(((fx - 0.5) ** 2 + (fy - 0.5) ** 2) * digest[2] * 6.0)
                r = int((fx * digest[3] + (wave + 1.0) * 42 + digest[4]) % 256)
                g = int((fy * digest[5] + (swirl + 1.0) * 55 + digest[6]) % 256)
                b = int(((1.0 - fx) * digest[7] + (1.0 - fy) * digest[8] + digest[9]) % 256)
                row.extend((r, g, b))
            rows.append(bytes(row))

        timestamp = int(time.time())
        filename = f"gen_procedural_{timestamp}.png"
        filepath = self.output_dir / filename
        self._write_png(filepath, width, height, rows)
        relative_url = f"/data/generated_images/{filename}"
        return {
            "ok": True,
            "url": relative_url,
            "path": str(filepath),
            "message": (
                "I generated a local procedural image because diffusion weights "
                "are not available in this runtime."
            ),
            "type": "image",
            "degraded": True,
            "generation_mode": "procedural_fallback",
            "model_id": None,
        }
        
    async def execute(self, goal: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Generate image locally."""
        prompt = goal.get("objective") or goal.get("params", {}).get("prompt")
        
        if not prompt:
            return {"ok": False, "error": "No prompt provided."}
            
        try:
            from core.skills.image_gen import ImageGenSkill

            if self._canonical_skill is None:
                self._canonical_skill = ImageGenSkill()
            result = await self._canonical_skill.execute(
                {
                    "prompt": str(prompt),
                    "negative_prompt": goal.get("negative_prompt"),
                    "style": goal.get("style"),
                    "steps": 40,
                    "guidance_scale": 8.0,
                    "width": 768,
                    "height": 512,
                },
                context,
            )
            if result.get("ok"):
                return {
                    **result,
                    "degraded": False,
                    "generation_mode": result.get("mode", "diffusion"),
                }
            logger.warning(
                "Canonical image generation unavailable; using procedural fallback: %s",
                result.get("error"),
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "local_media_generation",
                exc,
                severity="warning",
                action="used procedural fallback after canonical image generation failed",
            )
        return self._generate_procedural_image(str(prompt))

    async def on_stop_async(self) -> None:
        if self._canonical_skill is not None:
            await self._canonical_skill.on_stop_async()
            self._canonical_skill = None
