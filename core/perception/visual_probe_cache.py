"""A failed visual probe belongs to its measured pixels and interpretation inputs."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.verify import invariant


@dataclass(frozen=True)
class VisualProbeWitness:
    context: tuple[Any, ...]
    inputs: str
    shape: tuple[int, ...]
    digest: bytes

    @classmethod
    def measure(cls, picture: Any, *, context: tuple[Any, ...], inputs: str) -> VisualProbeWitness | None:
        """Use all captured pixels; absent or unsupported measurements cannot suppress another probe."""
        if not isinstance(picture, np.ndarray) or picture.dtype != np.uint8 or not picture.size:
            return None
        if picture.ndim not in (2, 3) or picture.nbytes > 16 * 1024 * 1024:
            return None
        if picture.ndim == 3 and picture.shape[2] not in (3, 4):
            return None
        return cls(context, inputs, picture.shape, hashlib.sha256(picture.tobytes()).digest())


@dataclass(frozen=True)
class VisualProbeMiss:
    witness: VisualProbeWitness
    at: float

    def applies(self, witness: VisualProbeWitness | None, *, now: float, seconds: float = 12.0) -> bool:
        """A rejection expires and cannot travel to another picture, surface or interpretation."""
        return (witness is not None and witness == self.witness and math.isfinite(now) and math.isfinite(self.at)
                and math.isfinite(seconds) and 0 <= now - self.at < seconds)


@invariant("perception.visual_probe_misses_require_current_pixels", scope="perception",
           owner="core/perception/visual_probe_cache.py", observational=False)
def _visual_probe_miss_invariant() -> tuple:
    picture = np.zeros((4, 4, 3), dtype=np.uint8)
    witness = VisualProbeWitness.measure(picture, context=("surface",), inputs="instructions")
    assert witness is not None
    miss = VisualProbeMiss(witness, 10.0)
    assert miss.applies(witness, now=11.0)
    picture[1, 1, 0] = 1
    assert not miss.applies(VisualProbeWitness.measure(picture, context=("surface",), inputs="instructions"), now=11.0)
    assert not miss.applies(None, now=11.0)
    assert not miss.applies(witness, now=22.0)
    return ()
