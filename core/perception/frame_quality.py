"""Measure whether a frame could support the detail being claimed about it.

The abstention fix upstream made "cannot tell" expressible. This makes it
CHECKABLE, because a model's own confidence is not evidence about the
physical conditions it was looking at. A vision model handed a
motion-blurred frame of a dim room will often still answer "two people at a
desk" in a perfectly confident tone, and nothing downstream has any way to
know the pixels could not have supported that.

So the conditions get measured directly, from the frame, before the answer
is trusted:

  * **lighting** — mean luminance plus the fraction of pixels crushed to
    black or blown to white. Detail in a clipped region is not dim, it is
    absent; no model can recover it and a confident claim about it is
    fabrication.
  * **motion / focus** — variance of the Laplacian, the standard sharpness
    proxy. A blurred frame supports "someone is there" and not "someone in
    a red jacket".
  * **distance / resolvable size** — pixels per unit of frame. A face
    twenty feet away occupies too few pixels to carry the features a
    description would assert.
  * **occlusion / uniformity** — how much of the frame is a single flat
    region, which is what a hand over the lens, a closed shutter, or a wall
    looks like.

Deliberately numpy-only. cv2 cannot be imported in Aura's primary macOS
process at all (the PyAV/AVFoundation class collision), and a quality check
that only runs in the sidecar would be absent exactly where readings are
consumed.

Nothing here decides what is in the frame. It decides what the frame is
capable of showing, which is the half that can be measured without a model
and the half that was missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# Below this mean luminance (0-255) a frame is too dark for reliable detail.
# Not a preference: at this level most of the histogram sits in the sensor's
# noise floor, so "detail" recovered from it is amplified noise.
DARK_MEAN = 40.0
BRIGHT_MEAN = 225.0

# Fraction of pixels at the extremes before detail is considered destroyed.
CLIPPED_FRACTION = 0.35

# Variance of the Laplacian. The conventional blur threshold; below it,
# edges have been smeared past the point where features are separable.
SHARP_VARIANCE = 100.0
VERY_BLURRED_VARIANCE = 20.0

# Frames smaller than this cannot carry fine detail regardless of quality.
MIN_DETAIL_PIXELS = 240 * 180

# Above this, the frame is essentially one flat region — a covered lens, a
# closed laptop, a blank wall.
UNIFORM_FRACTION = 0.92


@dataclass(frozen=True)
class FrameQuality:
    """What this frame can and cannot support a claim about."""

    mean_luminance: float
    dark_fraction: float
    bright_fraction: float
    sharpness: float
    uniformity: float
    pixels: int
    limits: tuple[str, ...] = ()

    @property
    def supports_detail(self) -> bool:
        """Can fine detail — counts, colours, text, faces — be asserted?"""
        return not self.limits

    @property
    def supports_presence(self) -> bool:
        """Can gross presence — "something moved", "someone is there" — be
        asserted? A frame can be far too poor for detail and still carry
        this, and collapsing the two loses real information."""
        return "lens_obstructed" not in self.limits and "no_signal" not in self.limits

    @property
    def evidence_score(self) -> float:
        """Comparable physical quality for selecting among adjacent frames.

        This is not semantic confidence and must not be presented as model
        accuracy. It ranks frames from the same camera burst by exposure,
        clipping, edge resolution, and available pixels so an autofocus or
        auto-exposure transient does not become the authoritative observation.
        """
        if "no_signal" in self.limits or "lens_obstructed" in self.limits:
            return 0.0
        exposure = 1.0 - min(1.0, abs(self.mean_luminance - 127.5) / 127.5)
        unclipped = 1.0 - min(1.0, self.dark_fraction + self.bright_fraction)
        sharp = min(1.0, math.log1p(max(0.0, self.sharpness)) / math.log1p(1000.0))
        resolution = min(1.0, self.pixels / max(1, MIN_DETAIL_PIXELS))
        return round(
            0.30 * exposure + 0.25 * unclipped + 0.35 * sharp + 0.10 * resolution,
            6,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_luminance": round(self.mean_luminance, 2),
            "dark_fraction": round(self.dark_fraction, 4),
            "bright_fraction": round(self.bright_fraction, 4),
            "sharpness": round(self.sharpness, 2),
            "uniformity": round(self.uniformity, 4),
            "pixels": self.pixels,
            "limits": list(self.limits),
            "supports_detail": self.supports_detail,
            "supports_presence": self.supports_presence,
            "evidence_score": self.evidence_score,
        }

    def why(self) -> str:
        if not self.limits:
            return ""
        readable = {
            "too_dark": "the frame is too dark to resolve detail",
            "too_bright": "the frame is blown out",
            "clipped": "much of the frame is crushed to black or white",
            "motion_blur": "the frame is blurred",
            "too_small": "the frame is too low-resolution for fine detail",
            "lens_obstructed": "the lens appears covered",
            "no_signal": "the frame carries no image",
        }
        return "; ".join(readable.get(limit, limit) for limit in self.limits)


def _luminance(frame: np.ndarray) -> np.ndarray:
    """Rec. 601 luma, matching what cv2's BGR2GRAY produces."""
    array = np.asarray(frame)
    if array.ndim == 2:
        return array.astype(np.float64)
    if array.ndim == 3 and array.shape[2] >= 3:
        blue = array[..., 0].astype(np.float64)
        green = array[..., 1].astype(np.float64)
        red = array[..., 2].astype(np.float64)
        return 0.114 * blue + 0.587 * green + 0.299 * red
    raise ValueError(f"unsupported frame shape {array.shape}")


def _laplacian_variance(gray: np.ndarray) -> float:
    """Sharpness proxy, computed without cv2.

    The 4-neighbour Laplacian: each pixel minus the average of its
    neighbours. On a sharp frame edges produce large responses and the
    variance is high; blur smears them and the variance collapses.
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    centre = gray[1:-1, 1:-1]
    response = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * centre
    )
    return float(np.var(response))


def assess_frame(frame: Any) -> FrameQuality:
    """Measure what this frame can support. Never raises on a bad frame."""
    try:
        gray = _luminance(frame)
    except (ValueError, TypeError, AttributeError):
        return FrameQuality(0.0, 1.0, 0.0, 0.0, 1.0, 0, ("no_signal",))

    if gray.size == 0:
        return FrameQuality(0.0, 1.0, 0.0, 0.0, 1.0, 0, ("no_signal",))

    pixels = int(gray.size)
    mean = float(np.mean(gray))
    dark_fraction = float(np.mean(gray < 16.0))
    bright_fraction = float(np.mean(gray > 239.0))
    sharpness = _laplacian_variance(gray)

    # The most common value's share of the frame. A covered lens is one
    # value everywhere; a real scene is not.
    counts = np.bincount(
        np.clip(gray, 0, 255).astype(np.uint8).ravel(), minlength=256
    )
    uniformity = float(counts.max() / pixels)

    limits: list[str] = []
    # A uniform frame has two very different causes and the remedies are
    # opposite: "move your hand off the lens" versus "point away from the
    # window". Saturation is not obstruction — a hand or a wall reflects
    # something, so it lands in the mid range, while a blown-out frame
    # pegs at white. Getting this wrong sends the owner to the wrong
    # switch, which is worse than not naming a cause at all.
    obstructed = uniformity >= UNIFORM_FRACTION and mean <= BRIGHT_MEAN

    if obstructed:
        # Named on its own: it explains the darkness rather than being a
        # separate finding, and reporting both would double-count one cause.
        limits.append("lens_obstructed")
    else:
        if mean < DARK_MEAN:
            limits.append("too_dark")
        elif mean > BRIGHT_MEAN:
            limits.append("too_bright")
        if (dark_fraction + bright_fraction) >= CLIPPED_FRACTION:
            limits.append("clipped")
        if sharpness < SHARP_VARIANCE:
            limits.append("motion_blur")
        if pixels < MIN_DETAIL_PIXELS:
            limits.append("too_small")

    return FrameQuality(
        mean_luminance=mean,
        dark_fraction=dark_fraction,
        bright_fraction=bright_fraction,
        sharpness=sharpness,
        uniformity=uniformity,
        pixels=pixels,
        limits=tuple(limits),
    )


def temper_reading(reading: dict[str, Any], quality: FrameQuality) -> dict[str, Any]:
    """Downgrade detail claims the frame could not have supported.

    This is the causal half. Without it the quality measurement is a number
    in a log: the model still says "2 people" about a frame that is mostly
    noise, and the count is still consumed as fact.

    Presence survives — a blurred frame can carry "someone is there" — but a
    COUNT, a colour, or a text reading cannot be recovered from pixels that
    do not contain it. Those become unknown, with the measured reason
    attached so the abstention can be explained rather than merely asserted.
    """
    reading = dict(reading)
    reading["frame_quality"] = quality.to_dict()

    if quality.supports_detail:
        return reading

    tempered: list[str] = []
    for field in ("objects_detected", "text_detected", "faces_detected"):
        if reading.get(field) is not None:
            tempered.append(field)
            reading[field] = None

    if tempered:
        reading["tempered_fields"] = tempered
        reading["temper_reason"] = quality.why()
        # The description is kept. It is the model's honest impression and
        # is still useful; what is removed is the structured detail a
        # consumer would otherwise treat as measured fact.
        reading["detail_supported"] = False
    else:
        reading["detail_supported"] = False

    return reading


__all__ = [
    "BRIGHT_MEAN",
    "CLIPPED_FRACTION",
    "DARK_MEAN",
    "MIN_DETAIL_PIXELS",
    "SHARP_VARIANCE",
    "UNIFORM_FRACTION",
    "FrameQuality",
    "assess_frame",
    "temper_reading",
]
