"""A model's confidence is not evidence about what it was looking at.

The abstention work made "cannot tell" expressible. It still left the
decision to the model, and a vision model handed a motion-blurred frame of a
dim room will answer "two people at a desk" in exactly the tone it uses for
a sharp one. Nothing downstream could tell that the pixels never contained
what was being asserted.

`frame_quality` measures the conditions directly — lighting, focus,
resolvable size, occlusion — and `temper_reading` removes the detail claims
those conditions could not support. That is the causal half: without it the
measurement is a number in a log while the count is still consumed as fact.

These use real numpy frames, because the entire point is what the pixels
do, and a mocked frame would test nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.frame_quality import (
    MIN_DETAIL_PIXELS,
    FrameQuality,
    assess_frame,
    temper_reading,
)


def _noise(shape=(480, 640, 3), scale: float = 1.0, seed: int = 7) -> np.ndarray:
    """High-frequency content — a stand-in for a sharp, well-lit scene."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, shape, dtype=np.uint8)
    if scale != 1.0:
        frame = np.clip(frame.astype(np.float64) * scale, 0, 255).astype(np.uint8)
    return frame


def _gradient(shape=(480, 640)) -> np.ndarray:
    """Smooth content — what blur leaves behind."""
    row = np.linspace(60, 200, shape[1])
    plane = np.tile(row, (shape[0], 1))
    return np.stack([plane] * 3, axis=-1).astype(np.uint8)


# ────────────────────────────────────────────────── lighting


def test_a_well_lit_sharp_frame_supports_detail():
    quality = assess_frame(_noise())

    assert quality.limits == ()
    assert quality.supports_detail is True
    assert quality.supports_presence is True


def test_a_dark_frame_does_not_support_detail():
    quality = assess_frame(_noise(scale=0.06))

    assert "too_dark" in quality.limits
    assert quality.supports_detail is False


def test_a_blown_out_frame_does_not_support_detail():
    frame = np.full((480, 640, 3), 250, dtype=np.uint8)
    # Break the uniformity so this is read as over-exposure, not a covered
    # lens — the two have different remedies.
    frame[::7, ::7] = 235

    quality = assess_frame(frame)

    assert "too_bright" in quality.limits or "clipped" in quality.limits
    assert quality.supports_detail is False


def test_heavy_clipping_is_caught_even_at_a_normal_mean():
    """Half black and half white averages to a perfectly ordinary mid-grey.

    Detail in a clipped region is not dim, it is absent, and a mean-only
    check would call this frame well exposed.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, 320:] = 255

    quality = assess_frame(frame)

    assert "clipped" in quality.limits
    assert 100 < quality.mean_luminance < 155, "the mean alone looks fine"


# ────────────────────────────────────────────────── motion and focus


def test_a_smooth_frame_reads_as_blurred():
    quality = assess_frame(_gradient())

    assert "motion_blur" in quality.limits
    assert quality.supports_detail is False


def test_sharpness_separates_a_blurred_frame_from_a_sharp_one():
    sharp = assess_frame(_noise())
    blurred = assess_frame(_gradient())

    assert sharp.sharpness > blurred.sharpness * 10


def test_evidence_score_prefers_a_resolved_frame_over_an_opening_transient():
    sharp = assess_frame(_noise())
    opening_transient = assess_frame(np.full((480, 640, 3), 8, dtype=np.uint8))

    assert sharp.evidence_score > opening_transient.evidence_score
    assert sharp.to_dict()["evidence_score"] == sharp.evidence_score


# ────────────────────────────────────────────────── distance / size


def test_a_tiny_frame_cannot_carry_fine_detail():
    small = _noise((40, 60, 3))

    quality = assess_frame(small)

    assert "too_small" in quality.limits
    assert quality.pixels < MIN_DETAIL_PIXELS


# ────────────────────────────────────────────────── occlusion


def test_a_covered_lens_is_named_as_such():
    """Not "too dark" — the remedy is "move your hand", not "turn on a
    light", and a system that says the wrong one sends its owner to the
    wrong switch."""
    quality = assess_frame(np.full((480, 640, 3), 12, dtype=np.uint8))

    assert quality.limits == ("lens_obstructed",)
    assert quality.supports_presence is False


def test_obstruction_is_not_double_counted_as_darkness():
    quality = assess_frame(np.full((480, 640, 3), 5, dtype=np.uint8))

    assert "too_dark" not in quality.limits


# ────────────────────────────────────────────────── degenerate input


def test_an_empty_frame_is_no_signal():
    quality = assess_frame(np.zeros((0, 0, 3), dtype=np.uint8))

    assert quality.limits == ("no_signal",)
    assert quality.supports_presence is False


def test_a_non_frame_never_raises():
    """This runs on the capture path; an exception here would turn a bad
    frame into a lost turn."""
    quality = assess_frame("not a frame")

    assert quality.limits == ("no_signal",)


def test_a_greyscale_frame_is_handled():
    quality = assess_frame(_noise((480, 640)))

    assert quality.limits == ()


# ──────────────────────────────── the measurement actually changes the claim


def test_detail_claims_are_removed_when_the_frame_cannot_support_them():
    """The causal half. Without this the quality check is a log line."""
    reading = {
        "scene_description": "two people at a desk",
        "objects_detected": ["desk", "laptop"],
        "text_detected": ["INBOX"],
        "faces_detected": 2,
    }

    tempered = temper_reading(reading, assess_frame(_gradient()))

    assert tempered["faces_detected"] is None
    assert tempered["objects_detected"] is None
    assert tempered["text_detected"] is None
    assert tempered["detail_supported"] is False


def test_the_description_survives_tempering():
    """It is the model's honest impression and is still useful.

    What is removed is the structured detail a consumer would treat as a
    measured fact.
    """
    reading = {"scene_description": "looks like someone moved", "faces_detected": 3}

    tempered = temper_reading(reading, assess_frame(_gradient()))

    assert tempered["scene_description"] == "looks like someone moved"


def test_tempering_explains_itself():
    reading = {"faces_detected": 2}

    tempered = temper_reading(reading, assess_frame(_noise(scale=0.06)))

    assert tempered["temper_reason"]
    assert "faces_detected" in tempered["tempered_fields"]


def test_a_good_frame_is_left_alone():
    """Over-tempering would make Aura unable to report anything she saw."""
    reading = {
        "scene_description": "a desk",
        "objects_detected": ["lamp"],
        "faces_detected": 1,
    }

    tempered = temper_reading(reading, assess_frame(_noise()))

    assert tempered["faces_detected"] == 1
    assert tempered["objects_detected"] == ["lamp"]


def test_quality_is_attached_even_when_nothing_is_tempered():
    """The measurement is evidence and should be recoverable either way."""
    tempered = temper_reading({"faces_detected": 1}, assess_frame(_noise()))

    assert tempered["frame_quality"]["supports_detail"] is True


def test_presence_survives_conditions_that_kill_detail():
    """A blurred frame can carry "someone is there" and not "someone in a
    red jacket". Collapsing those loses real information."""
    quality = assess_frame(_gradient())

    assert quality.supports_detail is False
    assert quality.supports_presence is True


# ──────────────────────────────────────── wired into the capture path


@pytest.mark.asyncio
async def test_analyze_tempers_a_confident_count_on_a_bad_frame(monkeypatch):
    """End to end: the model is confident, the pixels are not."""
    from core.perception import sensory_integration as si

    class _Brain:
        async def think(self, prompt, images=None):
            return '{"scene": "two people", "objects": ["desk"], "people": 2}'

    monkeypatch.setattr(si, "optional_service", lambda name: _Brain())

    quality = assess_frame(_gradient()).to_dict()
    result = await si.VisionSystem().analyze(
        {"data": "Zm9v", "frame_quality": quality}
    )

    assert result["faces_detected"] is None, (
        "a confident count survived a frame that could not support it"
    )
    assert result["detail_supported"] is False


@pytest.mark.asyncio
async def test_analyze_keeps_the_count_on_a_good_frame(monkeypatch):
    from core.perception import sensory_integration as si

    class _Brain:
        async def think(self, prompt, images=None):
            return '{"scene": "two people", "people": 2}'

    monkeypatch.setattr(si, "optional_service", lambda name: _Brain())

    result = await si.VisionSystem().analyze(
        {"data": "Zm9v", "frame_quality": assess_frame(_noise()).to_dict()}
    )

    assert result["faces_detected"] == 2


def test_the_quality_check_needs_no_cv2():
    """cv2 cannot be imported in Aura's primary macOS process at all.

    A quality check that only ran in the sidecar would be absent exactly
    where readings are consumed.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "perception"
        / "frame_quality.py"
    ).read_text("utf-8")

    assert "import cv2" not in source


def test_quality_is_measured_where_the_pixels_still_exist():
    """`analyze` receives base64 and a path; the array is gone by then, and
    re-decoding to assess it would need cv2."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "perception"
        / "sensory_integration.py"
    ).read_text("utf-8")

    assert "capture_best_still(lease)" in source
    authority = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "perception"
        / "camera_authority.py"
    ).read_text("utf-8")
    assert "quality = assess_frame(frame)" in authority


def test_frame_quality_of_a_dataclass_round_trips():
    quality = assess_frame(_noise())
    restored = FrameQuality(
        mean_luminance=quality.mean_luminance,
        dark_fraction=quality.dark_fraction,
        bright_fraction=quality.bright_fraction,
        sharpness=quality.sharpness,
        uniformity=quality.uniformity,
        pixels=quality.pixels,
        limits=quality.limits,
    )

    assert restored.supports_detail == quality.supports_detail
