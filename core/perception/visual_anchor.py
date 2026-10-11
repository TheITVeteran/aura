"""Reacquire a remembered label from fresh pixels before offering its location.

The label supplies a search hypothesis, not evidence that the control persists.
Small motion and uniform fading are allowed; context, distinct appearance and
an unambiguous current match are required. Presence does not imply enabledness.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.perception.observed_transfer import (
    Box,
    ScreenSnapshot,
    Surface,
    _distinctive,
    _matches,
    _overlap,
    _patch,
    freeze_screen,
)
from core.verify.invariants import invariant

MAX_CAPTURE_GAP = 120.0
MAX_SEARCH_RADIUS = 8
MAX_NORMALIZED_ERROR = .025


@dataclass(frozen=True)
class VisualAnchor:
    label: str
    box: Box
    appearance: bytes = field(repr=False)
    surface: Surface
    bounds: tuple[float, float, float, float]
    pixel_size: tuple[int, int]
    capture_at: float
    epoch: str
    work_epoch: str
    episode: str


def _measured(screen: ScreenSnapshot) -> bool:
    return (screen.picture is not None and bool(screen.surface) and screen.bounds is not None
            and screen.capture_at is not None and bool(screen.epoch) and screen.settled and not screen.occluded)


def remember_label(label: str, observation: Mapping[str, Any] | ScreenSnapshot, *, episode: str = "") -> VisualAnchor | None:
    """Bind a label's unique measured extent and immutable appearance to its capture."""
    screen = freeze_screen(observation)
    if not _measured(screen):
        return None
    boxes = {region.bbox for region in screen.regions if region.label == label.casefold().strip()}
    if len(boxes) != 1:
        return None
    box = next(iter(boxes))
    appearance = _patch(screen.picture, box)
    if not _distinctive(appearance):
        return None
    return VisualAnchor(label, box, appearance, screen.surface, screen.bounds,
                        (screen.picture.width, screen.picture.height), screen.capture_at,
                        screen.epoch, screen.work_epoch, episode)


def _appearance_error(before: bytes, now: bytes) -> float | None:
    if not _distinctive(now) or len(before) != len(now):
        return None
    old = np.frombuffer(before, np.uint8).astype(np.float32) / 255.
    new = np.frombuffer(now, np.uint8).astype(np.float32) / 255.
    if _matches(before, now):
        return float(np.mean(np.abs(old - new)))
    # A faded button retains its structure. Fit one positive contrast and
    # brightness change, with bounded residual rather than a best-match rank.
    centered = old - old.mean()
    denominator = float(centered @ centered)
    if denominator <= 1e-9:
        return None
    gain = float(centered @ (new - new.mean())) / denominator
    shift = float(new.mean() - gain * old.mean())
    if not .35 <= gain <= 1.5 or abs(shift) > .5:
        return None
    residual = np.abs(gain * old + shift - new)
    error = float(residual.mean())
    return error if error <= MAX_NORMALIZED_ERROR and float(np.quantile(residual, .9)) <= .06 else None


def reacquire(anchor: VisualAnchor, observation: Mapping[str, Any] | ScreenSnapshot, *, episode: str = "") -> Box | None:
    """Return one current measured match, or unknown when evidence is insufficient."""
    screen = freeze_screen(observation)
    if (not _measured(screen) or screen.surface != anchor.surface or screen.bounds != anchor.bounds
            or screen.work_epoch != anchor.work_epoch or episode != anchor.episode
            or screen.epoch == anchor.epoch or not 0 < screen.capture_at - anchor.capture_at <= MAX_CAPTURE_GAP
            or (screen.picture.width, screen.picture.height) != anchor.pixel_size):
        return None
    x, y, width, height = anchor.box
    rx = min(MAX_SEARCH_RADIUS, max(2, round(width * screen.picture.width * .15)))
    ry = min(MAX_SEARCH_RADIUS, max(2, round(height * screen.picture.height * .15)))
    candidates = {anchor.box}
    for dx in range(-rx, rx + 1):
        for dy in range(-ry, ry + 1):
            candidates.add((x + dx / screen.picture.width, y + dy / screen.picture.height, width, height))
    # Measured extents outside the local search also expose duplicate matches.
    candidates.update(region.bbox for region in screen.regions
                      if .9 <= region.bbox[2] / width <= 1.1 and .9 <= region.bbox[3] / height <= 1.1)
    matches = []
    for box in candidates:
        if any(region.label and region.label != anchor.label.casefold().strip() and _overlap(region.bbox, box) >= .8
               for region in screen.regions):
            continue
        error = _appearance_error(anchor.appearance, _patch(screen.picture, box))
        if error is not None:
            matches.append((error, box))
    distinct: list[Box] = []
    for _, box in sorted(matches):
        if not any(_overlap(box, other) >= .35 for other in distinct):
            distinct.append(box)
    if len(distinct) != 1:
        return None
    box = distinct[0]
    if abs(box[0] - x) > rx / screen.picture.width or abs(box[1] - y) > ry / screen.picture.height:
        return None
    return box


@invariant("perception.remembered_labels_need_current_pixels", scope="perception",
           owner="core/perception/visual_anchor.py", observational=False)
def _remembered_labels_invariant() -> tuple:
    from core.perception.observed_transfer import ScreenReading, make_snapshot

    picture = np.random.default_rng(17).integers(0, 255, (40, 40, 3), dtype=np.uint8)
    before = ScreenReading({"layout": [{"text": "Continue", "x": .25, "y": .25, "width": .5, "height": .5}],
                            "surface_id": "editor", "bounds": [0, 0, 40, 40], "capture_at": 1.,
                            "_capture_epoch": "one", "settled": True}, picture_snapshot=make_snapshot(picture))
    anchor = remember_label("Continue", before)
    assert anchor is not None
    after = ScreenReading({**before, "layout": [], "capture_at": 2., "_capture_epoch": "two"},
                          picture_snapshot=make_snapshot(picture))
    assert reacquire(anchor, after) is not None
    assert reacquire(anchor, before) is None
    after.picture_snapshot = make_snapshot(np.zeros_like(picture))
    assert reacquire(anchor, after) is None, "memory supplied a disappeared control"
    return ()
