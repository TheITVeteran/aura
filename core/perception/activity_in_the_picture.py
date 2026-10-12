"""Observe visible activity without claiming what moved or what caused it.

Offline 2026-10-12 a body visibly jumped while its changing outline supplied
no certified velocity. Play ended as still. Translation, activity and a
covered quiet interval are separate observations.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from core.verify import invariant


class PictureActivity:
    """A contiguous quiet interval, measured from consecutive RGB pictures."""

    def __init__(self, *, longest_gap: float = .5, difference: float = 12., pixels: int = 4) -> None:
        self.longest_gap, self.difference, self.pixels = longest_gap, difference, pixels
        self.at: float | None = None
        self.changed: bool | None = None
        self.quiet_since: float | None = None
        self._picture: np.ndarray | None = None

    def saw(self, picture: np.ndarray, at: float) -> None:
        """Missing, resized or interrupted pictures cannot establish continuous quiet."""
        self.changed = None
        if (not math.isfinite(at) or not isinstance(picture, np.ndarray) or picture.ndim != 3
                or picture.shape[2] < 3 or min(picture.shape[:2]) < 1
                or not np.isfinite(picture[..., :3]).all()):
            self._picture, self.at, self.quiet_since = None, None, None
            return
        if self.at is not None and at <= self.at:
            self._picture, self.at, self.quiet_since = None, None, None
            return
        previous, before = self._picture, self.at
        self._picture, self.at = picture[..., :3].copy(), at
        if previous is None or previous.shape != self._picture.shape or not 0 < at - before <= self.longest_gap:
            self.quiet_since = None
            return
        mask = np.abs(self._picture.astype(np.float32) - previous.astype(np.float32)).max(axis=2) > self.difference
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), bool))
        self.changed = bool(count and np.bincount(labels.ravel())[1:].max() >= self.pixels)
        if self.changed:
            self.quiet_since = at
        elif self.quiet_since is None:
            self.quiet_since = before

    def quiet_for(self, at: float) -> float | None:
        """Seconds covered by current quiet observations; None when the interval is unmeasured."""
        if self.at != at or self.quiet_since is None or self.changed is None:
            return None
        return at - self.quiet_since


@invariant("perception.visible_activity_does_not_require_body_translation", scope="perception",
           owner="core/perception/activity_in_the_picture.py", observational=False)
def _picture_activity_invariant() -> tuple:
    picture = np.full((32, 48, 3), 30, np.uint8)
    activity = PictureActivity()
    activity.saw(picture, 1.)
    assert activity.quiet_for(1.) is None
    activity.saw(picture, 1.2)
    assert activity.quiet_for(1.2) == 1.2 - 1.
    changed = picture.copy()
    changed[12:18, 15:22] = 220
    activity.saw(changed, 1.4)
    assert activity.changed is True and activity.quiet_for(1.4) == 0.
    activity.saw(changed, 2.4)
    assert activity.quiet_for(2.4) is None
    return ()
