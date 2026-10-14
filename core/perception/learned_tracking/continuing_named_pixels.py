"""Use Aura's original RGB subpixel measurements for supplied continuing queries."""
from __future__ import annotations

from scipy import ndimage

from core.perception.learned_tracking.patches_with_independent_matches import PatchMatch
from core.perception.points_before_the_picture_is_reduced import PointsBeforeReduction, _colours
from core.perception.points_that_keep_their_motion import SeenPoint, _pyramid


class ContinuingNamedPixels:
    """No corner discovery or new names can enter these supplied correspondence slots."""

    def __init__(self, limit: int, longest_gap: float) -> None:
        self._pixels = PointsBeforeReduction(limit=limit, pixel_budget=1_000_000, longest_gap=longest_gap)

    def _retain(self, frame) -> tuple:
        if frame.shape[0] * frame.shape[1] > self._pixels.pixel_budget:
            self._pixels.clear()
            return None, None
        colours = _colours(frame.pixels)
        image = _pyramid(colours.mean(axis=2))
        return colours, image

    def start(self, frame) -> None:
        self._pixels.clear()
        colours, image = self._retain(frame)
        self._pixels._colours, self._pixels._image, self._pixels._at = colours, image, frame.at

    def measure(self, frame, queries: dict[int, tuple[float, float]]) -> dict[int, PatchMatch]:
        colours, image = self._retain(frame)
        tracker = self._pixels
        measured = []
        if tracker._image is not None and image is not None and queries:
            tracker._native_points = tuple(SeenPoint(i, x, y, x, y, tracker._at, tracker._at,
                                                    tracker._at, 0., 1) for i, (x, y) in queries.items())
            measured = tracker._continued_source(image, colours, frame.at)
            found = {p.number for p in measured}
            remaining = tuple(p for p in tracker._native_points if p.number not in found)
            if remaining:
                # Raster motion adds sampling blur. A second common scale
                # retains the same forward/reverse residual checks.
                before_colours, before_image = tracker._colours, tracker._image
                tracker._native_points = remaining
                tracker._colours = ndimage.gaussian_filter(before_colours, (.8, .8, 0.))
                tracker._image = _pyramid(tracker._colours.mean(axis=2))
                coarse = ndimage.gaussian_filter(colours, (.8, .8, 0.))
                try:
                    measured.extend(tracker._continued_source(_pyramid(coarse.mean(axis=2)), coarse, frame.at))
                finally:
                    tracker._colours, tracker._image = before_colours, before_image
        tracker._colours, tracker._image, tracker._at = colours, image, frame.at
        return {p.number: PatchMatch(p.x, p.y, min(.99, max(.5, 1.-p.error*10.))) for p in measured}

    def close(self) -> None:
        self._pixels.clear()
