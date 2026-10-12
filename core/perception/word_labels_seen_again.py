"""Read a remembered label again only where its word pixels are still visible.

Offline 2026-10-12 the same end-screen button was read clearly once and then
as broken text. Its label may be kept while its measured drawing stays the
same. Words that merely sound alike supply no evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from core.perception.observed_transfer import Box, ScreenSnapshot, _overlap, _pixels
from core.perception.visual_anchor import MAX_CAPTURE_GAP, VisualAnchor, reacquire, remember_label
from core.verify import invariant

LABELS_KEPT = 32


def _word_crop(screen: ScreenSnapshot, box: Box) -> tuple[bytes, tuple[int, int]] | None:
    """Keep a whole word's pixels; a small average patch can hide a changed letter."""
    if screen.picture is None:
        return None
    x, y, width, height = box
    wide, tall = screen.picture.width, screen.picture.height
    left, top, right, bottom = round(x * wide), round(y * tall), round((x + width) * wide), round((y + height) * tall)
    if left < 0 or top < 0 or right > wide or bottom > tall or not 2 <= right - left <= 256 or not 2 <= bottom - top <= 96:
        return None
    crop = _pixels(screen.picture)[top:bottom, left:right]
    return crop.tobytes(), (bottom - top, right - left)


def _same_word(before: bytes, after: bytes) -> bool:
    """Allow uniform contrast changes and small image noise, while retaining the word's fine strokes."""
    if not before or len(before) != len(after):
        return False
    if before == after:
        return True
    old = np.frombuffer(before, np.uint8).astype(np.float32) / 255
    new = np.frombuffer(after, np.uint8).astype(np.float32) / 255
    centered = old - old.mean()
    denominator = float(centered @ centered)
    if denominator <= 1e-9:
        return False
    gain = float(centered @ (new - new.mean())) / denominator
    shift = float(new.mean() - gain * old.mean())
    if not .35 <= gain <= 1.5 or abs(shift) > .5:
        return False
    residual = np.abs(gain * old + shift - new)
    return bool(float(residual.mean()) <= .015 and float(np.quantile(residual, .99)) <= .10
                and float(np.mean(residual > .12)) <= .005)


def _fresh(screen: ScreenSnapshot, now: float) -> bool:
    return (screen.picture is not None and bool(screen.surface) and screen.bounds is not None
            and screen.capture_at is not None and math.isfinite(now) and math.isfinite(screen.capture_at)
            and 0 <= now - screen.capture_at <= 3.0 and bool(screen.epoch) and not screen.occluded)


def _same_view(before: ScreenSnapshot, after: ScreenSnapshot) -> bool:
    return (before.surface == after.surface and before.bounds == after.bounds and before.work_epoch == after.work_epoch
            and before.picture is not None and after.picture is not None
            and (before.picture.width, before.picture.height) == (after.picture.width, after.picture.height))


@dataclass(frozen=True)
class KeptWordLabel:
    anchor: VisualAnchor
    pixels: bytes = field(repr=False)
    shape: tuple[int, int]


@dataclass
class WordLabelsSeenAgain:
    """A bounded collection of measured labels; recovery cannot renew an old label's lifetime."""

    anchors: list[KeptWordLabel] = field(default_factory=list)

    def _current(self, screen: ScreenSnapshot, now: float) -> list[KeptWordLabel]:
        self.anchors[:] = [kept for kept in self.anchors if kept.anchor.surface == screen.surface
                           and kept.anchor.bounds == screen.bounds and kept.anchor.work_epoch == screen.work_epoch
                           and 0 <= now - kept.anchor.capture_at <= MAX_CAPTURE_GAP]
        return self.anchors

    def remember(self, screen: ScreenSnapshot, *, now: float, again: ScreenSnapshot | None = None) -> None:
        """Keep a label read on a settled view, or a word whose local pixels agree in two fresh captures."""
        if not _fresh(screen, now):
            return
        self._current(screen, now)
        confirmed = (again is not None and _fresh(again, now) and _same_view(screen, again)
                     and screen.capture_at < again.capture_at and screen.epoch != again.epoch)
        if not screen.settled and not confirmed:
            return
        for region in screen.regions:
            if not region.label or len(region.label) > 120:
                continue
            crop = _word_crop(screen, region.bbox)
            if crop is None:
                continue
            if not screen.settled:
                after = _word_crop(again, region.bbox)
                if after is None or after[1] != crop[1] or not _same_word(crop[0], after[0]):
                    continue
            # Settledness here is measured for this word, even when scenery elsewhere moves.
            anchor = remember_label(region.label, replace(screen, settled=True))
            if anchor is None:
                continue
            if any(kept.anchor.label == anchor.label and _overlap(kept.anchor.box, anchor.box) >= .8
                   and kept.shape == crop[1] and _same_word(kept.pixels, crop[0]) for kept in self.anchors):
                continue
            self.anchors.append(KeptWordLabel(anchor, *crop))
        del self.anchors[:-LABELS_KEPT]

    def recovered(self, screen: ScreenSnapshot, *, now: float,
                  again: ScreenSnapshot | None = None) -> list[dict[str, Any]]:
        """Return current label extents supported by a unique unchanged word drawing."""
        if not _fresh(screen, now):
            return []
        confirmed = (again is not None and _fresh(again, now) and _same_view(screen, again)
                     and screen.capture_at < again.capture_at and screen.epoch != again.epoch)
        if not screen.settled and not confirmed:
            return []
        # OCR labels are the readings being checked. Their measured extents still expose duplicate matches.
        unlabelled = tuple(replace(region, label="") for region in screen.regions)
        measured = replace(screen, regions=unlabelled, settled=True)
        matches = []
        for kept in self._current(screen, now):
            box = reacquire(kept.anchor, measured)
            crop = _word_crop(screen, box) if box is not None else None
            if crop is None or crop[1] != kept.shape or not _same_word(kept.pixels, crop[0]):
                continue
            if not screen.settled:
                after = _word_crop(again, box)
                if after is None or after[1] != kept.shape or not _same_word(kept.pixels, after[0]):
                    continue
            matches.append((kept.anchor.label, box))
        found = []
        for label, box in matches:
            if any(other != label and _overlap(box, other_box) >= .35 for other, other_box in matches):
                continue
            if any(region["text"] == label and _overlap(box, (region["x"], region["y"], region["width"], region["height"])) >= .8
                   for region in found):
                continue
            x, y, width, height = box
            found.append({"text": label, "x": x, "y": y, "width": width, "height": height,
                          "center_x": x + width / 2, "center_y": y + height / 2, "label_from_pixels": True})
        return found


@invariant("perception.visible_word_labels_require_their_current_strokes", scope="perception",
           owner="core/perception/word_labels_seen_again.py", observational=False)
def _visible_word_labels_invariant() -> tuple:
    from core.perception.observed_transfer import ScreenReading, freeze_screen, make_snapshot

    picture = np.random.default_rng(73).integers(0, 255, (64, 96, 3), dtype=np.uint8)
    before = ScreenReading({"layout": [{"text": "Continue", "x": .25, "y": .25, "width": .5, "height": .25}],
                            "surface_id": "a-view", "bounds": [0, 0, 96, 64], "capture_at": 1., "settled": True},
                           picture_snapshot=make_snapshot(picture))
    words = WordLabelsSeenAgain()
    words.remember(freeze_screen(before), now=1.1)
    after = ScreenReading({**before, "layout": [], "capture_at": 2.}, picture_snapshot=make_snapshot(picture))
    assert words.recovered(freeze_screen(after), now=2.1)[0]["text"] == "continue"
    after.picture_snapshot = make_snapshot(np.zeros_like(picture))
    assert words.recovered(freeze_screen(after), now=2.1) == [], "an absent word was restored from memory"
    return ()
