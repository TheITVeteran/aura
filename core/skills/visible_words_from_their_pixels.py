"""Keep a page's measured word labels through another reading of the same pixels."""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import replace
from typing import Any

import numpy as np

from core.perception.observed_transfer import (
    ScreenReading,
    ScreenSnapshot,
    _box,
    _overlap,
    _pixels,
    freeze_screen,
    make_snapshot,
)
from core.perception.word_labels_seen_again import WordLabelsSeenAgain


def _surface(reflexes: Any) -> tuple:
    return (("owner", str(id(reflexes.page))), ("url", str(getattr(reflexes.page, "url", ""))))


def _memory(reflexes: Any) -> WordLabelsSeenAgain | None:
    keep = getattr(reflexes, "keep", None)
    if not isinstance(keep, dict):
        return None
    words = keep.get("visible_word_labels")
    if not isinstance(words, WordLabelsSeenAgain):
        words = keep["visible_word_labels"] = WordLabelsSeenAgain()
    return words


def _local_frame(reflexes: Any, regions: list[dict], bgr: Any, at: float) -> ScreenSnapshot:
    read = ScreenReading({"layout": regions}, picture_snapshot=make_snapshot(bgr))
    frozen = freeze_screen(read)
    return replace(frozen, surface=_surface(reflexes), bounds=tuple(reflexes.band),
                   capture_at=at, epoch=f"{at:.9f}", settled=False)


def _viewport(reflexes: Any) -> tuple[float, float] | None:
    size = getattr(reflexes.page, "viewport_size", None)
    if isinstance(size, dict) and size.get("width") and size.get("height"):
        return float(size["width"]), float(size["height"])
    clip = getattr(reflexes, "_clip", None)
    left, top, right, bottom = reflexes.band
    if isinstance(clip, dict) and right > left and bottom > top:
        return float(clip["width"]) / (right - left), float(clip["height"]) / (bottom - top)
    return None


def _normal_frame(reflexes: Any, observation: Any, *, now: float, wall_now: float) -> tuple[ScreenSnapshot, bool] | None:
    """The page reader declares cropped geometry and wall-clock capture time; other clock domains remain unknown."""
    frozen = freeze_screen(observation)
    viewport = _viewport(reflexes)
    if (observation.get("on_her_page") is not True or frozen.picture is None or frozen.bounds is None
            or frozen.capture_at is None or not frozen.surface or viewport is None
            or not math.isfinite(wall_now) or not 0 <= wall_now - frozen.capture_at <= 3):
        return None
    wide, tall = viewport
    left, top, right, bottom = reflexes.band
    wanted = (left * wide, top * tall, (right - left) * wide, (bottom - top) * tall)
    within = observation.get("read_within")
    whole = within == "the page" and all(abs(a - b) <= 1 for a, b in zip(frozen.bounds, (0, 0, wide, tall), strict=True))
    part = within == "the part" and all(abs(a - b) <= 1 for a, b in zip(frozen.bounds, wanted, strict=True))
    if not (whole or part) or right <= left or bottom <= top:
        return None
    picture, regions = frozen.picture, frozen.regions
    if whole:
        pixels = _pixels(picture)
        y0, y1 = round(top * picture.height), round(bottom * picture.height)
        x0, x1 = round(left * picture.width), round(right * picture.width)
        picture = make_snapshot(pixels[y0:y1, x0:x1])
        local = []
        for region in regions:
            x, y, w, h = region.bbox
            if x >= left and y >= top and x + w <= right + 1e-6 and y + h <= bottom + 1e-6:
                local.append(replace(region, bbox=((x - left) / (right - left), (y - top) / (bottom - top),
                                                  w / (right - left), h / (bottom - top))))
        regions = tuple(local)
    expected_ratio = wanted[2] / wanted[3]
    if picture is None or abs(picture.width / picture.height / expected_ratio - 1) > .02:
        return None
    return replace(frozen, picture=picture, regions=regions, surface=_surface(reflexes), bounds=tuple(reflexes.band),
                   capture_at=now - (wall_now - frozen.capture_at)), whole


def _layout_with_labels(regions: list[dict], recovered: list[dict]) -> list[dict]:
    result = [dict(region) for region in regions]
    for label in recovered:
        box = _box(label)
        nearby = [i for i, region in enumerate(result) if _box(region) is not None and _overlap(_box(region), box) >= .6]
        if len(nearby) > 1:
            continue
        if nearby:
            i = nearby[0]
            if str(result[i].get("text") or "").strip().casefold() == label["text"]:
                continue
            result[i] = {**result[i], **label}
        else:
            result.append(dict(label))
    return result


def _without_recovered(screen: ScreenSnapshot, layout: list[dict]) -> ScreenSnapshot:
    boxes = [_box(region) for region in layout if region.get("label_from_pixels")]
    return replace(screen, regions=tuple(region for region in screen.regions
                                        if not any(box is not None and _overlap(region.bbox, box) >= .6 for box in boxes)))


def current_visible_words(reflexes: Any, observation: Any) -> Any:
    """Check retained labels before the page reading chooses a menu or starts body control."""
    if getattr(observation, "label_recovery_done", False):
        return observation
    now = time.monotonic()
    normalized = _normal_frame(reflexes, observation, now=now, wall_now=time.time())
    memory = _memory(reflexes) if normalized is not None else None
    if memory is None:
        return observation
    screen, whole = normalized
    recovered = memory.recovered(screen, now=now)
    if whole:
        left, top, right, bottom = reflexes.band
        for region in recovered:
            for key in ("x", "center_x"):
                region[key] = left + region[key] * (right - left)
            for key in ("y", "center_y"):
                region[key] = top + region[key] * (bottom - top)
            region["width"] *= right - left
            region["height"] *= bottom - top
    original = observation.get("layout") or []
    layout = _layout_with_labels(original, recovered)
    # Only the original readings teach labels; recovery never teaches itself.
    remembered_layout = layout
    if whole:
        remembered_layout = [{**region, "x": (region["x"] - left) / (right - left),
                              "y": (region["y"] - top) / (bottom - top),
                              "width": region["width"] / (right - left), "height": region["height"] / (bottom - top)}
                             for region in layout if _box(region) is not None and region.get("label_from_pixels")]
    memory.remember(_without_recovered(screen, remembered_layout), now=now)
    if layout == original:
        return observation
    result = ScreenReading({**observation, "layout": layout, "text": " ".join(str(r.get("text") or "") for r in layout)},
                           picture_snapshot=getattr(observation, "picture_snapshot", None))
    return result


async def verified_fast_labels(reflexes: Any, regions: list[dict], bgr: Any, at: float) -> list[dict]:
    """A second fresh capture measures whether the word itself stayed still while other scenery moved."""
    memory = _memory(reflexes)
    now = time.monotonic()
    if memory is None or not math.isfinite(at) or not 0 <= now - at <= .25:
        return regions
    first = _local_frame(reflexes, regions, bgr, at)
    if not any(_box(region) is not None for region in regions) and not memory.anchors:
        return regions
    await asyncio.sleep(.03)
    looked = await reflexes.look()
    now = time.monotonic()
    if looked is None or not math.isfinite(looked[1]) or not at < looked[1] <= now or now - at > .25:
        return regions
    pixels = np.asarray(looked[0])
    if pixels.ndim != 3 or pixels.shape[2] < 3 or pixels.dtype != np.uint8:
        return regions
    again = _local_frame(reflexes, [], pixels[:, :, 2::-1], looked[1])
    recovered = memory.recovered(first, now=now, again=again)
    layout = _layout_with_labels(regions, recovered)
    memory.remember(_without_recovered(first, layout), now=now, again=again)
    return layout
