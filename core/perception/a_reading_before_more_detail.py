"""Keep a finished reading if extra inspection cannot finish in time.

Offline 2026-10-12 reading took under two seconds, then a pointer sweep ran
past its deadline. The completed reading was discarded on each attempt.
A retained reading can be used only if its surface and pixels still agree.
"""
from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from core.perception.observed_transfer import PixelSnapshot, ScreenReading, freeze_screen
from core.verify import invariant

logger = logging.getLogger(__name__)
_CURRENT: ContextVar[_Reading | None] = ContextVar("a_finished_screen_reading", default=None)
CHECK_WITHIN_S = 1.0


@dataclass
class _Reading:
    finished: ScreenReading | None = None
    agrees: Callable[[], Awaitable[bool]] | None = None
    open: bool = True


def keep_the_finished_reading(reading: Any, agrees: Callable[[], Awaitable[bool]]) -> None:
    """Save private capture evidence within the call that requested this read."""
    current = _CURRENT.get()
    if current is None or not current.open or not isinstance(reading, ScreenReading) or reading.get("ok") is not True:
        return
    screen = freeze_screen(reading)
    if (screen.picture is None or not screen.surface or screen.bounds is None or screen.capture_at is None
            or not reading.get("_capture_epoch") or not screen.settled or screen.occluded
            or reading.get("her_window_showing") is False or reading.get("refused_because")):
        return
    current.finished, current.agrees = copy.deepcopy(reading), agrees


async def read_within_its_time(read: Callable[[], Awaitable[Any]], seconds: float) -> Any:
    """A finished read, or a retained one checked against a new capture; otherwise the timeout propagates."""
    current = _Reading()
    token = _CURRENT.set(current)
    try:
        try:
            return await asyncio.wait_for(read(), timeout=seconds)
        except TimeoutError:
            current.open = False
            if current.finished is None or current.agrees is None:
                raise
            try:
                agrees = await asyncio.wait_for(current.agrees(), timeout=CHECK_WITHIN_S)
            except (TimeoutError, OSError, RuntimeError, TypeError, ValueError) as why:
                logger.info("a retained reading could not be checked: %s", why)
                raise TimeoutError("the finished reading could not be checked") from why
            if agrees is not True:
                raise TimeoutError("the surface changed during extra inspection") from None
            current.finished["extra_inspection_incomplete"] = True
            logger.info("using the finished reading after extra inspection exceeded %.1fs; its pixels still agree", seconds)
            return current.finished
    finally:
        current.open = False
        current.finished, current.agrees = None, None
        _CURRENT.reset(token)


@invariant("perception.finished_readings_keep_their_capture", scope="perception",
           owner="core/perception/a_reading_before_more_detail.py", observational=False)
def _finished_reading_invariant() -> tuple:
    current = _Reading()
    token = _CURRENT.set(current)
    try:
        keep_the_finished_reading({"ok": True, "text": "unmeasured"}, lambda: None)
        assert current.finished is None
        reading = ScreenReading({"ok": True, "layout": [{"text": "start"}], "bounds": [0, 0, 2, 2],
                                 "capture_at": 1., "_capture_epoch": "capture", "surface_id": "example", "settled": True},
                                picture_snapshot=PixelSnapshot(bytes(range(12)), 2, 2))
        keep_the_finished_reading(reading, lambda: None)
        reading["layout"][0]["text"] = "unfinished"
        assert current.finished["layout"][0]["text"] == "start"
        assert current.finished.picture_snapshot == reading.picture_snapshot
        current.finished = None
        current.open = False
        keep_the_finished_reading(reading, lambda: None)
        assert current.finished is None
    finally:
        _CURRENT.reset(token)
    return ()
