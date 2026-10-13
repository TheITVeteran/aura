"""Optional inspection cannot erase a completed, independently rechecked read."""
from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import numpy as np
import pytest

import core.perception.a_reading_before_more_detail as stages
from core.perception.observed_transfer import ScreenReading, make_snapshot

pytestmark = pytest.mark.unit


def _reading(text="START"):
    pixels = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    return ScreenReading({"ok": True, "text": text, "layout": [{"text": text}], "bounds": [0, 0, 12, 12],
                          "capture_at": 1., "_capture_epoch": "capture", "surface_id": "test surface", "settled": True},
                         picture_snapshot=make_snapshot(pixels))


async def _yes():
    return True


@pytest.mark.asyncio
async def test_a_completed_read_is_detached_before_slow_detail_changes_it():
    reading = _reading()

    async def read():
        stages.keep_the_finished_reading(reading, _yes)
        reading["layout"][0]["text"] = "unfinished detail"
        reading["shapes"] = [{"partial": True}]
        await asyncio.sleep(10.)

    result = await stages.read_within_its_time(read, .02)
    assert result["text"] == result["layout"][0]["text"] == "START"
    assert result["extra_inspection_incomplete"] is True and "shapes" not in result
    assert result.picture_snapshot == reading.picture_snapshot
    assert result["capture_at"] == 1. and result["_capture_epoch"] == "capture"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["pixels", "missing_capture", "missing_epoch", "unsettled", "occluded", "hidden", "refused", "plain_dict"])
async def test_an_incomplete_or_changed_capture_cannot_survive_a_timeout(change):
    reading = _reading()
    if change == "missing_capture":
        reading.picture_snapshot = None
    elif change == "missing_epoch":
        reading.pop("_capture_epoch")
    elif change == "unsettled":
        reading["settled"] = False
    elif change == "occluded":
        reading["occluded"] = True
    elif change == "hidden":
        reading["her_window_showing"] = False
    elif change == "refused":
        reading["ok"] = False
    elif change == "plain_dict":
        reading = dict(reading)

    async def agrees():
        return change != "pixels"

    async def read():
        stages.keep_the_finished_reading(reading, agrees)
        await asyncio.sleep(10.)

    with pytest.raises(TimeoutError):
        await stages.read_within_its_time(read, .02)


@pytest.mark.asyncio
async def test_fast_complete_detail_is_returned_without_a_second_capture():
    complete = _reading()

    async def forbidden():
        pytest.fail("no second capture was needed")

    async def read():
        stages.keep_the_finished_reading(complete, forbidden)
        complete["shapes"] = [{"text": "triangle"}]
        return complete

    assert await stages.read_within_its_time(read, 1.) is complete
    assert "extra_inspection_incomplete" not in complete


@pytest.mark.asyncio
async def test_external_cancellation_is_never_converted_to_a_reading():
    entered = asyncio.Event()

    async def read():
        stages.keep_the_finished_reading(_reading(), _yes)
        entered.set()
        await asyncio.sleep(10.)

    task = asyncio.create_task(stages.read_within_its_time(read, 1.))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_parallel_reads_keep_their_own_captures_and_cancel_their_own_detail():
    cancelled = set()

    async def one(label):
        async def read():
            stages.keep_the_finished_reading(_reading(label), _yes)
            try:
                await asyncio.sleep(10.)
            finally:
                cancelled.add(label)
        return await stages.read_within_its_time(read, .02)

    first, second = await asyncio.gather(one("first"), one("second"))
    assert first["text"] == "first" and second["text"] == "second"
    assert cancelled == {"first", "second"}


@pytest.mark.asyncio
async def test_an_old_call_cannot_publish_into_a_new_call():
    old_context = None

    async def first():
        nonlocal old_context
        from contextvars import copy_context
        old_context = copy_context()
        return _reading("first")

    await stages.read_within_its_time(first, 1.)

    async def second():
        old_context.run(stages.keep_the_finished_reading, _reading("old"), _yes)
        await asyncio.sleep(10.)

    with pytest.raises(TimeoutError):
        await stages.read_within_its_time(second, .02)


@pytest.mark.asyncio
async def test_validation_has_its_own_bound(monkeypatch):
    monkeypatch.setattr(stages, "CHECK_WITHIN_S", .02)

    async def stuck():
        await asyncio.sleep(10.)

    async def read():
        stages.keep_the_finished_reading(_reading(), stuck)
        await asyncio.sleep(10.)

    with pytest.raises(TimeoutError, match="could not be checked"):
        await stages.read_within_its_time(read, .02)


@pytest.mark.asyncio
async def test_a_validator_that_cannot_measure_does_not_count_as_agreement():
    async def unknown():
        return None

    async def read():
        stages.keep_the_finished_reading(_reading(), unknown)
        await asyncio.sleep(10.)

    with pytest.raises(TimeoutError, match="surface changed"):
        await stages.read_within_its_time(read, .02)


@pytest.mark.asyncio
async def test_a_child_read_does_not_replace_its_parents_completed_reading():
    async def child():
        stages.keep_the_finished_reading(_reading("child"), _yes)
        await asyncio.sleep(10.)

    async def parent():
        stages.keep_the_finished_reading(_reading("parent"), _yes)
        assert (await stages.read_within_its_time(child, .02))["text"] == "child"
        await asyncio.sleep(10.)

    assert (await stages.read_within_its_time(parent, .05))["text"] == "parent"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["none", "pixels", "url", "size"])
async def test_the_real_page_read_survives_only_when_a_new_capture_agrees(monkeypatch, change):
    from core.perception import a_picture_of_her_page as camera
    from core.perception import what_the_pixels_show as pixels
    from core.skills.screen_pursuit_on_a_page import OnAPage

    picture = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    page = SimpleNamespace(viewport_size={"width": 12, "height": 12}, url="https://example.test/editor")
    captures = []

    async def picture_of(*args, **kwargs):
        captures.append(1)
        return picture.copy()

    async def settled(looker, take, *args, **kwargs):
        measured = await take()
        reading = _reading()
        reading.picture_snapshot = make_snapshot(measured)
        return reading, True, False, 2, (12, 12), .001

    async def watch(*args):
        return None

    async def more_detail(*args):
        if change == "pixels":
            picture[:] = 0
        elif change == "url":
            page.url = "https://example.test/another"
        elif change == "size":
            page.viewport_size = {"width": 14, "height": 12}
        await asyncio.sleep(10.)

    monkeypatch.setattr(camera, "picture_of", picture_of)
    monkeypatch.setattr(pixels, "_read_until_settled", settled)
    monkeypatch.setattr(pixels, "looker_for", lambda _: object())
    monkeypatch.setattr(OnAPage, "_watch", watch)
    monkeypatch.setattr(OnAPage, "_where_the_pointer_turns", more_detail)
    if change == "none":
        result = await stages.read_within_its_time(OnAPage(page, "editor").read, .02)
        assert result["on_her_page"] is True and result["text"] == "START"
        assert result["extra_inspection_incomplete"] is True and len(captures) == 2
    else:
        with pytest.raises(TimeoutError):
            await stages.read_within_its_time(OnAPage(page, "editor").read, .02)


def test_finished_reading_invariant_and_private_bytes():
    assert stages._finished_reading_invariant() == ()
    reading = copy.deepcopy(_reading())
    assert "picture_snapshot" not in dict(reading)


@pytest.mark.asyncio
async def test_the_pursuit_uses_the_finished_reading_after_slow_detail(monkeypatch):
    from core.skills import screen_pursuit_observing as observing
    from tests.screen_pursuit_support import patch_pursuit

    async def read(*args, **kwargs):
        stages.keep_the_finished_reading(_reading("editor"), _yes)
        await asyncio.sleep(10.)

    async def in_front(*args):
        return True

    patch_pursuit(monkeypatch, "read_screen", read)
    monkeypatch.setattr(observing, "_ensure_frontmost", in_front)
    monkeypatch.setattr(observing, "_ensure_page", in_front)
    monkeypatch.setattr(observing, "_how_long_a_look_takes", lambda _: .02)
    run = SimpleNamespace(anchor={"app": "editor", "page": "example", "settled": True},
                          at_rest={"reading": None}, drawn={"where": None, "last_seen": ""}, expect_page="example",
                          lost_page={"value": False}, open_page=None, reading_took=[], target_app="")
    seen = await observing.observe_the_screen(run)
    assert seen["text"] == run.drawn["last_seen"] == "editor"
    assert seen["extra_inspection_incomplete"] is True and len(run.reading_took) == 1
