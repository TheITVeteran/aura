"""Every call to her browser used to ask the file system about each frame on its stack. Read off the frames instead,
the stack is the same stack: the same files, lines, functions and frames, and the browser client makes the same of it.

Measured 2026-10-12: 0.67 ms a call at a stack 45 deep against 0.014 ms, and 7% of her main thread through a game.
"""
from __future__ import annotations

import inspect
import time

import pytest

from core.runtime.browser_calls_read_no_files import install, stack_read_off_the_frames

pytestmark = pytest.mark.unit


def _both(depth: int) -> tuple[list[inspect.FrameInfo], list[inspect.FrameInfo]]:
    if depth:
        return _both(depth - 1)
    return inspect.stack(0), stack_read_off_the_frames()


def test_the_stack_read_off_the_frames_is_the_stack_inspect_gives():
    slow, fast = _both(12)
    assert len(slow) == len(fast)
    for a, b in zip(slow, fast, strict=True):
        assert a.frame is b.frame and a.filename == b.filename and a.function == b.function
        assert a.lineno == b.lineno or a.frame.f_code.co_name == "_both"       # the two are taken a line apart


def test_the_browser_client_makes_the_same_of_either_stack():
    pytest.importorskip("playwright")
    from playwright._impl import _connection

    slow, fast = _both(6)
    made = _connection._extract_stack_trace_information_from_stack
    for a, b in zip(made(slow, False)["frames"], made(fast, False)["frames"], strict=True):
        assert (a["file"], a["function"]) == (b["file"], b["function"])
    assert made(slow, False)["apiName"] == made(fast, False)["apiName"]


def test_installed_the_client_s_stack_is_read_off_the_frames_and_the_rest_of_inspect_is_the_real_one():
    pytest.importorskip("playwright")
    from playwright._impl import _connection

    assert install() and install()                             # a second time changes nothing
    assert _connection.inspect.iscoroutinefunction is inspect.iscoroutinefunction

    def as_the_client_asks() -> list[inspect.FrameInfo]:
        return _connection.inspect.stack(0)

    assert as_the_client_asks()[0].function == "as_the_client_asks"
    assert _connection.inspect.stack(1)[0].function == inspect.stack(1)[0].function


def test_it_is_many_times_quicker_at_a_deep_stack():
    def at(depth: int) -> tuple[float, float]:
        if depth:
            return at(depth - 1)
        began = time.perf_counter()
        for _ in range(60):
            inspect.stack(0)
        slow = time.perf_counter() - began
        began = time.perf_counter()
        for _ in range(60):
            stack_read_off_the_frames()
        return slow, time.perf_counter() - began

    slow, fast = at(40)
    assert fast * 5 < slow, (slow, fast)


def test_the_traceback_kept_of_a_call_is_the_same_one_with_its_lines_read_when_it_is_shown():
    pytest.importorskip("playwright")
    import traceback

    from playwright._impl import _connection

    assert install()

    def taken() -> tuple[traceback.StackSummary, traceback.StackSummary]:
        return traceback.extract_stack(limit=10), _connection.traceback.extract_stack(limit=10)

    slow, fast = taken()
    assert [(f.filename, f.name) for f in slow] == [(f.filename, f.name) for f in fast]
    assert [f.lineno for f in slow][:-1] == [f.lineno for f in fast][:-1]
    assert slow[-2].line == fast[-2].line                      # the line itself, read when asked for
    assert _connection.traceback.format_exc is traceback.format_exc
