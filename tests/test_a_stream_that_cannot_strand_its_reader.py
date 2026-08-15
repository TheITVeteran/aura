"""The nucleus streaming bridge, and the bounds around generation.

The worker thread caught four exception classes and posted the None sentinel
only on the happy path and those four, so an OSError, an ImportError, a
TimeoutError, a cancellation or a closed loop exited the thread without
enqueueing it — and the consumer's `await queue.get()` waited forever for a
producer that had already died.

It also scheduled `put_nowait` into an unbounded queue, so a fast model outran a
slow consumer and every chunk it had ever produced stayed resident.

The generate loop applied no stop sequences, so a model that started writing the
next turn streamed `<|im_start|>user` to the consumer as its own answer.

And after the 60-second sentinel acquisition, nothing bounded how long the call
could then run.
"""
from __future__ import annotations

import math

import pytest

from core.brain.llm.nucleus_manager import (
    _DEFAULT_GENERATION_BUDGET_S,
    _MAX_GENERATION_BUDGET_S,
    _MAX_PREFILL_CHARS,
    _MIN_GENERATION_BUDGET_S,
    _STREAM_QUEUE_HIGH_WATER,
    _accepted_generation_budget_s,
    _bounded_prefill,
    _stream_stop_index,
)

SOURCE = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "core" / "brain" / "llm" / "nucleus_manager.py"
).read_text("utf-8")


# ── stop sequences ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    ["hello <|im_end|>", "hello <|im_start|>user", "answer\nUser: next question"],
)
def test_a_role_marker_stops_the_stream(text):
    assert _stream_stop_index(text) is not None


def test_ordinary_text_does_not_stop_the_stream():
    assert _stream_stop_index("a perfectly normal sentence about users") is None


def test_the_earliest_stop_wins():
    text = "abc<|im_end|>def<|im_start|>"

    assert _stream_stop_index(text) == 3


# ── the sentinel is always posted ─────────────────────────────────────────


def test_the_worker_posts_the_sentinel_in_a_finally():
    """This is the property that keeps a reader from waiting forever."""
    marker = SOURCE.index("def _thread_worker():")
    block = SOURCE[marker : marker + 2000]

    assert "finally:" in block
    assert "_post(None)" in block
    assert "except (RuntimeError, AttributeError, TypeError, ValueError) as e:" not in block


def test_a_closed_loop_does_not_wedge_the_worker():
    marker = SOURCE.index("def _post(item: Any) -> None:")
    block = SOURCE[marker : marker + 900]

    assert "except RuntimeError:" in block
    assert "stop_event.set()" in block


# ── backpressure ──────────────────────────────────────────────────────────


def test_the_producer_waits_when_the_reader_falls_behind():
    marker = SOURCE.index("def _thread_worker():")
    block = SOURCE[marker : marker + 2000]

    assert "_STREAM_QUEUE_HIGH_WATER" in block
    assert "queue.qsize()" in block
    assert _STREAM_QUEUE_HIGH_WATER > 0


# ── prefill ───────────────────────────────────────────────────────────────


def test_a_prefill_is_bounded():
    assert len(_bounded_prefill("x" * 10_000)) == _MAX_PREFILL_CHARS


@pytest.mark.parametrize("value", [None, 42, 3.5, ["a"], {"b": 1}])
def test_a_non_string_prefill_does_not_raise(value):
    assert isinstance(_bounded_prefill(value), str)


def test_an_ordinary_prefill_is_untouched():
    assert _bounded_prefill("Sure, ") == "Sure, "


# ── the generation deadline ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (None, _DEFAULT_GENERATION_BUDGET_S),
        (0, _DEFAULT_GENERATION_BUDGET_S),
        (-1, _DEFAULT_GENERATION_BUDGET_S),
        ("nonsense", _DEFAULT_GENERATION_BUDGET_S),
        (float("nan"), _DEFAULT_GENERATION_BUDGET_S),
        (float("inf"), _DEFAULT_GENERATION_BUDGET_S),
        (1.0, _MIN_GENERATION_BUDGET_S),
        (60.0, 60.0),
        (10_000.0, _MAX_GENERATION_BUDGET_S),
    ],
)
def test_the_generation_budget_is_clamped(requested, expected):
    assert _accepted_generation_budget_s(requested) == expected


def test_the_budget_is_always_finite_and_positive():
    for candidate in (None, 0, -5, "x", float("nan"), 1e30):
        value = _accepted_generation_budget_s(candidate)
        assert math.isfinite(value) and value > 0.0


def test_generation_is_awaited_under_that_budget():
    """mlx generation cannot be interrupted mid-call, so this cannot kill the
    work — it stops the caller waiting forever and marks the lane, which is the
    difference between a slow answer and a runtime that has stopped answering.
    """
    marker = SOURCE.index("_accepted_generation_budget_s(kwargs.get(\"deadline_s\"))")
    block = SOURCE[marker : marker + 1600]

    assert "asyncio.wait_for(" in block
    assert "timeout=budget" in block
    assert "except TimeoutError:" in block
    assert "not preemptible" in block
