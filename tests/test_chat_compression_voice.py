"""She has to still sound like herself after a long conversation.

The live compressor fires at 50% of the window and flattens the oldest 70% of
the thread into a <state_snapshot> — structured, third-person, informational.
Her own replies were inside that span, so the verbatim examples of how she
talks were replaced by a description of what she said, and the model then had
nothing to imitate but its own meeting minutes.

Summarising is a register-destroying operation. These pin the property that
some of her actual voice always survives it.
"""
from __future__ import annotations

import pytest

from core.context.chat_compression import (
    VOICE_ANCHOR_TURNS,
    find_compress_split_point,
    preserve_voice_split,
)

pytestmark = pytest.mark.unit


def _thread(turns: int):
    """A conversation: alternating user asks and her replies."""
    history = []
    for i in range(turns):
        history.append({"role": "user", "content": f"question {i} " + "x" * 200})
        history.append({"role": "assistant", "content": f"her reply {i} " + "y" * 200})
    return history


def _surviving_own_turns(history, split_point):
    return [m for m in history[split_point:] if m["role"] == "assistant"]


# ── the property ───────────────────────────────────────────────────────────


def test_her_recent_turns_survive_the_split():
    history = _thread(40)
    raw = find_compress_split_point(history, 0.7)

    adjusted = preserve_voice_split(history, raw)

    assert len(_surviving_own_turns(history, adjusted)) >= VOICE_ANCHOR_TURNS


def test_the_split_moves_earlier_only_when_it_has_to():
    history = _thread(40)
    raw = find_compress_split_point(history, 0.7)

    adjusted = preserve_voice_split(history, raw)

    assert adjusted <= raw


def test_a_split_that_already_keeps_enough_is_untouched():
    history = _thread(40)
    generous = len(history) - 20  # tail already holds many of her turns

    assert preserve_voice_split(history, generous) == generous


def test_the_surviving_turns_are_her_most_recent_ones():
    """Stale examples of her voice are worth less than current ones."""
    history = _thread(40)
    raw = find_compress_split_point(history, 0.7)

    adjusted = preserve_voice_split(history, raw)

    survivors = _surviving_own_turns(history, adjusted)
    assert survivors[-1] == history[-1]


def test_her_replies_stay_next_to_what_prompted_them():
    """Moving the boundary rather than extracting her messages: lifting them
    out would leave a transcript of her talking to nobody."""
    history = _thread(40)
    adjusted = preserve_voice_split(history, find_compress_split_point(history, 0.7))

    kept = history[adjusted:]

    for i, message in enumerate(kept):
        if message["role"] == "assistant" and i > 0:
            assert kept[i - 1]["role"] == "user"


# ── bounded, so compression still happens ──────────────────────────────────


def test_preservation_never_eats_the_whole_history():
    """A context that does not fit is not a context."""
    history = [{"role": "assistant", "content": "her turn"}] * 50

    adjusted = preserve_voice_split(history, 40, anchors=100)

    assert adjusted > 0


def test_the_floor_bounds_how_much_is_protected():
    history = _thread(40)

    adjusted = preserve_voice_split(history, 70, anchors=999, floor_fraction=0.5)

    assert adjusted >= int(len(history) * 0.5)


def test_something_is_always_left_to_compress():
    history = _thread(40)

    adjusted = preserve_voice_split(history, find_compress_split_point(history, 0.7))

    assert adjusted > 0
    assert len(history[:adjusted]) > 0


# ── switches and edges ─────────────────────────────────────────────────────


def test_anchoring_can_be_switched_off():
    history = _thread(40)
    raw = find_compress_split_point(history, 0.7)

    assert preserve_voice_split(history, raw, anchors=0) == raw


def test_an_empty_history_is_not_an_error():
    assert preserve_voice_split([], 0) == 0


def test_a_thread_with_none_of_her_turns_does_not_hang():
    history = [{"role": "user", "content": "hi"}] * 20

    adjusted = preserve_voice_split(history, 15)

    assert 0 <= adjusted <= 15


def test_the_anchored_role_is_configurable():
    history = _thread(40)
    raw = find_compress_split_point(history, 0.7)

    adjusted = preserve_voice_split(history, raw, role="user", anchors=3)

    kept_users = [m for m in history[adjusted:] if m["role"] == "user"]
    assert len(kept_users) >= 3
