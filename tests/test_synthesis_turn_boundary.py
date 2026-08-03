"""What synthesis tells the user when tools half-worked, and what it keeps.

Seven CP126 findings in ConversationalSynthesizer, all about the same seam: the
boundary that owns a user's turn was narrating unverified claims, truncating
mid-structure, retaining everything, and dying on inputs it did not expect.
"""
from __future__ import annotations

import pytest

from core.synthesis import (
    ConversationalSynthesizer,
    _readable_mood,
    _render_tool_results,
    _synthesis_failure_reply,
    _tool_result_verification,
    generate_offline_fallback_response,
)

pytestmark = pytest.mark.unit


class _Thought:
    def __init__(self, content):
        self.content = content


class _Brain:
    def __init__(self, reply="Here's the answer.", raises=None):
        self.reply = reply
        self.raises = raises
        self.prompts: list[str] = []

    async def think(self, prompt):
        self.prompts.append(prompt)
        if self.raises:
            raise self.raises
        return _Thought(self.reply)


# --- a failed tool is not narrated as a success (7c5c33ea) --------------


def test_a_failed_tool_is_flagged_to_the_writer():
    block = _tool_result_verification([{"ok": False, "tool": "search_web"}])

    assert "1 reported failure" in block
    assert "FAILED" in block


def test_an_unreceipted_success_is_flagged_as_unverified():
    block = _tool_result_verification([{"ok": True, "tool": "send_email"}])

    assert "no execution receipt" in block


def test_a_receipted_success_is_not_flagged():
    block = _tool_result_verification(
        [{"ok": True, "tool": "send_email", "receipt": "R-1"}]
    )

    assert "no execution receipt" not in block


@pytest.mark.asyncio
async def test_the_prompt_carries_the_outcome_summary():
    brain = _Brain()

    await ConversationalSynthesizer().synthesize_response(
        "did it send?", [{"ok": False, "tool": "send_email"}], brain=brain
    )

    assert "TOOL OUTCOMES" in brain.prompts[0]
    assert "1 reported failure" in brain.prompts[0]


# --- truncation does not fabricate a whole world (08551d41) -------------


def test_a_result_is_included_whole_or_not_at_all():
    big = {"tool": "x", "payload": "y" * 5000}
    rendered, dropped = _render_tool_results([big, big, big])

    assert dropped == 2
    assert rendered.count("[0]") == 1
    assert "...(truncated)" not in rendered


def test_the_dropped_count_reaches_the_writer():
    """A reply must not describe results it was never shown."""
    rendered, dropped = _render_tool_results(
        [{"payload": "z" * 5000}, {"payload": "z" * 5000}]
    )

    assert dropped == 1
    assert rendered


def test_small_results_all_survive():
    rendered, dropped = _render_tool_results([{"a": 1}, {"b": 2}])

    assert dropped == 0
    assert "[0]" in rendered and "[1]" in rendered


# --- an unreadable mood is no mood (665d669f) ---------------------------


@pytest.mark.parametrize(
    "context",
    [
        None,
        {},
        {"affective_state": None},
        {"affective_state": "angry"},
        {"affective_state": ["angry"]},
        {"affective_state": {"mood": "low"}},
        {"affective_state": {"mood": None}},
        {"affective_state": {"mood": True}},
        {"affective_state": {"mood": float("nan")}},
    ],
)
def test_an_unreadable_mood_reads_as_none(context):
    assert _readable_mood(context) is None


def test_a_readable_mood_is_clamped():
    assert _readable_mood({"affective_state": {"mood": 0.2}}) == 0.2
    assert _readable_mood({"affective_state": {"mood": 5.0}}) == 1.0
    assert _readable_mood({"affective_state": {"mood": -3.0}}) == 0.0


@pytest.mark.asyncio
async def test_a_wrong_shaped_affective_state_does_not_kill_the_turn():
    """It used to raise AttributeError from inside synthesis, over a cosmetic
    tone adjustment."""
    brain = _Brain(reply="What a wonderful result.")

    reply = await ConversationalSynthesizer().synthesize_response(
        "how did it go?",
        [{"ok": True}],
        context={"affective_state": "unstable"},
        brain=brain,
    )

    assert "wonderful" in reply


@pytest.mark.asyncio
async def test_a_low_mood_still_dampens_cheer():
    brain = _Brain(reply="What a wonderful result.")

    reply = await ConversationalSynthesizer().synthesize_response(
        "how did it go?",
        [{"ok": True}],
        context={"affective_state": {"mood": 0.1}},
        brain=brain,
    )

    assert "wonderful" not in reply


# --- the turn boundary holds (bacf3543, 8daf2d65) -----------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [ValueError("bad shape"), KeyError("missing"), TypeError("nope")]
)
async def test_an_unexpected_error_does_not_escape_the_turn(error):
    """Only OSError/ConnectionError/TimeoutError were caught, so a malformed
    result killed the turn and the user got a blank screen."""
    brain = _Brain(raises=error)

    reply = await ConversationalSynthesizer().synthesize_response(
        "what are the tide tables", [{"ok": True}], brain=brain
    )

    assert reply
    assert "synthesis failed" in reply


def test_the_failure_reply_names_the_task_and_the_stage():
    reply = _synthesis_failure_reply("what are the tide tables", ValueError("x"))

    assert "tide tables" in reply
    assert "ValueError" in reply
    assert "Nothing was sent or changed" in reply


def test_the_failure_reply_survives_an_empty_task():
    reply = _synthesis_failure_reply("", RuntimeError("x"))

    assert "RuntimeError" in reply


# --- history is bounded and not verbatim (157f7188) ---------------------


def test_history_does_not_retain_the_conversation_verbatim():
    record = ConversationalSynthesizer._turn_record(
        "my card number is 4111 1111 1111 1111 and the pin is 4242",
        "I won't repeat that back.",
        [{"tool": "noop"}],
    )

    assert "4242" not in str(record)
    assert "4111" not in str(record)
    assert record["user_chars"] > 0
    assert len(record["user_digest"]) == 12
    assert record["tools_used"] == ["noop"]


def test_history_is_capped():
    from core.synthesis import _MAX_HISTORY_TURNS

    synth = ConversationalSynthesizer()
    for i in range(_MAX_HISTORY_TURNS * 3):
        synth._remember_turn(f"message {i}", "reply", [])

    assert len(synth.conversation_history) == _MAX_HISTORY_TURNS
    # The most recent turns are the ones kept.
    last = ConversationalSynthesizer._turn_record(
        f"message {_MAX_HISTORY_TURNS * 3 - 1}", "reply", []
    )
    assert synth.conversation_history[-1]["user_digest"] == last["user_digest"]


# --- offline subject selection matches words, not substrings (42cb9651) --


@pytest.mark.parametrize(
    "prompt,unexpected",
    [
        ("what time is brunch", "technical request"),
        ("don't forget the milk", "lookup"),
        ("tell me about the whole thing", "question"),
        ("that was a terror of a day", "technical request"),
    ],
)
def test_an_incidental_substring_no_longer_picks_the_subject(prompt, unexpected):
    assert unexpected not in generate_offline_fallback_response(prompt)


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("debug this stack trace", "technical request"),
        ("search for the ferry timetable", "lookup"),
        ("why does the tide turn", "question"),
        ("is it nine or ten?", "question"),
    ],
)
def test_a_real_instruction_still_picks_its_subject(prompt, expected):
    assert expected in generate_offline_fallback_response(prompt)


def test_the_offline_reply_never_promises_work_it_will_not_do():
    for prompt in ("search for x", "debug this", "why", ""):
        reply = generate_offline_fallback_response(prompt).lower()
        assert reply
        for promise in ("let me think", "i'm searching", "i am analyzing", "i'll get back"):
            assert promise not in reply
