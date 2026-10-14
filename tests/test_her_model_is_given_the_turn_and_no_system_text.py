"""Her model is given what was said and what her instruments returned, and no system text.

Bryan, 13 October 2026: no internal system prompts at all. The mechanism is
core/brain/what_she_is_given.py, applied at the last point before her model.
"""
from __future__ import annotations

from core.brain.llm.chat_format import format_chatml_messages
from core.brain.what_she_is_given import (
    STILL_WORDED,
    for_her_model,
    her_turn_is_still_worded,
    readings_in,
    what_she_is_given,
)
from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE, is_stamped_grounding, stamp_grounding

MARKERS = ("## PRESENT MOMENT", "## REFERENCE PASSAGES FROM THE LOCAL CORPUS")
TOLD = (
    "You are Aura Luna. You have strong opinions and you share them freely.\n"
    "## LIVE TONE\nMood: empathy\nTone: inquisitive_engaged\n\n"
    "## HOLDING A VIEW\nYou may hold and state views.\n\n"
    "## PRESENT MOMENT\n- It is Tuesday 13 October 2026, 15:06 local.\n\n"
    "## SOMATIC STATE\nSomatically comfortable and engaged."
)


def _turn():
    return [
        {"role": "system", "content": "You are Aura speaking through the live desktop CognitiveEngine."},
        {"role": "system", "content": TOLD},
        {"role": "user", "content": "an earlier thing somebody said"},
        {"role": "assistant", "content": "what she answered"},
        {"role": "user", "content": "Why do limpets come back to the same scar?"},
    ]


def test_nothing_the_runtime_wrote_to_tell_her_who_to_be_is_dispatched():
    messages, system_prompt = what_she_is_given(_turn(), "You are Aura. Be witty.", markers=MARKERS)
    assert system_prompt == ""
    assert all(message["role"] != "system" for message in messages)
    sent = " ".join(str(message["content"]) for message in messages)
    for told in ("You are Aura", "strong opinions", "LIVE TONE", "HOLDING A VIEW", "SOMATIC STATE", "Be witty"):
        assert told not in sent


def test_what_was_said_is_untouched_and_in_order():
    messages, _ = what_she_is_given(_turn(), "", markers=MARKERS)
    said = [m["content"] for m in messages if m["role"] in {"user", "assistant"}]
    assert said == ["an earlier thing somebody said", "what she answered", "Why do limpets come back to the same scar?"]


def test_a_reading_travels_as_a_reading_just_before_what_was_last_said():
    messages, _ = what_she_is_given(_turn(), "", markers=MARKERS)
    evidence = [m for m in messages if m["role"] == RUNTIME_EVIDENCE_ROLE]
    assert len(evidence) == 1 and is_stamped_grounding(evidence[0])
    assert evidence[0]["content"] == "## PRESENT MOMENT\n- It is Tuesday 13 October 2026, 15:06 local."
    assert messages.index(evidence[0]) == len(messages) - 2


def test_a_message_the_runtime_stamped_as_evidence_is_a_reading_whole():
    turn = _turn()
    turn.insert(2, stamp_grounding({"role": "system", "content": "[TOOL RESULT: clock] 15:06"}))
    messages, _ = what_she_is_given(turn, "", markers=MARKERS)
    evidence = next(m for m in messages if m["role"] == RUNTIME_EVIDENCE_ROLE)
    assert "[TOOL RESULT: clock] 15:06" in evidence["content"]


def test_a_turn_with_no_reading_is_only_what_was_said():
    turn = [m for m in _turn() if m["content"] != TOLD]
    messages, _ = what_she_is_given(turn, "", markers=MARKERS)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]


def test_only_registered_blocks_count_as_readings():
    assert readings_in(TOLD, MARKERS) == ["## PRESENT MOMENT\n- It is Tuesday 13 October 2026, 15:06 local."]
    assert readings_in(TOLD, ()) == []


def test_a_lane_that_still_words_its_contract_is_named_and_left_as_it_was(monkeypatch):
    assert not her_turn_is_still_worded({})
    for flag in STILL_WORDED:
        assert her_turn_is_still_worded({flag: True})
    monkeypatch.setenv("AURA_SYSTEM_TEXT", "keep")
    assert her_turn_is_still_worded({})


def test_the_formatter_adds_no_identity_of_its_own():
    rendered = format_chatml_messages([{"role": "user", "content": "Why do limpets come back?"}])
    assert rendered == "<|im_start|>user\nWhy do limpets come back?<|im_end|>\n<|im_start|>assistant\n"
    assert "system" not in rendered and "You are" not in rendered


def test_a_stamp_on_a_message_does_not_make_the_instructions_in_it_readings():
    """The runtime stamps the whole living-context message because the present moment is in it."""
    turn = [
        stamp_grounding({"role": "system", "content": TOLD}),
        {"role": "user", "content": "Why do limpets come back to the same scar?"},
    ]
    messages, _ = what_she_is_given(turn, "", markers=MARKERS)
    evidence = next(m for m in messages if m["role"] == RUNTIME_EVIDENCE_ROLE)
    assert evidence["content"] == "## PRESENT MOMENT\n- It is Tuesday 13 October 2026, 15:06 local."


def test_her_own_state_travels_as_evidence_when_the_turn_turns_to_her_and_only_then():
    live = TOLD + "\n[LIVE MIND CONTEXT]\n{\"valence\": 0.2}\n[END LIVE MIND CONTEXT]\n"

    def given(said: str) -> str:
        turn = [{"role": "system", "content": live}, {"role": "user", "content": said}]
        messages, _ = for_her_model(turn, "", {}, True, (), said)
        assert all(message["role"] != "system" for message in messages)
        return " ".join(str(m["content"]) for m in messages if m["role"] == RUNTIME_EVIDENCE_ROLE)

    about_the_world = given("Why do limpets come back to the same scar?")
    assert "## LIVE TONE" not in about_the_world and "[LIVE MIND CONTEXT]" not in about_the_world
    about_her = given("How are you feeling right now?")
    assert "## LIVE TONE" in about_her and "## SOMATIC STATE" in about_her
    assert "[LIVE MIND CONTEXT]" in about_her
    # Asked about herself or not, she is not told who to be or what she may do.
    for evidence in (about_the_world, about_her):
        assert "You are Aura" not in evidence and "HOLDING A VIEW" not in evidence


def test_a_turn_nobody_sent_or_a_lane_still_worded_is_left_as_it_was():
    turn = [{"role": "system", "content": TOLD}, {"role": "user", "content": "plan this"}]
    assert for_her_model(turn, "sys", {}, False, ()) == (turn, "sys")
    assert for_her_model(turn, "sys", {"output_shape": "json"}, True, ()) == (turn, "sys")
