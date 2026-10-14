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


# The last hop. The gate's removal was live on 13 October and her model still
# read 577 characters of system text: the client wrote "You are Aura Luna..."
# in front of every foreground turn after the gate had finished.

def _client_that_records(sent: dict):
    from core.brain.llm.mlx_client import MLXLocalClient

    client = object.__new__(MLXLocalClient)

    async def generate(prompt, **kwargs):
        sent["prompt"], sent["messages"] = prompt, kwargs.get("messages")
        return "ok"

    client.generate = generate
    return client


def test_the_client_hands_the_worker_the_messages_it_was_handed():
    import asyncio

    sent: dict = {}
    client = _client_that_records(sent)
    turn = [
        {"role": "user", "content": "an earlier thing somebody said"},
        {"role": "assistant", "content": "what she answered"},
        {"role": "user", "content": "Why do limpets come back to the same scar?"},
    ]
    asyncio.run(
        client.generate_text_async(
            "", messages=[dict(m) for m in turn], system_prompt="", foreground_request=True
        )
    )
    assert sent["messages"] == turn
    assert "<|im_start|>system" not in sent["prompt"]
    assert "You are" not in sent["prompt"]


def test_from_the_gate_to_the_worker_a_turn_of_hers_carries_no_system_message():
    import asyncio

    sent: dict = {}
    client = _client_that_records(sent)
    said = "Why do limpets come back to the same scar?"
    messages, system_prompt = for_her_model(_turn(), "You are Aura. Be witty.", {}, True, (), said)
    asyncio.run(
        client.generate_text_async(
            "", messages=messages, system_prompt=system_prompt, foreground_request=True
        )
    )
    roles = [message["role"] for message in sent["messages"]]
    assert "system" not in roles
    assert roles[-1] == "user" and sent["messages"][-1]["content"] == said
    assert "<|im_start|>system" not in sent["prompt"]


def test_readings_are_named_by_heading_with_their_size():
    from core.brain.what_she_is_given import readings_by_size

    messages, _ = what_she_is_given(_turn(), "", markers=MARKERS)
    reading = next(m["content"] for m in messages if m["role"] == RUNTIME_EVIDENCE_ROLE)
    # One newline is counted after each of its lines.
    assert readings_by_size(messages) == [("PRESENT MOMENT", len(reading) + 1)]
    assert readings_by_size([{"role": "user", "content": "## PRESENT MOMENT\nnot a reading"}]) == []


def test_a_measurement_turn_keeps_its_readings(monkeypatch):
    from core.brain import aura_persona

    monkeypatch.setattr(aura_persona, "this_turn_is_a_measurement", lambda: True)
    monkeypatch.setattr(aura_persona, "left_out", lambda: frozenset())
    messages, _ = what_she_is_given(_turn(), "", markers=MARKERS)
    kept = aura_persona.as_measured(messages)
    assert [m["role"] for m in kept] == [RUNTIME_EVIDENCE_ROLE, "user"]


HER_MODEL = __import__("pathlib").Path.home() / ".aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15"


def test_her_own_chat_template_writes_no_system_block_for_such_a_turn():
    """Her template adds a sentence about reasoning effort unless told `medium`."""
    import pytest

    template = HER_MODEL / "chat_template.jinja"
    if not template.exists():
        pytest.skip("her model is not on this host")
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    from core.brain.llm.chat_format import reasoning_effort_for_generation

    def raise_exception(message):
        raise ValueError(message)

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.globals["raise_exception"] = raise_exception
    turn = [
        {"role": "user", "content": "an earlier thing somebody said"},
        {"role": "assistant", "content": "what she answered"},
        {"role": "tool", "content": "## PRESENT MOMENT\n- Clock: 15:06"},
        {"role": "user", "content": "Why do limpets come back to the same scar?"},
    ]
    render = environment.from_string(template.read_text()).render
    open_channel = render(
        messages=turn,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort=reasoning_effort_for_generation(thinking=True),
    )
    closed_channel = render(messages=turn, add_generation_prompt=True, enable_thinking=False)
    for rendered in (open_channel, closed_channel):
        assert "<|im_start|>system" not in rendered
        assert "<tool_response>\n## PRESENT MOMENT" in rendered
    # The default the worker must never fall back to.
    assert "<|im_start|>system" in render(messages=turn, add_generation_prompt=True)
