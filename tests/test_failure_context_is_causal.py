"""The failure facts actually reach the prompt.

This repository has a recurring defect class worth naming: a mechanism that
is built, tested in isolation, and never wired — reporting itself present
while the thing it was supposed to change goes on behaving exactly as before.
The clause-streaming carve-out was one (a full test suite, zero production
callers). Recording capability failures as narratable facts is a prime
candidate for the same fate, because everything about it looks correct from
inside its own unit tests while the model never sees a word of it.

So this test does not check that a failure was recorded. It checks that the
readings appear in the messages the model is actually handed, and that a turn
with nothing wrong is not polluted with an empty block.
"""
from __future__ import annotations

import pytest

from core.conversation.failure_context import (
    bind_failure_ledger,
    record_capability_failure,
)


@pytest.fixture()
def assembler():
    from core.brain.llm.context_assembler import ContextAssembler

    return ContextAssembler


@pytest.fixture()
def state():
    from core.state.aura_state import AuraState

    return AuraState.default()


def _system_text(messages) -> str:
    return "\n".join(
        str(m.get("content", "")) for m in messages if m.get("role") == "system"
    )


def test_recorded_failures_reach_the_system_prompt(assembler, state) -> None:
    """The whole point: she cannot explain what she was never told."""
    with bind_failure_ledger():
        record_capability_failure(
            "web_search",
            intent="look up tonight's train times",
            cause="offline",
            detail="probe to 1.1.1.1:53 has failed for 240s",
            still_possible=("everything already in memory",),
        )
        messages = assembler.build_messages(state, "when is the last train?")

    system = _system_text(messages)
    assert "probe to 1.1.1.1:53 has failed for 240s" in system
    assert "look up tonight's train times" in system
    assert "everything already in memory" in system


def test_a_clean_turn_carries_no_failure_block(assembler, state) -> None:
    """An empty block every turn is noise that trains her to ignore the real one."""
    with bind_failure_ledger():
        messages = assembler.build_messages(state, "hello")
    assert "as facts rather than phrasing" not in _system_text(messages)


def test_no_ledger_bound_is_not_an_error(assembler, state) -> None:
    """Background work and tests build prompts with nothing collecting."""
    messages = assembler.build_messages(state, "hello")
    assert messages
    assert "as facts rather than phrasing" not in _system_text(messages)


def test_the_block_survives_a_long_prompt(assembler, state) -> None:
    """It is appended last so middle-out truncation cannot eat it.

    A failure she is not told about is one she will paper over, which is
    strictly worse than a slightly longer prompt.
    """
    with bind_failure_ledger():
        record_capability_failure(
            "media_playback",
            intent="play Kind of Blue",
            cause="offline",
            detail="no network path",
        )
        messages = assembler.build_messages(
            state, "explain " + ("something at length " * 400), max_tokens=8192
        )
    assert "no network path" in _system_text(messages)


def test_failures_from_a_previous_turn_do_not_reappear(assembler, state) -> None:
    with bind_failure_ledger():
        record_capability_failure("web_search", intent="search", cause="offline")
    with bind_failure_ledger():
        messages = assembler.build_messages(state, "and now something else")
    assert "as facts rather than phrasing" not in _system_text(messages)
