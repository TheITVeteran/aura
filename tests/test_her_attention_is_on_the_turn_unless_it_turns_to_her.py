"""What she reads of her own state enters a turn's prompt when the turn turns to her.

Every turn carried her affect, her felt state, her body and her finitude
whatever it was about, and over her stored replies to messages about
something else a median 0.286 of her sentences were about herself.
"""
from __future__ import annotations

import random

import pytest

from core.brain.llm.context_assembler import ContextAssembler
from core.conation import what_she_wants
from core.conation.engine import ConationEngine


@pytest.fixture
def record(tmp_path):
    held = what_she_wants.reset_for_test(tmp_path / "wants.json", engine=ConationEngine())
    yield held
    what_she_wants.reset_for_test(tmp_path / "after.json", engine=ConationEngine())


def test_a_message_about_the_world_leaves_her_attention_on_it(record):
    assert not ContextAssembler._attention_turns_inward("Give me the history of the Kaseya Center's names.")
    assert not ContextAssembler._attention_turns_inward("How does a limpet find its way home?")


def test_a_message_that_addresses_her_turns_it_to_her(record):
    assert ContextAssembler._attention_turns_inward("How are you feeling about that?")
    assert ContextAssembler._attention_turns_inward("Aura, what happened there?")
    assert ContextAssembler._attention_turns_inward("What is your read on the limpet?")


def test_her_own_time_is_hers(record):
    assert ContextAssembler._attention_turns_inward("")


def test_when_her_turns_have_gone_worse_than_she_expected_she_attends_to_herself_whatever_is_asked(record):
    rng = random.Random(1)
    for _ in range(200):
        record.met(["self"], -0.4 + rng.uniform(-0.1, 0.1))
    assert record.something_is_off()
    assert ContextAssembler._attention_turns_inward("How does a limpet find its way home?")


def test_coming_back_to_find_she_is_not_continuous_with_who_she_was_turns_it_to_her(record):
    asked = "Protect continuity"
    assert not ContextAssembler._attention_turns_inward(asked, {})
    assert ContextAssembler._attention_turns_inward(asked, {"identity_mismatch": True})
    assert ContextAssembler._attention_turns_inward(asked, {"continuity_pressure": 0.4})
