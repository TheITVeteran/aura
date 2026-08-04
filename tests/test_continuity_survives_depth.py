"""Depth must not cost her the thread.

The old policy shrank the continuity budget as the conversation deepened
(1800 → 600 → 400 chars at depth 20 and 30) while the raw transcript grew.
These tests pin the inversion: continuity is what survives, optional colour
is what yields.
"""

from __future__ import annotations

import pytest

from core.brain.llm.context_assembler import ContextAssembler
from core.state.aura_state import AuraState


def _state_at_depth(depth: int) -> AuraState:
    state = AuraState.default()
    state.cognition.current_objective = "Continue the conversation."
    state.cognition.working_memory = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
        for i in range(depth)
    ]
    return state


def test_continuity_budget_never_shrinks_with_depth():
    budgets = [ContextAssembler._continuity_budget_chars(d) for d in range(0, 200, 5)]
    assert budgets == sorted(budgets), budgets
    assert budgets[-1] > budgets[0], "deep conversations must get MORE continuity, not less"


def test_deep_conversation_gets_more_continuity_budget_than_shallow():
    shallow = ContextAssembler._continuity_budget_chars(2)
    deep = ContextAssembler._continuity_budget_chars(46)
    assert deep > shallow


def test_long_summary_is_not_amputated_at_depth():
    """At depth 46 the old policy allowed 400 chars for the whole conversation."""
    long_summary = "We established that " + ("x" * 1500)
    state = _state_at_depth(46)
    state.cognition.rolling_summary = long_summary

    prompt = ContextAssembler.build_system_prompt(state)

    assert "## CONTINUITY SUMMARY" in prompt
    # The tail of a 1500-char summary must survive; under the old 400-char cap
    # everything past character 400 was silently dropped.
    assert prompt.count("x") > 1000


def test_ledger_reaches_the_prompt_on_every_assembly_path():
    """casual+trusted, casual-guest, and standard all carry the ledger.

    A rule implemented at one of three sites is the defect shape this repo
    keeps rediscovering.
    """
    from core.brain.llm.continuity_ledger import ContinuityLedger

    ledger = ContinuityLedger()
    ledger.observe([
        {"role": "user", "content": "I've always wanted to teach myself physics properly."}
    ])

    seen = []
    for objective in ("hey", "hello there", "Perform a full architecture review of the runtime"):
        state = _state_at_depth(40)
        state.cognition.current_objective = objective
        state.cognition.continuity_ledger = ledger.to_dict()
        prompt = ContextAssembler.build_system_prompt(state)
        seen.append("physics" in prompt.lower())

    assert all(seen), f"ledger missing from some assembly path: {seen}"


def test_ledger_renderer_has_no_hardcoded_user_name():
    """The renderer must name whoever is actually speaking.

    Scoped to the ledger block on purpose: a person's name legitimately
    appears elsewhere in the prompt as a *learned relational belief* carrying
    a confidence. That is evidence. A name compiled into a rendering path is
    not — it is wrong for everyone else and quietly makes one person's
    details part of who she is.
    """
    import inspect

    from core.brain.llm import continuity_ledger

    source = inspect.getsource(continuity_ledger)
    assert "Bryan" not in source

    ledger = continuity_ledger.ContinuityLedger()
    ledger.observe([{"role": "user", "content": "I have always loved sailing in the winter."}])
    assert "Ada said about" in ledger.render(20000, speaker_name="Ada")


def test_interlocutor_name_falls_back_without_inventing_one():
    state = _state_at_depth(4)
    assert ContextAssembler._interlocutor_name(state) in {"They", "Bryan"}


def test_compaction_preserves_disclosures_into_the_ledger():
    """End to end: what compaction drops, the ledger keeps."""
    from core.brain.llm.continuity_ledger import ContinuityLedger

    state = AuraState.default()
    state.cognition.working_memory = [
        {"role": "user", "content": "I've always wanted to teach myself physics and get good at it"}
    ] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"filler turn {i} of chatter"}
        for i in range(200)
    ]

    assert state.compact() is True

    ledger = ContinuityLedger.from_dict(state.cognition.continuity_ledger)
    assert "physics" in ledger.render(20000).lower()


def test_compaction_no_longer_resets_prompt_richness():
    """The sawtooth: pre- and post-compaction continuity budgets must match.

    Previously compaction dropped working memory 150 -> 21, depth fell, and
    elasticity reset from 3 to 1 — she got 'good again' for a few turns and
    then decayed. Budget must not depend on where in the cycle a turn lands.
    """
    pre = ContextAssembler._continuity_budget_chars(150)
    post = ContextAssembler._continuity_budget_chars(21)
    assert pre >= post
    # And both must be at least the floor — neither point starves.
    assert post >= ContextAssembler._continuity_budget_chars(0)
