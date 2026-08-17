"""The viability ledger capped the budget, and one lane simply ignored it.

`resource_stakes.action_envelope()` is the runtime saying how much output it
can still afford. A denied envelope means the machine is out of resources and
caps `max_tokens` at 128. The protected capability lane — a capability-
inventory turn on the desktop foreground — skipped that cap ENTIRELY, on both
the denied and the allowed branch, with no ceiling of any kind behind it.

The lane exists for a real reason: an inventory answer truncated to 128 tokens
is useless, and that lane is what the person is looking at. But "this answer
needs room" is a bounded claim, and what was there was unbounded — an
exemption from viability control, on exactly the branch where the ledger has
already said no.

The bound comes from the request. `_foreground_compute_profile` already
computes a token floor for this prompt's shape; that floor is what the lane
needs to finish a sentence, and it is now the ceiling of the override.
"""
from __future__ import annotations

from core.brain.inference_gate import _STAKES_DENIED_TOKEN_CAP, InferenceGate


def test_an_unprotected_turn_is_capped_by_the_envelope():
    context: dict = {}

    tokens, ceiling = InferenceGate._stakes_capped_tokens(
        4096,
        envelope_cap=_STAKES_DENIED_TOKEN_CAP,
        protected=False,
        prompt="hello",
        context=context,
        reason="envelope_denied",
    )

    assert tokens == _STAKES_DENIED_TOKEN_CAP
    assert ceiling == _STAKES_DENIED_TOKEN_CAP
    assert "resource_stakes_protected_override" not in context


def test_a_protected_turn_still_gets_a_ceiling():
    """It used to get none at all."""
    context: dict = {}

    tokens, ceiling = InferenceGate._stakes_capped_tokens(
        100_000,
        envelope_cap=_STAKES_DENIED_TOKEN_CAP,
        protected=True,
        prompt="what can you actually do right now?",
        context=context,
        reason="envelope_denied",
    )

    assert tokens == ceiling
    assert ceiling < 100_000, "the protected lane is still unbounded"
    assert ceiling >= _STAKES_DENIED_TOKEN_CAP


def test_the_protected_ceiling_comes_from_this_turn_s_own_profile():
    prompt = "what can you actually do right now?"
    floor, _cap, _loops = InferenceGate._foreground_compute_profile(prompt)
    context: dict = {}

    _tokens, ceiling = InferenceGate._stakes_capped_tokens(
        100_000,
        envelope_cap=_STAKES_DENIED_TOKEN_CAP,
        protected=True,
        prompt=prompt,
        context=context,
        reason="envelope_denied",
    )

    assert ceiling == max(_STAKES_DENIED_TOKEN_CAP, floor)


def test_structural_completion_floor_can_raise_bounded_surface_ceiling():
    context: dict = {}

    tokens, ceiling = InferenceGate._stakes_capped_tokens(
        2560,
        envelope_cap=1536,
        protected=True,
        completion_floor=2560,
        prompt="Explain a five-part algorithm in one complete response.",
        context=context,
        reason="envelope_allowed",
    )

    assert tokens == 2560
    assert ceiling == 2560
    assert context["resource_stakes_protected_override"]["derived_from"] == (
        "user_surface_completion_floor"
    )


def test_the_raise_is_recorded_where_the_turn_can_be_audited():
    context: dict = {}

    InferenceGate._stakes_capped_tokens(
        100_000,
        envelope_cap=_STAKES_DENIED_TOKEN_CAP,
        protected=True,
        prompt="what can you actually do right now?",
        context=context,
        reason="envelope_denied",
    )

    receipt = context["resource_stakes_protected_override"]
    assert receipt["reason"] == "envelope_denied"
    assert receipt["envelope_ceiling"] == _STAKES_DENIED_TOKEN_CAP
    assert receipt["override_ceiling"] > receipt["envelope_ceiling"]
    assert receipt["derived_from"] == "foreground_compute_profile_floor"


def test_no_override_is_recorded_when_the_envelope_is_already_generous():
    """A receipt for a raise that did not happen would be noise in the audit."""
    context: dict = {}

    _tokens, ceiling = InferenceGate._stakes_capped_tokens(
        100_000,
        envelope_cap=100_000,
        protected=True,
        prompt="hello",
        context=context,
        reason="envelope_allowed",
    )

    assert ceiling == 100_000
    assert "resource_stakes_protected_override" not in context


def test_a_caller_asking_for_less_than_the_ceiling_keeps_its_request():
    """The ceiling is a ceiling, not a floor — it must never RAISE a budget."""
    context: dict = {}

    tokens, _ceiling = InferenceGate._stakes_capped_tokens(
        64,
        envelope_cap=_STAKES_DENIED_TOKEN_CAP,
        protected=True,
        prompt="what can you actually do right now?",
        context=context,
        reason="envelope_denied",
    )

    assert tokens == 64


def test_both_envelope_branches_go_through_the_same_ceiling():
    """The bypass was duplicated on the denied and the allowed branch, which is
    how a fix to one of them would have missed the other."""
    import ast
    import inspect

    from core.brain import inference_gate as gate_mod

    source = inspect.getsource(gate_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        if "envelope.allowed" not in rendered or "resource_stakes_blocked" not in rendered:
            continue
        assert rendered.count("_stakes_capped_tokens") == 2, (
            "one of the two envelope branches does not go through the ceiling"
        )
        assert "protected_compact_capability_contract:" not in rendered, (
            "a branch still skips token limiting outright for the protected lane"
        )
        return
    raise AssertionError("the resource-stakes envelope branch was not found")
