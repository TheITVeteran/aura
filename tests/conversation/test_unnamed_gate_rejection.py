"""A gate that cannot name a violation has not proven one.

LIVE, 2026-08-10. A turn died with this failure class:

    reply_reliability_gate_failed:

The separator, and nothing after it. assessment.reasons was empty, so the
reliability gate rejected a complete reply without naming a single violation,
the turn was recorded as exhausted, and the person got "I couldn't get to an
answer I'd stand behind on that one."

The runtime already refuses the mirror image of this — absence of a check must
never be reported as a passed check, which is written into five subsystems. The
same rule holds pointing the other way: absence of a finding must not be
reported as a failure. A reply in hand beats a canned refusal justified by
nothing.

The gate keeps every power it had. It simply has to say what is wrong.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from interface.routes.chat import _named_gate_failure, _reply_gate_proved_a_violation


@pytest.mark.parametrize(
    "reasons",
    [(), [], None, ("",), ("   ",), ["", "  "]],
)
def test_an_empty_or_blank_reason_list_proves_nothing(reasons) -> None:
    assessment = SimpleNamespace(reasons=reasons)

    assert _reply_gate_proved_a_violation(assessment) is False


@pytest.mark.parametrize(
    "reasons",
    [
        ("truncated_tail",),
        ("fabricated_shared_history", "reply_abandons_thread"),
        ["runtime_boilerplate"],
    ],
)
def test_a_named_violation_still_rejects(reasons) -> None:
    """The gate keeps every power it had."""
    assessment = SimpleNamespace(reasons=reasons)

    assert _reply_gate_proved_a_violation(assessment) is True


def test_an_assessment_without_a_reasons_attribute_proves_nothing() -> None:
    assert _reply_gate_proved_a_violation(object()) is False


def test_the_failure_class_carries_its_reasons() -> None:
    assessment = SimpleNamespace(reasons=("truncated_tail", "runtime_boilerplate"))

    assert _named_gate_failure(assessment) == (
        "reply_reliability_gate_failed:truncated_tail,runtime_boilerplate"
    )


def test_an_unnamed_failure_class_says_so_instead_of_trailing_a_colon() -> None:
    """"reply_reliability_gate_failed:" told an operator nothing at all."""
    assessment = SimpleNamespace(reasons=())

    assert _named_gate_failure(assessment) == (
        "reply_reliability_gate_failed:unnamed_violation"
    )


def test_the_rejection_path_serves_the_reply_when_nothing_is_named() -> None:
    import inspect

    from interface.routes import chat

    source = inspect.getsource(chat)
    marker = "_record_exhausted_cognitive_failure(\n                    _named_gate_failure(assessment),"
    assert marker in source
    guarded = source[: source.find(marker)]
    assert "_reply_gate_proved_a_violation(assessment)" in guarded
