"""A clipboard objective had no way to be verified, so it was always refused.

LIVE, 2026-08-10:

    "Put the text ORION-7 on my clipboard, then tell me what you put there."
    "os_automation failed: OS automation refused to act because the objective
     has no complete observable acceptance contract. Completed 0/1 steps."

An EffectContract is verifiable only when it carries a strong required check.
There was no clipboard EffectKind at all, so a clipboard goal could never carry
one — unverifiable by construction, refused every time, for an effect the
runtime performs and the observation snapshot already reads back.

The acceptance criterion is exact and the objective states it: the text the
person named is on the clipboard afterwards.
"""

from __future__ import annotations

import pytest

from core.runtime.os_automation_effects import EffectKind, build_effect_contract


@pytest.mark.parametrize(
    ("goal", "expected_payload"),
    [
        ("Put the text ORION-7 on my clipboard, then tell me what you put there.", "ORION-7"),
        ('copy "hello there" to my clipboard', "hello there"),
        ("put the value BUILD-42 on the pasteboard", "BUILD-42"),
    ],
)
def test_a_clipboard_goal_is_verifiable(goal: str, expected_payload: str) -> None:
    contract = build_effect_contract(goal)

    assert contract.verifiable is True
    clipboard = [
        requirement
        for requirement in contract.requirements
        if requirement.kind is EffectKind.CLIPBOARD_CONTAINS
    ]
    assert clipboard, "no clipboard acceptance criterion"
    assert clipboard[0].expected == expected_payload
    assert clipboard[0].required and clipboard[0].strong


def test_copy_to_the_clipboard_is_not_a_filesystem_mutation() -> None:
    """"copy" alone marked the objective unsupported — "filesystem mutation
    lacks source, destination, and artifact verification" — for a request that
    moves no file and has an exact postcondition."""
    contract = build_effect_contract('copy "hello there" to my clipboard')

    assert contract.unsupported_reasons == ()


def test_a_real_file_copy_still_needs_its_artifact() -> None:
    """The exclusion is for the clipboard destination only."""
    contract = build_effect_contract("move the file to Documents")

    assert contract.verifiable is False
    assert any("filesystem mutation" in reason for reason in contract.unsupported_reasons)


def test_a_goal_with_no_clipboard_gets_no_clipboard_criterion() -> None:
    contract = build_effect_contract("open Notes and write a paragraph")

    assert not any(
        requirement.kind is EffectKind.CLIPBOARD_CONTAINS
        for requirement in contract.requirements
    )


def test_a_clipboard_goal_naming_no_text_is_not_invented() -> None:
    """Nothing to check means nothing to claim — it must not pass vacuously."""
    from core.runtime.os_automation_effects import _clipboard_payload

    assert _clipboard_payload("clear my clipboard", "") == ""


def test_the_check_reads_the_clipboard_back() -> None:
    import inspect

    from core.runtime import os_automation_effects

    source = inspect.getsource(os_automation_effects._evaluate_requirement)

    assert "EffectKind.CLIPBOARD_CONTAINS" in source
    assert "after.clipboard_excerpt" in source
