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


# ── A one-line script does not need a 32B model ────────────────────────────

def test_the_clipboard_script_is_deterministic() -> None:
    """LIVE, 2026-08-10, once the contract was accepted:

        "os_automation failed: Skill error: TimeoutError"  (55,302ms)

    The AppleScript compiler asks the resident 32B to WRITE
    `set the clipboard to "ORION-7"` under a 35-second budget. A one-line
    script whose only variable is a string the person typed does not need a
    model, and every second spent generating it is latency and a chance to
    fail.
    """
    from core.skills.os_automation import OSAutomationCompilerSkill

    script = OSAutomationCompilerSkill._deterministic_script_for_goal(
        "Put the text ORION-7 on my clipboard, then tell me what you put there.", {}
    )

    assert 'set the clipboard to "ORION-7"' in script


def test_the_deterministic_script_satisfies_the_contract() -> None:
    """If it did not cover the contract the compiler would call the model anyway."""
    from core.skills.os_automation import OSAutomationCompilerSkill

    goal = "Put the text ORION-7 on my clipboard, then tell me what you put there."
    script = OSAutomationCompilerSkill._deterministic_script_for_goal(goal, {})
    covered, _reason = OSAutomationCompilerSkill._deterministic_script_covers_contract(
        goal=goal,
        context={},
        contract=build_effect_contract(goal),
        script=script,
    )

    assert covered is True


def test_the_script_and_the_verifier_share_one_payload_definition() -> None:
    """Two definitions of "what text" would be one more place to disagree."""
    import inspect

    from core.skills.os_automation import OSAutomationCompilerSkill

    source = inspect.getsource(OSAutomationCompilerSkill._clipboard_payload_for_goal)

    assert "_clipboard_payload" in source


def test_a_goal_without_a_clipboard_gets_no_clipboard_line() -> None:
    from core.skills.os_automation import OSAutomationCompilerSkill

    script = OSAutomationCompilerSkill._deterministic_script_for_goal("open Notes", {})

    assert "clipboard" not in script


def test_the_safety_guard_accepts_the_clipboard_criterion() -> None:
    """LIVE, 2026-08-10, with the contract and script both correct:

        "Script blocked by safety guard: clipboard write is not represented by
         the effect contract"

    The guard predates CLIPBOARD_CONTAINS, so the only criterion that could
    justify a clipboard write was TEXT_VISIBLE — a check about what is on
    SCREEN. The contract carried a clipboard_contains requirement for that
    exact text and the guard had no way to recognise it.
    """
    from core.skills.os_automation import OSAutomationCompilerSkill

    goal = "Put the text ORION-7 on my clipboard, then tell me what you put there."
    script = OSAutomationCompilerSkill._deterministic_script_for_goal(goal, {})

    allowed, reason = OSAutomationCompilerSkill._validate_script_scope(
        script, build_effect_contract(goal)
    )

    assert allowed is True, reason


def test_a_clipboard_write_with_no_clipboard_criterion_is_still_blocked() -> None:
    """The guard keeps guarding: a script may not do what the contract never
    promised."""
    from core.skills.os_automation import OSAutomationCompilerSkill

    unrelated = build_effect_contract("open Notes")
    allowed, reason = OSAutomationCompilerSkill._validate_script_scope(
        'set the clipboard to "SNEAKY"', unrelated
    )

    assert allowed is False
    assert "clipboard write" in reason


def test_a_bare_identifier_payload_is_recognised() -> None:
    """"put ORION-7 on my clipboard" names its payload without quoting it.

    Requiring quotes or the literal word "text" produced NO acceptance
    criterion, so the contract had nothing to verify and the snapshot never
    read the clipboard back.
    """
    from core.runtime.os_automation_effects import _clipboard_payload

    assert _clipboard_payload("put ORION-7 on my clipboard", "") == "ORION-7"


def test_the_snapshot_reads_the_clipboard_when_the_contract_asks() -> None:
    """LIVE, 2026-08-10: the script SET the clipboard, the text was really
    there, and the check failed — nothing captured clipboard_excerpt, so the
    verifier compared its expectation against an empty string and went looking
    for a repair.

    An effect that happened and cannot be observed is indistinguishable from
    one that did not.
    """
    assert build_effect_contract("put ORION-7 on my clipboard").needs_clipboard is True
    assert build_effect_contract("open Notes").needs_clipboard is False


def test_the_capture_is_wired_into_the_snapshot() -> None:
    import inspect

    from core.skills import os_automation

    source = inspect.getsource(os_automation)

    assert "contract.needs_clipboard" in source
    assert 'values["clipboard_excerpt"]' in source
