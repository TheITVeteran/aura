"""Every multi-step desktop objective inherited a contract its parts cannot meet.

Measured live 2026-07-27, on "create a file on my Desktop called
aura_hello.txt", once the cause was finally being reported:

    create_folder failed: expectation incomplete: steps_requested;
    steps_completed (expected: Folder exists.)

``steps_requested`` and ``steps_completed`` are fields of the *task* result.
No individual step produces them, and none ever could. They arrived because
each step's context was built with ``dict(task_context)``, which carries the
task-level action expectation down into every child call — so the contract
layer asked a folder step to prove how many steps the task had run, it could
not, and the objective died on a step that had actually worked.

This was never one action misbehaving. Any desktop objective whose caller
declared an expectation handed that expectation to every step inside it.

A step proves the step's effect; the task proves the task's.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.skills.desktop_task import DesktopTaskSkill

SOURCE = Path("core/skills/desktop_task.py")


@pytest.mark.parametrize(
    "key",
    [
        "action_expectation",
        "expectation",
        "acceptance_criteria",
        "criteria",
        "required_evidence",
        "evidence_required",
        "required_evidence_present",
        "user_visible_effect",
        "visible_effect",
        "repair_hint",
        "rollback_hint",
        "allow_partial",
    ],
)
def test_no_task_level_contract_key_reaches_a_step(key: str) -> None:
    child = DesktopTaskSkill._child_step_context({key: "task-level", "origin": "desktop_ui"})
    assert key not in child


def test_the_live_failure_cannot_recur() -> None:
    """The exact contract that failed create_folder."""
    child = DesktopTaskSkill._child_step_context(
        {
            "required_evidence": ["steps_requested", "steps_completed"],
            "acceptance_criteria": ["desktop objective complete"],
            "objective": "create a file on my Desktop called aura_hello.txt",
        }
    )
    assert "required_evidence" not in child
    assert "acceptance_criteria" not in child


def test_everything_a_step_actually_needs_survives() -> None:
    """Stripping too much would break authorization and routing."""
    child = DesktopTaskSkill._child_step_context(
        {
            "origin": "desktop_ui",
            "objective": "write the file",
            "user_explicitly_authorized": True,
            "user_requested_action": True,
            "cognitive_engine": object(),
            "required_evidence": ["steps_completed"],
        }
    )
    for key in (
        "origin",
        "objective",
        "user_explicitly_authorized",
        "user_requested_action",
        "cognitive_engine",
    ):
        assert key in child


def test_the_caller_context_is_not_mutated() -> None:
    """The task still needs its own contract after building a step's."""
    task_context = {"required_evidence": ["steps_completed"], "origin": "desktop_ui"}
    DesktopTaskSkill._child_step_context(task_context)
    assert task_context["required_evidence"] == ["steps_completed"]


def test_a_missing_context_is_not_an_error() -> None:
    assert DesktopTaskSkill._child_step_context(None) == {}


def test_every_child_context_goes_through_the_helper() -> None:
    """A single `dict(task_context)` left anywhere reopens the whole defect."""
    src = SOURCE.read_text(encoding="utf-8")
    assert "step_context = dict(task_context)" not in src
    assert "step_context = dict(context or {})" not in src
    assert src.count("_child_step_context(") >= 4
