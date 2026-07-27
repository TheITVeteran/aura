"""A composite's effect scope belongs to what it does, not what it could do.

``desktop_task`` can write files, drive AppleScript and read the screen, so its
declared scope is the widest thing any of its steps might do. Governing every
invocation at that width blocks a screen-reading objective as though it were a
filesystem write — governance refusing work because the scope was computed at
the wrong layer, which is the same shape as the step that was asked to prove
``steps_completed`` because it inherited the task's contract.

The honest scope of a plan is the widest scope among the steps it actually
contains, computed from the steps, at the layer that has them.

The fallback direction matters more than the narrowing. An unrecognised step
could do anything, so the plan does not get narrowed at all — it keeps the
skill's declared scope. Over-blocking is a nuisance; under-blocking is a
governance hole, and a resolver that guesses narrow on unfamiliar input is a
resolver that can be widened by inventing a step name.
"""
from __future__ import annotations

import pytest

from core.executive.execution_policy import resolve_execution_effect_scope


def _scope(steps) -> str:
    return resolve_execution_effect_scope("desktop_task", {"steps": steps})


def test_a_reading_plan_is_read_only() -> None:
    assert _scope([{"action": "read_screen_text"}, {"action": "inspect_screen"}]) == "read_only"


def test_one_click_makes_it_desktop_control() -> None:
    assert (
        _scope([{"action": "read_screen_text"}, {"action": "click"}])
        == "foreground_desktop_control"
    )


def test_one_write_makes_it_file_io() -> None:
    """The widest step governs: a plan is as consequential as its worst step."""
    assert _scope([{"action": "click"}, {"action": "write_text_file"}]) == "desktop_file_io"


def test_order_does_not_change_the_scope() -> None:
    assert _scope([{"action": "write_text_file"}, {"action": "click"}]) == _scope(
        [{"action": "click"}, {"action": "write_text_file"}]
    )


@pytest.mark.parametrize(
    "steps",
    [
        [{"action": "teleport"}],
        [{"action": "read_screen_text"}, {"action": "teleport"}],
        [{"action": ""}],
        "not-a-list",
        [],
        None,
    ],
)
def test_anything_unrecognised_keeps_the_declared_scope(steps) -> None:
    """Never narrow on input the resolver does not understand.

    A resolver that guesses narrow can be widened by inventing a step name.
    """
    assert _scope(steps) == "foreground_desktop_control"


def test_a_json_encoded_plan_is_still_read() -> None:
    import json

    assert _scope(json.dumps([{"action": "read_screen_text"}])) == "read_only"


def test_malformed_json_keeps_the_declared_scope() -> None:
    assert _scope("{not json") == "foreground_desktop_control"


def test_other_skills_are_untouched() -> None:
    """The change is per-invocation for one composite, not a policy rewrite."""
    assert resolve_execution_effect_scope("web_search", {"query": "x"}) == "read_only"
    assert (
        resolve_execution_effect_scope("computer_use", {"action": "read_screen_text"})
        == "read_only"
    )
    assert (
        resolve_execution_effect_scope("computer_use", {"action": "write_text_file"})
        == "desktop_file_io"
    )
