"""Notes has a scripting interface. Use it rather than typing at a window.

Keystroke automation for note-writing was never going to be reliable: it
needs Notes to hold the front from cmd+n through cmd+v, and on a real
desktop the browser takes focus back mid-sequence. Live 2026-07-28 that
failed repeatedly with "did not become frontmost (observed=Google Chrome)",
and it will keep failing, because during a demo the person is watching
Aura's own UI in a browser.

The Notes dictionary makes it one atomic call — no focus, no clipboard, no
timing — and the executor reads the note back afterwards, so a silent
failure cannot be reported as success. Verified by hand on this machine: the
scripted call created and returned a note in under a second while Chrome
held the front.

This registers and covers the mechanism. Wiring it into the planner is NOT
done here: the existing plan contract deliberately requires watchable
keystroke staging (a launch wait between open_app and the first hotkey), and
five tests assert it. Changing that is a design decision about what a demo
should look like, not a bug fix, so it is left for a deliberate pass rather
than made under time pressure.
"""

import pytest

from core.runtime.desktop_task_contract import (
    DESKTOP_TASK_ALLOWED_ACTIONS,
    DESKTOP_TASK_RETRY_SAFE_ACTIONS,
)
from core.skills.computer_use import ComputerUseSkill

pytestmark = pytest.mark.unit


def test_create_note_is_a_governed_action():
    assert "create_note" in DESKTOP_TASK_ALLOWED_ACTIONS


def test_create_note_is_retry_safe():
    """A retry writes the same note again rather than half of one."""
    assert "create_note" in DESKTOP_TASK_RETRY_SAFE_ACTIONS


def test_the_executor_exists_and_verifies_by_reading_back():
    import inspect

    source = inspect.getsource(ComputerUseSkill._create_note)
    assert "make new note" in source
    # A note that cannot be found afterwards was not created.
    assert "return name of note" in source
    assert "effect_verified" in source


@pytest.mark.asyncio
async def test_a_note_without_a_body_is_refused():
    skill = ComputerUseSkill()
    result = await skill._create_note({"title": "Empty"})
    assert result["ok"] is False
    assert "body" in result["error"]
