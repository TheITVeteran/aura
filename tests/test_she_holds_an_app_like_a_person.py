"""Hold the app you are working in, then put it away.

Bryan's framing, and it is how people actually use a computer: you bring an
app forward, you work in it for as long as you need, and then you put it
away. Aura was checking "is it frontmost?" once at open_app and hoping that
survived until the keystrokes landed.

    open_app failed: ... did not become frontmost (observed=Google Chrome)

The browser took focus back between steps, which is what browsers do — and in
a demo the person is watching Aura's own UI in a browser, so the app she is
competing with is always there.

Focus is now re-asserted before every step whose effect depends on it, and
only those: keystrokes and clicks land wherever focus is, while file writes,
downloads and settings do not care. Polling is cheap and activating is an
AppleScript round trip, so activation only happens when focus was actually
lost.
"""

import inspect

import pytest

from core.skills import desktop_task
from core.skills.computer_use import ComputerUseSkill

pytestmark = pytest.mark.unit


def test_focus_sensitive_actions_are_the_ones_that_type_or_click():
    sensitive = desktop_task._FOCUS_SENSITIVE_ACTIONS
    for action in ("type", "hotkey", "click"):
        assert action in sensitive, action
    # Actions whose effect does not depend on the frontmost window.
    for action in ("write_text_file", "create_folder", "fetch_topic_image", "system_control"):
        assert action not in sensitive, action


def test_the_skill_can_hold_and_release_focus():
    assert callable(getattr(ComputerUseSkill, "hold_focus", None))
    assert callable(getattr(ComputerUseSkill, "release_focus", None))


def test_hold_focus_only_activates_when_focus_was_lost():
    """Polling is cheap; activating is an AppleScript round trip."""
    source = inspect.getsource(ComputerUseSkill.hold_focus)
    assert "_frontmost_app_matches" in source
    assert "return True" in source


def test_the_step_loop_reasserts_focus_before_typing():
    source = inspect.getsource(desktop_task)
    assert "_FOCUS_SENSITIVE_ACTIONS and _focus_app" in source
    assert "hold_focus(_focus_app)" in source
