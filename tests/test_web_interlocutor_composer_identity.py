"""Before pressing Return into somebody's browser, know which box you are in.

Two CP126 findings in how a composer is chosen and confirmed.

8598d4b2 — the DOM path scored every visible text field by label fragments and
vertical position, took the top one, typed, and pressed Enter. The label test
matched the bare words "message" and "reply", there was no per-site contract,
and nothing confirmed the chosen element was the one that actually took focus.

de352148 — the coordinate fallback accepted "the perception model was 0.76
confident this is a text input" plus "the accessibility snapshot was sparse" as
proof of a chat composer, then pasted and pressed Return. Nothing re-checked
focus between verification and send.
"""
from __future__ import annotations

import re

import pytest

from core.capabilities.web_interlocutor import ChromeVisibleDialogueBrowser

pytestmark = pytest.mark.unit


def _focus_script() -> str:
    """The JS the CDP adapter evaluates to choose and focus a composer."""
    import inspect

    from core.capabilities.web_interlocutor import ChromeCDPDialogueBrowser

    return inspect.getsource(ChromeCDPDialogueBrowser.send_message)


# --- the DOM composer is chosen against a contract (8598d4b2) -----------


def test_the_label_test_no_longer_matches_the_bare_nouns():
    script = _focus_script()
    prompt_like = re.search(r"const promptLike = \(label\) => /([^/]+)/", script)

    assert prompt_like, "the composer label test must still exist"
    alternatives = prompt_like.group(1).split("|")
    assert "message" not in alternatives
    assert "reply" not in alternatives
    assert "ask anything" in alternatives


@pytest.mark.parametrize(
    "host", ["chatgpt.com", "chat.openai.com", "claude.ai", "gemini.google.com"]
)
def test_the_known_surfaces_have_a_composer_contract(host):
    assert f"'{host}':" in _focus_script()


def test_a_site_with_a_contract_refuses_rather_than_guesses():
    """If the contracted element is missing, the generic scorer must not run —
    that is the path that types into the wrong box."""
    script = _focus_script()

    assert "site_composer_contract_not_satisfied" in script
    contract_refusal = script.index("site_composer_contract_not_satisfied")
    generic_scorer = script.index("document.querySelectorAll(\n")
    assert contract_refusal < generic_scorer


def test_the_send_requires_the_chosen_element_to_have_taken_focus():
    script = _focus_script()

    assert "composer_did_not_take_focus" in script
    focus_check = script.index("composer_did_not_take_focus")
    insert = script.index('"Input.insertText"')
    assert focus_check < insert, "focus must be confirmed before text is inserted"


# --- the coordinate fallback proves identity, not confidence (de352148) --


def test_perception_confidence_is_no_longer_composer_proof():
    import inspect

    source = inspect.getsource(
        ChromeVisibleDialogueBrowser._visible_keyboard_send_message
    )
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )

    assert "target_confidence" not in code
    assert "_focused_snapshot_is_sparse_browser" not in code


def test_focus_is_rechecked_between_verification_and_send():
    import inspect

    source = inspect.getsource(
        ChromeVisibleDialogueBrowser._visible_keyboard_send_message
    )

    recheck = source.index("_focused_snapshot_matches")
    paste = source.index("_paste_and_submit")
    assert recheck < paste


_COMPOSER = "\n".join(
    [
        "process:Google Chrome",
        "AXRole:AXTextArea",
        "AXTitle:",
        "AXDescription:Ask anything",
        "AXPlaceholderValue:Ask anything",
        "AXValue:",
        "AXPosition:100,900",
        "AXSize:800,60",
    ]
)


def test_the_same_element_still_matches_after_the_user_types():
    typed = _COMPOSER.replace("AXValue:", "AXValue:hello there")

    assert ChromeVisibleDialogueBrowser._focused_snapshot_matches(_COMPOSER, typed)


def test_the_same_element_still_matches_after_the_composer_reflows():
    """A streaming reply moves the composer; that is not a different box."""
    moved = _COMPOSER.replace("AXPosition:100,900", "AXPosition:100,840")

    assert ChromeVisibleDialogueBrowser._focused_snapshot_matches(_COMPOSER, moved)


def test_a_different_element_does_not_match():
    other = _COMPOSER.replace("AXRole:AXTextArea", "AXRole:AXTextField").replace(
        "AXDescription:Ask anything", "AXDescription:Search"
    )

    assert not ChromeVisibleDialogueBrowser._focused_snapshot_matches(_COMPOSER, other)


def test_focus_moving_to_another_application_does_not_match():
    elsewhere = _COMPOSER.replace("process:Google Chrome", "process:Mail")

    assert not ChromeVisibleDialogueBrowser._focused_snapshot_matches(
        _COMPOSER, elsewhere
    )


@pytest.mark.parametrize("unreadable", ["", "   ", "process:Chrome\nerror:no focus"])
def test_an_unreadable_snapshot_fails_closed(unreadable):
    """Not knowing which element has focus is not knowing it is the right one."""
    assert not ChromeVisibleDialogueBrowser._focused_snapshot_matches(
        _COMPOSER, unreadable
    )
    assert not ChromeVisibleDialogueBrowser._focused_snapshot_matches(
        unreadable, _COMPOSER
    )
