"""This module types into a real browser and presses send. Where it types, and
what it presses, has to be something it can prove.

Four CP126 findings, all the same shape: the least reversible action in the
codebase gated on the weakest available signal.

d4d1b84d — a blank attach returned targets[0]: not the frontmost tab, not the
requested one, not one anybody confirmed. Chrome's target list is not ordered
by visibility, so an arbitrary page received every subsequent send.

25c6379f — navigation returned a snapshot without asserting the requested page
was the one that ended up active.

268fd6dc — the bare words "message" and "reply" counted as evidence of a chat
composer, so an inbox, a settings page or the transcript itself authorized
coordinate typing and an external submit.

7b616aa0 — popup cleanup clicked any visible "Cancel", "No thanks" or "Not
now" button in the front browser window, which are also the decline options in
consent flows, security prompts, purchase confirmations and unsaved-edit
warnings.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from core.capabilities.web_interlocutor import (
    ChromeCDPDialogueBrowser,
    _clipboard_borrow_script,
    _origin_of,
    _screen_text_suggests_chat_composer,
)

pytestmark = pytest.mark.unit


def _adapter(targets):
    adapter = ChromeCDPDialogueBrowser.__new__(ChromeCDPDialogueBrowser)
    adapter.endpoint = "http://127.0.0.1:9222"
    adapter.timeout = 1.0
    adapter._target_ws_url = ""
    adapter._target_id = ""
    adapter._target_origin = ""
    adapter._page_targets = lambda: [dict(t) for t in targets]
    return adapter


# --- the tab is chosen, not stumbled into (d4d1b84d) --------------------


def test_a_single_page_target_is_unambiguous():
    adapter = _adapter(
        [{"id": "A", "url": "https://chat.example/c/1", "webSocketDebuggerUrl": "ws://127.0.0.1:9222/A"}]
    )

    assert adapter._active_target()["id"] == "A"


def test_several_tabs_refuse_rather_than_pick_one():
    """targets[0] is not the frontmost tab and not the user's choice."""
    adapter = _adapter(
        [
            {"id": "A", "url": "https://bank.example/transfer", "webSocketDebuggerUrl": "ws://127.0.0.1:9222/A"},
            {"id": "B", "url": "https://chat.example/c/1", "webSocketDebuggerUrl": "ws://127.0.0.1:9222/B"},
        ]
    )

    with pytest.raises(RuntimeError, match="explicit URL"):
        adapter._active_target()


def test_uncontrollable_targets_do_not_count_toward_ambiguity():
    adapter = _adapter(
        [
            {"id": "A", "url": "https://chat.example/c/1", "webSocketDebuggerUrl": "ws://127.0.0.1:9222/A"},
            {"id": "B", "url": "chrome://newtab"},
        ]
    )

    assert adapter._active_target()["id"] == "A"


# --- the destination is re-proved before the send (d4d1b84d, 25c6379f) --


def test_a_tab_that_navigated_elsewhere_is_refused():
    adapter = _adapter([{"id": "A", "url": "https://elsewhere.example/x"}])
    adapter._target_id = "A"
    adapter._target_origin = "https://chat.example"

    with pytest.raises(RuntimeError, match="moved from"):
        adapter._assert_attached_origin()


def test_a_tab_that_closed_is_refused():
    adapter = _adapter([{"id": "B", "url": "https://chat.example/c/2"}])
    adapter._target_id = "A"
    adapter._target_origin = "https://chat.example"

    with pytest.raises(RuntimeError, match="is gone"):
        adapter._assert_attached_origin()


def test_moving_within_the_same_service_is_allowed():
    """A new conversation on the same site is the same destination."""
    adapter = _adapter([{"id": "A", "url": "https://chat.example/c/999"}])
    adapter._target_id = "A"
    adapter._target_origin = "https://chat.example"

    assert adapter._assert_attached_origin()["id"] == "A"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://chat.example/c/1?x=2#f", "https://chat.example"),
        ("HTTPS://Chat.Example:443/c/1", "https://chat.example:443"),
        ("about:blank", ""),
        ("", ""),
        (None, ""),
        ("not a url", ""),
    ],
)
def test_origin_extraction(url, expected):
    assert _origin_of(url) == expected


# --- composer evidence is a placeholder, not a noun (268fd6dc) ----------


@pytest.mark.parametrize(
    "screen_text",
    [
        "Inbox — 12 unread. Message from Marta about the ferry.",
        "Settings > Notifications > Message previews",
        "She said reply when you can.",
        "Reply-To: nobody@example.com",
        "This article is about how message queues work.",
    ],
)
def test_an_ordinary_page_is_not_a_composer(screen_text):
    assert _screen_text_suggests_chat_composer(screen_text) is False


@pytest.mark.parametrize(
    "screen_text",
    [
        "Ask anything",
        "Message ChatGPT",
        "Send a message...",
        "Type a message",
        "Enter a prompt",
        "Ask Gemini",
    ],
)
def test_a_real_composer_placeholder_still_counts(screen_text):
    assert _screen_text_suggests_chat_composer(screen_text) is True


# --- no pressing a button we cannot identify (7b616aa0) -----------------


def _method_source(name: str) -> str:
    source = pathlib.Path("core/capabilities/web_interlocutor.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name} not found")


@pytest.mark.parametrize("label", ["Cancel", "No thanks", "Not now"])
def test_popup_cleanup_does_not_click_unidentified_buttons(label):
    """Those labels are also the decline option in consent, security, purchase
    and unsaved-edit dialogs."""
    body = _method_source("_dismiss_common_popups")
    code = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("#")
    )
    docstring_free = code.split('"""')[0] + "".join(code.split('"""')[2:])

    assert f'click button "{label}"' not in docstring_free


def test_popup_cleanup_still_dismisses_with_escape():
    assert "key code 53" in _method_source("_dismiss_common_popups")


# --- the clipboard is given back even when the paste fails (297aa442) ---


def test_the_clipboard_is_restored_on_the_failure_path():
    """The restore used to be the last statement with nothing guarding it, so
    an error in the paste left Aura's outbound text in the user's clipboard."""
    script = _clipboard_borrow_script('"secret text"', '    do shell script "false"')

    assert "on error errMsg" in script
    restore_at = script.index("set the clipboard to aura_saved_clip")
    error_at = script.index("on error errMsg")
    assert error_at < restore_at, "the restore must run after the error is caught"


def test_the_original_error_is_re_raised_after_the_restore():
    script = _clipboard_borrow_script('"x"', "    beep")

    restore_at = script.index("set the clipboard to aura_saved_clip")
    raise_at = script.index('if aura_error is not "" then error aura_error')
    assert restore_at < raise_at


def test_every_clipboard_path_goes_through_the_guarded_helper():
    source = pathlib.Path("core/capabilities/web_interlocutor.py").read_text(
        encoding="utf-8"
    )
    # The only place that borrows the clipboard is the helper itself.
    assert source.count("set aura_saved_clip to (the clipboard as text)") == 1
