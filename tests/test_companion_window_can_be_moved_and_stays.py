"""The companion has to move where she is put, and stay where she is left.

Reported live 2026-08-10, one sitting AFTER the fix that added dragging:
"i still cant drag across the screen" and "when i click on another window,
the companion goes away".

Both surfaces asked for the drag with `-webkit-app-region: drag`, an Electron
property WKWebView does not implement. The declaration was inert on every
surface it appeared on, and the comment beside it described a mechanism that
was never present — which is why the first fix went looking in the wrong
layer and the second report was identical to the first.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUBBLE_JS = (ROOT / "interface/static/bubble.js").read_text(encoding="utf-8")
BUBBLE_HTML = (ROOT / "interface/static/bubble.html").read_text(encoding="utf-8")
COMPANION_HTML = (ROOT / "interface/static/companion_chat.html").read_text(encoding="utf-8")
LAUNCHER = (ROOT / "scripts/AuraLauncher.swift").read_text(encoding="utf-8")


def test_no_surface_relies_on_the_electron_drag_property():
    """It does nothing in WKWebView, and reads as though it does everything."""
    for name, source in (
        ("bubble.html", BUBBLE_HTML),
        ("companion_chat.html", COMPANION_HTML),
    ):
        # Prose explaining why it is gone is fine; a live declaration is not.
        declarations = re.findall(r"^\s*-webkit-app-region\s*:", source, re.MULTILINE)
        assert not declarations, f"{name} still declares -webkit-app-region"


def test_the_bubble_drags_from_anywhere_on_its_surface():
    """The guarantee moved to the window server; the test follows it there.

    Both of the tests that used to live here read bubble.js for a JS drag
    recognizer — a mousedown handler, a `drag` object, a `dragJustEnded`
    suppression window. That implementation is gone, and deliberately: a 56x56
    WKWebView only receives mousemove for points inside itself, so a JS drag
    died about 28px in no matter how it was written. The pan recognizer on the
    host owns it now, in global screen coordinates.

    A test pinned to a deleted mechanism fails without anything being wrong,
    which is the same amount of information as not existing — and worse,
    because it trains the reader to ignore it.
    """

    assert "installWindowDrag(on: webView, topStrip:" in LAUNCHER
    assert "final class TopStripPanGestureRecognizer: NSPanGestureRecognizer" in LAUNCHER
    # topStrip 0 means the whole surface drags. The bubble is all glyph, which
    # is exactly what the old JS could not express: excluding #glyph from the
    # handle left only the pixels the controls did not claim.
    # Two installs, and the difference between them is the whole point: the
    # bubble takes the default topStrip of 0 (drag from anywhere, it is all
    # glyph) and the companion takes a strip so dragging across the transcript
    # still selects text.
    assert "installWindowDrag(on: webView)\n" in LAUNCHER, (
        "the bubble no longer drags from its whole surface"
    )
    assert "installWindowDrag(on: webView, topStrip: 37)" in LAUNCHER, (
        "the companion no longer reserves a strip, so its transcript is a handle"
    )


def test_a_drag_does_not_also_open_the_chat():
    """A pan and a click are different gestures, decided by the recognizer.

    `delaysPrimaryMouseButtonEvents = false` is what keeps a plain tap opening
    the chat. The separation in the other direction — a drag not ALSO counting
    as a tap — is the pan recognizer claiming the gesture once the pointer
    moves, which is AppKit's behaviour rather than something this page can
    assert about itself. What is checkable here is that the page holds no
    competing drag of its own: two recognizers for one gesture is how a move
    ends with the window it was moved aside for.
    """

    assert "delaysPrimaryMouseButtonEvents = false" in LAUNCHER

    # Code only. The prose above these lines quotes the removed approach
    # verbatim, including `{action:"move"}`, and a check that reads comments
    # cannot tell an explanation of a mistake from the mistake.
    code = re.sub(r"/\*.*?\*/", "", BUBBLE_JS, flags=re.DOTALL)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.MULTILINE)
    for gesture in ('addEventListener("mousemove"', 'addEventListener("mousedown"'):
        assert gesture not in code, (
            f"bubble.js is recognising drags again ({gesture}) alongside the "
            "host recognizer; two recognizers for one gesture is how a move "
            "ends with the window it was moved aside for"
        )
    # `forwardMove` stays: that is the HOST commanding a position, which is how
    # a remembered position is restored. It is not a drag the page recognised.
    assert "function forwardMove(" in code


def test_the_bubble_has_exactly_one_move_mechanism():
    """A native gesture plus the page's own drag moves the panel twice per motion."""
    bubble_block = LAUNCHER.split("bubblePanel = panel", 1)[0]
    # The bubble's panel construction must not install a native drag: only the
    # page knows whether the pointer went down on × or the reply control.
    assert "installWindowDrag(on: webView)" not in bubble_block


def test_the_companion_window_is_draggable_by_its_title_strip():
    assert "installWindowDrag(on: webView, topStrip:" in LAUNCHER
    assert "TopStripPanGestureRecognizer" in LAUNCHER
    # A pan that delays the primary mouse button would swallow the clicks the
    # composer and the FULL button need.
    assert "delaysPrimaryMouseButtonEvents = false" in LAUNCHER


def test_the_companion_does_not_hide_itself_when_another_app_is_clicked():
    """NSPanel defaults hidesOnDeactivate to TRUE, unlike NSWindow.

    Unset, every click into another app ordered the companion out, which reads
    as the window closing itself.
    """
    companion_block = LAUNCHER.split("let panel = KeyablePanel(", 1)[1].split(
        "companionPanel = panel", 1
    )[0]
    assert "hidesOnDeactivate = false" in companion_block
