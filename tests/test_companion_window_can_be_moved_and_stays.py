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


def test_the_glyph_is_draggable_because_at_rest_it_is_the_whole_bubble():
    """The one element excluded from the drag was the entire grabbable surface.

    ``#pill.dormant`` is a 6px scrim around a 28px mark and nothing else, so
    excluding ``#glyph`` meant ``drag`` was never assigned in the state the
    bubble spends all its time in.
    """
    mousedown = BUBBLE_JS.split('pill.addEventListener("mousedown"', 1)[1].split("});", 1)[0]

    assert "closest(" in mousedown, "the handler must still protect real controls"
    # × and the reply control stay targets.
    assert "#close" in mousedown and "#say" in mousedown
    # The glyph must not be among them.
    assert "#glyph" not in mousedown, "the glyph is excluded from the drag again"


def test_a_drag_that_ends_on_the_glyph_does_not_also_open_the_chat():
    """Otherwise every move is followed by the window it was moved aside for."""
    assert "dragJustEnded" in BUBBLE_JS
    click_handler = BUBBLE_JS.split('glyph.addEventListener("click"', 1)[1].split("});", 1)[0]
    assert "dragJustEnded()" in click_handler
    # And the suppression window must actually be armed when a drag ends.
    assert re.search(r"drag\.moved\s*\)\s*draggedAt\s*=", BUBBLE_JS)


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
