"""Hiding her stopped the drawing and not the looking.

``AmbientPresence`` honoured HIDDEN correctly: no observation, no utterance,
no overlay. ``/api/ambient/visibility`` existed and applied it correctly. And
nothing in the entire product ever called it — not the launcher, not the
bubble, not the chat window. ``hideBubble()`` was one line, ``orderOut(nil)``,
so the panel vanished and the observation loop kept reading the screen of
someone who had just dismissed her.

There was also no way to ask. The launcher had a ``case "hide"`` handler and
no surface ever sent that message; the bubble's × clears a MESSAGE, which is
the weaker of the two controls and was the only one reachable.

The tests below are deliberately split between the organ and the surfaces,
because the organ was never the part that was broken. Asserting only that
HIDDEN suppresses observation is what passed while the feature did nothing —
so the surface assertions are the ones with teeth here.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.perception.ambient_presence import (
    AmbientPresence,
    PresenceMode,
    ScreenContext,
    SkipReason,
)

_LAUNCHER = Path("scripts/AuraLauncher.swift")
_BUBBLE_JS = Path("interface/static/bubble.js")


@pytest.fixture
def presence():
    instance = AmbientPresence()
    instance.set_mode(PresenceMode.BUBBLE)
    return instance


@pytest.fixture(autouse=True)
def _allow_proactivity(monkeypatch):
    monkeypatch.setattr(
        "core.perception.ambient_presence._proactivity_suppressed", lambda: False
    )


def _with_context(presence, app="Google Chrome", title="GitHub"):
    async def _context():
        return ScreenContext(app=app, title=title)

    async def _read():
        _read.called = True
        return "screen text"

    _read.called = False
    presence._current_context = _context
    presence._read_screen_text = _read
    return _read


# ───────────────────────────────────────────── the organ honours it (it did)


def test_hidden_stops_the_capture_not_just_the_render(presence):
    reader = _with_context(presence)
    presence.hide()

    result = asyncio.run(presence.tick())

    assert result.skip_reason is SkipReason.HIDDEN
    assert reader.called is False


def test_unhiding_lets_her_look_again(presence):
    reader = _with_context(presence)
    presence.hide()
    asyncio.run(presence.tick())
    presence.show()

    assert asyncio.run(presence.tick()).observed is True
    assert reader.called is True


# ──────────────────────────────────── the surfaces actually ask for it (new)


def test_the_launcher_tells_the_runtime_when_she_is_hidden():
    """The half that was missing. Without this, hide is a layout change."""
    source = _LAUNCHER.read_text(encoding="utf-8")

    assert "/api/ambient/visibility" in source, (
        "hiding the panel never reached the runtime, so she kept observing"
    )
    assert "postAmbientMode" in source


def test_opening_the_full_window_does_not_read_as_being_dismissed():
    """Opening the desktop window retires the bubble. That is not a dismissal.

    ``openNativeDesktopWindow`` retires the bubble on its way in. If it did
    that through the dismissal path, every person who opened her would have
    told the runtime she had been sent away, and observation would stop for
    the one surface most clearly indicating they want her.
    """
    source = _LAUNCHER.read_text(encoding="utf-8")

    assert "orderBubbleOut" in source, (
        "a surface change and a dismissal must not share one code path"
    )
    opening = _function_body(source, "private func openNativeDesktopWindow()")
    assert "orderBubbleOut()" in opening
    assert "hideBubble()" not in opening, (
        "opening the window would report her as hidden and stop her looking"
    )
    assert 'postAmbientMode("window")' in opening


def _function_body(source: str, signature: str) -> str:
    """The text of one Swift function, bounded by the next declaration.

    A fixed character window is the wrong bound: it silently shrinks the
    assertion's reach as the function grows, so a rule can slide out of scope
    and the test keeps passing while checking less.
    """
    after = source.split(signature, 1)[1]
    # The EARLIEST following declaration, of any visibility. Stopping only at
    # `private func` ran straight past the `func userContentController` that
    # follows, swallowing its `case "hide": hideBubble()` and making the
    # assertion below fail against code that was already correct.
    ends = [
        offset
        for offset in (
            after.find(marker)
            for marker in ("\n    private func ", "\n    func ", "\n    // MARK:")
        )
        if offset != -1
    ]
    return after[: min(ends)] if ends else after


def test_showing_the_bubble_clears_hidden():
    """Otherwise hiding her once is permanent until a restart."""
    source = _LAUNCHER.read_text(encoding="utf-8")
    showing = _function_body(source, "private func showBubble()")

    assert 'postAmbientMode("bubble")' in showing


def test_there_is_a_way_to_ask_her_to_go_away():
    """The launcher's hide handler had no sender for its whole life."""
    source = _BUBBLE_JS.read_text(encoding="utf-8")

    assert "contextmenu" in source, "no surface offers hide at all"
    assert 'action: "hide"' in source, (
        "the launcher's hide handler is still unreachable"
    )


def test_a_message_withdraws_to_the_dot_instead_of_sitting_there():
    """The unread dot lit only when the text was already spelled out.

    Which made it decoration: it announced "there is something here" while
    the something was legible an inch to its right. Meanwhile an unread
    remark stayed expanded over the person's work indefinitely.

    Both halves are one fix. The message withdraws after a while and the dot
    is what remains — non-modal, and the only surviving trace of a sentence
    nobody acknowledged, which is what "open me when you can" means.
    """
    source = _BUBBLE_JS.read_text(encoding="utf-8")

    assert "WITHDRAW_AFTER_S" in source
    assert "utterance_age_s" in source, "nothing measures how long it has sat there"
    assert '"unread"' in source

    html = Path("interface/static/bubble.html").read_text(encoding="utf-8")
    assert "#pill.unread #dot" in html
    assert "#pill.speaking #dot" not in html, (
        "the dot would light beside text that is already on screen"
    )


def test_a_withdrawn_message_is_still_reachable():
    """Withdrawing must not mean losing it.

    A remark that vanishes silently can only be learned by having been
    looking at the moment it appeared.
    """
    source = _BUBBLE_JS.read_text(encoding="utf-8")

    assert "if (withdrawn) pill.title = text;" in source


def test_hiding_and_clearing_stay_different_acts():
    """× dismisses a message. Right-click dismisses her.

    Collapsing them would mean either that clearing a message silently stops
    her observing, or that the only reachable control is the weaker one —
    which is the state this shipped in.
    """
    source = _BUBBLE_JS.read_text(encoding="utf-8")

    assert "/api/ambient/clear" in source
    assert "/api/ambient/visibility" in source
    assert "hideHer" in source
