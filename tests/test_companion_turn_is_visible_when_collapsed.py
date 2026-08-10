"""A turn you cannot see still has to be legible somewhere.

Reported live 2026-08-10: "also no typing indicator, no indicator when a
message has arrived or is waiting".

The companion's own indicator was present and working. The gap was the
COLLAPSED case: the companion page keeps running when the host orders its
window out, so a message sent and then collapsed is answered correctly into a
window nobody is looking at. The bubble it collapsed into is the only thing
left on screen, and it showed neither that she was working nor that an answer
had landed.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.perception.ambient_presence import (
    _COMPANION_WORKING_TTL_S,
    AmbientPresence,
    PresenceMode,
)

ROOT = Path(__file__).resolve().parents[1]
BUBBLE_JS = (ROOT / "interface/static/bubble.js").read_text(encoding="utf-8")
BUBBLE_HTML = (ROOT / "interface/static/bubble.html").read_text(encoding="utf-8")
COMPANION_JS = (ROOT / "interface/static/companion_chat.js").read_text(encoding="utf-8")
LAUNCHER = (ROOT / "scripts/AuraLauncher.swift").read_text(encoding="utf-8")


def _bubble() -> AmbientPresence:
    presence = AmbientPresence()
    presence.set_mode(PresenceMode.BUBBLE)
    return presence


def test_the_three_states_are_distinct_on_the_wire():
    presence = _bubble()

    idle = presence.state(surface="native-bubble")
    assert idle["companion_working"] is False
    assert idle["companion_reply_waiting"] is False

    presence.note_companion_turn(working=True)
    working = presence.state(surface="native-bubble")
    assert working["companion_working"] is True
    assert working["companion_reply_waiting"] is False

    presence.note_companion_reply_waiting()
    waiting = presence.state(surface="native-bubble")
    # Working and waiting are different facts and must not both be true:
    # nothing is waiting to be read until the answer exists.
    assert waiting["companion_working"] is False
    assert waiting["companion_reply_waiting"] is True
    assert waiting["companion_reply_age_s"] is not None


def test_opening_the_window_is_reading_the_reply():
    presence = _bubble()
    presence.note_companion_reply_waiting()
    presence.clear_companion_reply_waiting()
    assert presence.state(surface="native-bubble")["companion_reply_waiting"] is False


def test_a_new_question_supersedes_the_last_answer():
    presence = _bubble()
    presence.note_companion_reply_waiting()
    presence.note_companion_turn(working=True)
    state = presence.state(surface="native-bubble")
    assert state["companion_working"] is True
    assert state["companion_reply_waiting"] is False


def test_a_window_that_never_comes_back_stops_claiming_she_is_working():
    """A crashed window must not leave the bubble working forever."""
    presence = _bubble()
    presence.note_companion_turn(working=True)
    assert presence.state(surface="native-bubble")["companion_working"] is True

    presence._companion_working_at -= _COMPANION_WORKING_TTL_S + 1.0
    assert presence.state(surface="native-bubble")["companion_working"] is False


def test_the_companion_reports_its_turn_and_renews_it():
    assert '"/api/ambient/companion-turn"' in COMPANION_JS
    assert 'reportTurn("working")' in COMPANION_JS
    # Renewal, so the signal can expire on its own if this window goes away.
    assert re.search(r"setInterval\(\s*\(\)\s*=>\s*reportTurn\(\"working\"\)", COMPANION_JS)
    # A reply that lands out of sight is reported differently from one that
    # lands in front of the person.
    assert 'reportTurn("reply_waiting")' in COMPANION_JS
    assert "windowVisible" in COMPANION_JS


def test_only_the_host_can_say_whether_the_companion_is_visible():
    """An ordered-out WKWebView is not 'hidden' by any measure the page has."""
    assert "aura-companion-visibility" in LAUNCHER
    assert "aura-companion-visibility" in COMPANION_JS
    assert "postCompanionVisibility(false)" in LAUNCHER
    assert "postCompanionVisibility(true)" in LAUNCHER


def test_the_bubble_shows_working_without_claiming_something_is_waiting():
    """Working is not unread — nothing is there to read yet."""
    assert "companion_working" in BUBBLE_JS
    assert "companion_reply_waiting" in BUBBLE_JS
    # The dot means "there is something here", so it must not light for working.
    assert re.search(r'toggle\("working",\s*working\s*&&\s*!replyWaiting\)', BUBBLE_JS)
    # A reply answered into a collapsed window is unread by another route.
    assert re.search(r'toggle\("unread",\s*withdrawn\s*\|\|\s*replyWaiting\)', BUBBLE_JS)


def test_the_working_state_survives_reduced_motion():
    """Otherwise reduced motion means no working indicator at all."""
    assert "#pill.working #glyph" in BUBBLE_HTML
    reduced = BUBBLE_HTML.split("prefers-reduced-motion", 1)[1]
    assert "#pill.working #glyph" in reduced
    assert "animation: none" in reduced
