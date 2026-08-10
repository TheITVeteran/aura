"""791 consecutive failures produced no signal of any kind.

LIVE DEFECT, 2026-08-10. Bryan reported companion mode "doesn't work". The
live ``/api/ambient/state`` said:

    {"ticks": 794, "observations": 0,
     "skips": {"capture_failed": 791, "private_window": 3}}

The organ had run for 794 ticks and had never once observed anything. Nothing
said so. Specifically:

  * FOUR distinct branches raise ``CAPTURE_FAILED`` — no frontmost window, the
    read raising, the receipt reporting failure, and an empty capture — and
    all four incremented the same integer. The endpoint could tell you
    observation was dead and not what killed it, which is the only part that
    leads to a fix;
  * two of those four recorded no degradation at all, and the ``detail`` that
    ``_skip`` already accepted was dropped on the floor by ``state()``;
  * ``record_degradation`` fires on the raising branch only, so a capture that
    "succeeded" with empty text was silent forever.

The tests here are about the REPORTING, not the capture. Capture can fail for
reasons outside the process — permissions, a locked screen, no display. Being
unable to see is allowed. Being unable to see while reporting nothing is not.
"""
from __future__ import annotations

import asyncio

import pytest

from core.perception.ambient_presence import (
    _BROKEN_SKIP_ESCALATION,
    AmbientPresence,
    PresenceMode,
    ScreenContext,
    SkipReason,
)


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


def _capture_returns(presence, text: str, *, app: str = "Google Chrome") -> None:
    """Wire a tick that reaches the capture and gets `text` back."""

    async def _context():
        return ScreenContext(app=app, title="a public window")

    async def _read():
        return text

    presence._current_context = _context
    presence._read_screen_text = _read


def test_an_empty_capture_says_which_adapter_returned_nothing(presence):
    """The branch that reported literally nothing."""
    _capture_returns(presence, "   ")

    result = asyncio.run(presence.tick())

    assert result.skip_reason is SkipReason.CAPTURE_FAILED
    assert result.detail, "an empty capture must say something"
    state = presence.state()
    assert "capture_failed" in state["skip_details"]
    assert state["last_skip"]["reason"] == "capture_failed"
    assert state["last_skip"]["detail"]


def test_no_frontmost_window_is_distinguishable_from_an_empty_capture(presence):
    """Both were "capture_failed: N" and nothing else."""

    async def _no_context():
        return None

    presence._current_context = _no_context
    asyncio.run(presence.tick())
    from_no_window = presence.state()["skip_details"]["capture_failed"]

    _capture_returns(presence, "")
    asyncio.run(presence.tick())
    from_empty = presence.state()["skip_details"]["capture_failed"]

    assert from_no_window != from_empty


def test_persistent_total_failure_is_escalated_once(presence, monkeypatch):
    """791 failures recorded zero degradations. One is the right number.

    Not zero, because a blind companion is a broken one. Not one per tick,
    because at a 6s cadence that is its own outage.
    """
    recorded: list[tuple] = []
    monkeypatch.setattr(
        "core.perception.ambient_presence.record_degradation",
        lambda subsystem, exc, **kw: recorded.append((subsystem, str(exc), kw)),
    )
    _capture_returns(presence, "")

    for _ in range(_BROKEN_SKIP_ESCALATION * 3):
        asyncio.run(presence.tick())

    assert len(recorded) == 1, recorded
    subsystem, message, kwargs = recorded[0]
    assert subsystem == "ambient_presence"
    assert str(_BROKEN_SKIP_ESCALATION) in message
    assert kwargs.get("severity") == "warning"


def test_blindness_is_visible_in_the_state_endpoint(presence):
    """The one field that answers Bryan's question directly."""
    _capture_returns(presence, "")

    assert presence.state()["blind"] is False
    for _ in range(_BROKEN_SKIP_ESCALATION):
        asyncio.run(presence.tick())

    state = presence.state()
    assert state["blind"] is True
    assert state["consecutive_broken_skips"] >= _BROKEN_SKIP_ESCALATION


def test_a_real_observation_clears_blindness(presence):
    _capture_returns(presence, "")
    for _ in range(_BROKEN_SKIP_ESCALATION):
        asyncio.run(presence.tick())
    assert presence.state()["blind"] is True

    _capture_returns(presence, "a screen with words on it", app="Preview")
    asyncio.run(presence.tick())

    state = presence.state()
    assert state["observations"] == 1
    assert state["blind"] is False
    assert state["consecutive_broken_skips"] == 0


@pytest.mark.parametrize(
    ("reason", "app", "title"),
    [(SkipReason.PRIVATE_WINDOW, "1Password", "vault")],
)
def test_the_organ_working_correctly_never_raises_an_alarm(
    presence, monkeypatch, reason, app, title
):
    """A privacy skip is the design succeeding. It must not read as an outage.

    If working-as-intended skips escalated, the alarm would fire constantly
    and stop being read — which is how the real one went unnoticed.
    """
    recorded: list = []
    monkeypatch.setattr(
        "core.perception.ambient_presence.record_degradation",
        lambda *a, **k: recorded.append(a),
    )

    async def _context():
        return ScreenContext(app=app, title=title)

    presence._current_context = _context
    for _ in range(_BROKEN_SKIP_ESCALATION * 2):
        asyncio.run(presence.tick())

    assert presence.state()["skips"].get(reason.value)
    assert presence.state()["blind"] is False
    assert not recorded


def test_hidden_is_not_an_outage(presence, monkeypatch):
    """Someone dismissing her is not a failure to report."""
    recorded: list = []
    monkeypatch.setattr(
        "core.perception.ambient_presence.record_degradation",
        lambda *a, **k: recorded.append(a),
    )
    presence.hide()

    for _ in range(_BROKEN_SKIP_ESCALATION * 2):
        asyncio.run(presence.tick())

    assert presence.state()["blind"] is False
    assert not recorded
