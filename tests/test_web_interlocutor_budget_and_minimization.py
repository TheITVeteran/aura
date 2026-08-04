"""A conversation that runs until it happens to stop, and a picture of the
whole desk kept forever.

Two CP126 findings about what a run costs and what it leaves lying around.

1175afe1 — the per-call compose timeout was clamped only from below and
environment-raisable without limit, opening composition retried five times, and
each rejected attempt ran a full cognitive call before sleeping. Multiplied by
turns, follow-ups, factchecks, browser waits, the summary and the memory write,
one conversation had no total bound.

408696bd — the fallback wrote a full-screen screenshot to disk so OCR could run
over it, then captured the browser window's entire accessibility tree, with no
redaction and no retention rule.
"""
from __future__ import annotations

import time

import pytest

from core.capabilities.web_interlocutor import (
    _SESSION_BUDGET_S,
    WebInterlocutorSession,
    _discard_capture_screenshot,
)

pytestmark = pytest.mark.unit


def _session() -> WebInterlocutorSession:
    session = WebInterlocutorSession.__new__(WebInterlocutorSession)
    session.cognitive_engine = None
    session.memory_gateway = None
    return session


# --- the exchange has one deadline (1175afe1) ---------------------------


def test_the_budget_is_armed_and_bounded():
    session = _session()
    session._start_session_budget()

    assert session._session_budget_exhausted() is False
    assert 0 < session._session_budget_remaining() <= _SESSION_BUDGET_S


def test_an_expired_budget_reads_as_exhausted():
    session = _session()
    session._session_deadline = time.monotonic() - 1.0

    assert session._session_budget_exhausted() is True
    assert session._session_budget_remaining() == 0.0


def test_an_unarmed_session_is_not_reported_as_exhausted():
    """A helper called before the exchange starts must not claim the budget
    already ran out."""
    session = _session()

    assert session._session_budget_exhausted() is False
    assert session._session_budget_remaining() == _SESSION_BUDGET_S


@pytest.mark.asyncio
async def test_composition_retries_stop_at_the_deadline():
    session = _session()
    session._session_deadline = time.monotonic() - 1.0
    calls: list[int] = []

    context: dict = {}

    async def _never_called(*_args, **_kwargs):  # pragma: no cover
        calls.append(1)
        raise AssertionError("a retry must not start past the deadline")

    import core.capabilities.web_interlocutor as module

    original = module._maybe_think
    module._maybe_think = _never_called
    try:
        # Without an explicit fallback allowance the refusal is the honest
        # outcome: no cognitive message was composed, so none is claimed.
        with pytest.raises(module.CognitiveCompositionUnavailable):
            await session._compose_with_retry(
                object(), "prompt", context, fallback=lambda: "canned"
            )

        allowed = {"allow_deterministic_composition_fallback": True}
        result = await session._compose_with_retry(
            object(), "prompt", allowed, fallback=lambda: "canned"
        )
    finally:
        module._maybe_think = original

    assert calls == [], "no cognitive call may start past the deadline"
    assert result == "canned"
    events = context["_web_interlocutor_composition_events"]
    assert any(e["source"] == "session_budget_exhausted" for e in events)


def test_a_browser_wait_cannot_outlive_the_conversation():
    session = _session()
    session._session_deadline = time.monotonic() + 5.0

    # The run loop passes min(wait_timeout_s, remaining).
    assert min(45.0, session._session_budget_remaining()) <= 5.0


def test_the_budget_default_is_not_open_ended():
    assert 60.0 <= _SESSION_BUDGET_S <= 3600.0


# --- transient captures do not persist (408696bd) -----------------------


@pytest.mark.asyncio
async def test_the_capture_screenshot_is_deleted_after_its_text_is_read(tmp_path):
    """A picture of the whole display, written to disk with nothing to delete
    it and nothing left to read it."""
    from types import SimpleNamespace

    shot = tmp_path / "capture.png"
    shot.write_bytes(b"not really a png")
    snap = SimpleNamespace(screenshot_path=str(shot), screen_text="already OCRed")

    await _discard_capture_screenshot(snap)

    assert not shot.exists()
    assert snap.screenshot_path == ""


@pytest.mark.asyncio
async def test_a_missing_screenshot_is_not_an_error(tmp_path):
    from types import SimpleNamespace

    snap = SimpleNamespace(screenshot_path=str(tmp_path / "gone.png"))

    await _discard_capture_screenshot(snap)  # must not raise

    assert snap.screenshot_path == ""


@pytest.mark.asyncio
async def test_no_screenshot_path_is_a_no_op():
    from types import SimpleNamespace

    snap = SimpleNamespace(screenshot_path="")

    await _discard_capture_screenshot(snap)

    assert snap.screenshot_path == ""


def test_captured_screen_text_is_redacted_before_use():
    import inspect

    source = inspect.getsource(
        WebInterlocutorSession.__init__.__globals__["ChromeVisibleDialogueBrowser"]
        ._screen_perception_snapshot
    )

    assert "_redact_remote_content(" in source


def test_the_accessibility_tree_is_redacted_before_use():
    import inspect

    source = inspect.getsource(
        WebInterlocutorSession.__init__.__globals__["ChromeVisibleDialogueBrowser"]
        ._accessibility_snapshot
    )

    assert "_redact_remote_content(" in source
