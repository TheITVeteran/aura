"""Boot grace must not protect a boot that has stopped moving.

On 2026-08-03 the desktop could not finish booting: the loop parked in the
skill catalog behind an ABBA deadlock. The stall watchdog noticed 33 times in
three minutes and escalated none of them, because the hard-exit ceiling is
suppressed for the first 1200s of boot. The reaper killed the kernel long
before that grace expired, the launcher started another one, and the race ran
again.

Boot grace was a pure time budget, so a boot that is progressing slowly and a
boot parked on one frame looked identical to it. They are not: a progressing
boot moves. The watchdog now watches the loop's innermost frame, and a loop
that has not moved for longer than the wedge ceiling loses the grace.
"""
from __future__ import annotations

import time

import pytest

from core.resilience.stall_watchdog import StallWatchdog


class _FakeLoop:
    def is_closed(self) -> bool:
        return False

    def is_running(self) -> bool:
        return True


@pytest.fixture
def watchdog():
    return StallWatchdog(_FakeLoop())


class TestFrameMotionIsTheEvidence:
    def test_a_moving_frame_never_accumulates_stuck_time(self, watchdog, monkeypatch):
        frames = iter(["boot.py:10:a", "boot.py:20:b", "boot.py:30:c"])
        monkeypatch.setattr(watchdog, "_loop_frame_signature", lambda: next(frames))

        for _ in range(3):
            stuck_for, _frame = watchdog._boot_frame_stuck_for()
            assert stuck_for == 0.0, "a loop that moved must reset the clock"

    def test_an_unchanged_frame_accumulates(self, watchdog, monkeypatch):
        monkeypatch.setattr(
            watchdog, "_loop_frame_signature", lambda: "capability_engine.py:2409:_reload"
        )

        first, frame = watchdog._boot_frame_stuck_for()
        assert first == 0.0, "the first sighting establishes the anchor"
        assert frame == "capability_engine.py:2409:_reload"

        watchdog._boot_frame_signature_since = time.time() - 42.0
        stuck_for, frame = watchdog._boot_frame_stuck_for()
        assert stuck_for >= 42.0
        assert frame == "capability_engine.py:2409:_reload"

    def test_an_unreadable_frame_is_not_evidence_of_a_wedge(self, watchdog, monkeypatch):
        """The loop thread id is only learned once the heartbeat has run."""

        monkeypatch.setattr(watchdog, "_loop_frame_signature", lambda: "")
        watchdog._boot_frame_signature_since = time.time() - 600.0

        stuck_for, frame = watchdog._boot_frame_stuck_for()
        assert stuck_for == 0.0
        assert frame == ""


class TestBootGraceStopsProtectingAWedge:
    def _in_boot_grace(self, watchdog) -> None:
        watchdog._started_at = time.time()  # boot grace is 1200s by default
        assert watchdog._hard_exit_boot_grace_s() > 0

    def test_a_progressing_boot_is_still_protected(self, watchdog, monkeypatch):
        self._in_boot_grace(watchdog)
        frames = iter(["a.py:1:x", "b.py:2:y"] * 10)
        monkeypatch.setattr(watchdog, "_loop_frame_signature", lambda: next(frames))

        silence = watchdog._hard_exit_ceiling_s() + 10.0
        assert watchdog._should_force_exit(silence) is False, (
            "a boot whose loop is still moving must keep its grace"
        )

    def test_a_wedged_boot_loses_the_grace(self, watchdog, monkeypatch):
        self._in_boot_grace(watchdog)
        monkeypatch.setattr(
            watchdog, "_loop_frame_signature", lambda: "capability_engine.py:2409:_reload"
        )
        # Prime the anchor, then age it past the ceiling.
        watchdog._boot_frame_stuck_for()
        ceiling = watchdog._hard_exit_ceiling_s()
        watchdog._boot_frame_signature_since = time.time() - (ceiling + 5.0)

        assert watchdog._should_force_exit(ceiling + 10.0) is True, (
            "a loop parked on one frame past the wedge ceiling is wedged, not "
            "booting slowly — this is the escalation that never fired"
        )

    def test_silence_below_the_ceiling_still_returns_early(self, watchdog, monkeypatch):
        """A stuck frame is not on its own a reason to kill a live process."""

        self._in_boot_grace(watchdog)
        monkeypatch.setattr(watchdog, "_loop_frame_signature", lambda: "stuck.py:1:f")
        watchdog._boot_frame_stuck_for()
        watchdog._boot_frame_signature_since = time.time() - 10_000.0

        assert watchdog._should_force_exit(1.0) is False, (
            "the loop answering within the ceiling is a live loop"
        )
