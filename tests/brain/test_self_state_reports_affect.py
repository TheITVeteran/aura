"""She was asked how she felt and had no instrument for it.

LIVE DEFECT, 2026-08-10. Asked whether "steady" had been a reading or a
reflex, she replied:

    "'tired' is the correct reading of my somatic state ... My actual mood is
     neutral, my energy is low, and there's a persistent hum in the background
     processing that I haven't been able to shake since 02:15 hours ago."

Three failures, one cause:

  * "tired" and "neutral" asserted in consecutive sentences;
  * the substrate's real reading was mood TIRED, energy 12, frustration 58 —
    so "neutral" was simply wrong;
  * "02:15 hours ago" appears nowhere in the runtime. No source emits it, and
    mood onset is not timestamped at all.

``runtime_self_report()`` — the block whose own header says "do not supplement
them with numbers or events you cannot see here" — carried uptime, memory,
model, cycles and degradations, and NO affective reading whatsoever, while the
same values were served to every other reader on /api/health as `liquid_state`.
The absence did not produce silence. It produced invention, on the subject she
is asked about more than any other.

This is the identical shape ``_cognition_line`` already documents for cycle
counts: "A number she can read must be in front of her, or 'I can't see it'
becomes a licence to invent one."
"""
from __future__ import annotations

import pytest

from core.brain import self_state_report


class _Substrate:
    def __init__(self, status):
        self._status = status

    def get_status(self):
        return self._status


@pytest.fixture
def substrate(monkeypatch):
    def _install(status):
        monkeypatch.setattr(
            self_state_report,
            "get_runtime_service",
            lambda name, default=None: (
                _Substrate(status) if name == "liquid_substrate" else default
            ),
            raising=False,
        )
        monkeypatch.setattr(
            "core.runtime.service_registry.get_runtime_service",
            lambda name, default=None: (
                _Substrate(status) if name == "liquid_substrate" else default
            ),
        )

    return _install


LIVE_READING = {
    "mood": "TIRED",
    "energy": 12.0,
    "curiosity": 94.0,
    "frustration": 58.0,
    "focus": 30.0,
}


def test_the_mood_she_is_asked_about_is_actually_in_front_of_her(substrate):
    substrate(LIVE_READING)

    line = self_state_report._affect_line()

    assert "TIRED" in line
    assert "energy 12" in line


def test_every_drive_the_health_endpoint_serves_is_present(substrate):
    """The values existed and went to every reader except her."""
    substrate(LIVE_READING)

    line = self_state_report._affect_line()

    for drive in ("energy", "curiosity", "frustration", "focus"):
        assert drive in line, drive


def test_mood_duration_is_declared_unreadable_rather_than_estimated(substrate):
    """The exact hole "02:15 hours ago" was invented to fill.

    Nothing timestamps a mood change, so the honest report is that it cannot
    be told — not a plausible-looking number.
    """
    substrate(LIVE_READING)

    line = self_state_report._affect_line()

    assert "NOT recorded" in line
    assert "cannot tell" in line


def test_an_unreadable_substrate_says_so_instead_of_going_silent(monkeypatch):
    """Silence in this block is what licenses invention."""
    monkeypatch.setattr(
        "core.runtime.service_registry.get_runtime_service",
        lambda name, default=None: default,
    )

    line = self_state_report._affect_line()

    assert "not readable" in line
    assert "do not describe a mood you did not measure" in line


def test_a_stale_reading_is_labelled_stale(substrate):
    substrate({**LIVE_READING, "snapshot_stale": True, "snapshot_age_s": 42.0})

    line = self_state_report._affect_line()

    assert "stale" in line
    assert "42" in line


def test_a_fresh_reading_makes_no_staleness_claim(substrate):
    substrate({**LIVE_READING, "snapshot_stale": False, "snapshot_age_s": 0.2})

    assert "stale" not in self_state_report._affect_line()


def test_the_report_carries_the_affect_line(substrate):
    """Wiring: the line must reach the block she actually reads."""
    substrate(LIVE_READING)

    report = self_state_report.runtime_self_report()

    assert "TIRED" in report
    assert self_state_report.SELF_STATE_HEADER in report


def test_a_partial_reading_reports_what_it_has(substrate):
    """A missing drive must not discard the mood, or vice versa."""
    substrate({"mood": "INQUISITIVE"})

    line = self_state_report._affect_line()

    assert "INQUISITIVE" in line
    assert "not readable" not in line
