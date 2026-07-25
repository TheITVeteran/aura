"""A rescue that fires every tick is a standing condition, not housekeeping.

The 2026-07-25 boot logged ``UnifiedField saturation rescue #4212`` with
mean|F| still 0.906 and spectral entropy 0.455. The anti-degeneracy kick was
pulling the field off the rails and the dynamics were putting it straight back,
every tick, for the entire run — a pinned field carrying almost no information.
The log coalesced to one line a minute, so it read as occasional housekeeping.

The guard cannot fix dynamics it has no evidence about, but it must stop
pretending a permanent condition is a series of small recoveries.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture()
def field():
    from core.consciousness.unified_field import UnifiedField

    return UnifiedField()


def _saturate(f) -> None:
    f.F = np.ones_like(f.F, dtype=np.float32) * 0.99


class TestPersistentSaturationIsEscalated:
    def test_a_brief_saturation_is_not_escalated(self, field, monkeypatch):
        recorded: list[dict] = []
        monkeypatch.setattr(
            "core.consciousness.unified_field.record_degradation",
            lambda *a, **kw: recorded.append(kw),
        )

        for _ in range(10):
            _saturate(field)
            field._update_coherence()

        assert recorded == [], "brief saturation is what the rescue is FOR"

    def test_a_permanently_saturated_field_is_reported_once(self, field, monkeypatch):
        recorded: list[dict] = []
        monkeypatch.setattr(
            "core.consciousness.unified_field.record_degradation",
            lambda *a, **kw: recorded.append(kw),
        )

        for _ in range(field._SATURATION_PERSISTENT_TICKS * 3):
            _saturate(field)  # dynamics undo the rescue every tick, as observed
            field._update_coherence()

        assert len(recorded) == 1, "report the standing condition once, not 600 times"
        assert recorded[0]["extra"]["consecutive_ticks"] >= (
            field._SATURATION_PERSISTENT_TICKS
        )
        assert "not restoring the field" in recorded[0]["action"]

    def test_recovery_clears_the_streak_and_re_arms(self, field, monkeypatch):
        recorded: list[dict] = []
        monkeypatch.setattr(
            "core.consciousness.unified_field.record_degradation",
            lambda *a, **kw: recorded.append(kw),
        )

        for _ in range(field._SATURATION_PERSISTENT_TICKS + 5):
            _saturate(field)
            field._update_coherence()
        assert len(recorded) == 1

        # The field recovers: streak resets, escalation re-arms.
        field.F = np.zeros_like(field.F, dtype=np.float32)
        field._update_coherence()
        assert field._consecutive_saturation_ticks == 0
        assert field._saturation_escalated is False

        for _ in range(field._SATURATION_PERSISTENT_TICKS + 5):
            _saturate(field)
            field._update_coherence()
        assert len(recorded) == 2, "a condition that returns must be reported again"

    def test_the_rescue_still_pulls_the_field_off_the_rails(self, field):
        """The escalation must not replace the corrective action."""
        _saturate(field)
        before = float(np.mean(np.abs(field.F)))
        field._update_coherence()
        assert float(np.mean(np.abs(field.F))) < before
