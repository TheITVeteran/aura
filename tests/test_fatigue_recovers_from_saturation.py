"""Rest must actually restore (2026-07-25 idle window).

A clean idle hour — zero failures, zero criticals, nothing charging a
consequence — still showed ``body_fatigue`` pinned at 0.996. Saturated
fatigue holds ``welfare.recovery_drive`` above the Will's defer threshold,
so the log carried a standing
``welfare_recovery_required_before_action`` storm: her own belief updates,
memory writes and initiative deferred during a completely quiet stretch.

The mechanism is that a FLAT decay can be exactly cancelled by the
ordinary drip of idle-loop body costs. That is the same class the earlier
0.002 → 0.01 retune addressed, and a constant only postpones it. Recovery
now scales with fatigue, so deep fatigue always escapes saturation while
ordinary dynamics are untouched.
"""
from __future__ import annotations

import time

import pytest

from core.being.body_state_service import BodyStateService

pytestmark = pytest.mark.unit


@pytest.fixture()
def service():
    BodyStateService.reset()
    svc = BodyStateService.get()
    yield svc
    BodyStateService.reset()


def _rest(svc: BodyStateService, seconds: float) -> float:
    svc._last_decay_time = time.monotonic() - seconds
    return svc.snapshot().fatigue


class TestRestRestores:
    def test_saturated_fatigue_recovers_in_a_quiet_minute(self, service):
        service._metabolic.fatigue = 1.0
        assert _rest(service, 60.0) < 0.4, "a quiet minute must make real progress"

    def test_saturation_escapes_a_constant_idle_drip(self, service):
        """Replay the observed shape: fatigue at 0.996 while an idle drip
        charges at exactly the old flat decay rate. Recovery must still win."""
        service._metabolic.fatigue = 0.996
        drip = service._fatigue_decay_rate  # per second, cancels a flat rate

        for _ in range(30):  # 30 x 2s of quiet cognition
            service._metabolic.fatigue = min(
                1.0, service._metabolic.fatigue + drip * 2.0
            )
            _rest(service, 2.0)

        assert service._metabolic.fatigue < 0.9, (
            "proportional recovery must beat a constant drip, or fatigue is a "
            "ratchet and her maintenance work is deferred forever"
        )

    def test_low_fatigue_dynamics_are_not_distorted(self, service):
        """The fix must not make her implausibly tireless at low fatigue."""
        service._metabolic.fatigue = 0.1
        after = _rest(service, 10.0)
        assert 0.0 <= after < 0.1
        assert after > 0.0, "a brief rest should not erase all light fatigue"

    def test_fatigue_still_accumulates_under_real_load(self, service):
        """Recovery is not immunity: sustained work still tires her."""
        service._metabolic.fatigue = 0.0
        service._last_decay_time = time.monotonic()
        for _ in range(40):
            service._commit_cost_locked(
                {"fatigue": 0.02}, receipt_id=f"work-{_}"
            )
        assert service._metabolic.fatigue > 0.5

    def test_recovery_never_drives_fatigue_negative(self, service):
        service._metabolic.fatigue = 0.05
        assert _rest(service, 600.0) == 0.0
