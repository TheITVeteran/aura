"""A resting state is not an emergency, and a resident model is not an OOM.

The 2026-07-25 verification run had zero degradations, zero incidents and every
probe passing — and still blocked 15 ``STATE_MUTATION`` requests on
``neurochemical_cortisol_crisis``, fourteen of them from the adaptive immune
system.

The driver: a resident 32B holds ~20GB against a 40GB existential limit, so
``memory_threat`` rests near 0.5 forever by design. The heartbeat's flat
``threat > 0.2`` gate turned that resting level into a cortisol surge on every
tick; surges outrun the tonic return and eventually cross the 0.85 crisis line.
Holding the model resident is the architecture working.

This is the same distinction ``existential_stakes`` already draws for CPU and
loop lag ("NORMAL during heavy 32B generation"), applied to the resting memory
footprint: what signals danger is the RISE toward the limit, plus any level
high enough to be dangerous on its own.
"""
from __future__ import annotations

import pytest

from core.consciousness.heartbeat import (
    _THREAT_ALWAYS_SIGNAL,
    _THREAT_RISE_TO_SIGNAL,
)

pytestmark = pytest.mark.unit


class Signaller:
    """Replays the heartbeat's gate over a threat series."""

    def __init__(self):
        self._last_signalled_threat = 0.0
        self.surges: list[float] = []

    def tick(self, threat: float) -> None:
        previous = float(self._last_signalled_threat)
        rising = threat >= previous + _THREAT_RISE_TO_SIGNAL
        dangerous = threat >= _THREAT_ALWAYS_SIGNAL
        if threat > 0.2 and (rising or dangerous):
            self.surges.append(threat)
        self._last_signalled_threat = (
            threat if (rising or dangerous) else min(previous, threat)
        )


class TestASteadyStateIsNotAnEmergency:
    def test_a_resident_model_does_not_surge_every_tick(self):
        """The live shape: memory_threat parked at 0.5 for the whole session."""
        s = Signaller()
        for _ in range(200):
            s.tick(0.50)
        assert len(s.surges) == 1, (
            "a stable resting footprint must announce itself once, not two "
            f"hundred times (got {len(s.surges)} surges)"
        )

    def test_small_wobble_around_a_resting_level_is_not_acute(self):
        s = Signaller()
        for i in range(200):
            s.tick(0.50 + (0.02 if i % 2 else -0.02))
        assert len(s.surges) <= 2


class TestRealDangerStillSignals:
    def test_a_rise_toward_the_limit_signals(self):
        s = Signaller()
        for threat in (0.30, 0.45, 0.60, 0.72):
            s.tick(threat)
        assert len(s.surges) == 4, "a climb toward the limit is exactly the signal"

    def test_a_dangerous_level_signals_every_tick(self):
        """Near the limit, sustained is worse than rising — never go quiet."""
        s = Signaller()
        for _ in range(20):
            s.tick(0.90)
        assert len(s.surges) == 20

    def test_a_slow_climb_still_trips_the_rise_test(self):
        s = Signaller()
        for i in range(40):
            s.tick(0.30 + i * 0.01)
        assert s.surges, "a slow creep toward the limit must not be invisible"

    def test_recovery_re_arms_the_signal(self):
        s = Signaller()
        s.tick(0.60)
        for _ in range(10):
            s.tick(0.20)  # pressure released
        s.tick(0.45)
        assert len(s.surges) == 2, "a threat that returns must be signalled again"

    def test_a_quiet_runtime_never_signals(self):
        s = Signaller()
        for _ in range(100):
            s.tick(0.05)
        assert s.surges == []
