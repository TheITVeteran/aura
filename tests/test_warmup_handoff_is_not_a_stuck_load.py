"""The mechanism that prevents GPU thrash must not be counted as GPU thrash.

The 2026-07-25 endurance probe attempted 62 cortex loads and completed none,
and 173 of 200 turns went unanswered. The chain:

  1. ``ensure_foreground_ready`` caps a RECOVERY wait at 15s. That is a
     designed handoff — the warmup task is shielded, the cortex keeps loading
     in the background, and this turn falls to the ready fallback.
  2. The ``TimeoutError`` from that deliberate 15s cap called
     ``_note_cortex_stuck_kill()``, the same counter a force-killed stuck load
     feeds.
  3. At the threshold, the backoff deferred cortex warmup for 240s.
  4. So the load never finished, the lane was never ready, so the next turn
     was also a "recovery", so it timed out at 15s again, forever.

A deliberate deferral counted as damage — the same category error as the
admission-backpressure degradations, one layer down.
"""
from __future__ import annotations

import inspect
import re

import pytest

pytestmark = pytest.mark.unit


def _ensure_foreground_ready_source() -> str:
    from core.brain.inference_gate import InferenceGate

    return inspect.getsource(InferenceGate.ensure_foreground_ready)


class TestTheHandoffIsDistinguished:
    def test_a_recovery_handoff_is_identified_from_lane_history(self):
        src = _ensure_foreground_ready_source()
        assert re.search(
            r"recovery_handoff\s*=.*last_ready_at", src
        ), "the handoff case must be derived from whether the lane was ever ready"

    def test_the_stuck_load_counter_is_guarded_by_it(self):
        src = _ensure_foreground_ready_source()
        guarded = re.search(
            r"if not recovery_handoff:\s*\n\s*self\._note_cortex_stuck_kill\(\)", src
        )
        assert guarded, (
            "a designed 15s handoff must not feed the stuck-load backoff that "
            "then prevents the load it was waiting for"
        )

    def test_a_cold_boot_timeout_still_counts(self):
        """The backoff must keep working for the case it exists for."""
        src = _ensure_foreground_ready_source()
        assert "_note_cortex_stuck_kill()" in src, (
            "a cold load that overruns its full budget is still a stuck load"
        )

    def test_the_handoff_is_visible_in_the_log(self):
        src = _ensure_foreground_ready_source()
        assert "recovery handoff" in src.lower()


class TestTheRecoveryCapItself:
    def test_a_lane_that_was_ready_gets_the_short_cap(self):
        from core.brain.inference_gate import InferenceGate

        gate = InferenceGate.__new__(InferenceGate)
        assert gate._foreground_warmup_timeout({"last_ready_at": 1.0}, 180.0) == 15.0

    def test_a_cold_lane_gets_the_full_budget(self):
        from core.brain.inference_gate import InferenceGate

        gate = InferenceGate.__new__(InferenceGate)
        assert gate._foreground_warmup_timeout({"last_ready_at": 0.0}, 180.0) >= 180.0

    def test_a_cold_lane_respects_a_larger_caller_budget(self):
        from core.brain.inference_gate import InferenceGate

        gate = InferenceGate.__new__(InferenceGate)
        assert gate._foreground_warmup_timeout({}, 300.0) == 300.0
