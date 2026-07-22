"""Cortex warmup backoff — stop the reload thrash from starving the fallback.

Live 2026-07-15 soak: under thermal throttle + GPU contention a 32B load
overran its deadline, got force-killed, and was re-spawned immediately —
thrashing the single GPU slot with repeated 20GB weight loads and starving the
foreground fallback that was actually serving the turn (210s walls). After a
cluster of stuck-load kills the gate must cool down: defer warmup so the
resident fallback carries smoothly, then take one clean reload shot once
thermal recovers. A successful serve clears the cooldown.
"""
from __future__ import annotations

import time
from collections import deque

import pytest

from core.brain.inference_gate import InferenceGate


def _bare_gate() -> InferenceGate:
    gate = InferenceGate.__new__(InferenceGate)
    gate._cortex_stuck_kill_times = deque(maxlen=16)
    gate._cortex_warmup_backoff_until = 0.0
    gate._cortex_warmup_backoff_streak = 0
    return gate


@pytest.fixture(autouse=True)
def _backoff_env(monkeypatch):
    monkeypatch.setenv("AURA_CORTEX_STUCK_KILL_THRESHOLD", "2")
    monkeypatch.setenv("AURA_CORTEX_STUCK_KILL_WINDOW_S", "300")
    monkeypatch.setenv("AURA_CORTEX_WARMUP_BACKOFF_S", "90")
    monkeypatch.setenv("AURA_CORTEX_WARMUP_BACKOFF_CAP_S", "240")
    monkeypatch.delenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", raising=False)


def test_single_stuck_kill_does_not_arm_backoff():
    gate = _bare_gate()
    gate._note_cortex_stuck_kill()
    assert gate._cortex_warmup_backoff_reason() is None


def test_backoff_arms_at_threshold_and_defers_warmup():
    gate = _bare_gate()
    gate._note_cortex_stuck_kill()
    gate._note_cortex_stuck_kill()
    reason = gate._cortex_warmup_backoff_reason()
    assert reason is not None and reason.startswith("warmup_backoff:")
    # The deferral reason surfaced to every warmup path short-circuits to the
    # backoff before consulting the memory-admission snapshot, so the turn
    # reroutes to the resident fallback instead of re-spawning the cortex.
    assert gate._cortex_warmup_deferral_reason("foreground") == reason


def test_backoff_escalates_then_caps():
    gate = _bare_gate()
    # First cluster → base cooldown.
    gate._note_cortex_stuck_kill()
    gate._note_cortex_stuck_kill()
    assert gate._cortex_warmup_backoff_streak == 1
    first = float(gate._cortex_warmup_backoff_reason().split(":")[1].rstrip("s"))
    assert 80.0 <= first <= 90.0
    # Expire, thrash again → longer cooldown, capped.
    gate._cortex_warmup_backoff_until = time.monotonic() - 1.0
    gate._note_cortex_stuck_kill()
    gate._note_cortex_stuck_kill()
    gate._note_cortex_stuck_kill()
    capped = float(gate._cortex_warmup_backoff_reason().split(":")[1].rstrip("s"))
    assert capped <= 240.0
    assert capped > first


def test_stale_kills_age_out_of_window(monkeypatch):
    gate = _bare_gate()
    # Two kills far in the past should not, on their own, arm a cooldown now.
    old = time.monotonic() - 10_000.0
    gate._cortex_stuck_kill_times.extend([old, old])
    gate._note_cortex_stuck_kill()  # only one recent kill inside the window
    assert gate._cortex_warmup_backoff_reason() is None


def test_successful_serve_clears_backoff():
    gate = _bare_gate()
    gate._note_cortex_stuck_kill()
    gate._note_cortex_stuck_kill()
    assert gate._cortex_warmup_backoff_reason() is not None
    gate._reset_cortex_warmup_backoff()
    assert gate._cortex_warmup_backoff_reason() is None
    assert gate._cortex_warmup_backoff_streak == 0
    assert len(gate._cortex_stuck_kill_times) == 0


def _snapshot(*, available_gb: float, measured: bool = True, can_admit: bool = False):
    """A memory-admission snapshot with a chosen available-RAM reading."""
    return lambda context="background": {
        "context": context,
        "available_gb": available_gb,
        "measured": measured,
        "can_admit": can_admit,
        "reason": "" if can_admit else "memory_pressure:90%/8GB",
    }


def test_force_warmup_env_overrides_backoff(monkeypatch):
    from core.brain.llm import emergency_override

    emergency_override.reset_overrides_for_test()
    gate = _bare_gate()
    gate._note_cortex_stuck_kill()
    gate._note_cortex_stuck_kill()
    assert gate._cortex_warmup_backoff_reason() is not None
    # Plenty of headroom above the survival floor, and the probe is measured.
    monkeypatch.setattr(
        InferenceGate, "_cortex_warmup_admission_snapshot",
        staticmethod(_snapshot(available_gb=32.0)),
    )
    monkeypatch.setenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "1")
    # Operator override wins mid-cooldown — but only above the survival floor.
    assert gate._cortex_warmup_deferral_reason("foreground") is None
    emergency_override.reset_overrides_for_test()


def test_force_warmup_denied_below_survival_floor(monkeypatch):
    """The override may skip soft admission and backoff, never the RAM floor."""
    from core.brain.llm import emergency_override

    emergency_override.reset_overrides_for_test()
    gate = _bare_gate()
    monkeypatch.setattr(
        InferenceGate, "_cortex_warmup_admission_snapshot",
        staticmethod(_snapshot(available_gb=6.0)),  # below the 10GB floor
    )
    monkeypatch.setenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "1")
    reason = gate._cortex_warmup_deferral_reason("foreground")
    assert reason is not None and reason.startswith("forced_warmup_denied_survival_floor:")
    emergency_override.reset_overrides_for_test()


def test_force_warmup_denied_when_memory_unmeasured(monkeypatch):
    """A failed probe means the floor cannot be confirmed → fail closed."""
    from core.brain.llm import emergency_override

    emergency_override.reset_overrides_for_test()
    gate = _bare_gate()
    monkeypatch.setattr(
        InferenceGate, "_cortex_warmup_admission_snapshot",
        staticmethod(_snapshot(available_gb=0.0, measured=False)),
    )
    monkeypatch.setenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "1")
    assert (
        gate._cortex_warmup_deferral_reason("foreground")
        == "forced_warmup_denied_survival_floor_unmeasured"
    )
    emergency_override.reset_overrides_for_test()


def test_force_warmup_override_is_use_bounded(monkeypatch):
    """The override is a bounded decision: after its budget the guard re-arms."""
    from core.brain.llm import emergency_override

    emergency_override.reset_overrides_for_test()
    gate = _bare_gate()
    gate._note_cortex_stuck_kill()
    gate._note_cortex_stuck_kill()  # a real deferral the override must bypass
    monkeypatch.setattr(
        InferenceGate, "_cortex_warmup_admission_snapshot",
        staticmethod(_snapshot(available_gb=32.0)),
    )
    monkeypatch.setenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "1")
    bypasses = 0
    for _ in range(emergency_override.DEFAULT_MAX_USES + 4):
        if gate._cortex_warmup_deferral_reason("foreground") is None:
            bypasses += 1
    # The flag cannot authorize an unbounded stream of bypasses.
    assert bypasses == emergency_override.DEFAULT_MAX_USES
    # Once exhausted, the normal deferral (backoff) stands again.
    assert gate._cortex_warmup_deferral_reason("foreground").startswith("warmup_backoff:")
    emergency_override.reset_overrides_for_test()


def test_no_override_needed_when_memory_is_calm(monkeypatch):
    """With headroom and no backoff, warmup is admitted without touching the override."""
    from core.brain.llm import emergency_override

    emergency_override.reset_overrides_for_test()
    gate = _bare_gate()
    monkeypatch.setattr(
        InferenceGate, "_cortex_warmup_admission_snapshot",
        staticmethod(_snapshot(available_gb=32.0, can_admit=True)),
    )
    monkeypatch.setenv("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "1")
    assert gate._cortex_warmup_deferral_reason("foreground") is None
    # The override budget was never spent, since it was never needed.
    status = emergency_override.override_status("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE")
    assert status.get("seen") is False
    emergency_override.reset_overrides_for_test()
