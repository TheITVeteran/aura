"""CP126 hardening contracts for core/soul.py (intrinsic-motivation drives).

Covers the schema-safe subset: finite/bounded surprise input, lock-guarded
state, and surfacing a failed self-diagnosis instead of swallowing it. The
wall-clock→monotonic drive-timing fix (522aad80) is deliberately left as
follow-up (it needs a None-sentinel refactor verified against the consciousness
suite).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import core.soul as soul_mod
from core.soul import Drive, Soul, _finite_surprise

# ── 5124cb9f: surprise is finite and bounded ───────────────────────────────


@pytest.mark.parametrize(
    "value,expected",
    [(math.nan, 0.0), (math.inf, 0.0), (-math.inf, 0.0), (2.0, 1.0), (-1.0, 0.0), ("x", 0.0), (0.5, 0.5)],
)
def test_finite_surprise(value, expected):
    assert _finite_surprise(value) == expected


def test_nonfinite_surprise_keeps_drive_urgency_finite(monkeypatch):
    monkeypatch.setattr(
        soul_mod, "get_runtime_service",
        lambda name, default=None: SimpleNamespace(get_surprise_signal=lambda: math.inf) if name == "self_prediction" else default,
    )
    s = Soul(SimpleNamespace(boredom=0.1))
    dominant = s.get_dominant_drive()
    assert math.isfinite(dominant.urgency)
    assert 0.0 <= dominant.urgency <= 1.0


# ── 6913714d: state access is lock-guarded ─────────────────────────────────


def test_state_has_a_lock():
    s = Soul(SimpleNamespace(boredom=0.0))
    assert hasattr(s, "_lock") and hasattr(s._lock, "acquire")
    s.update_state("user_message")  # exercises the locked path
    s.update_state("error")


# ── 624dadf6: a failed self-diagnosis is surfaced, not swallowed ───────────


@pytest.mark.asyncio
async def test_failed_competence_diagnosis_is_recorded():
    from core.runtime.errors import get_degradation_tracker

    get_degradation_tracker().reset()

    async def _boom(name, params):
        raise RuntimeError("health probe exploded")

    orch = SimpleNamespace(execute_tool=_boom)
    s = Soul(orch)
    await s.satisfy_drive(Drive("competence", 0.9, "diagnose"))

    recent = get_degradation_tracker().recent(subsystem="soul", limit=1)
    assert recent
    assert "self-diagnosis failed" in recent[0].action
