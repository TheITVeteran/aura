"""Regression: ensure_foreground_ready must cap the warmup wait for a
RECOVERING cortex, so a turn falls to the fast fallback instead of blocking
the full cold-boot budget on a lane that is merely re-warming.

Lived 2026-07-15 paced soak: the chat caller passed a 180s budget, and
ensure_foreground_ready honored it — every turn blocked ~206s on a
recovering cortex ("Protected foreground lane failed (lane_warming): Cortex
timed out after 206s"), the 200-turn wall. The recovery cap (15s when the
lane was EVER ready) existed but was not wired into this path.
"""
from __future__ import annotations

import inspect

import pytest

from core.brain.inference_gate import InferenceGate

pytestmark = pytest.mark.unit

_cap = InferenceGate._foreground_warmup_timeout


def _gate():
    return InferenceGate.__new__(InferenceGate)


def test_recovery_lane_caps_short():
    # last_ready_at > 0 => the cortex WAS ready and is now re-warming.
    lane = {"last_ready_at": 123.0, "state": "warming"}
    assert _cap(_gate(), lane, 180.0) == pytest.approx(15.0)


def test_cold_lane_keeps_full_budget():
    lane = {"last_ready_at": 0.0, "state": "cold"}
    assert _cap(_gate(), lane, 180.0) >= 180.0


def test_ensure_foreground_ready_applies_the_cap():
    # Source pin: the min(timeout, cap) must be present so a large passed
    # timeout can't override the recovery cap.
    src = inspect.getsource(InferenceGate.ensure_foreground_ready)
    assert "self._foreground_warmup_timeout(lane, timeout)" in src
    assert "min(timeout" in src
