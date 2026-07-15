"""Regression: the cascade-cleanup path must not force-kill a cortex worker
that is legitimately LOADING the model.

Lived 2026-07-15: a 200-turn soak found the 32B cortex starved for a full
hour — every turn's exhaustion path killed the worker mid-warmup (spawn →
load → killed on the next turn → warmup_deferred → repeat, 216s/turn, zero
real cortex answers). Only genuinely wedged workers may be killed.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from core.brain.inference_gate import InferenceGate

pytestmark = pytest.mark.unit

_check = InferenceGate._cortex_worker_is_legitimately_loading


def test_warming_worker_within_deadline_is_protected():
    client = SimpleNamespace(
        _warmup_in_flight=True,
        _lane_state="warming",
        _lane_transition_at=time.time() - 5.0,  # started 5s ago, mid-load
    )
    assert _check(client) is True


def test_recovering_lane_within_deadline_is_protected():
    client = SimpleNamespace(
        _warmup_in_flight=False,
        _lane_state="recovering",
        _lane_transition_at=time.time() - 30.0,
    )
    assert _check(client) is True


def test_wedged_warming_worker_past_deadline_is_killable():
    client = SimpleNamespace(
        _warmup_in_flight=True,
        _lane_state="warming",
        _lane_transition_at=time.time() - 500.0,  # far past the 200s deadline
    )
    assert _check(client) is False


def test_idle_running_worker_is_killable():
    # Not warming, not recovering: a running-but-idle worker is the original
    # nwait-wedge bug — killing it is correct.
    client = SimpleNamespace(
        _warmup_in_flight=False,
        _lane_state="ready",
        _lane_transition_at=time.time(),
    )
    assert _check(client) is False


def test_none_client_is_not_loading():
    assert _check(None) is False


def test_deadline_is_env_tunable(monkeypatch):
    monkeypatch.setenv("AURA_CORTEX_LOAD_DEADLINE_S", "10")
    client = SimpleNamespace(
        _warmup_in_flight=True,
        _lane_state="warming",
        _lane_transition_at=time.time() - 30.0,  # 30s > 10s tuned deadline
    )
    assert _check(client) is False
