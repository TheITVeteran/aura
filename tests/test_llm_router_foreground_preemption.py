"""Foreground turns preempt background generations cooperatively.

Live incident: the conversation lane sat cold for the full 75s gate window
behind a background 32B generation, then force-aborted it — killing the
warm worker and paying a model reload. The preemption ladder must instead
soft-cancel the background holder after a short grace (the worker yields
between tokens and stays warm) and only fall back to the old escalation.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from core.brain import llm_health_router as router_module
from core.brain.llm_health_router import HealthAwareLLMRouter


GATED_OK = {"ok": True, "text": "reply", "endpoint": "local", "tokens": 3}


@pytest.fixture()
def gate_state(monkeypatch):
    """Fresh single-slot gate with clean lease bookkeeping."""
    gate = threading.BoundedSemaphore(1)
    monkeypatch.setattr(router_module, "_GENERATION_GATE", gate)
    monkeypatch.setattr(router_module, "_GENERATION_GATE_ACTIVE_LEASES", {})
    monkeypatch.setattr(router_module, "_GENERATION_GATE_FORCED_LEASES", set())
    monkeypatch.setattr(router_module, "_GENERATION_GATE_NEXT_LEASE_ID", 0)
    # Small budgets so tests run in milliseconds while preserving ordering.
    monkeypatch.setattr(router_module, "_FOREGROUND_GATE_GRACE_S", 0.05)
    monkeypatch.setattr(router_module, "_FOREGROUND_SOFT_CANCEL_WAIT_S", 0.5)
    monkeypatch.setattr(router_module, "_GENERATION_GATE_WAIT_S", 0.3)
    return gate


def _router(monkeypatch) -> HealthAwareLLMRouter:
    router = HealthAwareLLMRouter()

    async def fake_gated(*args, **kwargs):
        return dict(GATED_OK)

    monkeypatch.setattr(router, "_generate_with_metadata_gated", fake_gated)
    monkeypatch.setattr(
        router, "_background_suppression_result", lambda **_kw: None
    )
    return router


def _hold_gate_as(gate, owner: str) -> int:
    assert gate.acquire(False) is True
    return router_module._mark_generation_gate_acquired(owner)


def test_foreground_soft_cancels_background_holder(monkeypatch, gate_state):
    """Rung 2: background holder → soft-cancel → gate freed → no force-abort."""
    router = _router(monkeypatch)
    lease = _hold_gate_as(gate_state, "stream_narrative:background")

    cancel_calls: list[str] = []

    def fake_soft_cancel(*, reason: str) -> bool:
        cancel_calls.append(reason)
        # Simulate the worker yielding between tokens: the background
        # generation completes early and releases its lease.
        router_module._release_generation_gate_after_call(lease)
        return True

    monkeypatch.setattr(router, "_soft_cancel_local_generations", fake_soft_cancel)
    force_aborts: list[str] = []
    monkeypatch.setattr(
        router,
        "force_abort_active_generation",
        lambda reason="": force_aborts.append(reason) or 0,
    )

    result = asyncio.run(
        router.generate_with_metadata(
            "hello", origin="user", purpose="response_generation_user",
            foreground_request=True,
        )
    )

    assert result == GATED_OK
    assert len(cancel_calls) == 1
    assert "foreground_preempts_background" in cancel_calls[0]
    assert "stream_narrative:background" in cancel_calls[0]
    assert force_aborts == [], "cooperative yield must not escalate to worker kill"
    # Lease bookkeeping drained: nothing left holding the gate.
    assert router_module._GENERATION_GATE_ACTIVE_LEASES == {}


def test_foreground_never_soft_cancels_foreground_holder(monkeypatch, gate_state):
    """A user's active turn is sacred: no cancel, fall through to busy result."""
    router = _router(monkeypatch)
    _hold_gate_as(gate_state, "user:response_generation_user")

    cancel_calls: list[str] = []
    monkeypatch.setattr(
        router,
        "_soft_cancel_local_generations",
        lambda *, reason: cancel_calls.append(reason) or True,
    )

    result = asyncio.run(
        router.generate_with_metadata(
            "hello again", origin="user", purpose="response_generation_user",
            foreground_request=True,
        )
    )

    assert cancel_calls == [], "must never cancel an active user foreground turn"
    assert result["ok"] is False
    assert result["endpoint"] == "generation_gate_busy_foreground"


def test_background_request_gets_no_preemption_ladder(monkeypatch, gate_state):
    """Background requests keep the plain full-window wait: no soft-cancel."""
    router = _router(monkeypatch)
    _hold_gate_as(gate_state, "stream_narrative:background")

    cancel_calls: list[str] = []
    monkeypatch.setattr(
        router,
        "_soft_cancel_local_generations",
        lambda *, reason: cancel_calls.append(reason) or True,
    )
    monkeypatch.setattr(
        router, "force_abort_active_generation", lambda reason="": 0
    )

    result = asyncio.run(
        router.generate_with_metadata(
            "idle musing", origin="stream_narrative", is_background=True
        )
    )

    assert cancel_calls == [], "background traffic must not preempt anything"
    assert result["ok"] is False


def test_failed_soft_cancel_falls_back_to_existing_escalation(monkeypatch, gate_state):
    """Rung 3: cancel refused → remaining wait → existing force-abort path."""
    router = _router(monkeypatch)
    lease = _hold_gate_as(gate_state, "stream_narrative:background")

    monkeypatch.setattr(
        router, "_soft_cancel_local_generations", lambda *, reason: False
    )

    def fake_force_abort(reason: str = "") -> int:
        router_module._release_generation_gate_after_call(lease)
        return 1

    monkeypatch.setattr(router, "force_abort_active_generation", fake_force_abort)

    result = asyncio.run(
        router.generate_with_metadata(
            "hello", origin="user", purpose="response_generation_user",
            foreground_request=True,
        )
    )

    assert result == GATED_OK, "escalation ladder must still recover the gate"


def test_uncontended_foreground_takes_gate_without_cancel(monkeypatch, gate_state):
    router = _router(monkeypatch)
    cancel_calls: list[str] = []
    monkeypatch.setattr(
        router,
        "_soft_cancel_local_generations",
        lambda *, reason: cancel_calls.append(reason) or True,
    )

    result = asyncio.run(
        router.generate_with_metadata(
            "hello", origin="user", purpose="response_generation_user",
            foreground_request=True,
        )
    )

    assert result == GATED_OK
    assert cancel_calls == []
    assert router_module._GENERATION_GATE_ACTIVE_LEASES == {}
