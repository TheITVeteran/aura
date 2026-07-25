"""Regression: the serving-latency ceiling (2026-07-18 soak).

Measured anatomy — 65/200 turns in 180 min, p50 167 s, p95 210 s, deaths 0,
and **32 turns that produced no reply at all** while a 1.5B reflex sat ready:

1. Repeated 32B stuck-load kills put the cortex lane in warmup BACKOFF with
   reason ``foreground_warmup_deferred_memory_pressure`` — the runtime had
   deliberately decided not to load the cortex.
2. ``_foreground_timeout_for_lane`` read only the lane STATE (``recovering``)
   and granted the 210 s cold-boot budget anyway.
3. The cortex tier then waited on the global spawn gate up to its fixed 330 s
   process bound — longer than the entire turn — because the gate wait
   ignored the caller's own deadline.
4. The escalation ladder is SERIAL, so the Brainstem inherited 56 s and the
   Reflex 14.5 s. Neither could load-and-generate in a scrap. No tier
   answered: ``canonical_chat_no_reply``.

Fixed here: the turn budget follows what the lane is actually DOING, so a
deferred cortex stops consuming a cold-boot budget and the ladder inherits
real time.

The spawn gate additionally gained the ability to bound its wait by the
caller's budget (``timeout_s``), which is pinned below — but that bound is
deliberately NOT wired into the live model-load path yet: a short scoped
wait moves other paths onto the timeout branch, and one of those leaves the
durable model-lane owner unreconciled (lane FENCED, admission blocked),
which is the lease-outlives-holder failure in a new costume. Wiring it needs
the durable-owner path made timeout-safe first.
"""
from __future__ import annotations

import asyncio
import time

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Half 1: a deliberately-deferred lane must not get the cold-boot budget
# ---------------------------------------------------------------------------

def _lane(state: str, reason: str = "", *, ready: bool = False) -> dict:
    return {
        "state": state,
        "conversation_ready": ready,
        "last_failure_reason": reason,
        "desired_model": "Cortex (32B)",
    }


class TestDeferredLaneBudget:
    def test_genuine_cold_boot_still_gets_the_full_budget(self):
        """A cortex actually warming toward ready has earned the wait."""
        from interface.routes import chat as chat_routes

        assert chat_routes._foreground_timeout_for_lane(_lane("cold")) == 210.0
        assert chat_routes._foreground_timeout_for_lane(_lane("warming")) == 210.0
        assert (
            chat_routes._foreground_timeout_for_lane(_lane("spawning", "spawn_in_flight"))
            == 210.0
        )

    def test_backoff_deferred_lane_yields_the_budget_to_the_ladder(self):
        """The exact soak reason string must shorten the turn budget."""
        from interface.routes import chat as chat_routes

        budget = chat_routes._foreground_timeout_for_lane(
            _lane("recovering", "foreground_warmup_deferred_memory_pressure")
        )
        assert budget == chat_routes._DEFERRED_CORTEX_TURN_TIMEOUT_S
        assert budget < 210.0

    @pytest.mark.parametrize(
        "reason",
        [
            "warmup_backoff:180s",
            "foreground_warmup_deferred:warmup_backoff:120s",
            "foreground_warmup_timeout",
            "foreground_warmup_deferred_memory_pressure",
        ],
    )
    def test_every_deferral_reason_class_is_recognized(self, reason):
        from interface.routes import chat as chat_routes

        assert chat_routes._lane_warmup_is_deliberately_deferred(_lane("recovering", reason))
        assert (
            chat_routes._foreground_timeout_for_lane(_lane("recovering", reason))
            == chat_routes._DEFERRED_CORTEX_TURN_TIMEOUT_S
        )

    def test_progressing_reasons_are_not_treated_as_deferral(self):
        """Only DELIBERATE hold-offs shorten the budget — a lane mid-load or
        recovering from an ordinary fault still gets the cold-boot room."""
        from interface.routes import chat as chat_routes

        for reason in ("", "worker_died_unexpectedly", "handshake_in_progress"):
            assert not chat_routes._lane_warmup_is_deliberately_deferred(
                _lane("recovering", reason)
            )
            assert chat_routes._foreground_timeout_for_lane(_lane("recovering", reason)) == 210.0

    def test_deferred_budget_still_allows_a_real_fallback_answer(self):
        """The point is a genuine lower-rung answer, not a faster apology:
        the budget must comfortably exceed a small-model load+generate."""
        from interface.routes import chat as chat_routes

        assert chat_routes._DEFERRED_CORTEX_TURN_TIMEOUT_S >= 60.0

    def test_ready_lane_is_unaffected(self):
        from interface.routes import chat as chat_routes

        ready = chat_routes._foreground_timeout_for_lane(
            _lane("ready", "warmup_backoff:90s", ready=True)
        )
        assert ready != chat_routes._DEFERRED_CORTEX_TURN_TIMEOUT_S


# ---------------------------------------------------------------------------
# Half 2: the spawn gate must never be waited past the caller's own deadline
# ---------------------------------------------------------------------------

class TestSpawnGateHonorsCallerDeadline:
    def test_wait_is_bounded_by_the_caller_budget(self):
        """A tier with 2s left must fail in ~2s, not the 330s process bound —
        the ladder needs the remaining time more than this tier does."""
        from core.brain.llm import mlx_client

        async def _scenario() -> float:
            async with mlx_client._spawn_gate_context(owner="holder"):
                started = time.monotonic()
                with pytest.raises(TimeoutError) as excinfo:
                    async with mlx_client._spawn_gate_context(
                        owner="starved_fallback", timeout_s=2.0
                    ):
                        pass  # pragma: no cover - must never be reached
                waited = time.monotonic() - started
                assert "spawn_gate_timeout:2." in str(excinfo.value)
                assert "holder=holder" in str(excinfo.value)
                return waited

        waited = asyncio.run(_scenario())
        assert waited < 10.0, f"caller-scoped gate wait ran {waited:.1f}s"

    def test_default_keeps_the_process_bound(self):
        from core.brain.llm import mlx_client

        async def _scenario() -> None:
            async with mlx_client._spawn_gate_context(owner="holder"):
                # No caller budget: the long process-level bound still applies,
                # so a legitimate cold boot is never cut short.
                task = asyncio.ensure_future(
                    _acquire(mlx_client, owner="patient", timeout_s=None)
                )
                await asyncio.sleep(0.2)
                assert not task.done(), "unbounded waiter must keep waiting"
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, TimeoutError):
                    pass

        async def _acquire(module, *, owner: str, timeout_s: float | None) -> None:
            async with module._spawn_gate_context(owner=owner, timeout_s=timeout_s):
                pass

        asyncio.run(_scenario())

    def test_gate_is_released_after_a_scoped_timeout(self):
        """A timed-out waiter must not leave the semaphore or its ownership
        metadata poisoned for the next tier."""
        from core.brain.llm import mlx_client

        async def _scenario() -> None:
            async with mlx_client._spawn_gate_context(owner="holder"):
                with pytest.raises(TimeoutError):
                    async with mlx_client._spawn_gate_context(
                        owner="loser", timeout_s=0.3
                    ):
                        pass  # pragma: no cover
            # Holder released: the gate must be immediately acquirable again.
            async with mlx_client._spawn_gate_context(
                owner="next_tier", timeout_s=5.0
            ) as snapshot:
                assert snapshot["owner"] == "next_tier"

        asyncio.run(_scenario())

    def test_zero_budget_fails_immediately_without_acquiring(self):
        from core.brain.llm import mlx_client

        async def _scenario() -> None:
            async with mlx_client._spawn_gate_context(owner="holder"):
                started = time.monotonic()
                with pytest.raises(TimeoutError):
                    async with mlx_client._spawn_gate_context(
                        owner="no_budget", timeout_s=0.0
                    ):
                        pass  # pragma: no cover
                assert time.monotonic() - started < 2.0

        asyncio.run(_scenario())
