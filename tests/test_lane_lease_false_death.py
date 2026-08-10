"""A blocked event loop must not be able to kill a healthy worker.

LIVE DEFECT, 2026-08-10. A user turn returned "I couldn't get to an answer I'd
stand behind on that one." The health pulse for that second::

    conversation_lane: cold (worker_not_alive)
    inference_gate (is_inference_ready() returned False)
    event_loop_monitor.last_lag_s 10.46 >= 5.00
    mlx_client (critical): TimeoutError
        -> stopped MLX worker after durable lane heartbeat failed

Nothing was wrong with the worker or the lease. The loop had been blocked for
ten seconds, so a renewal awaited with a five second budget measured on that
loop never got scheduled. The timeout was read as a lost fence, which killed a
healthy 32B and failed every in-flight turn.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain.llm.mlx_client import MLXLocalClient


class _Controller:
    """A lane controller whose renewals fail a given number of times first."""

    def __init__(self, *, timeouts: int, alive_after: bool = True):
        self.timeouts = timeouts
        self.alive_after = alive_after
        self.calls = 0

    async def heartbeat_owner(self, owner_id, *, fencing_token):
        self.calls += 1
        if self.calls <= self.timeouts:
            # Stand in for a renewal the loop never got round to running.
            await asyncio.sleep(3600)
        return self.alive_after


def _client() -> MLXLocalClient:
    return MLXLocalClient.__new__(MLXLocalClient)


@pytest.mark.asyncio
async def test_a_starved_loop_does_not_kill_a_live_lease(monkeypatch):
    """The exact live case: one timeout, then the truth."""
    monkeypatch.setattr("core.brain.llm.mlx_client._LEASE_RENEWAL_TIMEOUT_S", 0.01)
    controller = _Controller(timeouts=1, alive_after=True)

    alive = await _client()._renew_durable_lane_lease(controller, "owner-1", 7)

    assert alive is True
    assert controller.calls == 2, "the renewal must be re-asked, not assumed dead"


@pytest.mark.asyncio
async def test_a_genuinely_lost_lease_still_reports_lost(monkeypatch):
    """The retry must not paper over a real fence loss."""
    monkeypatch.setattr("core.brain.llm.mlx_client._LEASE_RENEWAL_TIMEOUT_S", 0.01)
    controller = _Controller(timeouts=1, alive_after=False)

    alive = await _client()._renew_durable_lane_lease(controller, "owner-1", 7)

    assert alive is False
    assert controller.calls == 2


@pytest.mark.asyncio
async def test_a_loop_that_never_recovers_still_fails_closed(monkeypatch):
    """One retry, not a loop.

    A second timeout is itself evidence the loop is not recovering, and a
    foreground turn cannot wait indefinitely to find out.
    """
    monkeypatch.setattr("core.brain.llm.mlx_client._LEASE_RENEWAL_TIMEOUT_S", 0.01)
    controller = _Controller(timeouts=99)

    with pytest.raises(TimeoutError):
        await _client()._renew_durable_lane_lease(controller, "owner-1", 7)

    assert controller.calls == 2


@pytest.mark.asyncio
async def test_a_healthy_lease_is_renewed_without_a_second_call(monkeypatch):
    """No extra round trip on the ordinary path."""
    monkeypatch.setattr("core.brain.llm.mlx_client._LEASE_RENEWAL_TIMEOUT_S", 5.0)
    controller = _Controller(timeouts=0, alive_after=True)

    assert await _client()._renew_durable_lane_lease(controller, "owner-1", 7) is True
    assert controller.calls == 1
