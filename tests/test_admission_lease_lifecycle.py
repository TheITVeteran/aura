"""Regression: admission leases must die with their holders (2026-07-15 P0).

The soak's serving-path P0: a MODEL_LOAD lease conflicts with every other
MODEL_LOAD lease, and lease lifetime was tied only to a TTL — never to the
holder's life. A cortex-load worker killed mid-load left its lease standing;
every recovery load then burned its own ``timeout_s`` against that wall and
returned ``resource_timeout``; the K1 reconciler retried into the same wall
all night while RAM sat at 40% and the ladder answered every turn at the
216s timeout wall (deaths=0, latency unusable).

The contract now: the worker-death seam reaps the dead holder's lease, so
the next foreground cortex load is admitted within one poll interval — not
one TTL.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from core.runtime.control_plane import (
    AdmissionPriority,
    AdmissionRequest,
    PressureSnapshot,
    ResourceAdmissionController,
    WorkClass,
)

pytestmark = pytest.mark.unit


def _controller() -> ResourceAdmissionController:
    return ResourceAdmissionController(
        pressure_provider=lambda: PressureSnapshot(),
        poll_interval_s=0.02,
    )


def _load_request(*, lane: str = "cortex", owner: str = "mlx.model_load:cortex",
                  timeout_s: float = 0.4, ttl_s: float = 600.0) -> AdmissionRequest:
    return AdmissionRequest(
        owner=owner,
        work_class=WorkClass.MODEL_LOAD,
        lane=lane,
        priority=AdmissionPriority.FOREGROUND,
        timeout_s=timeout_s,
        lease_ttl_s=ttl_s,
    )


class TestDeadHolderReaping:
    def test_replays_the_soak_anatomy(self):
        """Holder dies unreleased → without the reap the retry times out
        against the orphan; with the reap the next load admits within one
        poll interval, not one TTL."""

        async def _scenario() -> None:
            admission = _controller()
            first = await admission.acquire(_load_request(owner="mlx.model_load:attempt-1"))
            assert first.admitted
            # The holder dies without releasing (worker killed mid-load).

            # Precondition — the P0 itself: the orphan walls the retry into
            # resource_timeout for the full lease TTL.
            walled = await admission.acquire(
                _load_request(owner="mlx.model_load:attempt-2")
            )
            assert not walled.admitted
            assert walled.reason == "resource_timeout"

            # The fix: the worker-death seam reaps the dead holder's lease…
            reaped = admission.reap_dead_holder_leases_sync(
                lane="cortex", reason="process_died_unexpectedly"
            )
            assert reaped == 1

            # …and the next foreground cortex load is admitted promptly.
            started = time.monotonic()
            recovered = await admission.acquire(
                _load_request(owner="mlx.model_load:attempt-3", timeout_s=2.0)
            )
            assert recovered.admitted
            assert time.monotonic() - started < 1.0, (
                "admission after reap must take one poll interval, not one TTL"
            )

        asyncio.run(_scenario())

    def test_reap_is_lane_scoped(self):
        async def _scenario() -> None:
            admission = _controller()
            healthy = await admission.acquire(_load_request(lane="solver", owner="mlx.model_load:solver"))
            assert healthy.admitted
            assert admission.reap_dead_holder_leases_sync(lane="cortex") == 0
            assert admission.active_lease_count(WorkClass.MODEL_LOAD) == 1
            # The healthy lane's own release still works normally.
            released = await admission.release(healthy.lease_id, reason="completed")
            assert released.outcome.value == "released"

        asyncio.run(_scenario())

    def test_reap_leaves_history_naming_the_death(self):
        async def _scenario() -> None:
            admission = _controller()
            first = await admission.acquire(_load_request())
            assert first.admitted
            admission.reap_dead_holder_leases_sync(
                lane="cortex", reason="killed_signal_9(likely_oom)"
            )
            entries = [
                entry
                for entry in admission.status().get("history", [])
                if str(entry.get("reason", "")).startswith("holder_died:")
            ]
            assert entries, "the reap must leave an honest history entry"
            assert "killed_signal_9" in entries[-1]["reason"]

        asyncio.run(_scenario())

    def test_dead_holders_late_release_stays_tolerated(self):
        """The dead holder's own finally-release must hit the KeyError path
        the mlx admission context already tolerates — never a new lease."""

        async def _scenario() -> None:
            admission = _controller()
            first = await admission.acquire(_load_request())
            admission.reap_dead_holder_leases_sync(lane="cortex")
            with pytest.raises(KeyError):
                await admission.release(first.lease_id, reason="completed")

        asyncio.run(_scenario())

    def test_reap_never_raises_on_empty_controller(self):
        admission = _controller()
        assert admission.reap_dead_holder_leases_sync(lane="cortex") == 0


class TestWorkerDeathSeam:
    def test_death_note_reaps_the_lane_lease(self, monkeypatch):
        """_note_lane_worker_death must reach the reap with the worker's
        classified lane — the same seam that feeds the K4 breaker."""
        from core.brain.llm import mlx_client as mlx_module

        calls: dict[str, object] = {}

        class _FakeAdmission:
            def reap_dead_holder_leases_sync(self, *, lane, work_class, reason):
                calls.update(lane=lane, work_class=work_class, reason=reason)
                return 1

        monkeypatch.setattr(
            "core.runtime.control_plane.get_runtime_control_plane",
            lambda: SimpleNamespace(admission=_FakeAdmission()),
        )

        client = SimpleNamespace(
            model_path="/models/Aura-32B-cortex-4bit",
            _process_started_at=time.time() - 5.0,
        )
        mlx_module._note_lane_worker_death(client, "process_died_unexpectedly")

        assert calls, "worker death must trigger the lease reap"
        assert calls["work_class"] is WorkClass.MODEL_LOAD
        assert calls["reason"] == "process_died_unexpectedly"
        from core.brain.lane_admission import classify_lane

        expected_lane, _qos = classify_lane(client.model_path)
        assert calls["lane"] == expected_lane

    def test_death_note_never_raises_when_reap_unavailable(self, monkeypatch):
        from core.brain.llm import mlx_client as mlx_module

        def _boom():
            raise RuntimeError("control plane offline")

        monkeypatch.setattr(
            "core.runtime.control_plane.get_runtime_control_plane", _boom
        )
        client = SimpleNamespace(
            model_path="/models/Aura-32B-cortex-4bit",
            _process_started_at=time.time() - 5.0,
        )
        mlx_module._note_lane_worker_death(client, "handshake_failure")  # must not raise
