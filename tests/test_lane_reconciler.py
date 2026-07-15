"""Contracts for the K1 lane reconciler + K4 crash-loop breaker.

The doom loop this kills: stall → force-kill → cold reload → stall. Every
spawn SUCCEEDS, so spawn-failure backoff never engages — the signature is
short-lived worker runs. The breaker counts exactly those; the reconciler
is the one control loop that converges the serving lane onto its desired
state instead of five watchdogs fighting imperatively.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain.lane_admission import ActiveLane, QoSClass
from core.runtime.lane_reconciler import (
    CrashLoopBreaker,
    LaneReconciler,
    death_is_deliberate,
    get_crash_loop_breaker,
    get_lane_reconciler,
)
from core.runtime.shutdown_coordinator import clear_shutdown_request, request_shutdown

pytestmark = pytest.mark.unit


class Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def time(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    import core.runtime.lane_reconciler as lr

    c = Clock()
    monkeypatch.setattr(lr, "time", c)
    return c


@pytest.fixture
def breaker():
    return CrashLoopBreaker()


LANE = "/models/Aura-32B-cortex"


class TestDeathClassification:
    def test_administrative_kills_are_deliberate(self):
        for reason in (
            "yield_to_qwen-7b",
            "promoted_artifact_swap",
            "idle_vram_scavenge",
            "manual_reboot",
            "memory_pressure_guard",
            "reconcile_evict:budget",
        ):
            assert death_is_deliberate(reason), reason

    def test_crash_reasons_are_not_deliberate(self):
        for reason in (
            "hard_generation_deadline",
            "process_died_unexpectedly",
            "init_timeout_hard",
            "stale_handshake",
        ):
            assert not death_is_deliberate(reason), reason


class TestCrashLoopBreaker:
    def test_three_young_deaths_trip_the_breaker(self, breaker, clock):
        for _ in range(3):
            breaker.note_death(LANE, lifetime_s=30.0, reason="hard_generation_deadline")
        blocked = breaker.blocked(LANE)
        assert blocked is not None and blocked.startswith("crash_loop_backoff:trip=1")
        assert "hard_generation_deadline" in blocked

    def test_deliberate_deaths_never_trip(self, breaker, clock):
        for _ in range(10):
            breaker.note_death(LANE, lifetime_s=5.0, reason="yield_to_brainstem")
        assert breaker.blocked(LANE) is None

    def test_long_lived_death_closes_the_breaker(self, breaker, clock):
        for _ in range(2):
            breaker.note_death(LANE, lifetime_s=30.0, reason="hard_generation_deadline")
        # A worker that served for 20 minutes then died is NOT a crash loop.
        breaker.note_death(LANE, lifetime_s=1200.0, reason="hard_generation_deadline")
        breaker.note_death(LANE, lifetime_s=30.0, reason="hard_generation_deadline")
        assert breaker.blocked(LANE) is None, "history must reset after a long run"

    def test_backoff_expires_then_half_open_probe_allowed(self, breaker, clock):
        for _ in range(3):
            breaker.note_death(LANE, lifetime_s=10.0, reason="stall")
        assert breaker.blocked(LANE) is not None
        clock.now += 31.0  # past the 30s base backoff
        assert breaker.blocked(LANE) is None, "half-open: one probe spawn allowed"

    def test_half_open_young_death_retrips_at_double_backoff(self, breaker, clock):
        for _ in range(3):
            breaker.note_death(LANE, lifetime_s=10.0, reason="stall")
        clock.now += 31.0
        assert breaker.blocked(LANE) is None  # half-open probe
        breaker.note_death(LANE, lifetime_s=10.0, reason="stall")
        blocked = breaker.blocked(LANE)
        assert blocked is not None and "trip=2" in blocked
        clock.now += 31.0
        assert breaker.blocked(LANE) is not None, "trip 2 backoff is 60s, not 30s"
        clock.now += 31.0
        assert breaker.blocked(LANE) is None

    def test_note_healthy_closes_fully(self, breaker, clock):
        for _ in range(3):
            breaker.note_death(LANE, lifetime_s=10.0, reason="stall")
        clock.now += 31.0
        assert breaker.blocked(LANE) is None
        breaker.note_healthy(LANE)
        # Fully closed: a single young death does not immediately re-trip.
        breaker.note_death(LANE, lifetime_s=10.0, reason="stall")
        assert breaker.blocked(LANE) is None

    def test_kill_switch_records_but_never_blocks(self, breaker, clock, monkeypatch):
        monkeypatch.setenv("AURA_CRASHLOOP_BREAKER", "0")
        for _ in range(5):
            breaker.note_death(LANE, lifetime_s=10.0, reason="stall")
        assert breaker.blocked(LANE) is None
        snap = breaker.snapshot()
        assert snap["enforcing"] is False
        assert snap["lanes"][LANE]["trips"] >= 1, "still records for observability"

    def test_backoff_is_capped(self, breaker, clock, monkeypatch):
        monkeypatch.setenv("AURA_CRASHLOOP_MAX_BACKOFF_S", "120")
        for trip in range(6):
            for _ in range(3):
                breaker.note_death(LANE, lifetime_s=10.0, reason="stall")
            snap = breaker.snapshot()["lanes"][LANE]
            assert snap["blocked_for_s"] <= 120.0
            clock.now += 121.0
            assert breaker.blocked(LANE) is None  # drain to half-open

    def test_lanes_are_independent(self, breaker, clock):
        for _ in range(3):
            breaker.note_death(LANE, lifetime_s=10.0, reason="stall")
        assert breaker.blocked(LANE) is not None
        assert breaker.blocked("/models/qwen-7b") is None

    def test_singleton_accessor(self):
        assert get_crash_loop_breaker() is get_crash_loop_breaker()


def _reconciler(**overrides):
    defaults = dict(
        observe_lanes=lambda: [],
        primary_alive=lambda: True,
        primary_key=lambda: LANE,
        primary_age_s=lambda: 0.0,
        foreground_active=lambda: False,
        breaker=CrashLoopBreaker(),
    )

    async def _spawn():
        return True

    async def _evict(path):
        return True

    defaults["spawn_primary"] = _spawn
    defaults["evict_lane"] = _evict
    defaults.update(overrides)
    return LaneReconciler(**defaults)


class TestLaneReconciler:
    def test_dead_primary_triggers_warmup(self):
        calls = []

        async def spawn():
            calls.append("spawn")
            return True

        rec = _reconciler(primary_alive=lambda: False, spawn_primary=spawn)
        actions = asyncio.run(rec.reconcile_once())
        assert calls == ["spawn"]
        assert actions[0]["action"] == "warm_requested"

    def test_blocked_breaker_holds_healing(self, clock):
        breaker = CrashLoopBreaker()
        for _ in range(3):
            breaker.note_death(LANE, lifetime_s=10.0, reason="stall")
        calls = []

        async def spawn():
            calls.append("spawn")
            return True

        rec = _reconciler(
            primary_alive=lambda: False, spawn_primary=spawn, breaker=breaker
        )
        actions = asyncio.run(rec.reconcile_once())
        assert calls == [], "healing must respect the breaker"
        assert actions[0]["action"] == "held"
        assert "crash_loop_backoff" in actions[0]["detail"]

    def test_dead_primary_converges_despite_foreground_owner(self):
        """The inverted pin, corrected after it deadlocked live (2026-07-10):
        each waiting turn owned the foreground lane, which deferred the very
        convergence it was waiting on — 75 minutes of 216s fallback turns.
        A dead primary cannot be disrupted; ownership never blocks revival."""
        calls = []

        async def spawn():
            calls.append("spawn")
            return True

        rec = _reconciler(
            primary_alive=lambda: False,
            spawn_primary=spawn,
            foreground_active=lambda: True,
        )
        actions = asyncio.run(rec.reconcile_once())
        assert calls == ["spawn"]
        assert actions[0]["action"] == "warm_requested"
        assert actions[0]["detail"] == "reconciler_prewarm_foreground_waiting"

    def test_healthy_old_primary_closes_breaker(self, clock):
        breaker = CrashLoopBreaker()
        for _ in range(3):
            breaker.note_death(LANE, lifetime_s=10.0, reason="stall")
        rec = _reconciler(
            primary_alive=lambda: True,
            primary_age_s=lambda: 600.0,
            breaker=breaker,
        )
        asyncio.run(rec.reconcile_once())
        assert breaker.snapshot()["lanes"][LANE]["trips"] == 0

    def test_over_budget_evicts_best_effort_first(self, monkeypatch):
        monkeypatch.setenv("AURA_LANE_BUDGET_GB", "46")
        lanes = [
            ActiveLane("cortex", QoSClass.GUARANTEED, 20.0, model_path="/m/cortex"),
            ActiveLane("brainstem", QoSClass.BURSTABLE, 5.0, model_path="/m/brainstem"),
            ActiveLane("trainer", QoSClass.BEST_EFFORT, 25.0, model_path="/m/trainer"),
        ]
        evicted = []

        async def evict(path):
            evicted.append(path)
            return True

        rec = _reconciler(observe_lanes=lambda: lanes, evict_lane=evict)
        actions = asyncio.run(rec.reconcile_once())
        assert evicted == ["/m/trainer"], "best-effort goes first; 45GB then fits"
        assert [a["action"] for a in actions] == ["evicted"]

    def test_guaranteed_lane_is_never_evicted(self, monkeypatch):
        monkeypatch.setenv("AURA_LANE_BUDGET_GB", "10")
        lanes = [
            ActiveLane("cortex", QoSClass.GUARANTEED, 20.0, model_path="/m/cortex"),
        ]
        evicted = []

        async def evict(path):
            evicted.append(path)
            return True

        rec = _reconciler(observe_lanes=lambda: lanes, evict_lane=evict)
        asyncio.run(rec.reconcile_once())
        assert evicted == []

    def test_recently_user_facing_lane_is_shielded(self, monkeypatch):
        monkeypatch.setenv("AURA_LANE_BUDGET_GB", "10")
        lanes = [
            ActiveLane(
                "brainstem",
                QoSClass.BURSTABLE,
                12.0,
                model_path="/m/brainstem",
                last_user_facing_age_s=10.0,
            ),
        ]
        evicted = []

        async def evict(path):
            evicted.append(path)
            return True

        rec = _reconciler(observe_lanes=lambda: lanes, evict_lane=evict)
        asyncio.run(rec.reconcile_once())
        assert evicted == [], "a lane that served the user 10s ago must not churn"

    def test_within_budget_takes_no_action(self, monkeypatch):
        monkeypatch.setenv("AURA_LANE_BUDGET_GB", "46")
        lanes = [
            ActiveLane("cortex", QoSClass.GUARANTEED, 20.0, model_path="/m/cortex"),
        ]
        rec = _reconciler(observe_lanes=lambda: lanes)
        actions = asyncio.run(rec.reconcile_once())
        assert actions == []

    def test_single_flight_collapses_overlapping_calls(self):
        rec = _reconciler()
        inner_actions = []

        async def spawn():
            inner_actions.extend(await rec.reconcile_once())
            return True

        rec._spawn_primary = spawn
        rec._primary_alive = lambda: False
        asyncio.run(rec.reconcile_once())
        assert inner_actions[0]["action"] == "skipped"
        assert inner_actions[0]["detail"] == "reconcile_already_inflight"

    def test_snapshot_shape(self):
        rec = _reconciler()
        snap = rec.snapshot()
        assert set(snap) >= {
            "alive",
            "ready",
            "running",
            "enabled",
            "interval_s",
            "recent_actions",
            "breaker",
        }
        assert rec.is_alive() is False
        assert rec.is_ready() is False

    @pytest.mark.asyncio
    async def test_lifecycle_probe_tracks_background_loop(self):
        rec = _reconciler()

        await rec.start()
        assert rec.is_alive() is True
        assert rec.is_ready() is True

        await rec.stop()
        assert rec.is_alive() is False
        assert rec.is_ready() is False

    @pytest.mark.asyncio
    async def test_shutdown_latch_prevents_start_and_reconcile_work(self):
        calls = []

        async def spawn():
            calls.append("spawn")
            return True

        rec = _reconciler(primary_alive=lambda: False, spawn_primary=spawn)
        clear_shutdown_request()
        try:
            request_shutdown("unit-test")
            await rec.start()
            assert rec.is_alive() is False
            actions = await rec.reconcile_once()
        finally:
            clear_shutdown_request()

        assert calls == []
        assert actions[0]["action"] == "skipped"
        assert actions[0]["detail"] == "runtime_shutdown"

    def test_kill_switch(self, monkeypatch):
        monkeypatch.setenv("AURA_LANE_RECONCILER", "0")
        assert LaneReconciler.enabled() is False
        monkeypatch.setenv("AURA_LANE_RECONCILER", "1")
        assert LaneReconciler.enabled() is True

    def test_singleton_accessor(self):
        assert get_lane_reconciler() is get_lane_reconciler()


class TestMlxSeam:
    """The death-report and spawn-consult seams in mlx_client."""

    class _FakeClient:
        def __init__(self, model_path, started_at):
            self.model_path = model_path
            self._process_started_at = started_at

    def test_death_report_counts_young_crashes(self, monkeypatch):
        import time as real_time

        from core.brain.llm import mlx_client as mc

        get_crash_loop_breaker().reset_for_test()
        client = self._FakeClient(LANE, real_time.time() - 30.0)
        for _ in range(3):
            mc._note_lane_worker_death(client, "hard_generation_deadline")
        assert mc._crash_loop_blocks_worker_spawn(client) is not None

    def test_death_report_ignores_deliberate_kills(self):
        import time as real_time

        from core.brain.llm import mlx_client as mc

        get_crash_loop_breaker().reset_for_test()
        client = self._FakeClient(LANE, real_time.time() - 30.0)
        for _ in range(5):
            mc._note_lane_worker_death(client, "yield_to_qwen-7b")
        assert mc._crash_loop_blocks_worker_spawn(client) is None

    def test_death_report_skips_never_started_worker(self):
        from core.brain.llm import mlx_client as mc

        get_crash_loop_breaker().reset_for_test()
        client = self._FakeClient(LANE, 0.0)
        for _ in range(5):
            mc._note_lane_worker_death(client, "hard_generation_deadline")
        assert mc._crash_loop_blocks_worker_spawn(client) is None


class TestGenerationTimeoutNotCrashLoop:
    """A generation force-abort is a policy recycle, not a crash — it must
    NOT trip the crash-loop breaker and back off the fast fallback under
    contention (2026-07-15 soak: reflex/brainstem timing out queued behind
    a busy foreground 32B cascaded into deeper contention)."""

    def test_generation_timeout_reasons_are_deliberate(self):
        for reason in (
            "inference_gate_generation_timeout:Reflex:14.7s",
            "inference_gate_generation_timeout:Brainstem:53.8s",
            "first_token_wall_clock_watchdog",
        ):
            assert death_is_deliberate(reason), reason

    def test_hard_deadline_and_crashes_still_trip(self):
        # The escalated hard ceiling and genuine crashes must still count.
        for reason in (
            "hard_generation_deadline",
            "process_died_unexpectedly",
            "init_timeout_hard",
        ):
            assert not death_is_deliberate(reason), reason
