"""Dead contract loops must be revivable from health-pulse threads.

Observed live 2026-07-05: the runtime sat DEGRADED for 84 minutes with three
'important' contract entries dead (event_loop_monitor, mind_tick,
unified_runtime_pressure). Two had working restart machinery that silently
no-oped because health pulses run on plain threads where asyncio task
creation raises; the third was a phantom requirement with no provider
anywhere in the tree.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import core.runtime.runtime_pressure as rp
from core.runtime.runtime_pressure import UnifiedRuntimePressure


class TestUnifiedRuntimePressure:
    def test_snapshot_shape_and_nominal_alive(self, monkeypatch):
        pressure = UnifiedRuntimePressure()
        snap = pressure.runtime_pressure_snapshot()
        for key in ("loop_lag_s", "memory_pct", "thermal_level", "red_zones", "pressure_ok"):
            assert key in snap, f"snapshot missing {key}"
        assert isinstance(snap["red_zones"], list)

    def test_red_zone_memory_makes_it_not_alive(self, monkeypatch):
        pressure = UnifiedRuntimePressure()

        class _FakeVM:
            percent = 97.0

        fake_psutil = SimpleNamespace(virtual_memory=lambda: _FakeVM())
        monkeypatch.setitem(__import__("sys").modules, "psutil", fake_psutil)
        assert pressure.is_alive() is False
        snap = pressure.get_status()
        assert any(zone.startswith("memory_") for zone in snap["red_zones"])

    def test_is_alive_never_raises(self, monkeypatch):
        pressure = UnifiedRuntimePressure()
        monkeypatch.setattr(
            pressure, "runtime_pressure_snapshot", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        assert pressure.is_alive() is False

    def test_singleton_accessor(self):
        assert rp.get_unified_runtime_pressure() is rp.get_unified_runtime_pressure()


class TestEventLoopMonitorThreadRevival:
    def test_foreign_thread_hands_restart_to_owner_loop(self):
        from core.utils.concurrency import EventLoopMonitor

        class _DoneTask:
            def done(self):
                return True

        scheduled: list = []

        class _FakeLoop:
            def is_closed(self):
                return False

            def call_soon_threadsafe(self, fn, *args):
                scheduled.append(fn)

        monitor = EventLoopMonitor(threshold=0.5)
        monitor._task = _DoneTask()
        monitor._owner_loop = _FakeLoop()

        def _raising_start():
            raise RuntimeError("no running event loop")

        monitor.start = _raising_start

        # Called from a plain (non-asyncio) test thread: must not raise, must
        # schedule the restart onto the owning loop, and must stay honest
        # (False) until the restart actually lands.
        assert monitor.ensure_running() is False
        assert scheduled == [_raising_start]

    def test_owner_loop_captured_on_start(self):
        from core.utils.concurrency import EventLoopMonitor

        async def _scenario():
            monitor = EventLoopMonitor(threshold=0.5)
            monitor.start()
            try:
                assert monitor._owner_loop is asyncio.get_running_loop()
            finally:
                await monitor.stop()

        asyncio.run(_scenario())


class TestMindTickThreadRevival:
    def test_repair_from_foreign_thread_schedules_onto_owner_loop(self):
        from core.mind_tick import MindTick

        scheduled: list = []

        class _FakeLoop:
            def is_closed(self):
                return False

            def call_soon_threadsafe(self, fn, *args):
                scheduled.append(fn)

        class _DoneTask:
            def done(self):
                return True

            def exception(self):
                return None

        tick = MindTick.__new__(MindTick)
        tick._running = True
        tick._task = _DoneTask()
        tick._owner_loop = _FakeLoop()
        tick._consecutive_loop_failures = 0
        tick._last_liveness_repair_at = 0.0
        tick._started_at = 1.0

        # Plain thread: no running loop here. The repair must hand itself to
        # the owning loop instead of returning False and leaving the mind dead.
        assert tick._attempt_liveness_repair() is False
        assert len(scheduled) == 1

    def test_background_kernel_tick_yields_under_foreground_inference(self, monkeypatch):
        # A soak's back-to-back turns saturate the generation gate; the
        # background kernel tick must yield instead of blocking (which froze
        # the iteration and falsely marked mind_tick dead, 2026-07-06).
        import core.runtime.backpressure as bp
        from core.mind_tick import MindTick

        monkeypatch.setattr(bp, "foreground_inference_active", lambda: True)
        tick = MindTick.__new__(MindTick)
        tick.orchestrator = SimpleNamespace(_flow_controller=None)
        assert tick._background_reasoning_pause_reason() == "foreground_inference_active"


class TestGuiDegradedReadyKeepsUI:
    def test_degraded_ready_when_conversation_works_but_loop_degraded(self):
        from interface.gui_actor import _heartbeat_response_state

        class _Resp:
            status_code = 503

            @staticmethod
            def json():
                return {
                    "healthy": False,
                    "status": "booting",
                    "runtime_probe_healthy": True,
                    "conversation_ready": True,
                    "boot_phase": "kernel_warming",
                    "blockers": ["healthy", "runtime_contract_healthy", "important:mind_tick"],
                    "required_probes": {"all_passed": True},
                }

        # Conversation works; only a background loop is degraded → keep the UI.
        assert _heartbeat_response_state(_Resp()) == "degraded_ready"

    def test_truly_down_is_still_unhealthy(self):
        from interface.gui_actor import _heartbeat_response_state

        class _Resp:
            status_code = 503

            @staticmethod
            def json():
                return {
                    "healthy": False,
                    "status": "booting",
                    "runtime_probe_healthy": False,
                    "conversation_ready": False,
                    "boot_phase": "kernel_booting",
                    "blockers": ["healthy", "inference"],
                    "required_probes": {"all_passed": False},
                }

        assert _heartbeat_response_state(_Resp()) == "unhealthy"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
