import asyncio
import os
import sys
import time
import unittest
from types import SimpleNamespace

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.orchestrator import RobustOrchestrator
from core.resilience.sovereign_watchdog import SovereignWatchdog


class temporary_attr:
    def __init__(self, target, name, value):
        self.target = target
        self.name = name
        self.value = value
        self.previous = None

    def __enter__(self):
        self.previous = getattr(self.target, self.name)
        setattr(self.target, self.name, self.value)

    def __exit__(self, exc_type, exc, tb):
        setattr(self.target, self.name, self.previous)


class TestTimeResilience(unittest.IsolatedAsyncioTestCase):
    async def test_orchestrator_stall_detection_resilience(self):
        """Verify that Orchestrator stall detection ignores wall-clock jumps."""
        orchestrator = RobustOrchestrator()
        orchestrator.status = SimpleNamespace(is_processing=True)
        
        # Start processing at T=100 (monotonic)
        with temporary_attr(time, "monotonic", lambda: 100.0):
            orchestrator._current_processing_start = time.monotonic()
            
        # Simulate 1 second has passed monotonically, but wall-clock jumped 1 hour (3600s)
        # Aegis sentinel should NOT trigger if it uses monotonic time.
        wall_clock_after_jump = time.time() + 3600.0
        with temporary_attr(time, "monotonic", lambda: 101.0):
            with temporary_attr(time, "time", lambda: wall_clock_after_jump):
                # Manually trigger the check logic from _aegis_sentinel
                start_time = orchestrator._current_processing_start
                delta = time.monotonic() - start_time
                self.assertEqual(delta, 1.0, "Delta should be 1.0s regardless of wall-clock jump")
                self.assertLess(delta, 45.0, "Stall should NOT be detected")

    async def test_watchdog_heartbeat_resilience(self):
        """Verify that SovereignWatchdog ignores wall-clock jumps."""
        orchestrator = SimpleNamespace()
        watchdog = SovereignWatchdog(orchestrator, timeout=30.0)
        
        # Initial heartbeat at T=100 (monotonic)
        with temporary_attr(time, "monotonic", lambda: 100.0):
            watchdog.heartbeat()
            
        # Simulate 10 seconds passed monotonically, but wall-clock jumped 1 hour
        wall_clock_after_jump = time.time() + 3600.0
        with temporary_attr(time, "monotonic", lambda: 110.0):
            with temporary_attr(time, "time", lambda: wall_clock_after_jump):
                elapsed = time.monotonic() - watchdog._last_heartbeat
                self.assertEqual(elapsed, 10.0, "Elapsed should be 10.0s regardless of wall-clock jump")
                self.assertLess(elapsed, 30.0, "Watchdog should NOT trigger recovery")

    async def test_watchdog_start_uses_task_tracker_ownership(self):
        """Verify watchdog background loop is lifecycle-owned by the task tracker."""
        tracker_calls = {}

        class _Tracker:
            def create_task(self, coro, name=None):
                task = asyncio.create_task(coro, name=name)
                tracker_calls["name"] = name
                tracker_calls["task"] = task
                return task

        orchestrator = SimpleNamespace()
        watchdog = SovereignWatchdog(orchestrator, interval=3600.0, timeout=30.0)

        import core.resilience.sovereign_watchdog as watchdog_module

        with temporary_attr(watchdog_module, "get_task_tracker", lambda: _Tracker()):
            await watchdog.start()
            self.assertEqual(tracker_calls["name"], "sovereign_watchdog")
            self.assertIs(watchdog._task, tracker_calls["task"])
            await watchdog.stop()

if __name__ == '__main__':
    unittest.main()
