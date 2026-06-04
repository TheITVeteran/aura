################################################################################

"""tests/test_ghost_probe.py
Unit test for Ghost Probe deployment logic.
"""
import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from core.collective.probe_manager import ProbeManager
from core.skills.ghost_probe import GhostProbeParams, GhostProbeSkill
from core.container import ServiceContainer

class TestGhostProbe(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ServiceContainer.clear()
        self.orchestrator = MagicMock()
        # Mock loop for run_in_executor
        self.orchestrator.loop = asyncio.get_running_loop()
        self.manager = ProbeManager(self.orchestrator)
        ServiceContainer.register_instance("probe_manager", self.manager)
        self.skill = GhostProbeSkill(self.orchestrator)

    async def asyncTearDown(self):
        """Ensure all probes are killed."""
        probe_ids = list(self.manager.probes.keys())
        cleanup_errors = []
        for pid in probe_ids:
            try:
                ok = await self.manager.cleanup_probe(pid)
                if not ok:
                    cleanup_errors.append(f"{pid}: cleanup returned false")
            except (AttributeError, LookupError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                cleanup_errors.append(f"{pid}: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            self.fail("Probe cleanup failed: " + "; ".join(cleanup_errors))

    async def test_execute_deploys_and_cleans_file_probe(self):
        with tempfile.NamedTemporaryFile(prefix="aura_ghost_probe_", delete=False) as handle:
            target = Path(handle.name)

        try:
            params = GhostProbeParams(
                probe_id="unit_probe",
                target=str(target),
                type="file",
                duration=5,
            )
            result = await self.skill.execute(params)
            self.assertTrue(result["ok"], result)
            self.assertIn("unit_probe", self.manager.probes)
            self.assertIn("unit_probe", self.manager.probe_metadata)
            await asyncio.sleep(0.2)
            self.assertTrue(self.orchestrator.enqueue_message.called)
        finally:
            ok = await self.manager.cleanup_probe("unit_probe")
            target.unlink(missing_ok=True)

        self.assertTrue(ok)
        self.assertNotIn("unit_probe", self.manager.probes)
        self.assertNotIn("unit_probe", self.manager.probe_metadata)

    
