################################################################################

"""tests/test_ghost_probe.py
Unit test for Ghost Probe deployment logic.
"""
import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from core.collective.probe_manager import ProbeManager
from core.skills.ghost_probe import GhostProbeParams, GhostProbeSkill
from core.container import ServiceContainer


class OrchestratorHarness:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.messages = []

    def enqueue_message(self, message):
        self.messages.append(message)


class TestGhostProbe(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ServiceContainer.clear()
        self.orchestrator = OrchestratorHarness()
        self.manager = ProbeManager(self.orchestrator)
        ServiceContainer.register_instance("probe_manager", self.manager)
        self.skill = GhostProbeSkill(self.orchestrator)

    async def asyncTearDown(self):
        """Ensure all probes are killed."""
        try:
            ok = await self.manager.stop()
        except (AttributeError, LookupError, OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            self.fail(f"Probe manager stop failed: {type(exc).__name__}: {exc}")
        self.assertTrue(ok)

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
            probe_path = Path(self.manager.probe_metadata["unit_probe"]["path"])
            self.assertEqual(probe_path.parent, Path(tempfile.gettempdir()))
            await asyncio.sleep(0.2)
            self.assertTrue(self.orchestrator.messages)
        finally:
            ok = await self.manager.cleanup_probe("unit_probe")
            target.unlink(missing_ok=True)

        self.assertTrue(ok)
        self.assertNotIn("unit_probe", self.manager.probes)
        self.assertNotIn("unit_probe", self.manager.probe_metadata)
        self.assertFalse(probe_path.exists())

    async def test_execute_sanitizes_probe_id_and_stop_cleans_script(self):
        with tempfile.NamedTemporaryFile(prefix="aura_ghost_probe_", delete=False) as handle:
            target = Path(handle.name)

        try:
            params = GhostProbeParams(
                probe_id="../unsafe/probe",
                target=str(target),
                type="file",
                duration=5,
            )
            result = await self.skill.execute(params)
            self.assertTrue(result["ok"], result)
            probe_path = Path(self.manager.probe_metadata["../unsafe/probe"]["path"])
            self.assertEqual(probe_path.parent, Path(tempfile.gettempdir()))
            self.assertNotIn("/", probe_path.name)
            self.assertTrue(probe_path.exists())

            ok = await self.manager.stop()
        finally:
            target.unlink(missing_ok=True)

        self.assertTrue(ok)
        self.assertFalse(probe_path.exists())
        self.assertEqual(self.manager.probes, {})
        self.assertEqual(self.manager.probe_metadata, {})
