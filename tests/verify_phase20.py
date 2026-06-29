import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# Add core to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.curiosity_engine import CuriosityEngine
from core.ops.singularity_monitor import SingularityMonitor
from core.volition import VolitionEngine


class KnowledgeGraphProbe:
    def __init__(self):
        self.sparse_nodes = ["Quantum Entanglement"]

    def get_sparse_nodes(self, *args, **kwargs):
        return list(self.sparse_nodes)


class CognitiveEngineProbe:
    def __init__(self):
        self.singularity_factor = 1.0


class AuditMirrorProbe:
    def __init__(self):
        self.summary = {"health_score": 0.95}

    def get_audit_summary(self):
        return dict(self.summary)


class OrchestratorProbe:
    def __init__(self):
        self.cognitive_engine = CognitiveEngineProbe()
        self.knowledge_graph = KnowledgeGraphProbe()
        self.metacognition = SimpleNamespace(mirror=AuditMirrorProbe())
        self.is_busy = False


class SensoriumProbe:
    pass


class TestPhase20(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.orchestrator = OrchestratorProbe()

    async def test_roadmap_awareness(self):
        """Verify that VolitionEngine can scan the brain directory for phases."""
        brain_root = Path(tempfile.mkdtemp(prefix="aura_phase20_brain_"))
        try:
            phase19 = brain_root / "phase19" / "task.md"
            phase20 = brain_root / "phase20" / "task.md"
            phase19.parent.mkdir(parents=True, exist_ok=True)
            phase20.parent.mkdir(parents=True, exist_ok=True)
            phase19.write_text("# Phase 19: The Hall of Mirrors\n", encoding="utf-8")
            phase20.write_text("# Phase 20: Singularity Prep\n", encoding="utf-8")

            volition = VolitionEngine(self.orchestrator)
            volition.brain_base = brain_root
            milestones = volition._scan_roadmap()
            self.assertIn("Phase 19: The Hall of Mirrors", milestones)
            self.assertIn("Phase 20: Singularity Prep", milestones)

            volition.milestones = milestones
            import core.volition as volition_module

            original_random = volition_module.random.random
            try:
                volition_module.random.random = lambda: 0.01
                goal = volition._check_roadmap()
            finally:
                volition_module.random.random = original_random

            self.assertIsNotNone(goal)
            self.assertIn("Phase 20: Singularity Prep", goal["objective"])
        finally:
            shutil.rmtree(brain_root, ignore_errors=True)

    async def test_kg_driven_curiosity(self):
        """Verify that CuriosityEngine targets sparse nodes in the KG."""
        curiosity = CuriosityEngine(self.orchestrator, SensoriumProbe())
        # Clear queue to trigger novelty search
        curiosity.curiosity_queue.clear()
        
        topic = curiosity._get_next()
        self.assertIsNotNone(topic)
        self.assertIn("Quantum Entanglement", topic.topic)
        self.assertEqual(topic.reason, "sparse region of persistent knowledge")

    async def test_singularity_heartbeat(self):
        """Verify that SingularityMonitor enables acceleration."""
        monitor = SingularityMonitor(self.orchestrator)
        monitor.improvement_rate = 0.05 # Pre-set to trigger acceleration
        
        monitor.pulse()
        self.assertTrue(monitor.is_accelerated)
        self.assertEqual(monitor.acceleration_factor, 1.5)
        self.assertEqual(self.orchestrator.cognitive_engine.singularity_factor, 1.5)

if __name__ == "__main__":
    unittest.main()
