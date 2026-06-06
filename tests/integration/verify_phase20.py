################################################################################

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add core to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.curiosity_engine import CuriosityEngine
from core.ops.singularity_monitor import SingularityMonitor
from core.volition import VolitionEngine


class CognitiveEngineFixture:
    pass


class KnowledgeGraphFixture:
    def __init__(self):
        self.sparse_nodes = []

    def get_sparse_nodes(self):
        return list(self.sparse_nodes)


class MirrorFixture:
    def __init__(self):
        self.health_score = 1.0

    def get_audit_summary(self):
        return {"health_score": self.health_score}


class MetacognitionFixture:
    def __init__(self):
        self.mirror = MirrorFixture()


class OrchestratorFixture:
    def __init__(self):
        self.cognitive_engine = CognitiveEngineFixture()
        self.knowledge_graph = KnowledgeGraphFixture()
        self.metacognition = MetacognitionFixture()
        self.is_busy = False


class ProactiveCommFixture:
    def get_boredom_level(self):
        return 0.0


class TestPhase20(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.orchestrator = OrchestratorFixture()

    async def test_roadmap_awareness(self):
        """Verify that VolitionEngine can scan the brain directory for phases."""
        with tempfile.TemporaryDirectory() as tmp:
            brain_root = Path(tmp) / "brain"
            phase19 = brain_root / "phase19" / "task.md"
            phase20 = brain_root / "phase20" / "task.md"
            phase19.parent.mkdir(parents=True, exist_ok=True)
            phase20.parent.mkdir(parents=True, exist_ok=True)
            phase19.write_text("# Phase 19: The Hall of Mirrors\n", encoding="utf-8")
            phase20.write_text("# Phase 20: Singularity Prep\n", encoding="utf-8")

            volition = VolitionEngine(self.orchestrator)
            volition.brain_base = brain_root
            milestones = volition._scan_roadmap()
            volition.milestones = milestones
            self.assertIn("Phase 19: The Hall of Mirrors", milestones)
            self.assertIn("Phase 20: Singularity Prep", milestones)

            found = False
            for _ in range(100):
                g = volition._check_roadmap()
                if g:
                    self.assertIn("Phase 20: Singularity Prep", g["objective"])
                    found = True
                    break
            self.assertTrue(found, "Roadmap goal should eventually fire")

    async def test_kg_driven_curiosity(self):
        """Verify that CuriosityEngine targets sparse nodes in the KG."""
        self.orchestrator.knowledge_graph.sparse_nodes = ["Quantum Entanglement"]
        
        curiosity = CuriosityEngine(self.orchestrator, ProactiveCommFixture())
        # Clear queue to trigger novelty search
        curiosity.curiosity_queue.clear()
        
        topic = curiosity._get_next()
        self.assertIsNotNone(topic)
        self.assertIn("Quantum Entanglement", topic.topic)
        self.assertEqual(topic.reason, "knowledge graph novelty search")

    async def test_singularity_heartbeat(self):
        """Verify that SingularityMonitor enables acceleration."""
        self.orchestrator.metacognition.mirror.health_score = 0.95
        
        monitor = SingularityMonitor(self.orchestrator)
        monitor.improvement_rate = 0.05 # Pre-set to trigger acceleration
        
        monitor.pulse()
        self.assertTrue(monitor.is_accelerated)
        self.assertEqual(monitor.acceleration_factor, 1.5)
        self.assertEqual(self.orchestrator.cognitive_engine.singularity_factor, 1.5)

if __name__ == "__main__":
    unittest.main()


##
