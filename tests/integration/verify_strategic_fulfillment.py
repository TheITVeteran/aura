################################################################################

"""tests/verify_strategic_fulfillment.py
Simulation of a multi-step project fulfillment with error recovery.
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.container import ServiceContainer
from core.data.project_store import ProjectStore
from core.strategic_planner import StrategicPlanner


class StrategicBrainProbe:
    def __init__(self, content: str):
        self.content = content
        self.think_calls = []

    async def think(self, *args, **kwargs):
        self.think_calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(content=self.content)


class NeuralFeedProbe:
    def __init__(self):
        self.events = []

    def push(self, message, **kwargs):
        self.events.append({"message": message, "kwargs": kwargs})


class TestStrategicFulfillment(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ServiceContainer.clear()
        db_path = Path(tempfile.gettempdir()) / "test_strategic_fulfillment.db"
        db_path.unlink(missing_ok=True)
        self.db_path = str(db_path)
        
        replan_json = {
            "project_name": "Project Singularity",
            "tasks": [
                {"description": "Consult external advisors", "priority": 15},
                {"description": "Alternative algorithm trial", "priority": 14},
            ],
        }

        self.store = ProjectStore(self.db_path)
        self.brain = StrategicBrainProbe(f"```json\n{json.dumps(replan_json)}\n```")
        self.planner = StrategicPlanner(self.brain, self.store)
        self.neural_feed = NeuralFeedProbe()
        ServiceContainer.register_instance("strategic_planner", self.planner)
        ServiceContainer.register_instance("neural_feed", self.neural_feed)

    async def test_full_fulfillment_cycle(self):
        # 1. Manually create a 5-step project
        project = self.store.create_project("Project Singularity", "Achieve world peace")
        tasks = [
            "Gather global data",
            "Identify conflict points",
            "Propose solutions",
            "Negotiate treaties",
            "Final verification"
        ]
        for i, desc in enumerate(tasks):
            self.store.add_task(project.id, desc, priority=len(tasks)-i)
        
        # 2. Simulate completion of first 2 tasks
        t1 = self.planner.get_next_task(project.id)
        self.assertEqual(t1.description, "Gather global data")
        self.planner.mark_task_complete(t1.id, "Data gathered successfully.")
        
        t2 = self.planner.get_next_task(project.id)
        self.assertEqual(t2.description, "Identify conflict points")
        self.planner.mark_task_complete(t2.id, "Conflicts indexed.")
        
        # 3. Simulate failure on 3rd task
        t3 = self.planner.get_next_task(project.id)
        self.assertEqual(t3.description, "Propose solutions")
        self.planner.mark_task_failed(t3.id, "Algorithm deadlock: No solutions found.")
        
        # 4. Trigger Reflection Loop (Replanning)
        success = await self.planner.replan_project(project.id, "Algorithm deadlock")
        self.assertTrue(success)
        self.assertEqual(len(self.brain.think_calls), 1)
        self.assertEqual(len(self.neural_feed.events), 1)
        
        # 5. Verify next task is the NEW one
        next_t = self.planner.get_next_task(project.id)
        self.assertEqual(next_t.description, "Consult external advisors")
        
        # 6. Verify status report
        report = await self.planner.get_project_status_report(project.id)
        print("\n--- Project Status Report ---\n", report)
        self.assertIn("Consult external advisors", report)
        self.assertIn("Gather global data", report)

if __name__ == '__main__':
    unittest.main()


##
