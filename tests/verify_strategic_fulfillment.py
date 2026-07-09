"""tests/verify_strategic_fulfillment.py
Simulation of a multi-step project fulfillment with error recovery.
"""
import json
import tempfile
import unittest
from pathlib import Path

from core.container import ServiceContainer
from core.data.project_store import ProjectStore
from core.planning.strategic_planner import StrategicPlanner


class ThoughtProbe:
    def __init__(self, content: str):
        self.content = content


class StrategicBrainProbe:
    def __init__(self, plan_payload: dict):
        self.plan_payload = plan_payload
        self.prompts = []

    async def think(self, prompt, mode=None):
        self.prompts.append({"prompt": prompt, "mode": mode})
        return ThoughtProbe(f"```json\n{json.dumps(self.plan_payload)}\n```")


class NeuralFeedProbe:
    def __init__(self):
        self.events = []

    def push(self, content, category=None):
        self.events.append({"content": content, "category": category})


class TestStrategicFulfillment(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ServiceContainer.clear()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "strategic_fulfillment.db")
        
        self.store = ProjectStore(self.db_path)
        self.brain = StrategicBrainProbe(
            {
                "project_name": "Project Singularity",
                "tasks": [
                    {"description": "Consult external advisors", "priority": 15},
                    {"description": "Alternative algorithm trial", "priority": 14},
                ],
            }
        )
        self.planner = StrategicPlanner(self.brain, self.store)
        self.feed = NeuralFeedProbe()
        ServiceContainer.register_instance("strategic_planner", self.planner)
        ServiceContainer.register_instance("neural_feed", self.feed)

    async def asyncTearDown(self):
        ServiceContainer.clear()
        self.temp_dir.cleanup()

    async def test_full_fulfillment_cycle(self):
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
        
        t1 = self.planner.get_next_task(project.id)
        self.assertEqual(t1.description, "Gather global data")
        self.planner.mark_task_complete(t1.id, "Data gathered successfully.")
        
        t2 = self.planner.get_next_task(project.id)
        self.assertEqual(t2.description, "Identify conflict points")
        self.planner.mark_task_complete(t2.id, "Conflicts indexed.")
        
        t3 = self.planner.get_next_task(project.id)
        self.assertEqual(t3.description, "Propose solutions")
        self.planner.mark_task_failed(t3.id, "Algorithm deadlock: No solutions found.")
        
        success = await self.planner.replan_project(project.id, "Algorithm deadlock")
        self.assertTrue(success)
        self.assertEqual(self.brain.prompts[0]["mode"], "SLOW")
        self.assertIn("Algorithm deadlock", self.brain.prompts[0]["prompt"])
        
        next_t = self.planner.get_next_task(project.id)
        self.assertEqual(next_t.description, "Consult external advisors")
        
        report = await self.planner.get_project_status_report(project.id)
        self.assertIn("Consult external advisors", report)
        self.assertIn("Gather global data", report)
        self.assertTrue(
            any(event["category"] == "STRATEGY" for event in self.feed.events),
            self.feed.events,
        )

if __name__ == '__main__':
    unittest.main()
