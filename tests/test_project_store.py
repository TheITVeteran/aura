################################################################################

"""tests/test_project_store.py
Verification for Phase 17.1: ProjectStore persistence and task retrieval.
"""
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pytest

from core.data.project_store import ProjectStore


class TestProjectStore(unittest.TestCase):
    def setUp(self):
        self.db_path = str(Path(tempfile.gettempdir()) / "test_aura_projects.db")
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.store = ProjectStore(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_project_lifecycle(self):
        # 1. Create Project
        proj = self.store.create_project("Test Project", "Reach the singularity")
        self.assertEqual(proj.name, "Test Project")
        
        # 2. Add Tasks
        self.store.add_task(proj.id, "Gather data", priority=10)
        self.store.add_task(proj.id, "Train model", priority=5)
        
        # 3. Retrieve and Verify
        active = self.store.get_active_projects()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].id, proj.id)
        
        tasks = self.store.get_tasks_for_project(proj.id)
        self.assertEqual(len(tasks), 2)
        # Check priority sorting
        self.assertEqual(tasks[0].description, "Gather data")
        
    def test_cross_project_task_retrieval(self):
        p1 = self.store.create_project("Project Alpha", "Goal A")
        p2 = self.store.create_project("Project Beta", "Goal B")
        
        self.store.add_task(p1.id, "Alpha Task 1", priority=1)
        self.store.add_task(p2.id, "Beta Task 1", priority=10) # Highest priority
        
        next_task = self.store.get_next_strategic_task()
        self.assertIsNotNone(next_task)
        self.assertEqual(next_task.description, "Beta Task 1")
        
    def test_persistence_reboot(self):
        # Create data
        proj = self.store.create_project("Persistent Proj", "Goal P")
        self.store.add_task(proj.id, "Task 1")
        
        # Close and "reboot" by creating a new store instance with same DB
        new_store = ProjectStore(self.db_path)
        active = new_store.get_active_projects()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].name, "Persistent Proj")
        
        tasks = new_store.get_tasks_for_project(proj.id)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].description, "Task 1")

    def test_invalid_statuses_are_rejected(self):
        proj = self.store.create_project("Strict Proj", "Goal")
        task = self.store.add_task(proj.id, "Task 1")

        with pytest.raises(ValueError):
            self.store.update_project_status(proj.id, "maybe")
        with pytest.raises(ValueError):
            self.store.update_task_status(task.id, "maybe")

    def test_missing_updates_report_false(self):
        self.assertFalse(self.store.update_project_status("missing-project", "completed"))
        self.assertFalse(self.store.update_task_status("missing-task", "completed"))

    def test_orphan_task_is_rejected_by_foreign_key(self):
        with pytest.raises(sqlite3.IntegrityError):
            self.store.add_task("missing-project", "cannot be orphaned")

if __name__ == '__main__':
    unittest.main()


##
