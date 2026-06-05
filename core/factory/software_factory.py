"""core/factory/software_factory.py — Software Development Factory.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from core.factory.repo_cartographer import RepoCartographer
from core.factory.patch_planner import PatchPlanner
from core.factory.code_writer import CodeWriter
from core.factory.test_runner import FactoryTestRunner
from core.factory.regression_guard import RegressionGuard
from core.factory.pr_builder import PRBuilder
from core.factory.rollback_manager import RollbackManager

logger = logging.getLogger("Aura.SoftwareFactory")


class SoftwareFactory:
    """Manages the autonomous coding task lifecycle: plan → edit → test → review."""

    def __init__(self) -> None:
        self.cartographer = RepoCartographer()

    async def execute_refactor(self, issue_description: str) -> Dict[str, Any]:
        """Runs the software factory loop: map -> plan -> write -> lint -> test -> commit/rollback."""
        logger.info("🏭 SoftwareFactory initiated refactoring loop for: '%s'", issue_description)
        
        # 1. Map repository
        code_map = self.cartographer.map_repository()

        # 2. Plan changes
        tasks = PatchPlanner.plan_changes(issue_description, code_map)
        if not tasks:
            return {"ok": False, "reason": "no_tasks_planned"}

        task = tasks[0]
        # 3. Write patch draft
        patch_code = await CodeWriter.write_patch(task)

        # 4. Verify syntaxes and lints
        guard_res = RegressionGuard.verify_patch(task.file_path)
        if not guard_res.get("passed"):
            logger.warning("🏭 Regression check failed on patch. Discarding edits.")
            # Discard local file modification
            return {"ok": False, "reason": "regression_failed", "details": guard_res}

        # 5. Execute test suites
        test_res = FactoryTestRunner.run_tests("tests/runtime/")
        if not test_res.get("passed"):
            logger.warning("🏭 Tests failed post-patch! Rolling back.")
            RollbackManager.discard_changes(task.file_path)
            return {"ok": False, "reason": "tests_failed", "details": test_res}

        # 6. Build PR draft
        pr_res = PRBuilder.create_branch_and_draft(
            branch_name="refactor/" + task.file_path.replace("/", "_").replace(".py", ""),
            file_path=task.file_path,
            commit_message=f"Refactored {task.file_path} to address: {issue_description}",
        )

        return {
            "ok": True,
            "pr_created": True,
            "branch_name": pr_res.get("branch_name"),
            "details": pr_res,
        }


# Singleton
_factory_instance: SoftwareFactory | None = None


def get_software_factory() -> SoftwareFactory:
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = SoftwareFactory()
    return _factory_instance
