"""core/factory/test_runner.py — Software Factory Test Runner.
"""
from __future__ import annotations

import logging
import subprocess
from typing import Dict

from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.FactoryTestRunner")


class FactoryTestRunner:
    """Runs codebase test suites using standard pytest commands."""

    @staticmethod
    def run_tests(test_path: str = "tests/") -> Dict[str, Any]:
        logger.info("🏭 TestRunner running tests at path: %s", test_path)
        gateway = get_subprocess_gateway()
        
        # Dispatch command via subprocess gateway
        proc = gateway.run(
            argv=["pytest", test_path, "-q"],
            timeout=45.0,
            source="factory_test_runner",
        )
        passed = proc.returncode == 0
        logger.info("🏭 Tests outcome: %s (Exit Code: %d)", "PASSED" if passed else "FAILED", proc.returncode)

        return {
            "passed": passed,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
