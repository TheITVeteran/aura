"""core/factory/test_runner.py — Test Suite Executor.

Executes tests in localized sub-environments, capturing stdout/stderr,
pass/fail counts, and timing.
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.TestRunner")


class TestRunner:
    """Executes project test suites and captures structured results."""

    async def run_tests(
        self,
        repo_path: str,
        *,
        test_command: str = "python -m pytest --tb=short -q",
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Run the test suite and return structured results."""
        logger.info("🧪 TestRunner: executing tests in %s", repo_path)
        started = time.time()

        try:
            result = subprocess.run(
                test_command.split(),
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.time() - started
            stdout = result.stdout[-2000:] if result.stdout else ""
            stderr = result.stderr[-1000:] if result.stderr else ""

            # Parse pytest output for pass/fail counts
            passed = 0
            failed = 0
            for line in stdout.splitlines():
                if "passed" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "passed" and i > 0:
                            try:
                                passed = int(parts[i - 1])
                            except ValueError:
                                pass
                        if p == "failed" and i > 0:
                            try:
                                failed = int(parts[i - 1])
                            except ValueError:
                                pass

            return {
                "all_passed": result.returncode == 0,
                "return_code": result.returncode,
                "passed": passed,
                "failed": failed,
                "duration_s": round(duration, 2),
                "summary": stdout.splitlines()[-1] if stdout.strip() else "no output",
                "stderr_tail": stderr[-500:],
            }

        except subprocess.TimeoutExpired:
            return {"all_passed": False, "error": "timeout", "duration_s": timeout}
        except (OSError, RuntimeError) as e:
            record_degradation("test_runner", e, action="test execution failed")
            return {"all_passed": False, "error": str(e), "duration_s": time.time() - started}
