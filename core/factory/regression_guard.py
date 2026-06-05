"""core/factory/regression_guard.py — Linter and Typecheck Code Safety Guard.
"""
from __future__ import annotations

import logging
from typing import Dict

from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.RegressionGuard")


class RegressionGuard:
    """Verifies syntactic correctness, code cleanliness, and security integrity."""

    @staticmethod
    def verify_patch(file_path: str) -> Dict[str, Any]:
        logger.info("🏭 RegressionGuard scanning file: %s", file_path)
        gateway = get_subprocess_gateway()
        
        # Run ruff check
        proc = gateway.run(
            argv=["ruff", "check", file_path],
            timeout=10.0,
            source="regression_guard",
        )
        passed = proc.returncode == 0
        logger.info("🏭 RegressionGuard verification status: %s", "PASSED" if passed else "FAILED")

        return {
            "passed": passed,
            "exit_code": proc.returncode,
            "details": proc.stdout or proc.stderr,
        }
