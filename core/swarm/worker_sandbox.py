"""core/swarm/worker_sandbox.py — Distributed Worker Sandboxing.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any, Dict, List, Optional

from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.SwarmSandbox")


class WorkerSandbox:
    """Executes code or commands inside a restricted sandboxed process."""

    def __init__(self, workspace_path: Optional[str] = None) -> None:
        self.workspace_path = workspace_path

    def run_command(
        self,
        argv: List[str],
        env: Optional[Dict[str, str]] = None,
        timeout: float = 30.0
    ) -> subprocess.CompletedProcess:
        """Runs a command with limited environment variables and directory confinement."""
        logger.info("🔒 Worker Sandbox running command: %s", " ".join(argv))
        
        # Enforce sandbox environment overrides
        sandbox_env = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "AURA_SANDBOX_MODE": "1",
            "AURA_ENABLE_CAMERA": "0",
            "AURA_ENABLE_MIC": "0",
        }
        if env:
            # Only allow non-sensitive environment variables
            for k, v in env.items():
                if "SECRET" not in k.upper() and "KEY" not in k.upper() and "TOKEN" not in k.upper():
                    sandbox_env[k] = v

        try:
            # Execute with low priority (nice) or standard resource limit
            # On macOS we can Nice the process
            proc_argv = argv
            if sys.platform != "win32":
                proc_argv = ["nice", "-n", "10"] + argv

            result = get_subprocess_gateway().run(
                proc_argv,
                cwd=self.workspace_path,
                env=sandbox_env,
                capture_output=True,
                timeout=timeout,
                source="swarm_worker_sandbox",
            )
            return result
        except subprocess.TimeoutExpired as exc:
            logger.error("Sandbox command timed out after %ds", timeout)
            raise TimeoutError(f"Sandbox command timed out: {exc}") from exc
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.error("Sandbox execution failed: %s", exc)
            raise RuntimeError(f"Sandbox execution failed: {exc}") from exc
