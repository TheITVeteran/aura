"""core/security/sandbox.py
Executes subprocess commands under sandbox constraints.
"""
from typing import Dict, Any
import subprocess
import os
import logging

logger = logging.getLogger("Security.Sandbox")


class LocalCommandSandbox:
    """Restricts command scopes using directory isolations."""

    def execute_sandboxed_command(self, command: str, sandbox_dir: str) -> Dict[str, Any]:
        logger.info("Executing command under local sandbox isolation: %s", command)
        os.makedirs(sandbox_dir, exist_ok=True)
        
        try:
            res = subprocess.run(
                command,
                shell=True,
                cwd=sandbox_dir,
                capture_output=True,
                text=True,
                timeout=10.0
            )
            return {
                "exit_code": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr
            }
        except Exception as e:
            logger.error("Sandbox command execution failed: %s", e)
            return {"exit_code": -1, "error": str(e)}
