import os
import logging
import shlex
from subprocess import SubprocessError
from typing import Any, Dict, Sequence

from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Security.Sandbox")

_SANDBOX_EXECUTION_ERRORS = (OSError, RuntimeError, SubprocessError, TimeoutError, TypeError, ValueError)


class LocalCommandSandbox:
    """Restricts command scopes using directory isolations."""

    def execute_sandboxed_command(self, command: str | Sequence[str], sandbox_dir: str) -> Dict[str, Any]:
        logger.info("Executing command under local sandbox isolation: %s", command)
        os.makedirs(sandbox_dir, exist_ok=True)
        argv = [str(part) for part in command] if isinstance(command, (list, tuple)) else shlex.split(str(command))
        if not argv:
            return {"exit_code": -1, "error": "empty command"}
        
        try:
            res = get_subprocess_gateway().run(
                argv,
                cwd=sandbox_dir,
                capture_output=True,
                timeout=10.0,
                source="security.sandbox",
                accelerator_capability="auto",
            )
            return {
                "exit_code": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr
            }
        except _SANDBOX_EXECUTION_ERRORS as e:
            record_degradation("security.sandbox", e)
            logger.error("Sandbox command execution failed: %s", e)
            return {"exit_code": -1, "error": str(e)}
