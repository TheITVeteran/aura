"""core/body/terminal_motor.py
Terminal execution motor. Executes shell commands and processes safely.
"""
import logging
import shlex
from subprocess import SubprocessError
from typing import Any, Dict

from core.body.motor_controller import BaseMotor
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Body.TerminalMotor")

_TERMINAL_MOTOR_ERRORS = (OSError, RuntimeError, SubprocessError, TimeoutError, TypeError, ValueError)


class TerminalMotor(BaseMotor):
    """Actuator for running commands in the system shell."""

    @property
    def name(self) -> str:
        return "terminal"

    async def actuate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        command = params.get("argv") or params.get("command")
        cwd = params.get("cwd", ".")
        timeout = params.get("timeout", 5.0)

        if not command:
            return {"status": "error", "message": "Missing command string"}

        argv = [str(part) for part in command] if isinstance(command, (list, tuple)) else shlex.split(str(command))
        if not argv:
            return {"status": "error", "message": "Command produced no argv entries"}

        logger.info("Executing terminal command: %s (cwd=%s)", argv, cwd)

        try:
            res = await get_subprocess_gateway().run_async(
                argv,
                cwd=cwd,
                timeout=timeout,
                source="body.terminal_motor",
            )
            return {
                "status": "success",
                "exit_code": res.returncode,
                "stdout": res.stdout[:5000],  # Truncate overly long outputs
                "stderr": res.stderr[:2000]
            }
        except TimeoutError:
            return {"status": "timeout", "message": f"Command timed out after {timeout}s"}
        except _TERMINAL_MOTOR_ERRORS as e:
            record_degradation("body.terminal_motor", e)
            logger.error("Terminal motor execution failed: %s", e)
            return {"status": "error", "message": str(e)}
