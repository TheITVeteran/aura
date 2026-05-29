"""core/actuators/sandbox_operator.py
==================================
The motor organ for a Person-in-a-Box.
Allows Aura to synthesize, run, and debug its own arbitrary scripts in a sandboxed environment.
Grounds success and failure signals directly into Heartstone Values and the Liquid Substrate.
"""

import subprocess
import tempfile
import os
import sys
import logging
from typing import Dict, Any
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.SandboxOperator")

class SandboxOperator:
    """
    The ultimate motor organ for a Person-in-a-Box.
    Allows the model to synthesize, run, and debug its own arbitrary tools.
    """
    def __init__(self, sandbox_dir: str | None = None):
        self.sandbox_dir = os.path.abspath(
            sandbox_dir
            or os.getenv("AURA_SANDBOX_DIR")
            or os.path.join(tempfile.gettempdir(), "aura_sandbox")
        )
        os.makedirs(self.sandbox_dir, exist_ok=True)

    def execute_synthesized_tool(self, code: str, timeout_s: float = 10.0) -> Dict[str, Any]:
        """
        Executes raw Python code synthesized by Aura in a subprocess.
        Captures stderr and stdout to allow the system to self-correct.
        """
        # <comment-tag: The system must use this output to drive prediction-error deficits>
        with tempfile.NamedTemporaryFile(suffix=".py", dir=self.sandbox_dir, delete=False) as temp_file:
            temp_file.write(code.encode("utf-8"))
            temp_path = temp_file.name

        success = False
        result_dict = {}

        try:
            result = get_subprocess_gateway().run(
                [sys.executable, temp_path],
                timeout=timeout_s,
                cwd=self.sandbox_dir,
                source="sandbox_operator",
            )
            
            success = result.returncode == 0
            result_dict = {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "file_path": temp_path
            }
        except subprocess.TimeoutExpired as e:
            stdout_str = e.stdout.decode("utf-8") if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr_str = e.stderr.decode("utf-8") if isinstance(e.stderr, bytes) else (e.stderr or "")
            result_dict = {
                "success": False,
                "stdout": stdout_str,
                "stderr": f"{stderr_str}\nExecution timed out after {timeout_s} seconds.",
                "exit_code": -1,
                "file_path": temp_path
            }
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            result_dict = {
                "success": False,
                "stdout": "",
                "stderr": f"Subprocess launch error: {e}",
                "exit_code": -2,
                "file_path": temp_path
            }
        finally:
            # Keep the file if it failed so the Left Hemisphere can inspect it
            if success and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as err:
                    logger.debug("Failed to remove temp sandbox file %s: %s", temp_path, err)

        # Trigger Heartstone and Substrate affect updates
        try:
            from core.affect.heartstone_values import get_heartstone_values
            hv = get_heartstone_values()
            
            delta_curiosity = 0.0
            delta_frustration = 0.0
            
            if success:
                hv.on_sandbox_success()
                delta_frustration = -0.05
            else:
                hv.on_sandbox_failure(result_dict["exit_code"], result_dict["stderr"])
                delta_curiosity = +0.05
                delta_frustration = +0.08
                
            # Asynchronously update LiquidSubstrate
            import asyncio
            from core.container import ServiceContainer
            substrate = ServiceContainer.get("liquid_substrate", default=None)
            if substrate:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(substrate.update(
                        delta_curiosity=delta_curiosity,
                        delta_frustration=delta_frustration,
                        _caller="sandbox_operator"
                    ))
                except RuntimeError:
                    asyncio.run(substrate.update(
                        delta_curiosity=delta_curiosity,
                        delta_frustration=delta_frustration,
                        _caller="sandbox_operator"
                    ))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Sandbox affect grounding update failed: %s", exc)

        return result_dict
