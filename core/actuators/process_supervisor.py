"""core/actuators/process_supervisor.py
======================================
Spawns and manages long-running background processes.
Streams stdout/stderr to files under logs/processes/.
Integrates with SubstrateGovernor to scale clock speed during execution.
"""

import subprocess
import os
import shlex
import time
import uuid
import logging
from typing import Any, Dict
from core.actuators.actuator_registry import BaseActuator, ActuatorResult
from core.container import ServiceContainer
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.ProcessSupervisor")


class ProcessSupervisorActuator(BaseActuator):
    """Actuator to manage, inspect, and kill background jobs."""

    requires_authority = True

    def __init__(self, logs_dir: str = "logs/processes"):
        self.logs_dir = os.path.abspath(logs_dir)
        os.makedirs(self.logs_dir, exist_ok=True)
        self._processes: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "process_supervisor"

    @property
    def description(self) -> str:
        return "Spawns, lists, queries, and kills background tasks, streaming output to log files."

    def validate_params(self, params: dict[str, Any]) -> bool:
        if not isinstance(params, dict) or "action" not in params:
            return False
        action = params["action"]
        if action not in ("spawn", "list", "query", "kill"):
            return False
        if action == "spawn":
            if not bool(params.get("allow_spawn")):
                return False
            if "command" not in params:
                return False
            cmd = params["command"]
            if not isinstance(cmd, (str, list)):
                return False
            if isinstance(cmd, list) and not all(isinstance(x, str) for x in cmd):
                return False
        if action in ("query", "kill"):
            if "process_id" not in params or not isinstance(params["process_id"], str):
                return False
        return True

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not params.get("_aura_authorized"):
            return ActuatorResult(False, "Process supervision requires ActuatorRegistry/AuthorityGateway authorization.", {})
        if not self.validate_params(params):
            return ActuatorResult(False, "Parameter validation failed for process supervisor.", {})

        action = params["action"]

        if action == "spawn":
            return self._handle_spawn(params)
        elif action == "list":
            return self._handle_list()
        elif action == "query":
            return self._handle_query(params["process_id"])
        elif action == "kill":
            return self._handle_kill(params["process_id"])

        return ActuatorResult(False, f"Unsupported action: {action}", {})

    def _handle_spawn(self, params: dict[str, Any]) -> ActuatorResult:
        command = params["command"]
        if isinstance(command, str):
            command = shlex.split(command)
        if not command:
            return ActuatorResult(False, "Process command must not be empty.", {})
        process_id = f"proc_{uuid.uuid4().hex[:8]}"
        
        stdout_path = os.path.join(self.logs_dir, f"{process_id}_stdout.log")
        stderr_path = os.path.join(self.logs_dir, f"{process_id}_stderr.log")

        cwd = params.get("cwd") or os.getcwd()
        env = os.environ.copy()
        if "env" in params and isinstance(params["env"], dict):
            env.update(params["env"])

        try:
            proc = get_subprocess_gateway().spawn(
                command,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                cwd=cwd,
                env=env,
                text=True,
                start_new_session=True,
                source="process_supervisor",
            )
            
            self._processes[process_id] = {
                "proc": proc,
                "command": command,
                "start_time": time.time(),
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "log_streams": getattr(proc, "_aura_gateway_streams", ()),
                "cwd": cwd
            }

            # Scale up alacrity via SubstrateGovernor to request higher frequency focus ticks
            governor = ServiceContainer.get("governor", default=None)
            if governor:
                try:
                    governor.apply_volition_profile(3) # Scale to 20.0 Hz Focus mode
                except (RuntimeError, AttributeError, TypeError, ValueError) as ex:
                    logger.warning("Failed to scale substrate governor frequency: %s", ex)

            msg = f"Background process spawned successfully with ID: {process_id} (PID: {proc.pid})."
            return ActuatorResult(
                success=True,
                message=msg,
                updates={
                    "process_id": process_id,
                    "pid": proc.pid,
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path
                }
            )
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            return ActuatorResult(False, f"Process spawn failed: {e}", {})

    def _handle_list(self) -> ActuatorResult:
        result = {}
        for pid, pinfo in list(self._processes.items()):
            proc = pinfo["proc"]
            exit_code = proc.poll()
            is_running = exit_code is None
            
            result[pid] = {
                "command": pinfo["command"],
                "start_time": pinfo["start_time"],
                "is_running": is_running,
                "exit_code": exit_code,
                "stdout_path": pinfo["stdout_path"],
                "stderr_path": pinfo["stderr_path"]
            }
            
            # Clean up finished process file handles
            if not is_running:
                try:
                    for stream in pinfo.get("log_streams", ()):
                        stream.close()
                except (OSError, ValueError) as exc:
                    logger.debug("Process log close failed for %s: %s", pid, exc)

        # If all processes have completed, we can suggest stepping down alacrity if appropriate
        running_count = sum(1 for p in result.values() if p["is_running"])
        if running_count == 0:
            governor = ServiceContainer.get("governor", default=None)
            if governor:
                try:
                    governor.apply_volition_profile(1) # Revert to 10.0 Hz Reflective mode
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    logger.debug("Failed to restore substrate governor frequency: %s", exc)

        return ActuatorResult(True, f"Listed {len(result)} managed processes.", {"processes": result})

    def _handle_query(self, process_id: str) -> ActuatorResult:
        if process_id not in self._processes:
            return ActuatorResult(False, f"Process with ID {process_id} not found.", {})

        pinfo = self._processes[process_id]
        proc = pinfo["proc"]
        exit_code = proc.poll()
        is_running = exit_code is None

        # Clean file handles if completed
        if not is_running:
            try:
                for stream in pinfo.get("log_streams", ()):
                    stream.close()
            except (OSError, ValueError) as exc:
                logger.debug("Process log close failed for %s: %s", process_id, exc)

        details = {
            "process_id": process_id,
            "pid": proc.pid,
            "command": pinfo["command"],
            "start_time": pinfo["start_time"],
            "is_running": is_running,
            "exit_code": exit_code,
            "stdout_path": pinfo["stdout_path"],
            "stderr_path": pinfo["stderr_path"]
        }

        # Try to read some lines of stdout/stderr as a preview
        stdout_preview = ""
        stderr_preview = ""
        try:
            if os.path.exists(pinfo["stdout_path"]):
                with open(pinfo["stdout_path"], "r", encoding="utf-8") as f:
                    # Read last 20 lines
                    lines = f.readlines()
                    stdout_preview = "".join(lines[-20:])
            if os.path.exists(pinfo["stderr_path"]):
                with open(pinfo["stderr_path"], "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    stderr_preview = "".join(lines[-20:])
        except (OSError, ValueError) as e:
            logger.warning("Failed to read process log preview: %s", e)

        details["stdout_preview"] = stdout_preview
        details["stderr_preview"] = stderr_preview

        msg = f"Process {process_id} is running (PID: {proc.pid})." if is_running else f"Process {process_id} completed with exit code {exit_code}."
        return ActuatorResult(True, msg, {"process_details": details})

    def _handle_kill(self, process_id: str) -> ActuatorResult:
        if process_id not in self._processes:
            return ActuatorResult(False, f"Process with ID {process_id} not found.", {})

        pinfo = self._processes[process_id]
        proc = pinfo["proc"]
        exit_code = proc.poll()

        if exit_code is not None:
            return ActuatorResult(True, f"Process {process_id} was already stopped (exit code {exit_code}).", {"exit_code": exit_code})

        try:
            proc.terminate()
            # Wait up to 3 seconds for graceful shutdown
            for _ in range(30):
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
            else:
                proc.kill()
                proc.wait()

            exit_code = proc.poll()
            
            try:
                pinfo["stdout_file"].close()
                pinfo["stderr_file"].close()
            except (OSError, ValueError) as exc:
                logger.debug("Process log close failed for %s: %s", process_id, exc)

            # Update alacrity mapping if no running tasks left
            self._handle_list()

            return ActuatorResult(True, f"Process {process_id} killed successfully.", {"exit_code": exit_code})
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            return ActuatorResult(False, f"Failed to kill process {process_id}: {e}", {})
