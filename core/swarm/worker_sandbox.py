"""core/swarm/worker_sandbox.py — Swarm Sandboxed Environment.

Enforces absolute isolation of task executions to safeguard external actuation.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict

logger = logging.getLogger("Aura.WorkerSandbox")

_RUNNER_SOURCE = r'''
import json
import sys

payload = json.loads(sys.stdin.read())
safe_globals = {
    "__builtins__": {
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "hash": hash,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "str": str,
        "sum": sum,
        "tuple": tuple,
    }
}
safe_globals.update(payload.get("globals", {}))
exec(payload["code"], safe_globals)
print(json.dumps({"ok": True, "result": safe_globals.get("result")}, default=str))
'''


class WorkerSandbox:
    """Restricts directory access and execution scope for workers."""

    def __init__(self, allowed_directory: str) -> None:
        self.allowed_directory = os.path.abspath(allowed_directory)

    def is_safe_path(self, target_path: str) -> bool:
        """Validate if a file path stays strictly inside the sandbox directory."""
        abs_target = os.path.abspath(target_path)
        return abs_target.startswith(self.allowed_directory)

    def execute_code_sandboxed(self, code_str: str, globals_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python snippets in a child interpreter, not the live process."""
        logger.warning("Executing sandboxed python payload in isolated child process")
        runner_path = os.path.join(self.allowed_directory, f".aura_worker_{int(time.time() * 1000)}.py")
        if not self.is_safe_path(runner_path):
            return {"ok": False, "error": "sandbox runner path escaped allowed directory"}

        try:
            json.dumps(globals_dict)
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": f"sandbox globals must be JSON-serializable: {exc}"}

        try:
            from core.runtime.file_write_gateway import get_file_write_gateway
            from core.runtime.subprocess_gateway import get_subprocess_gateway

            get_file_write_gateway().write_text(
                runner_path,
                _RUNNER_SOURCE,
                source="worker_sandbox.write_runner",
            )
            proc = get_subprocess_gateway().run(
                [sys.executable, "-I", "-B", runner_path],
                input=json.dumps({"code": code_str, "globals": globals_dict}),
                timeout=5.0,
                source="worker_sandbox.execute_code",
            )
            if proc.returncode != 0:
                return {"ok": False, "error": proc.stderr[-1000:] if proc.stderr else "sandbox failed"}
            lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
            return json.loads(lines[-1]) if lines else {"ok": True, "result": None}
        except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError) as e:
            logger.error("Sandbox execution failed: %s", e)
            return {"ok": False, "error": str(e)}
        finally:
            try:
                os.remove(runner_path)
            except OSError:
                pass
