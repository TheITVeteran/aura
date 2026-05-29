"""core/actuators/code_execution_actuator.py
=========================================
Runs arbitrary Python code inside the sandbox.
Ensures safety by validating AST for banned imports/functions.
"""

import ast
import hashlib
from typing import Any
from core.actuators.actuator_registry import BaseActuator, ActuatorResult


class CodeExecutionActuator(BaseActuator):
    """Actuator that runs Python code in a sandbox with security controls."""

    requires_authority = True

    @property
    def name(self) -> str:
        return "code_execution"

    @property
    def description(self) -> str:
        return "Executes arbitrary Python code in a sandboxed environment with import and parameter validation."

    def validate_params(self, params: dict[str, Any]) -> bool:
        if not isinstance(params, dict) or "code" not in params:
            return False
        code = params["code"]
        if not isinstance(code, str):
            return False

        # Parse AST to check imports and safety
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, TypeError, MemoryError):
            return False  # Syntax error in code

        banned_modules = {
            "ctypes",
            "importlib",
            "os",
            "pathlib",
            "pty",
            "shutil",
            "subprocess",
            "sys",
        }

        # If network_access parameter is False or omitted, ban network imports
        network_access = bool(params.get("network_access", False))
        if not network_access:
            banned_modules.update({"socket", "urllib", "requests", "httpx", "http"})

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name.split('.')[0] in banned_modules:
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in banned_modules:
                    return False
                for name in node.names:
                    if name.name.split('.')[0] in banned_modules:
                        return False
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in (
                        "__import__",
                        "compile",
                        "eval",
                        "exec",
                        "globals",
                        "input",
                        "locals",
                        "open",
                        "vars",
                    ):
                        return False
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("system", "popen", "spawn", "remove", "unlink", "rmdir"):
                        return False

        return True

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not params.get("_aura_authorized"):
            return ActuatorResult(False, "Code execution requires ActuatorRegistry/AuthorityGateway authorization.", {})
        if not self.validate_params(params):
            return ActuatorResult(False, "Safety validation failed: code contains banned imports or functions.", {})

        from core.actuators.sandbox_operator import SandboxOperator
        operator = SandboxOperator()
        
        code = params["code"]
        timeout_s = float(params.get("timeout_s", 15.0))
        
        res = operator.execute_synthesized_tool(code, timeout_s=timeout_s)
        
        # Calculate digest for receipts
        output_combined = f"{res.get('stdout', '')}\n{res.get('stderr', '')}"
        output_hash = hashlib.sha256(output_combined.encode("utf-8")).hexdigest()
        
        updates = {
            "exit_code": res.get("exit_code"),
            "stdout": res.get("stdout"),
            "stderr": res.get("stderr"),
            "output_hash": output_hash,
            "success": res.get("success")
        }
        
        msg = f"Code executed successfully (exit code {res.get('exit_code')})." if res.get("success") else f"Code execution failed (exit code {res.get('exit_code')}): {res.get('stderr')}"
        
        return ActuatorResult(
            success=res.get("success", False),
            message=msg,
            updates=updates
        )
