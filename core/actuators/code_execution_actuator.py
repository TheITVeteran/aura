"""core/actuators/code_execution_actuator.py
=========================================
Runs arbitrary Python code inside the sandbox.
Ensures safety by validating AST for banned imports/functions.
"""

import ast
import hashlib
from typing import Any

from core.actuators.actuator_registry import ActuatorResult, BaseActuator
from core.actuators.authority import verify_actuator_authority

_BANNED_MODULES = {
    "ctypes", "importlib", "os", "pathlib", "pty", "shutil", "subprocess", "sys",
}
_BANNED_NETWORK_MODULES = {"socket", "urllib", "requests", "httpx", "http"}
_BANNED_CALLS = {
    "__import__", "compile", "eval", "exec", "globals", "input", "locals", "open", "vars",
}
_BANNED_ATTR_CALLS = {"system", "popen", "spawn", "remove", "unlink", "rmdir"}


def code_is_ast_safe(code: Any, *, network_access: bool = False) -> bool:
    """Shared AST safety gate for synthesized code.

    Rejects banned imports (filesystem/process/interpreter, plus network unless
    explicitly allowed) and dangerous call names. Used by both the code-execution
    actuator and the sandbox operator so no execution path skips the check.
    """
    if not isinstance(code, str):
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError, MemoryError):
        return False
    banned = set(_BANNED_MODULES)
    if not network_access:
        banned |= _BANNED_NETWORK_MODULES
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(n.name.split(".")[0] in banned for n in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in banned:
                return False
            if any(n.name.split(".")[0] in banned for n in node.names):
                return False
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BANNED_CALLS:
                return False
            if isinstance(node.func, ast.Attribute) and node.func.attr in _BANNED_ATTR_CALLS:
                return False
    return True


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
        return code_is_ast_safe(
            params["code"], network_access=bool(params.get("network_access", False))
        )

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        _authorized, _auth_reason = verify_actuator_authority(params, actuator=self.name)
        if not _authorized:
            return ActuatorResult(False, _auth_reason, {})
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
