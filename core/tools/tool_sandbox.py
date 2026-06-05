"""core/tools/tool_sandbox.py — Tool Sandbox Runner."""
from __future__ import annotations

import logging
from typing import Any, Dict

from core.sandbox.runner import run_untrusted
from core.tools.tool_manifest import ToolManifest

logger = logging.getLogger("Aura.ToolSandbox")


class ToolSandbox:
    """Safely executes third-party tool scripts in resource-constrained sub-processes."""

    @staticmethod
    def run(code: str, manifest: ToolManifest, params: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the tool code with the specified parameters using safe untrusted execution."""
        logger.info("Running tool %s version %s in sandbox", manifest.name, manifest.version)
        
        import ast
        
        # We wrap the tool execution code so that parameters are passed and returned cleanly.
        # Since __import__ is banned in the sandbox, we do not use any import statements in the wrapper.
        execution_wrapper = f"""
params = {repr(params)}

{code}

try:
    result = main(params)
    print("TOOL_OUT:" + repr(result))
except Exception as e:
    print("TOOL_ERR:" + str(e))
"""
        
        # Run untrusted python code
        raw_res = run_untrusted(
            code=execution_wrapper,
            timeout=10,  # 10s timeout
            mem_bytes=250 * 1024 * 1024,  # 250MB
        )
        
        if raw_res.get("status") != "ok":
            logger.error("Sandbox execution failed for tool %s: %s", manifest.name, raw_res.get("stderr"))
            return {
                "ok": False,
                "status": raw_res.get("status"),
                "error": raw_res.get("stderr") or "sandbox_execution_failed",
            }
        
        # Parse output from stdout
        stdout_lines = raw_res.get("stdout", "").strip().splitlines()
        for line in reversed(stdout_lines):
            if line.startswith("TOOL_OUT:"):
                try:
                    val = ast.literal_eval(line[len("TOOL_OUT:"):])
                    return {"ok": True, "result": val}
                except Exception as e:
                    logger.error("Failed to parse tool output: %s", e)
            elif line.startswith("TOOL_ERR:"):
                return {"ok": False, "error": line[len("TOOL_ERR:"):]}

        return {
            "ok": True,
            "stdout": raw_res.get("stdout"),
            "stderr": raw_res.get("stderr"),
        }
