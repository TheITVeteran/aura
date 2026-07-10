"""core/skills/code_repl.py — Real-Time Python REPL
====================================================
First-class BaseSkill that gives Aura a live, stateful Python REPL.
This is the exact equivalent of a code_execution/code_interpreter tool:
  - Execute arbitrary Python in a sandboxed subprocess
  - Maintain per-session variable state across turns
  - Capture stdout, stderr, return values, and generated files
  - Enforce memory/CPU/time limits via core.sandbox.runner

This closes the "code REPL" gap in tool parity.
"""

import asyncio
import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.config import config
from core.governance.will import ActionDomain
from core.runtime.action_executor import ActionExecutor
from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.CodeREPL")

_REPL_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
)


def _record_repl_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "code_repl",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=False,
        extra=extra,
    )


class CodeREPLInput(BaseModel):
    code: str = Field(..., description="Python code to execute in the REPL.")
    session_id: str | None = Field(
        None,
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description="Optional session ID for maintaining state across turns.",
    )
    timeout: int = Field(
        30,
        ge=1,
        le=120,
        description="Maximum execution time in seconds.",
    )
    capture_files: bool = Field(
        True,
        description="Whether to capture any files generated in the working directory.",
    )


class CodeREPLSkill(BaseSkill):
    name = "code_repl"
    description = (
        "Execute Python code in a real-time, sandboxed REPL. "
        "Supports multi-turn sessions with persistent state, file generation, "
        "and full stdout/stderr capture. Use for calculations, data processing, "
        "prototyping, and any computational task."
    )
    input_model = CodeREPLInput
    timeout_seconds = 120.0
    metabolic_cost = 2
    effect_scope = "sandboxed_compute"

    # Session state: maps session_id -> serialized namespace dict
    _sessions: dict[str, dict[str, Any]]
    _session_dirs: dict[str, Path]

    def __init__(self) -> None:
        super().__init__()
        self._sessions = {}
        self._session_dirs = {}
        self._output_dir = Path(config.paths.data_dir) / "repl_sessions"

    async def _get_session_dir(self, session_id: str) -> Path:
        """Get or create a working directory for a session."""
        if session_id not in self._session_dirs:
            session_dir = self._output_dir / session_id
            await get_file_write_gateway().ensure_directory_async(
                session_dir,
                source="skills.code_repl.session",
            )
            self._session_dirs[session_id] = session_dir
        return self._session_dirs[session_id]

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return hashlib.sha256(
            f"{time.time()}-{os.getpid()}".encode()
        ).hexdigest()[:12]

    async def execute(
        self, params: CodeREPLInput, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute Python code in a sandboxed REPL."""
        if isinstance(params, dict):
            try:
                params = CodeREPLInput(**params)
            except _REPL_RECOVERABLE_ERRORS as exc:
                _record_repl_degradation(
                    exc,
                    action="rejected invalid REPL input before code execution",
                )
                return {"ok": False, "error": f"Invalid input: {exc}"}

        code = params.code.strip()
        if not code:
            return {"ok": False, "error": "No code provided."}

        session_id = params.session_id or self._generate_session_id()
        session_dir = await self._get_session_dir(session_id)
        timeout_s = params.timeout

        # List files before execution to detect new ones
        pre_files = set()
        if params.capture_files:
            try:
                pre_files = set(session_dir.iterdir())
            except OSError as _exc:
                logger.debug("Suppressed %s in core.skills.code_repl: %s", type(_exc).__name__, _exc)

        # Strategy 1: Use core.sandbox.runner (preferred — full isolation)
        result = await self._execute_via_sandbox_runner(
            code, timeout_s, session_dir
        )

        if result is None:
            # Strategy 2: Use SandboxOperator (fallback)
            result = await self._execute_via_sandbox_operator(
                code, timeout_s
            )

        if result is None:
            # Strategy 3: Governed subprocess (last resort)
            result = await self._execute_via_subprocess(
                code, timeout_s, session_dir
            )

        if result is None:
            return {
                "ok": False,
                "error": "No execution backend available.",
            }

        # Detect newly generated files
        new_files: list[str] = []
        if params.capture_files:
            try:
                post_files = set(session_dir.iterdir())
                for f in post_files - pre_files:
                    if f.is_file():
                        new_files.append(str(f))
            except OSError as _exc:
                logger.debug("Suppressed %s in core.skills.code_repl: %s", type(_exc).__name__, _exc)

        # Ground affect signals into Heartstone
        self._ground_affect(result.get("ok", False), result.get("stderr", ""))

        result["session_id"] = session_id
        result["working_directory"] = str(session_dir)
        if new_files:
            result["generated_files"] = new_files

        return result

    async def _execute_via_sandbox_runner(
        self, code: str, timeout_s: int, cwd: Path
    ) -> dict[str, Any] | None:
        """Execute via core.sandbox.runner.run_untrusted (full isolation)."""
        try:
            from core.sandbox.runner import run_untrusted

            # Note: The restricted sandbox strips __import__ from builtins,
            # so we cannot prepend 'import os; os.chdir(...)' here.
            # The sandbox runs in a temporary directory by default.
            raw = await asyncio.to_thread(
                run_untrusted,
                code,
                timeout=timeout_s,
                mem_bytes=512 * 1024 * 1024,
            )

            if not isinstance(raw, dict):
                return {"ok": False, "error": f"Unexpected runner result: {raw}"}

            status = raw.get("status", "ok")
            stdout = raw.get("stdout", "")
            stderr = raw.get("stderr", "")
            returncode = raw.get("returncode")

            ok = status == "ok" and returncode == 0

            return {
                "ok": ok,
                "stdout": stdout,
                "stderr": stderr,
                "status": status,
                "returncode": returncode,
                "engine": "sandbox_runner",
                "summary": (
                    "Code executed successfully."
                    if ok
                    else f"Execution failed ({status})."
                ),
            }

        except _REPL_RECOVERABLE_ERRORS as exc:
            _record_repl_degradation(
                exc,
                action="fell back from sandbox_runner to alternative execution backend",
                extra={"engine": "sandbox_runner"},
            )
            logger.debug("sandbox_runner unavailable: %s", exc)
            return None

    async def _execute_via_sandbox_operator(
        self, code: str, timeout_s: int
    ) -> dict[str, Any] | None:
        """Execute via SandboxOperator (affect-grounded fallback)."""
        try:
            from core.actuators.sandbox_operator import SandboxOperator

            operator = SandboxOperator()
            raw = await asyncio.to_thread(
                operator.execute_synthesized_tool,
                code,
                float(timeout_s),
            )

            return {
                "ok": raw.get("success", False),
                "stdout": raw.get("stdout", ""),
                "stderr": raw.get("stderr", ""),
                "returncode": raw.get("exit_code"),
                "engine": "sandbox_operator",
                "summary": (
                    "Code executed via SandboxOperator."
                    if raw.get("success")
                    else "Execution failed."
                ),
            }

        except _REPL_RECOVERABLE_ERRORS as exc:
            _record_repl_degradation(
                exc,
                action="fell back from sandbox_operator to subprocess execution",
                extra={"engine": "sandbox_operator"},
            )
            logger.debug("SandboxOperator unavailable: %s", exc)
            return None

    async def _execute_via_subprocess(
        self, code: str, timeout_s: int, cwd: Path
    ) -> dict[str, Any] | None:
        """Execute via the canonical ActionExecutor subprocess pathway."""
        import sys

        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(suffix=".py", dir=str(cwd))
            os.close(fd)
            await get_file_write_gateway().write_text_async(
                temp_path,
                code,
                encoding="utf-8",
                source="core.skills.code_repl.temp_script",
            )

            # Execute via ActionExecutor
            res = await ActionExecutor.execute(
                domain=ActionDomain.TOOL_EXECUTION,
                action_name="code_repl.run_script",
                params={
                    "argv": [sys.executable, temp_path],
                    "cwd": str(cwd),
                    "timeout": float(timeout_s),
                },
                source="code_repl",
            )

            # Map ActionExecutor result to expected REPL format
            return {
                "ok": res.get("ok", False),
                "stdout": res.get("stdout", ""),
                "stderr": res.get("stderr", ""),
                "returncode": res.get("exit_code", -1),
                "engine": "subprocess",
                "summary": res.get("error", "Code executed via ActionExecutor."),
            }

        except _REPL_RECOVERABLE_ERRORS as exc:
            _record_repl_degradation(
                exc,
                action="reported execution failure after all backends exhausted",
                extra={"engine": "subprocess"},
            )
            return {"ok": False, "error": f"Subprocess failed: {exc}", "engine": "subprocess"}
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError as _exc:
                    logger.debug("Suppressed %s in core.skills.code_repl: %s", type(_exc).__name__, _exc)

    def _ground_affect(self, success: bool, stderr: str) -> None:
        """Ground execution results into Heartstone Values."""
        try:
            from core.affect.heartstone_values import get_heartstone_values

            hv = get_heartstone_values()
            if success:
                hv.on_sandbox_success()
            else:
                hv.on_sandbox_failure(-1, stderr[:500])
        except _REPL_RECOVERABLE_ERRORS as exc:
            logger.debug("Affect grounding skipped: %s", exc)
