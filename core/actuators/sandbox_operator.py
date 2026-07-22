"""core/actuators/sandbox_operator.py
==================================
The motor organ for a Person-in-a-Box.
Allows Aura to synthesize, run, and debug its own arbitrary scripts in a
sandboxed environment. Grounds success/failure signals into Heartstone Values
and the Liquid Substrate.

Hardening (CP126): synthesized code is AST-validated (shared gate with the
code-execution actuator) before it is ever written or run; the sandbox root is
a confined, private-mode trust root; timeout and code/output sizes are bounded;
failed scripts are reaped under a retention/quota policy; local paths are not
returned; and the affect-grounding evidence is sanitized and bounded before it
reaches Heartstone.
"""

import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from typing import Any

from core.actuators.code_execution_actuator import code_is_ast_safe
from core.runtime.service_registry import get_runtime_service
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.SandboxOperator")

_MAX_CODE_BYTES = 512 * 1024
_MAX_OUTPUT_CHARS = 32 * 1024
_MAX_EVIDENCE_CHARS = 2000
_MIN_TIMEOUT_S = 1.0
_MAX_TIMEOUT_S = 300.0
_DEFAULT_TIMEOUT_S = 10.0
_SANDBOX_RETENTION_S = 3600.0
_SANDBOX_MAX_FILES = 100


def _clamp_timeout(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S
    if not math.isfinite(num):
        return _DEFAULT_TIMEOUT_S
    return max(_MIN_TIMEOUT_S, min(_MAX_TIMEOUT_S, num))


def _bound_output(text: Any) -> str:
    s = str(text or "")
    return s if len(s) <= _MAX_OUTPUT_CHARS else s[:_MAX_OUTPUT_CHARS] + "\n...[truncated]"


def _safe_evidence(text: Any) -> str:
    """Sanitize untrusted program output before it grounds affect."""
    s = "".join(ch for ch in str(text or "") if ch == "\n" or ch == "\t" or ch >= " ")
    return s[:_MAX_EVIDENCE_CHARS]


class SandboxOperator:
    """The motor organ for a Person-in-a-Box: synthesize, run, and debug tools."""

    def __init__(self, sandbox_dir: str | None = None):
        from core.runtime.flags import FlagKind, declare

        configured_dir = str(
            declare(
                "AURA_SANDBOX_DIR",
                kind=FlagKind.STRING,
                default="",
                description="Root directory for synthesized-tool sandbox execution",
                owner="core.actuators.sandbox_operator",
            ).value()
        )
        self.sandbox_dir = self._resolve_trust_root(sandbox_dir or configured_dir)

    @staticmethod
    def _resolve_trust_root(requested: str) -> str:
        """Confine the sandbox to a private-mode directory we own."""
        default = os.path.join(tempfile.gettempdir(), "aura_sandbox")
        root = os.path.realpath(os.path.abspath(requested or default))
        os.makedirs(root, exist_ok=True)
        try:
            os.chmod(root, 0o700)  # owner-only — no group/other read/exec
        except OSError as exc:
            logger.debug("Could not tighten sandbox dir mode: %s", exc)
        return root

    def execute_synthesized_tool(
        self, code: str, timeout_s: float = _DEFAULT_TIMEOUT_S, *, expected_output: str | None = None
    ) -> dict[str, Any]:
        """Validate, run, and ground a synthesized Python tool.

        The code is AST-validated and size-checked BEFORE anything is written to
        disk or executed, so an unsafe or oversized script never reaches a
        subprocess.
        """
        if not isinstance(code, str) or not code.strip():
            return self._refused("empty or non-string code")
        if len(code.encode("utf-8", errors="ignore")) > _MAX_CODE_BYTES:
            return self._refused(f"code exceeds the {_MAX_CODE_BYTES}-byte sandbox limit")
        if not code_is_ast_safe(code, network_access=False):
            return self._refused("code failed AST safety validation (banned import or call)")

        timeout_s = _clamp_timeout(timeout_s)
        self._prune_sandbox()

        with tempfile.NamedTemporaryFile(suffix=".py", dir=self.sandbox_dir, delete=False) as temp_file:
            temp_file.write(code.encode("utf-8"))
            temp_path = temp_file.name

        success = False
        result_dict: dict[str, Any] = {}
        try:
            result = get_subprocess_gateway().run(
                [sys.executable, temp_path],
                timeout=timeout_s,
                cwd=self.sandbox_dir,
                source="sandbox_operator",
            )
            success = result.returncode == 0
            stdout, stderr, exit_code = _bound_output(result.stdout), _bound_output(result.stderr), result.returncode
        except subprocess.TimeoutExpired as e:
            stdout = _bound_output(e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))
            stderr = _bound_output((e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")) + f"\nExecution timed out after {timeout_s}s.")
            exit_code = -1
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            stdout, stderr, exit_code = "", f"Subprocess launch error: {e}", -2

        # Postcondition: exit-zero alone is not proof the tool did its job.
        if success and expected_output is not None and expected_output not in stdout:
            success = False
            stderr = (stderr + "\n[postcondition] expected output not found in stdout.").strip()

        result_dict = {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "sandbox_file": os.path.basename(temp_path),  # basename only, never the abs path
        }

        # Keep only failing scripts for inspection; successes are removed.
        if success and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as err:
                logger.debug("Failed to remove temp sandbox file %s: %s", temp_path, err)

        self._ground_affect(success, exit_code, stderr)
        return result_dict

    def _refused(self, reason: str) -> dict[str, Any]:
        # A refusal is not an execution: it must not move affect.
        return {"success": False, "stdout": "", "stderr": f"Refused: {reason}", "exit_code": -3, "refused": True}

    def _ground_affect(self, success: bool, exit_code: int, stderr: str) -> None:
        """Ground the outcome into Heartstone/Substrate with sanitized evidence."""
        try:
            from core.affect.heartstone_values import get_heartstone_values
            hv = get_heartstone_values()

            delta_curiosity = 0.0
            delta_frustration = 0.0
            if success:
                hv.on_sandbox_success()
                delta_frustration = -0.05
            else:
                # Untrusted program output is sanitized and bounded before it can
                # touch value/affect paths.
                hv.on_sandbox_failure(int(exit_code), _safe_evidence(stderr))
                delta_curiosity = +0.05
                delta_frustration = +0.08

            import asyncio
            substrate = get_runtime_service("liquid_substrate", default=None)
            if substrate:
                try:
                    get_task_tracker().create_task(
                        substrate.update(
                            delta_curiosity=delta_curiosity,
                            delta_frustration=delta_frustration,
                            _caller="sandbox_operator",
                        ),
                        name="sandbox_operator.substrate_update",
                    )
                except RuntimeError:
                    asyncio.run(
                        substrate.update(
                            delta_curiosity=delta_curiosity,
                            delta_frustration=delta_frustration,
                            _caller="sandbox_operator",
                        )
                    )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Sandbox affect grounding update failed: %s", exc)

    def _prune_sandbox(self) -> None:
        """Reap old / excess sandbox artifacts so failures don't accumulate."""
        try:
            entries = []
            for name in os.listdir(self.sandbox_dir):
                path = os.path.join(self.sandbox_dir, name)
                if os.path.isfile(path):
                    try:
                        entries.append((os.path.getmtime(path), path))
                    except OSError:
                        continue
            now = time.time()
            entries.sort()
            # Age-based expiry.
            survivors = []
            for mtime, path in entries:
                if (now - mtime) > _SANDBOX_RETENTION_S:
                    self._safe_unlink(path)
                else:
                    survivors.append(path)
            # Count-based quota (drop oldest beyond the cap).
            if len(survivors) > _SANDBOX_MAX_FILES:
                for path in survivors[: len(survivors) - _SANDBOX_MAX_FILES]:
                    self._safe_unlink(path)
        except OSError as exc:
            logger.debug("Sandbox prune failed: %s", exc)

    @staticmethod
    def _safe_unlink(path: str) -> None:
        try:
            os.remove(path)
        except OSError as exc:
            logger.debug("Sandbox unlink failed for %s: %s", path, exc)
