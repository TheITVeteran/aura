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

**Honest isolation bound (CP126 f52d8430).** This is NOT an OS sandbox. The
script runs as Aura's own user and interpreter, so it inherits filesystem,
keychain and device reach that no in-process check can revoke. What this module
actually provides is *constrained execution*: an AST gate that refuses
dangerous constructs before anything is written, a scrubbed environment, POSIX
resource limits (CPU, address space, file size, descriptor and process counts),
its own process group so descendants can be reaped, and a bounded timeout. Real
isolation needs a container, a jail or a separate low-privilege user; until one
exists, ``isolation_level`` reports ``"constrained_process"`` and never
``"sandboxed"``, so no caller can mistake this for containment.

CP126 f52d8430 / 84fc4f9d / 0f681b67 / 688c0259.
"""

import logging
import math
import os
import signal
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

#: What this module honestly provides. Not "sandboxed" (CP126 f52d8430).
ISOLATION_LEVEL = "constrained_process"

#: POSIX resource limits applied to the child. These are real kernel-enforced
#: bounds, unlike the AST gate which only inspects source.
_RLIMIT_CPU_S = 30
_RLIMIT_ADDRESS_SPACE_BYTES = 2 * 1024 * 1024 * 1024   # 2 GiB
_RLIMIT_FILE_SIZE_BYTES = 64 * 1024 * 1024             # 64 MiB
_RLIMIT_OPEN_FILES = 128
_RLIMIT_PROCESSES = 32

#: Environment handed to the child: no inherited secrets, no proxy settings.
_SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "TZ")


def _child_preexec() -> None:
    """Apply POSIX limits and start a new process group in the child.

    Runs between fork and exec. A new process group (CP126 84fc4f9d) is what
    makes it possible to reap the whole descendant tree on timeout rather than
    only the direct child.
    """
    try:
        os.setsid()
    except (AttributeError, OSError):
        pass
    try:
        import resource

        for limit, value in (
            (resource.RLIMIT_CPU, _RLIMIT_CPU_S),
            (resource.RLIMIT_AS, _RLIMIT_ADDRESS_SPACE_BYTES),
            (resource.RLIMIT_FSIZE, _RLIMIT_FILE_SIZE_BYTES),
            (resource.RLIMIT_NOFILE, _RLIMIT_OPEN_FILES),
            (getattr(resource, "RLIMIT_NPROC", resource.RLIMIT_NOFILE), _RLIMIT_PROCESSES),
            (getattr(resource, "RLIMIT_CORE", resource.RLIMIT_FSIZE), 0),
        ):
            try:
                soft, hard = resource.getrlimit(limit)
                ceiling = value if hard in (resource.RLIM_INFINITY, -1) else min(value, hard)
                resource.setrlimit(limit, (ceiling, ceiling))
            except (OSError, ValueError):
                continue
    except ImportError:
        pass


def _scrubbed_env() -> dict[str, str]:
    """A minimal environment, so the child inherits no ambient credentials."""
    env = {key: os.environ[key] for key in _SAFE_ENV_KEYS if key in os.environ}
    env.setdefault("PATH", "/usr/bin:/bin")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["AURA_SANDBOX"] = "1"
    return env


def _reap_process_group(pgid: int | None, process: Any = None) -> dict[str, Any]:
    """Terminate a process group and report whether it actually died.

    CP126 84fc4f9d: the operator relied on gateway timeout behaviour and
    recorded no process-group identity or evidence that spawned children were
    terminated, so a timed-out script could leave descendants running.
    """
    receipt: dict[str, Any] = {"pgid": pgid, "attempted": False, "reaped": None}
    if not pgid or not hasattr(os, "killpg"):
        receipt["reason"] = "no process group recorded"
        return receipt
    # Never signal our own group: that would take Aura down with the script.
    try:
        if pgid == os.getpgrp():
            receipt["reason"] = "refused to signal aura's own process group"
            return receipt
    except OSError:
        pass
    receipt["attempted"] = True
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
            receipt["signal"] = sig.name
        except ProcessLookupError:
            receipt["reaped"] = True
            receipt["signal"] = sig.name
            return receipt
        except (OSError, PermissionError) as exc:
            # On macOS a group whose members are zombies answers EPERM rather
            # than ESRCH, so the signal result alone cannot decide this.
            receipt["signal_error"] = f"{type(exc).__name__}: {exc}"
            break
        time.sleep(0.05)

    # Confirm by STATE, not by the signal's return: the direct child having
    # exited is the evidence that matters, and the group is then unaddressable.
    if process is not None:
        try:
            process.wait(timeout=2)
            receipt["reaped"] = True
            receipt["confirmed_by"] = "child_exit"
            return receipt
        except Exception:  # noqa: BLE001 - any wait failure leaves it unconfirmed
            pass
    try:
        os.killpg(pgid, 0)
        receipt["reaped"] = False
        receipt["confirmed_by"] = "group_still_addressable"
    except ProcessLookupError:
        receipt["reaped"] = True
        receipt["confirmed_by"] = "group_gone"
    except OSError as exc:
        receipt["reaped"] = None
        receipt["confirmed_by"] = f"indeterminate: {type(exc).__name__}"
    return receipt


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
        # CP126 84fc4f9d: spawn into its OWN session/process group with real
        # POSIX limits, so a timeout can reap the whole descendant tree and
        # produce evidence that it did.
        pgid: int | None = None
        reap_receipt: dict[str, Any] = {"attempted": False, "reaped": None}
        process = None
        try:
            process = get_subprocess_gateway().spawn(
                [sys.executable, "-I", "-S", temp_path],
                cwd=self.sandbox_dir,
                env=_scrubbed_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                preexec_fn=_child_preexec,
                source="sandbox_operator",
            )
            # With start_new_session=True the child IS its own session and
            # group leader, so its pgid is its pid by definition. Calling
            # os.getpgid() here races the child's setsid and can return OUR
            # group — which would aim the reap at Aura itself.
            pgid = process.pid
            try:
                raw_out, raw_err = process.communicate(timeout=timeout_s)
                exit_code = process.returncode
                success = exit_code == 0
                stdout, stderr = _bound_output(raw_out), _bound_output(raw_err)
            except subprocess.TimeoutExpired:
                reap_receipt = _reap_process_group(pgid, process)
                try:
                    raw_out, raw_err = process.communicate(timeout=5)
                except (subprocess.TimeoutExpired, ValueError, OSError):
                    raw_out, raw_err = "", ""
                stdout = _bound_output(raw_out)
                stderr = _bound_output(
                    (raw_err or "")
                    + f"\nExecution timed out after {timeout_s}s."
                    + (
                        "\n[cleanup] descendant process group reaped."
                        if reap_receipt.get("reaped")
                        else "\n[cleanup] WARNING: descendants may still be running "
                        f"({reap_receipt.get('error') or 'unconfirmed'})."
                    )
                )
                exit_code = -1
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            stdout, stderr, exit_code = "", f"Subprocess launch error: {e}", -2
        finally:
            if process is not None and process.poll() is None:
                reap_receipt = _reap_process_group(pgid, process)

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
            # CP126 f52d8430: name what this actually is, so no caller reads
            # "sandbox" as containment.
            "isolation_level": ISOLATION_LEVEL,
            # CP126 84fc4f9d: evidence about the descendant tree, not a hope.
            "process_group": pgid,
            "cleanup": reap_receipt,
        }

        # Keep only failing scripts for inspection; successes are removed.
        if success and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as err:
                logger.debug("Failed to remove temp sandbox file %s: %s", temp_path, err)

        # CP126 0f681b67: an exit code is not a task outcome. Affect only
        # moves when the caller declared a postcondition we could actually
        # check; otherwise the result is recorded as ungrounded.
        verified = expected_output is not None
        result_dict["outcome_verified"] = verified
        if verified:
            self._ground_affect(success, exit_code, stderr)
        else:
            result_dict["affect_grounded"] = False
            logger.debug(
                "Sandbox result not grounded into affect: no expected_output "
                "postcondition was supplied, so exit code %s is not evidence of "
                "task success.", exit_code,
            )
        return result_dict

    def _refused(self, reason: str) -> dict[str, Any]:
        # A refusal is not an execution: it must not move affect.
        return {"success": False, "stdout": "", "stderr": f"Refused: {reason}", "exit_code": -3, "refused": True}

    def _ground_affect(self, success: bool, exit_code: int, stderr: str) -> bool:
        """Ground a VERIFIED outcome into Heartstone/Substrate.

        CP126 0f681b67: any zero exit reduced frustration and any non-zero
        raised curiosity and frustration, regardless of intent, expected result
        or whether the task had actually been done. Callers reach this only
        after a postcondition was checked.

        CP126 688c0259: the substrate update was fired into a detached task and
        never awaited, so a failure there was invisible and the update was not
        bound to the execution result. It is now awaited and its success is
        returned.
        """
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

            substrate = get_runtime_service("liquid_substrate", default=None)
            if substrate is None:
                return True
            return self._apply_substrate_delta(
                substrate, delta_curiosity, delta_frustration
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Sandbox affect grounding update failed: %s", exc)
            return False

    @staticmethod
    def _apply_substrate_delta(
        substrate: Any, delta_curiosity: float, delta_frustration: float
    ) -> bool:
        """Apply the substrate delta and report whether it actually landed."""
        import asyncio

        coro = substrate.update(
            delta_curiosity=delta_curiosity,
            delta_frustration=delta_frustration,
            _caller="sandbox_operator",
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop on this thread: run it to completion so the caller's
            # result and the substrate state cannot diverge.
            try:
                asyncio.run(coro)
                return True
            except (RuntimeError, TypeError, ValueError) as exc:
                logger.warning("Sandbox substrate update failed: %s", exc)
                return False

        # On a running loop this method is synchronous, so the update is
        # tracked and its failure is surfaced rather than swallowed.
        task = get_task_tracker().create_task(
            coro, name="sandbox_operator.substrate_update"
        )

        def _report(done: Any) -> None:
            if done.cancelled():
                return
            exc = done.exception()
            if exc is not None:
                logger.warning("Sandbox substrate update failed: %s", exc)

        task.add_done_callback(_report)
        return True

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
