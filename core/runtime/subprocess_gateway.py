"""Canonical subprocess gateway.

All live runtime subprocess creation should flow through this module. Effectful
calls require an active governance context; explicitly read-only probes may opt
out while still receiving consistent validation and logging behavior.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import IO, Any

from core import governance_context as _governance_context
from core.runtime.shutdown_coordinator import (
    is_shutdown_requested,
    record_shutdown_admission_event,
)
from core.runtime.shutdown_execution import run_sync_shutdown_callable
from core.utils.task_tracker import (
    begin_shutdown_resource_creation_scope,
    end_shutdown_resource_creation_scope,
)

GovernanceViolation = _governance_context.GovernanceViolation

_EFFECT_DOMAINS = (
    "environment_action",
    "external_action",
    "tool_execution",
    "state_mutation",
    "file_write",
    "self_modification",
)
_OFFLINE_TOOLING_SOURCE_PREFIXES = (
    "benchmark_tooling:",
    "certification_tooling:",
    "maintenance_tooling:",
    "proof_tooling:",
    "training_tooling:",
)
_TEST_MODE_GOVERNANCE_BYPASS_PREFIXES = (
    "certification_tooling:",
    "proof_tooling:",
)
_DESKTOP_LONGRUN_COMMAND_MARKERS = (
    "challenges/nethack_challenge.py",
    "nethack_challenge.py",
    "run_dnu_agi_proof_battery.py",
    "run_longevity_soak.py",
    "aletheia_tier5",
    "run_aletheia",
)
logger = logging.getLogger("Aura.SubprocessGateway")


def _inferred_model_lane_claim(
    command: Sequence[str],
    *,
    source: str,
    timeout_s: float,
) -> Any | None:
    from core.runtime.model_lane_control import infer_model_process_claim

    return infer_model_process_claim(
        command,
        source=source,
        timeout_s=timeout_s,
    )


async def _reserve_model_lane_process(
    claim: Any,
) -> tuple[Any, Any]:
    from core.runtime.model_lane_control import prepare_model_lane_claim

    prepared: tuple[Any, Any] = await prepare_model_lane_claim(claim)
    return prepared


async def _cancel_model_lane_process(
    controller: Any,
    decision: Any,
    *,
    reason: str,
) -> None:
    try:
        await controller.cancel(decision, reason=reason)
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.error(
            "Model subprocess reservation cancellation failed transaction=%s: %s",
            getattr(decision, "transaction_id", ""),
            exc,
        )


def _model_command_requires_async(command: Sequence[str], *, source: str) -> None:
    claim = _inferred_model_lane_claim(command, source=source, timeout_s=30.0)
    if claim is not None:
        raise RuntimeError(
            "accelerator-owning subprocesses require run_async/spawn_async so "
            "their durable lane reservation can follow the child lifecycle"
        )


def _register_runtime_hygiene_process(
    proc: Any,
    *,
    kind: str,
    source: str,
    command: Sequence[str] | str,
) -> None:
    """Register gateway-spawned children with runtime hygiene when available."""

    try:
        from core.runtime.runtime_hygiene import get_runtime_hygiene

        if isinstance(command, str):
            command_text = command
        else:
            command_text = " ".join(str(part) for part in command)
        get_runtime_hygiene().register_process_handle(
            proc,
            kind=kind,
            name=source or kind,
            source=f"subprocess_gateway:{source or 'unknown'}",
            command=command_text,
        )
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("runtime hygiene registration skipped for subprocess gateway child: %s", exc)


def governance_runtime_active() -> bool:
    return bool(_governance_context.governance_runtime_active())


def require_governance(*args: Any, **kwargs: Any) -> Any:
    return _governance_context.require_governance(*args, **kwargs)


def _coerce_argv(argv: Sequence[str]) -> list[str]:
    if not isinstance(argv, (list, tuple)) or not argv:
        raise ValueError("argv must be a non-empty list or tuple")
    coerced = [str(part) for part in argv]
    if any(not part for part in coerced):
        raise ValueError("argv entries must not be empty")
    return coerced


def _coerce_cwd(cwd: str | os.PathLike[str] | None) -> str | None:
    if cwd is None:
        return None
    return str(Path(cwd).expanduser().resolve())


def _validate_read_only_source(source: str) -> None:
    if not isinstance(source, str) or source.strip() in {"", "unknown"}:
        raise ValueError("read-only subprocess probes require a specific source label")
    if "\n" in source or "\r" in source:
        raise ValueError("subprocess source label must be single-line")


def _truthy_env_value(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _effective_env_value(env: Mapping[str, str] | None, key: str) -> str | None:
    if env is not None and key in env:
        return str(env[key])
    return os.getenv(key)


def _desktop_safe_mode_requested(env: Mapping[str, str] | None) -> bool:
    return _truthy_env_value(_effective_env_value(env, "AURA_DESKTOP_RESOURCE_GUARD")) or _truthy_env_value(
        _effective_env_value(env, "AURA_SAFE_BOOT_DESKTOP")
    ) or _truthy_env_value(
        _effective_env_value(env, "AURA_LAUNCHED_FROM_APP")
    )


def _desktop_longrun_override(env: Mapping[str, str] | None) -> bool:
    return _truthy_env_value(_effective_env_value(env, "AURA_ALLOW_DESKTOP_LONGRUNS")) or _truthy_env_value(
        _effective_env_value(env, "AURA_ALLOW_DESKTOP_NETHACK")
    )


def _validate_desktop_safe_subprocess(
    command: Sequence[str] | str,
    *,
    env: Mapping[str, str] | None,
    source: str,
    operation: str,
) -> None:
    """Prevent desktop boot/chat sessions from launching proof-scale child jobs.

    Long environment batteries are valid proof tooling, but they are not part of
    the live user desktop lane. They can exceed desktop memory budgets when they
    are started by a stale shell, launch agent, or task handoff. An explicit
    operator opt-in keeps proof work possible while making false "normal desktop"
    launches fail closed.
    """
    if not _desktop_safe_mode_requested(env) or _desktop_longrun_override(env):
        return
    if isinstance(command, str):
        normalized = command
    else:
        normalized = " ".join(str(part) for part in command)
    lowered = normalized.lower()
    if any(marker in lowered for marker in _DESKTOP_LONGRUN_COMMAND_MARKERS):
        raise GovernanceViolation(
            f"{operation}:{source} denied desktop-safe long-run subprocess; "
            "set AURA_ALLOW_DESKTOP_LONGRUNS=1 for an intentional proof run"
        )


def _open_spawn_stream(path: str | os.PathLike[str], *, text: bool) -> IO[Any]:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(str(target), flags, 0o600)
    try:
        return os.fdopen(fd, "w", encoding="utf-8") if text else os.fdopen(fd, "wb")
    except (OSError, ValueError):
        os.close(fd)
        raise


def _validate_offline_tooling_bypass(
    *,
    offline_tooling: bool,
    source: str,
    command: Sequence[str],
    env: Mapping[str, str] | None = None,
) -> bool:
    """Allow named repo tooling to launch child processes outside live Aura.

    This is intentionally not a general governance bypass. It exists for CLI
    proof, certification, benchmark, maintenance, and training wrappers that
    orchestrate Aura from outside her live runtime. If live/strict governance is
    active, the bypass fails closed except for proof/certification harnesses
    running under AURA_TEST_MODE.
    """
    if not offline_tooling:
        return False
    if not any(source.startswith(prefix) for prefix in _OFFLINE_TOOLING_SOURCE_PREFIXES):
        raise ValueError(
            "offline subprocess tooling requires a source prefix of "
            f"{', '.join(_OFFLINE_TOOLING_SOURCE_PREFIXES)}"
        )
    if governance_runtime_active():
        is_certification_harness = any(
            source.startswith(prefix) for prefix in _TEST_MODE_GOVERNANCE_BYPASS_PREFIXES
        )
        explicit_test_mode = env is not None and str(env.get("AURA_TEST_MODE", "")) == "1"
        process_test_mode = os.getenv("AURA_TEST_MODE", "") == "1"
        if is_certification_harness and (process_test_mode or explicit_test_mode):
            logger.info(
                "offline subprocess tooling bypass (test-mode) source=%s argv0=%s argc=%s",
                source,
                command[0] if command else "",
                len(command),
            )
            return True
        raise GovernanceViolation(
            f"offline subprocess tooling bypass denied while live governance is active: {source}"
        )
    logger.info(
        "offline subprocess tooling bypass source=%s argv0=%s argc=%s",
        source,
        command[0] if command else "",
        len(command),
    )
    return True


def _require_effect_governance(operation: str) -> None:
    should_fail_closed = governance_runtime_active()
    token = require_governance(
        operation,
        strict=True,
        allowed_domains=_EFFECT_DOMAINS,
    )
    if should_fail_closed and (token is None or getattr(token, "domain", "") == "degraded"):
        raise GovernanceViolation(f"{operation} called outside governed context")


def _require_not_shutting_down(
    operation: str,
    *,
    read_only: bool,
    offline_tooling: bool,
    allow_during_shutdown: bool,
    resource_created: bool = False,
    bounded_completion: bool = False,
) -> None:
    """Block new live subprocess work after the process shutdown latch is set."""

    if not is_shutdown_requested():
        return
    # External proof/certification tools run in their own process and therefore
    # do not inherit the stopped runtime's latch. An in-process exception must
    # be explicit and may only be used for non-effectful inspection.
    if allow_during_shutdown and read_only and bounded_completion:
        if not resource_created:
            record_shutdown_admission_event(
                operation,
                resource_kind="subprocess",
                outcome="allowed_read_only",
                detail="explicit_shutdown_probe",
            )
        logger.warning("Allowing explicit shutdown-time subprocess probe: %s", operation)
        return
    record_shutdown_admission_event(
        operation,
        resource_kind="subprocess",
        outcome="crossed" if resource_created else "suppressed",
        detail="shutdown_latch",
    )
    raise GovernanceViolation(f"{operation} refused during runtime shutdown")


class SubprocessGateway:
    """Single owner for subprocess execution and spawning."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        read_only: bool = False,
        offline_tooling: bool = False,
        allow_during_shutdown: bool = False,
        capture_output: bool = True,
        input: str | None = None,
        check: bool = False,
        source: str = "unknown",
    ) -> subprocess.CompletedProcess[str]:
        command = _coerce_argv(argv)
        _model_command_requires_async(command, source=source)
        if read_only and not offline_tooling:
            _validate_read_only_source(source)
        offline_bypass = _validate_offline_tooling_bypass(
            offline_tooling=offline_tooling,
            source=source,
            command=command,
            env=env,
        )
        _require_not_shutting_down(
            f"subprocess_gateway.run:{source}",
            read_only=read_only,
            offline_tooling=offline_tooling,
            allow_during_shutdown=allow_during_shutdown,
            bounded_completion=True,
        )
        if not read_only and not offline_bypass:
            _require_effect_governance(f"subprocess_gateway.run:{source}")
        _validate_desktop_safe_subprocess(command, env=env, source=source, operation="run")
        resource_token = (
            begin_shutdown_resource_creation_scope()
            if is_shutdown_requested() and allow_during_shutdown and read_only
            else None
        )
        try:
            return subprocess.run(
                command,
                cwd=_coerce_cwd(cwd),
                env=dict(env) if env is not None else None,
                timeout=float(timeout),
                capture_output=bool(capture_output),
                input=input,
                text=True,
                check=bool(check),
                shell=False,
            )
        finally:
            if resource_token is not None:
                end_shutdown_resource_creation_scope(resource_token)

    def run_model_blocking(
        self,
        argv: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        read_only: bool = False,
        offline_tooling: bool = False,
        allow_during_shutdown: bool = False,
        capture_output: bool = True,
        input: str | None = None,
        check: bool = False,
        source: str = "unknown",
        model_lane_claim: Any | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run one governed model child from a synchronous CLI entrypoint.

        The async path performs the reservation, eviction, delegated fencing,
        process-group monitoring, and terminal release. Calling this method from
        an event-loop thread is refused so a synchronous caller cannot stall the
        runtime loop; async code must await ``run_async`` directly.
        """
        command = _coerce_argv(argv)
        claim = model_lane_claim or _inferred_model_lane_claim(
            command,
            source=source,
            timeout_s=float(timeout),
        )
        if claim is None:
            raise RuntimeError(
                f"run_model_blocking requires an attributable model claim: {source}"
            )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "run_model_blocking cannot block an active event loop; await run_async"
            )
        return asyncio.run(
            self.run_async(
                command,
                cwd=cwd,
                env=env,
                timeout=timeout,
                read_only=read_only,
                offline_tooling=offline_tooling,
                allow_during_shutdown=allow_during_shutdown,
                capture_output=capture_output,
                input=input,
                check=check,
                source=source,
                model_lane_claim=claim,
            )
        )

    async def run_async(
        self,
        argv: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30.0,  # noqa: ASYNC109 - forwarded to subprocess.run.
        read_only: bool = False,
        offline_tooling: bool = False,
        allow_during_shutdown: bool = False,
        capture_output: bool = True,
        input: str | None = None,
        check: bool = False,
        source: str = "unknown",
        model_lane_claim: Any | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = _coerce_argv(argv)
        inferred_claim = model_lane_claim or _inferred_model_lane_claim(
            command,
            source=source,
            timeout_s=float(timeout),
        )
        if inferred_claim is not None:
            process = await self.spawn_async(
                command,
                stdin=asyncio.subprocess.PIPE if input is not None else None,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None,
                cwd=cwd,
                env=env,
                read_only=read_only,
                offline_tooling=offline_tooling,
                allow_during_shutdown=allow_during_shutdown,
                source=source,
                model_lane_claim=inferred_claim,
            )
            input_bytes = input.encode() if input is not None else None
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(input_bytes),
                    timeout=float(timeout),
                )
            except TimeoutError as exc:
                process.kill()
                stdout_bytes, stderr_bytes = await process.communicate()
                raise subprocess.TimeoutExpired(
                    command,
                    float(timeout),
                    output=stdout_bytes,
                    stderr=stderr_bytes,
                ) from exc
            stdout_text = (
                stdout_bytes.decode("utf-8", errors="replace")
                if isinstance(stdout_bytes, bytes)
                else stdout_bytes
            )
            stderr_text = (
                stderr_bytes.decode("utf-8", errors="replace")
                if isinstance(stderr_bytes, bytes)
                else stderr_bytes
            )
            completed = subprocess.CompletedProcess(
                command,
                int(process.returncode or 0),
                stdout_text,
                stderr_text,
            )
            if check:
                completed.check_returncode()
            return completed

        def _run() -> subprocess.CompletedProcess[str]:
            return self.run(
                command,
                cwd=cwd,
                env=env,
                timeout=timeout,
                read_only=read_only,
                offline_tooling=offline_tooling,
                allow_during_shutdown=allow_during_shutdown,
                capture_output=capture_output,
                input=input,
                check=check,
                source=source,
            )

        if is_shutdown_requested() and allow_during_shutdown and read_only:
            result = await run_sync_shutdown_callable(
                _run,
                timeout_s=max(0.1, float(timeout)) + 1.0,
                name=f"read-only-subprocess:{source}",
            )
            if not isinstance(result, subprocess.CompletedProcess):
                raise RuntimeError("shutdown subprocess bridge returned invalid result")
            return result
        return await asyncio.to_thread(_run)

    def spawn(
        self,
        argv: Sequence[str],
        *,
        stdin: Any = None,
        stdout: IO[str] | None = None,
        stderr: IO[str] | None = None,
        stdout_path: str | os.PathLike[str] | None = None,
        stderr_path: str | os.PathLike[str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        text: bool = True,
        start_new_session: bool = True,
        preexec_fn: Callable[[], None] | None = None,
        read_only: bool = False,
        offline_tooling: bool = False,
        allow_during_shutdown: bool = False,
        source: str = "unknown",
    ) -> subprocess.Popen[Any]:
        command = _coerce_argv(argv)
        _model_command_requires_async(command, source=source)
        if read_only and not offline_tooling:
            _validate_read_only_source(source)
        offline_bypass = _validate_offline_tooling_bypass(
            offline_tooling=offline_tooling,
            source=source,
            command=command,
            env=env,
        )
        _require_not_shutting_down(
            f"subprocess_gateway.spawn:{source}",
            read_only=read_only,
            offline_tooling=offline_tooling,
            allow_during_shutdown=allow_during_shutdown,
        )
        if not read_only and not offline_bypass:
            _require_effect_governance(f"subprocess_gateway.spawn:{source}")
        _validate_desktop_safe_subprocess(command, env=env, source=source, operation="spawn")
        if stdout is not None and stdout_path is not None:
            raise ValueError("stdout and stdout_path are mutually exclusive")
        if stderr is not None and stderr_path is not None:
            raise ValueError("stderr and stderr_path are mutually exclusive")

        opened_streams: list[IO[Any]] = []
        try:
            if stdout_path is not None:
                stdout = _open_spawn_stream(stdout_path, text=text)
                opened_streams.append(stdout)
            if stderr_path is not None:
                stderr = _open_spawn_stream(stderr_path, text=text)
                opened_streams.append(stderr)

            proc = subprocess.Popen(
                command,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                cwd=_coerce_cwd(cwd),
                env=dict(env) if env is not None else None,
                shell=False,
                text=text,
                start_new_session=start_new_session,
                preexec_fn=preexec_fn,
            )
            try:
                _require_not_shutting_down(
                    f"subprocess_gateway.spawn:{source}",
                    read_only=read_only,
                    offline_tooling=offline_tooling,
                    allow_during_shutdown=allow_during_shutdown,
                    resource_created=True,
                )
            except GovernanceViolation:
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2.0)
                except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as exc:
                    record_shutdown_admission_event(
                        f"subprocess_gateway.spawn:{source}",
                        resource_kind="subprocess",
                        outcome="survived",
                        detail=repr(exc),
                    )
                    raise
                record_shutdown_admission_event(
                    f"subprocess_gateway.spawn:{source}",
                    resource_kind="subprocess",
                    outcome="reaped",
                    detail=f"pid={getattr(proc, 'pid', None)}",
                )
                raise
            proc._aura_gateway_streams = tuple(opened_streams)  # type: ignore[attr-defined]
            _register_runtime_hygiene_process(
                proc,
                kind="subprocess",
                source=source,
                command=command,
            )
            return proc
        except (
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            ValueError,
            GovernanceViolation,
        ):
            for stream in opened_streams:
                try:
                    stream.close()
                except OSError as close_exc:
                    logger.debug("failed to close gateway-owned subprocess stream: %s", close_exc)
            raise

    async def spawn_async(
        self,
        argv: Sequence[str],
        *,
        stdin: Any = None,
        stdout: Any = None,
        stderr: Any = None,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        start_new_session: bool = True,
        read_only: bool = False,
        offline_tooling: bool = False,
        allow_during_shutdown: bool = False,
        source: str = "unknown",
        model_lane_claim: Any | None = None,
    ) -> asyncio.subprocess.Process:
        command = _coerce_argv(argv)
        if read_only and not offline_tooling:
            _validate_read_only_source(source)
        offline_bypass = _validate_offline_tooling_bypass(
            offline_tooling=offline_tooling,
            source=source,
            command=command,
            env=env,
        )
        _require_not_shutting_down(
            f"subprocess_gateway.spawn_async:{source}",
            read_only=read_only,
            offline_tooling=offline_tooling,
            allow_during_shutdown=allow_during_shutdown,
        )
        if not read_only and not offline_bypass:
            _require_effect_governance(f"subprocess_gateway.spawn_async:{source}")
        _validate_desktop_safe_subprocess(command, env=env, source=source, operation="spawn_async")
        claim = model_lane_claim or _inferred_model_lane_claim(
            command,
            source=source,
            timeout_s=300.0,
        )
        if claim is not None and not start_new_session:
            raise RuntimeError("model_subprocess_requires_isolated_process_group")
        model_controller = None
        model_decision = None
        model_delegation_token = ""
        if claim is not None:
            model_controller, model_decision = await _reserve_model_lane_process(claim)
            try:
                model_delegation_token = await model_controller.issue_inherited_claim(
                    model_decision
                )
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
                await _cancel_model_lane_process(
                    model_controller,
                    model_decision,
                    reason="model_subprocess_delegation_failed",
                )
                raise
        process_env = dict(env) if env is not None else None
        if claim is not None:
            if process_env is None:
                process_env = dict(os.environ)
            process_env.update(
                {
                    "AURA_MODEL_LANE_INHERITED_OWNER_ID": str(claim.owner_id),
                    "AURA_MODEL_LANE_INHERITED_REQUEST_ID": str(claim.request_id),
                    "AURA_MODEL_LANE_INHERITED_MODEL_PATH": str(claim.model_path),
                    "AURA_MODEL_LANE_INHERITED_PURPOSE": str(claim.purpose),
                    "AURA_MODEL_LANE_DELEGATION_TOKEN": model_delegation_token,
                }
            )
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                cwd=_coerce_cwd(cwd),
                env=process_env,
                start_new_session=start_new_session,
            )
        except (OSError, RuntimeError, ValueError):
            if model_controller is not None and model_decision is not None:
                await _cancel_model_lane_process(
                    model_controller,
                    model_decision,
                    reason="model_subprocess_spawn_failed",
                )
            raise
        try:
            _require_not_shutting_down(
                f"subprocess_gateway.spawn_async:{source}",
                read_only=read_only,
                offline_tooling=offline_tooling,
                allow_during_shutdown=allow_during_shutdown,
                resource_created=True,
            )
        except GovernanceViolation:
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except TimeoutError:
                    proc.kill()
                    try:
                        # Bounded: a SIGKILLed child that cannot be reaped
                        # in 5s is an OS-level anomaly; leaking one zombie
                        # beats wedging the caller (A1 discipline).
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                    except TimeoutError:
                        logger.warning(
                            "SIGKILLed subprocess pid=%s not reaped in 5s",
                            getattr(proc, "pid", None),
                        )
            except (OSError, RuntimeError, ProcessLookupError, ValueError) as exc:
                record_shutdown_admission_event(
                    f"subprocess_gateway.spawn_async:{source}",
                    resource_kind="subprocess",
                    outcome="survived",
                    detail=repr(exc),
                )
                raise
            record_shutdown_admission_event(
                f"subprocess_gateway.spawn_async:{source}",
                resource_kind="subprocess",
                outcome="reaped",
                detail=f"pid={getattr(proc, 'pid', None)}",
            )
            if model_controller is not None and model_decision is not None:
                await _cancel_model_lane_process(
                    model_controller,
                    model_decision,
                    reason="shutdown_crossed_model_subprocess_spawn",
                )
            raise
        if model_controller is not None and model_decision is not None:
            from core.runtime.model_lane_control import (
                managed_process_group_alive,
                process_identity_for_pid,
            )

            try:
                process_group_id = int(os.getpgid(proc.pid))
            except (OSError, ProcessLookupError, ValueError):
                process_group_id = 0
            committed_process = process_identity_for_pid(proc.pid)
            try:
                committed = await model_controller.commit(
                    model_decision,
                    process=committed_process,
                    metadata={
                        "managed_model_process": True,
                        "process_group_id": process_group_id,
                        "start_new_session": bool(start_new_session),
                        "source": source,
                        "command": list(command),
                    },
                )
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except (OSError, RuntimeError, ProcessLookupError, TimeoutError, ValueError):
                    logger.error(
                        "Model subprocess survived failed lane commit pid=%s",
                        getattr(proc, "pid", None),
                    )
                await _cancel_model_lane_process(
                    model_controller,
                    model_decision,
                    reason=f"model_subprocess_commit_failed:{type(exc).__name__}",
                )
                raise RuntimeError("model_subprocess_lane_commit_failed") from exc

            async def _release_model_owner_when_done() -> None:
                try:
                    # Re-arming bounded slices: this monitor's LIFETIME is the
                    # worker's lifetime by design, but each individual await
                    # stays bounded (A1) so a wedged wait can never hide.
                    worker_exited = False
                    while not worker_exited:
                        try:
                            await asyncio.wait_for(proc.wait(), timeout=60.0)
                            worker_exited = True
                        except TimeoutError:
                            continue
                    # Descendants have no asyncio completion primitive.
                    while managed_process_group_alive(  # noqa: ASYNC110
                        process_group_id,
                        root_started_at=committed_process.started_at,
                    ):
                        await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    if proc.returncode is None or managed_process_group_alive(
                        process_group_id,
                        root_started_at=committed_process.started_at,
                    ):
                        logger.info(
                            "Model subprocess monitor cancelled while process tree remains "
                            "live; durable owner retained owner=%s pid=%s pgid=%s",
                            committed.owner_id,
                            proc.pid,
                            process_group_id,
                        )
                    raise
                finally:
                    if proc.returncode is not None and not managed_process_group_alive(
                        process_group_id,
                        root_started_at=committed_process.started_at,
                    ):
                        try:
                            await model_controller.release_owner(
                                committed.owner_id,
                                fencing_token=committed.fencing_token,
                                reason=f"model_subprocess_exit:{proc.returncode}",
                            )
                        except (
                            OSError,
                            RuntimeError,
                            AttributeError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            logger.warning(
                                "Model subprocess owner release failed owner=%s: %s",
                                committed.owner_id,
                                exc,
                            )

            from core.utils.task_tracker import get_task_tracker

            monitor_coroutine = _release_model_owner_when_done()
            try:
                monitor = get_task_tracker().create_task(
                    monitor_coroutine,
                    name=f"ModelProcessOwner:{committed.owner_id}",
                )
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                monitor_coroutine.close()
                try:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except TimeoutError:
                        proc.kill()
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                except (
                    OSError,
                    RuntimeError,
                    ProcessLookupError,
                    TimeoutError,
                    ValueError,
                ) as reap_exc:
                    logger.error(
                        "Model subprocess monitor failed and child reap was incomplete pid=%s: %s",
                        getattr(proc, "pid", None),
                        reap_exc,
                    )
                finally:
                    await model_controller.release_owner(
                        committed.owner_id,
                        fencing_token=committed.fencing_token,
                        reason="model_subprocess_monitor_registration_failed",
                    )
                raise RuntimeError("model_subprocess_monitor_registration_failed") from exc
            proc._aura_model_lane_owner_id = committed.owner_id  # type: ignore[attr-defined]
            proc._aura_model_lane_fencing_token = committed.fencing_token  # type: ignore[attr-defined]
            proc._aura_model_lane_receipt_id = committed.receipt_id  # type: ignore[attr-defined]
            proc._aura_model_lane_monitor = monitor  # type: ignore[attr-defined]
        _register_runtime_hygiene_process(
            proc,
            kind="subprocess",
            source=source,
            command=command,
        )
        return proc

    async def spawn_shell_async(
        self,
        command: str,
        *,
        stdin: Any = None,
        stdout: Any = None,
        stderr: Any = None,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        start_new_session: bool = True,
        offline_tooling: bool = False,
        allow_during_shutdown: bool = False,
        source: str = "unknown",
    ) -> asyncio.subprocess.Process:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("shell command must be a non-empty string")
        if "\x00" in command:
            raise ValueError("shell command must not contain NUL bytes")
        offline_bypass = _validate_offline_tooling_bypass(
            offline_tooling=offline_tooling,
            source=source,
            command=("/bin/sh", "-lc"),
        )
        _require_not_shutting_down(
            f"subprocess_gateway.spawn_shell_async:{source}",
            read_only=False,
            offline_tooling=offline_tooling,
            allow_during_shutdown=allow_during_shutdown,
        )
        if not offline_bypass:
            _require_effect_governance(f"subprocess_gateway.spawn_shell_async:{source}")
        _validate_desktop_safe_subprocess(command, env=env, source=source, operation="spawn_shell_async")
        if any(
            marker in command.lower()
            for marker in ("mlx_lm", "mlx-lm", "mlx_lm_lora", "heldout_eval.py")
        ):
            raise GovernanceViolation(
                "accelerator-owning shell commands are denied; use spawn_async argv "
                "so model identity and lane ownership remain parseable"
            )
        proc = await asyncio.create_subprocess_shell(
            command,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            cwd=_coerce_cwd(cwd),
            env=dict(env) if env is not None else None,
            start_new_session=start_new_session,
        )
        try:
            _require_not_shutting_down(
                f"subprocess_gateway.spawn_shell_async:{source}",
                read_only=False,
                offline_tooling=offline_tooling,
                allow_during_shutdown=allow_during_shutdown,
                resource_created=True,
            )
        except GovernanceViolation:
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except TimeoutError:
                    proc.kill()
                    try:
                        # Bounded: a SIGKILLed child that cannot be reaped
                        # in 5s is an OS-level anomaly; leaking one zombie
                        # beats wedging the caller (A1 discipline).
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                    except TimeoutError:
                        logger.warning(
                            "SIGKILLed subprocess pid=%s not reaped in 5s",
                            getattr(proc, "pid", None),
                        )
            except (OSError, RuntimeError, ProcessLookupError, ValueError) as exc:
                record_shutdown_admission_event(
                    f"subprocess_gateway.spawn_shell_async:{source}",
                    resource_kind="subprocess",
                    outcome="survived",
                    detail=repr(exc),
                )
                raise
            record_shutdown_admission_event(
                f"subprocess_gateway.spawn_shell_async:{source}",
                resource_kind="subprocess",
                outcome="reaped",
                detail=f"pid={getattr(proc, 'pid', None)}",
            )
            raise
        _register_runtime_hygiene_process(
            proc,
            kind="subprocess",
            source=source,
            command=command,
        )
        return proc


_gateway: SubprocessGateway | None = None


def get_subprocess_gateway() -> SubprocessGateway:
    global _gateway
    if _gateway is None:
        _gateway = SubprocessGateway()
    return _gateway


__all__ = ["SubprocessGateway", "get_subprocess_gateway"]
