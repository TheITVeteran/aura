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


def governance_runtime_active() -> bool:
    return _governance_context.governance_runtime_active()


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
    return _truthy_env_value(_effective_env_value(env, "AURA_SAFE_BOOT_DESKTOP")) or _truthy_env_value(
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
        capture_output: bool = True,
        input: str | None = None,
        check: bool = False,
        source: str = "unknown",
    ) -> subprocess.CompletedProcess[str]:
        command = _coerce_argv(argv)
        if read_only and not offline_tooling:
            _validate_read_only_source(source)
        offline_bypass = _validate_offline_tooling_bypass(
            offline_tooling=offline_tooling,
            source=source,
            command=command,
            env=env,
        )
        if not read_only and not offline_bypass:
            _require_effect_governance(f"subprocess_gateway.run:{source}")
        _validate_desktop_safe_subprocess(command, env=env, source=source, operation="run")
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

    async def run_async(
        self,
        argv: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30.0,  # noqa: ASYNC109 - forwarded to subprocess.run.
        read_only: bool = False,
        offline_tooling: bool = False,
        capture_output: bool = True,
        input: str | None = None,
        check: bool = False,
        source: str = "unknown",
    ) -> subprocess.CompletedProcess[str]:
        return await asyncio.to_thread(
            self.run,
            argv,
            cwd=cwd,
            env=env,
            timeout=timeout,
            read_only=read_only,
            offline_tooling=offline_tooling,
            capture_output=capture_output,
            input=input,
            check=check,
            source=source,
        )

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
        source: str = "unknown",
    ) -> subprocess.Popen[Any]:
        command = _coerce_argv(argv)
        if read_only and not offline_tooling:
            _validate_read_only_source(source)
        offline_bypass = _validate_offline_tooling_bypass(
            offline_tooling=offline_tooling,
            source=source,
            command=command,
            env=env,
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
            proc._aura_gateway_streams = tuple(opened_streams)  # type: ignore[attr-defined]
            return proc
        except (OSError, subprocess.SubprocessError, ValueError):
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
        source: str = "unknown",
    ) -> asyncio.subprocess.Process:
        command = _coerce_argv(argv)
        if read_only and not offline_tooling:
            _validate_read_only_source(source)
        offline_bypass = _validate_offline_tooling_bypass(
            offline_tooling=offline_tooling,
            source=source,
            command=command,
        )
        if not read_only and not offline_bypass:
            _require_effect_governance(f"subprocess_gateway.spawn_async:{source}")
        _validate_desktop_safe_subprocess(command, env=env, source=source, operation="spawn_async")
        return await asyncio.create_subprocess_exec(
            *command,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            cwd=_coerce_cwd(cwd),
            env=dict(env) if env is not None else None,
            start_new_session=start_new_session,
        )

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
        if not offline_bypass:
            _require_effect_governance(f"subprocess_gateway.spawn_shell_async:{source}")
        _validate_desktop_safe_subprocess(command, env=env, source=source, operation="spawn_shell_async")
        return await asyncio.create_subprocess_shell(
            command,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            cwd=_coerce_cwd(cwd),
            env=dict(env) if env is not None else None,
            start_new_session=start_new_session,
        )


_gateway: SubprocessGateway | None = None


def get_subprocess_gateway() -> SubprocessGateway:
    global _gateway
    if _gateway is None:
        _gateway = SubprocessGateway()
    return _gateway


__all__ = ["SubprocessGateway", "get_subprocess_gateway"]
