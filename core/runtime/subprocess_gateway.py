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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any

from core.governance_context import (
    GovernanceViolation,
    governance_runtime_active,
    require_governance,
)

_EFFECT_DOMAINS = (
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
logger = logging.getLogger("Aura.SubprocessGateway")


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


def _validate_offline_tooling_bypass(
    *,
    offline_tooling: bool,
    source: str,
    command: Sequence[str],
) -> bool:
    """Allow named repo tooling to launch child processes outside live Aura.

    This is intentionally not a general governance bypass. It exists for CLI
    proof, certification, benchmark, maintenance, and training wrappers that
    orchestrate Aura from outside her live runtime. If live/strict governance is
    active, the bypass fails closed and callers must enter a governed scope.
    """
    if not offline_tooling:
        return False
    if governance_runtime_active():
        raise GovernanceViolation(
            f"offline subprocess tooling bypass denied while live governance is active: {source}"
        )
    if not any(source.startswith(prefix) for prefix in _OFFLINE_TOOLING_SOURCE_PREFIXES):
        raise ValueError(
            "offline subprocess tooling requires a source prefix of "
            f"{', '.join(_OFFLINE_TOOLING_SOURCE_PREFIXES)}"
        )
    logger.info(
        "offline subprocess tooling bypass source=%s argv0=%s argc=%s",
        source,
        command[0] if command else "",
        len(command),
    )
    return True


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
        source: str = "unknown",
    ) -> subprocess.CompletedProcess[str]:
        command = _coerce_argv(argv)
        offline_bypass = _validate_offline_tooling_bypass(
            offline_tooling=offline_tooling,
            source=source,
            command=command,
        )
        if not read_only and not offline_bypass:
            require_governance(
                f"subprocess_gateway.run:{source}",
                strict=True,
                allowed_domains=_EFFECT_DOMAINS,
            )
        return subprocess.run(
            command,
            cwd=_coerce_cwd(cwd),
            env=dict(env) if env is not None else None,
            timeout=float(timeout),
            capture_output=bool(capture_output),
            text=True,
            check=False,
            shell=False,
        )

    def spawn(
        self,
        argv: Sequence[str],
        *,
        stdout: IO[str] | None = None,
        stderr: IO[str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        text: bool = True,
        start_new_session: bool = True,
        offline_tooling: bool = False,
        source: str = "unknown",
    ) -> subprocess.Popen[Any]:
        command = _coerce_argv(argv)
        offline_bypass = _validate_offline_tooling_bypass(
            offline_tooling=offline_tooling,
            source=source,
            command=command,
        )
        if not offline_bypass:
            require_governance(
                f"subprocess_gateway.spawn:{source}",
                strict=True,
                allowed_domains=_EFFECT_DOMAINS,
            )
        return subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            cwd=_coerce_cwd(cwd),
            env=dict(env) if env is not None else None,
            shell=False,
            text=text,
            start_new_session=start_new_session,
        )

    async def spawn_async(
        self,
        argv: Sequence[str],
        *,
        stdout: Any = None,
        stderr: Any = None,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        start_new_session: bool = True,
        offline_tooling: bool = False,
        source: str = "unknown",
    ) -> asyncio.subprocess.Process:
        command = _coerce_argv(argv)
        offline_bypass = _validate_offline_tooling_bypass(
            offline_tooling=offline_tooling,
            source=source,
            command=command,
        )
        if not offline_bypass:
            require_governance(
                f"subprocess_gateway.spawn_async:{source}",
                strict=True,
                allowed_domains=_EFFECT_DOMAINS,
            )
        return await asyncio.create_subprocess_exec(
            *command,
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
