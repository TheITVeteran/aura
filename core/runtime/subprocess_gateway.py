"""Canonical subprocess gateway.

All live runtime subprocess creation should flow through this module. Effectful
calls require an active governance context; explicitly read-only probes may opt
out while still receiving consistent validation and logging behavior.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import IO, Any, Mapping, Sequence

from core.governance_context import require_governance


_EFFECT_DOMAINS = (
    "tool_execution",
    "state_mutation",
    "file_write",
    "self_modification",
)


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
        source: str = "unknown",
    ) -> subprocess.CompletedProcess[str]:
        command = _coerce_argv(argv)
        if not read_only:
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
            capture_output=True,
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
        source: str = "unknown",
    ) -> subprocess.Popen[Any]:
        command = _coerce_argv(argv)
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


_gateway: SubprocessGateway | None = None


def get_subprocess_gateway() -> SubprocessGateway:
    global _gateway
    if _gateway is None:
        _gateway = SubprocessGateway()
    return _gateway


__all__ = ["SubprocessGateway", "get_subprocess_gateway"]
