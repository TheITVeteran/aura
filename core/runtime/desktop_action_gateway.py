"""core/runtime/desktop_action_gateway.py — Canonical Desktop Action Gateway.

All desktop/computer-use actions should flow through this module to ensure correct governance, logging, and audit.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any, Optional

from core.governance_context import (
    governance_runtime_active,
    require_governance,
)
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.DesktopActionGateway")
_DESKTOP_ACTION_RECOVERABLE_ERRORS = (
    FileNotFoundError,
    OSError,
    subprocess.SubprocessError,
    subprocess.TimeoutExpired,
    TypeError,
    ValueError,
)
_DESKTOP_ACTION_DOMAINS = (
    "environment_action",
    "external_action",
    "tool_execution",
)
_MAX_APPLESCRIPT_CHARS = 50_000


class DesktopActionGateway:
    """Single canonical owner for desktop control and AppleScript actions."""

    def __init__(self) -> None:
        self._allowed_domains = _DESKTOP_ACTION_DOMAINS

    def run_applescript(
        self,
        script: str,
        *,
        source: str = "unknown",
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Execute a desktop command via AppleScript."""
        script = _coerce_script(script)
        timeout_s = _coerce_timeout(timeout)
        if governance_runtime_active():
            require_governance(
                f"desktop_action_gateway.run_applescript:{source}",
                strict=True,
                allowed_domains=self._allowed_domains,
            )

        if shutil.which("osascript") is None:
            return {
                "ok": False,
                "stdout": "",
                "stderr": "osascript is unavailable on this host",
                "exit_code": 127,
            }

        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                shell=False,
            )
            return {
                "ok": proc.returncode == 0,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "exit_code": proc.returncode,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "stdout": "",
                "stderr": f"AppleScript timed out: {exc}",
                "exit_code": -1,
            }
        except _DESKTOP_ACTION_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "desktop_action_gateway",
                exc,
                action="returned failed desktop action receipt",
            )
            logger.warning("AppleScript execution failed: %s", exc)
            return {
                "ok": False,
                "stdout": "",
                "stderr": str(exc),
                "exit_code": -2,
            }


def _coerce_script(script: str) -> str:
    if not isinstance(script, str):
        raise TypeError("AppleScript payload must be a string")
    script = script.strip()
    if not script:
        raise ValueError("AppleScript payload must not be empty")
    if len(script) > _MAX_APPLESCRIPT_CHARS:
        raise ValueError(f"AppleScript payload exceeds {_MAX_APPLESCRIPT_CHARS} characters")
    return script


def _coerce_timeout(timeout: float) -> float:
    try:
        timeout_s = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("desktop action timeout must be numeric") from exc
    if timeout_s <= 0:
        raise ValueError("desktop action timeout must be positive")
    return min(timeout_s, 120.0)


_gateway: Optional[DesktopActionGateway] = None


def get_desktop_action_gateway() -> DesktopActionGateway:
    global _gateway
    if _gateway is None:
        _gateway = DesktopActionGateway()
    return _gateway


__all__ = ["DesktopActionGateway", "get_desktop_action_gateway"]
