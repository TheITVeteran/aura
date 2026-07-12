"""Native macOS computer-use bridge owned by the signed Aura.app identity.

macOS TCC evaluates the executable performing an action.  Aura's cognitive
runtime is Python, while the user grants Screen Recording and Accessibility to
``Aura.app``.  This module routes primitive desktop effects through the native
launcher executable so the visible app's grant and the acting process are the
same identity.  Planning, authority, receipts, and verification remain in the
canonical Python runtime.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from collections import namedtuple
from pathlib import Path
from typing import Any

from core import governance_context as _governance_context
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.subprocess_gateway import get_subprocess_gateway

_BRIDGE_FLAG = "--native-desktop-bridge"
_PROBE_READY_TTL_S = 30.0
_PROBE_DEGRADED_TTL_S = 2.0
_PROBE_LOCK = threading.Lock()
_PROBE_CACHE: tuple[float, dict[str, Any]] = (0.0, {})
_EFFECT_DOMAINS = (
    "environment_action",
    "external_action",
    "tool_execution",
    "state_mutation",
    "file_write",
    "self_modification",
)
_Size = namedtuple("Size", "width height")
_Point = namedtuple("Point", "x y")


def _code_signature_summary(executable: Path | None) -> dict[str, Any]:
    """Return signing facts that explain macOS TCC retention behavior.

    Screen Recording and Accessibility grants are tied to the app identity
    macOS sees.  Ad-hoc signed rebuilds can leave System Settings showing an
    old "Aura" row while the current executable is still denied.  The desktop
    access panel needs those facts so it can report identity drift instead of a
    vague blocked state.
    """
    if executable is None or not executable.exists():
        return {"available": False, "reason": "bridge_executable_missing"}
    if sys.platform != "darwin":
        return {"available": False, "reason": "not_macos"}
    try:
        completed = get_subprocess_gateway().run(
            ["codesign", "-dv", "--verbose=4", str(executable)],
            timeout=5.0,
            read_only=True,
            capture_output=True,
            source="native_desktop_bridge.codesign_identity",
        )
    except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    text = "\n".join(
        str(part or "") for part in (completed.stdout, completed.stderr)
    )
    summary: dict[str, Any] = {
        "available": completed.returncode == 0,
        "returncode": int(completed.returncode),
        "adhoc": "Signature=adhoc" in text,
        "signature": "",
        "team_identifier": "",
        "authorities": [],
        "identifier": "",
        "cdhash": "",
        "stable_tcc_identity": True,
    }
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "Signature":
            summary["signature"] = value
        elif key == "TeamIdentifier":
            summary["team_identifier"] = value
        elif key == "Identifier":
            summary["identifier"] = value
        elif key == "CDHash":
            summary["cdhash"] = value
        elif key == "Authority":
            summary["authorities"].append(value)
    team = str(summary.get("team_identifier") or "").strip().lower()
    signature = str(summary.get("signature") or "").strip().lower()
    authorities = [
        str(value or "").strip()
        for value in summary.get("authorities", [])
        if str(value or "").strip()
    ]
    summary["stable_tcc_identity"] = bool(
        completed.returncode == 0
        and not summary["adhoc"]
        and (
            authorities
            or (signature and signature != "adhoc")
            or (team and team not in {"not set", "none"})
        )
    )
    if not summary["stable_tcc_identity"]:
        if summary["adhoc"]:
            summary["tcc_repair_hint"] = (
                "This Aura.app build is ad-hoc signed. macOS may show a stale "
                "permission row after rebuilds; install a stable signing identity "
                "before granting Screen Recording and Accessibility again."
            )
        else:
            summary["tcc_repair_hint"] = (
                "Aura.app does not expose a stable signing authority. Rebuild and "
                "install it with a persistent local or Developer ID certificate."
            )
    cert_name = os.getenv("AURA_CODESIGN_IDENTITY", "").strip()
    if cert_name:
        summary["configured_codesign_identity"] = cert_name
    if re.search(r"\badhoc\b", text, flags=re.IGNORECASE):
        summary["adhoc"] = True
    return summary


def _candidate_executables() -> tuple[Path, ...]:
    configured = str(os.getenv("AURA_NATIVE_DESKTOP_BRIDGE", "") or "").strip()
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path("/Applications/Aura.app/Contents/MacOS/aura-launcher"),
        project_root / "dist" / "Aura.app" / "Contents" / "MacOS" / "aura-launcher",
    ]
    return tuple(
        candidate.resolve()
        for candidate in candidates
        if candidate is not None and candidate.is_file() and os.access(candidate, os.X_OK)
    )


def bridge_executable() -> Path | None:
    candidates = _candidate_executables()
    return candidates[0] if candidates else None


def _probe_cache_ttl(result: dict[str, Any]) -> float:
    """Cache good bridge evidence longer than denials or launch races.

    The signed Aura.app bridge is the durable macOS TCC identity.  During app
    launch, however, the bridge can temporarily miss, time out, or report a
    partial grant while System Settings has already been toggled.  A long-lived
    negative cache makes the desktop UI look blocked after the bridge is ready.
    """
    if not result or not result.get("ok"):
        return _PROBE_DEGRADED_TTL_S
    required = ("screen_recording", "accessibility", "automation")
    if all(bool(result.get(key)) for key in required):
        return _PROBE_READY_TTL_S
    return _PROBE_DEGRADED_TTL_S


def _bridge_ipc_dirs() -> tuple[Path, Path]:
    base = Path(os.getenv("AURA_NATIVE_BRIDGE_DIR", "~/.aura/native_bridge")).expanduser()
    return base / "requests", base / "responses"


def _resident_bridge_process_running(executable: Path | None = None) -> bool:
    """Return True only when the signed Aura launcher bridge is actually alive.

    The IPC directories are persistent by design.  Treating their presence as a
    live resident bridge makes direct-python launches wait on stale request
    folders until the proof/readiness probe times out.  If the resident process
    is not observable, the caller should use the one-shot signed bridge path.
    """
    executable = executable or bridge_executable()
    if executable is None:
        return False
    try:
        executable_resolved = executable.resolve()
    except OSError:
        executable_resolved = executable
    try:
        from core.runtime.resource_observation import get_resource_observer

        for process in get_resource_observer().processes():
            try:
                proc_exe = process.exe
                if proc_exe:
                    try:
                        if Path(proc_exe).resolve() == executable_resolved:
                            return True
                    except OSError:
                        if Path(proc_exe) == executable:
                            return True
                cmdline = process.cmdline
                if cmdline:
                    try:
                        if Path(str(cmdline[0])).resolve() == executable_resolved:
                            return True
                    except OSError:
                        if Path(str(cmdline[0])) == executable:
                            return True
            except (OSError, TypeError, ValueError):
                continue
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return False


def _require_effect_governance(command: str) -> None:
    should_fail_closed = _governance_context.governance_runtime_active()
    token = _governance_context.require_governance(
        f"native_desktop_bridge.{command}",
        strict=True,
        allowed_domains=_EFFECT_DOMAINS,
    )
    if should_fail_closed and (
        token is None or getattr(token, "domain", "") == "degraded"
    ):
        raise _governance_context.GovernanceViolation(
            f"native_desktop_bridge.{command} called outside governed context"
        )


def _invoke_resident_bridge(
    command: str,
    *,
    timeout: float,
    **payload: Any,
) -> dict[str, Any] | None:
    request_dir, response_dir = _bridge_ipc_dirs()
    if not request_dir.is_dir() or not response_dir.is_dir():
        return None
    if not _resident_bridge_process_running():
        return None

    request_id = uuid.uuid4().hex
    request_path = request_dir / f"{request_id}.json"
    response_path = response_dir / f"{request_id}.json"
    request = {"command": str(command or "probe"), **payload}
    try:
        atomic_write_text(
            request_path,
            json.dumps(request, separators=(",", ":")),
            encoding="utf-8",
        )
    except OSError:
        request_path.unlink(missing_ok=True)
        return None

    deadline = time.monotonic() + max(0.05, float(timeout))
    try:
        while time.monotonic() < deadline:
            if response_path.is_file():
                text = response_path.read_text(encoding="utf-8")
                result = json.loads(text or "{}")
                if isinstance(result, dict):
                    result.setdefault("returncode", int(result.get("returncode", 0) or 0))
                    return result
                return {"ok": False, "error": "resident_bridge_non_object_response"}
            time.sleep(0.02)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"ok": False, "error": f"resident_bridge_error:{type(exc).__name__}: {exc}"}
    finally:
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)
    return None


def invoke_native_desktop_bridge(
    command: str,
    *,
    read_only: bool = False,
    timeout: float = 12.0,
    prefer_one_shot: bool = False,
    **payload: Any,
) -> dict[str, Any]:
    if not read_only:
        _require_effect_governance(command)

    command_name = str(command or "probe").lower()
    resident_timeout = (
        max(0.25, float(timeout))
        if command_name.startswith("request_")
        else min(max(0.25, float(timeout)), 3.0)
    )
    resident = None if prefer_one_shot else _invoke_resident_bridge(command, timeout=resident_timeout, **payload)
    if resident is not None:
        resident.setdefault("bridge_transport", "resident_ipc")
        return resident

    return _invoke_one_shot_bridge(
        command,
        read_only=read_only,
        timeout=timeout,
        **payload,
    )


def _invoke_one_shot_bridge(
    command: str,
    *,
    read_only: bool,
    timeout: float,
    **payload: Any,
) -> dict[str, Any]:
    executable = bridge_executable()
    if executable is None:
        return {"ok": False, "error": "native_desktop_bridge_unavailable"}

    request = {"command": str(command or "probe"), **payload}
    completed = get_subprocess_gateway().run(
        [str(executable), _BRIDGE_FLAG, json.dumps(request, separators=(",", ":"))],
        timeout=max(1.0, float(timeout)),
        read_only=bool(read_only),
        capture_output=True,
        source=f"native_desktop_bridge.{command}",
    )
    text = str(completed.stdout or "").strip()
    try:
        result = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": "native_desktop_bridge_invalid_response",
            "returncode": completed.returncode,
            "stderr": str(completed.stderr or "")[:240],
        }
    if not isinstance(result, dict):
        return {"ok": False, "error": "native_desktop_bridge_non_object_response"}
    result.setdefault("returncode", completed.returncode)
    result.setdefault("bridge_transport", "one_shot_subprocess")
    return result


def probe_native_desktop_bridge(*, force: bool = False, prefer_one_shot: bool = False) -> dict[str, Any]:
    global _PROBE_CACHE

    now = time.monotonic()
    with _PROBE_LOCK:
        captured_at, cached = _PROBE_CACHE
        if not force and not prefer_one_shot and cached and (now - captured_at) < _probe_cache_ttl(cached):
            cached_result = dict(cached)
            cached_result["cache_hit"] = True
            cached_result["cache_age_s"] = round(max(0.0, now - captured_at), 3)
            return cached_result
        try:
            result = invoke_native_desktop_bridge(
                "probe",
                read_only=True,
                timeout=5.0,
                prefer_one_shot=prefer_one_shot,
            )
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            result = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        result["bridge_executable"] = str(bridge_executable() or "")
        result["code_signature"] = _code_signature_summary(bridge_executable())
        result["cache_hit"] = False
        result["captured_at_unix"] = time.time()
        result["cache_ttl_s"] = _probe_cache_ttl(result)
        _PROBE_CACHE = (now, dict(result))
        return result


class NativePyAutoGUI:
    """Small PyAutoGUI-compatible facade backed by Aura.app CoreGraphics."""

    FAILSAFE = True
    PAUSE = 0.1

    @staticmethod
    def easeInOutQuad(value: float) -> float:  # noqa: N802 - pyautogui API compatibility
        value = max(0.0, min(1.0, float(value)))
        return 2 * value * value if value < 0.5 else 1 - ((-2 * value + 2) ** 2) / 2

    def _invoke(self, command: str, **payload: Any) -> dict[str, Any]:
        result = invoke_native_desktop_bridge(command, **payload)
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or f"native {command} failed"))
        pause = max(0.0, float(self.PAUSE or 0.0))
        if pause:
            time.sleep(pause)
        return result

    def size(self) -> _Size:
        result = invoke_native_desktop_bridge("size", read_only=True)
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or "native size failed"))
        return _Size(int(result.get("width", 0)), int(result.get("height", 0)))

    def position(self) -> _Point:
        result = invoke_native_desktop_bridge("position", read_only=True)
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or "native position failed"))
        return _Point(int(result.get("x", 0)), int(result.get("y", 0)))

    def screenshot(self) -> Any:
        from PIL import Image

        fd, raw_path = tempfile.mkstemp(prefix="aura-screen-", suffix=".png")
        os.close(fd)
        target = Path(raw_path)
        try:
            self._invoke("screenshot", path=str(target))
            with Image.open(target) as image:
                return image.copy()
        finally:
            target.unlink(missing_ok=True)

    def moveTo(  # noqa: N802 - pyautogui API compatibility
        self,
        x: float,
        y: float,
        duration: float = 0.0,
        tween: Any = None,
    ) -> None:
        del duration, tween
        self._invoke("move", x=float(x), y=float(y))

    def click(
        self,
        x: float | None = None,
        y: float | None = None,
        clicks: int = 1,
        interval: float = 0.0,
        button: str = "left",
    ) -> None:
        payload: dict[str, Any] = {
            "clicks": max(1, int(clicks)),
            "interval": max(0.0, float(interval)),
            "button": str(button),
        }
        if x is not None:
            payload["x"] = float(x)
        if y is not None:
            payload["y"] = float(y)
        self._invoke("click", **payload)

    def write(self, text: str, interval: float = 0.0) -> None:
        self._invoke("write", text=str(text), interval=max(0.0, float(interval)))

    def press(self, key: str, presses: int = 1, interval: float = 0.0) -> None:
        self._invoke(
            "press",
            key=str(key),
            presses=max(1, int(presses)),
            interval=max(0.0, float(interval)),
        )

    def hotkey(self, *keys: str, interval: float = 0.0) -> None:
        del interval
        self._invoke("hotkey", keys=[str(key) for key in keys])

    def scroll(self, amount: int) -> None:
        self._invoke("scroll", amount=int(amount))


_NATIVE_PYAUTOGUI = NativePyAutoGUI()


def get_native_pyautogui() -> NativePyAutoGUI | None:
    probe = probe_native_desktop_bridge()
    if probe.get("ok") and probe.get("accessibility"):
        return _NATIVE_PYAUTOGUI
    return None


__all__ = [
    "NativePyAutoGUI",
    "bridge_executable",
    "get_native_pyautogui",
    "invoke_native_desktop_bridge",
    "probe_native_desktop_bridge",
]
