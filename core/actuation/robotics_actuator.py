"""core/actuation/robotics_actuator.py — Robotics Devices Actuator.

This boundary moves PHYSICAL things. Motion, force, thermal and electrical
effects are not undoable by a later software decision, so the checks here are
the last ones that can prevent them.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

#: A physical command must complete or be abandoned within a bounded window;
#: an un-deadlined motion command has no defined end state.
DEFAULT_COMMAND_DEADLINE_S = 15.0
MAX_COMMAND_DEADLINE_S = 120.0

#: Bounds on the command envelope crossing this boundary.
MAX_COMMAND_CHARS = 512
MAX_PARAM_KEYS = 32


class RoboticsActuationError(ValueError):
    """A physical-device command was refused at the boundary."""


def _registered_device(device_id: str) -> tuple[Any, Any]:
    """Resolve a device from the hardware registry, or refuse.

    CP126 15c5b221: a FREE-FORM identifier was forwarded with no binding to an
    owned device, driver, or interlock state — the governed label and the
    thing that actually moved were unrelated strings.
    """
    name = str(device_id or "").strip()
    if not name:
        raise RoboticsActuationError("device_id is empty")
    try:
        from core.container import ServiceContainer

        manager = ServiceContainer.get("hardware_manager", default=None)
    except (ImportError, AttributeError, RuntimeError) as exc:
        # No registry means no capability check. For a PHYSICAL actuator that
        # is a refusal, not a pass.
        raise RoboticsActuationError(
            f"hardware registry unavailable; refusing physical command: {exc}"
        ) from exc
    if manager is None:
        raise RoboticsActuationError(
            "hardware registry is not running; refusing physical command"
        )
    device = manager.get_device(name) if hasattr(manager, "get_device") else None
    if device is None:
        raise RoboticsActuationError(f"device is not registered: {name}")
    return manager, device


def _actuation_coordinator() -> Any:
    try:
        from core.container import ServiceContainer

        coordinator = ServiceContainer.get("reality_actuation", default=None)
    except (ImportError, AttributeError, RuntimeError) as exc:
        raise RoboticsActuationError(
            f"Reality Reach actuation coordinator unavailable: {exc}"
        ) from exc
    if coordinator is None or not callable(getattr(coordinator, "execute", None)):
        raise RoboticsActuationError(
            "Reality Reach actuation coordinator is not running"
        )
    return coordinator


def _device_state_snapshot(device: Any) -> dict[str, Any]:
    """Pre-command state, so the receipt says what the world looked like."""
    snapshot: dict[str, Any] = {}
    for attr in ("device_id", "device_type", "status", "connected", "interlock"):
        value = getattr(device, attr, None)
        if value is not None and not callable(value):
            snapshot[attr] = value
    to_dict = getattr(device, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
            if isinstance(payload, dict):
                snapshot.setdefault("status", payload.get("status"))
                snapshot["connected"] = payload.get("connected", snapshot.get("connected"))
        except (RuntimeError, AttributeError, TypeError, ValueError):
            snapshot["snapshot_error"] = "device_to_dict_failed"
    return snapshot


def _interlock_blocked(snapshot: dict[str, Any]) -> str:
    """Refuse when the device's own state says it must not move."""
    if snapshot.get("connected") is False:
        return "device_not_connected"
    status = str(snapshot.get("status") or "").strip().lower()
    if status in {"error", "fault", "estop", "emergency_stop", "locked", "disabled"}:
        return f"device_status:{status}"
    interlock = snapshot.get("interlock")
    if interlock is True or str(interlock or "").strip().lower() in {"engaged", "closed"}:
        return "interlock_engaged"
    return ""


class RoboticsActuator:
    """Wrapper for external physical/robotics device interactions."""

    @classmethod
    async def command_device(
        cls,
        device_id: str,
        command: str,
        params: dict[str, Any],
        source: str = "robotics_actuator",
        *,
        deadline_s: float = DEFAULT_COMMAND_DEADLINE_S,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Command a REGISTERED physical device under an explicit contract.

        CP126 67b7a5b1 / 94b4cea8 / 15c5b221 / 384bd18d — every command:
          * is dispatched as HIGH RISK (physical effects are irreversible);
          * carries the device and command the boundary classified, applied
            LAST so caller params cannot smuggle a different target;
          * names a device in the hardware registry whose interlock state
            permits motion;
          * carries a state snapshot, a bounded deadline, an idempotency key
            and the compensating stop action to run if it does not complete.
        """
        name = str(device_id or "").strip()
        text = str(command or "").strip()
        if not text:
            raise RoboticsActuationError("command is empty")
        if len(text) > MAX_COMMAND_CHARS:
            raise RoboticsActuationError(
                f"command exceeds {MAX_COMMAND_CHARS} characters"
            )
        caller_params = dict(params or {})
        caller_params.pop("device_id", None)
        caller_params.pop("command", None)
        if len(caller_params) > MAX_PARAM_KEYS:
            raise RoboticsActuationError(
                f"command carries too many parameters ({len(caller_params)})"
            )
        try:
            bounded_deadline = float(deadline_s)
        except (TypeError, ValueError):
            bounded_deadline = DEFAULT_COMMAND_DEADLINE_S
        if not (0.0 < bounded_deadline <= MAX_COMMAND_DEADLINE_S):
            bounded_deadline = min(MAX_COMMAND_DEADLINE_S, DEFAULT_COMMAND_DEADLINE_S)

        manager, device = _registered_device(name)
        snapshot = _device_state_snapshot(device)
        blocked = _interlock_blocked(snapshot)
        if blocked:
            raise RoboticsActuationError(
                f"device state refuses physical commands: {blocked}"
            )

        adapter = (
            manager.get_reality_adapter(name)
            if callable(getattr(manager, "get_reality_adapter", None))
            else None
        )
        if adapter is None and callable(getattr(manager, "activate_device", None)):
            adapter = await manager.activate_device(name)
        if adapter is None:
            raise RoboticsActuationError(
                "device has no registered explicit Reality Reach capability"
            )
        service = getattr(manager, "reality_service", None)
        if service is None or not callable(getattr(service, "status", None)):
            raise RoboticsActuationError("hardware inventory service is unavailable")
        await adapter.refresh_readback()
        await asyncio.to_thread(service.refresh)
        inventory_sha256 = str(service.status().get("registry_sha256") or "")
        try:
            typed_command = await adapter.compile_command(
                text,
                caller_params,
                inventory_sha256=inventory_sha256,
                deadline_s=bounded_deadline,
                idempotency_key=str(idempotency_key or uuid.uuid4().hex),
                source=source,
            )
            result = await _actuation_coordinator().execute(typed_command)
        except (LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RoboticsActuationError(str(exc)) from exc
        if not isinstance(result, dict):
            raise RoboticsActuationError("Reality Reach returned an invalid result")
        return result

    @classmethod
    async def emergency_stop(
        cls, device_id: str, source: str = "robotics_actuator", reason: str = "operator_stop"
    ) -> dict[str, Any]:
        """The compensating action referenced by every command's contract."""
        return await cls.command_device(
            device_id,
            "emergency_stop",
            {"reason": str(reason or "")[:200]},
            source=source,
            deadline_s=DEFAULT_COMMAND_DEADLINE_S,
        )
