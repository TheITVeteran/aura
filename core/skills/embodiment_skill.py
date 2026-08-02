"""Aura's governed introspection and control surface for physical reach."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from core.container import ServiceContainer
from core.embodiment.world_bridge import Channel, get_world_bridge
from core.reality_reach.attachments import AttachmentAccess
from core.runtime.errors import record_degradation
from core.skills.base_skill import BaseSkill

_HOME_ASSISTANT_CONTROL_DOMAINS = frozenset(
    {"climate", "fan", "input_boolean", "light", "switch"}
)


def _service(name: str) -> Any | None:
    try:
        return ServiceContainer.get(name, default=None)
    except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
        record_degradation(
            "embodiment_skill",
            exc,
            severity="warning",
            action=f"reported {name} unavailable without inventing physical capability",
        )
        return None


def _world_result(result: Any) -> dict[str, Any]:
    return {
        "ok": bool(getattr(result, "ok", False)),
        "receipt_id": str(getattr(result, "receipt_id", "") or ""),
        "status": str(getattr(result, "status", "") or ""),
        "data": getattr(result, "data", None),
        "error": str(getattr(result, "error", "") or ""),
        "transport_succeeded": getattr(result, "transport_succeeded", None),
        "effect_verified": getattr(result, "effect_verified", None),
        "manual_reconciliation_required": bool(
            getattr(result, "manual_reconciliation_required", False)
        ),
    }


class EmbodimentSkill(BaseSkill):  # type: ignore[misc]  # skipped import is untyped
    """Observe, discover, attach, focus, and control declared physical surfaces."""

    name = "embodiment"
    description = (
        "Use my live physical body and Reality Reach fabric: inspect connected "
        "sensors and actuators, discover nearby configured devices, propose a "
        "connection, focus attention on a sensor, read observations, or execute "
        "a governed and verified physical command."
    )
    effect_scope = "external_io"
    retry_safe = False
    timeout_seconds = 90.0
    inputs = {
        "action": (
            "inventory | discover | candidates | connection_requests | "
            "request_connection | focus_sensor | pause_sensors | resume_sensors | "
            "latest_observations | query_device | command_device"
        ),
        "device_id": "Hardware device id for query or command.",
        "candidate_id": "Discovered candidate id for request_connection.",
        "channel_id": "Reality Reach channel or prefix for query/focus.",
        "access": "observe, or observe+control when proposing a connection.",
        "command": "Declared hardware command or Home Assistant operation.",
        "parameters": "Bounded command/effect parameters.",
    }

    async def execute(
        self,
        goal: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        del context
        params = goal.get("params", goal)
        if not isinstance(params, Mapping):
            return {"ok": False, "error": "embodiment parameters must be a mapping"}
        action = str(params.get("action") or "inventory").strip().lower()
        if action in {"inventory", "list_devices"}:
            return self._inventory()
        if action == "discover":
            broker = _service("reality_attachment_broker")
            if broker is None:
                return {"ok": False, "error": "physical attachment broker is offline"}
            candidates = await broker.discover()
            return {
                "ok": True,
                "candidates": [item.to_dict() for item in candidates],
                "connection_requests": [item.to_dict() for item in broker.requests()],
                "summary": f"Discovered {len(candidates)} declared physical surfaces.",
            }
        if action == "candidates":
            broker = _service("reality_attachment_broker")
            if broker is None:
                return {"ok": False, "error": "physical attachment broker is offline"}
            candidates = broker.candidates()
            return {"ok": True, "candidates": [item.to_dict() for item in candidates]}
        if action == "connection_requests":
            broker = _service("reality_attachment_broker")
            if broker is None:
                return {"ok": False, "error": "physical attachment broker is offline"}
            requests = broker.requests()
            return {"ok": True, "connection_requests": [item.to_dict() for item in requests]}
        if action == "request_connection":
            return await self._request_connection(params)
        if action == "focus_sensor":
            router = _service("reality_observation_router")
            selector = str(params.get("channel_id") or params.get("selector") or "").strip()
            if router is None or not selector:
                return {"ok": False, "error": "a live router and channel selector are required"}
            subscription = router.focus(
                selector,
                duration_s=float(params.get("duration_s") or 30.0),
                max_rate_hz=float(params.get("max_rate_hz") or 4.0),
                min_salience=float(params.get("min_salience") or 0.0),
            )
            return {
                "ok": True,
                "subscription": subscription.to_dict(),
                "summary": f"Focused physical attention on {selector} for a bounded interval.",
            }
        if action in {"pause_sensors", "resume_sensors"}:
            router = _service("reality_observation_router")
            if router is None:
                return {"ok": False, "error": "physical observation router is offline"}
            (router.pause if action == "pause_sensors" else router.resume)()
            return {"ok": True, "observation_router": router.status()}
        if action == "latest_observations":
            router = _service("reality_observation_router")
            if router is None:
                return {"ok": False, "error": "physical observation router is offline"}
            prefix = str(params.get("channel_id") or "").strip().lower()
            latest = router.latest()
            if prefix:
                latest = {
                    key: value for key, value in latest.items() if key.startswith(prefix)
                }
            return {"ok": True, "observations": latest, "router": router.status()}
        if action == "query_device":
            return await self._query(params)
        if action == "command_device":
            return await self._command(params)
        return {"ok": False, "error": f"unknown embodiment action: {action}"}

    @staticmethod
    def _inventory() -> dict[str, Any]:
        manager = _service("hardware_manager")
        reality = _service("reality_reach")
        router = _service("reality_observation_router")
        broker = _service("reality_attachment_broker")
        body = _service("body_schema")
        devices = manager.list_devices() if manager is not None else []
        declarations = (
            [item.to_dict() for item in reality.declarations()]
            if reality is not None
            else []
        )
        body_map = body.get_body_map() if body is not None else {}
        physical_limbs = {
            name: limb
            for name, limb in body_map.items()
            if str(limb.get("source") or "").startswith("reality:")
        }
        return {
            "ok": True,
            "devices": devices,
            "channels": declarations,
            "physical_limbs": physical_limbs,
            "observation_router": router.status() if router is not None else None,
            "attachments": broker.status() if broker is not None else None,
            "summary": (
                f"My physical body currently exposes {len(declarations)} channels "
                f"across {len(physical_limbs)} sensor or actuator limbs."
            ),
        }

    @staticmethod
    async def _request_connection(params: Mapping[str, Any]) -> dict[str, Any]:
        broker = _service("reality_attachment_broker")
        candidate_id = str(params.get("candidate_id") or "").strip()
        if broker is None or not candidate_id:
            return {"ok": False, "error": "candidate_id and attachment broker are required"}
        raw_access = params.get("access", "observe")
        tokens = (
            [str(item).strip().lower() for item in raw_access]
            if isinstance(raw_access, (list, tuple))
            else str(raw_access).replace("+", ",").split(",")
        )
        access = tuple(dict.fromkeys(AttachmentAccess(item.strip()) for item in tokens if item.strip()))
        if AttachmentAccess.CONTROL in access and AttachmentAccess.OBSERVE not in access:
            access = (AttachmentAccess.OBSERVE, *access)
        request = await broker.request_connection(
            candidate_id,
            requested_access=access or (AttachmentAccess.OBSERVE,),
            initiated_by="aura",
            reason=str(params.get("reason") or "I chose to request this physical capability")[:320],
        )
        return {
            "ok": True,
            "connection_request": request.to_dict(),
            "summary": (
                "The device was attached through existing trust."
                if request.state.value == "attached"
                else "I proposed the connection; trust has not been invented or assumed."
            ),
        }

    @staticmethod
    async def _query(params: Mapping[str, Any]) -> dict[str, Any]:
        channel_id = str(params.get("channel_id") or "").strip().lower()
        if channel_id:
            reality = _service("reality_reach")
            if reality is None:
                return {"ok": False, "error": "Reality Reach is offline"}
            reading = reality.reading(channel_id)
            declaration = next(
                (item for item in reality.declarations() if item.channel_id == channel_id),
                None,
            )
            if reading is None or declaration is None:
                return {"ok": False, "error": f"physical channel not found: {channel_id}"}
            return {
                "ok": True,
                "declaration": declaration.to_dict(),
                "reading": reading.to_dict(),
            }
        device_id = str(params.get("device_id") or "").strip()
        manager = _service("hardware_manager")
        device = manager.get_device(device_id) if manager is not None else None
        if device is None:
            return {"ok": False, "error": f"hardware device not found: {device_id}"}
        status = await device.get_status()
        return {"ok": bool(status.get("ok", True)), "device_id": device_id, "status": status}

    @staticmethod
    async def _command(params: Mapping[str, Any]) -> dict[str, Any]:
        device_id = str(params.get("device_id") or params.get("target") or "").strip().lower()
        command = str(params.get("command") or params.get("op") or "").strip().lower()
        raw_parameters = params.get("parameters", params.get("effect", {}))
        if not device_id or not command or not isinstance(raw_parameters, Mapping):
            return {
                "ok": False,
                "error": "device_id, command, and a parameter mapping are required",
            }
        requested_transport = str(params.get("transport") or "").strip().lower()
        domain = device_id.partition(".")[0]
        use_home_assistant = (
            requested_transport == "home_assistant"
            or domain in _HOME_ASSISTANT_CONTROL_DOMAINS
        )
        operation = "home_assistant_apply" if use_home_assistant else "hardware_apply"
        payload: dict[str, Any] = {
            "operation": operation,
            "target": device_id,
            "op": command,
            "parameters": dict(raw_parameters),
            "reason": str(params.get("reason") or "Aura selected a physical action")[:240],
            "idempotency_key": str(
                params.get("idempotency_key")
                or f"embodiment.{uuid.uuid4().hex}"
            ),
        }
        if operation == "home_assistant_apply":
            payload["transport"] = "home_assistant"
            payload["effect"] = payload.pop("parameters")
        result = await get_world_bridge().call(
            Channel.ENVIRONMENTAL_CHANGE,
            action=f"physical:{device_id}:{command}",
            intent=payload["reason"],
            payload=payload,
        )
        return _world_result(result)


__all__ = ["EmbodimentSkill"]
