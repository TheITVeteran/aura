"""Aura's governed introspection and control surface for physical reach."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from typing import Any

from core.container import ServiceContainer
from core.embodiment.world_bridge import Channel, get_world_bridge
from core.governance.capability_chain import CapabilityViolation, get_capability_issuer
from core.governance.will import ActionDomain, get_will
from core.reality_reach.attachment_authority import ATTACHMENT_AUTHORITY_ACTION
from core.reality_reach.attachments import AttachmentAccess
from core.runtime.audit_chain import canonical_json
from core.runtime.errors import record_degradation
from core.skills.base_skill import BaseSkill

_HOME_ASSISTANT_CONTROL_DOMAINS = frozenset(
    {"climate", "fan", "input_boolean", "light", "switch"}
)


def _boolean_parameter(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("boolean parameter must be true or false")


def _bounded_int_parameter(
    value: Any,
    *,
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must lie inside [{minimum}, {maximum}]")
    return parsed


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
        "a governed and verified physical command. I can also inspect durable "
        "sensor history, active alarms, and quarantined physical evidence."
    )
    effect_scope = "external_io"
    retry_safe = False
    timeout_seconds = 90.0
    inputs = {
        "action": (
            "inventory | discover | candidates | connection_requests | "
            "request_connection | authorize_connection | rotate_trust_custody | "
            "focus_sensor | pause_sensors | resume_sensors | "
            "pause_sensor_attention | resume_sensor_attention | latest_observations | "
            "observation_history | active_alarms | acknowledge_alarm | "
            "observation_quarantine | query_device | command_device"
        ),
        "device_id": "Hardware device id for query or command.",
        "candidate_id": "Discovered candidate id for request_connection.",
        "request_id": "Pending request id for authorize_connection.",
        "channel_id": "Reality Reach channel or prefix for query/focus.",
        "access": "observe, or observe+control when proposing a connection.",
        "persistent": "Whether bounded trust may survive a runtime migration.",
        "grant_ttl_s": "Requested trust lifetime within the enforced policy ceiling.",
        "command": "Declared hardware command or Home Assistant operation.",
        "parameters": "Bounded command/effect parameters.",
    }

    async def execute(
        self,
        goal: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
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
        if action == "authorize_connection":
            return await self._authorize_connection(params, context)
        if action == "rotate_trust_custody":
            broker = _service("reality_attachment_broker")
            if broker is None:
                return {"ok": False, "error": "physical attachment broker is offline"}
            receipt = await broker.rotate_trust_custody()
            return {
                "ok": True,
                "rotation_receipt": dict(receipt),
                "attachments": broker.status(),
            }
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
        if action in {"pause_sensor_attention", "resume_sensor_attention"}:
            router = _service("reality_observation_router")
            if router is None:
                return {"ok": False, "error": "physical observation router is offline"}
            (
                router.pause_attention
                if action == "pause_sensor_attention"
                else router.resume_attention
            )()
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
        if action == "observation_history":
            historian = _service("reality_historian")
            if historian is None:
                return {"ok": False, "error": "physical historian is offline"}
            try:
                limit = _bounded_int_parameter(
                    params.get("limit"),
                    name="limit",
                    default=100,
                    minimum=1,
                    maximum=500,
                )
                before_row_id = params.get("before_row_id")
                if before_row_id is not None and before_row_id != "":
                    before_row_id = _bounded_int_parameter(
                        before_row_id,
                        name="before_row_id",
                        default=1,
                        minimum=1,
                        maximum=2**63 - 1,
                    )
                else:
                    before_row_id = None
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            history = await historian.replay_history(
                channel_id=str(params.get("channel_id") or "").strip().lower() or None,
                before_row_id=before_row_id,
                limit=limit,
            )
            historian_status = await asyncio.to_thread(historian.status)
            return {"ok": True, "history": history, "historian": historian_status}
        if action == "active_alarms":
            historian = _service("reality_historian")
            if historian is None:
                return {"ok": False, "error": "physical historian is offline"}
            try:
                limit = _bounded_int_parameter(
                    params.get("limit"),
                    name="limit",
                    default=100,
                    minimum=1,
                    maximum=500,
                )
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            alarms = await historian.active_alarms(limit=limit)
            historian_status = await asyncio.to_thread(historian.status)
            return {
                "ok": True,
                "active_alarms": list(alarms),
                "historian": historian_status,
            }
        if action == "acknowledge_alarm":
            historian = _service("reality_historian")
            channel_id = str(params.get("channel_id") or "").strip().lower()
            if historian is None or not channel_id:
                return {
                    "ok": False,
                    "error": "physical historian and channel_id are required",
                }
            try:
                receipt = await historian.acknowledge_alarm(
                    channel_id,
                    actor="aura",
                )
            except LookupError:
                return {
                    "ok": False,
                    "error": "no active physical alarm exists for that channel",
                }
            return {
                "ok": True,
                "acknowledgement": receipt,
                "summary": "I acknowledged the alarm without clearing its physical state.",
            }
        if action == "observation_quarantine":
            historian = _service("reality_historian")
            if historian is None:
                return {"ok": False, "error": "physical historian is offline"}
            try:
                limit = _bounded_int_parameter(
                    params.get("limit"),
                    name="limit",
                    default=100,
                    minimum=1,
                    maximum=500,
                )
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            quarantined = await historian.quarantine(limit=limit)
            historian_status = await asyncio.to_thread(historian.status)
            return {
                "ok": True,
                "quarantine": list(quarantined),
                "historian": historian_status,
            }
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
        historian = _service("reality_historian")
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
            "historian": (
                historian.health_snapshot()
                if historian is not None
                and callable(getattr(historian, "health_snapshot", None))
                else historian.status()
                if historian is not None
                else None
            ),
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
    async def _authorize_connection(
        params: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        broker = _service("reality_attachment_broker")
        request_id = str(params.get("request_id") or "").strip()
        if broker is None or not request_id:
            return {"ok": False, "error": "request_id and attachment broker are required"}
        try:
            persistent = _boolean_parameter(params.get("persistent"), default=True)
            raw_ttl = params.get("grant_ttl_s")
            if raw_ttl is None:
                grant_ttl_s = None
            elif isinstance(raw_ttl, bool):
                raise ValueError("grant_ttl_s must be an integer")
            elif isinstance(raw_ttl, int):
                grant_ttl_s = raw_ttl
            elif isinstance(raw_ttl, str) and raw_ttl.strip().isdigit():
                grant_ttl_s = int(raw_ttl.strip())
            else:
                raise ValueError("grant_ttl_s must be an integer")
            intent = broker.authority_intent(
                request_id,
                persistent=persistent,
                grant_ttl_s=grant_ttl_s,
            )
            decision_context = dict(context)
            decision_context.update(
                {
                    "physical_attachment_request_id": request_id,
                    "physical_attachment_scope": intent["scope"],
                    "persistent_physical_trust": persistent,
                    "verification_required": True,
                }
            )
            decision = get_will().decide(
                content=(
                    "Authorize this exact bounded physical attachment relationship: "
                    + canonical_json(intent).decode("utf-8")
                ),
                source="embodiment_skill",
                domain=ActionDomain.ENVIRONMENT_ACTION,
                priority=0.7 if "control" in intent["requested_access"] else 0.5,
                context=decision_context,
            )
            capability = get_capability_issuer().issue_from_decision(
                decision,
                action=ATTACHMENT_AUTHORITY_ACTION,
                payload=intent,
                scope=str(intent["scope"]),
            )
            attached = await broker.authorize_and_attach(
                request_id,
                authority_capability=capability.to_dict(),
                persistent=persistent,
                grant_ttl_s=grant_ttl_s,
            )
            return {
                "ok": attached.state.value == "attached",
                "connection_request": attached.to_dict(),
                "authority_receipt_id": attached.authority_receipt_id,
                "grant_ttl_s": int(intent["grant_ttl_s"]),
                "persistent": persistent,
                "summary": (
                    "The declared physical relationship is attached under bounded trust."
                    if attached.state.value == "attached"
                    else "Authority was valid, but the physical attachment did not complete."
                ),
            }
        except (
            CapabilityViolation,
            LookupError,
            PermissionError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}:{exc}"[:320],
                "request_id": request_id,
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
