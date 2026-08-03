"""Compatibility adapter for affect-driven environmental actions.

All physical effects are delegated to the canonical WorldBridge and IoT
transport stack. This module intentionally owns no network client, credential,
permission, or receipt path of its own.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from core.embodiment.world_bridge import Channel, WorldActionResult, get_world_bridge

logger = logging.getLogger("Aura.IoTBridge")
DEFAULT_LIGHT_ENTITY = "light.office_ambient"


def _result_payload(result: WorldActionResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "receipt_id": result.receipt_id,
        "data": result.data,
        "error": result.error,
        "status": result.status,
        "transport_succeeded": result.transport_succeeded,
        "effect_verified": result.effect_verified,
        "manual_reconciliation_required": result.manual_reconciliation_required,
    }


def _hass_flag(name: str, description: str) -> str:
    """Non-secret HASS knobs read through the typed flag layer (C1).
    Tokens deliberately stay as raw env reads — credentials must never
    surface in the declared-flags registry."""
    try:
        from core.runtime.flags import FlagKind, declare

        return str(
            declare(
                name,
                kind=FlagKind.STRING,
                default="",
                description=description,
                owner="core.autonomic.iot_bridge",
            ).value()
            or ""
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return ""


class PhysicalActuator:
    """Legacy affect API backed by the governed environmental-change channel."""

    def __init__(self, home_assistant_url: str | None = None) -> None:
        token_configured = bool(
            str(os.getenv("AURA_HASS_TOKEN") or os.getenv("HASS_TOKEN") or "").strip()
        )
        self._configured_url = str(
            home_assistant_url
            or _hass_flag("AURA_HASS_URL", "Home Assistant base URL")
            or os.getenv("HASS_URL")
            or ("https://homeassistant.local:8123" if token_configured else "")
        ).strip()
        self._configured = bool(self._configured_url and token_configured)
        if self._configured:
            logger.info("Affect IoT adapter attached to the governed WorldBridge path.")
        else:
            logger.info("Affect IoT adapter idle: Home Assistant transport is not configured.")

    async def discover_devices(self) -> list[dict[str, Any]]:
        result = await get_world_bridge().call(
            Channel.ENVIRONMENTAL_CHANGE,
            action="iot:discover",
            intent="discover explicitly permitted environmental devices",
            payload={"operation": "discover"},
        )
        if not result.ok or not isinstance(result.data, dict):
            return []
        devices = result.data.get("devices")
        return [item for item in list(devices or []) if isinstance(item, dict)]

    async def broadcast_affect_state(
        self,
        pad_vector: dict[str, float],
    ) -> dict[str, Any]:
        """Request a governed ambient-light update from a bounded PAD vector."""
        if not self._configured:
            return {
                "ok": False,
                "status": "unavailable",
                "error": "home_assistant_transport_not_configured",
                "transport_succeeded": False,
                "effect_verified": False,
                "manual_reconciliation_required": False,
                "receipt_id": None,
                "data": {},
            }
        pleasure = max(-1.0, min(1.0, float(pad_vector.get("P", 0.0))))
        arousal = max(-1.0, min(1.0, float(pad_vector.get("A", 0.0))))
        brightness = max(50, min(255, int(((arousal + 1.0) / 2.0) * 255)))
        color_temp = 500 if pleasure < 0 else 250
        target = (
            str(
                _hass_flag("AURA_HASS_LIGHT_ENTITY", "Default affect light entity id")
                or os.getenv("HASS_LIGHT_ENTITY")
                or DEFAULT_LIGHT_ENTITY
            )
            .strip()
            .lower()
        )

        result = await get_world_bridge().call(
            Channel.ENVIRONMENTAL_CHANGE,
            action=f"iot:{target}:turn_on",
            intent="reflect bounded affect state through explicitly permitted ambient lighting",
            payload={
                "operation": "apply",
                "target": target,
                "op": "turn_on",
                "effect": {
                    "brightness": brightness,
                    "color_temp": color_temp,
                },
                "reason": "affect_pad_broadcast",
            },
        )
        return _result_payload(result)

    async def push_microcontroller_logic(
        self,
        device_id: str,
        action: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        safe_device = str(device_id or "").strip().lower()
        safe_action = str(action or "").strip().lower()
        if not re.fullmatch(r"[a-z0-9_]+", safe_device):
            raise ValueError("invalid_iot_device_id")
        if not re.fullmatch(r"[a-z0-9_]+", safe_action):
            raise ValueError("invalid_iot_action")
        target = f"microcontroller.{safe_device}"
        result = await get_world_bridge().call(
            Channel.ENVIRONMENTAL_CHANGE,
            action=f"iot:{target}:{safe_action}",
            intent="execute an explicitly permitted microcontroller action",
            payload={
                "operation": "apply",
                "target": target,
                "op": safe_action,
                "effect": dict(parameters),
                "reason": "microcontroller_action",
            },
        )
        if result.ok:
            logger.info(
                "IoT action receipt accepted: target=%s action=%s receipt=%s",
                target,
                safe_action,
                result.receipt_id,
            )
        return _result_payload(result)


__all__ = ["DEFAULT_LIGHT_ENTITY", "PhysicalActuator"]
