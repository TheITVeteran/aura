"""Governed REST smart-plug driver with explicit Reality Reach readback."""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
from collections.abc import Mapping
from typing import Any

from core.embodiment.base_device import BaseHardwareDevice
from core.embodiment.reality_adapter import (
    HardwareCommandContract,
    HardwareRealityManifest,
)
from core.reality_reach import (
    ActuatorCapability,
    ChannelDeclaration,
    ChannelKind,
    CouplingClass,
    EvidenceLevel,
    NumericDomain,
    RealityLayer,
    Reversibility,
)
from core.runtime.action_executor import ActionExecutor
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.network_gateway import get_network_gateway

logger = logging.getLogger("Embodiment.RestSmartPlug")


def _canonical_device_id(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_.:-]+", "-", str(value).strip().lower()).strip("-.")
    if not normalized or len(normalized) > 80:
        raise ValueError("smart-plug device_id is not a bounded canonical identifier")
    return normalized


class RestSmartPlug(BaseHardwareDevice):
    """REST relay whose command and status routes are separately observed.

    The endpoint must implement ``{"action": "status"}`` and return a JSON
    mapping with ``state`` equal to ``on`` or ``off``. Transport acceptance is
    never treated as state confirmation; Reality Reach calls ``get_status``
    again after every command.
    """

    def __init__(
        self,
        device_id: str = "generic_relay_01",
        name: str = "REST API Relay",
    ) -> None:
        canonical_id = _canonical_device_id(device_id)
        super().__init__(canonical_id, name, "iot.relay")
        self.power_state = False
        self.current_draw_watts = 0.0
        self.endpoint_url = str(os.environ.get("AURA_IOT_ENDPOINT") or "").strip()
        self.api_key = str(os.environ.get("AURA_IOT_KEY") or "").strip()

    def _validated_endpoint(self) -> str:
        parsed = urllib.parse.urlparse(self.endpoint_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise RuntimeError("AURA_IOT_ENDPOINT must be an absolute credential-free HTTP(S) URL")
        return self.endpoint_url

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _status_request(self) -> dict[str, Any]:
        response = await get_network_gateway().request_async(
            "POST",
            self._validated_endpoint(),
            headers=self._headers(),
            data=json.dumps(
                {"action": "status", "device_id": self.device_id},
                sort_keys=True,
                separators=(",", ":"),
            ),
            timeout=5.0,
            source="hardware:rest_smart_plug.status",
            read_only=True,
        )
        if not response.get("ok") or not 200 <= int(response.get("status_code") or 0) < 300:
            return {
                "ok": False,
                "error": str(response.get("error") or "smart_plug_status_transport_failed")[:300],
            }
        content = response.get("content") or b"{}"
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="strict")
        decoded = json.loads(str(content))
        if not isinstance(decoded, dict):
            raise ValueError("smart_plug_status_response_not_mapping")
        state = str(decoded.get("state") or "").strip().lower()
        if state not in {"on", "off"}:
            raise ValueError("smart_plug_status_response_has_invalid_state")
        try:
            watts = float(decoded.get("power_draw_watts") or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("smart_plug_status_response_has_invalid_power") from exc
        self.power_state = state == "on"
        self.current_draw_watts = max(0.0, watts)
        return {
            "ok": True,
            "status": state,
            "power_state_numeric": 1.0 if self.power_state else 0.0,
            "power_draw_watts": self.current_draw_watts,
            "connected": self.is_connected,
        }

    async def connect(self) -> bool:
        try:
            status = await self._status_request()
        except (OSError, RuntimeError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("REST relay status handshake failed: %s", exc)
            self.is_connected = False
            return False
        self.is_connected = status.get("ok") is True
        return self.is_connected

    async def disconnect(self) -> bool:
        self.is_connected = False
        return True

    async def get_status(self) -> dict[str, Any]:
        if not self.is_connected:
            return {"ok": False, "error": "smart_plug_not_connected"}
        try:
            return await self._status_request()
        except (OSError, RuntimeError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            return {"ok": False, "error": f"{type(exc).__name__}:{exc}"[:300]}

    async def check_interlocks(
        self,
        command: str,
        parameters: Mapping[str, Any],
        status: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        allowed = (
            self.is_connected
            and status.get("ok") is True
            and command in {"turn_on", "turn_off", "emergency_stop"}
            and set(parameters).issubset({"reason"})
        )
        body = {
            "device_id": self.device_id,
            "connected": self.is_connected,
            "status_ok": status.get("ok") is True,
            "command": command,
            "parameter_keys": sorted(parameters),
        }
        return {
            "ok": allowed,
            "reason": "smart_plug_interlock_refused" if not allowed else "",
            "interlock_sha256": str(sha256_hex(canonical_json(body))),
        }

    async def execute_command(self, command: str, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        target_state = {
            "turn_on": "on",
            "turn_off": "off",
            "emergency_stop": "off",
        }.get(command)
        if target_state is None:
            return {"ok": False, "error": "smart_plug_command_not_declared"}
        response = await ActionExecutor.request_network_transport(
            method="POST",
            url=self._validated_endpoint(),
            headers=self._headers(),
            data=json.dumps(
                {"action": target_state, "device_id": self.device_id},
                sort_keys=True,
                separators=(",", ":"),
            ),
            timeout_s=5.0,
            source="world_bridge:hardware.rest_smart_plug.apply",
            read_only=False,
        )
        status_code = int(response.get("status_code") or 0)
        accepted = bool(response.get("ok") and 200 <= status_code < 300)
        return {
            "ok": accepted,
            "transport_completed": status_code > 0,
            "status_code": status_code,
            "error": "" if accepted else str(response.get("error") or "transport_refused")[:300],
        }

    def reality_manifest(self) -> HardwareRealityManifest:
        prefix = f"hardware.{self.device_id}"
        actuator = ChannelDeclaration(
            channel_id=f"{prefix}.relay.command",
            kind=ChannelKind.ACTUATOR,
            observable="relay_power_state",
            unit="binary",
            domain=NumericDomain(0.0, 1.0),
            coupling=CouplingClass.ELECTRICAL,
            reality_layers=(RealityLayer.EFFECTIVE,),
            evidence_level=EvidenceLevel.P1,
            owner="core.embodiment.mock_iot_plug",
            stale_after_s=10.0,
            coupling_validated=True,
        )
        observation = ChannelDeclaration(
            channel_id=f"{prefix}.relay.readback",
            kind=ChannelKind.SENSOR,
            observable="relay_power_state",
            unit="binary",
            domain=NumericDomain(0.0, 1.0),
            coupling=CouplingClass.NETWORK,
            reality_layers=(RealityLayer.EFFECTIVE,),
            evidence_level=EvidenceLevel.P1,
            owner="core.embodiment.mock_iot_plug",
            resolution=1.0,
            sample_rate_hz=0.2,
            max_latency_s=5.0,
            stale_after_s=10.0,
            reference_id=f"{prefix}.status_api",
            coupling_validated=True,
        )
        capability = ActuatorCapability(
            adapter_id=f"{prefix}.adapter",
            channel_id=actuator.channel_id,
            reversibility=Reversibility.REVERSIBLE,
            magnitude_domain=NumericDomain(0.0, 1.0),
            max_commands_per_minute=12,
            observation_channels=(observation.channel_id,),
            required_permissions=("hardware.iot", "network.local"),
            failure_modes=("transport_failure", "readback_mismatch"),
            watchdog_timeout_s=8.0,
            compensation_action="emergency_stop",
        )
        turn_on = HardwareCommandContract(
            command="turn_on",
            target=1.0,
            magnitude=1.0,
            tolerance=0.0,
            safe_envelope=NumericDomain(0.0, 1.0),
            expected_effects=("relay_on_observed",),
            rollback_command="turn_off",
        )
        turn_off = HardwareCommandContract(
            command="turn_off",
            target=0.0,
            magnitude=1.0,
            tolerance=0.0,
            safe_envelope=NumericDomain(0.0, 1.0),
            expected_effects=("relay_off_observed",),
            rollback_command="turn_on",
        )
        emergency_stop = HardwareCommandContract(
            command="emergency_stop",
            target=0.0,
            magnitude=0.0,
            tolerance=0.0,
            safe_envelope=NumericDomain(0.0, 0.0),
            allowed_parameters=("reason",),
            expected_effects=("relay_safe_state_observed",),
            rollback_command="emergency_stop",
        )
        return HardwareRealityManifest(
            adapter_id=capability.adapter_id,
            actuator=actuator,
            observation=observation,
            capability=capability,
            observation_field="power_state_numeric",
            command_transport_id=f"{prefix}.command_transport",
            readback_transport_id=f"{prefix}.status_transport",
            commands=(turn_on, turn_off, emergency_stop),
            safe_state_command="emergency_stop",
            safe_state_target=0.0,
        )


__all__ = ["RestSmartPlug"]
