"""core/embodiment/iot_bridge.py

Physical IoT Bridge (Causal Grounding)
========================================
A two-way coupling between Aura's homeostatic state and the physical
environment. Examples:

  * Aura's CPU temperature climbs past a threshold → request a thermostat
    setpoint drop on the local network.
  * Sustained prediction-error storm → tint the room red and lower
    activity-level lights (mood signal).
  * High curiosity + idle → pull research feed brightness up so the user
    sees a quiet "I'm reading" state.
  * The environment changes (door opens, light turns on) → that change
    enters Aura's prediction-error stream as a real exteroceptive signal.

The bridge is transport-agnostic; concrete transports register through
``register_transport()``. Stock support is provided for:

  * Home Assistant REST (``HassTransport``)
  * MQTT (``MQTTTransport``) — only constructed when paho-mqtt is present
  * a "noop" transport for development and tests

Every effect goes through ``WorldBridge.call(Channel.ENVIRONMENTAL_CHANGE,
...)`` so it inherits permission, conscience, and capability-token gates.

The reverse direction — env → substrate — uses ``observe()`` to inject an
event into the prediction-error stream tagged with provenance so it never
gets confused with internal state.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import urllib.parse
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.action_executor import ActionExecutor
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.IoTBridge")


# ─── transports ─────────────────────────────────────────────────────────────


@dataclass
class IoTEffect:
    target: str  # e.g. "thermostat.living_room", "light.studio"
    op: str  # set, increment, scene, etc.
    payload: dict[str, Any] = field(default_factory=dict)
    reason: str = ""  # human-readable rationale (logged, never user-visible)


class IoTTransport(ABC):
    name: str = "abstract"

    @abstractmethod
    async def apply(self, effect: IoTEffect) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    @abstractmethod
    async def observe(self) -> dict[str, Any] | None:  # pragma: no cover - interface
        raise NotImplementedError

    async def discover(self) -> list[dict[str, Any]]:
        return []


class NoopTransport(IoTTransport):
    name = "noop"

    def __init__(self) -> None:
        self.applied: list[IoTEffect] = []
        self.events: list[dict[str, Any]] = []

    async def apply(self, effect: IoTEffect) -> dict[str, Any]:
        self.applied.append(effect)
        return {
            "applied": True,
            "effect_verified": True,
            "transport": "noop",
            "target": effect.target,
            "op": effect.op,
        }

    async def observe(self) -> dict[str, Any] | None:
        if not self.events:
            return None
        return self.events.pop(0)


def _hass_values_match(expected: Any, observed: Any) -> bool:
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(float(observed) - float(expected)) <= 2.0
        except (TypeError, ValueError):
            return False
    if isinstance(expected, (list, tuple)):
        return list(observed or []) == list(expected)
    return str(observed) == str(expected)


def _hass_expected_attributes(effect: IoTEffect) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    for key, value in effect.payload.items():
        if key == "transition":
            continue
        if key == "brightness_pct":
            try:
                expected["brightness"] = round(
                    max(0.0, min(100.0, float(value))) * 255.0 / 100.0
                )
            except (TypeError, ValueError):
                expected["brightness"] = value
            continue
        expected[key] = value
    return expected


def _hass_state_matches_effect(state: dict[str, Any], effect: IoTEffect) -> bool:
    if str(state.get("entity_id") or "") != effect.target:
        return False
    state_value = str(state.get("state") or "").lower()
    if effect.op == "turn_on" and state_value != "on":
        return False
    if effect.op == "turn_off" and state_value != "off":
        return False
    attributes = state.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}
    return all(
        _hass_values_match(expected, attributes.get(key))
        for key, expected in _hass_expected_attributes(effect).items()
    )


def _bounded_hass_state(
    state: dict[str, Any] | None,
    effect: IoTEffect,
) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    attributes = state.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}
    expected_attributes = _hass_expected_attributes(effect)
    return {
        "entity_id": str(state.get("entity_id") or "")[:160],
        "state": str(state.get("state") or "")[:80],
        "attributes": {
            key: attributes.get(key)
            for key in expected_attributes
            if key in attributes
        },
        "last_changed": str(state.get("last_changed") or "")[:80],
    }


class HassTransport(IoTTransport):
    """Home Assistant REST transport.

    Configured by environment variables:
      AURA_HASS_URL    — e.g. "http://homeassistant.local:8123"
      AURA_HASS_TOKEN  — long-lived access token

    Refuses to operate without both.
    """

    name = "home_assistant"

    def __init__(self) -> None:
        self.token = str(
            os.getenv("AURA_HASS_TOKEN") or os.getenv("HASS_TOKEN") or ""
        ).strip()
        self.base = str(
            os.getenv("AURA_HASS_URL")
            or os.getenv("HASS_URL")
            or ("https://homeassistant.local:8123" if self.token else "")
        ).strip().rstrip("/")
        if not self.base or not self.token:
            raise RuntimeError("hass_credentials_missing")
        parsed = urllib.parse.urlparse(self.base)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise RuntimeError("hass_url_must_be_origin_only_http_or_https")
        if parsed.scheme == "http" and str(
            os.getenv("AURA_HASS_ALLOW_HTTP", "")
        ).strip().lower() not in {"1", "true", "yes", "on"}:
            raise RuntimeError("hass_insecure_http_requires_explicit_opt_in")

    async def apply(self, effect: IoTEffect) -> dict[str, Any]:
        domain, _, entity = effect.target.partition(".")
        if (
            not re.fullmatch(r"[a-z0-9_]+", domain)
            or not re.fullmatch(r"[a-z0-9_]+", entity)
            or not re.fullmatch(r"[a-z0-9_]+", effect.op)
        ):
            raise ValueError("invalid_home_assistant_effect")
        url = f"{self.base}/api/services/{domain}/{effect.op}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        body = {"entity_id": effect.target, **effect.payload}
        response = await ActionExecutor.request_network_transport(
            method="POST",
            url=url,
            headers=headers,
            data=json.dumps(body, separators=(",", ":")),
            timeout_s=8.0,
            source="world_bridge:iot.home_assistant.apply",
            read_only=False,
        )
        status = int(response.get("status_code") or 0)
        content = response.get("content") or b""
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        accepted = bool(response.get("ok") and 200 <= status < 300)
        observed_state: dict[str, Any] | None = None
        effect_verified = False
        if accepted:
            for attempt in range(3):
                state_response = await ActionExecutor.request_network_transport(
                    method="GET",
                    url=f"{self.base}/api/states/{effect.target}",
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout_s=8.0,
                    source="world_bridge:iot.home_assistant.verify",
                    read_only=True,
                )
                state_content = state_response.get("content") or b"{}"
                if isinstance(state_content, bytes):
                    state_content = state_content.decode("utf-8", errors="replace")
                try:
                    candidate = json.loads(str(state_content))
                except json.JSONDecodeError:
                    candidate = None
                if isinstance(candidate, dict):
                    observed_state = candidate
                    if _hass_state_matches_effect(candidate, effect):
                        effect_verified = True
                        break
                if attempt < 2:
                    await asyncio.sleep(0.2)
        return {
            "applied": accepted,
            "effect_verified": effect_verified,
            "status": status,
            "body": str(content)[:1024],
            "target": effect.target,
            "op": effect.op,
            "observed_state": _bounded_hass_state(observed_state, effect),
        }

    async def observe(self) -> dict[str, Any] | None:
        # Polling-style observation. A push variant would subscribe via
        # WebSocket; this minimal version exposes the structure.
        return None

    async def discover(self) -> list[dict[str, Any]]:
        response = await ActionExecutor.request_network_transport(
            method="GET",
            url=f"{self.base}/api/states",
            headers={"Authorization": f"Bearer {self.token}"},
            timeout_s=8.0,
            source="world_bridge:iot.home_assistant.discover",
            read_only=False,
        )
        if not bool(response.get("ok")):
            raise RuntimeError(str(response.get("error") or "hass_discovery_failed"))
        content = response.get("content") or b"[]"
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        decoded = json.loads(str(content))
        if not isinstance(decoded, list):
            raise ValueError("hass_discovery_response_not_list")
        return [item for item in decoded if isinstance(item, dict)][:5000]


# ─── policy: substrate → effect ────────────────────────────────────────────


@dataclass
class PolicyRule:
    name: str
    when: Callable[[dict[str, Any]], bool]
    effect: Callable[[dict[str, Any]], IoTEffect]
    cooldown_s: float = 60.0


def _read_substrate() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from core.container import ServiceContainer
        affect = ServiceContainer.get("affect_engine", default=None)
        if affect is not None and hasattr(affect, "snapshot"):
            out["affect"] = affect.snapshot() or {}
        homeo = ServiceContainer.get("homeostasis_engine", default=None) or ServiceContainer.get("homeostatic_engine", default=None)
        if homeo is not None and hasattr(homeo, "snapshot"):
            out["homeo"] = homeo.snapshot() or {}
    except (ImportError, AttributeError, RuntimeError):
        pass  # no-op: intentional
    try:
        import psutil
        out["cpu_pct"] = psutil.cpu_percent(interval=None)
        out["ram_pct"] = psutil.virtual_memory().percent
    except (ImportError, AttributeError, RuntimeError):
        pass  # no-op: intentional
    try:
        from core.organism.viability import get_viability
        out["viability"] = get_viability().state.value
    except (ImportError, AttributeError, RuntimeError):
        pass  # no-op: intentional
    return out


_DEFAULT_POLICY: list[PolicyRule] = [
    PolicyRule(
        name="cpu_hot_drop_thermostat",
        when=lambda s: float(s.get("cpu_pct", 0.0)) > 85.0,
        effect=lambda s: IoTEffect(
            target="climate.studio",
            op="set_temperature",
            payload={"temperature": 21.0},
            reason="cpu_hot",
        ),
        cooldown_s=180.0,
    ),
    PolicyRule(
        name="threat_red_room",
        when=lambda s: (s.get("affect", {}) or {}).get("prediction_error", 0.0) > 0.85,
        effect=lambda s: IoTEffect(
            target="light.studio",
            op="turn_on",
            payload={"rgb_color": [255, 32, 32], "brightness_pct": 25},
            reason="prediction_error_storm",
        ),
        cooldown_s=120.0,
    ),
    PolicyRule(
        name="curiosity_reading_light",
        when=lambda s: (s.get("affect", {}) or {}).get("curiosity", 0.0) > 0.75 and s.get("viability") == "healthy",
        effect=lambda s: IoTEffect(
            target="light.desk",
            op="turn_on",
            payload={"brightness_pct": 80, "color_temp_kelvin": 5200},
            reason="curiosity_high_idle",
        ),
        cooldown_s=600.0,
    ),
]


# ─── bridge ─────────────────────────────────────────────────────────────────


class IoTBridge:
    def __init__(self) -> None:
        self._transports: dict[str, IoTTransport] = {}
        self._policy: list[PolicyRule] = list(_DEFAULT_POLICY)
        self._last_fired: dict[str, float] = {}
        self._last_attempted: dict[str, float] = {}
        self._task: asyncio.Task[Any] | None = None
        self._observe_task: asyncio.Task[Any] | None = None
        self._running = False
        try:
            self.register_transport("home_assistant", HassTransport())
            logger.info("Home Assistant IoT transport configured.")
        except RuntimeError as exc:
            logger.info("Home Assistant IoT transport disabled: %s", exc)

    def register_transport(self, name: str, transport: IoTTransport) -> None:
        self._transports[name] = transport

    def replace_policy(self, rules: list[PolicyRule]) -> None:
        self._policy = list(rules)

    def append_rule(self, rule: PolicyRule) -> None:
        self._policy.append(rule)

    async def apply_authorized(
        self,
        effect: IoTEffect,
        *,
        capability_token: str,
    ) -> dict[str, Any]:
        """Apply one already-authorized effect to configured physical transports."""
        if not str(capability_token or "").strip():
            raise PermissionError("iot_capability_token_required")
        if not self._transports:
            raise RuntimeError("iot_transport_unavailable")

        results: list[dict[str, Any]] = []
        failures: list[str] = []
        for transport_name, transport in self._transports.items():
            try:
                output = await transport.apply(effect)
                if output.get("applied") is not True:
                    failures.append(
                        f"{transport_name}:{output.get('status') or output.get('error') or 'not_applied'}"
                    )
                elif output.get("effect_verified") is not True:
                    failures.append(f"{transport_name}:effect_unverified")
                results.append(
                    {
                        "transport": transport_name,
                        "target": effect.target,
                        "op": effect.op,
                        "output": output,
                    }
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation("iot_bridge", exc)
                failures.append(f"{transport_name}:{type(exc).__name__}:{exc}")
        transport_succeeded = any(
            isinstance(item.get("output"), dict)
            and item["output"].get("applied") is True
            for item in results
        )
        effect_verified = bool(
            results
            and not failures
            and all(
                isinstance(item.get("output"), dict)
                and item["output"].get("effect_verified") is True
                for item in results
            )
        )
        return {
            "effects": results,
            "transport_succeeded": transport_succeeded,
            "effect_verified": effect_verified,
            "failures": failures,
        }

    async def discover_authorized(
        self,
        *,
        capability_token: str,
    ) -> list[dict[str, Any]]:
        if not str(capability_token or "").strip():
            raise PermissionError("iot_capability_token_required")
        if not self._transports:
            raise RuntimeError("iot_transport_unavailable")
        devices: list[dict[str, Any]] = []
        for transport_name, transport in self._transports.items():
            discovered = await transport.discover()
            devices.extend(
                {"transport": transport_name, **item}
                for item in discovered
                if isinstance(item, dict)
            )
        return devices[:5000]

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "configured": bool(self._transports),
            "transports": sorted(self._transports),
            "policy_rules": len(self._policy),
        }

    async def tick(self) -> list[dict[str, Any]]:
        snapshot = _read_substrate()
        results: list[dict[str, Any]] = []
        now = time.time()
        for rule in self._policy:
            try:
                if not rule.when(snapshot):
                    continue
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('iot_bridge', exc)
                logger.debug("iot rule predicate failed: %s", exc)
                continue
            last = max(
                self._last_fired.get(rule.name, 0.0),
                self._last_attempted.get(rule.name, 0.0),
            )
            if (now - last) < rule.cooldown_s:
                continue
            try:
                effect = rule.effect(snapshot)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('iot_bridge', exc)
                logger.debug("iot rule effect build failed: %s", exc)
                continue
            from core.embodiment.world_bridge import Channel, get_world_bridge

            self._last_attempted[rule.name] = now
            result = await get_world_bridge().call(
                Channel.ENVIRONMENTAL_CHANGE,
                action=f"iot:{effect.target}:{effect.op}",
                intent=effect.reason or rule.name,
                payload={
                    "operation": "apply",
                    "target": effect.target,
                    "op": effect.op,
                    "effect": dict(effect.payload),
                    "reason": effect.reason or rule.name,
                },
            )
            results.append(
                {
                    "rule": rule.name,
                    "ok": result.ok,
                    "receipt_id": result.receipt_id,
                    "data": result.data,
                    "error": result.error,
                    "status": result.status,
                    "transport_succeeded": result.transport_succeeded,
                    "effect_verified": result.effect_verified,
                    "manual_reconciliation_required": (
                        result.manual_reconciliation_required
                    ),
                }
            )
            if result.ok:
                self._last_fired[rule.name] = now
        return results

    async def observe_loop(self) -> None:
        """Drain observations from all transports into the substrate's
        prediction-error stream. Each observation is tagged with
        ``source="iot:<transport>"`` so it never gets confused with
        internal-only signals.
        """
        while self._running:
            for tname, transport in self._transports.items():
                try:
                    obs = await transport.observe()
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('iot_bridge', exc)
                    logger.debug("iot observe failed (%s): %s", tname, exc)
                    obs = None
                if obs is None:
                    continue
                self._inject_to_substrate(tname, obs)
            await asyncio.sleep(2.0)

    @staticmethod
    def _inject_to_substrate(transport_name: str, observation: dict[str, Any]) -> None:
        try:
            from core.container import ServiceContainer
            sg = ServiceContainer.get("sensory_gate", default=None)
            if sg is not None and hasattr(sg, "ingest"):
                sg.ingest({"source": f"iot:{transport_name}", "observation": observation, "when": time.time()})
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('iot_bridge', exc)
            logger.debug("iot substrate inject failed: %s", exc)

    async def start(self, *, interval: float = 5.0) -> None:
        if self._running:
            return
        self._running = True

        async def _loop() -> None:
            while self._running:
                try:
                    await self.tick()
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation('iot_bridge', exc)
                    logger.debug("iot bridge tick failed: %s", exc)
                await asyncio.sleep(interval)

        self._task = get_task_tracker().create_task(_loop(), name="IoTBridge")
        self._observe_task = get_task_tracker().create_task(
            self.observe_loop(),
            name="IoTBridgeObserve",
        )

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # no-op: intentional
            self._task = None
        if self._observe_task is not None:
            self._observe_task.cancel()
            try:
                await self._observe_task
            except asyncio.CancelledError:
                pass
            self._observe_task = None


async def _environmental_change_handler(
    payload: dict[str, Any],
    *,
    capability_token: str,
) -> dict[str, Any]:
    bridge = get_iot_bridge()
    operation = str(payload.get("operation") or "apply").strip().lower()
    if operation == "discover":
        return {
            "devices": await bridge.discover_authorized(
                capability_token=capability_token,
            ),
            "transport_succeeded": True,
            "effect_verified": True,
        }
    if operation != "apply":
        raise ValueError(f"unknown_iot_operation:{operation}")
    target = str(payload.get("target") or "").strip().lower()
    action = str(payload.get("op") or "").strip().lower()
    effect_payload = payload.get("effect")
    if not target or not action or not isinstance(effect_payload, dict):
        raise ValueError("iot_effect_target_op_payload_required")
    effect = IoTEffect(
        target=target,
        op=action,
        payload=dict(effect_payload),
        reason=str(payload.get("reason") or "")[:240],
    )
    outcome = await bridge.apply_authorized(
        effect,
        capability_token=capability_token,
    )
    return outcome


_BRIDGE: IoTBridge | None = None


def get_iot_bridge() -> IoTBridge:
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = IoTBridge()
    return _BRIDGE


__all__ = [
    "IoTEffect",
    "IoTTransport",
    "NoopTransport",
    "HassTransport",
    "PolicyRule",
    "IoTBridge",
    "get_iot_bridge",
]
