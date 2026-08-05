from __future__ import annotations

import json
from typing import Any

import pytest

from core.embodiment.openhab_connector import OpenHABConnector, OpenHABTransport
from core.embodiment.reality_connectors import (
    build_configured_reality_connector_catalog,
)
from core.reality_reach.attachments import AttachmentAccess
from core.reality_reach.scalar_adapter import ScalarRealityAdapter
from core.runtime.action_executor import ActionExecutor


def _items() -> list[dict[str, Any]]:
    return [
        {
            "name": "DeskLight",
            "label": "Desk light",
            "type": "Dimmer",
            "state": "20",
            "groupNames": ["Office"],
            "tags": ["Lighting"],
        },
        {
            "name": "OfficeTemperature",
            "label": "Office temperature",
            "type": "Number",
            "state": "21.5 °C",
            "stateDescription": {"minimum": -50, "maximum": 100, "step": 0.1},
            "groupNames": ["Office"],
            "tags": ["Measurement"],
        },
        {
            "name": "DeskLightFeedback",
            "label": "Desk light feedback",
            "type": "Dimmer",
            "state": "20",
            "groupNames": ["Office"],
            "tags": ["Measurement"],
        },
        {"name": "Unusable", "type": "String", "state": "hello"},
    ]


@pytest.fixture
def openhab_env(monkeypatch):
    monkeypatch.setenv("AURA_OPENHAB_URL", "https://openhab.example.test")
    monkeypatch.setenv("AURA_OPENHAB_TOKEN", "private-token")
    monkeypatch.setenv("AURA_OPENHAB_INSTALLATION_ID", "house-alpha")
    monkeypatch.setenv("AURA_OPENHAB_CONTROL_ITEMS", "DeskLight")
    monkeypatch.setenv("AURA_OPENHAB_FEEDBACK_ITEMS", "DeskLight:DeskLightFeedback")
    monkeypatch.delenv("AURA_OPENHAB_ITEMS", raising=False)
    monkeypatch.delenv("AURA_OPENHAB_ALLOW_HTTP", raising=False)


@pytest.mark.asyncio
async def test_openhab_discovers_attaches_and_uses_separate_command_readback(
    monkeypatch,
    openhab_env,
) -> None:
    state = {item["name"]: dict(item) for item in _items()}
    calls: list[dict[str, Any]] = []

    async def request_network_transport(**kwargs):
        calls.append(kwargs)
        url = kwargs["url"]
        method = kwargs["method"]
        if method == "GET" and url.endswith("/rest/items"):
            return {"ok": True, "status_code": 200, "content": json.dumps(_items()).encode()}
        item_name = url.rsplit("/", 1)[-1]
        if method == "GET":
            return {
                "ok": True,
                "status_code": 200,
                "content": json.dumps(state[item_name]).encode(),
            }
        command = str(kwargs["data"])
        state[item_name]["state"] = command
        if item_name == "DeskLight":
            state["DeskLightFeedback"]["state"] = command
        return {"ok": True, "status_code": 202, "content": b""}

    monkeypatch.setattr(
        ActionExecutor,
        "request_network_transport",
        request_network_transport,
    )
    transport = OpenHABTransport()
    connector = OpenHABConnector(transport)
    candidates = await connector.discover()

    assert len(candidates) == 2
    light = next(item for item in candidates if item.metadata["item_name"] == "DeskLight")
    temperature = next(
        item for item in candidates if item.metadata["item_name"] == "OfficeTemperature"
    )
    assert light.access == (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL)
    assert temperature.access == (AttachmentAccess.OBSERVE,)
    assert light.persistent_identity is True

    adapter = await connector.attach(
        light,
        (AttachmentAccess.OBSERVE, AttachmentAccess.CONTROL),
    )
    assert isinstance(adapter, ScalarRealityAdapter)
    assert len(adapter.actuator_capabilities()) == 1
    reading = await adapter.refresh_readback()
    assert reading.value == 20.0
    assert reading.unit == "percent"

    result = await transport.write_scalar(
        "item.desklight",
        55.0,
        idempotency_key="test.openhab.write",
    )
    assert result.accepted is True
    assert state["DeskLight"]["state"] == "55"
    readback = await transport.read_scalar("item.desklight")
    assert readback.value == 55.0
    post = next(call for call in calls if call["method"] == "POST")
    assert post["source"] == "reality_reach:openhab.actuate"
    assert post["headers"]["X-Aura-Idempotency-Key"] == "test.openhab.write"
    assert "private-token" not in json.dumps(result.receipt)


@pytest.mark.asyncio
async def test_openhab_observe_only_attachment_removes_actuation(
    monkeypatch,
    openhab_env,
) -> None:
    item = _items()[0]

    async def request_network_transport(**kwargs):
        content = [item] if kwargs["url"].endswith("/rest/items") else item
        return {"ok": True, "status_code": 200, "content": json.dumps(content).encode()}

    monkeypatch.setattr(
        ActionExecutor,
        "request_network_transport",
        request_network_transport,
    )
    connector = OpenHABConnector(OpenHABTransport())
    candidate = (await connector.discover())[0]
    adapter = await connector.attach(candidate, (AttachmentAccess.OBSERVE,))
    assert adapter.actuator_capabilities() == ()
    assert len(adapter.declarations()) == 1


def test_openhab_rejects_plain_http_without_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("AURA_OPENHAB_URL", "http://openhab.example.test")
    monkeypatch.setenv("AURA_OPENHAB_TOKEN", "private-token")
    monkeypatch.delenv("AURA_OPENHAB_ALLOW_HTTP", raising=False)
    with pytest.raises(RuntimeError, match="insecure_http"):
        OpenHABTransport()


def test_openhab_connector_catalog_registers_only_complete_configuration(
    openhab_env,
) -> None:
    registered = []

    class Broker:
        @staticmethod
        def register_connector(connector):
            registered.append(connector)

    catalog = build_configured_reality_connector_catalog()
    before = catalog.status()
    catalog.register_with(Broker())
    after = catalog.status()

    assert before["configured"] == 1
    assert before["registered"] == 0
    assert after["ready"] is True
    assert after["registered"] == 1
    assert after["connectors"][0]["connector_id"] == "openhab.local"
    assert registered[0].connector_id == "openhab.local"


def test_openhab_connector_catalog_exposes_partial_configuration(monkeypatch) -> None:
    monkeypatch.setenv("AURA_OPENHAB_URL", "https://openhab.example.test")
    monkeypatch.delenv("AURA_OPENHAB_TOKEN", raising=False)

    status = build_configured_reality_connector_catalog().status()

    assert status["ready"] is False
    assert status["configured"] == 1
    assert status["registered"] == 0
    assert status["connectors"][0]["state"] == "invalid"
    assert status["connectors"][0]["error"] == "AURA_OPENHAB_TOKEN is missing"


@pytest.mark.asyncio
async def test_openhab_control_without_distinct_feedback_stays_observe_only(
    monkeypatch,
    openhab_env,
) -> None:
    monkeypatch.delenv("AURA_OPENHAB_FEEDBACK_ITEMS", raising=False)

    async def request_network_transport(**kwargs):
        content = _items() if kwargs["url"].endswith("/rest/items") else _items()[0]
        return {"ok": True, "status_code": 200, "content": json.dumps(content).encode()}

    monkeypatch.setattr(
        ActionExecutor,
        "request_network_transport",
        request_network_transport,
    )
    candidate = next(
        item
        for item in await OpenHABConnector(OpenHABTransport()).discover()
        if item.metadata["item_name"] == "DeskLight"
    )

    assert candidate.access == (AttachmentAccess.OBSERVE,)
    assert candidate.metadata["control_available"] is False
    assert candidate.metadata["independent_readback"] is False


@pytest.mark.asyncio
async def test_openhab_numeric_manifest_does_not_change_with_reading(
    monkeypatch,
    openhab_env,
) -> None:
    items = _items()

    async def request_network_transport(**kwargs):
        return {
            "ok": True,
            "status_code": 200,
            "content": json.dumps(items).encode(),
        }

    monkeypatch.setattr(
        ActionExecutor,
        "request_network_transport",
        request_network_transport,
    )
    connector = OpenHABConnector(OpenHABTransport())
    before = next(
        item
        for item in await connector.discover()
        if item.metadata["item_name"] == "OfficeTemperature"
    )
    items[1]["state"] = "87.2 °C"
    after = next(
        item
        for item in await connector.discover()
        if item.metadata["item_name"] == "OfficeTemperature"
    )

    assert after.identity_fingerprint == before.identity_fingerprint
    assert after.manifest_sha256 == before.manifest_sha256
    assert after.candidate_id == before.candidate_id
