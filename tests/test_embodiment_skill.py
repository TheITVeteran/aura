from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.reality_reach.attachments import AttachmentAccess, ConnectionState
from core.skills.embodiment_skill import EmbodimentSkill


class WorldBridge:
    def __init__(self) -> None:
        self.calls = []

    async def call(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(
            ok=True,
            receipt_id="post-test-physical",
            status="completed",
            data={"effect_verified": True},
            error="",
            transport_succeeded=True,
            effect_verified=True,
            manual_reconciliation_required=False,
        )


@pytest.mark.asyncio
async def test_physical_command_uses_world_bridge_and_never_direct_device_execution(
    monkeypatch,
) -> None:
    import core.skills.embodiment_skill as skill_module

    world = WorldBridge()
    monkeypatch.setattr(skill_module, "get_world_bridge", lambda: world)
    skill = EmbodimentSkill()

    result = await skill.execute(
        {
            "action": "command_device",
            "device_id": "relay-1",
            "command": "turn_on",
            "parameters": {},
        },
        {},
    )

    assert result["ok"] is True
    assert result["effect_verified"] is True
    payload = world.calls[0][1]["payload"]
    assert payload["operation"] == "hardware_apply"
    assert payload["target"] == "relay-1"
    assert payload["idempotency_key"].startswith("embodiment.")


@pytest.mark.asyncio
async def test_aura_can_discover_and_choose_to_propose_a_connection(monkeypatch) -> None:
    import core.skills.embodiment_skill as skill_module

    candidate = SimpleNamespace(
        candidate_id="test.candidate",
        to_dict=lambda: {"candidate_id": "test.candidate"},
    )
    request = SimpleNamespace(
        state=ConnectionState.PENDING_TRUST,
        to_dict=lambda: {
            "request_id": "test.request",
            "state": "pending_trust",
        },
    )

    class Broker:
        async def discover(self):
            return (candidate,)

        def requests(self):
            return (request,)

        async def request_connection(self, candidate_id, **kwargs):
            assert candidate_id == "test.candidate"
            assert kwargs["initiated_by"] == "aura"
            assert kwargs["requested_access"] == (
                AttachmentAccess.OBSERVE,
                AttachmentAccess.CONTROL,
            )
            return request

    broker = Broker()
    monkeypatch.setattr(
        skill_module,
        "_service",
        lambda name: broker if name == "reality_attachment_broker" else None,
    )
    skill = EmbodimentSkill()

    discovery = await skill.execute({"action": "discover"}, {})
    proposal = await skill.execute(
        {
            "action": "request_connection",
            "candidate_id": "test.candidate",
            "access": "control",
            "reason": "I want to use this device",
        },
        {},
    )

    assert discovery["ok"] is True
    assert discovery["candidates"] == [{"candidate_id": "test.candidate"}]
    assert proposal["ok"] is True
    assert proposal["connection_request"]["state"] == "pending_trust"
    assert "not been invented" in proposal["summary"]


def test_inventory_reports_physical_limbs_as_part_of_auras_body(monkeypatch) -> None:
    import core.skills.embodiment_skill as skill_module

    services = {
        "hardware_manager": SimpleNamespace(
            list_devices=lambda: [{"device_id": "relay-1"}]
        ),
        "reality_reach": SimpleNamespace(
            declarations=lambda: (
                SimpleNamespace(to_dict=lambda: {"channel_id": "relay-1.power"}),
            )
        ),
        "reality_observation_router": SimpleNamespace(
            status=lambda: {"ready": True}
        ),
        "reality_attachment_broker": SimpleNamespace(
            status=lambda: {"attached": 1}
        ),
        "body_schema": SimpleNamespace(
            get_body_map=lambda: {
                "reality_sensor_a": {
                    "source": "reality:relay-1",
                    "limb_type": "sensor",
                },
                "web_search": {"source": "core.skills.web_search"},
            }
        ),
    }
    monkeypatch.setattr(skill_module, "_service", lambda name: services.get(name))

    inventory = EmbodimentSkill._inventory()

    assert inventory["ok"] is True
    assert list(inventory["physical_limbs"]) == ["reality_sensor_a"]
    assert "1 channels" in inventory["summary"]
