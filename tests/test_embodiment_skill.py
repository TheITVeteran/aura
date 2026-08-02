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


@pytest.mark.asyncio
async def test_aura_can_authorize_pending_attachment_through_her_will(monkeypatch) -> None:
    import core.skills.embodiment_skill as skill_module

    intent = {
        "schema": "aura.reality-attachment-authority.intent.v1",
        "requested_access": ["observe"],
        "scope": "reality_attachment.observe",
        "grant_ttl_s": 3600,
    }
    attached = SimpleNamespace(
        state=ConnectionState.ATTACHED,
        authority_receipt_id="will-physical-1",
        to_dict=lambda: {"request_id": "request-1", "state": "attached"},
    )

    class Broker:
        def authority_intent(self, request_id, **kwargs):
            assert request_id == "request-1"
            assert kwargs == {"persistent": True, "grant_ttl_s": 3600}
            return intent

        async def authorize_and_attach(self, request_id, **kwargs):
            assert request_id == "request-1"
            assert kwargs["authority_capability"] == {"signed": "capability"}
            assert kwargs["persistent"] is True
            assert kwargs["grant_ttl_s"] == 3600
            return attached

    class Will:
        def decide(self, **kwargs):
            assert kwargs["source"] == "embodiment_skill"
            assert kwargs["domain"].value == "environment_action"
            assert kwargs["context"]["foreground_request"] is True
            assert kwargs["context"]["verification_required"] is True
            return SimpleNamespace(receipt_id="will-physical-1")

    class Capability:
        @staticmethod
        def to_dict():
            return {"signed": "capability"}

    class Issuer:
        def issue_from_decision(self, decision, **kwargs):
            assert decision.receipt_id == "will-physical-1"
            assert kwargs["payload"] == intent
            assert kwargs["scope"] == "reality_attachment.observe"
            return Capability()

    monkeypatch.setattr(
        skill_module,
        "_service",
        lambda name: Broker() if name == "reality_attachment_broker" else None,
    )
    monkeypatch.setattr(skill_module, "get_will", lambda: Will())
    monkeypatch.setattr(skill_module, "get_capability_issuer", lambda: Issuer())

    result = await EmbodimentSkill().execute(
        {
            "action": "authorize_connection",
            "request_id": "request-1",
            "persistent": "true",
            "grant_ttl_s": "3600",
        },
        {"foreground_request": True},
    )

    assert result["ok"] is True
    assert result["authority_receipt_id"] == "will-physical-1"
    assert result["grant_ttl_s"] == 3600


@pytest.mark.asyncio
async def test_trust_rotation_uses_broker_custody_boundary(monkeypatch) -> None:
    import core.skills.embodiment_skill as skill_module

    class Broker:
        async def rotate_trust_custody(self):
            return {"sequence": 4, "key_version": 2}

        def status(self):
            return {"persistent_trust_ready": True}

    broker = Broker()
    monkeypatch.setattr(
        skill_module,
        "_service",
        lambda name: broker if name == "reality_attachment_broker" else None,
    )

    result = await EmbodimentSkill().execute({"action": "rotate_trust_custody"}, {})

    assert result["ok"] is True
    assert result["rotation_receipt"] == {"sequence": 4, "key_version": 2}
