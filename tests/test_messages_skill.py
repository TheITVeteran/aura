from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.skills.messages import MessagesInput, MessagesSkill


class _Transport:
    def __init__(self) -> None:
        self.calls = []

    def status(self):
        return {"running": True, "configured": True, "outbound_ready": True}

    async def send_authorized(self, **kwargs):
        self.calls.append(("send_authorized", kwargs))
        return {"ok": True, "accepted": True, "state": "accepted_unverified"}

    async def send_from_governed_context(self, **kwargs):
        self.calls.append(("send_governed", kwargs))
        return {"ok": True, "accepted": True, "state": "accepted_unverified"}

    async def set_paused_authorized(self, **kwargs):
        self.calls.append(("control", kwargs))
        return {"ok": True, "paused": kwargs["paused"]}


@pytest.mark.asyncio
async def test_messages_skill_exposes_status_send_and_control(service_container) -> None:
    transport = _Transport()
    service_container.register_instance("messages_transport", transport, required=False)
    skill = MessagesSkill()

    status = await skill.execute(MessagesInput(action="status"), {"source": "user"})
    sent = await skill.execute(
        MessagesInput(action="send", body="Hello from Aura", idempotency_key="skill-1"),
        {
            "source": "user",
            "foreground_request": True,
            "signed_capability": {"bound": True},
            "capability_token_id": "tool-capability-1",
        },
    )
    paused = await skill.execute(MessagesInput(action="pause"), {"source": "user"})

    assert status["status"]["configured"] is True
    assert sent["accepted"] is True
    assert paused["paused"] is True
    assert [kind for kind, _payload in transport.calls] == ["send_governed", "control"]


@pytest.mark.asyncio
async def test_messages_skill_authorizes_when_called_outside_capability_engine(
    service_container,
) -> None:
    transport = _Transport()
    service_container.register_instance("messages_transport", transport, required=False)

    result = await MessagesSkill().execute(
        MessagesInput(action="send", body="Hello", idempotency_key="skill-direct-1"),
        {"source": "messages"},
    )

    assert result["accepted"] is True
    assert [kind for kind, _payload in transport.calls] == ["send_authorized"]


def test_messages_skill_rejects_raw_destination_arguments() -> None:
    with pytest.raises(ValidationError):
        MessagesInput(action="send", body="hello", destination="+15550001111")


def test_messages_skill_is_discovered_with_external_effect_policy() -> None:
    from core.skills.discovery import build_skill_catalog

    catalog = build_skill_catalog(try_rust=False)
    declaration = next(item for item in catalog.accepted if item.name == "messages")
    assert declaration.effect_scope == "external_io"
    assert declaration.authority_class == "external_effect"
