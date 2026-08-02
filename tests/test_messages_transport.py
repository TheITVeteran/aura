from __future__ import annotations

import asyncio
import json
import sqlite3
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.communication.contact_directory import MessagesContact
from core.communication.messages_journal import (
    MessagesDeliveryJournal,
    MessagesJournalCorruptionError,
    content_digest,
)
from core.communication.messages_transport import (
    HistoryMessage,
    MessagesHistoryReader,
    MessagesJXADriver,
    MessagesTransport,
    MessagesTransportError,
)
from core.executive.execution_policy import (
    canonical_authority_arguments,
    canonical_authority_context,
)


def _contact() -> MessagesContact:
    return MessagesContact(
        alias="primary_operator",
        destination="+15550001111",
        destination_kind="phone",
        endpoint_ref="msg_" + "a" * 32,
        service_preference="auto",
        allow_inbound=True,
        allow_outbound=True,
        created_at=1000.0,
        updated_at=1000.0,
    )


def _send_arguments(body: str, idempotency_key: str) -> dict[str, object]:
    return canonical_authority_arguments(
        "messages",
        {
            "action": "send",
            "alias": "primary_operator",
            "body": body,
            "idempotency_key": idempotency_key,
        },
    )


class _Directory:
    async def load_async(self, alias: str) -> MessagesContact:
        assert alias == "primary_operator"
        return _contact()


def _history_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE handle (id TEXT NOT NULL)")
        connection.execute(
            "CREATE TABLE message (guid TEXT, text TEXT, handle_id INTEGER, is_from_me INTEGER)"
        )
        owner = connection.execute(
            "INSERT INTO handle(id) VALUES(?)",
            ("+15550001111",),
        ).lastrowid
        stranger = connection.execute(
            "INSERT INTO handle(id) VALUES(?)",
            ("+15559990000",),
        ).lastrowid
        connection.execute(
            "INSERT INTO message(guid, text, handle_id, is_from_me) VALUES(?, ?, ?, 0)",
            ("owner-in-1", "hello from owner", owner),
        )
        connection.execute(
            "INSERT INTO message(guid, text, handle_id, is_from_me) VALUES(?, ?, ?, 0)",
            ("stranger-in-1", "ignore me", stranger),
        )
        connection.execute(
            "INSERT INTO message(guid, text, handle_id, is_from_me) VALUES(?, ?, ?, 1)",
            ("owner-out-1", "hello from aura", owner),
        )
        connection.execute(
            "INSERT INTO message(guid, text, handle_id, is_from_me) VALUES(?, NULL, ?, 0)",
            ("rich-only", owner),
        )
        connection.commit()
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_history_reader_is_exact_read_only_and_directional(tmp_path: Path) -> None:
    path = tmp_path / "chat.db"
    _history_db(path)
    reader = MessagesHistoryReader(path)

    assert (await reader.probe())["readable"] is True
    inbound = await reader.messages_after(
        "+15550001111",
        from_me=False,
        after_row_id=0,
    )
    outbound = await reader.messages_after(
        "+15550001111",
        from_me=True,
        after_row_id=0,
    )

    assert [(item.guid, item.text) for item in inbound] == [
        ("owner-in-1", "hello from owner")
    ]
    assert [(item.guid, item.text) for item in outbound] == [
        ("owner-out-1", "hello from aura")
    ]


@pytest.mark.asyncio
async def test_jxa_driver_keeps_destination_and_body_out_of_argv(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class _Gateway:
        async def run_async(self, argv, **kwargs):
            observed["argv"] = tuple(argv)
            observed["input"] = kwargs["input"]
            return subprocess.CompletedProcess(
                argv,
                0,
                '{"ok":true,"accepted":true,"transport":"messages"}',
                "",
            )

    monkeypatch.setattr(MessagesJXADriver, "preflight", staticmethod(lambda: None))
    driver = MessagesJXADriver(_Gateway())
    result = await driver.send(
        destination="+15550001111",
        body="private body",
        service_preference="auto",
    )

    argv = " ".join(observed["argv"])
    assert "+15550001111" not in argv
    assert "private body" not in argv
    assert json.loads(observed["input"]) == {
        "body": "private body",
        "destination": "+15550001111",
        "service": "auto",
    }
    assert result == {"accepted": True, "transport": "messages"}


@pytest.mark.asyncio
async def test_jxa_driver_accepts_osascript_string_encoded_json(monkeypatch) -> None:
    class _Gateway:
        async def run_async(self, argv, **_kwargs):
            receipt = '{"ok":true,"accepted":true,"transport":"messages"}'
            return subprocess.CompletedProcess(argv, 0, json.dumps(receipt), "")

    monkeypatch.setattr(MessagesJXADriver, "preflight", staticmethod(lambda: None))

    assert await MessagesJXADriver(_Gateway()).send(
        destination="+15550001111",
        body="private body",
        service_preference="auto",
    ) == {"accepted": True, "transport": "messages"}


def test_messages_authority_envelope_is_content_hiding_and_idempotent() -> None:
    private_body = "A private sentence that must not enter governance receipts."
    envelope = _send_arguments(private_body, "authority-envelope-1")
    context = canonical_authority_context(
        "messages",
        {"message": private_body, "objective": private_body, "foreground_request": True},
    )

    assert private_body not in json.dumps(envelope, sort_keys=True)
    assert private_body not in json.dumps(context, sort_keys=True)
    assert envelope["body_chars"] == len(private_body)
    assert envelope["body_bytes"] == len(private_body.encode("utf-8"))
    assert canonical_authority_arguments("messages", envelope) == envelope
    assert context["foreground_request"] is True


class _History:
    def __init__(self) -> None:
        self.outbound: list[HistoryMessage] = []

    async def latest_row_id(self, _destination: str, *, from_me: bool) -> int:
        assert from_me is True
        return max((item.row_id for item in self.outbound), default=0)

    async def messages_after(
        self,
        _destination: str,
        *,
        from_me: bool,
        after_row_id: int,
        limit: int,
    ) -> list[HistoryMessage]:
        assert from_me is True
        return [item for item in self.outbound if item.row_id > after_row_id][:limit]


class _Driver:
    def __init__(self, history: _History, *, fail: bool = False) -> None:
        self.history = history
        self.fail = fail
        self.calls = 0

    def preflight(self) -> None:
        return None

    async def send(self, *, destination: str, body: str, service_preference: str):
        self.calls += 1
        if self.fail:
            raise MessagesTransportError("synthetic_ambiguous_failure")
        self.history.outbound.append(
            HistoryMessage(row_id=10 + self.calls, guid=f"out-{self.calls}", text=body)
        )
        return {"accepted": True}


@pytest.mark.asyncio
async def test_transport_verifies_local_history_and_replays_without_resend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    history = _History()
    driver = _Driver(history)
    journal = MessagesDeliveryJournal(tmp_path / "journal.sqlite3")
    transport = MessagesTransport(
        chat_turn=lambda *_args, **_kwargs: None,
        directory=_Directory(),
        journal=journal,
        history=history,
        driver=driver,
    )
    monkeypatch.setattr(
        "core.communication.messages_transport.enforce_capability",
        lambda *_args, **_kwargs: None,
    )
    authority = SimpleNamespace(signed_capability={"bound": True})
    arguments = _send_arguments("private body", "stable-send-1")

    first = await transport.send(
        alias="primary_operator",
        body="private body",
        idempotency_key="stable-send-1",
        authority=authority,
        arguments=arguments,
    )
    second = await transport.send(
        alias="primary_operator",
        body="private body",
        idempotency_key="stable-send-1",
        authority=authority,
        arguments=arguments,
    )

    assert first["state"] == "verified_local_history"
    assert first["remote_delivery_verified"] is False
    assert second["state"] == "verified_local_history"
    assert driver.calls == 1
    serialized = json.dumps(first, sort_keys=True)
    assert "private body" not in serialized
    assert "+15550001111" not in serialized


@pytest.mark.asyncio
async def test_governed_skill_send_reuses_existing_authority_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    history = _History()
    driver = _Driver(history)
    transport = MessagesTransport(
        chat_turn=lambda *_args, **_kwargs: None,
        directory=_Directory(),
        journal=MessagesDeliveryJournal(tmp_path / "journal.sqlite3"),
        history=history,
        driver=driver,
    )
    verified: list[tuple[str, str]] = []

    class _ExistingAuthorityGateway:
        def verify_tool_access(self, tool_name: str, token_id: str) -> bool:
            verified.append((tool_name, token_id))
            return True

    monkeypatch.setattr(
        "core.executive.authority_gateway.get_authority_gateway",
        lambda: _ExistingAuthorityGateway(),
    )
    monkeypatch.setattr(
        "core.communication.messages_transport.require_governance",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "core.communication.messages_transport.enforce_capability",
        lambda *_args, **_kwargs: None,
    )

    result = await transport.send_from_governed_context(
        alias="primary_operator",
        body="private governed body",
        idempotency_key="governed-send-1",
        context={
            "capability_token_id": "capability-1",
            "signed_capability": {"bound": True},
        },
    )

    assert result["state"] == "verified_local_history"
    assert verified == [("messages", "capability-1")]
    assert driver.calls == 1


@pytest.mark.asyncio
async def test_transport_never_retries_ambiguous_send(tmp_path: Path, monkeypatch) -> None:
    history = _History()
    driver = _Driver(history, fail=True)
    transport = MessagesTransport(
        chat_turn=lambda *_args, **_kwargs: None,
        directory=_Directory(),
        journal=MessagesDeliveryJournal(tmp_path / "journal.sqlite3"),
        history=history,
        driver=driver,
    )
    monkeypatch.setattr(
        "core.communication.messages_transport.enforce_capability",
        lambda *_args, **_kwargs: None,
    )
    authority = SimpleNamespace(signed_capability={"bound": True})
    arguments = _send_arguments("private body", "stable-send-2")

    first = await transport.send(
        alias="primary_operator",
        body="private body",
        idempotency_key="stable-send-2",
        authority=authority,
        arguments=arguments,
    )
    second = await transport.send(
        alias="primary_operator",
        body="private body",
        idempotency_key="stable-send-2",
        authority=authority,
        arguments=arguments,
    )

    assert first["state"] == "ambiguous"
    assert second["state"] == "ambiguous"
    assert driver.calls == 1


@pytest.mark.asyncio
async def test_journal_restart_marks_inflight_send_ambiguous(tmp_path: Path) -> None:
    path = tmp_path / "journal.sqlite3"
    endpoint_ref = _contact().endpoint_ref
    body_sha256 = content_digest("private body")
    journal = MessagesDeliveryJournal(path)
    await journal.admit_outbound(
        idempotency_key="restart-proof-1",
        endpoint_ref=endpoint_ref,
        content_sha256=body_sha256,
        baseline_row_id=4,
    )
    claimed = await journal.mark_outbound_sending("restart-proof-1")
    assert claimed.may_execute is True

    reopened = MessagesDeliveryJournal(path)
    recovered = await reopened.lookup_outbound(
        idempotency_key="restart-proof-1",
        endpoint_ref=endpoint_ref,
        content_sha256=body_sha256,
    )

    assert recovered is not None
    assert recovered.state == "ambiguous"
    assert recovered.may_execute is False
    assert recovered.attempts == 1
    assert recovered.error_code == "process_interrupted_after_effect_admission"


@pytest.mark.asyncio
async def test_journal_refuses_idempotency_reuse_with_different_content(
    tmp_path: Path,
) -> None:
    journal = MessagesDeliveryJournal(tmp_path / "journal.sqlite3")
    endpoint_ref = _contact().endpoint_ref
    await journal.admit_outbound(
        idempotency_key="conflict-proof-1",
        endpoint_ref=endpoint_ref,
        content_sha256=content_digest("first body"),
        baseline_row_id=None,
    )

    with pytest.raises(MessagesJournalCorruptionError, match="different content"):
        await journal.lookup_outbound(
            idempotency_key="conflict-proof-1",
            endpoint_ref=endpoint_ref,
            content_sha256=content_digest("different body"),
        )


@pytest.mark.asyncio
async def test_journal_is_owner_only_and_never_persists_plaintext(tmp_path: Path) -> None:
    path = tmp_path / "journal.sqlite3"
    private_body = "plaintext must not be in the delivery ledger"
    journal = MessagesDeliveryJournal(path)
    await journal.admit_outbound(
        idempotency_key="content-hiding-1",
        endpoint_ref=_contact().endpoint_ref,
        content_sha256=content_digest(private_body),
        baseline_row_id=None,
    )

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if candidate.exists():
            assert private_body.encode("utf-8") not in candidate.read_bytes()


@pytest.mark.asyncio
async def test_transport_rate_limit_does_not_hide_prior_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    history = _History()
    driver = _Driver(history)
    transport = MessagesTransport(
        chat_turn=lambda *_args, **_kwargs: None,
        directory=_Directory(),
        journal=MessagesDeliveryJournal(tmp_path / "journal.sqlite3"),
        history=history,
        driver=driver,
        clock=lambda: 1000.0,
    )
    monkeypatch.setattr(
        "core.communication.messages_transport.enforce_capability",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("core.communication.messages_transport._BURST_SEND_LIMIT", 1)
    authority = SimpleNamespace(signed_capability={"bound": True})

    first = await transport.send(
        alias="primary_operator",
        body="first private body",
        idempotency_key="rate-proof-1",
        authority=authority,
        arguments=_send_arguments("first private body", "rate-proof-1"),
    )
    blocked = await transport.send(
        alias="primary_operator",
        body="second private body",
        idempotency_key="rate-proof-2",
        authority=authority,
        arguments=_send_arguments("second private body", "rate-proof-2"),
    )
    replay = await transport.send(
        alias="primary_operator",
        body="first private body",
        idempotency_key="rate-proof-1",
        authority=authority,
        arguments=_send_arguments("first private body", "rate-proof-1"),
    )

    assert first["state"] == "verified_local_history"
    assert blocked["error_code"] == "messages_rate_limited"
    assert replay["state"] == "verified_local_history"
    assert driver.calls == 1


@pytest.mark.asyncio
async def test_concurrent_duplicate_send_executes_effect_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    history = _History()
    driver = _Driver(history)
    transport = MessagesTransport(
        chat_turn=lambda *_args, **_kwargs: None,
        directory=_Directory(),
        journal=MessagesDeliveryJournal(tmp_path / "journal.sqlite3"),
        history=history,
        driver=driver,
    )
    monkeypatch.setattr(
        "core.communication.messages_transport.enforce_capability",
        lambda *_args, **_kwargs: None,
    )
    authority = SimpleNamespace(signed_capability={"bound": True})
    arguments = _send_arguments("concurrent private body", "concurrent-proof-1")

    results = await asyncio.gather(
        *(
            transport.send(
                alias="primary_operator",
                body="concurrent private body",
                idempotency_key="concurrent-proof-1",
                authority=authority,
                arguments=arguments,
            )
            for _ in range(2)
        )
    )

    assert {result["state"] for result in results} == {"verified_local_history"}
    assert driver.calls == 1


@pytest.mark.asyncio
async def test_inbound_uses_canonical_messages_surface_and_durable_replay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed: list[dict[str, object]] = []

    async def chat_turn(message: str, **kwargs):
        observed.append({"message": message, **kwargs})
        return "Aura's reply"

    transport = MessagesTransport(
        chat_turn=chat_turn,
        directory=_Directory(),
        journal=MessagesDeliveryJournal(tmp_path / "journal.sqlite3"),
        history=_History(),
        driver=_Driver(_History()),
    )

    async def accepted(**_kwargs):
        return {"ok": True, "accepted": True, "state": "accepted_unverified"}

    monkeypatch.setattr(transport, "send_authorized", accepted)
    contact = _contact()
    await transport._journal.prime_cursor(contact.endpoint_ref, 0)
    message = HistoryMessage(row_id=7, guid="incoming-guid", text="Hello, Aura")

    assert await transport._process_inbound(contact, message) is True
    assert await transport._process_inbound(contact, message) is True
    assert len(observed) == 1
    assert observed[0]["surface"] == "messages"
    assert observed[0]["session_id"] == "messages-primary_operator"
    assert str(observed[0]["idempotency_key"]).startswith("messages-in-")
    assert await transport._journal.cursor(contact.endpoint_ref) == 7
    assert content_digest("Aura's reply")


def test_messages_source_contains_no_operator_destination() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "core" / "communication" / "contact_directory.py",
        root / "core" / "communication" / "messages_transport.py",
        root / "core" / "skills" / "messages.py",
        root / "tools" / "configure_messages_contact.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "DEFAULT_PHONE" not in source
    assert "phone_number" not in source
    assert "destination: str = Field" not in source
