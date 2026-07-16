from __future__ import annotations

import asyncio
import json
import sqlite3
import stat
from pathlib import Path

import pytest
from fastapi.responses import JSONResponse
from starlette.requests import Request

from core.runtime.chat_delivery_journal import (
    AdmissionKind,
    ChatDeliveryFenceLost,
    ChatDeliveryJournal,
    ChatDeliveryJournalCorruption,
    ChatDeliveryJournalUnavailable,
    DeliveryIdentity,
    DeliveryState,
    canonical_request_hash,
)


def _identity(
    key: str = "turn-1",
    *,
    principal: str = "owner:bryan",
    session_id: str = "session-1",
) -> DeliveryIdentity:
    return DeliveryIdentity.create(
        principal=principal,
        session_id=session_id,
        idempotency_key=key,
    )


def _request_hash(message: str = "hello") -> str:
    return canonical_request_hash({"message": message})


def _request(
    key: str,
    *,
    path: str = "/api/chat",
    method: str = "POST",
) -> Request:
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": [(b"x-idempotency-key", key.encode("ascii"))],
            "client": ("127.0.0.1", 50123),
            "server": ("127.0.0.1", 8000),
        }
    )


def _payload(response: JSONResponse) -> dict[str, object]:
    decoded = json.loads(bytes(response.body))
    assert isinstance(decoded, dict)
    return decoded


@pytest.fixture
def journal(tmp_path: Path) -> ChatDeliveryJournal:
    return ChatDeliveryJournal(
        tmp_path / "runtime" / "chat.sqlite3",
        stale_after_s=2.0,
        retention_s=60.0,
        abandon_after_s=30.0,
        poll_interval_s=0.01,
    )


def test_initialization_uses_private_filesystem_permissions(tmp_path: Path) -> None:
    path = tmp_path / "private" / "chat.sqlite3"

    ChatDeliveryJournal(path)

    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_existing_malformed_database_fails_closed_without_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chat.sqlite3"
    original = b"not-a-sqlite-database"
    path.write_bytes(original)

    with pytest.raises(ChatDeliveryJournalCorruption):
        ChatDeliveryJournal(path)

    assert path.read_bytes() == original


def test_cached_factory_does_not_resolve_away_symlink_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.runtime import chat_delivery_journal as journal_module

    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"")
    link = tmp_path / "chat.sqlite3"
    link.symlink_to(target)
    monkeypatch.setenv("AURA_CHAT_DELIVERY_DB", str(link))
    journal_module.reset_chat_delivery_journals_for_test()

    with pytest.raises(ChatDeliveryJournalCorruption):
        journal_module.get_chat_delivery_journal()


@pytest.mark.parametrize(
    ("principal", "session_id"),
    (
        ("p" * 241, "session"),
        ("principal", "s" * 241),
        ("principal", "session\x00collision"),
    ),
)
def test_identity_rejects_values_that_could_alias_or_poison_storage(
    principal: str,
    session_id: str,
) -> None:
    with pytest.raises(ValueError):
        DeliveryIdentity.create(
            principal=principal,
            session_id=session_id,
            idempotency_key="safe-key",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("stale_after_s", float("nan")),
        ("retention_s", float("inf")),
        ("poll_interval_s", 0.0),
        ("busy_timeout_s", -1.0),
        ("max_rows", True),
    ),
)
def test_invalid_configuration_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError):
        ChatDeliveryJournal(tmp_path / f"{field}.sqlite3", **kwargs)


@pytest.mark.asyncio
async def test_concurrent_same_request_executes_once_then_replays(
    journal: ChatDeliveryJournal,
) -> None:
    identity = _identity()
    digest = _request_hash()
    owner = await journal.reserve(identity, digest, wait_timeout_s=0)
    assert owner.kind is AdmissionKind.EXECUTE

    waiter = asyncio.create_task(journal.reserve(identity, digest, wait_timeout_s=1.0))
    await asyncio.sleep(0.03)
    assert not waiter.done()

    terminal = await journal.finalize(
        owner,
        state=DeliveryState.COMPLETED,
        http_status=200,
        response={"response": "hello", "status": "ok"},
    )
    replay = await waiter

    assert replay.kind is AdmissionKind.REPLAY
    assert replay.record.turn_id == owner.record.turn_id == terminal.turn_id
    assert replay.record.response == {"response": "hello", "status": "ok"}


@pytest.mark.asyncio
async def test_same_key_different_payload_is_rejected_without_wait(
    journal: ChatDeliveryJournal,
) -> None:
    identity = _identity()
    owner = await journal.reserve(identity, _request_hash("first"), wait_timeout_s=0)

    mismatch = await journal.reserve(
        identity,
        _request_hash("different"),
        wait_timeout_s=5.0,
    )

    assert owner.kind is AdmissionKind.EXECUTE
    assert mismatch.kind is AdmissionKind.MISMATCH
    assert mismatch.record.turn_id == owner.record.turn_id


@pytest.mark.asyncio
async def test_same_key_isolated_by_principal_and_session(
    journal: ChatDeliveryJournal,
) -> None:
    digest = _request_hash()
    owner = await journal.reserve(_identity(), digest, wait_timeout_s=0)
    other_principal = await journal.reserve(
        _identity(principal="paired:device-2"),
        digest,
        wait_timeout_s=0,
    )
    other_session = await journal.reserve(
        _identity(session_id="session-2"),
        digest,
        wait_timeout_s=0,
    )

    assert owner.kind is AdmissionKind.EXECUTE
    assert other_principal.kind is AdmissionKind.EXECUTE
    assert other_session.kind is AdmissionKind.EXECUTE
    assert (
        len(
            {
                owner.record.turn_id,
                other_principal.record.turn_id,
                other_session.record.turn_id,
            }
        )
        == 3
    )


@pytest.mark.asyncio
async def test_expired_running_owner_becomes_ambiguous_not_reexecuted(
    tmp_path: Path,
) -> None:
    short = ChatDeliveryJournal(
        tmp_path / "chat.sqlite3",
        stale_after_s=0.05,
        poll_interval_s=0.01,
    )
    identity = _identity()
    digest = _request_hash()
    owner = await short.reserve(identity, digest, wait_timeout_s=0)
    await asyncio.sleep(0.07)

    recovered = await short.reserve(identity, digest, wait_timeout_s=0)

    assert recovered.kind is AdmissionKind.REPLAY
    assert recovered.record.state is DeliveryState.AMBIGUOUS
    assert recovered.record.http_status == 409
    assert recovered.record.response is not None
    assert recovered.record.response["status"] == "delivery_ambiguous"
    with pytest.raises(ChatDeliveryFenceLost):
        await short.finalize(
            owner,
            state=DeliveryState.COMPLETED,
            http_status=200,
            response={"response": "late"},
        )


@pytest.mark.asyncio
async def test_status_read_reconciles_expired_running_owner(
    tmp_path: Path,
) -> None:
    short = ChatDeliveryJournal(
        tmp_path / "chat.sqlite3",
        stale_after_s=0.05,
        poll_interval_s=0.01,
    )
    identity = _identity()
    owner = await short.reserve(identity, _request_hash(), wait_timeout_s=0)
    await asyncio.sleep(0.07)

    status = await short.get(identity)

    assert status is not None
    assert status.state is DeliveryState.AMBIGUOUS
    assert status.turn_id == owner.record.turn_id
    assert status.http_status == 409


@pytest.mark.asyncio
async def test_terminal_receipt_survives_journal_recreation(tmp_path: Path) -> None:
    path = tmp_path / "chat.sqlite3"
    first = ChatDeliveryJournal(path)
    identity = _identity()
    digest = _request_hash()
    owner = await first.reserve(identity, digest, wait_timeout_s=0)
    await first.finalize(
        owner,
        state=DeliveryState.COMPLETED,
        http_status=200,
        response={"response": "durable", "status": "ok"},
    )

    second = ChatDeliveryJournal(path)
    replay = await second.reserve(identity, digest, wait_timeout_s=0)

    assert replay.kind is AdmissionKind.REPLAY
    assert replay.record.turn_id == owner.record.turn_id
    assert replay.record.response == {"response": "durable", "status": "ok"}


@pytest.mark.asyncio
async def test_tampered_terminal_receipt_fails_closed(
    journal: ChatDeliveryJournal,
) -> None:
    identity = _identity()
    owner = await journal.reserve(identity, _request_hash(), wait_timeout_s=0)
    await journal.finalize(
        owner,
        state=DeliveryState.COMPLETED,
        http_status=200,
        response={"response": "sealed"},
    )
    with sqlite3.connect(journal.db_path) as conn:
        conn.execute(
            "UPDATE chat_deliveries SET response_hash=? WHERE turn_id=?",
            ("0" * 64, owner.record.turn_id),
        )

    with pytest.raises(ChatDeliveryJournalCorruption):
        await journal.get(identity)


@pytest.mark.asyncio
async def test_active_capacity_fails_closed_instead_of_growing_unbounded(
    tmp_path: Path,
) -> None:
    bounded = ChatDeliveryJournal(tmp_path / "chat.sqlite3", max_rows=10)
    digest = _request_hash()
    for index in range(10):
        admission = await bounded.reserve(
            _identity(f"turn-{index}"),
            digest,
            wait_timeout_s=0,
        )
        assert admission.kind is AdmissionKind.EXECUTE

    with pytest.raises(ChatDeliveryJournalUnavailable):
        await bounded.reserve(_identity("turn-overflow"), digest, wait_timeout_s=0)


@pytest.mark.asyncio
async def test_capacity_evicts_oldest_terminal_receipt_before_rejecting_new_work(
    tmp_path: Path,
) -> None:
    bounded = ChatDeliveryJournal(tmp_path / "chat.sqlite3", max_rows=10)
    digest = _request_hash()
    first_identity = _identity("turn-0")
    for index in range(10):
        identity = _identity(f"turn-{index}")
        admission = await bounded.reserve(identity, digest, wait_timeout_s=0)
        await bounded.finalize(
            admission,
            state=DeliveryState.COMPLETED,
            http_status=200,
            response={"response": str(index)},
        )

    replacement = await bounded.reserve(
        _identity("turn-replacement"),
        digest,
        wait_timeout_s=0,
    )

    assert replacement.kind is AdmissionKind.EXECUTE
    assert await bounded.get(first_identity) is None
    with sqlite3.connect(bounded.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM chat_deliveries").fetchone()[0] == 10


def _patch_route_identity(monkeypatch: pytest.MonkeyPatch, journal: ChatDeliveryJournal) -> None:
    from interface.routes import chat as chat_mod

    monkeypatch.setattr(chat_mod, "get_chat_delivery_journal", lambda: journal)
    monkeypatch.setattr(
        chat_mod,
        "request_access_profile",
        lambda _request: {"surface": "owner", "conversation_only": False},
    )
    monkeypatch.setattr(
        chat_mod,
        "_authenticated_chat_principal",
        lambda _request: "owner:bryan",
    )
    monkeypatch.setattr(
        chat_mod,
        "_observe_authenticated_chat_turn",
        lambda _request, _body: "owner:bryan",
    )
    monkeypatch.setattr(
        chat_mod,
        "_attach_http_chat_delivery_receipt",
        lambda *_args, **_kwargs: None,
    )


@pytest.mark.asyncio
async def test_route_concurrent_duplicate_runs_handler_once(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    @chat_mod._paired_chat_response_boundary
    async def handler(*, body, request):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return JSONResponse({"response": "one", "status": "ok"})

    request = _request("route-race")
    body = chat_mod.ChatRequest(message="hello", session_id="session-1")
    first_task = asyncio.create_task(handler(body=body, request=request))
    await entered.wait()
    second_task = asyncio.create_task(handler(body=body, request=request))
    await asyncio.sleep(0.03)
    assert calls == 1
    release.set()

    first, second = await asyncio.gather(first_task, second_task)
    first_payload = _payload(first)
    second_payload = _payload(second)
    assert calls == 1
    assert first_payload["turn_id"] == second_payload["turn_id"]
    assert first_payload["delivery_replayed"] is False
    assert second_payload["delivery_replayed"] is True


@pytest.mark.asyncio
async def test_route_cancellation_seals_ambiguous_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)
    entered = asyncio.Event()
    calls = 0

    @chat_mod._paired_chat_response_boundary
    async def handler(*, body, request):
        nonlocal calls
        calls += 1
        entered.set()
        await asyncio.Event().wait()
        return JSONResponse({"response": "unreachable"})

    request = _request("route-cancel")
    body = chat_mod.ChatRequest(message="act", session_id="session-1")
    task = asyncio.create_task(handler(body=body, request=request))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    replay = await handler(body=body, request=request)
    payload = _payload(replay)
    assert calls == 1
    assert replay.status_code == 409
    assert payload["status"] == "delivery_ambiguous"
    assert payload["delivery_replayed"] is True


@pytest.mark.asyncio
async def test_route_same_key_different_message_returns_409_without_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)
    calls = 0

    @chat_mod._paired_chat_response_boundary
    async def handler(*, body, request):
        nonlocal calls
        calls += 1
        return JSONResponse({"response": body.message, "status": "ok"})

    request = _request("route-mismatch")
    first = chat_mod.ChatRequest(message="first", session_id="session-1")
    second = chat_mod.ChatRequest(message="second", session_id="session-1")
    assert (await handler(body=first, request=request)).status_code == 200

    mismatch = await handler(body=second, request=request)

    assert mismatch.status_code == 409
    assert _payload(mismatch)["status"] == "idempotency_payload_mismatch"
    assert calls == 1


def test_request_contract_binds_method_and_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from interface.routes import chat as chat_mod

    monkeypatch.setattr(
        chat_mod,
        "request_access_profile",
        lambda _request: {"surface": "owner", "conversation_only": False},
    )
    body = chat_mod.ChatRequest(message="hello", session_id="session-1")
    _, chat_hash, _ = chat_mod._chat_delivery_request_contract(
        _request("same-key", path="/api/chat"),
        body,
        exact_principal="owner:bryan",
    )
    _, regenerate_hash, _ = chat_mod._chat_delivery_request_contract(
        _request("same-key", path="/api/chat/regenerate"),
        body,
        exact_principal="owner:bryan",
    )

    assert chat_hash != regenerate_hash


@pytest.mark.asyncio
async def test_authenticated_status_endpoint_returns_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
    journal: ChatDeliveryJournal,
) -> None:
    from interface.routes import chat as chat_mod

    _patch_route_identity(monkeypatch, journal)
    identity = _identity("status-key")
    owner = await journal.reserve(identity, _request_hash(), wait_timeout_s=0)
    terminal = await journal.finalize(
        owner,
        state=DeliveryState.COMPLETED,
        http_status=200,
        response={"response": "recovered", "status": "ok"},
    )

    response = await chat_mod.api_chat_delivery_status(
        "status-key",
        _request("ignored", method="GET"),
        session_id="session-1",
    )
    payload = _payload(response)

    assert response.status_code == 200
    assert payload["turn_id"] == terminal.turn_id
    assert payload["result"] == {"response": "recovered", "status": "ok"}
