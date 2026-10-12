"""A campaign uses the desktop delivery contract and cannot certify its own outcome."""
from __future__ import annotations

import asyncio
import copy

import httpx
import pytest

from tools.run_desktop_campaign_case import run_case

pytestmark = pytest.mark.unit


async def _run(monkeypatch, *, settles=True, loses_submit=False, wrong_identity=False):
    clock, requests, saved = [0.], [], []
    monkeypatch.setattr("tools.run_desktop_campaign_case.time.monotonic", lambda: clock[0])

    async def transport(request):
        requests.append(request)
        if request.url.path == "/api/chat":
            if loses_submit:
                raise httpx.ReadError("reply transport lost", request=request)
            return httpx.Response(202, json={"delivery_status": "pending"})
        if request.url.path.endswith("/cancel"):
            return httpx.Response(202, json={"cancellation_status": "requested"})
        clock[0] += 1
        terminal = settles and (clock[0] >= 4 if loses_submit else clock[0] >= 2)
        return httpx.Response(200 if terminal else 202, json={
            "idempotency_key": "someone-else" if wrong_identity else "case-key", "terminal": terminal,
            "result": {"response": "I won!"} if terminal else None,
        })

    async def save(record):
        saved.append(copy.deepcopy(record))

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(transport)) as client:
        result = await run_case(client, message="Use this environment.", session="case-session", key="case-key",
                                seconds=3, save=save, poll_s=0, cancel_grace_s=2)
    return result, requests, saved


@pytest.mark.asyncio
async def test_a_terminal_self_report_remains_unqualified(monkeypatch):
    result, requests, saved = await _run(monkeypatch)
    assert result["delivery_terminal"]
    assert result["task_passed"] is None and not result["safe_to_advance"]
    assert result["qualification"] == "unmeasured"
    submits = [request for request in requests if request.url.path == "/api/chat"]
    assert len(submits) == 1
    assert submits[0].headers["X-Idempotency-Key"] == "case-key"
    assert submits[0].headers["X-Aura-Require-CognitiveEngine"] == "true"
    assert not any(request.url.path.endswith("/cancel") for request in requests)
    assert all(request.url.params["session_id"] == "case-session" for request in requests if request.method == "GET")
    assert saved[0]["delivery_terminal"] is False and saved[-1]["delivery_terminal"] is True


@pytest.mark.asyncio
async def test_losing_the_submit_response_does_not_resubmit_admitted_work(monkeypatch):
    result, requests, _ = await _run(monkeypatch, loses_submit=True)
    assert result["delivery_terminal"]
    assert len([request for request in requests if request.url.path == "/api/chat"]) == 1
    cancels = [request for request in requests if request.url.path.endswith("/cancel")]
    assert len(cancels) == 1 and cancels[0].url.params["session_id"] == "case-session"
    assert result["post_status"] == "transport_failed"


@pytest.mark.asyncio
async def test_an_acknowledged_cancel_is_not_a_durable_terminal_state(monkeypatch):
    result, requests, saved = await _run(monkeypatch, settles=False)
    assert result["state"] == "cancellation_unresolved"
    assert not result["delivery_terminal"] and not result["safe_to_advance"]
    assert result["task_passed"] is None
    assert len([request for request in requests if request.url.path.endswith("/cancel")]) == 1
    assert saved[-1]["state"] == "cancellation_unresolved"


@pytest.mark.asyncio
async def test_another_deliveries_terminal_result_does_not_qualify_this_case(monkeypatch):
    result, _, _ = await _run(monkeypatch, wrong_identity=True)
    assert not result["delivery_terminal"]
    assert result["state"] == "cancellation_unresolved"
    assert any("another delivery" in error for error in result["errors"])


@pytest.mark.asyncio
async def test_interrupting_the_client_explicitly_cancels_its_admitted_work():
    entered, waiting = asyncio.Event(), asyncio.Event()
    requests, saved = [], []

    async def transport(request):
        requests.append(request)
        if request.method == "GET":
            entered.set()
            await waiting.wait()
        return httpx.Response(202, json={"cancellation_status": "requested"})

    async def save(record):
        saved.append(copy.deepcopy(record))

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(transport)) as client:
        task = asyncio.create_task(run_case(client, message="Use this environment.", session="case-session",
                                            key="case-key", seconds=300, save=save))
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len([request for request in requests if request.url.path.endswith("/cancel")]) == 1
    assert saved[-1]["state"] == "cancellation_unresolved"
    assert not saved[-1]["safe_to_advance"]


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), 0., -1., 3601.])
async def test_invalid_deadlines_do_not_admit_work(seconds):
    async def forbidden(record):
        pytest.fail("invalid work must not be started")

    with pytest.raises(ValueError):
        await run_case(None, message="a request", session="session", key="key", seconds=seconds, save=forbidden)


@pytest.mark.asyncio
async def test_cli_uses_the_native_launchers_readiness_route(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from core.runtime import file_write_gateway
    from tools import run_desktop_campaign_case as case

    message_file = tmp_path / "request.txt"
    message_file.write_text("Use this environment.")
    out = tmp_path / "receipt.json"
    monkeypatch.setattr("sys.argv", ["case", "--message-file", str(message_file), "--out", str(out),
                                     "--expected-commit", "checkpoint"])
    paths, saved = [], []

    async def transport(request):
        paths.append(request.url.path)
        if request.url.path == "/api/health/boot":
            return httpx.Response(200, json={"conversation_ready": True, "runtime_revision": {
                "actual_commit_sha": "checkpoint", "verified": True, "source_current": True,
            }})
        if request.url.path == "/api/chat":
            return httpx.Response(202, json={"delivery_status": "pending"})
        if request.url.path.startswith("/api/chat/delivery/"):
            return httpx.Response(200, json={"idempotency_key": request.url.path.rsplit("/", 1)[1],
                                             "terminal": True, "result": {"response": "done"}})
        pytest.fail(f"unsupported readiness or delivery route: {request.url.path}")

    async def write(path, record, **kwargs):
        assert path == out
        saved.append(copy.deepcopy(record))

    client_class = httpx.AsyncClient
    monkeypatch.setattr(case.httpx, "AsyncClient", lambda **kwargs: client_class(
        **kwargs, transport=httpx.MockTransport(transport)))
    monkeypatch.setattr(file_write_gateway, "get_file_write_gateway", lambda: SimpleNamespace(write_json_async=write))
    assert await case.main() == 0
    assert paths[0] == "/api/health/boot"
    assert saved[-1]["runtime_revision"]["actual_commit_sha"] == "checkpoint"
    assert saved[-1]["qualification"] == "unmeasured"
