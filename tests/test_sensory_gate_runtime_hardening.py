import json
import urllib.error
from pathlib import Path

import pytest

from core.bus.sensory_gate import SensoryGateActor
from tools.audit_degradation import analyze_file


def _actor_without_bus() -> SensoryGateActor:
    actor = SensoryGateActor.__new__(SensoryGateActor)
    actor.browser = None
    actor._is_active = True
    actor._shutdown_event = None
    # CP126: authority/liveness state the real __init__ installs.
    actor._authorized_principals = ()
    actor._shutdown_token = ""
    actor._used_shutdown_nonces = set()
    actor._shutdown_reason = ""
    actor._supervisor_pid = 0
    actor._last_observation_ts = 0.0
    actor._heartbeat_failures = 0
    return actor


def test_sensory_gate_degradation_audit_is_clean():
    assert analyze_file(Path("core/bus/sensory_gate.py")) == []


def test_search_result_formatting_tolerates_mismatched_wikipedia_arrays():
    data = ["aura", ["Aura", "Aura 2"], ["first snippet"], ["https://example.com/aura"]]

    assert SensoryGateActor._format_search_results(data) == [
        "Aura: first snippet (https://example.com/aura)"
    ]


@pytest.mark.asyncio
async def test_browse_without_browser_fails_closed():
    actor = _actor_without_bus()

    result = await actor._handle_browse({"url": "https://github.com/anthropics"}, "trace-1")

    # CP126 c8c56e76: refusals are schema'd receipts, not bare error dicts.
    assert result["ok"] is False
    assert result["error"] == "browser_unavailable"
    assert result["kind"] == "browse"
    assert result["trace_id"] == "trace-1"


@pytest.mark.asyncio
async def test_search_handles_mismatched_response_without_crashing(monkeypatch):
    class FakeGateway:
        def request(self, method, url, **kwargs):
            return {
                "ok": True,
                "content": json.dumps(
                    ["aura", ["Aura", "Aura 2"], ["first snippet"], ["https://example.com/aura"]]
                ).encode(),
            }

    monkeypatch.setattr(
        "core.bus.sensory_gate.get_network_gateway", lambda: FakeGateway()
    )
    actor = _actor_without_bus()

    result = await actor._handle_search({"query": "aura"}, "trace-2")

    assert result["results"] == ["Aura: first snippet (https://example.com/aura)"]
    assert result["observation_only"] is True


@pytest.mark.asyncio
async def test_search_network_error_returns_structured_error(monkeypatch):
    urlopen_calls = []

    def fail_urlopen(*_args, **_kwargs):
        urlopen_calls.append((_args, _kwargs))
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    actor = _actor_without_bus()

    result = await actor._handle_search({"query": "aura"}, "trace-3")

    assert len(urlopen_calls) == 1
    assert "offline" in result["error"]


@pytest.mark.asyncio
async def test_shutdown_handler_flips_actor_state_and_event():
    import asyncio

    actor = _actor_without_bus()
    actor._shutdown_event = asyncio.Event()

    result = await actor._handle_shutdown({}, "trace-4")

    # CP126 4dc9dc31: unauthenticated shutdown still works only when no
    # supervisor token is configured, and the acknowledgement is durable.
    assert result["ok"] is True
    assert result["acknowledged"] is True
    assert result["trace_id"] == "trace-4"
    assert actor._is_active is False
    assert actor._shutdown_event.is_set()
