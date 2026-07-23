"""CP126 contract tests for the sensory gate actor."""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

from core.bus import sensory_gate as module
from core.bus.sensory_gate import SENSORY_RESULT_SCHEMA, SensoryGateActor
from core.runtime import url_policy


def _actor(**overrides) -> SensoryGateActor:
    actor = SensoryGateActor.__new__(SensoryGateActor)
    actor.browser = None
    actor.bus = SimpleNamespace(is_running=True, send=None)
    actor._is_active = True
    actor._shutdown_event = None
    actor._authorized_principals = ()
    actor._shutdown_token = ""
    actor._used_shutdown_nonces = set()
    actor._shutdown_reason = ""
    actor._supervisor_pid = os.getpid()
    actor._last_observation_ts = 0.0
    actor._last_heartbeat_ok_ts = 0.0
    actor._heartbeat_failures = 0
    actor._heartbeat_interval = 0.01
    actor._background_tasks = set()
    for key, value in overrides.items():
        setattr(actor, key, value)
    return actor


class _Browser:
    def __init__(self, *, navigates=True, content="Hello world", final_url=None):
        self._navigates = navigates
        self._content = content
        self.page = SimpleNamespace(url=final_url or "https://github.com/anthropics")
        self.is_active = True
        self.visited = []

    async def browse(self, url):
        self.visited.append(url)
        return self._navigates

    async def read_content(self):
        return self._content

    def get_status(self):
        return {"is_active": self.is_active}


@pytest.fixture(autouse=True)
def _allow_public_hosts(monkeypatch):
    """Keep the policy real but skip live DNS."""
    monkeypatch.setattr(url_policy, "ip_is_public", lambda addr: True)
    monkeypatch.setattr(
        url_policy.socket,
        "getaddrinfo",
        lambda *a, **k: [(0, 0, 0, "", ("93.184.216.34", 443))],
    )


# --- 3bba0f36: browse is under the canonical outbound policy ---------------


@pytest.mark.parametrize(
    "hostile",
    [
        "file:///etc/passwd",
        "http://127.0.0.1:8000/admin",
        "https://user:pass@github.com/x",
        "https://169.254.169.254/latest/meta-data/",
        "https://internal.corp.example/secrets",
        "javascript:alert(1)",
        "https://github.com:22/x",
    ],
)
def test_hostile_urls_never_reach_the_browser(hostile):
    browser = _Browser()
    actor = _actor(browser=browser)

    result = asyncio.run(actor._handle_browse({"url": hostile}, "trace"))

    assert result["ok"] is False
    assert browser.visited == []
    assert result["schema"] == SENSORY_RESULT_SCHEMA


def test_an_allowlisted_url_is_observed():
    browser = _Browser()
    actor = _actor(browser=browser)

    result = asyncio.run(actor._handle_browse({"url": "https://github.com/anthropics"}, "t"))

    assert result["ok"] is True
    assert browser.visited == ["https://github.com/anthropics"]
    assert result["policy"]["allowed"] is True


def test_the_decision_is_receipted():
    actor = _actor(browser=_Browser())

    result = asyncio.run(actor._handle_browse({"url": "https://evil.example/x"}, "t"))

    assert result["policy"]["policy"] == "core.runtime.url_policy"
    assert result["policy"]["host"] == "evil.example"
    assert "allowlist" in result["error"]


def test_an_unauthorized_principal_is_refused():
    actor = _actor(browser=_Browser(), _authorized_principals=("cognition",))

    refused = asyncio.run(
        actor._handle_browse({"url": "https://github.com/x", "principal": "nobody"}, "t")
    )
    allowed = asyncio.run(
        actor._handle_browse({"url": "https://github.com/x", "principal": "cognition"}, "t")
    )

    assert refused["ok"] is False and "not authorized" in refused["error"]
    assert allowed["ok"] is True and allowed["principal"] == "cognition"


def test_a_request_with_no_principal_is_refused_when_a_roster_exists():
    actor = _actor(browser=_Browser(), _authorized_principals=("cognition",))

    result = asyncio.run(actor._handle_browse({"url": "https://github.com/x"}, "t"))

    assert result["ok"] is False and "no principal" in result["error"]


def test_a_redirect_out_of_policy_discards_the_observation():
    browser = _Browser(final_url="https://internal.corp.example/secrets")
    actor = _actor(browser=browser)

    result = asyncio.run(actor._handle_browse({"url": "https://github.com/x"}, "t"))

    assert result["ok"] is False
    assert "redirect left policy" in result["error"]
    assert "content" not in result


# --- c8c56e76: the result says what was actually observed ------------------


def test_navigation_boolean_is_not_returned_as_content():
    actor = _actor(
        browser=_Browser(content="the real page text", final_url="https://github.com/x")
    )

    result = asyncio.run(actor._handle_browse({"url": "https://github.com/x"}, "t"))

    assert result["content"] == "the real page text"
    assert result["content"] is not True
    assert result["content_chars"] == len("the real page text")
    assert len(result["content_sha256"]) == 64
    assert result["complete"] is True
    assert result["final_url"]
    assert result["redirected"] is False
    assert result["observed_at"] > 0


def test_failed_navigation_is_distinguishable_from_an_observation():
    actor = _actor(browser=_Browser(navigates=False))

    result = asyncio.run(actor._handle_browse({"url": "https://github.com/x"}, "t"))

    assert result["ok"] is False
    assert result["error"] == "navigation_failed"
    assert "content" not in result


def test_extraction_failure_is_reported_as_incomplete():
    class NoText(_Browser):
        async def read_content(self):
            raise RuntimeError("page detached")

    actor = _actor(browser=NoText())

    result = asyncio.run(actor._handle_browse({"url": "https://github.com/x"}, "t"))

    assert result["ok"] is False
    assert result["navigated"] is True
    assert "page detached" in result["extraction_error"]
    assert result["complete"] is False


def test_oversized_content_is_truncated_and_declared():
    actor = _actor(browser=_Browser(content="x" * (module.MAX_CONTENT_CHARS + 500)))

    result = asyncio.run(actor._handle_browse({"url": "https://github.com/x"}, "t"))

    assert result["truncated"] is True
    assert result["complete"] is False
    assert result["content_chars"] == module.MAX_CONTENT_CHARS


# --- 1fb6515c: one deadline for the whole operation ------------------------


def test_a_slow_browse_is_bounded_by_its_deadline():
    class Slow(_Browser):
        async def browse(self, url):
            await asyncio.sleep(5)
            return True

    actor = _actor(browser=Slow())

    result = asyncio.run(
        actor._handle_browse({"url": "https://github.com/x", "deadline_s": 0.05}, "t")
    )

    assert result["ok"] is False
    assert "deadline" in result["error"]
    assert result["elapsed_s"] >= 0.0


def test_deadlines_are_clamped_and_validated():
    actor = _actor()

    assert actor._deadline({"deadline_s": 10_000}, 45.0) == module.MAX_REQUEST_DEADLINE_S
    assert actor._deadline({"deadline_s": -1}, 45.0) == 45.0
    assert actor._deadline({"deadline_s": "soon"}, 45.0) == 45.0
    assert actor._deadline({}, 45.0) == 45.0


def test_background_task_cancellation_is_bounded():
    async def scenario():
        actor = _actor()

        async def stubborn():
            while True:
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    continue

        task = asyncio.ensure_future(stubborn())
        task.set_name("stubborn")
        actor._background_tasks.add(task)
        loop = asyncio.get_running_loop()
        started = loop.time()
        await actor._cancel_background_tasks(timeout=0.2)
        elapsed = loop.time() - started
        task.cancel()
        return elapsed

    assert asyncio.run(scenario()) < 2.0


# --- 4dc9dc31: shutdown requires the supervisor ---------------------------


def test_unauthenticated_shutdown_is_refused_when_a_token_is_configured():
    async def scenario():
        actor = _actor(_shutdown_token="s3cret")
        actor._shutdown_event = asyncio.Event()
        return await actor._handle_shutdown({}, "t"), actor

    result, actor = asyncio.run(scenario())

    assert result["ok"] is False
    assert result["error"] == "shutdown_token_mismatch"
    assert actor._is_active is True


def test_a_shutdown_nonce_cannot_be_replayed():
    async def scenario():
        actor = _actor(_shutdown_token="s3cret")
        actor._shutdown_event = asyncio.Event()
        first = await actor._handle_shutdown({"token": "s3cret", "nonce": "n1"}, "t")
        actor._is_active = True
        second = await actor._handle_shutdown({"token": "s3cret", "nonce": "n1"}, "t")
        return first, second

    first, second = asyncio.run(scenario())

    assert first["ok"] is True and first["authenticated"] is True
    assert second["ok"] is False and second["error"] == "shutdown_nonce_replayed"


def test_a_token_without_a_nonce_is_refused():
    async def scenario():
        actor = _actor(_shutdown_token="s3cret")
        actor._shutdown_event = asyncio.Event()
        return await actor._handle_shutdown({"token": "s3cret"}, "t")

    assert asyncio.run(scenario())["error"] == "shutdown_nonce_required"


def test_shutdown_acknowledgement_is_durable():
    async def scenario():
        actor = _actor(_shutdown_token="s3cret")
        actor._shutdown_event = asyncio.Event()
        result = await actor._handle_shutdown(
            {"token": "s3cret", "nonce": "n2", "reason": "rolling restart"}, "trace-9"
        )
        return result, actor

    result, actor = asyncio.run(scenario())

    assert result["reason"] == "rolling restart"
    assert result["trace_id"] == "trace-9"
    assert result["pid"] == os.getpid()
    assert actor._shutdown_reason == "rolling restart"


# --- b25cb82b: a lost supervisor stops the actor ---------------------------


def test_a_dead_supervisor_requests_shutdown():
    async def scenario():
        actor = _actor(_supervisor_pid=999_999_999)
        actor._shutdown_event = asyncio.Event()
        await asyncio.wait_for(actor._liveness_loop(), timeout=2.0)
        return actor

    actor = asyncio.run(scenario())

    assert actor._is_active is False
    assert "gone" in actor._shutdown_reason


def test_repeated_heartbeat_failure_stops_the_actor():
    async def scenario():
        actor = _actor()
        actor._shutdown_event = asyncio.Event()

        async def failing_send(*args, **kwargs):
            raise OSError("pipe is gone")

        actor.bus = SimpleNamespace(is_running=True, send=failing_send)
        await asyncio.wait_for(actor._heartbeat_loop(), timeout=5.0)
        return actor

    actor = asyncio.run(scenario())

    assert actor._is_active is False
    assert "heartbeat_unreachable" in actor._shutdown_reason
    assert actor._heartbeat_failures >= module.MAX_HEARTBEAT_FAILURES


# --- 62c1e3f3: health reflects real readiness ------------------------------


def test_health_is_degraded_without_a_browser():
    snapshot = _actor()._health_snapshot()

    assert snapshot["status"] == "degraded"
    assert snapshot["browser"] == "absent"


def test_health_is_healthy_only_when_everything_is_ready():
    snapshot = _actor(browser=_Browser())._health_snapshot()

    assert snapshot["status"] == "healthy"
    assert snapshot["browser"] == "active"
    assert snapshot["bus_ready"] is True


def test_a_wedged_browser_is_not_reported_healthy():
    browser = _Browser()
    browser.is_active = False

    snapshot = _actor(browser=browser)._health_snapshot()

    assert snapshot["status"] == "degraded"
    assert snapshot["browser"] == "inactive"


def test_a_failing_browser_probe_is_reported():
    class BadProbe(_Browser):
        def get_status(self):
            raise RuntimeError("browser died")

    snapshot = _actor(browser=BadProbe())._health_snapshot()

    assert snapshot["status"] == "degraded"
    assert "probe_failed" in snapshot["browser"]


def test_health_reports_observation_recency():
    actor = _actor(browser=_Browser())
    asyncio.run(actor._handle_browse({"url": "https://github.com/x"}, "t"))

    snapshot = actor._health_snapshot()

    assert snapshot["last_observation_age_s"] is not None
    assert snapshot["last_observation_age_s"] < 5
