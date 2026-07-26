"""CP126 contract tests for the sanctioned raw CDP transport.

CDP can navigate, read any page, run arbitrary JavaScript and clear cookies.
Being the "approved adapter" only means something if the adapter checks.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from core.adapters import chrome_cdp_transport as module
from core.adapters.chrome_cdp_transport import (
    DEFAULT_TIMEOUT_S,
    MAX_FRAME_BYTES,
    MAX_RETAINED_EVENTS,
    MAX_TIMEOUT_S,
    CdpPolicyError,
    cdp_call,
    cdp_call_async,
    classify_method,
    validate_target,
    validated_timeout,
)

GOOD_URL = "ws://localhost:9222/devtools/page/ABC"


class _FakeWS:
    """A websocket stand-in that replays scripted frames."""

    def __init__(self, frames, *, recv_error=None):
        self.frames = list(frames)
        self.sent: list[str] = []
        self.closed = False
        self.recv_error = recv_error
        self.timeouts: list[float] = []

    def send(self, payload):
        self.sent.append(payload)

    def settimeout(self, value):
        self.timeouts.append(value)

    def recv(self):
        if self.recv_error is not None:
            raise self.recv_error
        if not self.frames:
            raise _LibTimeout("timed out")
        return self.frames.pop(0)

    def close(self):
        self.closed = True


class _LibTimeout(Exception):
    """Stands in for websocket-client's own WebSocketTimeoutException."""


@pytest.fixture()
def websocket(monkeypatch):
    """Install a fake `websocket` module and expose the connection it makes."""
    holder: dict = {}

    class _Module:
        @staticmethod
        def create_connection(url, timeout=None):
            holder["url"] = url
            holder["timeout"] = timeout
            return holder["ws"]

    import sys

    monkeypatch.setitem(sys.modules, "websocket", _Module)
    return holder


def _reply(**extra):
    return json.dumps({"id": 1, "result": {"ok": True}, **extra})


# --- 57ab9887: the destination is validated ------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "ws://evil.example.com:9222/x",
        "ws://10.0.0.5:9222/x",
        "http://localhost:9222/x",
        "wss://attacker.net/x",
        "file:///etc/passwd",
        "",
        "ws://localhost:9222/x\nInjected: header",
    ],
)
def test_a_non_loopback_or_malformed_target_is_refused(url):
    with pytest.raises(CdpPolicyError):
        validate_target(url)


@pytest.mark.parametrize(
    "url",
    ["ws://localhost:9222/devtools/page/A", "ws://127.0.0.1:9222/x", "ws://[::1]:9222/x"],
)
def test_loopback_targets_are_accepted(url):
    assert validate_target(url) == url


def test_the_target_is_checked_before_any_socket_opens(websocket):
    websocket["ws"] = _FakeWS([_reply()])

    with pytest.raises(CdpPolicyError):
        cdp_call("ws://evil.example.com:9222/x", "DOM.getDocument", {})

    assert "url" not in websocket


# --- 45ccffeb: methods are classified and gated -------------------------


@pytest.mark.parametrize(
    "method,expected",
    [
        ("DOM.getDocument", "read"),
        ("Page.navigate", "mutating"),
        ("Runtime.evaluate", "mutating"),
        ("Network.clearBrowserCookies", "destructive"),
        ("Browser.close", "destructive"),
        ("Totally.madeUp", "unknown"),
    ],
)
def test_methods_are_classified(method, expected):
    assert classify_method(method) == expected


def test_an_unknown_method_is_refused(websocket):
    websocket["ws"] = _FakeWS([_reply()])

    with pytest.raises(CdpPolicyError, match="allowlist"):
        cdp_call(GOOD_URL, "Totally.madeUp", {})

    assert "url" not in websocket


def test_a_destructive_method_needs_explicit_permission(websocket):
    websocket["ws"] = _FakeWS([_reply()])

    with pytest.raises(CdpPolicyError, match="destructive"):
        cdp_call(GOOD_URL, "Network.clearBrowserCookies", {})


def test_a_destructive_method_needs_a_reason(websocket):
    websocket["ws"] = _FakeWS([_reply()])

    with pytest.raises(CdpPolicyError, match="reason"):
        cdp_call(GOOD_URL, "Browser.close", {}, allow_destructive=True)


def test_a_permitted_destructive_call_proceeds(websocket):
    websocket["ws"] = _FakeWS([_reply()])

    result = cdp_call(
        GOOD_URL, "Browser.close", {},
        allow_destructive=True, reason="operator asked to close the browser",
    )

    assert result["receipt"]["method_class"] == "destructive"
    assert result["receipt"]["reason"].startswith("operator asked")


def test_every_call_produces_a_receipt(websocket):
    websocket["ws"] = _FakeWS([_reply()])

    receipt = cdp_call(GOOD_URL, "DOM.getDocument", {})["receipt"]

    assert receipt["method"] == "DOM.getDocument"
    assert receipt["method_class"] == "read"
    assert receipt["target"] == GOOD_URL
    assert receipt["ok"] is True
    assert receipt["elapsed_s"] >= 0


def test_the_live_caller_methods_are_all_allowlisted():
    """The web interlocutor's CDP vocabulary must keep working."""
    for method in (
        "Page.bringToFront", "Input.insertText",
        "Input.dispatchKeyEvent", "Runtime.evaluate",
    ):
        assert classify_method(method) != "unknown"


# --- fc411124: events are retained, not discarded -----------------------


def test_events_seen_while_waiting_are_returned(websocket):
    websocket["ws"] = _FakeWS([
        json.dumps({"method": "Page.frameNavigated", "params": {"n": 1}}),
        json.dumps({"method": "Network.responseReceived", "params": {"n": 2}}),
        _reply(),
    ])

    result = cdp_call(GOOD_URL, "DOM.getDocument", {})

    assert len(result["events"]) == 2
    assert result["events"][0]["method"] == "Page.frameNavigated"
    assert result["receipt"]["events_observed"] == 2


def test_the_event_buffer_is_bounded(websocket):
    noise = [json.dumps({"method": "X", "i": i}) for i in range(MAX_RETAINED_EVENTS + 20)]
    websocket["ws"] = _FakeWS([*noise, _reply()])

    events = cdp_call(GOOD_URL, "DOM.getDocument", {})["events"]

    assert len(events) == MAX_RETAINED_EVENTS + 1
    assert events[-1]["_truncated"] is True


def test_the_reply_is_still_returned_unchanged(websocket):
    """The existing caller reads result["result"]; that contract holds."""
    websocket["ws"] = _FakeWS([_reply()])

    result = cdp_call(GOOD_URL, "DOM.getDocument", {})

    assert result["result"]["id"] == 1
    assert result["result"]["result"] == {"ok": True}


# --- 7e9bb8a8: the timeout contract is normalized -----------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (float("nan"), DEFAULT_TIMEOUT_S),
        (float("inf"), DEFAULT_TIMEOUT_S),
        (0, DEFAULT_TIMEOUT_S),
        (-3, DEFAULT_TIMEOUT_S),
        ("soon", DEFAULT_TIMEOUT_S),
        (None, DEFAULT_TIMEOUT_S),
        (9999, MAX_TIMEOUT_S),
        (3.0, 3.0),
    ],
)
def test_timeouts_are_validated(value, expected):
    assert validated_timeout(value) == expected


def test_a_library_timeout_is_normalized_to_timeout_error(websocket):
    websocket["ws"] = _FakeWS([], recv_error=_LibTimeout("read timed out"))

    with pytest.raises(TimeoutError, match="DOM.getDocument"):
        cdp_call(GOOD_URL, "DOM.getDocument", {}, timeout=0.5)


def test_a_non_timeout_error_is_not_disguised(websocket):
    websocket["ws"] = _FakeWS([], recv_error=ConnectionResetError("peer reset"))

    with pytest.raises(ConnectionResetError):
        cdp_call(GOOD_URL, "DOM.getDocument", {}, timeout=0.5)


def test_the_socket_is_closed_on_every_path(websocket):
    ws = _FakeWS([], recv_error=_LibTimeout("nope"))
    websocket["ws"] = ws

    with pytest.raises(TimeoutError):
        cdp_call(GOOD_URL, "DOM.getDocument", {}, timeout=0.5)

    assert ws.closed is True


# --- e39c11e5: frames are size-bounded ----------------------------------


def test_an_oversized_frame_is_refused(websocket):
    huge = json.dumps({"id": 1, "result": {"blob": "x" * (MAX_FRAME_BYTES + 100)}})
    websocket["ws"] = _FakeWS([huge])

    with pytest.raises(RuntimeError, match="exceeds"):
        cdp_call(GOOD_URL, "DOM.getDocument", {})


def test_unparseable_json_is_a_typed_error(websocket):
    websocket["ws"] = _FakeWS(["{not json"])

    with pytest.raises(RuntimeError, match="unparseable"):
        cdp_call(GOOD_URL, "DOM.getDocument", {})


def test_a_cdp_level_error_is_raised(websocket):
    websocket["ws"] = _FakeWS([json.dumps({"id": 1, "error": {"message": "boom"}})])

    with pytest.raises(RuntimeError, match="boom"):
        cdp_call(GOOD_URL, "DOM.getDocument", {})


# --- f47118c0: an async caller need not block ---------------------------


def test_an_async_path_exists_and_works(websocket):
    websocket["ws"] = _FakeWS([_reply()])

    result = asyncio.run(cdp_call_async(GOOD_URL, "DOM.getDocument", {}))

    assert result["result"]["result"] == {"ok": True}


def test_the_async_path_refuses_before_spending_a_thread(websocket):
    websocket["ws"] = _FakeWS([_reply()])

    with pytest.raises(CdpPolicyError):
        asyncio.run(cdp_call_async(GOOD_URL, "Totally.madeUp", {}))

    assert "url" not in websocket


def test_the_async_path_validates_the_target(websocket):
    websocket["ws"] = _FakeWS([_reply()])

    with pytest.raises(CdpPolicyError):
        asyncio.run(cdp_call_async("ws://evil.example.com/x", "DOM.getDocument", {}))


def test_the_sync_docstring_warns_async_callers():
    assert "cdp_call_async" in (module.cdp_call.__doc__ or "")
