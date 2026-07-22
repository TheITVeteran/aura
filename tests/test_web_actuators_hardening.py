"""CP126 hardening contracts for core/actuators/web_actuators.py.

Web fetch is an SSRF surface. Tests cover exact-HTTPS + userinfo rejection, the
configurable allowlist, and — the core defense — rejecting an allowlisted host
that resolves to a non-public address (DNS rebinding). Also: bounded search
inputs, a deadline on the async bridge, dict-shape guarding, and forwarding the
authority/capability context to the browser skill. socket.getaddrinfo is mocked
so no real DNS or network happens.
"""
from __future__ import annotations

import asyncio
import socket

import pytest

from core.actuators.web_actuators import (
    WebFetchActuator,
    WebSearchActuator,
    _clamp_int,
    _ip_is_public,
    run_async_in_sync,
    validate_fetch_url,
)


def _mock_resolve(monkeypatch, ip: str):
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, **kw: [(2, 1, 6, "", (ip, port or 443))])


# ── 92341d1b: resolved-IP SSRF policy ──────────────────────────────────────


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1", "0.0.0.0"])
def test_non_public_ip_is_rejected(ip):
    assert _ip_is_public(ip) is False


def test_public_ip_is_accepted():
    assert _ip_is_public("93.184.216.34") is True


def test_allowlisted_host_resolving_to_private_ip_is_refused(monkeypatch):
    _mock_resolve(monkeypatch, "10.0.0.1")  # rebinding to internal
    url, err = validate_fetch_url("https://github.com/x")
    assert url is None and "non-public" in err


def test_allowlisted_host_resolving_to_public_ip_is_allowed(monkeypatch):
    _mock_resolve(monkeypatch, "140.82.112.3")
    url, err = validate_fetch_url("https://github.com/org/repo")
    assert url is not None, err


# ── 7b92dcc2: scheme + userinfo rules ──────────────────────────────────────


def test_http_scheme_refused(monkeypatch):
    _mock_resolve(monkeypatch, "140.82.112.3")
    url, err = validate_fetch_url("http://github.com/x")
    assert url is None and "scheme" in err


def test_userinfo_refused(monkeypatch):
    _mock_resolve(monkeypatch, "140.82.112.3")
    url, err = validate_fetch_url("https://evil@github.com/x")
    assert url is None and "credentials" in err


def test_non_allowlisted_host_refused(monkeypatch):
    _mock_resolve(monkeypatch, "1.2.3.4")
    url, err = validate_fetch_url("https://evil.example.com/x")
    assert url is None and "allowlist" in err


# ── c5d4e4b5: allowlist is configurable ────────────────────────────────────


def test_allowlist_is_env_extensible(monkeypatch):
    _mock_resolve(monkeypatch, "1.2.3.4")
    monkeypatch.setenv("AURA_WEB_FETCH_ALLOWLIST", "example.com")
    url, err = validate_fetch_url("https://example.com/x")
    assert url is not None, err


# ── f14d3a4a: bounded inputs ───────────────────────────────────────────────


@pytest.mark.parametrize("value,expected", [(-3, 1), (9999, 25), ("x", 5), (10, 10)])
def test_num_results_is_clamped(value, expected):
    assert _clamp_int(value, 5, 1, 25) == expected


# ── 8c42625: the async bridge honors a deadline ────────────────────────────


def test_bridge_deadline_times_out():
    async def _hang():
        await asyncio.sleep(5)

    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        run_async_in_sync(_hang(), deadline_s=0.2)


# ── authority ──────────────────────────────────────────────────────────────


def test_web_search_requires_authorization():
    assert WebSearchActuator().execute({"query": "hi"}).success is False


def test_web_fetch_requires_authorization():
    # Auth is checked before any DNS/validation, so this does no network.
    assert WebFetchActuator().execute({"url": "https://github.com/x"}).success is False


# ── fbb1c179: malformed (non-dict) results are handled ─────────────────────


def test_search_non_dict_result_is_failure(monkeypatch):
    class _Pipe:
        async def search(self, *a, **k):
            return "not a dict"

    monkeypatch.setattr(WebSearchActuator, "_get_pipeline", lambda self: _Pipe())
    res = WebSearchActuator().execute({"query": "hi", "_aura_authorized": True})
    assert res.success is False and "malformed" in res.message


# ── 5ea6fa36: authority/capability context reaches the browser skill ───────


def test_fetch_forwards_capability_context(monkeypatch):
    _mock_resolve(monkeypatch, "140.82.112.3")
    captured = {}

    class _FakeSkill:
        async def execute(self, params, context):
            captured.update(context)
            return {"ok": True, "message": "fetched"}

    import core.skills.sovereign_browser as sb
    monkeypatch.setattr(sb, "SovereignBrowserSkill", _FakeSkill)

    res = WebFetchActuator().execute({
        "url": "https://github.com/org/repo",
        "_aura_authorized": True,
        "_capability_token_id": "cap-123",
    })
    assert res.success is True
    assert captured.get("_capability_token_id") == "cap-123"
    assert captured.get("_aura_authorized") is True
