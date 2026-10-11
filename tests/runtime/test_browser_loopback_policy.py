"""A process owner can admit one local app without opening other destinations."""

from __future__ import annotations

import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from core.runtime import url_policy as policy


@pytest.fixture
def destinations(monkeypatch):
    monkeypatch.delenv(policy.LOOPBACK_OPT_IN_FLAG, raising=False)
    monkeypatch.delenv(policy.LOOPBACK_ORIGINS_FLAG, raising=False)
    monkeypatch.delenv("AURA_WEB_FETCH_ALLOW_HTTP", raising=False)
    answers = {
        "127.0.0.1": ["127.0.0.1"],
        "::1": ["::1"],
        "localhost": ["127.0.0.1", "::1"],
        "mixed.local": ["127.0.0.1", "10.0.0.5"],
        "mixed-public.local": ["127.0.0.1", "93.184.216.34"],
        "example.com": ["93.184.216.34"],
        "10.0.0.5": ["10.0.0.5"],
        "169.254.169.254": ["169.254.169.254"],
        "100.100.100.200": ["100.100.100.200"],
        "fd00:ec2::254": ["fd00:ec2::254"],
    }

    def resolve(host, port, *, proto):
        assert proto == socket.IPPROTO_TCP
        if host not in answers:
            raise socket.gaierror("unknown test host")
        return [(socket.AF_INET, socket.SOCK_STREAM, proto, "", (ip, port)) for ip in answers[host]]

    monkeypatch.setattr(policy.socket, "getaddrinfo", resolve)
    admitted = Mock()
    monkeypatch.setattr(policy, "record_degradation", admitted)
    return answers, admitted


def scoped(monkeypatch, origins="http://127.0.0.1:56740"):
    monkeypatch.setenv(policy.LOOPBACK_OPT_IN_FLAG, "1")
    monkeypatch.setenv(policy.LOOPBACK_ORIGINS_FLAG, origins)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:56740/app",
    "https://127.0.0.1/app",
    "https://localhost/app",
    "https://10.0.0.5/app",
    "https://169.254.169.254/latest",
    "http://example.com/app",
    "https://example.com:56740/app",
])
def test_defaults_refuse_local_http_and_unapproved_ports(destinations, url):
    assert policy.validate_browser_url(url)[0] is None
    destinations[1].assert_not_called()


def test_public_https_stays_available_with_scoped_loopback(monkeypatch, destinations):
    scoped(monkeypatch)
    assert policy.validate_browser_url("https://example.com/app") == ("https://example.com/app", "")
    assert policy.validate_fetch_url_static("https://example.com/app")[0] is None
    assert policy.validate_fetch_url_static("http://127.0.0.1:56740/app")[0] is None
    destinations[1].assert_not_called()


@pytest.mark.parametrize("path", ["", "/", "/app?scene=2#start", "/nested/file.html"])
def test_scoped_origin_admits_paths_and_records_the_opt_in(monkeypatch, destinations, path):
    scoped(monkeypatch)
    url = "http://127.0.0.1:56740" + path
    assert policy.validate_browser_url(url) == (url, "")
    destinations[1].assert_called_once()
    assert destinations[1].call_args.kwargs["enforce_failure_policy"] is False


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:56741/app",
    "http://127.0.0.1/app",
    "https://127.0.0.1:56740/app",
    "http://localhost:56740/app",
    "http://[::1]:56740/app",
])
def test_scope_distinguishes_scheme_host_and_port(monkeypatch, destinations, url):
    scoped(monkeypatch)
    validated, reason = policy.validate_browser_url(url)
    assert validated is None
    assert policy.LOOPBACK_ORIGINS_FLAG in reason
    destinations[1].assert_not_called()


def test_multiple_origins_and_default_ports_use_the_same_origin_rules(monkeypatch, destinations):
    scoped(monkeypatch, "http://LOCALHOST:80/, https://[::1]:443, http://127.0.0.1:56740")
    for url in ("http://localhost/app", "https://[::1]/app", "http://127.0.0.1:56740/app"):
        assert policy.validate_browser_url(url) == (url, "")
    assert policy.validate_browser_url("https://[::1]:444/app")[0] is None


def test_an_origin_list_alone_cannot_enable_loopback(monkeypatch, destinations):
    monkeypatch.setenv(policy.LOOPBACK_ORIGINS_FLAG, "http://127.0.0.1:56740")
    assert policy.validate_browser_url("http://127.0.0.1:56740/app")[0] is None


@pytest.mark.parametrize("origins", [
    "", " ", "http://127.0.0.1:56740,", "http://127.0.0.1:56740/app",
    "http://127.0.0.1:56740?scope=all", "http://127.0.0.1:56740#scope",
    "http://user@127.0.0.1:56740", "ftp://127.0.0.1:56740",
    "http://[::1", "http://127.0.0.1:99999",
])
def test_present_invalid_scope_fails_closed(monkeypatch, destinations, origins):
    scoped(monkeypatch, origins)
    validated, reason = policy.validate_browser_url("http://127.0.0.1:56740/app")
    assert validated is None
    assert policy.LOOPBACK_ORIGINS_FLAG in reason
    destinations[1].assert_not_called()


@pytest.mark.parametrize("scoped_config", [False, True])
@pytest.mark.parametrize("url", [
    "http://user:secret@127.0.0.1:56740/app",
    "http://user@127.0.0.1:56740/app",
    "http://@127.0.0.1:56740/app",
])
def test_credentials_never_bypass_policy(monkeypatch, destinations, scoped_config, url):
    monkeypatch.setenv(policy.LOOPBACK_OPT_IN_FLAG, "1")
    if scoped_config:
        monkeypatch.setenv(policy.LOOPBACK_ORIGINS_FLAG, "http://127.0.0.1:56740")
    validated, reason = policy.validate_browser_url(url)
    assert validated is None
    assert "credentials" in reason
    destinations[1].assert_not_called()


@pytest.mark.parametrize("url", [
    None, 17, "", "http://", "http://[::1", "http://127.0.0.1:bad/app",
    "http://127.0.0.1:65536/app", "http://127.0.0.1:0/app", "http://127.0.0.1:/app",
    "\nhttp://127.0.0.1:56740/app", "http://127.0.0.1:56740/app\r",
    "http://127.0.0.1:56740/\napp", "http://127.0.0.1:56740/\tapp",
    "http://127.0.0.1:56740/a\u00a0b",
    "http://127.0.0.1:56740/\x00app", "http://127.0.0.1:56740\\@example.com/app",
    "http://127.0.0.1%23:56740/app", "file://127.0.0.1:56740/app",
])
def test_malformed_urls_are_refused_under_opt_in(monkeypatch, destinations, url):
    scoped(monkeypatch)
    validated, reason = policy.validate_browser_url(url)
    assert validated is None
    assert reason
    destinations[1].assert_not_called()


@pytest.mark.parametrize("host", ["mixed.local", "mixed-public.local"])
def test_one_non_loopback_dns_answer_refuses_the_entire_origin(monkeypatch, destinations, host):
    scoped(monkeypatch, f"http://{host}:56740")
    assert policy.validate_browser_url(f"http://{host}:56740/app")[0] is None
    destinations[1].assert_not_called()


@pytest.mark.parametrize("host", ["10.0.0.5", "169.254.169.254", "100.100.100.200", "[fd00:ec2::254]"])
def test_scoped_configuration_cannot_admit_private_or_metadata_addresses(monkeypatch, destinations, host):
    scoped(monkeypatch, f"https://{host}:443")
    assert policy.validate_browser_url(f"https://{host}/app")[0] is None
    destinations[1].assert_not_called()


def test_legacy_opt_in_still_admits_only_loopback(monkeypatch, destinations):
    monkeypatch.setenv(policy.LOOPBACK_OPT_IN_FLAG, "1")
    for url in ("http://127.0.0.1:56740/app", "https://localhost:56741/app", "http://[::1]:56742/app"):
        assert policy.validate_browser_url(url) == (url, "")
    for url in ("https://10.0.0.5/app", "https://169.254.169.254/latest", "https://mixed-public.local/app"):
        assert policy.validate_browser_url(url)[0] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("landed,allowed", [
    ("http://127.0.0.1:56740/next", True),
    ("http://127.0.0.1:56741/next", False),
    ("http://localhost:56740/next", False),
    ("http://user@127.0.0.1:56740/next", False),
    ("https://169.254.169.254/latest", False),
])
async def test_browser_arrival_revalidates_redirects_with_the_scoped_policy(monkeypatch, destinations, landed, allowed):
    from core.capabilities.browser_authority import BrowserAction, authorize_browser_action
    from core.capabilities.phantom_browser import PhantomBrowser

    scoped(monkeypatch)
    requested = "http://127.0.0.1:56740/app"
    assert authorize_browser_action(BrowserAction.NAVIGATE, principal="offline_test_harness", url=requested).allowed
    browser = object.__new__(PhantomBrowser)
    browser.page = SimpleNamespace(url=landed)
    browser.is_blocked = AsyncMock(return_value=False)
    arrival = await browser._verify_arrival(requested, SimpleNamespace(status=200), principal="offline_test_harness")
    assert arrival["redirected"] is True
    assert arrival["ok"] is allowed
    if not allowed:
        assert arrival["reason"].startswith("redirect refused:")


@pytest.mark.asyncio
async def test_redirect_dns_is_checked_again_after_initial_admission(monkeypatch, destinations):
    from core.capabilities.browser_authority import BrowserAction, authorize_browser_action
    from core.capabilities.phantom_browser import PhantomBrowser

    scoped(monkeypatch, "http://localhost:56740")
    requested = "http://localhost:56740/app"
    assert authorize_browser_action(BrowserAction.NAVIGATE, principal="offline_test_harness", url=requested).allowed
    destinations[0]["localhost"].append("10.0.0.5")
    browser = object.__new__(PhantomBrowser)
    browser.page = SimpleNamespace(url="http://localhost:56740/next")
    browser.is_blocked = AsyncMock(return_value=False)
    arrival = await browser._verify_arrival(requested, SimpleNamespace(status=200), principal="offline_test_harness")
    assert arrival["ok"] is False
    assert arrival["reason"].startswith("redirect refused:")
    browser.is_blocked.assert_not_called()
