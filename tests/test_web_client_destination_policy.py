"""An allowlist that permits everything when it is empty is not an allowlist.

The populated path also compared raw ``netloc`` — userinfo and port
included — so a permitted host on a non-default port was refused and a
case difference was refused, while the parse-level spoofing surface stayed
open.
"""
from __future__ import annotations

import pytest

from core.capability_engine import WebClient


def test_an_empty_allowlist_denies_everything():
    """The defect: no destinations configured meant unrestricted egress."""
    client = WebClient()
    assert client._is_allowed("https://evil.com/x") is False
    assert client._is_allowed("https://anything.example/") is False


def test_the_wildcard_must_be_written_down():
    assert WebClient(["*"])._is_allowed("https://anything.example/") is True


def test_exact_and_subdomain_hosts_are_permitted():
    client = WebClient(["good.com"])
    assert client._is_allowed("https://good.com/x") is True
    assert client._is_allowed("https://api.good.com/x") is True


def test_an_unlisted_host_is_denied():
    assert WebClient(["good.com"])._is_allowed("https://evil.com/x") is False


def test_a_suffix_lookalike_is_denied():
    """notgood.com must not satisfy an allowlist entry of good.com."""
    assert WebClient(["good.com"])._is_allowed("https://notgood.com/x") is False


def test_userinfo_cannot_impersonate_an_allowed_host():
    client = WebClient(["good.com"])
    assert client._is_allowed("https://good.com@evil.com/x") is False
    assert client._is_allowed("https://good.com:pw@evil.com/x") is False


@pytest.mark.parametrize(
    "url",
    [
        "https://good.com:8443/x",
        "http://good.com:8080/x",
    ],
)
def test_an_explicit_port_does_not_deny_an_allowed_host(url):
    """netloc includes the port, so this used to refuse a permitted host."""
    assert WebClient(["good.com"])._is_allowed(url) is True


def test_host_matching_is_case_insensitive():
    assert WebClient(["good.com"])._is_allowed("HTTPS://GOOD.COM/x") is True
    assert WebClient(["GOOD.COM"])._is_allowed("https://good.com/x") is True


def test_a_trailing_root_dot_is_the_same_host():
    assert WebClient(["good.com"])._is_allowed("https://good.com./x") is True


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://good.com/x",
        "gopher://good.com/x",
        "data:text/plain;base64,QQ==",
    ],
)
def test_non_web_schemes_are_denied(url):
    """An allowlist naming web hosts never authorized these."""
    assert WebClient(["good.com", "*.good.com"])._is_allowed(url) is False


def test_a_malformed_url_is_denied():
    client = WebClient(["good.com"])
    assert client._is_allowed("") is False
    assert client._is_allowed("https://") is False
    assert client._is_allowed("not a url at all") is False


def test_blank_allowlist_entries_do_not_match_anything():
    """An entry of '' must not become a suffix that matches every host."""
    client = WebClient(["", "   "])
    assert client._is_allowed("https://evil.com/x") is False
