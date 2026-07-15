from __future__ import annotations

from collections.abc import Iterable

import pytest
from fastapi import HTTPException, Request

from interface import auth


def _request(
    path: str = "/api/memory/export",
    *,
    peer: str = "127.0.0.1",
    host: str | None = "127.0.0.1:8000",
    origin: str | None = None,
    method: str = "GET",
    scheme: str = "http",
    extra_headers: Iterable[tuple[str, str]] = (),
) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if host is not None:
        headers.append((b"host", host.encode("latin-1")))
    if origin is not None:
        headers.append((b"origin", origin.encode("latin-1")))
    headers.extend(
        (name.lower().encode("ascii"), value.encode("latin-1"))
        for name, value in extra_headers
    )
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": scheme,
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": headers,
            "client": (peer, 49152),
            "server": ("127.0.0.1", 8000),
        }
    )


@pytest.fixture(autouse=True)
def _runtime_security(monkeypatch):
    monkeypatch.setattr(auth.config.security, "internal_only_mode", False, raising=False)
    monkeypatch.setattr(auth.config, "api_token", "dev-secret", raising=False)


@pytest.mark.parametrize(
    ("path", "method"),
    (
        ("/api/memory/export", "GET"),
        ("/api/settings", "PATCH"),
    ),
)
def test_dns_rebinding_published_exploit_is_denied(path, method):
    request = _request(
        path,
        host="attacker.test:8000",
        origin="http://attacker.test:8000",
        method=method,
        extra_headers=(("Sec-Fetch-Site", "same-origin"),),
    )

    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(request)

    assert exc.value.status_code == 403
    assert "Host authority" in str(exc.value.detail)


@pytest.mark.parametrize(
    "path",
    ("/", "/api/health", "/pair", "/api/devices/pair/complete"),
)
def test_dns_rebinding_cannot_hide_behind_public_paths(path):
    request = _request(
        path,
        host="attacker.test:8000",
        origin="http://attacker.test:8000",
        method="POST" if path.endswith("complete") else "GET",
    )

    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(request)

    assert exc.value.status_code == 403


def test_dns_rebinding_without_origin_is_denied_before_local_bypass():
    request = _request(host="attacker.test:8000")

    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(request)

    assert exc.value.status_code == 403


def test_lan_address_rebinding_does_not_gain_tokenless_owner_access():
    rebound = _request(
        peer="192.168.1.44",
        host="attacker.test:8000",
        origin="http://attacker.test:8000",
        extra_headers=(("Sec-Fetch-Site", "same-origin"),),
    )
    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(rebound)
    assert exc.value.status_code == 403

    no_origin = _request(peer="192.168.1.44", host="attacker.test:8000")
    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(no_origin)
    assert exc.value.status_code == 401


def test_forged_desktop_markers_do_not_bypass_host_authority():
    request = _request(
        "/api/skill/execute",
        host="attacker.test:8000",
        origin="http://attacker.test:8000",
        method="POST",
        extra_headers=(
            ("Sec-Fetch-Site", "same-origin"),
            ("X-Aura-Surface", "desktop-ui"),
            ("X-Aura-Desktop-Request", "same-origin"),
        ),
    )

    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(request)

    assert exc.value.status_code == 403


def test_internal_only_mode_still_rejects_rebound_host(monkeypatch):
    monkeypatch.setattr(auth.config.security, "internal_only_mode", True, raising=False)
    request = _request(
        host="attacker.test:8000",
        origin="http://attacker.test:8000",
    )

    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(request)

    assert exc.value.status_code == 403


@pytest.mark.parametrize(
    ("peer", "host", "origin"),
    (
        ("127.0.0.1", "127.0.0.1:8000", "http://127.0.0.1:8000"),
        ("127.0.0.1", "LOCALHOST:8000", "http://localhost:8000"),
        ("::1", "[::1]:8000", "http://[::1]:8000"),
    ),
)
def test_exact_loopback_authorities_remain_available(peer, host, origin):
    auth.validate_runtime_security_request(
        _request(peer=peer, host=host, origin=origin)
    )
    auth.validate_runtime_security_request(
        _request(
            "/api/settings",
            peer=peer,
            host=host,
            origin=origin,
            method="PATCH",
        )
    )


def test_configured_local_dev_origin_remains_available():
    auth.validate_runtime_security_request(
        _request(
            host="127.0.0.1:8000",
            origin="http://127.0.0.1:5173",
            extra_headers=(("Sec-Fetch-Site", "same-site"),),
        )
    )


@pytest.mark.parametrize(
    "host",
    (
        "localhost.attacker.test:8000",
        "127.0.0.1.attacker.test:8000",
        "127.0.0.1.nip.io:8000",
        "127.0.0.1.:8000",
        "2130706433:8000",
        "0x7f000001:8000",
        "017700000001:8000",
        "0.0.0.0:8000",
        "[::ffff:127.0.0.1]:8000",
        "user@127.0.0.1:8000",
        "127.0.0.1:bad",
        "127.0.0.1:70000",
        "127.0.0.1:8000,attacker.test:8000",
    ),
)
def test_loopback_lookalikes_and_ambiguous_authorities_are_denied(host):
    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(_request(host=host))

    assert exc.value.status_code == 403


def test_missing_and_duplicate_host_headers_are_denied():
    with pytest.raises(HTTPException):
        auth.validate_runtime_security_request(_request(host=None))

    duplicate = _request(
        host="127.0.0.1:8000",
        extra_headers=(("Host", "attacker.test:8000"),),
    )
    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(duplicate)
    assert exc.value.status_code == 403


@pytest.mark.parametrize(
    "origin",
    (
        "null",
        "https://127.0.0.1:8000",
        "http://user@127.0.0.1:8000",
        "http://127.0.0.1:8000/path",
        "http://127.0.0.1:8000?query=1",
        "http://127.0.0.1:8000#fragment",
    ),
)
def test_malformed_or_cross_scheme_origins_are_denied(origin):
    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(
            _request(host="127.0.0.1:8000", origin=origin)
        )

    assert exc.value.status_code == 403


def test_valid_master_token_can_cross_proxy_authority_boundary():
    auth.validate_runtime_security_request(
        _request(
            host="api.internal.example:8443",
            origin="https://admin.internal.example",
            extra_headers=(("Authorization", "Bearer dev-secret"),),
        )
    )


def test_rebound_request_cannot_be_classified_as_owner():
    rebound = _request(
        host="attacker.test:8000",
        origin="http://attacker.test:8000",
    )
    assert auth.request_access_profile(rebound)["surface"] == "unknown"

    authenticated = _request(
        host="attacker.test:8000",
        origin="http://attacker.test:8000",
        extra_headers=(("Authorization", "Bearer dev-secret"),),
    )
    assert auth.request_access_profile(authenticated)["surface"] == "owner"


def test_internal_dependency_rechecks_host_when_token_is_unconfigured(monkeypatch):
    monkeypatch.setattr(auth.config.security, "internal_only_mode", True, raising=False)
    monkeypatch.setattr(auth.config, "api_token", None, raising=False)
    rebound = _request(
        host="attacker.test:8000",
        origin="http://attacker.test:8000",
    )

    with pytest.raises(HTTPException) as exc:
        auth._verify_token(rebound, x_api_token=None)
    assert exc.value.status_code == 503

    with pytest.raises(HTTPException) as exc:
        auth._require_internal(rebound)
    assert exc.value.status_code == 403


def test_websocket_local_trust_uses_same_rebinding_boundary():
    rebound = _request(
        "/ws",
        host="attacker.test:8000",
        origin="http://attacker.test:8000",
    )
    assert auth.request_has_allowed_local_browser_origin(rebound) is False

    trusted = _request(
        "/ws",
        host="127.0.0.1:8000",
        origin="http://127.0.0.1:8000",
    )
    assert auth.request_has_allowed_local_browser_origin(trusted) is True

    trusted_wss = _request(
        "/ws",
        host="[::1]:8443",
        origin="https://[::1]:8443",
        peer="::1",
        scheme="wss",
    )
    assert auth.request_has_allowed_local_browser_origin(trusted_wss) is True
