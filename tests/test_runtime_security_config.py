from types import SimpleNamespace

import pytest
from fastapi import HTTPException


class _Request:
    def __init__(self, path="/api/chat", host="203.0.113.10", headers=None, method="GET"):
        self.url = SimpleNamespace(path=path)
        self.client = SimpleNamespace(host=host)
        self.headers = headers or {}
        self.method = method


def test_runtime_security_fails_closed_when_token_disappears(monkeypatch):
    from interface import auth

    monkeypatch.setattr(auth.config.security, "internal_only_mode", False, raising=False)
    monkeypatch.setattr(auth.config, "api_token", None, raising=False)

    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(_Request())

    assert exc.value.status_code == 503


def test_runtime_security_keeps_health_probe_available_without_token(monkeypatch):
    from interface import auth

    monkeypatch.setattr(auth.config.security, "internal_only_mode", False, raising=False)
    monkeypatch.setattr(auth.config, "api_token", None, raising=False)

    auth.validate_runtime_security_request(_Request(path="/api/health"))


def test_runtime_security_rechecks_internal_only_per_request(monkeypatch):
    from interface import auth

    monkeypatch.setattr(auth.config.security, "internal_only_mode", True, raising=False)
    monkeypatch.setattr(auth.config, "api_token", "secret", raising=False)

    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(_Request(host="198.51.100.2"))

    assert exc.value.status_code == 403


def test_runtime_security_accepts_valid_bearer_token(monkeypatch):
    from interface import auth

    monkeypatch.setattr(auth.config.security, "internal_only_mode", False, raising=False)
    monkeypatch.setattr(auth.config, "api_token", "secret", raising=False)

    auth.validate_runtime_security_request(
        _Request(headers={"Authorization": "Bearer secret"})
    )


def test_runtime_security_rejects_invalid_external_token(monkeypatch):
    from interface import auth

    monkeypatch.setattr(auth.config.security, "internal_only_mode", False, raising=False)
    monkeypatch.setattr(auth.config, "api_token", "secret", raising=False)

    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(
            _Request(headers={"Authorization": "Bearer wrong"})
        )

    assert exc.value.status_code == 401


def test_verify_token_allows_trusted_local_internal_only_without_header(monkeypatch):
    from interface import auth

    monkeypatch.setattr(auth.config.security, "internal_only_mode", True, raising=False)
    monkeypatch.setattr(auth.config, "api_token", "secret", raising=False)

    auth._verify_token(_Request(host="127.0.0.1"), x_api_token=None)


def test_verify_token_allows_trusted_local_desktop_ui_without_header(monkeypatch):
    from interface import auth

    monkeypatch.setattr(auth.config.security, "internal_only_mode", False, raising=False)
    monkeypatch.setattr(auth.config, "api_token", "secret", raising=False)

    auth._verify_token(
        _Request(
            host="127.0.0.1",
            headers={
                "Origin": "http://127.0.0.1:8000",
                "X-Aura-Surface": "desktop-ui",
            },
            method="POST",
        ),
        x_api_token=None,
    )


def test_verify_token_rejects_bare_localhost_for_protected_skill_route(monkeypatch):
    from interface import auth

    monkeypatch.setattr(auth.config.security, "internal_only_mode", False, raising=False)
    monkeypatch.setattr(auth.config, "api_token", "secret", raising=False)

    with pytest.raises(HTTPException) as exc:
        auth._verify_token(
            _Request(path="/api/skill/execute", host="127.0.0.1", method="POST"),
            x_api_token=None,
        )

    assert exc.value.status_code == 401


def test_runtime_security_rejects_cross_site_localhost_csrf(monkeypatch):
    from interface import auth

    monkeypatch.setattr(auth.config.security, "internal_only_mode", False, raising=False)
    monkeypatch.setattr(auth.config, "api_token", "secret", raising=False)

    with pytest.raises(HTTPException) as exc:
        auth.validate_runtime_security_request(
            _Request(
                path="/api/skill/execute",
                host="127.0.0.1",
                headers={
                    "Origin": "http://evil.example.com",
                    "Sec-Fetch-Site": "cross-site",
                },
                method="POST",
            )
        )

    assert exc.value.status_code == 403


def test_verify_token_rejects_external_internal_only_without_header(monkeypatch):
    from interface import auth

    monkeypatch.setattr(auth.config.security, "internal_only_mode", True, raising=False)
    monkeypatch.setattr(auth.config, "api_token", "secret", raising=False)

    with pytest.raises(HTTPException) as exc:
        auth._verify_token(_Request(host="198.51.100.9"), x_api_token=None)

    assert exc.value.status_code == 401
