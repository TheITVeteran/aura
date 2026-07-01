from __future__ import annotations

from types import SimpleNamespace

from interface import auth


class _BadCookieJar:
    def get(self, _name: str):
        self.last_requested = _name
        raise ValueError("malformed cookie jar")


def test_owner_session_restore_handles_malformed_cookie_jar_without_authorizing():
    request = SimpleNamespace(cookies=_BadCookieJar(), headers={})

    assert auth._restore_owner_session_from_request(request) is False


def test_owner_session_restore_handles_malformed_cookie_header_without_authorizing():
    request = SimpleNamespace(cookies={}, headers={"cookie": "aura_owner_session=\x00bad"})

    assert auth._restore_owner_session_from_request(request) is False


def test_cheat_code_activation_reports_recoverable_runtime_failure(monkeypatch):
    import core.security.cheat_codes as cheat_codes

    def _raise_runtime_error(*_args, **_kwargs):
        _raise_runtime_error.called = True
        raise RuntimeError("trust engine unavailable")

    _raise_runtime_error.called = False
    monkeypatch.setattr(cheat_codes, "activate_cheat_code", _raise_runtime_error)

    result = auth._activate_cheat_code_for_request("anything", silent=True, source="test")

    assert result == {
        "ok": False,
        "status": "error",
        "message": "Cheat code activation failed.",
    }
    assert _raise_runtime_error.called is True


def _rate_limit_request(peer: str, headers: dict[str, str] | None = None):
    return SimpleNamespace(
        client=SimpleNamespace(host=peer),
        headers=headers or {},
        url=SimpleNamespace(path="/api/chat"),
    )


def test_rate_limit_bypass_cannot_be_spoofed_by_forwarded_header(monkeypatch):
    """A remote peer forging X-Forwarded-For: 127.0.0.1 must not bypass limits."""
    checked: list[str] = []

    def _record(ip: str) -> bool:
        checked.append(ip)
        return True

    monkeypatch.setattr(auth._rate_limiter, "check", _record)

    request = _rate_limit_request("203.0.113.9", {"X-Forwarded-For": "127.0.0.1"})
    auth._check_rate_limit(request)

    # The limiter is keyed on the real socket peer, not the forged header.
    assert checked == ["203.0.113.9"]


def test_rate_limit_direct_local_traffic_is_bypassed(monkeypatch):
    def _fail(_ip: str) -> bool:  # pragma: no cover - must not be called
        raise AssertionError("local traffic should bypass the limiter")

    monkeypatch.setattr(auth._rate_limiter, "check", _fail)

    auth._check_rate_limit(_rate_limit_request("127.0.0.1"))


def test_rate_limit_local_proxy_buckets_by_forwarded_client(monkeypatch):
    """A loopback reverse proxy gets per-real-client bucketing, no bypass."""
    checked: list[str] = []

    monkeypatch.setattr(
        auth._rate_limiter, "check", lambda ip: (checked.append(ip) or True)
    )

    request = _rate_limit_request("127.0.0.1", {"X-Forwarded-For": "198.51.100.7, 127.0.0.1"})
    auth._check_rate_limit(request)

    assert checked == ["198.51.100.7"]
