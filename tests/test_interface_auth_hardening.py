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
