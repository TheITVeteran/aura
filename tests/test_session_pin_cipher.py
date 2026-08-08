from __future__ import annotations

import json

import pytest

from core.memory.session_pin_cipher import SessionPinCipher, SessionPinCipherError


class _Keychain:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}
        self.reject_writes = False

    def get_password(self, service: str, account: str) -> str | None:
        return self.values.get((service, account))

    def set_password(self, service: str, account: str, password: str) -> bool:
        if self.reject_writes:
            return False
        self.values[(service, account)] = password
        return True


def _sealed(cipher: SessionPinCipher) -> dict[str, str]:
    return cipher.seal(
        content="the launch phrase is heliotrope seven",
        source="Remember the launch phrase.",
        timestamp="2026-08-08T12:00:00+00:00",
        session_id="owner-before-restart",
        principal_id="owner:bryan",
        principal_surface="owner",
    )


def test_session_pin_cipher_is_ciphertext_only_and_restartable() -> None:
    backend = _Keychain()
    first = SessionPinCipher.from_backend(backend)
    envelope = _sealed(first)
    serialized = json.dumps(envelope, sort_keys=True)

    assert "heliotrope" not in serialized
    assert "owner:bryan" not in serialized
    assert "Remember the launch phrase" not in serialized

    second = SessionPinCipher.from_backend(backend, create_if_missing=False)
    opened = second.open(envelope)
    assert opened["content"] == "the launch phrase is heliotrope seven"
    assert opened["principal_id"] == "owner:bryan"
    assert opened["session_id"] == "owner-before-restart"


def test_session_pin_cipher_rejects_tamper_and_wrong_key() -> None:
    envelope = _sealed(SessionPinCipher(b"a" * 32))
    tampered = dict(envelope)
    tampered["record_id"] = "f" * 32

    with pytest.raises(SessionPinCipherError, match="authentication"):
        SessionPinCipher(b"a" * 32).open(tampered)
    with pytest.raises(SessionPinCipherError, match="identity"):
        SessionPinCipher(b"b" * 32).open(envelope)


def test_session_pin_cipher_requires_confirmed_keychain_custody() -> None:
    backend = _Keychain()
    backend.reject_writes = True

    with pytest.raises(SessionPinCipherError, match="write_rejected"):
        SessionPinCipher.from_backend(backend)


@pytest.mark.parametrize(
    ("principal_id", "surface"),
    [("", "owner"), ("owner:bryan", "")],
)
def test_session_pin_cipher_requires_complete_principal_binding(
    principal_id: str,
    surface: str,
) -> None:
    cipher = SessionPinCipher(b"a" * 32)

    with pytest.raises(SessionPinCipherError, match="principal_binding_missing"):
        cipher.seal(
            content="remember me",
            source="remember this",
            timestamp="now",
            session_id="session-a",
            principal_id=principal_id,
            principal_surface=surface,
        )
