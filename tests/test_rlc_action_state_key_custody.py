"""Contracts for SPARK external snapshot-key custody."""

from __future__ import annotations

import base64
import json

import pytest

from core.brain.llm.latent_cortex.action_state_key_custody import (
    KeychainSnapshotKeyCustodian,
    SnapshotKeyCustodyError,
)


class _Backend:
    def __init__(self, value: str | None = None, *, accept_write: bool = True) -> None:
        self.value = value
        self.accept_write = accept_write
        self.writes = 0

    def get_password(self, _service: str, _account: str) -> str | None:
        return self.value

    def set_password(self, _service: str, _account: str, password: str) -> bool:
        self.writes += 1
        if self.accept_write:
            self.value = password
        return self.accept_write


def _backend() -> _Backend:
    return _Backend(base64.b64encode(b"K" * 32).decode("ascii"))


def test_existing_external_key_wraps_and_unwraps_without_plaintext_disclosure():
    custodian = KeychainSnapshotKeyCustodian(_backend())
    data_key = b"D" * 32
    context = "a" * 64

    envelope = custodian.wrap_data_key(data_key, context_sha256=context)

    assert custodian.unwrap_data_key(envelope, context_sha256=context) == bytearray(data_key)
    assert data_key not in json.dumps(envelope, sort_keys=True).encode("utf-8")
    assert envelope["custody_identity"]["custody_class"] == "macos_keychain"


def test_missing_key_is_created_only_after_confirmed_keychain_write():
    backend = _Backend()
    custodian = KeychainSnapshotKeyCustodian(backend)

    assert backend.writes == 1
    assert len(base64.b64decode(backend.value or "", validate=True)) == 32
    assert custodian.identity["wrapping_key_id"]


def test_system_constructor_uses_strict_keychain_backend(monkeypatch):
    backend = _backend()
    monkeypatch.setattr(
        "core.brain.llm.latent_cortex.action_state_key_custody.require_keychain_backend",
        lambda: backend,
    )

    custodian = KeychainSnapshotKeyCustodian.from_system()

    assert custodian.identity["wrapping_key_id"]


def test_system_constructor_refuses_to_race_provisioning(monkeypatch):
    monkeypatch.setattr(
        "core.brain.llm.latent_cortex.action_state_key_custody.require_keychain_backend",
        lambda: _Backend(),
    )

    with pytest.raises(SnapshotKeyCustodyError, match="not_provisioned"):
        KeychainSnapshotKeyCustodian.from_system()


def test_system_provisioner_creates_and_confirms_key(monkeypatch):
    backend = _Backend()
    monkeypatch.setattr(
        "core.brain.llm.latent_cortex.action_state_key_custody.require_keychain_backend",
        lambda: backend,
    )

    custodian = KeychainSnapshotKeyCustodian.provision_system()

    assert backend.writes == 1
    assert custodian.identity["wrapping_key_id"]


def test_unconfirmed_keychain_write_fails_closed():
    with pytest.raises(SnapshotKeyCustodyError, match="write_unconfirmed"):
        KeychainSnapshotKeyCustodian(_Backend(accept_write=False))


@pytest.mark.parametrize("field", ["context_sha256", "wrapped_key_b64", "envelope_sha256"])
def test_wrapped_key_tampering_fails_closed(field: str):
    custodian = KeychainSnapshotKeyCustodian(_backend())
    envelope = dict(custodian.wrap_data_key(b"D" * 32, context_sha256="a" * 64))
    envelope[field] = "b" * 64

    with pytest.raises(SnapshotKeyCustodyError):
        custodian.unwrap_data_key(envelope, context_sha256="a" * 64)


def test_wrong_context_cannot_unwrap_data_key():
    custodian = KeychainSnapshotKeyCustodian(_backend())
    envelope = custodian.wrap_data_key(b"D" * 32, context_sha256="a" * 64)

    with pytest.raises(SnapshotKeyCustodyError, match="wrapped_key_invalid"):
        custodian.unwrap_data_key(envelope, context_sha256="b" * 64)


def test_closed_custodian_cannot_wrap_or_unwrap():
    custodian = KeychainSnapshotKeyCustodian(_backend())
    envelope = custodian.wrap_data_key(b"D" * 32, context_sha256="a" * 64)
    custodian.close()

    with pytest.raises(SnapshotKeyCustodyError, match="custodian_closed"):
        custodian.wrap_data_key(b"E" * 32, context_sha256="a" * 64)
    with pytest.raises(SnapshotKeyCustodyError, match="custodian_closed"):
        custodian.unwrap_data_key(envelope, context_sha256="a" * 64)
