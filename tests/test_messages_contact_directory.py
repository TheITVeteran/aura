from __future__ import annotations

import json

import pytest

from core.communication.contact_directory import (
    ContactDirectoryError,
    ContactNotConfiguredError,
    KeychainContactDirectory,
)


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


def test_keychain_contact_round_trip_never_exposes_destination() -> None:
    backend = _Keychain()
    directory = KeychainContactDirectory(backend, clock=lambda: 1000.0)

    contact = directory.provision("primary_operator", "+15550001111")
    loaded = directory.load("primary_operator")

    assert loaded == contact
    assert loaded.destination == "+15550001111"
    public = loaded.public_status()
    assert public["alias"] == "primary_operator"
    assert public["endpoint_ref"].startswith("msg_")
    assert "destination" not in public
    assert "+15550001111" not in json.dumps(public, sort_keys=True)
    assert "+15550001111" not in repr(loaded)


def test_keychain_contact_normalizes_phone_only_at_provisioning_boundary() -> None:
    directory = KeychainContactDirectory(_Keychain(), clock=lambda: 1000.0)

    contact = directory.provision("primary_operator", "(555) 000-1111")

    assert contact.destination == "+15550001111"
    assert contact.destination_kind == "phone"


def test_keychain_contact_rejects_tamper_and_wrong_alias() -> None:
    backend = _Keychain()
    directory = KeychainContactDirectory(backend, clock=lambda: 1000.0)
    directory.provision("primary_operator", "+15550001111")
    contact_key = next(key for key in backend.values if key[1] == "contact.primary_operator")
    payload = json.loads(backend.values[contact_key])
    payload["allow_outbound"] = False
    backend.values[contact_key] = json.dumps(payload, sort_keys=True)

    with pytest.raises(ContactDirectoryError, match="integrity"):
        directory.load("primary_operator")
    with pytest.raises(ContactNotConfiguredError):
        directory.load("secondary_operator")


@pytest.mark.parametrize(
    "destination",
    ["", "555", "+0123456789", "not-an-address", "person@invalid"],
)
def test_keychain_contact_rejects_invalid_destinations(destination: str) -> None:
    directory = KeychainContactDirectory(_Keychain(), clock=lambda: 1000.0)
    with pytest.raises(ValueError):
        directory.provision("primary_operator", destination)


def test_keychain_contact_requires_write_confirmation() -> None:
    backend = _Keychain()
    backend.reject_writes = True
    directory = KeychainContactDirectory(backend, clock=lambda: 1000.0)

    with pytest.raises(ContactDirectoryError, match="integrity key"):
        directory.provision("primary_operator", "+15550001111")
