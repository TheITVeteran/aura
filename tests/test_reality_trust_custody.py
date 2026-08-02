from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from core.reality_reach.trust_custody import (
    AttachmentTrustStoreError,
    KeychainAttachmentTrustStore,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


class FakeKeychain:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}
        self.accept_writes = True
        self.confirm_writes = True

    def get_password(self, service: str, account: str) -> str | None:
        return self.values.get((service, account))

    def set_password(self, service: str, account: str, password: str) -> bool:
        if self.accept_writes and self.confirm_writes:
            self.values[(service, account)] = password
        return self.accept_writes


def _store(path: Path, backend: FakeKeychain) -> KeychainAttachmentTrustStore:
    return KeychainAttachmentTrustStore(backend, path)


def _body(label: str) -> dict[str, object]:
    return {
        "grants": [
            {
                "identity": "sha256:" + "a" * 64,
                "private_device_metadata": {
                    "display_name": label,
                    "room": "private-lab",
                },
            }
        ]
    }


def test_state_is_ciphertext_only_and_round_trips(tmp_path: Path) -> None:
    backend = FakeKeychain()
    state_path = tmp_path / "trust.json"
    store = _store(state_path, backend)

    receipt = store.save(_body("Hidden Sensor"))

    encoded = state_path.read_text(encoding="utf-8")
    assert "Hidden Sensor" not in encoded
    assert "private-lab" not in encoded
    assert store.load() == _body("Hidden Sensor")
    assert receipt["sequence"] == 1
    assert receipt["key_version"] == 1
    assert os.stat(state_path).st_mode & 0o077 == 0


def test_ciphertext_tamper_fails_even_if_public_digest_is_recomputed(tmp_path: Path) -> None:
    backend = FakeKeychain()
    state_path = tmp_path / "trust.json"
    store = _store(state_path, backend)
    store.save(_body("Sensor"))
    envelope = json.loads(state_path.read_text(encoding="utf-8"))
    ciphertext = envelope["ciphertext_b64"]
    replacement = "A" if ciphertext[-1] != "A" else "B"
    envelope["ciphertext_b64"] = ciphertext[:-1] + replacement
    body = {key: value for key, value in envelope.items() if key != "envelope_sha256"}
    envelope["envelope_sha256"] = _digest(body)
    state_path.write_bytes(_canonical(envelope))

    with pytest.raises(AttachmentTrustStoreError):
        store.load()


def test_valid_older_envelope_is_refused_as_rollback(tmp_path: Path) -> None:
    backend = FakeKeychain()
    state_path = tmp_path / "trust.json"
    store = _store(state_path, backend)
    store.save(_body("First"))
    first = state_path.read_bytes()
    store.save(_body("Second"))
    state_path.write_bytes(first)

    with pytest.raises(
        AttachmentTrustStoreError,
        match="rollback_or_replay_refused",
    ):
        store.load()


def test_single_file_ahead_of_anchor_crash_window_recovers_once(tmp_path: Path) -> None:
    backend = FakeKeychain()
    state_path = tmp_path / "trust.json"
    store = _store(state_path, backend)
    store.save(_body("First"))
    anchor_key = next(key for key in backend.values if key[1].endswith("anchor-v1"))
    first_anchor = backend.values[anchor_key]
    store.save(_body("Second"))
    backend.values[anchor_key] = first_anchor

    assert store.load() == _body("Second")
    assert store.status()["committed_sequence"] == 2
    assert store.status()["recovered_commits"] == 1


def test_missing_state_after_committed_anchor_fails_closed(tmp_path: Path) -> None:
    backend = FakeKeychain()
    state_path = tmp_path / "trust.json"
    store = _store(state_path, backend)
    store.save(_body("Sensor"))
    state_path.unlink()

    with pytest.raises(AttachmentTrustStoreError, match="state_missing_after_commit"):
        store.load()


def test_keychain_root_loss_does_not_replace_existing_state(tmp_path: Path) -> None:
    backend = FakeKeychain()
    state_path = tmp_path / "trust.json"
    store = _store(state_path, backend)
    store.save(_body("Sensor"))
    keyring_key = next(key for key in backend.values if key[1].endswith("keyring-v1"))
    del backend.values[keyring_key]

    with pytest.raises(
        AttachmentTrustStoreError,
        match="keyring_missing_for_existing_state",
    ):
        _store(state_path, backend)


def test_rotation_reencrypts_head_and_preserves_content(tmp_path: Path) -> None:
    backend = FakeKeychain()
    state_path = tmp_path / "trust.json"
    store = _store(state_path, backend)
    first = store.save(_body("Sensor"))
    old_envelope = state_path.read_bytes()

    rotated = store.rotate_and_save(_body("Sensor"))

    assert first["key_version"] == 1
    assert rotated["key_version"] == 2
    assert store.load() == _body("Sensor")
    assert store.status()["rotations"] == 1
    state_path.write_bytes(old_envelope)
    with pytest.raises(AttachmentTrustStoreError, match="rollback_or_replay_refused"):
        store.load()


def test_unconfirmed_keychain_write_fails_without_state_commit(tmp_path: Path) -> None:
    backend = FakeKeychain()
    state_path = tmp_path / "trust.json"
    store = _store(state_path, backend)
    backend.confirm_writes = False

    with pytest.raises(AttachmentTrustStoreError, match="anchor_write_unconfirmed"):
        store.save(_body("Sensor"))

    assert state_path.exists() is False
    backend.confirm_writes = True
    store.save(_body("Sensor"))
    assert store.load() == _body("Sensor")
    assert store.status()["recovered_commits"] == 0


def test_symlinked_state_is_never_followed(tmp_path: Path) -> None:
    backend = FakeKeychain()
    state_path = tmp_path / "trust.json"
    store = _store(state_path, backend)
    store.save(_body("Sensor"))
    real_path = tmp_path / "moved-envelope.json"
    state_path.rename(real_path)
    state_path.symlink_to(real_path)

    with pytest.raises(AttachmentTrustStoreError, match="state_read_failed"):
        store.load()


def test_group_or_world_readable_state_is_refused(tmp_path: Path) -> None:
    backend = FakeKeychain()
    state_path = tmp_path / "trust.json"
    store = _store(state_path, backend)
    store.save(_body("Sensor"))
    state_path.chmod(0o644)

    with pytest.raises(AttachmentTrustStoreError, match="permissions_invalid"):
        store.load()


def test_duplicate_json_keys_are_refused_before_envelope_use(tmp_path: Path) -> None:
    backend = FakeKeychain()
    state_path = tmp_path / "trust.json"
    store = _store(state_path, backend)
    store.save(_body("Sensor"))
    encoded = state_path.read_text(encoding="utf-8")
    encoded = encoded.replace(
        '"algorithm":"AES-256-GCM"',
        '"algorithm":"AES-256-GCM","algorithm":"AES-256-GCM"',
        1,
    )
    state_path.write_text(encoded, encoding="utf-8")
    state_path.chmod(0o600)

    with pytest.raises(AttachmentTrustStoreError, match="state_invalid"):
        store.load()
