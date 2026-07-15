"""BlackHole must never hand back plaintext from encrypt().

The defect: ``encrypt()`` logged "Encryption bypass active (no key)" and
returned ``data`` unchanged when Horcrux had not initialized. On first boot,
every memory the privacy story covers was written to disk in the clear — and
nothing in the return value said so, so the failure was invisible at the call
site and easy to forget while demoing privacy.

Two properties are tested here: encryption fails closed when there is no key,
and there is normally a key at all (first boot provisions one).
"""
from __future__ import annotations

import pytest

from core.memory.black_hole import (
    BlackHole,
    BlackHoleEncryptionUnavailable,
    _local_key_path,
)


@pytest.fixture(autouse=True)
def _isolated_key_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_BLACK_HOLE_KEY_DIR", str(tmp_path / "keys"))
    yield


def test_encrypt_without_a_key_raises_instead_of_returning_plaintext():
    """The headline: no key must mean no output, not clear output."""
    bh = BlackHole()  # never started — no key
    secret = b"Bryan's private memory"

    with pytest.raises(BlackHoleEncryptionUnavailable):
        bh.encrypt(secret)


def test_encrypt_never_returns_its_input():
    """Guards the exact old shape: `return data`."""
    bh = BlackHole()
    secret = b"Bryan's private memory"
    try:
        out = bh.encrypt(secret)
    except BlackHoleEncryptionUnavailable:
        return  # fail-closed is the correct behaviour
    assert out != secret, "encrypt() returned its plaintext input"


def test_decrypt_without_a_key_raises_instead_of_returning_the_blob():
    bh = BlackHole()
    with pytest.raises(BlackHoleEncryptionUnavailable):
        bh.decrypt(b"\x00" * 32)


def test_encryption_active_reports_the_truth():
    bh = BlackHole()
    assert bh.encryption_active is False
    assert bh.key_provenance == "none"


def test_first_boot_without_horcrux_still_encrypts(monkeypatch):
    """First boot must not be the boot where privacy quietly does not apply."""
    monkeypatch.setattr(
        "core.memory.black_hole.get_runtime_service",
        lambda _name, default=None: None,
    )

    bh = BlackHole()
    bh.on_start()

    assert bh.encryption_active is True
    assert bh.key_provenance == "local"

    secret = b"Bryan's private memory"
    blob = bh.encrypt(secret)
    assert blob != secret
    assert secret not in blob
    assert bh.decrypt(blob) == secret


def test_local_key_is_persisted_and_reused(monkeypatch):
    """A regenerated key each boot would make yesterday's memories unreadable."""
    monkeypatch.setattr(
        "core.memory.black_hole.get_runtime_service",
        lambda _name, default=None: None,
    )

    first = BlackHole()
    first.on_start()
    blob = first.encrypt(b"remember this")

    assert _local_key_path().exists()

    second = BlackHole()
    second.on_start()
    assert second.decrypt(blob) == b"remember this"


def test_local_key_file_is_not_world_readable(monkeypatch):
    import os
    import stat

    monkeypatch.setattr(
        "core.memory.black_hole.get_runtime_service",
        lambda _name, default=None: None,
    )
    BlackHole().on_start()

    mode = os.stat(_local_key_path()).st_mode
    assert not (mode & stat.S_IRGRP), "key is group-readable"
    assert not (mode & stat.S_IROTH), "key is world-readable"


def test_horcrux_key_is_preferred_when_available(monkeypatch):
    """The local key is a fallback, never a replacement."""

    class _Horcrux:
        derived_key = b"\x11" * 32

    monkeypatch.setattr(
        "core.memory.black_hole.get_runtime_service",
        lambda _name, default=None: _Horcrux(),
    )

    bh = BlackHole()
    bh.on_start()
    assert bh.key_provenance == "horcrux"
    assert not _local_key_path().exists(), "provisioned a local key despite Horcrux"


def test_encrypt_json_cannot_leak_plaintext():
    """The JSON convenience wrapper inherits the fail-closed guarantee."""
    bh = BlackHole()
    with pytest.raises(BlackHoleEncryptionUnavailable):
        bh.encrypt_json({"secret": "Bryan's private memory"})
