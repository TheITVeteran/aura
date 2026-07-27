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

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.memory.black_hole import (
    BlackHole,
    BlackHoleEncryptionUnavailable,
    _local_key_path,
    _provision_local_key,
    _read_local_key,
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
    assert bh.key_identity_sha256 == ""


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
    assert len(bh.key_identity_sha256) == 64

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


def test_concurrent_first_boot_converges_on_one_local_key():
    with ThreadPoolExecutor(max_workers=8) as pool:
        keys = list(pool.map(lambda _index: _provision_local_key(), range(8)))

    assert all(key is not None for key in keys)
    assert len(set(keys)) == 1
    assert _local_key_path().read_bytes() == keys[0]


def test_local_key_read_accepts_publish_link_cleanup(tmp_path, monkeypatch):
    import core.memory.black_hole as black_hole

    payload = os.urandom(32)
    staged = tmp_path / "publisher-temp"
    path = tmp_path / "published.key"
    staged.write_bytes(payload)
    staged.chmod(0o600)
    os.link(staged, path)

    real_read = black_hole.os.read
    cleaned = False

    def _unlink_staging_during_read(fd, size):
        nonlocal cleaned
        if not cleaned:
            staged.unlink()
            cleaned = True
        return real_read(fd, size)

    monkeypatch.setattr(black_hole.os, "read", _unlink_staging_during_read)

    assert _read_local_key(path) == payload
    assert path.stat().st_nlink == 1


def test_local_key_read_refuses_permission_change(tmp_path, monkeypatch):
    import core.memory.black_hole as black_hole

    path = tmp_path / "changing-mode.key"
    path.write_bytes(os.urandom(32))
    path.chmod(0o600)
    real_read = black_hole.os.read
    changed = False

    def _change_mode_during_read(fd, size):
        nonlocal changed
        if not changed:
            path.chmod(0o640)
            changed = True
        return real_read(fd, size)

    monkeypatch.setattr(black_hole.os, "read", _change_mode_during_read)
    try:
        with pytest.raises(RuntimeError, match="changed while reading"):
            _read_local_key(path)
    finally:
        path.chmod(0o600)


def test_malformed_local_key_is_preserved_and_encryption_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "core.memory.black_hole.get_runtime_service",
        lambda _name, default=None: None,
    )
    path = _local_key_path()
    path.parent.mkdir(parents=True, mode=0o700)
    path.write_bytes(b"truncated")
    path.chmod(0o600)
    before = path.read_bytes()

    bh = BlackHole()
    bh.on_start()

    assert bh.encryption_active is False
    assert bh.key_provenance == "none"
    with pytest.raises(BlackHoleEncryptionUnavailable):
        bh.encrypt(b"must not leak")
    assert path.read_bytes() == before


def test_local_key_symlink_is_rejected_without_touching_target(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "core.memory.black_hole.get_runtime_service",
        lambda _name, default=None: None,
    )
    outside = tmp_path / "outside.key"
    outside.write_bytes(b"x" * 32)
    path = _local_key_path()
    path.parent.mkdir(parents=True, mode=0o700)
    path.symlink_to(outside)

    bh = BlackHole()
    bh.on_start()

    assert bh.encryption_active is False
    assert outside.read_bytes() == b"x" * 32
    assert path.is_symlink()


def test_horcrux_key_is_preferred_when_available(monkeypatch):
    """The local key is a fallback, never a replacement."""

    class _Horcrux:
        derived_key = b"\x11" * 32
        key_identity_sha256 = hashlib.sha256(
            b"AURA-HORCRUX-KEY-IDENTITY-v1\x00" + derived_key
        ).hexdigest()

    monkeypatch.setattr(
        "core.memory.black_hole.get_runtime_service",
        lambda _name, default=None: _Horcrux(),
    )

    bh = BlackHole()
    bh.on_start()
    assert bh.key_provenance == "horcrux"
    assert bh.key_identity_sha256 == _Horcrux.key_identity_sha256
    assert not _local_key_path().exists(), "provisioned a local key despite Horcrux"


def test_horcrux_identity_must_be_a_canonical_sha256(monkeypatch):
    class _Horcrux:
        derived_key = b"\x12" * 32
        key_identity_sha256 = "Z" * 64

    monkeypatch.setattr(
        "core.memory.black_hole.get_runtime_service",
        lambda _name, default=None: _Horcrux(),
    )
    black_hole = BlackHole()

    with pytest.raises(RuntimeError, match="key identity is unavailable"):
        black_hole.on_start()

    assert black_hole.encryption_active is False
    assert black_hole.key_provenance == "none"
    assert black_hole.key_identity_sha256 == ""


def test_encrypt_json_cannot_leak_plaintext():
    """The JSON convenience wrapper inherits the fail-closed guarantee."""
    bh = BlackHole()
    with pytest.raises(BlackHoleEncryptionUnavailable):
        bh.encrypt_json({"secret": "Bryan's private memory"})
