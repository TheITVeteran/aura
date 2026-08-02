"""Vault MAC key: minted exclusively, read back exactly.

Found while auditing durable-write-gateway bypasses. The key writes looked like
policy violations; three were legitimate exclusive-create secret material, one
was not — and reading that one turned up a latent corruption bug.
"""
from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path

import pytest

from core.mycelium import _vault_mac_key

pytestmark = pytest.mark.unit

_WHITESPACE_BYTES = set(b" \t\n\r\x0b\x0c")


@pytest.fixture()
def base(tmp_path):
    (tmp_path / "data").mkdir(parents=True)
    return tmp_path


def _key_path(base: Path) -> Path:
    return base / "data" / "mycelium_vault.key"


# ── the corruption bug ─────────────────────────────────────────────────────


@pytest.mark.parametrize("hostile", [
    b"\x20" + b"\x01" * 31,          # leading space
    b"\x01" * 31 + b"\x0a",          # trailing newline
    b"\x09" + b"\x01" * 30 + b"\x0d",  # both ends
])
def test_a_key_is_read_back_byte_for_byte(base, hostile):
    """The read path used to .strip() the key. It is 32 raw random bytes, so
    stripping treats whitespace-VALUED bytes as padding and returns a shorter
    key than was written — after which the MAC is computed with a key that
    never existed, and vault tamper-evidence is broken forever."""
    path = _key_path(base)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with open(fd, "wb") as handle:
        handle.write(hostile)

    assert _vault_mac_key(base) == hostile


def test_the_corruption_was_not_rare(base):
    """~4.6% of random 32-byte keys begin or end with a whitespace byte, so
    this was roughly a one-in-twenty install."""
    affected = sum(
        1 for _ in range(20_000)
        if (lambda k: k[0] in _WHITESPACE_BYTES or k[-1] in _WHITESPACE_BYTES)(
            secrets.token_bytes(32)
        )
    )

    assert affected / 20_000 > 0.02, "the bug class must be materially likely"


# ── minting: exclusive, and never world-readable ───────────────────────────


def test_a_minted_key_is_owner_only(base):
    """The old sequence wrote the key and then chmod'd it, leaving it
    world-readable for the window in between."""
    key = _vault_mac_key(base)
    assert key is not None and len(key) == 32

    mode = stat.S_IMODE(_key_path(base).stat().st_mode)
    assert mode == 0o600, f"key mode {oct(mode)} should be 0o600"


def test_minting_is_idempotent(base):
    first = _vault_mac_key(base)
    second = _vault_mac_key(base)

    assert first == second


def test_an_existing_key_is_never_overwritten(base):
    """Exclusive creation is the point: a second process must adopt the
    existing key rather than replacing one that is already in use."""
    path = _key_path(base)
    original = b"\x07" * 32
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with open(fd, "wb") as handle:
        handle.write(original)

    assert _vault_mac_key(base) == original
    assert path.read_bytes() == original


def test_an_unwritable_location_degrades_rather_than_raising(tmp_path):
    """Tamper-evidence is a security property, not an availability one; losing
    it must not take the vault down."""
    unwritable = tmp_path / "nope"
    unwritable.write_text("this is a file, not a directory")

    assert _vault_mac_key(unwritable) is None
