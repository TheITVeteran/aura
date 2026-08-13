"""Power-loss durability: F_FULLFSYNC, and the reason it is opt-in.

macOS ``fsync(2)`` pushes a write to the drive and returns; it does not flush
the drive's own cache. ``fcntl(fd, F_FULLFSYNC)`` does, and Apple's guidance is
to use it when the write must survive power loss. Aura runs on macOS and called
plain ``os.fsync`` everywhere, so every "durable" write in the runtime —
mind-state backups and hash-chained ledgers included — survived process death
but not power loss, and nothing said so.

It is opt-in rather than the default because it is genuinely expensive.
Measured on this host over 40 writes of 4KB:

    durable=False          median 0.168 ms
    durable (fsync)        median 0.214 ms   max 27.1 ms
    power_safe (FULLSYNC)  median 8.006 ms   max 11.1 ms

Roughly 37x the median. Making every write pay that would trade a silent
correctness gap for a loud liveness one, and an on-loop fsync in this codebase
has already frozen the live event loop for twenty minutes. Note the tails
though: plain fsync's worst case was worse than F_FULLFSYNC's.
"""

from __future__ import annotations

import json

import pytest

from core.runtime import atomic_writer
from core.runtime.atomic_writer import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    read_json_envelope,
)


def test_power_safe_writes_the_same_bytes(tmp_path):
    """Durability level must not change content. Obvious, and worth pinning."""
    target = tmp_path / "payload.bin"
    atomic_write_bytes(target, b"identity", power_safe=True)
    assert target.read_bytes() == b"identity"


def test_power_safe_defaults_off(tmp_path, monkeypatch):
    """The hot path must not silently start paying 8ms a write."""
    calls: list[bool] = []
    real = atomic_writer._fsync_file

    def spy(fd, *, full=False):
        calls.append(full)
        return real(fd, full=full)

    monkeypatch.setattr(atomic_writer, "_fsync_file", spy)
    atomic_write_bytes(tmp_path / "a.bin", b"x")
    assert calls, "no fsync happened at all"
    assert not any(calls), "a default write requested F_FULLFSYNC"


def test_power_safe_reaches_both_the_file_and_the_directory(tmp_path, monkeypatch):
    """A durable rename is only durable if the directory entry is flushed too."""
    calls: list[bool] = []
    real = atomic_writer._fsync_file

    def spy(fd, *, full=False):
        calls.append(full)
        return real(fd, full=full)

    monkeypatch.setattr(atomic_writer, "_fsync_file", spy)
    atomic_write_bytes(tmp_path / "b.bin", b"x", power_safe=True)
    assert calls.count(True) >= 2, (
        f"expected the file and its parent directory to be full-synced: {calls}"
    )


def test_non_durable_writes_skip_syncing_entirely(tmp_path, monkeypatch):
    calls: list[bool] = []
    monkeypatch.setattr(
        atomic_writer, "_fsync_file", lambda fd, *, full=False: calls.append(full)
    )
    atomic_write_bytes(tmp_path / "c.bin", b"x", durable=False, power_safe=True)
    assert calls == [], "durable=False must not sync even when power_safe is asked for"


def test_text_and_json_carry_the_flag_through(tmp_path, monkeypatch):
    seen: list[bool] = []
    monkeypatch.setattr(
        atomic_writer,
        "_fsync_file",
        lambda fd, *, full=False: seen.append(full),
    )
    atomic_write_text(tmp_path / "t.txt", "hello", power_safe=True)
    assert any(seen), "atomic_write_text dropped power_safe"

    seen.clear()
    atomic_write_json(
        tmp_path / "j.json", {"a": 1}, schema_version=1, power_safe=True
    )
    assert any(seen), "atomic_write_json dropped power_safe"


def test_json_envelope_still_round_trips_when_power_safe(tmp_path):
    target = tmp_path / "ledger.json"
    atomic_write_json(
        target, {"commitments": []}, schema_version=3,
        schema_name="identity_ledger", power_safe=True,
    )
    envelope = read_json_envelope(target)
    assert envelope is not None
    raw = json.loads(target.read_text())
    assert raw["schema_version"] == 3


def test_an_unsupported_filesystem_falls_back_instead_of_losing_the_write(
    tmp_path, monkeypatch
):
    """Network and virtual filesystems reject F_FULLFSYNC. Never drop the write."""
    monkeypatch.setattr(atomic_writer, "_FSYNC_NEEDS_FULLSYNC", True)
    monkeypatch.setattr(atomic_writer, "_fullsync_unsupported", False)

    fsynced: list[int] = []

    def refuse(fd, op, *args):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(atomic_writer.fcntl, "fcntl", refuse)
    monkeypatch.setattr(atomic_writer.os, "fsync", lambda fd: fsynced.append(fd))

    target = tmp_path / "fallback.bin"
    atomic_write_bytes(target, b"kept", power_safe=True)

    assert target.read_bytes() == b"kept"
    assert fsynced, "fell back to nothing instead of to fsync"
    assert atomic_writer._fullsync_unsupported is True, (
        "an unsupported filesystem must be remembered, not retried per call"
    )


def test_the_identity_ledger_asks_for_power_safe():
    """The one lane where losing a write loses something unreconstructable."""
    from pathlib import Path

    source = Path(atomic_writer.__file__).parent.parent / "identity" / "identity_ledger.py"
    text = source.read_text(encoding="utf-8")
    assert "power_safe=True" in text, (
        "the identity ledger stopped requesting power-loss durability"
    )


@pytest.mark.parametrize("power_safe", [False, True])
def test_content_is_identical_across_durability_levels(tmp_path, power_safe):
    target = tmp_path / f"same-{power_safe}.json"
    atomic_write_json(target, {"k": [1, 2, 3]}, schema_version=1, power_safe=power_safe)
    assert json.loads(target.read_text())["payload"] == {"k": [1, 2, 3]}
