from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path

import pytest

from core.runtime.file_read_gateway import (
    StableFileReadError,
    open_stable_readonly_binary,
    read_stable_bytes,
    read_stable_directory_files,
)


def test_stable_reader_returns_exact_bounded_bytes(tmp_path: Path):
    target = tmp_path / "evidence.bin"
    target.write_bytes(b"evidence")

    assert read_stable_bytes(target, max_bytes=8) == b"evidence"


def test_stable_reader_rejects_symlink_and_nonregular_input(tmp_path: Path):
    target = tmp_path / "evidence.bin"
    target.write_bytes(b"evidence")
    link = tmp_path / "evidence-link"
    link.symlink_to(target)

    with pytest.raises(StableFileReadError, match="symlink_rejected"):
        read_stable_bytes(link, max_bytes=32)
    with pytest.raises(StableFileReadError, match="not_regular_file"):
        read_stable_bytes(tmp_path, max_bytes=32)


def test_stable_reader_rejects_fifo_without_waiting_for_a_writer(tmp_path: Path):
    target = tmp_path / "evidence.fifo"
    os.mkfifo(target)

    with pytest.raises(StableFileReadError, match="not_regular_file"):
        read_stable_bytes(target, max_bytes=32)


def test_stable_reader_translates_nofollow_race_to_symlink_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "evidence.bin"
    target.write_bytes(b"evidence")

    def raced_open(_path: object, _flags: int) -> int:
        raise OSError(errno.ELOOP, "symbolic link race")

    monkeypatch.setattr(os, "open", raced_open)
    with pytest.raises(StableFileReadError, match="symlink_rejected"):
        read_stable_bytes(target, max_bytes=32)


def test_stable_reader_enforces_bound_before_read(tmp_path: Path):
    target = tmp_path / "evidence.bin"
    target.write_bytes(b"too-large")

    with pytest.raises(StableFileReadError, match="size_exceeds_bound"):
        read_stable_bytes(target, max_bytes=3)


def test_stable_reader_detects_mutation_while_descriptor_is_open(tmp_path: Path):
    target = tmp_path / "evidence.bin"
    target.write_bytes(b"before")

    with pytest.raises(StableFileReadError, match="changed_during_read"):
        with open_stable_readonly_binary(target, max_bytes=64) as (handle, identity):
            assert handle.read() == b"before"
            assert identity.size == 6
            target.write_bytes(b"after-change")


def test_stable_directory_reader_returns_one_locked_generation(tmp_path: Path):
    from core.runtime.file_write_gateway import (
        DirectoryFileWriteBatchEntry,
        FileWriteGateway,
    )

    FileWriteGateway().write_bytes_batch_in_directory(
        tmp_path,
        (
            DirectoryFileWriteBatchEntry("data.bin", b"data"),
            DirectoryFileWriteBatchEntry("manifest.json", b"{}"),
        ),
        allowed_existing_names={"data.bin", "manifest.json"},
        commit_marker="manifest.json",
        source="unit.read_snapshot",
    )

    assert read_stable_directory_files(
        tmp_path,
        names={"data.bin", "manifest.json"},
        max_bytes_per_file=32,
    ) == {"data.bin": b"data", "manifest.json": b"{}"}


def test_stable_directory_reader_rejects_mixed_inventory(tmp_path: Path):
    (tmp_path / ".aura_file_write_batch.lock").write_bytes(b"")
    (tmp_path / ".aura_file_write_batch.lock").chmod(0o600)
    (tmp_path / "manifest.json").write_bytes(b"{}")
    (tmp_path / "unexpected.txt").write_text("foreign")

    with pytest.raises(
        StableFileReadError,
        match="directory_inventory_mismatch",
    ):
        read_stable_directory_files(
            tmp_path,
            names={"manifest.json"},
            max_bytes_per_file=32,
        )


def test_stable_directory_reader_rejects_replaced_lock_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import core.runtime.file_read_gateway as reader_module
    from core.runtime.file_write_gateway import (
        DirectoryFileWriteBatchEntry,
        FileWriteGateway,
    )

    FileWriteGateway().write_bytes_batch_in_directory(
        tmp_path,
        (DirectoryFileWriteBatchEntry("manifest.json", b"{}"),),
        allowed_existing_names={"manifest.json"},
        commit_marker="manifest.json",
        source="unit.read_lock_replace",
    )
    real_flock = reader_module.fcntl.flock
    replaced = False

    def replace_after_lock(descriptor, operation):
        nonlocal replaced
        result = real_flock(descriptor, operation)
        if operation == reader_module.fcntl.LOCK_SH and not replaced:
            replaced = True
            lock = tmp_path / ".aura_file_write_batch.lock"
            lock.unlink()
            lock.write_bytes(b"replacement")
            lock.chmod(0o600)
        return result

    monkeypatch.setattr(reader_module.fcntl, "flock", replace_after_lock)
    with pytest.raises(
        StableFileReadError,
        match="directory_lock_path_changed",
    ):
        read_stable_directory_files(
            tmp_path,
            names={"manifest.json"},
            max_bytes_per_file=32,
        )


def test_stable_directory_reader_rejects_hardlinked_artifact(
    tmp_path: Path,
):
    from core.runtime.file_write_gateway import (
        DirectoryFileWriteBatchEntry,
        FileWriteGateway,
    )

    FileWriteGateway().write_bytes_batch_in_directory(
        tmp_path,
        (DirectoryFileWriteBatchEntry("manifest.json", b"{}"),),
        allowed_existing_names={"manifest.json"},
        commit_marker="manifest.json",
        source="unit.read_hardlink",
    )
    os.link(
        tmp_path / "manifest.json",
        tmp_path.parent / f"{tmp_path.name}-manifest-hardlink",
    )

    with pytest.raises(StableFileReadError, match="not_regular_file"):
        read_stable_directory_files(
            tmp_path,
            names={"manifest.json"},
            max_bytes_per_file=32,
        )


def test_stable_directory_reader_rejects_locked_hardlinked_lock_without_waiting(
    tmp_path: Path,
):
    backing = tmp_path.parent / f"{tmp_path.name}-lock-backing"
    backing.write_bytes(b"lock")
    backing.chmod(0o600)
    os.link(backing, tmp_path / ".aura_file_write_batch.lock")
    (tmp_path / "manifest.json").write_bytes(b"{}")
    descriptor = os.open(backing, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        with pytest.raises(
            StableFileReadError,
            match="directory_lock_path_changed",
        ):
            read_stable_directory_files(
                tmp_path,
                names={"manifest.json"},
                max_bytes_per_file=32,
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
