from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from core.runtime.file_read_gateway import (
    StableFileReadError,
    open_stable_readonly_binary,
    read_stable_bytes,
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
