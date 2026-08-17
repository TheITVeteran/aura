"""Crash-durable path mutation primitives stay centralized in AtomicWriter."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from core.runtime.atomic_writer import (
    AtomicWriteError,
    async_durable_replace,
    async_durable_unlink,
    atomic_hardlink_replace,
    atomic_append_text,
    durable_replace,
    durable_unlink,
)


def test_atomic_hardlink_replace_reuses_immutable_inode(tmp_path: Path) -> None:
    source = tmp_path / "generation.safetensors"
    target = tmp_path / "checkpoint_latest.safetensors"
    source.write_bytes(b"immutable tensor bytes")
    source.chmod(0o400)
    target.write_bytes(b"obsolete mirror")

    assert atomic_hardlink_replace(source, target) is True
    assert source.samefile(target)
    assert target.read_bytes() == b"immutable tensor bytes"
    assert atomic_hardlink_replace(source, target) is False


def test_atomic_hardlink_replace_rejects_symlink_source(tmp_path: Path) -> None:
    source = tmp_path / "generation.safetensors"
    source.write_bytes(b"immutable tensor bytes")
    alias = tmp_path / "source-link"
    alias.symlink_to(source)

    with pytest.raises(AtomicWriteError, match="regular file"):
        atomic_hardlink_replace(alias, tmp_path / "compatibility")


def test_atomic_hardlink_replace_keeps_old_target_when_replace_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from core.runtime import atomic_writer

    source = tmp_path / "generation.safetensors"
    target = tmp_path / "checkpoint_latest.safetensors"
    source.write_bytes(b"new immutable bytes")
    target.write_bytes(b"old complete bytes")

    def fail_replace(_source, _target):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(atomic_writer.os, "replace", fail_replace)
    with pytest.raises(OSError, match="rename failure"):
        atomic_hardlink_replace(source, target)

    assert target.read_bytes() == b"old complete bytes"
    assert not any(
        child.name.startswith(atomic_writer.DEFAULT_TEMP_PREFIX)
        for child in tmp_path.iterdir()
    )


def _append_worker(path: str, prefix: str, count: int) -> None:
    for index in range(count):
        atomic_append_text(path, f"{prefix}:{index:03d}\n")


def test_durable_replace_atomically_moves_file(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    target = tmp_path / "nested" / "target.txt"
    source.write_text("new", encoding="utf-8")

    durable_replace(source, target)

    assert not source.exists()
    assert target.read_text(encoding="utf-8") == "new"


def test_durable_unlink_removes_broken_symlink(tmp_path: Path) -> None:
    link = tmp_path / "broken"
    link.symlink_to(tmp_path / "missing")

    assert durable_unlink(link) is True
    assert not link.is_symlink()
    assert durable_unlink(link, missing_ok=True) is False


def test_durable_unlink_refuses_directory(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()

    with pytest.raises(IsADirectoryError):
        durable_unlink(directory)
    assert directory.exists()


@pytest.mark.asyncio
async def test_async_durable_path_operations(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    source.write_text("async", encoding="utf-8")

    await async_durable_replace(source, target)
    assert target.read_text(encoding="utf-8") == "async"
    assert await async_durable_unlink(target) is True
    assert not target.exists()


def test_atomic_append_is_noninterleaving_across_processes(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_append_worker, args=(str(ledger), f"p{index}", 20))
        for index in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 80
    assert len(set(lines)) == 80
    assert all(line.startswith(("p0:", "p1:", "p2:", "p3:")) for line in lines)
