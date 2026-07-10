"""Crash-durable path mutation primitives stay centralized in AtomicWriter."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from core.runtime.atomic_writer import (
    async_durable_replace,
    async_durable_unlink,
    atomic_append_text,
    durable_replace,
    durable_unlink,
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
