"""Canonical no-follow boundary for stable reads of evidence and state files."""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

PathLike = str | Path


class StableFileReadError(OSError):
    """A file could not be proven regular, bounded, and stable while read."""

    def __init__(self, code: str, path: PathLike) -> None:
        super().__init__(f"{code}: {path}")
        self.code = code


@dataclass(frozen=True)
class StableFileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> StableFileIdentity:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            size=value.st_size,
            modified_ns=value.st_mtime_ns,
            changed_ns=value.st_ctime_ns,
        )


@contextmanager
def open_stable_readonly_binary(
    path: PathLike,
    *,
    max_bytes: int,
) -> Iterator[tuple[BinaryIO, StableFileIdentity]]:
    """Open a regular file without following its final symlink.

    The descriptor identity is checked after a successful read so replacement,
    truncation, or mutation cannot silently produce an accepted receipt.
    """

    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")
    target = Path(path).expanduser()
    if target.is_symlink():
        raise StableFileReadError("symlink_rejected", target)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(target, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise StableFileReadError("symlink_rejected", target) from exc
        raise
    handle: BinaryIO | None = None
    try:
        before_stat = os.fstat(fd)
        if not stat.S_ISREG(before_stat.st_mode):
            raise StableFileReadError("not_regular_file", target)
        before = StableFileIdentity.from_stat(before_stat)
        if before.size < 0 or before.size > max_bytes:
            raise StableFileReadError("size_exceeds_bound", target)
        handle = os.fdopen(fd, "rb", closefd=False)
        try:
            yield handle, before
        except BaseException:  # noqa: BLE001 - preserve the caller's validation failure
            raise
        else:
            after = StableFileIdentity.from_stat(os.fstat(fd))
            if after != before:
                raise StableFileReadError("changed_during_read", target)
    finally:
        try:
            if handle is not None:
                handle.close()
        finally:
            os.close(fd)


def read_stable_bytes(path: PathLike, *, max_bytes: int) -> bytes:
    """Read one stable regular file under an explicit byte bound."""

    with open_stable_readonly_binary(path, max_bytes=max_bytes) as (handle, identity):
        payload = handle.read(max_bytes + 1)
        if len(payload) > max_bytes or len(payload) != identity.size:
            raise StableFileReadError("bounded_read_length_mismatch", path)
        if handle.read(1):
            raise StableFileReadError("grew_beyond_bound", path)
        return payload


__all__ = [
    "StableFileIdentity",
    "StableFileReadError",
    "open_stable_readonly_binary",
    "read_stable_bytes",
]
