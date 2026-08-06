"""`with sqlite3.connect(...)` does not close the connection.

It reads like resource management and it is transaction management. From the
stdlib docs: the connection context manager commits on success and rolls back
on an exception — and "does not implicitly close the connection". Measured:

    for i in range(5):
        with sqlite3.connect(path) as conn:
            conn.execute(...)
    # -> 5 open file descriptors on that database

They come back when the connection object is finally collected, which is at
best the end of the enclosing function and at worst whenever the cyclic
collector runs. WAL adds two more handles per connection (`-wal`, `-shm`), so
the real cost is threefold. The suite's hermetic-resource guard finds these as
"open_files" leaks at test teardown; a long-running desktop runtime finds them
as a slow climb nobody attributes to a mood update.

This module is the one-line fix, and it PRESERVES the transaction semantics
rather than replacing them — `contextlib.closing` alone would close the handle
and silently drop the commit-on-success behaviour every call site depends on.

Usage::

    from core.runtime.sqlite_support import connecting

    with connecting(sqlite3.connect(path)) as conn:
        conn.execute(...)          # commits on success, rolls back on error,
                                   # and closes either way

It takes an already-constructed connection so the call site keeps full control
of its own pragmas, timeouts, URI flags and row factories — the alternative
(wrapping `connect` itself) would have to re-expose every one of those.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

__all__ = ["connecting"]


@contextmanager
def connecting(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Commit-or-rollback like ``with conn``, then always close.

    Exactly equivalent to::

        try:
            with connection:
                ...
        finally:
            connection.close()

    so converting a call site changes only whether the descriptor is returned,
    never when the transaction ends.
    """
    try:
        with connection:
            yield connection
    finally:
        try:
            connection.close()
        except sqlite3.Error:
            # A connection that cannot close is already unusable; raising here
            # would replace the caller's real exception with a cleanup one.
            pass
