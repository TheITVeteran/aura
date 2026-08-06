"""`with sqlite3.connect(...)` is a transaction manager, not a resource one.

It commits on success and rolls back on an exception, and it does NOT close the
connection — the stdlib says so plainly, and the shape reads exactly like every
other `with` that does. 127 call sites across this codebase relied on the
reading rather than the behaviour.

The descriptors come back when the connection object is collected, which is at
best the end of the enclosing function and at worst whenever the cyclic
collector runs; WAL adds `-wal` and `-shm` handles, so it is three per
connection. The suite's hermetic-resource guard surfaced it as
`open_files={.../adaptive_mood.sqlite3, -wal, -shm}` at teardown of a test that
had merely predicted a mood.

This is the same defect class as the warmup deadline and the corruption gate:
local code stating a contract the composition does not hold.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from core.runtime.sqlite_support import connecting

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parent.parent
SCANNED = ("core", "interface", "tools", "training")


def _bare_sqlite_context_lines(source: str, *, filename: str) -> list[int]:
    tree = ast.parse(source, filename=filename)
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            expression = item.context_expr
            if not isinstance(expression, ast.Call):
                continue
            function = expression.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "connect"
                and isinstance(function.value, ast.Name)
                and function.value.id == "sqlite3"
            ):
                lines.append(expression.lineno)
    return lines


class TestTheHelperBehaves:
    def test_it_closes(self, tmp_path):
        db = tmp_path / "t.db"
        with connecting(sqlite3.connect(db)) as conn:
            conn.execute("CREATE TABLE t(x)")
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    def test_it_still_commits_on_success(self, tmp_path):
        db = tmp_path / "t.db"
        with connecting(sqlite3.connect(db)) as conn:
            conn.execute("CREATE TABLE t(x)")
            conn.execute("INSERT INTO t VALUES (1)")
        with connecting(sqlite3.connect(db)) as check:
            assert check.execute("SELECT count(*) FROM t").fetchone()[0] == 1

    def test_it_still_rolls_back_on_error(self, tmp_path):
        db = tmp_path / "t.db"
        with connecting(sqlite3.connect(db)) as conn:
            conn.execute("CREATE TABLE t(x)")
        with pytest.raises(RuntimeError):
            with connecting(sqlite3.connect(db)) as conn:
                conn.execute("INSERT INTO t VALUES (1)")
                raise RuntimeError("boom")
        with connecting(sqlite3.connect(db)) as check:
            assert check.execute("SELECT count(*) FROM t").fetchone()[0] == 0

    def test_it_closes_even_when_the_body_raises(self, tmp_path):
        db = tmp_path / "t.db"
        conn = sqlite3.connect(db)
        with pytest.raises(RuntimeError):
            with connecting(conn):
                raise RuntimeError("boom")
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


class TestTheBareFormIsMeasurablyWorse:
    def test_the_bare_form_leaks_descriptors(self, tmp_path):
        """The measurement the fix rests on, run rather than cited."""
        psutil = pytest.importorskip("psutil")
        db = tmp_path / "leak.db"
        process = psutil.Process()

        def held() -> int:
            return sum(1 for f in process.open_files() if str(db) in f.path)

        conns = []
        for _ in range(5):
            conn = sqlite3.connect(db)
            conns.append(conn)
            with conn:  # the bare form, made explicit
                conn.execute("CREATE TABLE IF NOT EXISTS t(x)")
        assert held() >= 5, "the bare form should be holding every descriptor"

        for conn in conns:
            conn.close()
        assert held() == 0

    def test_the_wrapped_form_holds_none(self, tmp_path):
        psutil = pytest.importorskip("psutil")
        db = tmp_path / "clean.db"
        process = psutil.Process()

        for _ in range(5):
            with connecting(sqlite3.connect(db)) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS t(x)")

        assert sum(1 for f in process.open_files() if str(db) in f.path) == 0


class TestTheCodebaseUsesIt:
    """A ratchet: the count of bare connections may shrink, never grow."""

    @staticmethod
    def _offenders() -> list[str]:
        found: list[str] = []
        for root in SCANNED:
            for path in (REPO / root).rglob("*.py"):
                if path.name == "sqlite_support.py":
                    continue
                try:
                    body = path.read_text(encoding="utf-8")
                    lines = _bare_sqlite_context_lines(body, filename=str(path))
                except (OSError, SyntaxError, UnicodeError):
                    continue
                found.extend(f"{path.relative_to(REPO)}:{line}" for line in lines)
        return sorted(found)

    def test_scanner_ignores_prose_and_closing_helpers(self):
        source = '''
"""Never write ``with sqlite3.connect(path)`` directly."""
with self._connect() as conn:
    conn.execute("SELECT 1")
with connecting(sqlite3.connect(path)) as conn:
    conn.execute("SELECT 1")
'''
        assert _bare_sqlite_context_lines(source, filename="safe.py") == []

    def test_scanner_finds_only_the_direct_bare_context(self):
        source = '''
with sqlite3.connect(path) as conn:
    conn.execute("SELECT 1")
'''
        assert _bare_sqlite_context_lines(source, filename="unsafe.py") == [2]

    def test_no_connection_is_opened_without_being_closed(self):
        offenders = self._offenders()
        assert offenders == [], (
            "these open a sqlite connection in a `with` that never closes it; "
            "wrap with core.runtime.sqlite_support.connecting(...):\n  "
            + "\n  ".join(offenders)
        )
