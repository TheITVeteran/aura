"""tests/test_intention_loop_bounded_growth.py
===============================================
The IntentionLoop SQLite table must stay bounded over long-horizon runs.

prune_old() only removes 30-day-old rows and was never called, so under
continuous operation the `intentions` table grew unbounded within the window
(observed: a session-bloated 79MB / 9308-row table dragged a DNU run 7-10x
slower via slow DB ops). prune_excess() bounds the table by COUNT, keeps the
newest completed rows, always keeps active intentions, and self-maintains from
_persist.
"""
from __future__ import annotations

import time

from core.agency.intention_loop import IntentionLoop


def _make_loop(tmp_path):
    return IntentionLoop(db_path=str(tmp_path / "intention_loop_test.db"))


def _insert(il, n, *, completed: bool, start: float = 0.0):
    """Insert n rows directly via the DB for a deterministic count test."""
    for i in range(n):
        ts = start + i
        il._conn.execute(
            """INSERT OR REPLACE INTO intentions
               (id, intention, drive, intended_at, status, completed_at)
               VALUES (?,?,?,?,?,?)""",
            (
                f"{'c' if completed else 'a'}-{ts}-{i}",
                "x", "curiosity", ts,
                "completed" if completed else "intended",
                ts if completed else None,
            ),
        )
    il._conn.commit()


def test_prune_excess_bounds_completed_rows(tmp_path):
    il = _make_loop(tmp_path)
    try:
        _insert(il, 3000, completed=True, start=1000.0)
        assert il._conn.execute("SELECT COUNT(*) FROM intentions").fetchone()[0] == 3000
        deleted = il.prune_excess(max_rows=500)
        total = il._conn.execute("SELECT COUNT(*) FROM intentions").fetchone()[0]
        assert deleted == 2500
        assert total == 500
    finally:
        il.close()


def test_prune_excess_keeps_newest_and_all_active(tmp_path):
    il = _make_loop(tmp_path)
    try:
        _insert(il, 100, completed=True, start=1000.0)   # older completed
        _insert(il, 100, completed=True, start=5000.0)   # newer completed
        _insert(il, 20, completed=False)                 # active (never pruned)
        il.prune_excess(max_rows=50)
        # 50 newest completed + 20 active kept.
        total = il._conn.execute("SELECT COUNT(*) FROM intentions").fetchone()[0]
        active = il._conn.execute(
            "SELECT COUNT(*) FROM intentions WHERE completed_at IS NULL"
        ).fetchone()[0]
        kept_old = il._conn.execute(
            "SELECT COUNT(*) FROM intentions WHERE completed_at < 5000.0"
        ).fetchone()[0]
        assert active == 20            # every active intention survives
        assert total == 70            # 50 completed + 20 active
        assert kept_old == 0          # the oldest completed rows were dropped
    finally:
        il.close()


def test_prune_excess_noop_under_cap(tmp_path):
    il = _make_loop(tmp_path)
    try:
        _insert(il, 10, completed=True, start=1.0)
        assert il.prune_excess(max_rows=100) == 0
        assert il._conn.execute("SELECT COUNT(*) FROM intentions").fetchone()[0] == 10
    finally:
        il.close()


def test_completed_at_index_exists(tmp_path):
    il = _make_loop(tmp_path)
    try:
        idx = {
            r[0]
            for r in il._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_intentions_completed_at" in idx
    finally:
        il.close()
