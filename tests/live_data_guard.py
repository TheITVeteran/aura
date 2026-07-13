"""tests/live_data_guard.py — hermeticity guard for the LIVE data root.

Unit tests must never write into the real ~/.aura/data: on 2026-07-12 the
session-memory pin tests were found appending fixture pins to the LIVE
ledger (ember-vault-93 x195) — phantom memories Aura could recall to her
operator as real. config.paths.data_dir resolves to the live root even
under AURA_TEST_MODE, so any un-redirected consumer is one test away from
polluting live cognition state.

Two detection layers:
  - builtins.open write-hook (installed per-test by the autouse fixture in
    conftest): exact test attribution for Python-level file writes;
  - end-of-session mtime/size sweep over the live root (session fixture):
    catches SQLite's C-level writes that bypass builtins.open, attributed
    to the chunk.

Phase 1 (now): REPORT mode — violations append to
  $AURA_LOG_DIR/live_data_guard_violations.jsonl (never fails a test).
Phase 2 (after one full cert builds the ledger): the allowlist freezes,
  AURA_TEST_LIVE_DATA_GUARD=fail flips new violations to test failures.
"""

from __future__ import annotations

import builtins
import json
import os
import time
from pathlib import Path

LIVE_DATA_ROOT = (Path.home() / ".aura" / "data").resolve()
_WRITE_MODE_CHARS = ("w", "a", "x", "+")
_REAL_OPEN = builtins.open


def _guard_mode() -> str:
    return os.environ.get("AURA_TEST_LIVE_DATA_GUARD", "report").strip().lower()


def _violation_ledger() -> Path | None:
    log_dir = os.environ.get("AURA_LOG_DIR")
    if not log_dir:
        return None
    path = Path(log_dir) / "live_data_guard_violations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def record_violation(kind: str, path: str, test_id: str) -> None:
    entry = {
        "kind": kind,
        "path": path,
        "test": test_id,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    ledger = _violation_ledger()
    if ledger is not None:
        with _REAL_OPEN(ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    if _guard_mode() == "fail":
        raise AssertionError(
            f"live-data hermeticity violation: {kind} of {path} by {test_id} — "
            "redirect the path (monkeypatch/tmp_path); tests must never write "
            "into the real ~/.aura/data"
        )


def _is_live_write(file, mode: str) -> bool:
    if not any(c in mode for c in _WRITE_MODE_CHARS):
        return False
    if "r" in mode and "+" not in mode:
        return False
    try:
        resolved = Path(os.fspath(file)).resolve()
    except (TypeError, ValueError, OSError):
        return False  # fd-based or exotic opens: the mtime sweep covers them
    return resolved.is_relative_to(LIVE_DATA_ROOT)


def make_guarded_open(test_id: str):
    def guarded_open(file, mode="r", *args, **kwargs):
        if _is_live_write(file, str(mode)):
            record_violation("open-write", str(file), test_id)
        return _REAL_OPEN(file, mode, *args, **kwargs)

    return guarded_open


def snapshot_live_root(max_files: int = 20000) -> dict[str, tuple[float, int]]:
    """mtime/size snapshot of the live data root for the end-of-session sweep."""
    snap: dict[str, tuple[float, int]] = {}
    if not LIVE_DATA_ROOT.is_dir():
        return snap
    count = 0
    for dirpath, dirnames, filenames in os.walk(LIVE_DATA_ROOT):
        for fname in filenames:
            if count >= max_files:
                return snap
            p = Path(dirpath) / fname
            try:
                st = p.stat()
            except OSError:
                continue
            snap[str(p)] = (st.st_mtime, st.st_size)
            count += 1
    return snap


def diff_snapshots(
    before: dict[str, tuple[float, int]],
    after: dict[str, tuple[float, int]],
) -> list[str]:
    changed = [p for p, meta in after.items() if before.get(p) not in (None, meta)]
    created = [p for p in after if p not in before]
    return sorted(changed + created)
