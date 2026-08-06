"""Ratchet: no NEW locks that lockdep cannot see.

core/runtime/lockdep.py finds ABBA deadlocks without the deadlock having to
happen — but only for locks it wraps. CLAUDE.md says exactly that: "it only
sees locks it wraps." That is a coverage claim with no number behind it, and
an ABBA detector covering an unknown fraction of a system gives an assurance
nobody can size.

The number turned out to be 8 of 729. Lockdep can reason about roughly one
percent of the locking in core/ and interface/.

721 raw constructions is not something to fix in one pass. Migrating a lock
carelessly causes precisely the deadlock the migration exists to prevent,
and 721 careless migrations would be a catastrophe. So the debt is frozen
here and the gate stops it growing: new code uses checked_lock /
checked_async_lock, or adopts an existing critical section with
instrument(name).

If this fails on code you just wrote: use the checked constructors. If you
just MIGRATED one, delete its entry from config/lockdep_baseline.json — the
test enforces that too, because a baseline that only grows is a list of
excuses and the next regression hides behind a stale allowance.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.runtime.lockdep_coverage import RawLock, scan_lock_coverage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "config" / "lockdep_baseline.json"


def _baseline() -> list[str]:
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == "aura.lockdep_baseline.v1"
    return list(payload["entries"])


@pytest.fixture(scope="module")
def report():
    return scan_lock_coverage()


class TestTheDebtDoesNotGrow:
    def test_no_new_unchecked_locks(self, report):
        offenders = report.new_since(_baseline())
        assert not offenders, (
            "new lock(s) lockdep cannot see — use checked_lock / "
            "checked_async_lock, or adopt the section with instrument():\n  "
            + "\n  ".join(f"{o.path}:{o.line} {o.kind} in {o.function}" for o in offenders)
        )

    def test_the_baseline_only_shrinks(self, report):
        """Entries that were migrated must leave the baseline."""
        stale = report.fixed_since(_baseline())
        assert not stale, (
            "these baseline entries no longer exist; delete them from "
            "config/lockdep_baseline.json:\n  " + "\n  ".join(stale)
        )

    def test_the_baseline_is_not_silently_empty(self):
        """A truncated baseline would make the ratchet pass vacuously."""
        assert len(_baseline()) > 100


class TestCoverageIsMeasured:
    def test_coverage_is_reported_as_a_number(self, report):
        payload = report.to_dict()
        assert payload["schema"] == "aura.lockdep_coverage.v1"
        assert payload["total"] == payload["raw"] + payload["checked"]
        assert 0.0 <= payload["coverage"] <= 1.0

    def test_some_locks_are_actually_checked(self, report):
        """If this hits zero, the checked lane has been abandoned."""
        assert report.checked > 0

    def test_no_locks_found_reports_zero_not_full_coverage(self, tmp_path):
        """A codebase where nothing was found has not achieved full
        coverage; it has failed to measure."""
        (tmp_path / "core").mkdir()
        empty = scan_lock_coverage(roots=("core",), repo_root=tmp_path)
        assert empty.total == 0
        assert empty.coverage == 0.0


class TestTheScannerSeesBothLanes:
    def _module(self, tmp_path, body: str):
        source = tmp_path / "core" / "m.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(body, encoding="utf-8")
        return scan_lock_coverage(roots=("core",), repo_root=tmp_path)

    @pytest.mark.parametrize(
        "expr",
        [
            "threading.Lock()",
            "threading.RLock()",
            "asyncio.Lock()",
            "asyncio.Semaphore(4)",
            "threading.Condition()",
            "threading.BoundedSemaphore(2)",
        ],
    )
    def test_raw_primitives_are_counted_as_unseen(self, tmp_path, expr):
        report = self._module(tmp_path, f"def f():\n    return {expr}\n")
        assert len(report.raw) == 1
        assert report.checked == 0

    @pytest.mark.parametrize(
        "expr",
        [
            'checked_lock("x")',
            'checked_async_lock("x")',
            'checked_async_condition("x")',
            'instrument("x")',
        ],
    )
    def test_checked_constructors_count_as_covered(self, tmp_path, expr):
        report = self._module(tmp_path, f"def f():\n    return {expr}\n")
        assert report.raw == []
        assert report.checked == 1

    def test_a_migration_moves_the_number(self, tmp_path):
        before = self._module(tmp_path, "def f():\n    return threading.Lock()\n")
        after = self._module(tmp_path, 'def f():\n    return checked_lock("f")\n')
        assert before.coverage == 0.0
        assert after.coverage == 1.0

    def test_a_skipped_name_above_the_root_does_not_blind_the_scan(self, tmp_path):
        """A repo checked out under a skipped directory still gets scanned.

        CLAUDE.md tells agents to work in ``.claude/worktrees/<name>`` — a
        path whose parts include ``.claude``. Testing the skip list against
        the absolute path made every file in such a checkout invisible, so
        the scan found nothing and ``fixed_since`` read that as "all 674
        baseline entries were migrated". Silent, and in the direction that
        retires the ratchet.
        """
        root = tmp_path / ".claude" / "worktrees" / "wt"
        (root / "core").mkdir(parents=True)
        (root / "core" / "m.py").write_text(
            "def f():\n    return threading.Lock()\n", encoding="utf-8"
        )
        report = scan_lock_coverage(roots=("core",), repo_root=root)
        assert [lock.path for lock in report.raw] == ["core/m.py"]

    def test_skips_still_apply_inside_the_root(self, tmp_path):
        (tmp_path / "core" / "__pycache__").mkdir(parents=True)
        (tmp_path / "core" / "__pycache__" / "m.py").write_text(
            "def f():\n    return threading.Lock()\n", encoding="utf-8"
        )
        assert scan_lock_coverage(roots=("core",), repo_root=tmp_path).raw == []

    def test_the_scanner_records_where(self, tmp_path):
        report = self._module(
            tmp_path,
            "class Engine:\n    def setup(self):\n        self._lock = threading.Lock()\n",
        )
        assert report.raw[0].function == "Engine.setup"
        assert report.raw[0].kind == "threading.Lock"


class TestBaselineKeysAreStable:
    def test_a_key_ignores_line_drift(self):
        assert (
            RawLock("core/a.py", 10, "threading.Lock", "C.f").key()
            == RawLock("core/a.py", 90, "threading.Lock", "C.f").key()
        )

    def test_a_key_separates_functions(self):
        assert (
            RawLock("core/a.py", 10, "threading.Lock", "C.f").key()
            != RawLock("core/a.py", 10, "threading.Lock", "C.g").key()
        )
