"""A large write that would fill the volume must be declined, not attempted.

LIVE, 2026-08-13. The data volume sat at 99% with 19GB free of 1.8TB, and the
whole runtime was in permanent metabolic lockdown because of it:

    threat flagged by immune:resource_monitor (severity=0.90): resource strain:
    disk at 99%                                          [every ~15 seconds]
    Metabolism: Throttling due to resource pressure (Lockdown active)
    allostasis protecting: disk_percent is already past its red line

The cause was not her data. Sixty git worktrees under .claude/worktrees each
carried their own copy of training/fused-model — 17GB apiece, two of them
338GB — plus 157 one-gigabyte training checkpoints from a single run. All of it
git-ignored build output that nothing had a budget for, so nothing ever said
no. Clearing it took the volume from 19GB free to 516GB free.

Two halves, because either alone fails: a budget checked BEFORE a large write,
and retention applied to what accumulated before the budget existed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.runtime.disk_budget import (
    DEFAULT_FLOOR_GB,
    DiskBudgetRefusal,
    ensure_headroom_for,
    free_space,
    prune_superseded_artifacts,
)


def test_free_space_reads_the_real_volume() -> None:
    space = free_space(Path.home())

    assert space.total_gb > 0
    assert 0.0 <= space.used_fraction <= 1.0


def test_a_write_that_fits_is_allowed() -> None:
    ensure_headroom_for(1024, purpose="a tiny file", path=Path.home())


def test_a_write_that_would_fill_the_volume_is_refused() -> None:
    """The 17GB fuse that must not start with 19GB free."""
    space = free_space(Path.home())
    absurd = int((space.free_gb + DEFAULT_FLOOR_GB + 50.0) * (1024**3))

    with pytest.raises(DiskBudgetRefusal) as caught:
        ensure_headroom_for(absurd, purpose="fuse a 32B model", path=Path.home())

    message = str(caught.value)
    assert "fuse a 32B model" in message
    assert "floor" in message


def test_the_refusal_names_what_it_would_have_left() -> None:
    """A refusal that cannot be acted on gets retried verbatim."""
    space = free_space(Path.home())
    absurd = int((space.free_gb + DEFAULT_FLOOR_GB + 10.0) * (1024**3))

    with pytest.raises(DiskBudgetRefusal) as caught:
        ensure_headroom_for(absurd, purpose="x", path=Path.home())

    assert "GB" in str(caught.value)


# ── Retention ──────────────────────────────────────────────────────────────

def _generation(root: Path, name: str, *, size: int = 2048) -> Path:
    entry = root / name
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "weights.bin").write_bytes(b"0" * size)
    return entry


def test_retention_keeps_the_newest_generations(tmp_path: Path) -> None:
    import os
    import time

    for index in range(6):
        entry = _generation(tmp_path, f"gen{index}")
        os.utime(entry, (time.time() + index, time.time() + index))

    removed = prune_superseded_artifacts(tmp_path, keep=3)

    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert len(remaining) == 3
    assert remaining == ["gen3", "gen4", "gen5"]
    assert len(removed) == 3


def test_the_protected_entry_is_never_removed(tmp_path: Path) -> None:
    """The active model is protected by name whatever its age."""
    import os
    import time

    _generation(tmp_path, "active-and-oldest")
    os.utime(tmp_path / "active-and-oldest", (1, 1))
    for index in range(5):
        entry = _generation(tmp_path, f"gen{index}")
        os.utime(entry, (time.time() + index, time.time() + index))

    prune_superseded_artifacts(tmp_path, keep=2, protect=("active-and-oldest",))

    assert (tmp_path / "active-and-oldest").is_dir()


def test_nothing_is_removed_below_the_keep_count(tmp_path: Path) -> None:
    for index in range(3):
        _generation(tmp_path, f"gen{index}")

    assert prune_superseded_artifacts(tmp_path, keep=3) == []
    assert len(list(tmp_path.iterdir())) == 3


def test_a_dry_run_removes_nothing(tmp_path: Path) -> None:
    import os
    import time

    for index in range(5):
        entry = _generation(tmp_path, f"gen{index}")
        os.utime(entry, (time.time() + index, time.time() + index))

    planned = prune_superseded_artifacts(tmp_path, keep=2, dry_run=True)

    assert len(planned) == 3
    assert len(list(tmp_path.iterdir())) == 5


def test_a_missing_root_is_not_an_error(tmp_path: Path) -> None:
    assert prune_superseded_artifacts(tmp_path / "nope", keep=1) == []
