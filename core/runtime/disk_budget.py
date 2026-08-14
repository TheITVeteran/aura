"""Refuse the write that would fill the disk, and prune what already did.

LIVE, 2026-08-13. The data volume sat at 99% with 19GB free on 1.8TB, and the
consequences reached everything:

    threat flagged by immune:resource_monitor (severity=0.90): resource strain:
    disk at 99%                                         [every ~15 seconds]
    Metabolism: Throttling due to resource pressure (Lockdown active)
    Metabolism: deferring on allostasis signal — allostasis protecting:
    disk_percent is already past its red line

She was in permanent metabolic lockdown, and the reason was not her data. Sixty
git worktrees under .claude/worktrees each carried their own copy of
training/fused-model — 17GB apiece, two of them 338GB — for a total of 951GB of
duplicated, git-ignored build output. Nothing had a budget, so nothing ever
said no.

Two halves, because either alone fails:

  * a BUDGET, checked before a large artifact is written, so the volume cannot
    be filled by something that could have been declined;
  * RETENTION, applied to what accumulated anyway, because a budget added after
    the fact does not remove what predates it.

Deliberately conservative about what it will remove: only artifact trees it is
told to manage, only entries beyond the keep count, never the active one, and
never anything git-tracked. A cleaner that deletes something load-bearing is a
worse failure than a full disk.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

__all__ = [
    "DiskBudgetRefusal",
    "FreeSpace",
    "ensure_headroom_for",
    "free_space",
    "prune_superseded_artifacts",
]

#: Below this, a large optional write is refused rather than attempted. A model
#: fuse that lands on a full volume corrupts the artifact AND the host.
DEFAULT_FLOOR_GB = 60.0

#: How many generations of a versioned artifact tree to keep. One to run from,
#: one to roll back to, one for the comparison someone is mid-way through.
DEFAULT_KEEP = 3


class DiskBudgetRefusal(RuntimeError):
    """A write was declined because it would breach the free-space floor."""


@dataclass(frozen=True, slots=True)
class FreeSpace:
    path: str
    free_gb: float
    total_gb: float

    @property
    def used_fraction(self) -> float:
        if self.total_gb <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (self.free_gb / self.total_gb)))


def free_space(path: Any = "/") -> FreeSpace:
    """Free and total gigabytes on the volume holding `path`."""

    target = Path(str(path or "/")).expanduser()
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(str(target))
    return FreeSpace(
        path=str(target),
        free_gb=usage.free / float(1024**3),
        total_gb=usage.total / float(1024**3),
    )


def ensure_headroom_for(
    estimated_bytes: int,
    *,
    purpose: str,
    path: Any = "/",
    floor_gb: float = DEFAULT_FLOOR_GB,
) -> None:
    """Raise DiskBudgetRefusal when this write would breach the floor.

    The estimate does not have to be exact. What matters is that a 17GB artifact
    cannot be started with 19GB free and no one asking.
    """

    space = free_space(path)
    needed_gb = max(0.0, float(estimated_bytes)) / float(1024**3)
    remaining_after = space.free_gb - needed_gb
    if remaining_after >= floor_gb:
        return
    refusal = DiskBudgetRefusal(
        f"{purpose} needs {needed_gb:.1f}GB and would leave {remaining_after:.1f}GB "
        f"on {space.path}, below the {floor_gb:.1f}GB floor "
        f"(free now {space.free_gb:.1f}GB of {space.total_gb:.1f}GB)"
    )
    record_degradation(
        "disk_budget",
        refusal,
        action="declined a large artifact write to protect the volume",
    )
    raise refusal


def _is_git_tracked(entry: Path) -> bool:
    """True when git knows about this path, in which case it is not ours to remove."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(entry)],
            cwd=str(entry.parent),
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        # Cannot prove it is untracked, so treat it as tracked and leave it.
        return True
    return result.returncode == 0


def prune_superseded_artifacts(
    root: Any,
    *,
    keep: int = DEFAULT_KEEP,
    protect: tuple[str, ...] = (),
    dry_run: bool = False,
) -> list[tuple[str, float]]:
    """Remove the oldest generations under `root`, keeping the newest `keep`.

    Returns the (name, gigabytes) removed, newest-kept-first ordering applied by
    modification time. `protect` names entries that are never removed whatever
    their age — the active model, the one a config points at.
    """

    base = Path(str(root or "")).expanduser()
    if not base.is_dir():
        return []
    protected = {str(name) for name in protect}
    entries = [
        entry
        for entry in base.iterdir()
        if entry.is_dir() and entry.name not in protected
    ]
    if len(entries) <= max(0, int(keep)):
        return []
    entries.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    removed: list[tuple[str, float]] = []
    for entry in entries[max(0, int(keep)) :]:
        if _is_git_tracked(entry):
            continue
        try:
            size_gb = sum(
                item.stat().st_size for item in entry.rglob("*") if item.is_file()
            ) / float(1024**3)
        except OSError:
            size_gb = 0.0
        if dry_run:
            removed.append((entry.name, size_gb))
            continue
        try:
            shutil.rmtree(entry)
        except OSError as exc:
            record_degradation(
                "disk_budget", exc, action=f"could not prune {entry.name}"
            )
            continue
        removed.append((entry.name, size_gb))
    return removed
