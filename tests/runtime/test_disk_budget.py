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


def test_free_space_uses_the_canonical_resource_observer(tmp_path: Path) -> None:
    from core.runtime.resource_observation import (
        SimulatedResourceObserver,
        resource_observer_scope,
    )

    gib = 1024**3
    observer = SimulatedResourceObserver(
        disk_total_bytes=200 * gib,
        disk_free_bytes=50 * gib,
    )

    with resource_observer_scope(observer):
        space = free_space(tmp_path)

    assert space.total_gb == pytest.approx(200.0)
    assert space.free_gb == pytest.approx(50.0)
    assert space.percent == pytest.approx(75.0)


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


# ───────── a recursive delete belongs behind the write gateway


def test_pruning_goes_through_the_write_gateway_not_shutil():
    """`prune_superseded_artifacts` removes whole artifact DIRECTORIES.

    CLAUDE.md is explicit that every consequential file write goes through
    `file_write_gateway`, and a recursive delete is the most consequential
    of them. This called `shutil.rmtree` directly, which the governance
    lint reported as new `raw_file_mutation` debt in `core/runtime`.
    """
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "core" / "runtime" / "disk_budget.py"
    ).read_text("utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "rmtree":
            raise AssertionError(
                "disk_budget calls shutil.rmtree directly again; use "
                "get_file_write_gateway().delete_path(recursive=True)"
            )

    assert "delete_path(" in source


def test_the_git_probe_goes_through_the_subprocess_gateway():
    """A raw `subprocess.run` in core/runtime bypasses every shutdown,
    privilege and desktop-safety check the gateway performs."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "core" / "runtime" / "disk_budget.py"
    ).read_text("utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "run"
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
        ):
            raise AssertionError(
                "disk_budget calls subprocess.run directly again; use "
                "get_subprocess_gateway().run(..., read_only=True)"
            )

    assert "get_subprocess_gateway()" in source


def test_the_gateway_has_a_sync_recursive_delete():
    """The reason the raw call existed: `delete_file` refuses directories
    and `delete_path_async` is a coroutine, so a synchronous caller that
    needed to remove a tree had no governed option."""
    from core.runtime.file_write_gateway import get_file_write_gateway

    gateway = get_file_write_gateway()

    assert hasattr(gateway, "delete_path")


def test_the_sync_delete_still_insists_on_recursive_for_a_directory(tmp_path):
    """"Remove this file" and "remove everything under here" must not be
    the same call."""
    import pytest

    from core.runtime.file_write_gateway import get_file_write_gateway

    directory = tmp_path / "tree"
    (directory / "inner").mkdir(parents=True)

    with pytest.raises(IsADirectoryError):
        get_file_write_gateway().delete_path(directory, source="test")

    assert directory.exists()


def test_the_sync_delete_removes_a_tree_when_asked(tmp_path):
    from core.runtime.file_write_gateway import get_file_write_gateway

    directory = tmp_path / "tree"
    (directory / "inner").mkdir(parents=True)
    (directory / "inner" / "f.txt").write_text("x", encoding="utf-8")

    removed = get_file_write_gateway().delete_path(
        directory, recursive=True, source="test"
    )

    assert removed is True
    assert not directory.exists()


def test_the_sync_delete_reports_an_absent_path_rather_than_raising(tmp_path):
    from core.runtime.file_write_gateway import get_file_write_gateway

    assert (
        get_file_write_gateway().delete_path(tmp_path / "nope", source="test") is False
    )


def test_owned_readonly_tree_delete_handles_immutable_artifacts(tmp_path):
    import os

    from core.runtime.file_write_gateway import get_file_write_gateway

    generation = tmp_path / "generation"
    generation.mkdir(mode=0o700)
    weights = generation / "bundle.safetensors"
    weights.write_bytes(b"weights")
    os.chmod(weights, 0o400)
    os.chmod(generation, 0o500)

    assert get_file_write_gateway().delete_owned_readonly_tree(
        generation,
        source="test.checkpoint_retention",
    )
    assert not generation.exists()


def test_owned_readonly_tree_delete_rejects_symlinks(tmp_path):
    import os

    from core.runtime.file_write_gateway import (
        FileWriteTransactionError,
        get_file_write_gateway,
    )

    generation = tmp_path / "generation"
    generation.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.write_text("keep", encoding="utf-8")
    (generation / "linked").symlink_to(outside)
    os.chmod(generation, 0o500)

    with pytest.raises(FileWriteTransactionError, match="custody differs"):
        get_file_write_gateway().delete_owned_readonly_tree(
            generation,
            source="test.checkpoint_retention",
        )

    assert outside.read_text(encoding="utf-8") == "keep"
    assert generation.exists()
    os.chmod(generation, 0o700)
    (generation / "linked").unlink()


def test_owned_readonly_tree_delete_refuses_a_replaced_planned_inode(tmp_path):
    import os

    from core.runtime.file_write_gateway import (
        FileWriteTransactionError,
        get_file_write_gateway,
    )

    generation = tmp_path / "generation"
    generation.mkdir(mode=0o700)
    (generation / "old").write_bytes(b"old")
    os.chmod(generation / "old", 0o400)
    os.chmod(generation, 0o500)
    planned = generation.lstat()

    original = tmp_path / "original"
    generation.rename(original)
    generation.mkdir(mode=0o700)
    (generation / "replacement").write_bytes(b"keep")
    os.chmod(generation / "replacement", 0o400)
    os.chmod(generation, 0o500)

    with pytest.raises(FileWriteTransactionError, match="changed before quarantine"):
        get_file_write_gateway().delete_owned_readonly_tree(
            generation,
            source="test.checkpoint_retention",
            expected_device=planned.st_dev,
            expected_inode=planned.st_ino,
            expected_mtime_ns=planned.st_mtime_ns,
        )

    assert (generation / "replacement").read_bytes() == b"keep"
    assert (original / "old").read_bytes() == b"old"
