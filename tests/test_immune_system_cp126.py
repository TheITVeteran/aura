"""Legacy immune system: an unauthenticated overwrite of executable core code."""
from __future__ import annotations

import asyncio
import hashlib

import pytest

import core.adaptation.immune_system as ims

pytestmark = pytest.mark.unit


def _snapshot(directory, name="kernel.py", *, body="x = 1\n", manifest=True,
              digest=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body)
    if manifest:
        (directory / f"{name}.sha256").write_text(
            digest or hashlib.sha256(path.read_bytes()).hexdigest()
        )
    return path


def _system(tmp_path, monkeypatch, *, approved=True):
    system = ims.ImmuneSystem() if hasattr(ims, "ImmuneSystem") else ims.get_immune_system()
    system.data_dir = tmp_path / "backups"
    system.data_dir.mkdir(parents=True, exist_ok=True)

    class _Will:
        def decide(self, **kwargs):
            return type("D", (), {"is_approved": lambda self: approved})()

    monkeypatch.setattr("core.will.get_will", lambda: _Will())
    return system


# ── containment ────────────────────────────────────────────────────────────


def test_sibling_directory_cannot_masquerade_as_contained(tmp_path, monkeypatch):
    """The old check was startswith(str(base_dir)), which accepts any SIBLING
    whose name merely begins with the base — 'data/backups_evil' passes a
    'data/backups' prefix test."""
    system = _system(tmp_path, monkeypatch)
    evil = _snapshot(tmp_path / "backups_evil")

    assert asyncio.run(system.initiate_rollback(str(evil))) is False


def test_outside_paths_are_refused(tmp_path, monkeypatch):
    system = _system(tmp_path, monkeypatch)
    outside = _snapshot(tmp_path / "elsewhere")

    assert asyncio.run(system.initiate_rollback(str(outside))) is False


def test_missing_snapshot_is_refused(tmp_path, monkeypatch):
    system = _system(tmp_path, monkeypatch)

    assert asyncio.run(
        system.initiate_rollback(str(system.data_dir / "nope.py"))
    ) is False


# ── integrity ──────────────────────────────────────────────────────────────


def test_unsigned_snapshot_is_refused(tmp_path, monkeypatch):
    """Anything that landed in the backups directory became running code."""
    system = _system(tmp_path, monkeypatch)
    unsigned = _snapshot(system.data_dir, manifest=False)

    assert asyncio.run(system.initiate_rollback(str(unsigned))) is False


def test_digest_mismatch_is_refused(tmp_path, monkeypatch):
    system = _system(tmp_path, monkeypatch)
    tampered = _snapshot(system.data_dir, digest="0" * 64)

    assert asyncio.run(system.initiate_rollback(str(tampered))) is False


def test_unparseable_python_is_refused(tmp_path, monkeypatch):
    """Restoring it would leave the kernel unimportable — a rollback that
    bricks the thing it was rescuing."""
    system = _system(tmp_path, monkeypatch)
    broken = _snapshot(system.data_dir, body="def (((:\n")

    assert asyncio.run(system.initiate_rollback(str(broken))) is False


# ── authority ──────────────────────────────────────────────────────────────


def test_rollback_without_approval_is_refused(tmp_path, monkeypatch):
    """Overwriting core code had no governance decision at all."""
    system = _system(tmp_path, monkeypatch, approved=False)
    good = _snapshot(system.data_dir)

    assert asyncio.run(system.initiate_rollback(str(good))) is False


def test_rollback_fails_closed_when_governance_is_unreachable(tmp_path, monkeypatch):
    """An emergency is not authority."""
    system = _system(tmp_path, monkeypatch)
    good = _snapshot(system.data_dir)

    def _boom():
        raise RuntimeError("will unavailable")

    monkeypatch.setattr("core.will.get_will", _boom)

    assert asyncio.run(system.initiate_rollback(str(good))) is False


# ── the happy path must still work, and be undoable ────────────────────────


def test_an_approved_verified_contained_rollback_proceeds(tmp_path, monkeypatch):
    system = _system(tmp_path, monkeypatch)
    good = _snapshot(system.data_dir, body="KERNEL = 'restored'\n")

    target = tmp_path / "cognitive_kernel.py"
    target.write_text("KERNEL = 'broken'\n")
    monkeypatch.setattr(ims, "Path", ims.Path)
    original_path = ims.Path

    class _PathShim(original_path):
        def __new__(cls, *args, **kwargs):
            if args and str(args[0]) == "core/cognition/cognitive_kernel.py":
                return original_path(target)
            return original_path.__new__(cls, *args, **kwargs)

    monkeypatch.setattr(ims, "Path", _PathShim)

    assert asyncio.run(system.initiate_rollback(str(good))) is True
    assert target.read_text() == "KERNEL = 'restored'\n"
    # The overwritten file was preserved: an emergency restore that cannot
    # itself be undone is a one-way door.
    assert list(target.parent.glob("*.pre_rollback.*.py"))
