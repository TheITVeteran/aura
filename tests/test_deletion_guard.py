"""Deletion guard: versioned restore, protected-path gating, deletion-storm freeze."""
from __future__ import annotations

import pytest

from core.security.deletion_guard import DeletionGuard


@pytest.fixture
def guard(tmp_path):
    return DeletionGuard(recycle_dir=tmp_path / "recycle", storm_window_s=10.0, storm_threshold=5)


def test_ordinary_delete_is_allowed_and_recoverable(guard, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("important content")
    d = guard.guard_delete(str(f), now=1000.0)
    assert d.allowed
    assert d.version_id is not None
    # simulate the actual delete, then restore from the guard
    f.unlink()
    restored = guard.restore(d.version_id)
    assert restored is not None
    assert open(restored).read() == "important content"


def test_protected_path_requires_confirmation(guard, tmp_path):
    f = tmp_path / "aura_identity_core.json"   # 'identity' marker → protected
    f.write_text("{}")
    d = guard.guard_delete(str(f), now=1000.0)
    assert not d.allowed
    assert d.requires_confirmation
    assert d.version_id is not None  # still snapshotted, just gated


def test_confirmed_protected_delete_by_owner_allowed(guard, tmp_path):
    f = tmp_path / "memory_store.db"
    f.write_text("x")
    d = guard.guard_delete(str(f), confirmed=True, actor="bryan", now=1000.0)
    assert d.allowed


def test_forced_delete_of_protected_path_by_nonowner_blocked(guard, tmp_path):
    f = tmp_path / "governance_vault.bin"
    f.write_text("x")
    d = guard.guard_delete(str(f), forced=True, confirmed=True, actor="attacker", now=1000.0)
    assert not d.allowed
    assert "forced deletion" in d.reason


def test_deletion_storm_freezes(guard, tmp_path):
    # past the threshold of destructive ops in the window → freeze
    frozen_seen = False
    for i in range(8):
        f = tmp_path / f"f{i}.txt"
        f.write_text("x")
        d = guard.guard_delete(str(f), now=1000.0 + i * 0.1)
        frozen_seen = frozen_seen or d.frozen
    assert frozen_seen, "deletion storm did not trip the freeze"
    # subsequent deletes stay frozen
    f = tmp_path / "after.txt"
    f.write_text("x")
    d = guard.guard_delete(str(f), now=1001.0)
    assert d.frozen and not d.allowed


def test_unfreeze_restores_operation(guard, tmp_path):
    for i in range(8):
        f = tmp_path / f"f{i}.txt"
        f.write_text("x")
        guard.guard_delete(str(f), now=1000.0 + i * 0.1)
    guard.unfreeze()
    f = tmp_path / "ok.txt"
    f.write_text("x")
    d = guard.guard_delete(str(f), now=2000.0)
    assert d.allowed


def test_storm_flags_immune_system(guard, tmp_path, monkeypatch):
    flagged = []
    import core.security.immune_system as im

    class _Stub:
        def assess(self, *a, **k):
            flagged.append(k.get("threat_class"))
    monkeypatch.setattr(im, "get_immune_system", lambda: _Stub())

    for i in range(8):
        f = tmp_path / f"f{i}.txt"
        f.write_text("x")
        guard.guard_delete(str(f), now=1000.0 + i * 0.1)
    assert flagged, "deletion storm did not notify the immune system"
