import json
import os
import tempfile
from pathlib import Path

from core.reaper import (
    LEGACY_REAPER_MANIFEST,
    ReaperManifest,
    _execute_cleanup,
    _pid_cleanup_authorized,
    resolve_reaper_manifest_path,
)


def test_reaper_refuses_bare_legacy_pid_by_default(monkeypatch):
    monkeypatch.delenv("AURA_REAPER_ALLOW_LEGACY_PID_CLEANUP", raising=False)

    authorized, pid, reason = _pid_cleanup_authorized(os.getpid())

    assert authorized is False
    assert pid == os.getpid()
    assert reason == "legacy_pid_without_identity"


def test_reaper_authorizes_current_pid_only_with_identity_metadata(tmp_path):
    manifest = ReaperManifest(tmp_path / "manifest.json")
    manifest.register_pid(os.getpid())

    entry = manifest._data["child_pid_records"][0]
    authorized, pid, reason = _pid_cleanup_authorized(entry)

    assert authorized is True
    assert pid == os.getpid()
    assert reason == "identity_verified"


def test_reaper_rejects_reused_or_stale_pid_metadata(tmp_path):
    manifest = ReaperManifest(tmp_path / "manifest.json")
    manifest.register_pid(os.getpid())
    entry = dict(manifest._data["child_pid_records"][0])
    entry["create_time"] = float(entry["create_time"]) - 1000.0

    authorized, pid, reason = _pid_cleanup_authorized(entry)

    assert authorized is False
    assert pid == os.getpid()
    assert reason == "pid_reused_or_stale"


def test_reaper_cleanup_does_not_signal_legacy_manifest_pid(tmp_path, monkeypatch):
    manifest_path = tmp_path / "legacy.json"
    manifest_path.write_text(json.dumps({"child_pids": [os.getpid()], "shm_names": []}))
    manifest = ReaperManifest(manifest_path)
    signaled = []

    def fake_kill(pid, sig):
        signaled.append((pid, sig))
        raise AssertionError("legacy PID should not be signaled")

    monkeypatch.setattr("core.reaper.os.kill", fake_kill)

    summary = _execute_cleanup(manifest)

    assert signaled == []
    assert summary["skipped_pid_details"] == [
        {"pid": os.getpid(), "reason": "legacy_pid_without_identity"}
    ]
    assert not manifest_path.exists()


def test_legacy_reaper_env_resolves_to_unique_runtime_manifest(monkeypatch):
    monkeypatch.setenv("AURA_REAPER_MANIFEST", str(LEGACY_REAPER_MANIFEST))
    monkeypatch.setenv("AURA_RUNTIME_ID", "test-runtime")

    resolved = resolve_reaper_manifest_path()

    assert resolved != Path(tempfile.gettempdir()) / "aura_reaper_manifest.json"
    assert resolved.name == "manifest-test-runtime.json"
