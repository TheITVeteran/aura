from __future__ import annotations

import json
from pathlib import Path

from core.runtime import launch_provenance


def _app_contract(tmp_path: Path) -> tuple[Path, Path, dict[str, str], dict[str, object]]:
    app = tmp_path / "Aura.app"
    executable = app / "Contents" / "MacOS" / "aura-launcher"
    manifest_path = app / "Contents" / "Resources" / "aura-launch-provenance.json"
    executable.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    executable.write_text("launcher", encoding="utf-8")
    manifest: dict[str, object] = {
        "schema": launch_provenance.LAUNCH_PROVENANCE_SCHEMA,
        "source_root": str(tmp_path.resolve()),
        "commit_sha": "a" * 40,
        "branch": "main",
        "workspace_state_sha256": "b" * 64,
        "bundle_identifier": launch_provenance.EXPECTED_BUNDLE_ID,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    env = {
        "AURA_LAUNCHED_FROM_APP": "1",
        "AURA_LAUNCH_MANIFEST_PATH": str(manifest_path),
        "AURA_LAUNCH_APP_EXECUTABLE": str(executable),
        "AURA_LAUNCH_EXPECTED_ROOT": str(tmp_path.resolve()),
        "AURA_LAUNCH_EXPECTED_COMMIT": "a" * 40,
        "AURA_LAUNCH_EXPECTED_BRANCH": "main",
        "AURA_LAUNCH_EXPECTED_WORKSPACE_SHA256": "b" * 64,
        "AURA_LAUNCH_BUNDLE_ID": launch_provenance.EXPECTED_BUNDLE_ID,
    }
    return executable, manifest_path, env, manifest


def _stub_source(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(
        launch_provenance,
        "_git_identity",
        lambda _root: {
            "source_root": str(root.resolve()),
            "commit_sha": "a" * 40,
            "branch": "main",
        },
    )
    monkeypatch.setattr(
        launch_provenance,
        "_workspace_state",
        lambda _root, *, commit_sha: {
            "workspace_state_sha256": "b" * 64,
            "source_dirty": False,
            "source_change_count": 0,
            "source_changed_paths": [],
            "source_changed_paths_truncated": False,
        },
    )


def test_signed_app_source_preflight_accepts_exact_manifest(monkeypatch, tmp_path):
    _executable, _manifest_path, env, _manifest = _app_contract(tmp_path)
    _stub_source(monkeypatch, tmp_path)

    result = launch_provenance.validate_launch_source(tmp_path, env=env)

    assert result["required"] is True
    assert result["source_verified"] is True
    assert result["issues"] == []
    assert result["actual"]["commit_sha"] == "a" * 40


def test_signed_app_source_preflight_rejects_commit_drift(monkeypatch, tmp_path):
    _executable, _manifest_path, env, _manifest = _app_contract(tmp_path)
    _stub_source(monkeypatch, tmp_path)
    env["AURA_LAUNCH_EXPECTED_COMMIT"] = "c" * 40

    result = launch_provenance.validate_launch_source(tmp_path, env=env)

    assert result["source_verified"] is False
    assert "commit_sha_mismatch" in result["issues"]
    assert "manifest_commit_sha_mismatch" in result["issues"]


def test_signed_app_source_preflight_rejects_dirty_workspace_drift(monkeypatch, tmp_path):
    _executable, _manifest_path, env, _manifest = _app_contract(tmp_path)
    _stub_source(monkeypatch, tmp_path)
    env["AURA_LAUNCH_EXPECTED_WORKSPACE_SHA256"] = "d" * 64

    result = launch_provenance.validate_launch_source(tmp_path, env=env)

    assert result["source_verified"] is False
    assert "workspace_state_sha256_mismatch" in result["issues"]
    assert "manifest_workspace_state_sha256_mismatch" in result["issues"]


def test_signed_app_source_preflight_rejects_manifest_outside_bundle(monkeypatch, tmp_path):
    _executable, _manifest_path, env, manifest = _app_contract(tmp_path)
    _stub_source(monkeypatch, tmp_path)
    detached_manifest = tmp_path / "detached.json"
    detached_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    env["AURA_LAUNCH_MANIFEST_PATH"] = str(detached_manifest)

    result = launch_provenance.validate_launch_source(tmp_path, env=env)

    assert result["source_verified"] is False
    assert "manifest_outside_app_bundle" in result["issues"]


def test_runtime_provenance_requires_resident_stably_signed_strict_bundle(
    monkeypatch,
    tmp_path,
):
    executable, _manifest_path, env, _manifest = _app_contract(tmp_path)
    _stub_source(monkeypatch, tmp_path)
    from core.security import native_desktop_bridge

    monkeypatch.setattr(
        native_desktop_bridge,
        "native_desktop_bridge_identity",
        lambda *, executable: {
            "bridge_executable": str(executable),
            "resident_running": True,
            "code_signature": {
                "available": True,
                "stable_tcc_identity": True,
                "identifier": launch_provenance.EXPECTED_BUNDLE_ID,
            },
        },
    )
    monkeypatch.setattr(
        launch_provenance,
        "_strict_bundle_verification",
        lambda _executable: {"ok": True, "bundle_path": str(executable.parents[2])},
    )

    result = launch_provenance.collect_runtime_launch_provenance(tmp_path, env=env)

    assert result["app_executable"] == str(executable)
    assert result["source_verified"] is True
    assert result["verified"] is True
    assert result["issues"] == []


def test_runtime_provenance_rejects_orphaned_app(monkeypatch, tmp_path):
    _executable, _manifest_path, env, _manifest = _app_contract(tmp_path)
    _stub_source(monkeypatch, tmp_path)
    from core.security import native_desktop_bridge

    monkeypatch.setattr(
        native_desktop_bridge,
        "native_desktop_bridge_identity",
        lambda *, executable: {
            "bridge_executable": str(executable),
            "resident_running": False,
            "code_signature": {
                "available": True,
                "stable_tcc_identity": True,
                "identifier": launch_provenance.EXPECTED_BUNDLE_ID,
            },
        },
    )
    monkeypatch.setattr(
        launch_provenance,
        "_strict_bundle_verification",
        lambda _executable: {"ok": True},
    )

    result = launch_provenance.collect_runtime_launch_provenance(tmp_path, env=env)

    assert result["verified"] is False
    assert "resident_app_not_running" in result["issues"]


def test_boot_health_fails_closed_on_required_launch_provenance(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(
        launch_provenance,
        "collect_runtime_launch_provenance",
        lambda _root: {
            "required": True,
            "verified": False,
            "issues": ["commit_sha_mismatch"],
        },
    )

    payload, status = system_routes._attach_launch_provenance_contract(
        {
            "ready": True,
            "launcher_ready": True,
            "system_ready": True,
            "checks": {},
            "blockers": [],
        },
        200,
    )

    assert status == 503
    assert payload["ready"] is False
    assert payload["checks"]["launch_provenance"] is False
    assert payload["blockers"] == ["launch_provenance"]


def test_boot_health_keeps_direct_runtime_semantics(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(
        launch_provenance,
        "collect_runtime_launch_provenance",
        lambda _root: {"required": False, "verified": False, "launch_mode": "direct"},
    )

    payload, status = system_routes._attach_launch_provenance_contract(
        {"ready": True, "checks": {}, "blockers": []},
        200,
    )

    assert status == 200
    assert payload["ready"] is True
    assert payload["checks"]["launch_provenance"] is True


def test_boot_health_fallback_never_runs_blocking_provenance_probe(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    monkeypatch.setattr(
        launch_provenance,
        "collect_runtime_launch_provenance",
        lambda _root: (_ for _ in ()).throw(AssertionError("blocking probe called")),
    )

    fallback = system_routes._fallback_launch_provenance(
        {"required": True, "verified": True, "issues": []}
    )
    payload, status = system_routes._attach_launch_provenance_contract(
        {"ready": True, "checks": {}, "blockers": []},
        200,
        provenance=fallback,
    )

    assert status == 503
    assert payload["ready"] is False
    assert payload["launch_provenance"]["verified"] is False
    assert "launch_provenance_live_refresh_unavailable" in payload["launch_provenance"]["issues"]
