"""Aura keeps her own launcher current.

The installed app is a thin launcher over live source, so the only artifact
that can genuinely go stale is the compiled Swift binary. These pin that it is
detected, rebuilt without being asked, and never swapped underneath a running
process.
"""
from __future__ import annotations

import json

import pytest

from core.runtime import app_bundle_sync as module


def _bundle(tmp_path, *, launcher_sha="a" * 64, with_binary=True):
    bundle = tmp_path / "Aura.app"
    resources = bundle / "Contents" / "Resources"
    macos = bundle / "Contents" / "MacOS"
    resources.mkdir(parents=True)
    macos.mkdir(parents=True)
    (resources / "aura-launch-provenance.json").write_text(
        json.dumps({"launcher_source_sha256": launcher_sha}), encoding="utf-8"
    )
    if with_binary:
        (macos / "aura-launcher").write_text("binary", encoding="utf-8")
    return bundle


def _root(tmp_path, *, launcher_text="swift source"):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "AuraLauncher.swift").write_text(launcher_text, encoding="utf-8")
    build = root / "scripts" / "bundle_app.sh"
    build.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    build.chmod(0o755)
    return root


def _live_sha(root):
    import hashlib

    return hashlib.sha256(
        (root / "scripts" / "AuraLauncher.swift").read_bytes()
    ).hexdigest()


# --- drift detection ----------------------------------------------------


def test_a_launcher_built_from_the_current_source_is_not_stale(tmp_path):
    root = _root(tmp_path)
    bundle = _bundle(tmp_path, launcher_sha=_live_sha(root))

    assert module.launcher_drift(root, bundle)["stale"] is False


def test_a_launcher_built_from_older_source_is_stale(tmp_path):
    root = _root(tmp_path)
    bundle = _bundle(tmp_path, launcher_sha="b" * 64)

    drift = module.launcher_drift(root, bundle)

    assert drift["stale"] is True
    assert drift["launcher_source_sha256"] == _live_sha(root)


def test_a_bundle_with_no_recorded_digest_counts_as_stale(tmp_path):
    """Cannot tell is not current."""
    root = _root(tmp_path)
    bundle = _bundle(tmp_path)
    (bundle / "Contents" / "Resources" / "aura-launch-provenance.json").write_text(
        "{}", encoding="utf-8"
    )

    assert module.launcher_drift(root, bundle)["stale"] is True


def test_a_missing_bundle_is_not_stale(tmp_path):
    root = _root(tmp_path)

    drift = module.launcher_drift(root, tmp_path / "absent.app")

    assert drift["bundle_present"] is False
    assert drift["stale"] is False


# --- sync behaviour -----------------------------------------------------


@pytest.fixture()
def no_running_app(monkeypatch):
    monkeypatch.setattr(module, "bundle_is_running", lambda _bundle: False)


def _capture_build(monkeypatch, returncode=0):
    calls = []

    class _Completed:
        def __init__(self):
            self.returncode = returncode
            self.stdout = "built"
            self.stderr = ""

    class _Gateway:
        def run(self, cmd, **kwargs):
            calls.append({"cmd": list(cmd), **kwargs})
            return _Completed()

    monkeypatch.setattr(module, "get_subprocess_gateway", lambda: _Gateway())
    return calls


def test_a_current_launcher_triggers_no_build(tmp_path, monkeypatch, no_running_app):
    root = _root(tmp_path)
    bundle = _bundle(tmp_path, launcher_sha=_live_sha(root))
    calls = _capture_build(monkeypatch)

    receipt = module.sync_app_bundle(root, resident=bundle)

    assert receipt["action"] == "current"
    assert calls == []


def test_a_stale_launcher_is_rebuilt_and_installed(tmp_path, monkeypatch, no_running_app):
    root = _root(tmp_path)
    bundle = _bundle(tmp_path, launcher_sha="b" * 64)
    calls = _capture_build(monkeypatch)

    receipt = module.sync_app_bundle(root, resident=bundle)

    assert receipt["action"] == "rebuilt_and_installed"
    assert receipt["installed"] is True
    assert calls[0]["env"]["AURA_INSTALL_PATH"] == str(bundle)


def test_a_running_bundle_is_never_replaced_underneath(tmp_path, monkeypatch):
    """The update is built, but takes effect at the next start."""
    root = _root(tmp_path)
    bundle = _bundle(tmp_path, launcher_sha="b" * 64)
    monkeypatch.setattr(module, "bundle_is_running", lambda _bundle: True)
    calls = _capture_build(monkeypatch)

    receipt = module.sync_app_bundle(root, resident=bundle)

    assert receipt["action"] == "rebuilt_staged"
    assert receipt["installed"] is False
    assert "AURA_INSTALL_PATH" not in calls[0]["env"]


def test_a_failed_build_leaves_the_previous_launcher_in_place(
    tmp_path, monkeypatch, no_running_app
):
    root = _root(tmp_path)
    bundle = _bundle(tmp_path, launcher_sha="b" * 64)
    _capture_build(monkeypatch, returncode=1)

    receipt = module.sync_app_bundle(root, resident=bundle)

    assert receipt["action"] == "failed"
    assert receipt["installed"] is False


def test_sync_never_raises_on_the_boot_path(tmp_path, monkeypatch):
    """A launcher that cannot be refreshed must not stop Aura starting."""
    root = _root(tmp_path)
    bundle = _bundle(tmp_path, launcher_sha="b" * 64)
    monkeypatch.setattr(module, "bundle_is_running", lambda _b: False)

    class _Exploding:
        def run(self, *a, **k):
            raise OSError("no exec")

    monkeypatch.setattr(module, "get_subprocess_gateway", lambda: _Exploding())

    receipt = module.sync_app_bundle(root, resident=bundle)

    assert receipt["action"] == "failed"
    assert "OSError" in receipt["reason"]


def test_the_cli_never_fails_a_boot(tmp_path, monkeypatch, capsys):
    root = _root(tmp_path)
    bundle = _bundle(tmp_path, launcher_sha="b" * 64)
    monkeypatch.setattr(module, "bundle_is_running", lambda _b: True)
    _capture_build(monkeypatch, returncode=1)

    code = module.main(["--root", str(root), "--resident", str(bundle)])

    assert code == 0
    assert json.loads(capsys.readouterr().out)["action"] == "failed"


# --- staged install -----------------------------------------------------


def test_a_staged_build_is_installed_when_nothing_is_running(tmp_path, monkeypatch):
    root = _root(tmp_path)
    resident = _bundle(tmp_path, launcher_sha="old" + "0" * 61)
    staged = root / "dist" / "Aura.app"
    staged.parent.mkdir(parents=True)
    import shutil

    shutil.copytree(_bundle(tmp_path / "staged_src", launcher_sha="new" + "0" * 61), staged)
    monkeypatch.setattr(module, "bundle_is_running", lambda _b: False)

    receipt = module.install_staged_bundle(root, resident=resident)

    assert receipt["installed"] is True
    manifest = json.loads(
        (resident / "Contents" / "Resources" / "aura-launch-provenance.json").read_text()
    )
    assert manifest["launcher_source_sha256"].startswith("new")


def test_a_staged_build_is_not_installed_over_a_running_app(tmp_path, monkeypatch):
    root = _root(tmp_path)
    resident = _bundle(tmp_path, launcher_sha="old" + "0" * 61)
    staged = root / "dist" / "Aura.app"
    staged.parent.mkdir(parents=True)
    import shutil

    shutil.copytree(_bundle(tmp_path / "staged_src", launcher_sha="new" + "0" * 61), staged)
    monkeypatch.setattr(module, "bundle_is_running", lambda _b: True)

    receipt = module.install_staged_bundle(root, resident=resident)

    assert receipt["installed"] is False
    assert "running" in receipt["reason"]


def test_nothing_staged_is_a_quiet_no_op(tmp_path, monkeypatch):
    root = _root(tmp_path)
    resident = _bundle(tmp_path)
    monkeypatch.setattr(module, "bundle_is_running", lambda _b: False)

    assert module.install_staged_bundle(root, resident=resident)["installed"] is False


def test_an_identical_staged_build_is_not_reinstalled(tmp_path, monkeypatch):
    root = _root(tmp_path)
    resident = _bundle(tmp_path, launcher_sha="same" + "0" * 60)
    staged = root / "dist" / "Aura.app"
    staged.parent.mkdir(parents=True)
    import shutil

    shutil.copytree(_bundle(tmp_path / "staged_src", launcher_sha="same" + "0" * 60), staged)
    monkeypatch.setattr(module, "bundle_is_running", lambda _b: False)

    receipt = module.install_staged_bundle(root, resident=resident)

    assert receipt["installed"] is False
    assert "already matches" in receipt["reason"]


# --- safety: an unreadable process table must not authorize an install ---


def test_an_unknown_process_state_is_treated_as_running(tmp_path, monkeypatch):
    class _Exploding:
        def run(self, *a, **k):
            raise OSError("pgrep unavailable")

    monkeypatch.setattr(module, "get_subprocess_gateway", lambda: _Exploding())

    assert module.bundle_is_running(tmp_path / "Aura.app") is True
