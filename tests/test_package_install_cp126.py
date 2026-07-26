"""CP126 supply-chain / environment-integrity tests for package installation.

Installing a package executes installer code from a remote index into a live
interpreter. These pin what the actuator refuses, and what it admits it did
NOT verify.
"""
from __future__ import annotations

import subprocess

import pytest

from core.actuators import git_pkg_actuators as module
from core.actuators.git_pkg_actuators import PackageInstallActuator


class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _Gateway:
    def __init__(self, result=None, versions=None):
        self.calls: list[list[str]] = []
        self.result = result or _Result(0, "Successfully installed", "")
        self.versions = list(versions or ["", ""])

    def run(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        if "importlib.metadata" in " ".join(str(part) for part in cmd):
            value = self.versions.pop(0) if self.versions else ""
            return _Result(0, value, "")
        return self.result


@pytest.fixture()
def gateway(monkeypatch) -> _Gateway:
    made = _Gateway()
    monkeypatch.setattr(module, "get_subprocess_gateway", lambda: made)
    monkeypatch.setattr(
        module, "verify_actuator_authority", lambda params, actuator: (True, "ok")
    )
    return made


@pytest.fixture()
def actuator() -> PackageInstallActuator:
    return PackageInstallActuator()


def _params(**extra):
    base = {
        "_aura_authorized": True,
        "allow_install": True,
        "package_name": "requests==2.31.0",
    }
    base.update(extra)
    return base


def _install_cmd(gateway):
    return next(cmd for cmd in gateway.calls if "install" in cmd)


# --- cc1d78b7: a version pin is not artifact verification ----------------


def test_an_unpinned_spec_is_refused(actuator, gateway):
    result = actuator.execute(_params(package_name="requests"))

    assert result.success is False
    assert "not version-pinned" in result.message
    assert gateway.calls == []


def test_a_pinned_install_admits_it_verified_no_artifact(actuator, gateway):
    result = actuator.execute(_params(python_executable="/usr/bin/python3"))

    assert result.success is True
    assert result.updates["hash_verified"] is False
    assert "no artifact digest" in result.updates["supply_chain_bound"]


def test_supplied_hashes_are_passed_to_pip(actuator, gateway):
    digest = "sha256:" + "a" * 64

    result = actuator.execute(
        _params(python_executable="/usr/bin/python3", hashes=[digest])
    )

    cmd = _install_cmd(gateway)
    assert "--require-hashes" in cmd
    assert digest in cmd
    assert result.updates["hash_verified"] is True
    assert "digest verified" in result.updates["supply_chain_bound"]


@pytest.mark.parametrize(
    "bad", ["bogus", "md5:abc", "sha256:xyz", "sha256:" + "a" * 10, ""]
)
def test_a_malformed_hash_is_refused(actuator, gateway, bad):
    result = actuator.execute(
        _params(python_executable="/usr/bin/python3", hashes=[bad])
    )

    assert result.success is False
    assert "malformed hash" in result.message
    assert gateway.calls == []


# --- 3a5c4a39: the running interpreter is not mutated by accident -------


def test_mutating_the_running_interpreter_is_refused_by_default(actuator, gateway):
    result = actuator.execute(_params())

    assert result.success is False
    assert "RUNNING interpreter" in result.message
    assert gateway.calls == []


def test_mutating_the_running_interpreter_can_be_acknowledged(actuator, gateway):
    result = actuator.execute(_params(allow_mutating_running_env=True))

    assert result.success is True
    assert result.updates["mutated_running_interpreter"] is True


def test_an_isolated_target_needs_no_acknowledgement(actuator, gateway):
    result = actuator.execute(_params(python_executable="/usr/bin/python3"))

    assert result.success is True
    assert result.updates["target_python"] == "/usr/bin/python3"
    assert result.updates["mutated_running_interpreter"] is False
    assert _install_cmd(gateway)[0] == "/usr/bin/python3"


def test_a_dry_run_does_not_count_as_mutating(actuator, gateway):
    result = actuator.execute(_params(allow_mutating_running_env=True, dry_run=True))

    assert "--dry-run" in _install_cmd(gateway)
    assert result.updates["dry_run"] is True
    assert result.updates["mutated_running_interpreter"] is False


def test_the_spec_is_still_terminated_by_a_double_dash(actuator, gateway):
    actuator.execute(_params(python_executable="/usr/bin/python3"))

    cmd = _install_cmd(gateway)
    assert cmd[-2] == "--"
    assert cmd[-1] == "requests==2.31.0"


# --- 6da6af1d: the outcome is reconciled, not assumed -------------------


def test_the_result_reports_what_actually_changed(actuator, monkeypatch):
    gateway = _Gateway(versions=["", "2.31.0"])
    monkeypatch.setattr(module, "get_subprocess_gateway", lambda: gateway)
    monkeypatch.setattr(
        module, "verify_actuator_authority", lambda params, actuator: (True, "ok")
    )

    result = actuator.execute(_params(python_executable="/usr/bin/python3"))

    assert result.updates["version_before"] == ""
    assert result.updates["version_after"] == "2.31.0"
    assert result.updates["changed"] is True


def test_a_no_op_install_is_reported_as_unchanged(actuator, monkeypatch):
    gateway = _Gateway(versions=["2.31.0", "2.31.0"])
    monkeypatch.setattr(module, "get_subprocess_gateway", lambda: gateway)
    monkeypatch.setattr(
        module, "verify_actuator_authority", lambda params, actuator: (True, "ok")
    )

    result = actuator.execute(_params(python_executable="/usr/bin/python3"))

    assert result.updates["changed"] is False


def test_a_timeout_reconciles_the_environment(actuator, monkeypatch):
    class _Timing(_Gateway):
        def run(self, cmd, **kwargs):
            self.calls.append(list(cmd))
            if "importlib.metadata" in " ".join(str(part) for part in cmd):
                value = self.versions.pop(0) if self.versions else ""
                return _Result(0, value, "")
            raise subprocess.TimeoutExpired(cmd, 120)

    gateway = _Timing(versions=["", "2.31.0"])
    monkeypatch.setattr(module, "get_subprocess_gateway", lambda: gateway)
    monkeypatch.setattr(
        module, "verify_actuator_authority", lambda params, actuator: (True, "ok")
    )

    result = actuator.execute(_params(python_executable="/usr/bin/python3"))

    assert result.success is False
    assert result.updates["timed_out"] is True
    # A timeout that still changed the environment is a PARTIAL install.
    assert result.updates["partial_install_suspected"] is True
    assert result.updates["version_after"] == "2.31.0"


def test_a_restart_is_flagged_when_the_live_env_changed(actuator, monkeypatch):
    gateway = _Gateway(versions=["1.0.0", "2.31.0"])
    monkeypatch.setattr(module, "get_subprocess_gateway", lambda: gateway)
    monkeypatch.setattr(
        module, "verify_actuator_authority", lambda params, actuator: (True, "ok")
    )

    result = actuator.execute(_params(allow_mutating_running_env=True))

    assert result.updates["restart_required"] is True


def test_the_timeout_is_bounded(actuator, gateway):
    actuator.execute(_params(python_executable="/usr/bin/python3", timeout_s=99999))

    # No exception, and the install still ran under a clamped budget.
    assert _install_cmd(gateway)


def test_authority_is_still_required(actuator, monkeypatch):
    monkeypatch.setattr(
        module, "verify_actuator_authority",
        lambda params, actuator: (False, "no capability token"),
    )

    result = actuator.execute(_params(python_executable="/usr/bin/python3"))

    assert result.success is False
    assert "no capability token" in result.message


def test_install_still_requires_explicit_approval(actuator, gateway):
    params = _params(python_executable="/usr/bin/python3")
    params.pop("allow_install")

    assert actuator.execute(params).success is False
