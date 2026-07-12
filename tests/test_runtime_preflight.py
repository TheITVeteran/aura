"""Contract tests for tools/runtime_preflight.py.

The preflight is the daily-driver "will the app open cleanly" gate; every
check must be honest on the failure paths it exists for (missing .env,
stale model manifest, drifted lockfile, full disk), and the tool must stay
importable with stdlib only.
"""

from __future__ import annotations

import json
import socket
import sys
from collections import namedtuple
from pathlib import Path

import pytest

from core.runtime.subprocess_gateway import get_subprocess_gateway
from tools.runtime_preflight import (
    FAIL,
    INFO,
    OK,
    WARN,
    Check,
    check_disk,
    check_env_file,
    check_lock_drift,
    check_logs_writable,
    check_models,
    check_port,
    check_python,
    check_ram,
    main,
)

_Usage = namedtuple("_Usage", "total used free")


class TestDisk:
    def test_ok_above_thresholds(self, tmp_path):
        c = check_disk(tmp_path, usage_fn=lambda p: _Usage(100e9, 10e9, 90e9))
        assert c.status == OK

    def test_warn_below_comfort(self, tmp_path):
        c = check_disk(tmp_path, usage_fn=lambda p: _Usage(100e9, 80e9, 20e9))
        assert c.status == WARN

    def test_fail_below_floor(self, tmp_path):
        c = check_disk(tmp_path, usage_fn=lambda p: _Usage(100e9, 95e9, 5e9))
        assert c.status == FAIL

    def test_unstatable_path_fails(self, tmp_path):
        def boom(_p):
            raise OSError("gone")

        assert check_disk(tmp_path, usage_fn=boom).status == FAIL

    def test_canonical_observer_path_records_provenance(self, tmp_path):
        from core.runtime.resource_observation import SimulatedResourceObserver

        observer = SimulatedResourceObserver(
            scenario_id="runtime-preflight-disk",
            disk_total_bytes=100_000_000_000,
            disk_free_bytes=90_000_000_000,
        )
        check = check_disk(tmp_path, observer=observer)

        assert check.status == OK
        assert "source=simulated" in check.detail


class TestRam:
    def test_warns_on_low_canonical_observation(self):
        from core.runtime.resource_observation import SimulatedResourceObserver

        observer = SimulatedResourceObserver(
            scenario_id="runtime-preflight-low-memory",
            total_memory_bytes=64_000_000_000,
            available_memory_bytes=4_000_000_000,
        )
        check = check_ram(observer=observer)

        assert check.status == WARN
        assert "source=simulated" in check.detail

    def test_unavailable_observation_fails_visible(self):
        class _UnavailableObserver:
            @staticmethod
            def memory():
                return type(
                    "UnavailableMemory",
                    (),
                    {"available": False, "error": "probe denied"},
                )()

        check = check_ram(observer=_UnavailableObserver())

        assert check.status == FAIL
        assert "probe denied" in check.detail


class TestEnvFile:
    def test_missing_env_warns_with_recovery_pointer(self, tmp_path):
        c = check_env_file(tmp_path)
        assert c.status == WARN
        assert "backups/env" in c.detail

    def test_env_without_token_warns(self, tmp_path):
        (tmp_path / ".env").write_text("OTHER=1\n")
        c = check_env_file(tmp_path)
        assert c.status == WARN
        assert "AURA_API_TOKEN" in c.detail

    def test_env_with_token_ok_and_value_never_printed(self, tmp_path):
        (tmp_path / ".env").write_text("AURA_API_TOKEN=supersecretvalue\n")
        c = check_env_file(tmp_path)
        assert c.status == OK
        assert "supersecretvalue" not in c.detail

    def test_commented_token_does_not_count(self, tmp_path):
        (tmp_path / ".env").write_text("# AURA_API_TOKEN=old\n")
        assert check_env_file(tmp_path).status == WARN


class TestModels:
    def test_missing_models_dir_fails(self, tmp_path):
        assert check_models(tmp_path).status == FAIL

    def test_models_without_manifest_is_info(self, tmp_path):
        (tmp_path / "models").mkdir()
        assert check_models(tmp_path).status == INFO

    def test_manifest_resolving_to_real_artifact_ok(self, tmp_path):
        (tmp_path / "models").mkdir()
        artifact = tmp_path / "fused"
        artifact.mkdir()
        mdir = tmp_path / "training" / "fused-model"
        mdir.mkdir(parents=True)
        (mdir / "active.json").write_text(
            json.dumps({"active_model_path": str(artifact)})
        )
        c = check_models(tmp_path)
        assert c.status == OK

    def test_manifest_pointing_at_missing_artifact_warns(self, tmp_path):
        (tmp_path / "models").mkdir()
        mdir = tmp_path / "training" / "fused-model"
        mdir.mkdir(parents=True)
        (mdir / "active.json").write_text(
            json.dumps({"active_model_path": str(tmp_path / "vanished")})
        )
        c = check_models(tmp_path)
        assert c.status == WARN
        assert "promotion" in c.detail

    def test_corrupt_manifest_warns_not_crashes(self, tmp_path):
        (tmp_path / "models").mkdir()
        mdir = tmp_path / "training" / "fused-model"
        mdir.mkdir(parents=True)
        (mdir / "active.json").write_text("{not json")
        assert check_models(tmp_path).status == WARN


class TestLockDrift:
    def test_no_lockfile_is_info(self, tmp_path):
        assert check_lock_drift(tmp_path / "nope.txt").status == INFO

    def test_matching_pin_is_ok(self, tmp_path):
        import pytest as _pytest

        lock = tmp_path / "lock.txt"
        lock.write_text(f"pytest=={_pytest.__version__}\n")
        assert check_lock_drift(lock).status == OK

    def test_drifted_pin_warns_and_names_package(self, tmp_path):
        lock = tmp_path / "lock.txt"
        lock.write_text("pytest==0.0.1\n")
        c = check_lock_drift(lock)
        assert c.status == WARN
        assert "pytest" in c.detail

    def test_pip_compile_hash_continuations_parse(self, tmp_path):
        import pytest as _pytest

        lock = tmp_path / "lock.txt"
        lock.write_text(
            "# comment\n"
            f"pytest=={_pytest.__version__} \\\n"
            "    --hash=sha256:deadbeef\n"
        )
        assert check_lock_drift(lock).status == OK

    def test_pinned_but_not_installed_is_not_drift(self, tmp_path):
        lock = tmp_path / "lock.txt"
        lock.write_text("definitely-not-installed-pkg==9.9.9\n")
        c = check_lock_drift(lock)
        assert c.status == OK
        assert "not-installed" in c.detail


class TestPortAndLogs:
    def test_bound_port_reports_in_use(self):
        with socket.socket() as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            c = check_port(port=port)
            assert c.status == INFO
            assert "serving" in c.detail

    def test_free_port_reports_free(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        assert check_port(port=port).status == OK

    def test_logs_writable(self, tmp_path):
        c = check_logs_writable(tmp_path / "logs")
        assert c.status == OK
        assert not (tmp_path / "logs" / ".preflight_probe").exists()


class TestPython:
    def test_matches_running_interpreter(self):
        c = check_python(expected=(sys.version_info.major, sys.version_info.minor))
        assert c.status == OK

    def test_mismatch_fails(self):
        assert check_python(expected=(2, 7)).status == FAIL


class TestMain:
    @pytest.fixture()
    def healthy_root(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text("AURA_API_TOKEN=x\n")
        (tmp_path / "models").mkdir()
        monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
        monkeypatch.setattr(
            "tools.runtime_preflight.check_python",
            lambda: Check("python", OK, "supported interpreter supplied by assembly test"),
        )
        return tmp_path

    def _free_port(self) -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def test_json_output_and_exit_zero(self, healthy_root, capsys):
        rc = main(["--root", str(healthy_root), "--json",
                   "--port", str(self._free_port())])
        out = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert out["verdict"] == OK
        names = {c["name"] for c in out["checks"]}
        assert {"python", "disk", "env-file", "models", "lock-drift"} <= names

    def test_strict_promotes_warn_to_fail(self, healthy_root, capsys):
        (healthy_root / ".env").unlink()  # WARN source
        rc = main(["--root", str(healthy_root), "--json", "--strict",
                   "--port", str(self._free_port())])
        assert rc == 1

    def test_missing_models_fails_even_non_strict(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "logs"))
        rc = main(["--root", str(tmp_path), "--json",
                   "--port", str(self._free_port())])
        assert rc == 1

    def test_stdlib_only_import(self):
        """Core runtime dependencies are loaded only when checks execute."""
        code = (
            "import sys; sys.modules['psutil'] = None; "
            "import tools.runtime_preflight"
        )
        proc = get_subprocess_gateway().run(
            [sys.executable, "-c", code],
            timeout=30,
            cwd=str(Path(__file__).resolve().parents[1]),
            offline_tooling=True,
            source="certification_tooling:runtime_preflight.stdlib_import",
        )
        assert proc.returncode == 0, proc.stderr

    def test_direct_script_resolves_canonical_resource_observer(self, healthy_root):
        script = Path(__file__).resolve().parents[1] / "tools" / "runtime_preflight.py"
        proc = get_subprocess_gateway().run(
            [
                sys.executable,
                str(script),
                "--root",
                str(healthy_root),
                "--json",
                "--port",
                str(self._free_port()),
            ],
            timeout=30,
            offline_tooling=True,
            source="certification_tooling:runtime_preflight.direct_script",
        )
        report = json.loads(proc.stdout)
        resource_checks = {
            check["name"]: check for check in report["checks"] if check["name"] in {"disk", "ram"}
        }

        assert set(resource_checks) == {"disk", "ram"}
        assert all(
            "resource observer unavailable" not in check["detail"]
            for check in resource_checks.values()
        ), proc.stderr
        assert all("source=host" in check["detail"] for check in resource_checks.values())
