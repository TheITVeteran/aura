"""Signed pre-load sentinel barrier contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import tools.unified_intrinsic_preload_barrier as barrier
from tools import run_detached_step as detached
from tools.unified_intrinsic_preload_barrier import (
    UnifiedPreloadBarrierError,
    command_sha256,
    expand_pid_template,
    publish_ready,
    publish_release,
    verify_release,
)


def test_pid_template_binds_replayed_launch_to_new_artifacts() -> None:
    assert expand_pid_template("/tmp/ready-{pid}.json", target_pid=417) == (
        "/tmp/ready-417.json"
    )
    assert expand_pid_template("--marker={pid}", target_pid=417) == "--marker=417"


def test_replayed_wrapper_uses_fresh_pid_scoped_handshake(tmp_path: Path) -> None:
    key_path = _key(tmp_path)
    script = Path(__file__).parents[1] / "tools/unified_intrinsic_preload_barrier.py"
    config_sha256 = "a" * 64
    observed_pids: list[int] = []
    for _attempt in range(2):
        command = [
            sys.executable,
            str(script),
            "--ready",
            str(tmp_path / "ready-{pid}.json"),
            "--release",
            str(tmp_path / "release-{pid}.json"),
            "--key",
            str(key_path),
            "--config-sha256",
            config_sha256,
            "--timeout",
            "10",
            "--",
            sys.executable,
            "-c",
            (
                "import os; from pathlib import Path; "
                f"Path({str(tmp_path / 'child-{pid}.txt')!r}).write_text(str(os.getpid()))"
            ),
        ]
        process = subprocess.Popen(command)
        observed_pids.append(process.pid)
        ready_path = tmp_path / f"ready-{process.pid}.json"
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not ready_path.exists():
            time.sleep(0.01)
        assert ready_path.exists()
        ready = json.loads(ready_path.read_text(encoding="ascii"))
        publish_release(
            tmp_path / f"release-{process.pid}.json",
            ready_path=ready_path,
            key_path=key_path,
            sentinel_pid=os.getpid(),
            sentinel_start_token=detached._process_start_token(os.getpid()),
            sentinel_ring_entry_sha256="b" * 64,
            host_pressure={"available": True, "under_pressure": False},
            expected_target_pid=process.pid,
            expected_target_start_token=ready["target_start_token"],
            expected_command_sha256=ready["command_sha256"],
        )
        assert process.wait(timeout=5.0) == 0
        child_path = tmp_path / f"child-{process.pid}.txt"
        assert child_path.read_text(encoding="utf-8") == str(process.pid)

    assert observed_pids[0] != observed_pids[1]


def _key(tmp_path: Path) -> Path:
    path = tmp_path / "heartbeat.key"
    path.write_bytes(b"k" * 32)
    path.chmod(0o400)
    return path


def test_release_binds_exact_target_sentinel_and_host_pressure(tmp_path: Path) -> None:
    ready_path = tmp_path / "preload-ready.json"
    release_path = tmp_path / "preload-release.json"
    key_path = _key(tmp_path)
    config_sha256 = "a" * 64
    ready, _ready_raw = publish_ready(
        ready_path,
        config_sha256=config_sha256,
        command=("/usr/bin/python3", "trainer.py"),
        target_pid=os.getpid(),
    )
    published = publish_release(
        release_path,
        ready_path=ready_path,
        key_path=key_path,
        sentinel_pid=os.getpid() + 1,
        sentinel_start_token="123.25",
        sentinel_ring_entry_sha256="b" * 64,
        host_pressure={
            "available": True,
            "under_pressure": False,
            "reclaimable_gb": 42.0,
        },
        expected_target_pid=os.getpid(),
        expected_target_start_token=ready["target_start_token"],
        expected_command_sha256=command_sha256(
            ("/usr/bin/python3", "trainer.py")
        ),
    )
    verified = verify_release(
        release_path,
        ready_path=ready_path,
        key_path=key_path,
        config_sha256=config_sha256,
        expected_target_pid=os.getpid(),
        expected_command_sha256=command_sha256(
            ("/usr/bin/python3", "trainer.py")
        ),
    )

    assert verified == published
    assert verified["target_pid"] == os.getpid()
    assert verified["target_start_token"] == ready["target_start_token"]
    assert verified["sentinel_pid"] == os.getpid() + 1
    assert verified["expires_at_unix_ns"] > verified["issued_at_unix_ns"]


def test_release_rejects_ready_record_for_different_command(tmp_path: Path) -> None:
    ready_path = tmp_path / "preload-ready.json"
    release_path = tmp_path / "preload-release.json"
    key_path = _key(tmp_path)
    ready, _ready_raw = publish_ready(
        ready_path,
        config_sha256="a" * 64,
        command=("/usr/bin/python3", "unexpected.py"),
        target_pid=os.getpid(),
    )

    with pytest.raises(UnifiedPreloadBarrierError, match="command"):
        publish_release(
            release_path,
            ready_path=ready_path,
            key_path=key_path,
            sentinel_pid=os.getpid() + 1,
            sentinel_start_token="123.25",
            sentinel_ring_entry_sha256="b" * 64,
            host_pressure={"available": True, "under_pressure": False},
            expected_target_pid=os.getpid(),
            expected_target_start_token=ready["target_start_token"],
            expected_command_sha256=command_sha256(
                ("/usr/bin/python3", "expected.py")
            ),
        )

    assert not release_path.exists()


def test_release_rejects_tampered_pressure_even_with_valid_shape(tmp_path: Path) -> None:
    ready_path = tmp_path / "preload-ready.json"
    release_path = tmp_path / "preload-release.json"
    key_path = _key(tmp_path)
    config_sha256 = "c" * 64
    ready, _ready_raw = publish_ready(
        ready_path,
        config_sha256=config_sha256,
        command=("/usr/bin/python3", "trainer.py"),
        target_pid=os.getpid(),
    )
    publish_release(
        release_path,
        ready_path=ready_path,
        key_path=key_path,
        sentinel_pid=os.getpid() + 1,
        sentinel_start_token="123.25",
        sentinel_ring_entry_sha256="d" * 64,
        host_pressure={"available": True, "under_pressure": False},
        expected_target_pid=os.getpid(),
        expected_target_start_token=ready["target_start_token"],
        expected_command_sha256=command_sha256(("/usr/bin/python3", "trainer.py")),
    )
    document = json.loads(release_path.read_text(encoding="ascii"))
    document["host_pressure"]["under_pressure"] = True
    release_path.chmod(0o600)
    release_path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )

    with pytest.raises(UnifiedPreloadBarrierError, match="differs"):
        verify_release(
            release_path,
            ready_path=ready_path,
            key_path=key_path,
            config_sha256=config_sha256,
            expected_target_pid=os.getpid(),
        )


def test_release_rejects_wrong_process_incarnation(tmp_path: Path) -> None:
    ready_path = tmp_path / "preload-ready.json"
    release_path = tmp_path / "preload-release.json"
    key_path = _key(tmp_path)
    ready, _ready_raw = publish_ready(
        ready_path,
        config_sha256="e" * 64,
        command=("/usr/bin/python3", "trainer.py"),
        target_pid=os.getpid(),
    )

    with pytest.raises(UnifiedPreloadBarrierError, match="incarnation"):
        publish_release(
            release_path,
            ready_path=ready_path,
            key_path=key_path,
            sentinel_pid=os.getpid() + 1,
            sentinel_start_token="123.25",
            sentinel_ring_entry_sha256="f" * 64,
            host_pressure={"available": True, "under_pressure": False},
            expected_target_pid=os.getpid(),
            expected_target_start_token="wrong-incarnation",
            expected_command_sha256=ready["command_sha256"],
        )


def test_release_expires_for_model_load_but_remains_historical_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready_path = tmp_path / "preload-ready.json"
    release_path = tmp_path / "preload-release.json"
    key_path = _key(tmp_path)
    ready, _ready_raw = publish_ready(
        ready_path,
        config_sha256="1" * 64,
        command=("/usr/bin/python3", "trainer.py"),
        target_pid=os.getpid(),
    )
    release = publish_release(
        release_path,
        ready_path=ready_path,
        key_path=key_path,
        sentinel_pid=os.getpid() + 1,
        sentinel_start_token="123.25",
        sentinel_ring_entry_sha256="2" * 64,
        host_pressure={"available": True, "under_pressure": False},
        expected_target_pid=os.getpid(),
        expected_target_start_token=ready["target_start_token"],
        expected_command_sha256=ready["command_sha256"],
    )
    monkeypatch.setattr(
        barrier.time,
        "time_ns",
        lambda: int(release["expires_at_unix_ns"]) + 1,
    )

    with pytest.raises(UnifiedPreloadBarrierError, match="differs"):
        verify_release(
            release_path,
            ready_path=ready_path,
            key_path=key_path,
            config_sha256="1" * 64,
            expected_target_pid=os.getpid(),
        )
    assert verify_release(
        release_path,
        ready_path=ready_path,
        key_path=key_path,
        config_sha256="1" * 64,
        expected_target_pid=os.getpid(),
        require_fresh=False,
    ) == release
