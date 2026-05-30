import json
import os
import time
from types import SimpleNamespace


def test_request_shutdown_writes_pid_bound_grace_flag(tmp_path, monkeypatch):
    from core.runtime import shutdown_coordinator

    monkeypatch.setenv("HOME", str(tmp_path))
    shutdown_coordinator._shutdown_requested.clear()
    try:
        shutdown_coordinator.request_shutdown("unit_test")
        payload = json.loads(
            (tmp_path / ".aura" / "run" / "grace_exit.flag").read_text(encoding="utf-8")
        )
    finally:
        shutdown_coordinator._shutdown_requested.clear()

    assert payload["schema"] == "aura.shutdown_grace.v1"
    assert payload["pid"] == os.getpid()
    assert payload["reason"] == "unit_test"
    assert payload["created_at_unix"] <= time.time()


def test_supervisor_rejects_stale_or_wrong_pid_grace_flags(tmp_path, monkeypatch):
    from core.resilience.supervisor import SovereignSupervisor

    monkeypatch.setenv("HOME", str(tmp_path))
    grace_file = tmp_path / ".aura" / "run" / "grace_exit.flag"
    grace_file.parent.mkdir(parents=True)
    supervisor = SovereignSupervisor("aura_main.py")
    supervisor.process = SimpleNamespace(pid=1234)

    grace_file.write_text(
        json.dumps({"pid": 9999, "created_at_unix": time.time()}),
        encoding="utf-8",
    )
    assert supervisor._grace_flag_matches_child(grace_file) is False

    grace_file.write_text(
        json.dumps({"pid": 1234, "created_at_unix": time.time() - 3600}),
        encoding="utf-8",
    )
    assert supervisor._grace_flag_matches_child(grace_file) is False

    grace_file.write_text(
        json.dumps({"pid": 1234, "created_at_unix": time.time()}),
        encoding="utf-8",
    )
    assert supervisor._grace_flag_matches_child(grace_file) is True
