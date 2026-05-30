import os


def test_instance_lock_writes_identity_sidecar(tmp_path, monkeypatch):
    from core.utils import singleton as singleton_module

    singleton_module.release_instance_lock()
    monkeypatch.setenv("HOME", str(tmp_path))

    singleton_module.acquire_instance_lock("unit-test")
    try:
        assert singleton_module.read_instance_lock_pid("unit-test") == os.getpid()
        metadata = singleton_module.read_instance_lock_metadata("unit-test")
        assert metadata["schema"] == "aura.instance_lock.v1"
        assert metadata["lock_name"] == "unit-test"
        assert metadata["pid"] == os.getpid()
        assert metadata["cwd"]
    finally:
        singleton_module.release_instance_lock()


def test_stop_aura_rejects_stale_lock_identity():
    import psutil

    from aura_main import PROJECT_ROOT, _lock_pid_matches_aura_runtime

    proc = psutil.Process(os.getpid())
    stale_metadata = {
        "pid": os.getpid(),
        "create_time": proc.create_time() - 60.0,
        "cwd": str(PROJECT_ROOT),
    }

    ok, reason = _lock_pid_matches_aura_runtime(os.getpid(), stale_metadata)

    assert ok is False
    assert reason == "pid_reused_or_stale"


def test_stop_aura_accepts_current_lock_identity():
    import psutil

    from aura_main import PROJECT_ROOT, _lock_pid_matches_aura_runtime

    proc = psutil.Process(os.getpid())
    metadata = {
        "pid": os.getpid(),
        "create_time": proc.create_time(),
        "cwd": str(PROJECT_ROOT),
    }

    ok, reason = _lock_pid_matches_aura_runtime(os.getpid(), metadata)

    assert ok is True
    assert reason == "metadata_identity_verified"
