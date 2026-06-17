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


def test_reaped_orchestrator_lock_is_purged(tmp_path, monkeypatch):
    import aura_main

    monkeypatch.setenv("HOME", str(tmp_path))
    lock_dir = tmp_path / ".aura" / "locks"
    lock_dir.mkdir(parents=True)
    lock_file = lock_dir / "orchestrator.lock"
    metadata_file = lock_dir / "orchestrator.lock.meta.json"
    lock_file.write_text("4242\n", encoding="utf-8")
    metadata_file.write_text('{"pid": 4242}', encoding="utf-8")
    monkeypatch.setattr(aura_main, "_pid_still_runnable", lambda pid: False)

    purged = aura_main._purge_reaped_orchestrator_lock([4242], set())

    assert purged is True
    assert not lock_file.exists()
    assert not metadata_file.exists()


def test_reaped_orchestrator_lock_is_kept_while_pid_runnable(tmp_path, monkeypatch):
    import aura_main

    monkeypatch.setenv("HOME", str(tmp_path))
    lock_dir = tmp_path / ".aura" / "locks"
    lock_dir.mkdir(parents=True)
    lock_file = lock_dir / "orchestrator.lock"
    metadata_file = lock_dir / "orchestrator.lock.meta.json"
    lock_file.write_text("4242\n", encoding="utf-8")
    metadata_file.write_text('{"pid": 4242}', encoding="utf-8")
    monkeypatch.setattr(aura_main, "_pid_still_runnable", lambda pid: True)

    purged = aura_main._purge_reaped_orchestrator_lock([4242], {4242})

    assert purged is False
    assert lock_file.exists()
    assert metadata_file.exists()


def test_bootstrap_lock_does_not_reap_verified_live_runtime(monkeypatch):
    import aura_main

    calls: list[str] = []

    monkeypatch.setattr(aura_main, "_RUNTIME_LOCK_CLAIMED", False)
    monkeypatch.setattr(aura_main, "_verified_live_orchestrator_lock_pid", lambda: 4242)
    monkeypatch.setattr(
        aura_main,
        "_reap_orphaned_aura_processes",
        lambda: calls.append("reap"),
    )

    def fake_acquire_instance_lock(*, lock_name: str, skip_lock: bool = False) -> None:
        calls.append(f"acquire:{lock_name}:{skip_lock}")

    monkeypatch.setattr(aura_main, "acquire_instance_lock", fake_acquire_instance_lock)

    aura_main.bootstrap_lock()

    assert calls == ["acquire:orchestrator:False"]
