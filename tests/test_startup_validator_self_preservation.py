import time
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def clear_container():
    from core.container import ServiceContainer

    ServiceContainer.clear()
    yield
    ServiceContainer.clear()


@pytest.mark.asyncio
async def test_startup_validator_blocks_unsafe_self_preservation_files(tmp_path, monkeypatch):
    from core.container import ServiceContainer
    from core.resilience.startup_validator import StartupValidator, ValidationCheck

    legacy_file = tmp_path / "core" / "self_preservation_integration.py"
    legacy_file.parent.mkdir(parents=True)
    legacy_file.write_text("class SecurityBypassSystem: ...\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    ServiceContainer.register_instance("backup_system", object())

    check = ValidationCheck("safe_01", "Dangerous Files Purged", "", True)
    await StartupValidator(SimpleNamespace(backup_system=object()))._check_safe_01(check)

    assert check.passed is False
    assert "unsafe legacy self-preservation files are present" in check.message


@pytest.mark.asyncio
async def test_startup_validator_accepts_safe_backup_when_legacy_files_absent(tmp_path, monkeypatch):
    from core.container import ServiceContainer
    from core.resilience.startup_validator import StartupValidator, ValidationCheck

    monkeypatch.chdir(tmp_path)
    ServiceContainer.register_instance("backup_system", object())

    check = ValidationCheck("safe_01", "Dangerous Files Purged", "", True)
    await StartupValidator(SimpleNamespace(backup_system=object()))._check_safe_01(check)

    assert check.passed is True
    assert check.message == "Unsafe self-preservation path removed; safe backup active."


@pytest.mark.asyncio
async def test_state_validation_uses_authoritative_kernel_fallback_without_sleep():
    from core.container import ServiceContainer
    from core.resilience.startup_validator import StartupValidator, ValidationCheck

    calls = 0

    class Repository:
        _current = None

        async def get_current(self):
            nonlocal calls
            calls += 1
            return None

    state = SimpleNamespace(version=17)
    orchestrator = SimpleNamespace(
        kernel_interface=SimpleNamespace(kernel=SimpleNamespace(state=state))
    )
    ServiceContainer.register_instance("state_repository", Repository())
    check = ValidationCheck("core_03", "State Repository Bound", "", True)

    started = time.perf_counter()
    await StartupValidator(orchestrator)._check_core_03(check)
    elapsed = time.perf_counter() - started

    assert check.passed is True
    assert check.message == "State bound via authoritative fallback (v17)."
    assert calls == 1
    assert elapsed < 0.05


@pytest.mark.asyncio
async def test_startup_validator_memory_check_uses_attributed_observer(resource_observer):
    from core.resilience.startup_validator import StartupValidator, ValidationCheck

    resource_observer.configure_memory(available_bytes=256 * 1024**2)
    check = ValidationCheck("sys_01", "Memory Check", "", False)

    await StartupValidator(SimpleNamespace())._check_sys_01(check)

    assert check.passed is False
    assert check.message == "Low memory available: 256MB"

    resource_observer.configure_memory(available_bytes=8 * 1024**3)
    await StartupValidator(SimpleNamespace())._check_sys_01(check)
    assert check.passed is True
    assert "(simulated)" in check.message


@pytest.mark.asyncio
async def test_startup_validator_zombie_scan_fails_visible_when_process_table_is_blind(
    monkeypatch,
    resource_observer,
):
    import core.resilience.startup_validator as validator_module
    from core.runtime.resource_observation import ProcessTableObservation

    table = ProcessTableObservation(
        provenance=resource_observer.provenance,
        processes=(),
        available=False,
        error="induced-process-visibility-loss",
    )
    monkeypatch.setattr(
        validator_module,
        "get_resource_observer",
        lambda: SimpleNamespace(process_table=lambda: table),
    )
    check = validator_module.ValidationCheck("sys_03", "Zombie Reaper", "", False)

    await validator_module.StartupValidator(SimpleNamespace())._check_sys_03(check)

    assert check.passed is False
    assert "process_table_unavailable" in check.message


@pytest.mark.asyncio
async def test_startup_validator_reaps_only_observed_orphan_worker(
    monkeypatch,
    resource_observer,
):
    import core.resilience.startup_validator as validator_module
    from core.runtime.resource_observation import ProcessObservation

    orphan_pid = 94_321
    resource_observer.configure_processes(
        [
            ProcessObservation(
                provenance=resource_observer.provenance,
                pid=orphan_pid,
                ppid=1,
                create_time=1.0,
                status="running",
                name="python",
                cmdline=("python", "mlx_worker.py"),
                rss_bytes=1024,
            )
        ]
    )
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr("os.kill", lambda pid, sig: killed.append((pid, sig)))
    check = validator_module.ValidationCheck("sys_03", "Zombie Reaper", "", False)

    await validator_module.StartupValidator(SimpleNamespace())._check_sys_03(check)

    assert check.passed is True
    assert check.message == "Reaped 1 orphaned workers."
    assert [pid for pid, _signal in killed] == [orphan_pid]
