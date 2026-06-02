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
