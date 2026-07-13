from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import pytest

from core.capabilities.host_automation import (
    AppleScriptRunner,
    HostAutomationProvider,
    ScriptASTGuard,
)
from core.capabilities.permission_model import PermissionRiskModel, RiskLevel
from core.capabilities.post_action_verifier import PostActionVerifier


class _FakeProcess:
    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self, input: bytes | None = None):  # noqa: A002 - subprocess API.
        return self._stdout, self._stderr


class _FakeSubprocessGateway:
    def __init__(self, process: subprocess.CompletedProcess[str] | _FakeProcess) -> None:
        self.process = process
        self.run_calls: list[dict[str, object]] = []
        self.spawn_calls: list[dict[str, object]] = []
        self.shell_calls: list[dict[str, object]] = []

    async def run_async(self, argv, **kwargs):
        self.run_calls.append({"argv": list(argv), **kwargs})
        return self.process

    async def spawn_async(self, argv, **kwargs):
        self.spawn_calls.append({"argv": list(argv), **kwargs})
        return self.process

    async def spawn_shell_async(self, command, **kwargs):
        self.shell_calls.append({"command": command, **kwargs})
        return self.process


def test_script_ast_guard_blocks_destructive_applescript() -> None:
    ok, reason = ScriptASTGuard.validate_applescript(
        'do shell script "sudo rm -rf /"'
    )

    assert ok is False
    assert "Blocked pattern" in reason


@pytest.mark.asyncio
async def test_applescript_runner_uses_subprocess_gateway(monkeypatch) -> None:
    from core.capabilities import host_automation

    gateway = _FakeSubprocessGateway(
        subprocess.CompletedProcess(["osascript"], 0, "Notes\n", "")
    )
    monkeypatch.setattr(host_automation, "get_subprocess_gateway", lambda: gateway)

    receipt = await AppleScriptRunner.run(
        'tell application "System Events" to get name of first application process whose frontmost is true',
        read_only=True,
        source="unit.host_automation.frontmost",
    )

    assert receipt.success is True
    assert receipt.result == "Notes"
    assert gateway.run_calls[0]["argv"][:2] == ["osascript", "-e"]
    assert gateway.run_calls[0]["read_only"] is True
    assert gateway.run_calls[0]["source"] == "unit.host_automation.frontmost"
    assert gateway.run_calls[0]["timeout"] == 10.0


@pytest.mark.asyncio
async def test_host_applescript_inspection_is_read_only_and_not_action_logged(monkeypatch) -> None:
    from core.capabilities import host_automation

    gateway = _FakeSubprocessGateway(
        subprocess.CompletedProcess(["osascript"], 0, "Finder\n", "")
    )
    monkeypatch.setattr(host_automation, "get_subprocess_gateway", lambda: gateway)
    provider = HostAutomationProvider()

    receipt = await provider.inspect_applescript(
        'tell application "System Events" to get name of first application process whose frontmost is true',
        timeout_s=2.0,
        source="unit.host_automation.inspection",
    )

    assert receipt.success is True
    assert receipt.action == "inspect_applescript"
    assert gateway.run_calls[0]["read_only"] is True
    assert gateway.run_calls[0]["source"] == "unit.host_automation.inspection"
    assert provider.get_recent_receipts() == []


@pytest.mark.asyncio
async def test_ephemeral_ocr_capture_is_deleted_after_verification(
    monkeypatch,
    tmp_path,
) -> None:
    from core.capabilities import host_automation

    provider = HostAutomationProvider()
    capture_path = tmp_path / "aura-ephemeral-verification.png"
    screenshot = SimpleNamespace(
        success=True,
        result=str(capture_path),
        error="",
    )

    async def take_screenshot(*_args, **_kwargs):
        return screenshot

    deleted: list[tuple[str, str]] = []

    async def execute_action(*, params, source, **_kwargs):
        deleted.append((str(params["path"]), source))
        return {
            "ok": True,
            "effect_verified": True,
            "receipt_persisted": True,
            "post_action_receipt_id": "receipt-cleanup",
        }

    monkeypatch.setattr(provider, "take_screenshot", take_screenshot)
    monkeypatch.setattr(host_automation.ActionExecutor, "execute", execute_action)
    monkeypatch.setattr(
        provider,
        "_ocr_image_text",
        lambda _path: "Visible verification text",
    )

    receipt = await provider.get_screen_text(retain_screenshot=False)

    assert receipt.success is True
    assert receipt.result == "Visible verification text"
    assert receipt.target == "ephemeral_verification_capture"
    assert deleted == [
        (
            str(capture_path),
            "host_automation.ephemeral_ocr_cleanup",
        )
    ]


@pytest.mark.asyncio
async def test_screenshot_retention_bounds_capture_count(monkeypatch, tmp_path) -> None:
    from core.capabilities import host_automation

    monkeypatch.setenv("AURA_SCREENSHOT_RETENTION_MAX_FILES", "10")
    monkeypatch.setenv("AURA_SCREENSHOT_RETENTION_MAX_DAYS", "365")
    monkeypatch.setenv("AURA_SCREENSHOT_RETENTION_MAX_BYTES", str(32 * 1024 * 1024))
    paths = []
    for index in range(12):
        path = tmp_path / f"capture-{index:02d}.png"
        path.write_bytes(b"png")
        os.utime(path, (2_000_000_000 + index, 2_000_000_000 + index))
        paths.append(path)
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("not a capture")

    async def execute_action(*, params, **_kwargs):
        path = tmp_path / str(params["path"]).split("/")[-1]
        path.unlink(missing_ok=True)
        return {
            "ok": True,
            "effect_verified": True,
            "receipt_persisted": True,
            "post_action_receipt_id": f"delete-{path.name}",
        }

    monkeypatch.setattr(host_automation.ActionExecutor, "execute", execute_action)
    result = await HostAutomationProvider._enforce_screenshot_retention(
        tmp_path,
        keep_path=paths[-1],
    )

    assert result["kept"] == 10
    assert result["deleted"] == 2
    assert sum(path.exists() for path in paths) == 10
    assert unrelated.exists()


@pytest.mark.asyncio
async def test_post_action_verifier_clipboard_uses_read_only_gateway(monkeypatch) -> None:
    from core.capabilities import post_action_verifier

    gateway = _FakeSubprocessGateway(_FakeProcess(stdout=b"hello Aura"))
    monkeypatch.setattr(post_action_verifier, "get_subprocess_gateway", lambda: gateway)

    result = await PostActionVerifier().verify(
        "clipboard_contains",
        {"text": "Aura"},
    )

    assert result.success is True
    assert "contains target: True" in result.evidence
    assert gateway.spawn_calls[0]["argv"] == ["pbpaste"]
    assert gateway.spawn_calls[0]["read_only"] is True
    assert gateway.spawn_calls[0]["source"] == "post_action_verifier.clipboard_contains"


@pytest.mark.asyncio
async def test_post_action_verifier_command_uses_governed_shell_gateway(monkeypatch) -> None:
    from core.capabilities import post_action_verifier

    gateway = _FakeSubprocessGateway(_FakeProcess(stdout=b"ok", returncode=0))
    monkeypatch.setattr(post_action_verifier, "get_subprocess_gateway", lambda: gateway)

    result = await PostActionVerifier().verify(
        "command_succeeded",
        {"command": "printf ok"},
    )

    assert result.success is True
    assert gateway.shell_calls[0]["command"] == "printf ok"
    assert gateway.shell_calls[0]["source"] == "post_action_verifier.command_succeeded"


def test_permission_model_blocks_destructive_action() -> None:
    model = PermissionRiskModel()

    decision = model.check_permission("run command", "sudo rm -rf /")

    assert decision.risk_level == RiskLevel.BLOCKED
    assert decision.approved is False
    assert "BLOCKED" in decision.reason


def test_permission_model_modality_detection_uses_word_boundaries() -> None:
    model = PermissionRiskModel()

    assert model._detect_modality("response", "ZeroDivisionError in average(nums)") == "app_control"
    assert model._detect_modality("response", "Provisioning dependency graph") == "app_control"
    assert model._detect_modality("response", "shared reference mechanism") == "app_control"
    assert model._detect_modality("response", "camera capture request") == "camera"
    assert model._detect_modality("response", "upload this file") == "network_write"
