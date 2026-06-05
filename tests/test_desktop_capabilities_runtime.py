from __future__ import annotations

import pytest

from core.capabilities.host_automation import AppleScriptRunner, ScriptASTGuard
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
    def __init__(self, process: _FakeProcess) -> None:
        self.process = process
        self.spawn_calls: list[dict[str, object]] = []
        self.shell_calls: list[dict[str, object]] = []

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

    gateway = _FakeSubprocessGateway(_FakeProcess(stdout=b"Notes\n"))
    monkeypatch.setattr(host_automation, "get_subprocess_gateway", lambda: gateway)

    receipt = await AppleScriptRunner.run(
        'tell application "System Events" to get name of first application process whose frontmost is true',
        read_only=True,
        source="unit.host_automation.frontmost",
    )

    assert receipt.success is True
    assert receipt.result == "Notes"
    assert gateway.spawn_calls[0]["argv"][:2] == ["osascript", "-e"]
    assert gateway.spawn_calls[0]["read_only"] is True
    assert gateway.spawn_calls[0]["source"] == "unit.host_automation.frontmost"


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
