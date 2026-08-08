"""Contracts for the canonical governed desktop transport boundary."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest


def test_action_executor_desktop_transport_reuses_existing_governance(monkeypatch) -> None:
    import core.runtime.action_executor as executor

    checks: list[tuple[str, dict[str, Any]]] = []

    class Gateway:
        def run_applescript(self, script: str, *, source: str, timeout: float):
            return {
                "ok": True,
                "stdout": script,
                "stderr": "",
                "exit_code": 0,
                "source": source,
                "timeout": timeout,
            }

    monkeypatch.setattr(
        executor,
        "require_governance",
        lambda operation, **kwargs: checks.append((operation, kwargs)),
    )
    monkeypatch.setattr(executor, "get_desktop_action_gateway", lambda: Gateway())

    result = executor.ActionExecutor.request_desktop_transport(
        script='return "ok"',
        source="computer_use",
        timeout_s=7.0,
    )

    assert result["ok"] is True
    assert result["source"] == "computer_use"
    assert result["timeout"] == 7.0
    assert checks == [
        (
            "action_executor.request_desktop_transport",
            {
                "strict": True,
                "allowed_domains": (
                    "environment_action",
                    "external_action",
                    "tool_execution",
                ),
            },
        )
    ]


def test_action_executor_desktop_transport_rejects_unowned_callers() -> None:
    from core.runtime.action_executor import ActionExecutor

    with pytest.raises(ValueError, match="desktop transport source"):
        ActionExecutor.request_desktop_transport(
            script='return "no"',
            source="arbitrary_plugin",
        )


def test_computer_use_routes_applescript_through_action_executor(monkeypatch) -> None:
    from core.runtime.action_executor import ActionExecutor
    from core.skills.computer_use import ComputerUseSkill

    calls: list[dict[str, Any]] = []

    def request(**kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        return {"ok": True, "stdout": "menu clock", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(ActionExecutor, "request_desktop_transport", request)
    monkeypatch.setenv("AURA_COMPUTER_USE_NATIVE_APPLESCRIPT", "1")

    assert ComputerUseSkill()._run_applescript('return "menu clock"', timeout=6) == "menu clock"
    assert calls == [
        {
            "script": 'return "menu clock"',
            "source": "computer_use",
            "timeout_s": 6,
        }
    ]


def test_web_interlocutor_reuses_action_executor_receipt_and_run_identity(monkeypatch) -> None:
    import core.capabilities.web_interlocutor as interlocutor

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        interlocutor.ActionExecutor,
        "request_desktop_transport",
        lambda **kwargs: (
            calls.append(dict(kwargs)) or {"ok": True, "stdout": "", "stderr": "", "exit_code": 0}
        ),
    )

    with interlocutor._caller_authority("owner", "webchat-run-proof"):
        result = interlocutor._run_governed_applescript(
            'return "ok"',
            source="web_interlocutor.test",
            timeout=4.0,
        )

    assert result["ok"] is True
    assert calls == [
        {
            "script": 'return "ok"',
            "source": "web_interlocutor.test:run=webchat-run-proof",
            "timeout_s": 4.0,
        }
    ]


def test_native_applescript_error_is_not_replayed(monkeypatch) -> None:
    import core.runtime.desktop_action_gateway as gateway

    class NativeScript:
        @classmethod
        def alloc(cls):
            return cls()

        def initWithSource_(self, _script: str):  # noqa: N802 - PyObjC selector
            return self

        def executeAndReturnError_(self, _error: Any):  # noqa: N802 - PyObjC selector
            return None, {
                "NSAppleScriptErrorMessage": "permission denied",
                "NSAppleScriptErrorNumber": -1743,
            }

    monkeypatch.setitem(sys.modules, "Foundation", SimpleNamespace(NSAppleScript=NativeScript))
    monkeypatch.setenv("AURA_COMPUTER_USE_NATIVE_APPLESCRIPT", "1")
    monkeypatch.setattr(gateway, "governance_runtime_active", lambda: False)
    monkeypatch.setattr(gateway, "_refuse_if_untrusted_context", lambda *_args: None)
    monkeypatch.setattr(
        gateway,
        "get_subprocess_gateway",
        lambda: (_ for _ in ()).throw(AssertionError("native failure must not replay")),
    )

    result = gateway.DesktopActionGateway().run_applescript('return "x"')

    assert result["ok"] is False
    assert result["exit_code"] == -1743
    assert result["transport"] == "native_nsapplescript"
