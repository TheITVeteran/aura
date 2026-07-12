from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from core.skills import os_automation as os_automation_module
from core.skills.os_automation import OSAutomationCompilerSkill, OSAutomationInput


class _Engine:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        if not self.responses:
            raise AssertionError("unexpected compiler call")
        return self.responses.pop(0)


class _Host:
    def __init__(self, snapshots: list[dict[str, object]]) -> None:
        self.snapshots = list(snapshots)
        self.executed_scripts: list[str] = []
        self._current_snapshot: dict[str, object] = {}

    async def inspect_applescript(
        self,
        script: str,
        *,
        timeout_s: float = 5.0,
        source: str = "unit",
    ) -> SimpleNamespace:
        del script, timeout_s
        if source == "os_automation.desktop_snapshot":
            if not self.snapshots:
                raise AssertionError("unexpected desktop snapshot")
            self._current_snapshot = self.snapshots.pop(0)
            return SimpleNamespace(success=True, result=self._current_snapshot, error="")
        return SimpleNamespace(
            success=True,
            result=str(self._current_snapshot.get("browser_url") or ""),
            error="",
        )

    async def get_screen_text(self, *, retain_screenshot: bool = True) -> SimpleNamespace:
        del retain_screenshot
        text = str(self._current_snapshot.get("screen_text") or "")
        return SimpleNamespace(success=bool(text), result=text, error="")

    async def execute_applescript(self, script: str) -> SimpleNamespace:
        self.executed_scripts.append(script)
        return SimpleNamespace(
            success=True,
            result="transport ok",
            error="",
            receipt_id=f"receipt-{len(self.executed_scripts)}",
            adapter="applescript",
        )


def _snapshot(
    *,
    app: str = "Finder",
    frame: tuple[int, int, int, int] = (300, 100, 1100, 800),
    text: str = "",
) -> dict[str, object]:
    return {
        "frontmost_app": app,
        "frontmost_window": "Unit Window",
        "window_frame": frame,
        "desktop_frame": (0, 0, 1920, 1080),
        "window_minimized": False,
        "focused_value_excerpt": text,
        "screen_text": text,
        "running_apps": ("Finder", app),
    }


def _install_runtime(monkeypatch: pytest.MonkeyPatch, engine: _Engine, host: _Host | None) -> None:
    monkeypatch.setattr(
        os_automation_module.ServiceContainer,
        "get",
        lambda name, default=None: engine if name == "cognitive_engine" else default,
    )
    monkeypatch.setattr(os_automation_module, "get_host_automation", lambda: host)


def test_strict_script_parser_rejects_prose_multiple_blocks_and_wrong_language() -> None:
    extract = OSAutomationCompilerSkill._extract_single_script
    valid = '```applescript\ntell application "Notes" to activate\n```'

    assert extract(valid, "applescript") == 'tell application "Notes" to activate'
    with pytest.raises(ValueError, match="exactly one"):
        extract("Here you go\n" + valid, "applescript")
    with pytest.raises(ValueError, match="multiple"):
        extract(valid + "\n" + valid, "applescript")
    with pytest.raises(ValueError, match="wrong fenced language"):
        extract("```bash\necho hi\n```", "applescript")
    with pytest.raises(ValueError, match="exactly one"):
        extract('tell application "Notes" to activate', "applescript")


def test_input_contract_rejects_shell_lane() -> None:
    with pytest.raises(ValidationError):
        OSAutomationInput(goal="list files", script_type="bash")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_delegated_execution_uses_foreground_primary_and_verifies_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _Engine('```applescript\ntell application "Notes" to activate\n```')
    host = _Host([_snapshot(), _snapshot(app="Notes")])
    _install_runtime(monkeypatch, engine, host)

    async def duplicate_authority(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("delegated execution must not issue duplicate authority")

    monkeypatch.setattr(
        OSAutomationCompilerSkill,
        "_authorize",
        staticmethod(duplicate_authority),
    )
    result = await OSAutomationCompilerSkill().execute(
        OSAutomationInput(goal="Open Notes."),
        {
            "_capability_token_verified": True,
            "capability_token_id": "outer-token",
            "origin": "desktop_ui",
        },
    )

    assert result["ok"] is True
    assert result["status"] == "completed_verified"
    assert result["effect_verified"] is True
    assert result["authority"]["mode"] == "delegated"
    assert len(host.executed_scripts) == 1
    assert engine.calls[0]["is_background"] is False
    assert engine.calls[0]["prefer_tier"] == "primary"
    assert engine.calls[0]["use_strategies"] is False
    assert engine.calls[0]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_failed_effect_gets_one_evidence_driven_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _Engine(
        '```applescript\ntell application "Google Chrome" to activate\n```',
        '```applescript\ntell application "System Events" to set position of window 1 of first application process whose frontmost is true to {0, 24}\n```',
    )
    host = _Host(
        [
            _snapshot(app="Google Chrome"),
            _snapshot(app="Google Chrome"),
            _snapshot(app="Google Chrome", frame=(0, 24, 940, 1030)),
        ]
    )
    _install_runtime(monkeypatch, engine, host)

    result = await OSAutomationCompilerSkill().execute(
        OSAutomationInput(goal="Arrange the current Chrome window on the left side."),
        {"_capability_token_verified": True, "capability_token_id": "outer-token"},
    )

    assert result["ok"] is True
    assert result["effect_verified"] is True
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["verification"]["verified"] is False
    assert result["attempts"][1]["verification"]["verified"] is True
    assert len(host.executed_scripts) == 2
    assert len(engine.calls) == 2
    assert "Failed checks" in engine.calls[1]["prompt"]


@pytest.mark.asyncio
async def test_transport_success_without_effect_never_reports_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = '```applescript\ntell application "Google Chrome" to activate\n```'
    engine = _Engine(script, script)
    host = _Host(
        [
            _snapshot(app="Google Chrome"),
            _snapshot(app="Google Chrome"),
        ]
    )
    _install_runtime(monkeypatch, engine, host)

    result = await OSAutomationCompilerSkill().execute(
        OSAutomationInput(goal="Arrange the current Chrome window on the left side."),
        {"_capability_token_verified": True, "capability_token_id": "outer-token"},
    )

    assert result["ok"] is False
    assert result["status"] == "repair_made_no_change"
    assert result["effect_verified"] is False
    assert result["attempts"][0]["transport_success"] is True
    assert len(host.executed_scripts) == 1


@pytest.mark.asyncio
async def test_direct_compile_only_closes_authority_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _Engine('```applescript\ntell application "Notes" to activate\n```')
    _install_runtime(monkeypatch, engine, None)
    auth = {
        "approved": True,
        "reason": "unit",
        "delegated": False,
        "decision": SimpleNamespace(),
        "executive_intent_id": "intent-1",
        "capability_token_id": "token-1",
    }
    finalized: list[bool] = []

    async def authorize(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return auth

    def finalize(_auth: dict[str, Any], *, success: bool) -> dict[str, Any]:
        finalized.append(success)
        return {"closed": True, "mode": "unit", "success": success, "errors": []}

    monkeypatch.setattr(OSAutomationCompilerSkill, "_authorize", staticmethod(authorize))
    monkeypatch.setattr(OSAutomationCompilerSkill, "_finalize", staticmethod(finalize))
    result = await OSAutomationCompilerSkill().execute(
        OSAutomationInput(goal="Open Notes.", execute=False),
        {"origin": "unit"},
    )

    assert result["ok"] is True
    assert result["status"] == "compiled_validated_not_executed"
    assert finalized == [True]


@pytest.mark.asyncio
async def test_direct_closure_failure_stops_repair_and_requires_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _Engine('```applescript\ntell application "Notes" to activate\n```')
    host = _Host([_snapshot(), _snapshot(app="Notes")])
    _install_runtime(monkeypatch, engine, host)
    auth = {
        "approved": True,
        "reason": "unit",
        "delegated": False,
        "decision": SimpleNamespace(),
        "executive_intent_id": "intent-1",
        "capability_token_id": "token-1",
    }

    async def authorize(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return auth

    def fail_closure(_auth: dict[str, Any], *, success: bool) -> dict[str, Any]:
        return {
            "closed": False,
            "mode": "unit",
            "success": success,
            "errors": ["token revoke failed"],
        }

    monkeypatch.setattr(OSAutomationCompilerSkill, "_authorize", staticmethod(authorize))
    monkeypatch.setattr(OSAutomationCompilerSkill, "_finalize", staticmethod(fail_closure))
    result = await OSAutomationCompilerSkill().execute(
        OSAutomationInput(goal="Open Notes."),
        {"origin": "unit"},
    )

    assert result["ok"] is False
    assert result["status"] == "authority_closure_failed"
    assert result["manual_reconciliation_required"] is True
    assert len(host.executed_scripts) == 1
    assert len(engine.calls) == 1
