"""Canonical transaction and observed-effect contracts for browser work."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError


def test_browser_input_contracts_are_mode_specific_and_deterministic() -> None:
    from core.skills.sovereign_browser import (
        BrowserAction,
        BrowserInput,
        SovereignBrowserSkill,
    )

    with pytest.raises(ValidationError, match="non-empty query"):
        BrowserInput(mode="search", query="  ")
    with pytest.raises(ValidationError, match="requires a selector"):
        BrowserAction(type="click")
    with pytest.raises(ValidationError, match="between 0 and 10"):
        BrowserAction(type="wait", value="11")
    with pytest.raises(ValidationError, match="at least one action"):
        BrowserInput(mode="interact", url="example.test", actions=[])

    normalized = BrowserInput(mode="browse", url="example.test")
    assert normalized.url == "https://example.test"
    assert SovereignBrowserSkill()._pick_browser_type("auto") == "chromium"


@pytest.mark.asyncio
async def test_browser_startup_failure_is_not_treated_as_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.skills.sovereign_browser as browser_module

    class InactiveBrowser:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def ensure_ready(self) -> bool:
            return False

        def get_status(self) -> dict[str, Any]:
            return {"startup_error": "browser binary unavailable"}

    monkeypatch.setattr(browser_module, "PhantomBrowser", InactiveBrowser)
    with pytest.raises(RuntimeError, match="browser binary unavailable"):
        await browser_module.SovereignBrowserSkill()._create_browser("chromium")


@pytest.mark.asyncio
async def test_direct_browser_execution_uses_action_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.skills.sovereign_browser as browser_module

    skill = browser_module.SovereignBrowserSkill()
    raw_result = {
        "ok": True,
        "observed_url": "https://example.test/final",
        "navigation_confirmed": True,
        "content": "observed body",
    }
    monkeypatch.setattr(skill, "_execute_browser", AsyncMock(return_value=raw_result))
    captured: dict[str, Any] = {}

    async def fake_execute(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        handler_result = await kwargs["effect_handler"]({"will_receipt_id": "will-1"})
        verifier_result = kwargs["effect_verifier"](
            {
                "params": kwargs["params"],
                "result": handler_result,
            }
        )
        assert verifier_result["effect_verified"] is True
        return {"ok": True, "transaction_routed": True}

    monkeypatch.setattr(browser_module.ActionExecutor, "execute", fake_execute)
    result = await skill.execute(
        {"mode": "browse", "url": "https://example.test/start"},
        {"source": "browser_test"},
    )

    assert result == {"ok": True, "transaction_routed": True}
    assert captured["domain"] == browser_module.ActionDomain.NETWORK_CALL
    assert captured["action_name"] == "sovereign_browser.browse"
    assert captured["source"] == "browser_test"
    assert captured["execution_timeout_s"] > skill.BROWSE_TIMEOUT
    assert captured["expectation"].required_evidence == ["custom_verifier.observation"]


@pytest.mark.asyncio
async def test_action_executor_managed_browser_does_not_reenter_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.skills.sovereign_browser as browser_module

    skill = browser_module.SovereignBrowserSkill()
    expected = {"ok": True, "engine": "managed"}
    execute_browser = AsyncMock(return_value=expected)
    monkeypatch.setattr(skill, "_execute_browser", execute_browser)

    async def must_not_run(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("managed browser execution re-entered ActionExecutor")

    monkeypatch.setattr(browser_module.ActionExecutor, "execute", must_not_run)
    result = await skill.execute(
        {"mode": "search", "query": "bounded test"},
        {"action_executor_managed_welfare_transaction": True},
    )

    assert result == expected
    execute_browser.assert_awaited_once()


def test_browser_verifier_requires_observed_navigation_content_and_actions() -> None:
    from core.skills.sovereign_browser import SovereignBrowserSkill

    browse = SovereignBrowserSkill._verify_browser_effect(
        {
            "params": {"mode": "browse", "url": "https://example.test/start"},
            "result": {
                "ok": True,
                "observed_url": "https://example.test/final",
                "navigation_confirmed": True,
                "content": "readback",
            },
        }
    )
    assert browse["effect_verified"] is True
    assert browse["observation"]["content_sha256"]
    assert "content" not in browse["observation"]

    empty = SovereignBrowserSkill._verify_browser_effect(
        {
            "params": {"mode": "browse", "url": "https://example.test/start"},
            "result": {
                "ok": True,
                "observed_url": "https://example.test/final",
                "navigation_confirmed": True,
                "content": "",
            },
        }
    )
    assert empty["effect_verified"] is False

    interact = SovereignBrowserSkill._verify_browser_effect(
        {
            "params": {
                "mode": "interact",
                "url": "https://example.test",
                "actions": [{"type": "click"}, {"type": "type", "value": "secret"}],
            },
            "result": {
                "ok": True,
                "observed_url": "https://example.test/complete",
                "navigation_confirmed": True,
                "action_report": [
                    {"action": "click", "ok": True},
                    {"action": "type", "ok": True},
                ],
            },
        }
    )
    assert interact["effect_verified"] is True
    assert "secret" not in str(interact)


@pytest.mark.asyncio
async def test_interaction_failure_is_not_reported_as_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.skills.sovereign_browser import (
        BrowserAction,
        SovereignBrowserSkill,
    )

    skill = SovereignBrowserSkill()
    browser = SimpleNamespace(
        click=AsyncMock(return_value=False),
        page=SimpleNamespace(url="https://example.test"),
    )
    monkeypatch.setattr(skill, "_safe_read_content", AsyncMock(return_value="page body"))

    result = await skill._handle_interact(
        browser,
        "",
        [BrowserAction(type="click", selector="#missing")],
    )

    assert result["ok"] is False
    assert result["error"] == "browser_interaction_incomplete"
    assert result["action_report"] == [{"action": "click", "ok": False}]


@pytest.mark.asyncio
async def test_phantom_type_aborts_when_focus_click_fails() -> None:
    from core.capabilities.phantom_browser import PhantomBrowser

    browser = object.__new__(PhantomBrowser)
    browser.click = AsyncMock(return_value=False)
    browser.page = SimpleNamespace(keyboard=SimpleNamespace(type=AsyncMock()))
    browser._human_delay = AsyncMock()

    result = await browser.type("#password", "must-not-be-typed")

    assert result is False
    browser.page.keyboard.type.assert_not_awaited()
