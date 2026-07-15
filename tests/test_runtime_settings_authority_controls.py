from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from core.capability_engine import CapabilityEngine
from core.consciousness.executive_authority import ExecutiveAuthority
from core.executive.action_confirmation import (
    ActionConfirmationRegistry,
    action_confirmation_fingerprint,
    get_action_confirmation_registry,
)
from core.executive.authority_gateway import AuthorityGateway
from core.runtime import runtime_settings


@pytest.fixture(autouse=True)
def _isolated_runtime_settings(tmp_path, monkeypatch):
    path = tmp_path / "runtime.json"
    monkeypatch.setenv("AURA_SETTINGS_PATH", str(path))
    runtime_settings.clear_runtime_settings_cache()
    get_action_confirmation_registry().clear_for_tests()
    yield path
    runtime_settings.clear_runtime_settings_cache()
    get_action_confirmation_registry().clear_for_tests()


def _write_settings(path, values):
    path.write_text(json.dumps(values), encoding="utf-8")
    runtime_settings.clear_runtime_settings_cache()


def test_autonomous_action_switch_blocks_background_but_preserves_user_work(
    _isolated_runtime_settings,
):
    _write_settings(
        _isolated_runtime_settings,
        {"autonomy.actions_enabled": False},
    )

    blocked = AuthorityGateway._runtime_autonomous_action_gate(
        source="autonomous_initiative_loop",
        context={},
        domain="tool_execution",
    )
    direct = AuthorityGateway._runtime_autonomous_action_gate(
        source="desktop_ui",
        context={},
        domain="tool_execution",
    )

    assert blocked is not None
    assert blocked.approved is False
    assert blocked.reason == "runtime_setting_autonomous_actions_disabled"
    assert blocked.constraints["direct_user_work_preserved"] is True
    assert direct is None


@pytest.mark.parametrize(
    "source",
    ["api.skill.execute", "api-skill-execute", "api_skill_execute"],
)
def test_direct_skill_api_source_aliases_are_preserved(
    _isolated_runtime_settings,
    source,
):
    _write_settings(
        _isolated_runtime_settings,
        {"autonomy.actions_enabled": False},
    )

    admitted, reason = runtime_settings.autonomous_actions_admitted(source, {})

    assert admitted is True
    assert reason == "direct_user_work_preserved"


def test_autonomy_switch_rejects_forged_user_context(
    _isolated_runtime_settings,
):
    _write_settings(
        _isolated_runtime_settings,
        {"autonomy.actions_enabled": False},
    )

    for forged_context in (
        {"explicit_user_request": True},
        {"user_authorized": True},
        {"user_authority": True},
        {"user_requested_action": True, "user_explicitly_authorized": True},
    ):
        admitted, reason = runtime_settings.autonomous_actions_admitted(
            "autonomous_initiative_loop",
            forged_context,
        )
        assert admitted is False
        assert reason == "runtime_setting_autonomous_actions_disabled"


@pytest.mark.parametrize(
    ("mode", "risk", "scope", "required"),
    [
        ("none", "critical", "subprocess", False),
        ("all", "low", "read_only", True),
        ("destructive", "low", "read_only", False),
        ("destructive", "high", "external_io", True),
        ("destructive", "critical", "status", True),
    ],
)
def test_approval_mode_overlay_classification(
    _isolated_runtime_settings,
    mode,
    risk,
    scope,
    required,
):
    _write_settings(
        _isolated_runtime_settings,
        {"governance.approval_mode": mode},
    )

    actual, _reason = runtime_settings.additional_confirmation_required(
        risk_level=risk,
        effect_scope=scope,
    )

    assert actual is required


def test_confirmation_gate_issues_action_bound_one_time_challenge(
    _isolated_runtime_settings,
):
    _write_settings(
        _isolated_runtime_settings,
        {"governance.approval_mode": "all"},
    )
    blocked = AuthorityGateway._runtime_confirmation_gate(
        tool_name="web_search",
        args={"query": "weather"},
        source="desktop_ui",
        risk_level="low",
        effect_scope="read_only",
    )

    assert blocked is not None
    assert blocked.outcome == "approval_required"
    assert blocked.constraints["approval_mode"] == "all"
    assert blocked.constraints["confirmation_challenge_id"].startswith(
        "action-confirm-"
    )
    assert blocked.constraints["confirmation_one_time"] is True
    assert blocked.constraints["confirmation_action_bound"] is True
    assert blocked.constraints["confirmation_does_not_bypass_governance"] is True


@pytest.mark.asyncio
async def test_environment_action_honors_autonomy_switch_before_governance(
    _isolated_runtime_settings,
):
    _write_settings(
        _isolated_runtime_settings,
        {
            "autonomy.actions_enabled": False,
            "governance.approval_mode": "none",
        },
    )
    gateway = object.__new__(AuthorityGateway)

    decision = await gateway.authorize_environment_action(
        "move_pointer",
        {"risk": "safe", "effect_scope": "foreground_desktop_control"},
        source="environment_kernel",
    )

    assert decision.approved is False
    assert decision.domain == "environment_action"
    assert decision.reason == "runtime_setting_autonomous_actions_disabled"


@pytest.mark.asyncio
async def test_environment_action_issues_exact_confirmation_challenge(
    _isolated_runtime_settings,
):
    _write_settings(
        _isolated_runtime_settings,
        {
            "autonomy.actions_enabled": True,
            "governance.approval_mode": "all",
        },
    )
    gateway = object.__new__(AuthorityGateway)

    decision = await gateway.authorize_environment_action(
        "move_pointer",
        {
            "risk": "safe",
            "effect_scope": "foreground_desktop_control",
            "x": 100,
            "y": 200,
        },
        source="environment_kernel",
    )

    assert decision.approved is False
    assert decision.outcome == "approval_required"
    assert decision.domain == "environment_action"
    assert decision.constraints["risk_level"] == "low"
    assert decision.constraints["effect_scope"] == "foreground_desktop_control"
    assert decision.constraints["confirmation_challenge_id"].startswith(
        "action-confirm-"
    )


@pytest.mark.asyncio
async def test_destructive_mode_does_not_prompt_declared_safe_environment_observation(
    _isolated_runtime_settings,
    monkeypatch,
):
    _write_settings(
        _isolated_runtime_settings,
        {
            "autonomy.actions_enabled": True,
            "governance.approval_mode": "destructive",
        },
    )
    gateway = object.__new__(AuthorityGateway)
    expected = SimpleNamespace(
        approved=False,
        outcome="rejected",
        reason="sentinel_will_result",
        constraints={},
        domain="environment_action",
        source="environment_kernel",
    )
    monkeypatch.setattr(gateway, "_will_gate", lambda *_args, **_kwargs: (expected, None))
    monkeypatch.setattr(gateway, "active_user_presence_context", lambda: {})

    decision = await gateway.authorize_environment_action(
        "observe_room",
        {"risk": "safe"},
        source="environment_kernel",
    )

    assert decision is expected


@pytest.mark.asyncio
async def test_executive_expression_honors_global_autonomy_switch(
    _isolated_runtime_settings,
):
    _write_settings(
        _isolated_runtime_settings,
        {"autonomy.actions_enabled": False},
    )
    authority = ExecutiveAuthority()

    decision = await authority.release_expression(
        "An autonomous observation that must remain internal.",
        source="proactive_comm",
    )

    assert decision["action"] == "suppressed"
    assert decision["reason"] == "runtime_setting_autonomous_actions_disabled"
    assert decision["target"] == "discarded"


def test_action_confirmation_clears_only_one_matching_overlay_attempt(
    _isolated_runtime_settings,
):
    _write_settings(
        _isolated_runtime_settings,
        {"governance.approval_mode": "all"},
    )
    first = AuthorityGateway._runtime_confirmation_gate(
        tool_name="shell_command",
        args={"cmd": ["date"]},
        source="desktop_ui",
        risk_level="critical",
        effect_scope="subprocess",
    )
    assert first is not None
    challenge_id = first.constraints["confirmation_challenge_id"]
    get_action_confirmation_registry().authorize(challenge_id)

    assert AuthorityGateway._runtime_confirmation_gate(
        tool_name="shell_command",
        args={"cmd": ["date"]},
        source="desktop_ui",
        risk_level="critical",
        effect_scope="subprocess",
    ) is None
    second_attempt = AuthorityGateway._runtime_confirmation_gate(
        tool_name="shell_command",
        args={"cmd": ["date"]},
        source="desktop_ui",
        risk_level="critical",
        effect_scope="subprocess",
    )
    changed_action = AuthorityGateway._runtime_confirmation_gate(
        tool_name="shell_command",
        args={"cmd": ["whoami"]},
        source="desktop_ui",
        risk_level="critical",
        effect_scope="subprocess",
    )
    assert second_attempt is not None
    assert changed_action is not None


def test_action_confirmation_expiry_and_atomic_single_consumer():
    now = [100.0]
    registry = ActionConfirmationRegistry(
        clock=lambda: now[0],
        pending_ttl_seconds=5.0,
        authorized_ttl_seconds=2.0,
    )
    fingerprint = action_confirmation_fingerprint(
        tool_name="desktop_task",
        arguments={"objective": "open Notes"},
        source="desktop_ui",
        risk_level="high",
        effect_scope="foreground_desktop_control",
    )
    challenge = registry.issue(
        action_fingerprint=fingerprint,
        tool_name="desktop_task",
    )
    registry.authorize(challenge["challenge_id"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _index: registry.consume_authorized(fingerprint), range(2)))
    assert sum(1 for approved, _detail in outcomes if approved) == 1

    replacement = registry.issue(
        action_fingerprint=fingerprint,
        tool_name="desktop_task",
    )
    registry.authorize(replacement["challenge_id"])
    now[0] += 3.0
    assert registry.consume_authorized(fingerprint) == (
        False,
        "action_confirmation_missing_or_expired",
    )


def test_action_confirmation_cancel_removes_unconsumed_authorization():
    registry = ActionConfirmationRegistry()
    fingerprint = action_confirmation_fingerprint(
        tool_name="desktop_task",
        arguments={"objective": "open Notes"},
        source="desktop_ui",
        risk_level="high",
        effect_scope="foreground_desktop_control",
    )
    challenge = registry.issue(
        action_fingerprint=fingerprint,
        tool_name="desktop_task",
    )
    registry.authorize(challenge["challenge_id"])

    assert registry.cancel(challenge["challenge_id"]) is True
    assert registry.cancel(challenge["challenge_id"]) is False
    assert registry.consume_authorized(fingerprint)[0] is False


def test_capability_boundary_preserves_actionable_confirmation_contract():
    handle = SimpleNamespace(
        decision=SimpleNamespace(
            reason="runtime_setting_user_confirmation_required:approval_mode_all"
        ),
        constraints={
            "requires_user_confirmation": True,
            "approval_mode": "all",
            "risk_level": "low",
            "effect_scope": "read_only",
            "confirmation_endpoint": "/api/settings/auth/fresh",
            "confirmation_challenge_id": "action-confirm-test",
            "confirmation_pending_expires_in_seconds": 300.0,
            "confirmation_one_time": True,
            "confirmation_action_bound": True,
            "confirmation_does_not_bypass_governance": True,
        },
    )

    payload = CapabilityEngine._constitutional_denial_payload(handle)

    assert payload["ok"] is False
    assert payload["status"] == "approval_required"
    assert payload["approval"] == {
        "required": True,
        "mode": "all",
        "risk_level": "low",
        "effect_scope": "read_only",
        "confirmation_endpoint": "/api/settings/auth/fresh",
        "challenge_id": "action-confirm-test",
        "pending_expires_in_seconds": 300.0,
        "one_time": True,
        "action_bound": True,
        "confirmation_does_not_bypass_governance": True,
    }
    assert "Executive veto" not in payload["error"]


@pytest.mark.asyncio
async def test_none_mode_still_reaches_standing_authority_denial(
    _isolated_runtime_settings,
    monkeypatch,
):
    _write_settings(
        _isolated_runtime_settings,
        {
            "autonomy.actions_enabled": True,
            "governance.approval_mode": "none",
        },
    )
    gateway = object.__new__(AuthorityGateway)
    lease_calls = []

    class _StandingAuthority:
        async def issue_child_lease(self, *_args, **_kwargs):
            lease_calls.append((_args, _kwargs))
            return SimpleNamespace(
                approved=False,
                context={},
                reason="no_matching_standing_grant",
                receipt_id="denial-receipt",
                token=None,
            )

    gateway._standing_authority = _StandingAuthority()
    monkeypatch.setattr(gateway, "_social_governance_gate", lambda *_args: None)
    monkeypatch.setattr(gateway, "active_user_presence_context", lambda: {})
    monkeypatch.setattr(
        gateway,
        "_will_gate",
        lambda *_args, **_kwargs: (None, SimpleNamespace(receipt_id="will-ok")),
    )
    monkeypatch.setattr(
        "core.executive.authority_gateway.resolve_execution_effect_scope",
        lambda *_args, **_kwargs: "external_io",
    )
    monkeypatch.setattr(
        "core.executive.authority_gateway.classify_execution_risk",
        lambda *_args, **_kwargs: "high",
    )

    decision = await gateway.authorize_tool_execution(
        "email_adapter",
        {"to": "owner@example.com"},
        source="desktop_ui",
        context={"explicit_user_request": True},
    )

    assert lease_calls
    assert decision.approved is False
    assert decision.reason == "standing_authority_denied:no_matching_standing_grant"
