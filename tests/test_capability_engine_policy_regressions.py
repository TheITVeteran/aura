from types import SimpleNamespace

import pytest

from core.capability_engine import CapabilityEngine, SkillMetadata
from core.container import ServiceContainer


def _quiet_logger():
    return SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )


def _engine_with_skill(skill_name: str, *, metabolic_cost: int = 1) -> CapabilityEngine:
    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.logger = _quiet_logger()
    engine.error_boundary = lambda fn: fn
    engine.skills = {
        skill_name: SkillMetadata(
            name=skill_name,
            description="policy regression probe",
            skill_class=lambda: object(),
            metabolic_cost=metabolic_cost,
        )
    }
    engine.instances = {}
    engine.sandbox = None
    engine.rosetta_stone = None
    engine.temporal = None
    engine.orchestrator = SimpleNamespace(mycelium=None)
    engine.skill_last_errors = {}
    engine._emit_skill_status = lambda *args, **kwargs: None
    engine.max_retries = 1
    engine.retry_delay = 0.0
    engine.timeout = 1.0
    return engine


def test_stateless_sandbox_compute_is_not_irreversible_for_user_advocate():
    assert (
        CapabilityEngine._user_advocate_irreversible_for(
            "run_code",
            {"code": "print(120)", "stateful": False},
            "high",
            "sandboxed_compute",
        )
        is False
    )


def test_stateful_code_still_requires_irreversible_confirmation():
    assert (
        CapabilityEngine._user_advocate_irreversible_for(
            "run_code",
            {"code": "x = 1", "stateful": True},
            "critical",
            "sandboxed_compute",
        )
        is True
    )


def test_foreground_desktop_control_still_requires_irreversible_confirmation():
    assert (
        CapabilityEngine._user_advocate_irreversible_for(
            "computer_use",
            {"action": "type", "target": "hello"},
            "medium",
            "foreground_desktop_control",
        )
        is True
    )


def test_user_visible_desktop_task_auto_confirms_foreground_request():
    assert (
        CapabilityEngine._user_advocate_auto_confirmed_for(
            "desktop_task",
            {
                "origin": "desktop_ui",
                "route": "chat.live_runtime_proof.desktop_task",
                "user_visible_desktop_action": True,
                "local_desktop_action": True,
            },
            "desktop_ui",
            "foreground_desktop_control",
        )
        is True
    )


def test_background_desktop_task_does_not_auto_confirm():
    assert (
        CapabilityEngine._user_advocate_auto_confirmed_for(
            "desktop_task",
            {
                "user_visible_desktop_action": True,
                "local_desktop_action": True,
            },
            "background",
            "foreground_desktop_control",
        )
        is False
    )


def test_auto_refactor_scan_is_read_only_not_privileged_mutation():
    engine = _engine_with_skill("auto_refactor")
    meta = engine.skills["auto_refactor"]

    assert (
        engine._effect_scope_for_execution(
            "auto_refactor",
            meta,
            {"path": ".", "run_tests": False},
            {"origin": "overt_action_loop"},
        )
        == "read_only"
    )
    assert (
        engine._edi_risk_for(
            "auto_refactor",
            meta,
            {"path": ".", "run_tests": False},
            "read_only",
        )
        == "low"
    )
    assert (
        CapabilityEngine._user_advocate_irreversible_for(
            "auto_refactor",
            {"path": ".", "run_tests": False},
            "low",
            "read_only",
        )
        is False
    )


def test_auto_refactor_mutation_remains_privileged():
    engine = _engine_with_skill("auto_refactor")
    meta = engine.skills["auto_refactor"]

    scope = engine._effect_scope_for_execution(
        "auto_refactor",
        meta,
        {"path": ".", "apply": True},
        {"origin": "overt_action_loop"},
    )

    assert scope == "privileged_mutation"
    assert (
        engine._edi_risk_for(
            "auto_refactor",
            meta,
            {"path": ".", "apply": True},
            scope,
        )
        == "critical"
    )
    assert (
        CapabilityEngine._user_advocate_irreversible_for(
            "auto_refactor",
            {"path": ".", "apply": True},
            "critical",
            scope,
        )
        is True
    )


@pytest.mark.asyncio
async def test_foreground_exclusive_background_tool_defers_when_policy_fails(monkeypatch):
    engine = _engine_with_skill("web_search")

    def _policy_down(*args, **kwargs):
        return (_ for _ in ()).throw(RuntimeError("policy offline"))

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        _policy_down,
    )

    result = await CapabilityEngine.execute(
        engine,
        "web_search",
        {"query": "latest vulnerability"},
        context={"origin": "background"},
    )

    assert result["ok"] is False
    assert result["status"] == "deferred"
    assert result["reason"] == "background_policy_unavailable"


@pytest.mark.asyncio
async def test_high_cost_tool_blocks_when_self_preservation_check_fails(monkeypatch):
    ServiceContainer.clear()
    engine = _engine_with_skill("sovereign_terminal", metabolic_cost=3)

    monkeypatch.setattr(
        "core.capability_engine.ServiceContainer.has", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("constitution offline")),
    )

    def _metabolism_down(*args, **kwargs):
        return (_ for _ in ()).throw(RuntimeError("metabolism offline"))

    monkeypatch.setattr("core.capability_engine.resolve_metabolic_monitor", _metabolism_down)
    monkeypatch.setattr(
        "core.capability_engine.resolve_state_repository", lambda default=None: None
    )

    try:
        result = await CapabilityEngine.execute(
            engine,
            "sovereign_terminal",
            {"command": "stress test"},
            context={"origin": "background"},
        )
    finally:
        ServiceContainer.clear()

    assert result["ok"] is False
    assert result["status"] == "blocked_by_self_preservation_unavailable"
