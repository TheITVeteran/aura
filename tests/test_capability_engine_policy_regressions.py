from types import SimpleNamespace

import pytest

from core.capability_engine import CapabilityEngine, SkillMetadata
from core.container import ServiceContainer
from core.guardians.user_advocate import UserAdvocateWatchdog
from core.sim.outcome_simulator import OutcomeSimulationEngine


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


def test_web_search_execution_scope_is_read_only_not_external_mutation():
    engine = _engine_with_skill("web_search")
    meta = engine.skills["web_search"]

    scope = engine._effect_scope_for_execution(
        "web_search",
        meta,
        {"query": "latest climate research"},
        {"origin": "background"},
    )

    assert scope == "read_only"
    assert engine._edi_risk_for("web_search", meta, {"query": "latest climate research"}, scope) == "low"
    description = CapabilityEngine._action_description_for_user_advocate(
        "web_search",
        {"query": "latest climate research"},
        scope,
    )
    assert "read-only web_search information retrieval" in description


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


def test_user_visible_web_interlocutor_auto_confirms_foreground_request():
    assert (
        CapabilityEngine._user_advocate_auto_confirmed_for(
            "web_interlocutor",
            {
                "origin": "desktop_ui",
                "route": "chat.live_runtime_proof.web_interlocutor",
                "foreground_request": True,
                "user_requested_action": True,
                "user_visible_browser_action": True,
            },
            "desktop_ui",
            "foreground_browser_dialogue",
        )
        is True
    )


def test_background_web_interlocutor_does_not_auto_confirm():
    assert (
        CapabilityEngine._user_advocate_auto_confirmed_for(
            "web_interlocutor",
            {
                "foreground_request": True,
                "user_requested_action": True,
                "user_visible_browser_action": True,
            },
            "background",
            "foreground_browser_dialogue",
        )
        is False
    )


@pytest.mark.asyncio
async def test_execute_with_retry_uses_skill_execution_timeout(monkeypatch):
    engine = _engine_with_skill("web_interlocutor")
    engine.max_retries = 1
    engine.timeout = 1.0

    class SlowVisibleSkill:
        async def safe_execute(self, params, context):
            return {"ok": True, "status": "completed"}

    observed_timeouts = []

    async def fake_wait_for(coro, timeout):
        observed_timeouts.append(timeout)
        return await coro

    monkeypatch.setattr("core.capability_engine.asyncio.wait_for", fake_wait_for)

    result = await engine._execute_with_retry(
        SlowVisibleSkill(),
        "web_interlocutor",
        {},
        {},
        execution_timeout=420.0,
    )

    assert result["ok"] is True
    assert observed_timeouts == [420.0]


@pytest.mark.asyncio
async def test_execute_with_retry_reports_blank_timeout_with_skill_context(monkeypatch):
    engine = _engine_with_skill("web_interlocutor")
    engine.max_retries = 1
    engine.timeout = 1.0

    class TimingOutVisibleSkill:
        async def safe_execute(self, params, context):
            return {"ok": True}

    async def fake_wait_for(coro, timeout):
        coro.close()
        raise TimeoutError()

    monkeypatch.setattr("core.capability_engine.asyncio.wait_for", fake_wait_for)

    result = await engine._execute_with_retry(
        TimingOutVisibleSkill(),
        "web_interlocutor",
        {},
        {},
        execution_timeout=420.0,
    )

    assert result["ok"] is False
    assert result["error"] == "web_interlocutor timed out after 420.0s"


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


def test_auto_refactor_read_only_scan_presents_user_benefit_to_guardian():
    params = {"path": ".", "run_tests": False}
    desc = CapabilityEngine._action_description_for_user_advocate(
        "auto_refactor",
        params,
        "read_only",
    )
    benefit = CapabilityEngine._user_benefit_for_execution(
        "auto_refactor",
        params,
        {"origin": "overt_action_loop"},
        "overt_action_loop",
        "read_only",
    )

    review = UserAdvocateWatchdog().review_action(
        {
            "description": desc,
            "irreversible": CapabilityEngine._user_advocate_irreversible_for(
                "auto_refactor",
                params,
                "low",
                "read_only",
            ),
            "confirmed": False,
            "user_benefit": benefit,
            "explanation": "skill auto_refactor",
        }
    )

    assert "read-only" in desc
    assert "no source writes" in desc
    assert benefit
    assert review.verdict == "for_user"
    assert review.flags == []


def test_outcome_simulator_allows_read_only_external_web_search():
    result = OutcomeSimulationEngine().assess_fast(
        "web_search [read_only_external_io] {'query': 'latest research on Europa'}",
        context={
            "effect_scope": "read_only_external_io",
            "skill_name": "web_search",
            "tool_name": "web_search",
        },
    )

    assert result.recommendation == "act"
    assert result.worst_case_harm < OutcomeSimulationEngine.HOLD_HARM_THRESHOLD


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
async def test_background_web_search_uses_lightweight_io_preflight(monkeypatch):
    engine = _engine_with_skill("web_search")
    calls: list[dict] = []

    def _policy(*args, **kwargs):
        calls.append(dict(kwargs))
        return (_ for _ in ()).throw(RuntimeError("policy offline"))

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        _policy,
    )

    result = await CapabilityEngine.execute(
        engine,
        "web_search",
        {"query": "latest Turing Award"},
        context={"origin": "background"},
    )

    assert result["ok"] is False
    assert result["status"] == "deferred"
    assert calls
    assert calls[0]["min_idle_seconds"] == pytest.approx(30.0)
    assert calls[0]["max_memory_percent"] == pytest.approx(84.0)
    assert calls[0]["max_failure_pressure"] == pytest.approx(0.45)


@pytest.mark.asyncio
async def test_background_browser_dialogue_still_uses_strict_foreground_preflight(monkeypatch):
    engine = _engine_with_skill("web_interlocutor")
    calls: list[dict] = []

    def _policy(*args, **kwargs):
        calls.append(dict(kwargs))
        return (_ for _ in ()).throw(RuntimeError("policy offline"))

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        _policy,
    )

    result = await CapabilityEngine.execute(
        engine,
        "web_interlocutor",
        {"topic": "memory and agency"},
        context={"origin": "background"},
    )

    assert result["ok"] is False
    assert result["status"] == "deferred"
    assert calls
    assert calls[0]["min_idle_seconds"] == pytest.approx(600.0)
    assert calls[0]["max_memory_percent"] == pytest.approx(72.0)
    assert calls[0]["max_failure_pressure"] == pytest.approx(0.20)


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
