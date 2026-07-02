import asyncio
import time
from types import SimpleNamespace

import pytest

from core.runtime.errors import get_degradation_tracker
from core.autonomy import research_cycle as research_module
from core.autonomy.research_cycle import ResearchCycle, ResearchRecord


def test_research_cycle_respects_background_policy_gate_before_starting(monkeypatch):
    cycle = ResearchCycle.__new__(ResearchCycle)
    cycle.orchestrator = SimpleNamespace(_last_user_interaction_time=0.0, status=SimpleNamespace(is_processing=False))
    cycle._last_cycle_mono = 0.0
    cycle._get_state = lambda: SimpleNamespace(
        motivation=SimpleNamespace(budgets={"energy": {"level": 100.0}}),
        affect=SimpleNamespace(curiosity=0.8),
        cognition=SimpleNamespace(pending_initiatives=[{"goal": "Research continuity"}]),
    )

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "no_user_anchor",
    )

    assert cycle._should_run() is False


def test_research_cycle_defers_during_boot_grace(monkeypatch):
    cycle = ResearchCycle.__new__(ResearchCycle)
    cycle.orchestrator = SimpleNamespace(_last_user_interaction_time=0.0, status=SimpleNamespace(is_processing=False))
    cycle._last_cycle_mono = 0.0
    cycle._started_mono = time.monotonic()
    cycle._get_state = lambda: SimpleNamespace(
        motivation=SimpleNamespace(budgets={"energy": {"level": 100.0}}),
        affect=SimpleNamespace(curiosity=0.8),
        cognition=SimpleNamespace(pending_initiatives=[{"goal": "Research continuity"}]),
    )

    monkeypatch.setenv("AURA_RESEARCH_BOOT_GRACE_S", "300")
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "",
    )

    assert cycle._should_run() is False


def test_research_cycle_defers_when_background_policy_gate_fails(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        research_module,
        "_record_research_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )

    cycle = ResearchCycle.__new__(ResearchCycle)
    cycle.orchestrator = SimpleNamespace(_last_user_interaction_time=0.0, status=SimpleNamespace(is_processing=False))
    cycle._last_cycle_mono = 0.0
    cycle._started_mono = time.monotonic() - 1000
    cycle._last_cycle_error = None
    cycle._get_state = lambda: SimpleNamespace(
        motivation=SimpleNamespace(budgets={"energy": {"level": 100.0}}),
        affect=SimpleNamespace(curiosity=0.8),
        cognition=SimpleNamespace(pending_initiatives=[{"goal": "Research continuity"}]),
    )

    monkeypatch.setenv("AURA_RESEARCH_BOOT_GRACE_S", "0")
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("policy offline")),
    )

    assert cycle._should_run() is False
    assert cycle._last_cycle_error == "RuntimeError: policy offline"
    assert recorded[0][1]["action"] == (
        "deferred autonomous research because background policy gate was unavailable"
    )


def test_research_cycle_save_record_keeps_memory_copy_when_history_append_fails(
    monkeypatch,
    tmp_path,
):
    recorded = []
    monkeypatch.setattr(
        research_module,
        "_record_research_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )

    cycle = ResearchCycle.__new__(ResearchCycle)
    cycle._record_path = tmp_path
    cycle._last_cycle_error = None
    record = ResearchRecord(
        record_id="abc123",
        drive="curiosity",
        goal="Research continuity",
        findings=["Continuity depends on replayable records."],
        identity_impact="I remember why durability matters.",
        affect_before={},
        affect_after={},
    )

    cycle._save_record(record)

    assert cycle._last_cycle_error.startswith("IsADirectoryError:")
    assert recorded[0][1]["action"] == (
        "kept in-memory research record after durable history append failed"
    )


@pytest.mark.asyncio
async def test_research_cycle_extract_findings_falls_back_when_llm_json_is_malformed(
    monkeypatch,
):
    recorded = []
    monkeypatch.setattr(
        research_module,
        "_record_research_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )

    class FakeLLM:
        async def think(self, _prompt):
            return "[not valid json]"

    kernel = SimpleNamespace(organs={"llm": SimpleNamespace(get_instance=lambda: FakeLLM())})
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: kernel if name == "aura_kernel" else default,
    )

    cycle = ResearchCycle.__new__(ResearchCycle)
    result = await cycle._extract_findings(
        "A replayable log gives external evaluators enough detail to reproduce decisions.",
        "Research replayable evaluation logs",
    )

    assert result == [
        "A replayable log gives external evaluators enough detail to reproduce decisions"
    ]
    assert recorded[0][1]["action"] == (
        "used sentence-splitting findings fallback after LLM extraction failed"
    )


@pytest.mark.asyncio
async def test_start_research_daemon_registers_recoverable_service_policy(
    service_container,
    monkeypatch,
):
    monkeypatch.setenv("AURA_RESEARCH_BOOT_GRACE_S", "300")

    cycle = await research_module.start_research_daemon(SimpleNamespace())
    try:
        desc = service_container._services["research_cycle"]
        assert desc.required is True
        assert desc.failure_policy == "degrade_with_receipt"
        assert "full desktop autonomy" in desc.required_for
        assert cycle.is_alive() is True
    finally:
        await cycle.stop()


@pytest.mark.asyncio
async def test_research_cycle_start_recovers_dead_daemon_task(monkeypatch):
    async def failed_daemon():
        raise RuntimeError("research timeout")

    old_task = asyncio.create_task(failed_daemon(), name="aura.research_cycle")
    await asyncio.sleep(0)

    release = asyncio.Event()

    async def replacement_daemon():
        await release.wait()

    cycle = ResearchCycle.__new__(ResearchCycle)
    cycle._running = True
    cycle._task = old_task
    cycle._daemon = replacement_daemon

    await cycle.start()
    try:
        assert cycle._task is not old_task
        assert cycle.is_alive() is True
    finally:
        release.set()
        await cycle.stop()


def test_research_degradation_self_heals_stale_fail_closed_descriptor(service_container):
    get_degradation_tracker().reset()
    service_container.register_instance(
        "research_cycle",
        object(),
        required=True,
        failure_policy="fail-closed",
    )

    research_module._record_research_degradation(
        TimeoutError("temporary search timeout"),
        action="deferred one background research cycle after a bounded timeout",
    )

    desc = service_container._services["research_cycle"]
    records = get_degradation_tracker().recent(subsystem="research_cycle", limit=1)
    assert desc.failure_policy == "degrade_with_receipt"
    assert records
    assert records[-1].severity == "warning"
