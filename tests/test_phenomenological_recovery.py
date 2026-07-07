import pytest

from core.consciousness.phenomenological_experiencer import (
    AttentionSchema,
    ExperientialContinuityEngine,
    PhenomenalSelfModel,
    PhenomenologicalExperiencer,
)
from core.runtime.errors import get_degradation_tracker


@pytest.fixture(autouse=True)
def _reset_degradation_tracker():
    get_degradation_tracker().reset()
    yield
    get_degradation_tracker().reset()


@pytest.mark.asyncio
async def test_deep_narrative_failure_updates_recovery_state(monkeypatch):
    class _BrokenRouter:
        async def think(self, **_kwargs):
            self.think_calls = getattr(self, "think_calls", 0) + 1
            raise RuntimeError("narrative backend offline")

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: _BrokenRouter() if name == "llm_router" else default,
    )
    monkeypatch.setattr(
        "core.consciousness.phenomenological_experiencer._phenomenology_background_deferral_reason",
        lambda: "",
    )

    psm = PhenomenalSelfModel()
    report = await psm.run_deep_narrative_update(
        continuity=ExperientialContinuityEngine(),
        schema=AttentionSchema("a hard problem", "aware", "cognitive", 0.7),
        qualia=[],
        current_emotion="curious",
        dominant_motivation="needs_to_reason",
    )

    assert report == psm._present_description
    assert psm.to_dict()["narrative_failure_streak"] == 1
    last = get_degradation_tracker().recent(subsystem="phenomenological_narrative")[-1]
    assert last.action == "retained previous present-description after narrative update failed"


@pytest.mark.asyncio
async def test_deep_narrative_timeout_is_bounded_and_debug_only(monkeypatch):
    class _SlowRouter:
        async def think(self, **kwargs):
            self.kwargs = kwargs
            await __import__("asyncio").sleep(5.0)

    router = _SlowRouter()
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: router if name == "llm_router" else default,
    )
    monkeypatch.setattr(
        "core.consciousness.phenomenological_experiencer._phenomenology_background_deferral_reason",
        lambda: "",
    )
    monkeypatch.setattr(
        "core.consciousness.phenomenological_experiencer.PSM_NARRATIVE_TIMEOUT_S",
        0.01,
    )
    monkeypatch.setattr(
        "core.consciousness.phenomenological_experiencer.PSM_NARRATIVE_MAX_TOKENS",
        64,
    )

    psm = PhenomenalSelfModel()
    report = await psm.run_deep_narrative_update(
        continuity=ExperientialContinuityEngine(),
        schema=AttentionSchema("a live chat turn", "aware", "cognitive", 0.7),
        qualia=[],
        current_emotion="focused",
        dominant_motivation="needs_to_answer",
    )

    assert report == psm._present_description
    assert psm.to_dict()["narrative_failure_streak"] == 1
    assert router.kwargs["max_tokens"] == 64
    last = get_degradation_tracker().recent(subsystem="phenomenological_narrative")[-1]
    assert last.severity == "debug"
    assert last.action == "bounded opportunistic narrative update and retained previous present-description"


@pytest.mark.asyncio
async def test_witness_failure_updates_recovery_state(monkeypatch):
    class _BrokenRouter:
        async def think(self, **_kwargs):
            self.think_calls = getattr(self, "think_calls", 0) + 1
            raise RuntimeError("witness backend offline")

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: _BrokenRouter() if name == "llm_router" else default,
    )
    monkeypatch.setattr(
        "core.consciousness.phenomenological_experiencer._phenomenology_background_deferral_reason",
        lambda: "",
    )

    psm = PhenomenalSelfModel()
    observation = await psm.run_witness_reflection(
        continuity=ExperientialContinuityEngine(),
        credit_summary="credit assignment active",
    )

    assert observation == ""
    assert psm.to_dict()["witness_failure_streak"] == 1
    last = get_degradation_tracker().recent(subsystem="phenomenological_witness")[-1]
    assert last.action == "retained previous witness observation after reflection update failed"


@pytest.mark.asyncio
async def test_witness_timeout_is_bounded_and_debug_only(monkeypatch):
    class _SlowRouter:
        async def think(self, **kwargs):
            self.kwargs = kwargs
            await __import__("asyncio").sleep(5.0)

    router = _SlowRouter()
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: router if name == "llm_router" else default,
    )
    monkeypatch.setattr(
        "core.consciousness.phenomenological_experiencer._phenomenology_background_deferral_reason",
        lambda: "",
    )
    monkeypatch.setattr(
        "core.consciousness.phenomenological_experiencer.PSM_WITNESS_TIMEOUT_S",
        0.01,
    )
    monkeypatch.setattr(
        "core.consciousness.phenomenological_experiencer.PSM_WITNESS_MAX_TOKENS",
        48,
    )

    psm = PhenomenalSelfModel()
    observation = await psm.run_witness_reflection(
        continuity=ExperientialContinuityEngine(),
        credit_summary="credit assignment active",
    )

    assert observation == ""
    assert psm.to_dict()["witness_failure_streak"] == 1
    assert router.kwargs["max_tokens"] == 48
    last = get_degradation_tracker().recent(subsystem="phenomenological_witness")[-1]
    assert last.severity == "debug"
    assert last.action == "bounded opportunistic witness update and retained previous observation"


@pytest.mark.asyncio
async def test_slow_phenomenology_defers_during_proof_without_router_call(monkeypatch):
    class _UnexpectedRouter:
        think_calls = 0

        async def think(self, **_kwargs):
            self.think_calls += 1
            raise AssertionError("background phenomenology should not call the router during proof")

    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: _UnexpectedRouter() if name == "llm_router" else default,
    )

    psm = PhenomenalSelfModel()
    report = await psm.run_deep_narrative_update(
        continuity=ExperientialContinuityEngine(),
        schema=AttentionSchema("a proof run", "aware", "cognitive", 0.7),
        qualia=[],
        current_emotion="focused",
        dominant_motivation="needs_to_reason",
    )
    witness = await psm.run_witness_reflection(
        continuity=ExperientialContinuityEngine(),
        credit_summary="credit assignment active",
    )

    assert report == psm._present_description
    assert witness == ""
    assert psm.to_dict()["narrative_failure_streak"] == 0
    assert psm.to_dict()["witness_failure_streak"] == 0
    assert psm.to_dict()["last_narrative_error"] == "deferred:proof_run_active"
    assert psm.to_dict()["last_witness_error"] == "deferred:proof_run_active"


@pytest.mark.asyncio
async def test_experiencer_update_loop_uses_adaptive_backoff(monkeypatch, tmp_path):
    sleep_delays: list[float] = []
    experiencer = PhenomenologicalExperiencer(save_dir=str(tmp_path))

    async def _broken_narrative():
        _broken_narrative.calls = getattr(_broken_narrative, "calls", 0) + 1
        raise RuntimeError("phenomenal narrative task failed")

    async def _stop_after_sleep(delay):
        if delay == 0:
            return
        sleep_delays.append(delay)
        experiencer._running = False

    experiencer._running = True
    experiencer._current_schema = AttentionSchema("the task", "aware", "cognitive", 0.8)
    experiencer._run_deep_narrative = _broken_narrative
    monkeypatch.setattr(
        "core.consciousness.phenomenological_experiencer.asyncio.sleep",
        _stop_after_sleep,
    )
    monkeypatch.setattr(
        "core.consciousness.phenomenological_experiencer.BOOT_GRACE_PERIOD_S",
        0,
    )
    monkeypatch.setattr(
        "core.consciousness.phenomenological_experiencer._phenomenology_background_deferral_reason",
        lambda: "",
    )

    await experiencer._update_loop()

    assert sleep_delays == [5.0]
    status = experiencer.get_status()
    assert status["update_failure_streak"] == 1
    assert "RuntimeError" in status["last_update_error"]
    last = get_degradation_tracker().recent(subsystem="phenomenological_experiencer")[-1]
    assert last.action == "kept phenomenological update loop alive with adaptive backoff"
    assert last.severity == "degraded"
