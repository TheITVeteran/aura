"""Wiring the supervised cognitive loop into Aura's live Will."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.agency import cognitive_loop_pathway as clp


class _Container:
    def __init__(self, services):
        self._s = services

    def get(self, name, default=None):
        return self._s.get(name, default)


class _Router:
    def __init__(self, reply="the answer is 42"):
        self.reply = reply
        self.prompts = []
        self.options = []

    async def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.options.append(dict(kwargs))
        return self.reply


class _Memory:
    def search_sync(self, query, limit=5):
        return [{"content": "a relevant fact"}]


class _Agency:
    def __init__(self, monologue=""):
        self._current_monologue = monologue
        self.registered = []

    def register_pathway_hook(self, pathway, provider):
        if pathway != "cognitive_loop":
            raise ValueError("unknown pathway")
        self.registered.append(pathway)


# ── Gated OFF by default -- the live instance is unchanged ───────────────


def test_on_by_default_registers_the_pathways(monkeypatch):
    """Owner's decision: on unless explicitly disabled."""
    monkeypatch.delenv(clp.ENABLE_FLAG, raising=False)
    agency = _Agency()
    receipt = clp.register_if_enabled(agency)
    assert receipt["enabled"] is True
    assert set(agency.registered) == set(clp.TARGET_PATHWAYS)


def test_explicit_disable_registers_nothing(monkeypatch):
    monkeypatch.setenv(clp.ENABLE_FLAG, "0")
    agency = _Agency()
    receipt = clp.register_if_enabled(agency)
    assert receipt["enabled"] is False
    assert agency.registered == []


def test_enabled_registers_on_the_target_pathways(monkeypatch):
    monkeypatch.setenv(clp.ENABLE_FLAG, "1")
    agency = _Agency()
    receipt = clp.register_if_enabled(agency)
    assert receipt["enabled"] is True
    assert set(receipt["registered"]) == set(clp.TARGET_PATHWAYS)
    assert set(agency.registered) == set(clp.TARGET_PATHWAYS)


# ── Builds from live organs, degrades honestly without them ─────────────


def test_build_returns_none_without_an_llm():
    assert clp.build_live_loop(_Container({})) is None


def test_build_succeeds_with_router_and_memory():
    loop = clp.build_live_loop(_Container({
        "llm_router": _Router(), "memory_facade": _Memory(),
    }))
    assert loop is not None
    # retrieval producer wired from real memory
    assert len(loop.composer.producers) == 1


def test_build_works_with_llm_but_no_memory():
    """Deliberation without retrieval is degraded, not broken."""
    loop = clp.build_live_loop(_Container({"llm_router": _Router()}))
    assert loop is not None
    assert loop.composer.producers == []


def test_query_falls_back_to_pending_agency_goal():
    agency = _Agency(monologue="")
    agency.state = SimpleNamespace(
        pending_goals=[
            {"text": "already finished", "status": "completed"},
            {"description": "Understand the current memory fault", "status": "pending"},
        ]
    )

    assert clp._derive_query(agency) == "Understand the current memory fault"


# ── The live async loop runs end to end ─────────────────────────────────


def test_router_deliberator_bridges_async_generation():
    router = _Router("stepwise... 42")
    delib = clp._RouterDeliberator(router)
    out = asyncio.run(delib.deliberate("q", ["a fact"]))
    assert out == "stepwise... 42"
    # material and step-by-step framing reach the model
    assert "a fact" in router.prompts[0]
    assert "step by step" in router.prompts[0]
    assert router.options == [{
        "origin": "cognitive_loop_pathway",
        "purpose": "autonomous_internal_deliberation",
        "is_background": True,
        "foreground_request": False,
    }]


@pytest.mark.asyncio
async def test_provider_proposes_from_the_monologue_without_blocking_pulse(monkeypatch):
    monkeypatch.setenv(clp.ENABLE_FLAG, "1")
    monkeypatch.setattr(clp, "COOLDOWN_SECONDS", 0.0)
    router = _Router("42")
    container = _Container({"llm_router": router, "memory_facade": _Memory()})
    loop = clp.build_live_loop(container)
    monkeypatch.setattr(clp, "build_live_loop", lambda *a, **k: loop)
    async def _accepted(**_kwargs):
        return {"admitted": True, "reason": "accepted_for_competition"}
    monkeypatch.setattr(clp, "_publish_result_to_workspace", _accepted)
    agency = _Agency(monologue="I wonder how many moons Jupiter has")
    scheduled = await clp.cognitive_loop_provider(
        pathway="cognitive_loop", now=0.0, idle_seconds=0.0, agency=agency,
    )
    assert scheduled is None
    assert agency._cognitive_loop_task.done() is False
    await agency._cognitive_loop_task
    proposal = await clp.cognitive_loop_provider(
        pathway="cognitive_loop", now=0.0, idle_seconds=0.0, agency=agency,
    )
    assert proposal is not None
    assert proposal["type"] == "internal_reflection"
    assert proposal["internal_only"] is True
    assert proposal["trust"] == "unverified_hypothesis"
    assert proposal["content"] == "42"
    # unverified (no verifier organ) -> lower priority, never over-trusted
    assert proposal["verified"] is False
    assert proposal["priority"] == 0.3
    assert proposal["workspace_admitted"] is True
    assert len(router.prompts) == 1
    assert proposal["attempts"] == 1


def test_provider_proposes_nothing_without_a_query(monkeypatch):
    container = _Container({"llm_router": _Router()})
    loop = clp.build_live_loop(container)
    monkeypatch.setattr(clp, "build_live_loop", lambda *a, **k: loop)
    monkeypatch.setattr(clp, "COOLDOWN_SECONDS", 0.0)
    agency = _Agency(monologue="")  # nothing on her mind
    proposal = asyncio.run(clp.cognitive_loop_provider(
        pathway="cognitive_loop", now=0.0, idle_seconds=0.0, agency=agency,
    ))
    assert proposal is None


def test_provider_proposes_nothing_when_organs_missing(monkeypatch):
    monkeypatch.setattr(clp, "build_live_loop", lambda: None)
    monkeypatch.setattr(clp, "COOLDOWN_SECONDS", 0.0)
    proposal = asyncio.run(clp.cognitive_loop_provider(
        pathway="cognitive_loop", now=0.0, idle_seconds=0.0,
        agency=_Agency(monologue="a real question here"),
    ))
    assert proposal is None


@pytest.mark.asyncio
async def test_cooldown_prevents_firing_the_loop_every_pulse(monkeypatch):
    """On-by-default must not hammer the live 32B: one run, then a cooldown."""
    monkeypatch.setattr(clp, "COOLDOWN_SECONDS", 180.0)
    router = _Router("42")
    container = _Container({"llm_router": router, "memory_facade": _Memory()})
    loop = clp.build_live_loop(container)
    monkeypatch.setattr(clp, "build_live_loop", lambda *a, **k: loop)
    async def _accepted(**_kwargs):
        return {"admitted": True, "reason": "accepted_for_competition"}
    monkeypatch.setattr(clp, "_publish_result_to_workspace", _accepted)
    agency = _Agency(monologue="a real question to reason about")

    scheduled = await clp.cognitive_loop_provider(
        pathway="cognitive_loop", now=1000.0, idle_seconds=0.0, agency=agency)
    assert scheduled is None
    await agency._cognitive_loop_task
    first = await clp.cognitive_loop_provider(
        pathway="cognitive_loop", now=1000.0, idle_seconds=0.0, agency=agency)
    assert first is not None
    # a pulse 5s later is inside the cooldown -> no re-fire
    second = await clp.cognitive_loop_provider(
        pathway="cognitive_loop", now=1005.0, idle_seconds=0.0, agency=agency)
    assert second is None
    # well past the cooldown -> fires again
    scheduled_again = await clp.cognitive_loop_provider(
        pathway="cognitive_loop", now=1200.0, idle_seconds=0.0, agency=agency)
    assert scheduled_again is None
    await agency._cognitive_loop_task
    third = await clp.cognitive_loop_provider(
        pathway="cognitive_loop", now=1200.0, idle_seconds=0.0, agency=agency)
    assert third is not None
    assert len(router.prompts) == 2


@pytest.mark.asyncio
async def test_workspace_publication_marks_unverified_result_non_belief(monkeypatch):
    from core.container import ServiceContainer

    class Workspace:
        def __init__(self):
            self.calls = []

        async def publish(self, **kwargs):
            self.calls.append(kwargs)
            return True

    workspace = Workspace()
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: workspace if name == "global_workspace" else default
        ),
    )
    loop_receipt = {"schema": "aura.cognitive_loop.v1", "verified": False}

    receipt = await clp._publish_result_to_workspace(
        answer="A tentative internal conclusion",
        pathway="cognitive_loop",
        verified=False,
        priority=0.3,
        loop_receipt=loop_receipt,
    )

    assert receipt == {
        "admitted": True,
        "reason": "accepted_for_competition",
        "trust": "unverified_hypothesis",
    }
    assert len(workspace.calls) == 1
    call = workspace.calls[0]
    assert call["source"] == "cognitive_loop_pathway"
    assert call["content_type"].name == "META"
    assert call["reason"].startswith("[unverified_hypothesis]")
    assert call["payload"]["answer"] == "A tentative internal conclusion"
    assert call["payload"]["loop_receipt"] == loop_receipt
    assert call["payload"]["retained_as_belief"] is False


@pytest.mark.asyncio
async def test_task_tracker_rejection_closes_cycle_and_rolls_back_cooldown(monkeypatch):
    class RejectingTracker:
        def create_task(self, _cycle, *, name):
            assert name == "agency.cognitive_loop.cycle"
            return None

    degradations = []
    loop = clp.build_live_loop(_Container({"llm_router": _Router("42")}))
    monkeypatch.setattr(clp, "build_live_loop", lambda *args, **kwargs: loop)
    monkeypatch.setattr(clp, "_degrade", lambda error, action, **kwargs: degradations.append((error, action)))
    monkeypatch.setattr(
        "core.utils.task_tracker.get_task_tracker",
        lambda: RejectingTracker(),
    )
    agency = _Agency(monologue="A substantive question for the loop")

    result = await clp.cognitive_loop_provider(
        pathway="cognitive_loop",
        now=1000.0,
        idle_seconds=300.0,
        agency=agency,
    )

    assert result is None
    assert agency._cognitive_loop_last_run is None
    assert not hasattr(agency, "_cognitive_loop_task")
    assert degradations[-1][1] == "cognitive-loop supervised task creation rejected"
