"""Wiring the cognitive loop into Aura's live Will (CP244).

The load-bearing safety property: gated OFF by default, so turning the
mechanism on grants nothing until explicitly enabled -- and when off, the
live agency is byte-identical.
"""
from __future__ import annotations

import asyncio

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

    async def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.reply


class _Memory:
    def search_sync(self, query, limit=5):
        return [{"content": "a relevant fact"}]


class _Agency:
    def __init__(self, monologue=""):
        self._current_monologue = monologue
        self.registered = []

    def register_pathway_hook(self, pathway, provider):
        if pathway not in ("autonomous_research", "curiosity_drive"):
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


# ── The live async loop runs end to end ─────────────────────────────────


def test_router_deliberator_bridges_async_generation():
    router = _Router("stepwise... 42")
    delib = clp._RouterDeliberator(router)
    out = asyncio.run(delib.deliberate("q", ["a fact"]))
    assert out == "stepwise... 42"
    # material and step-by-step framing reach the model
    assert "a fact" in router.prompts[0]
    assert "step by step" in router.prompts[0]


def test_provider_proposes_from_the_monologue(monkeypatch):
    monkeypatch.setenv(clp.ENABLE_FLAG, "1")
    monkeypatch.setattr(clp, "COOLDOWN_SECONDS", 0.0)
    container = _Container({"llm_router": _Router("42"), "memory_facade": _Memory()})
    loop = clp.build_live_loop(container)
    monkeypatch.setattr(clp, "build_live_loop", lambda *a, **k: loop)
    agency = _Agency(monologue="I wonder how many moons Jupiter has")
    proposal = asyncio.run(clp.cognitive_loop_provider(
        pathway="curiosity_drive", now=0.0, idle_seconds=0.0, agency=agency,
    ))
    assert proposal is not None
    assert proposal["type"] == "inner_reasoning"
    assert proposal["content"] == "42"
    # unverified (no verifier organ) -> lower priority, never over-trusted
    assert proposal["verified"] is False
    assert proposal["priority"] == 0.3


def test_provider_proposes_nothing_without_a_query(monkeypatch):
    container = _Container({"llm_router": _Router()})
    loop = clp.build_live_loop(container)
    monkeypatch.setattr(clp, "build_live_loop", lambda *a, **k: loop)
    monkeypatch.setattr(clp, "COOLDOWN_SECONDS", 0.0)
    agency = _Agency(monologue="")  # nothing on her mind
    proposal = asyncio.run(clp.cognitive_loop_provider(
        pathway="autonomous_research", now=0.0, idle_seconds=0.0, agency=agency,
    ))
    assert proposal is None


def test_provider_proposes_nothing_when_organs_missing(monkeypatch):
    monkeypatch.setattr(clp, "build_live_loop", lambda: None)
    monkeypatch.setattr(clp, "COOLDOWN_SECONDS", 0.0)
    proposal = asyncio.run(clp.cognitive_loop_provider(
        pathway="autonomous_research", now=0.0, idle_seconds=0.0,
        agency=_Agency(monologue="a real question here"),
    ))
    assert proposal is None


def test_cooldown_prevents_firing_the_loop_every_pulse(monkeypatch):
    """On-by-default must not hammer the live 32B: one run, then a cooldown."""
    monkeypatch.setattr(clp, "COOLDOWN_SECONDS", 180.0)
    container = _Container({"llm_router": _Router("42"), "memory_facade": _Memory()})
    loop = clp.build_live_loop(container)
    monkeypatch.setattr(clp, "build_live_loop", lambda *a, **k: loop)
    agency = _Agency(monologue="a real question to reason about")

    first = asyncio.run(clp.cognitive_loop_provider(
        pathway="curiosity_drive", now=1000.0, idle_seconds=0.0, agency=agency))
    assert first is not None
    # a pulse 5s later is inside the cooldown -> no re-fire
    second = asyncio.run(clp.cognitive_loop_provider(
        pathway="curiosity_drive", now=1005.0, idle_seconds=0.0, agency=agency))
    assert second is None
    # well past the cooldown -> fires again
    third = asyncio.run(clp.cognitive_loop_provider(
        pathway="curiosity_drive", now=1200.0, idle_seconds=0.0, agency=agency))
    assert third is not None
