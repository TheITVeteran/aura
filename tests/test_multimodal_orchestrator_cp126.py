"""Multimodal orchestrator: 'Online' with nothing wired, and success before
anything rendered."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import core.brain.multimodal_orchestrator as mm
from core.brain.multimodal_orchestrator import MultimodalOrchestrator

pytestmark = pytest.mark.unit


def _with_services(monkeypatch, **services):
    monkeypatch.setattr(
        mm.ServiceContainer, "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
    )


# ── absence must not be cached as health ───────────────────────────────────


def test_setup_with_nothing_registered_does_not_report_online(monkeypatch):
    """ServiceContainer returns None rather than raising, so a boot where
    NOTHING was registered set _is_setup=True and logged 'Online'. Every later
    call then returned from that cache, so services registered afterwards were
    never picked up — permanently blind while reporting healthy."""
    _with_services(monkeypatch)
    orchestrator = MultimodalOrchestrator()

    assert orchestrator._setup() is False
    assert orchestrator._is_setup is False, "absence must not be cached"


def test_setup_is_retried_after_a_later_registration(monkeypatch):
    """The consequence that made the caching bug permanent."""
    orchestrator = MultimodalOrchestrator()
    _with_services(monkeypatch)
    assert orchestrator._setup() is False

    _with_services(monkeypatch, voice_engine=SimpleNamespace(name="voice"))
    assert orchestrator._setup() is True
    assert orchestrator._is_setup is True


def test_render_declines_without_raising_when_nothing_is_wired(monkeypatch):
    """A boot path must get a structured refusal, not an exception."""
    _with_services(monkeypatch)

    result = asyncio.run(MultimodalOrchestrator().render("hello"))

    assert result["ok"] is False
    assert result["reason"] == "setup_failed"


# ── success must mean something happened ───────────────────────────────────


def test_render_does_not_claim_completion_when_it_only_scheduled(monkeypatch):
    """ok=True was returned the instant tasks were SCHEDULED, so voice,
    expression and asset failures happened later and could not change the
    result — callers were told rendering succeeded when nothing had rendered."""
    _with_services(monkeypatch, input_bus=SimpleNamespace(publish=lambda *a, **k: None))
    orchestrator = MultimodalOrchestrator()
    orchestrator._pulse_expression = lambda text, meta: asyncio.sleep(0)

    result = asyncio.run(orchestrator.render("hello"))

    assert result["accepted"] is True
    assert result["completed"] is False, "scheduling is not completion"
    assert result["completion"] is not None


def test_render_reports_no_modality_rather_than_a_hollow_success(monkeypatch):
    """With setup satisfied by one service but that modality disabled, nothing
    is scheduled — and that is not a success."""
    _with_services(monkeypatch, voice_engine=SimpleNamespace(name="voice"))
    orchestrator = MultimodalOrchestrator()

    result = asyncio.run(orchestrator.render("hello", {"voice": False}))

    assert result["ok"] is False
    assert result["reason"] == "no_modality_available"
    assert result["scheduled"] == []


# ── one response must not be able to book hours of work ────────────────────


def test_manifestation_concepts_are_bounded():
    """Concept count and input size were unbounded and each concept carried its
    own 120s timeout, so one response could occupy the lane for hours."""
    text = " ".join(f"[Manifesting: concept number {i}]" for i in range(200))

    concepts = MultimodalOrchestrator._manifestation_concepts(text)

    assert len(concepts) <= mm._MAX_MANIFEST_CONCEPTS


def test_repeated_tags_are_deduplicated():
    text = "[Manifesting: a red door] " * 50

    concepts = MultimodalOrchestrator._manifestation_concepts(text)

    assert concepts == ["a red door"]


def test_tags_beyond_the_scan_window_are_ignored():
    """An enormous input cannot dominate the scan itself."""
    filler = "x" * (mm._MAX_MANIFEST_SCAN_CHARS + 5_000)
    text = filler + "[Manifesting: hidden concept]"

    assert MultimodalOrchestrator._manifestation_concepts(text) == []


def test_ordinary_tags_still_work():
    concepts = MultimodalOrchestrator._manifestation_concepts(
        "Here it is [Drawing: a quiet harbour] and done."
    )

    assert concepts == ["a quiet harbour"]
