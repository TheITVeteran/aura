"""CP126 hardening contracts for core/brain/metacognitive_monitor.py.

The dominant defect class here is 'absence of a check certified as a passed
check'. Tests verify the monitor returns an UNEVALUATED report (never a perfect
score) when it cannot actually audit, validates the judgment's fields and
ranges, triggers revision on declared incoherence / low critical metrics, and
guards the revised output.
"""
from __future__ import annotations

import math

import pytest

from core.brain.metacognitive_monitor import (
    MetacognitiveMonitor,
    _clamp01,
    _extract_json_object,
)
from core.state.aura_state import AuraState


class _FakeRouter:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    async def think(self, prompt, **kw):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(r, Exception):
            raise r
        return r


def _state():
    st = AuraState.default()
    st.identity.current_narrative = "Aura, a reflective mind."
    return st


def _monitor(router):
    m = MetacognitiveMonitor()
    m.router = router
    return m


# ── babdcc4e: missing router does not certify perfect coherence ────────────


@pytest.mark.asyncio
async def test_missing_router_is_unevaluated():
    m = MetacognitiveMonitor()
    m.router = None
    # Force _get_router to find nothing.
    m._get_router = lambda: None  # type: ignore[assignment]
    rep = await m.evaluate("hi", _state())
    assert rep.evaluated is False
    assert rep.coherence_score == 0.0  # NOT a fake 1.0


# ── de1decbc: router error is unevaluated, not perfect ─────────────────────


@pytest.mark.asyncio
async def test_router_error_is_unevaluated():
    rep = await _monitor(_FakeRouter(RuntimeError("network down"))).evaluate("hi", _state())
    assert rep.evaluated is False
    assert rep.metrics == {}


# ── 75501e58: unparseable judgment is unevaluated, not approved ────────────


@pytest.mark.asyncio
async def test_non_json_judgment_is_unevaluated():
    rep = await _monitor(_FakeRouter("I think it's fine, no JSON here")).evaluate("hi", _state())
    assert rep.evaluated is False


# ── bda83c10: missing verdict fields are not approval ──────────────────────


@pytest.mark.asyncio
async def test_missing_coherent_field_is_unevaluated():
    rep = await _monitor(_FakeRouter('{"score": 0.9, "violations": []}')).evaluate("hi", _state())
    assert rep.evaluated is False


@pytest.mark.asyncio
async def test_missing_score_field_is_unevaluated():
    rep = await _monitor(_FakeRouter('{"coherent": true, "violations": []}')).evaluate("hi", _state())
    assert rep.evaluated is False


# ── 4a93dddd: score/metric range validation ────────────────────────────────


@pytest.mark.asyncio
async def test_out_of_range_score_is_clamped():
    rep = await _monitor(_FakeRouter('{"coherent": true, "score": 1.5, "violations": [], "metrics": {}}')).evaluate("hi", _state())
    assert rep.evaluated is True and rep.coherence_score == 1.0


@pytest.mark.asyncio
async def test_non_numeric_score_is_unevaluated():
    rep = await _monitor(_FakeRouter('{"coherent": true, "score": "high", "violations": []}')).evaluate("hi", _state())
    assert rep.evaluated is False


def test_clamp01():
    assert _clamp01(True) is None
    assert _clamp01(math.nan) is None
    assert _clamp01(1.5) == 1.0
    assert _clamp01(-0.2) == 0.0
    assert _clamp01(0.4) == 0.4


# ── 5dfaeb2a: revision fires on declared incoherence / low critical metric ─


@pytest.mark.asyncio
async def test_declared_incoherence_triggers_revision():
    audit = '{"coherent": false, "score": 0.9, "violations": ["tone"], "metrics": {"logic": 0.9}}'
    rep = await _monitor(_FakeRouter(audit, "a revised response")).evaluate("hi", _state())
    assert rep.is_coherent is False
    assert rep.revision_needed is True
    assert rep.revised_response == "a revised response"


@pytest.mark.asyncio
async def test_low_logic_metric_triggers_revision():
    audit = '{"coherent": true, "score": 0.9, "violations": [], "metrics": {"logic": 0.2}}'
    rep = await _monitor(_FakeRouter(audit, "fixed")).evaluate("hi", _state())
    assert rep.revision_needed is True


# ── a8965003: empty revision is rejected (content preservation) ────────────


@pytest.mark.asyncio
async def test_empty_revision_keeps_original():
    audit = '{"coherent": false, "score": 0.9, "violations": ["x"], "metrics": {}}'
    rep = await _monitor(_FakeRouter(audit, "   ")).evaluate("original text", _state())
    assert rep.revised_response == "original text"


# ── 75501e58: robust JSON extraction ───────────────────────────────────────


def test_extract_json_handles_prose_and_multiple_objects():
    assert _extract_json_object('prose {"a": 1} more {"b": 2}') == {"a": 1}
    assert _extract_json_object('```json\n{"ok": true}\n```') == {"ok": True}
    assert _extract_json_object("no object here") is None
    assert _extract_json_object('{"s": "brace } inside string"}') == {"s": "brace } inside string"}
