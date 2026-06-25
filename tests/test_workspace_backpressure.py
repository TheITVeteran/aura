"""Salience-ranked backpressure in the global workspace.

The seizure guard used to blanket-drop *every* new bid once the candidate pool hit
``_MAX_CANDIDATES``. A flood of low-salience submissions could therefore lock out a
genuinely urgent bid that arrived a moment later. Backpressure now keeps the N most
salient bids: a late-but-important candidate evicts the weakest queued one.
"""
from __future__ import annotations

import asyncio

import pytest

from core.consciousness.global_workspace import GlobalWorkspace, CognitiveCandidate


def _fill(ws: GlobalWorkspace, n: int, priority: float) -> None:
    for i in range(n):
        ok = asyncio.run(ws.submit(CognitiveCandidate(
            content=f"noise-{i}", source=f"noise-{i}", priority=priority,
        )))
        assert ok


def test_high_salience_bid_evicts_weakest_when_flooded():
    ws = GlobalWorkspace()
    cap = ws._MAX_CANDIDATES
    _fill(ws, cap, priority=0.10)            # flood with low-salience bids
    assert len(ws._candidates) == cap

    # An urgent bid arrives late — it must NOT be dropped; it should evict a weak one.
    admitted = asyncio.run(ws.submit(CognitiveCandidate(
        content="URGENT", source="affect_distress", priority=0.95,
    )))
    assert admitted is True
    assert len(ws._candidates) == cap                 # cap respected
    sources = {c.source for c in ws._candidates}
    assert "affect_distress" in sources               # urgent bid is in
    assert sum(1 for s in sources if s.startswith("noise")) == cap - 1  # one noise evicted


def test_low_salience_bid_is_dropped_when_flooded():
    ws = GlobalWorkspace()
    cap = ws._MAX_CANDIDATES
    _fill(ws, cap, priority=0.80)            # flood with strong bids

    # A weak bid arriving into a strong field is genuinely least important → dropped.
    admitted = asyncio.run(ws.submit(CognitiveCandidate(
        content="meh", source="weakling", priority=0.05,
    )))
    assert admitted is False
    assert len(ws._candidates) == cap
    assert "weakling" not in {c.source for c in ws._candidates}


def test_same_source_rebids_without_counting_as_flood_pressure():
    ws = GlobalWorkspace()
    cap = ws._MAX_CANDIDATES
    _fill(ws, cap, priority=0.50)

    # An existing source updating its own bid replaces in place — never dropped,
    # never grows the pool past the cap.
    existing = next(iter(ws._candidates)).source
    admitted = asyncio.run(ws.submit(CognitiveCandidate(
        content="updated", source=existing, priority=0.51,
    )))
    assert admitted is True
    assert len(ws._candidates) == cap
    updated = [c for c in ws._candidates if c.source == existing]
    assert len(updated) == 1 and updated[0].content == "updated"
