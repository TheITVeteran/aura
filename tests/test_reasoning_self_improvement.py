"""Tests for the internal STaR/RLVR reasoning self-improvement flywheel."""
from __future__ import annotations

import asyncio

import pytest

from core.brain.reasoning_self_improvement import ReasoningSelfImprovement


@pytest.fixture
def si(tmp_path):
    return ReasoningSelfImprovement(tmp_path / "traces.json", min_confidence=0.7)


def test_records_verified_cacheable_win(si):
    assert si.record_win("compute 12 * 12", "math", answer="144", confidence=0.95, mode="deep", verified=True)
    assert si.pending_count() == 1
    ex = si.export_training_examples()
    assert ex[0]["prompt"] == "compute 12 * 12" and ex[0]["completion"] == "144"


def test_rejects_unverified_low_conf_and_source_dependent(si):
    assert si.record_win("q", "math", answer="x", confidence=0.95, mode="deep", verified=False) is False
    assert si.record_win("q", "math", answer="x", confidence=0.3, mode="deep", verified=True) is False
    assert si.record_win("q", "repo_audit", answer="core/x.py", confidence=0.95, mode="deep", verified=True) is False
    assert si.pending_count() == 0


def test_dedup_by_problem_key(si):
    si.record_win("solve x", "math", answer="1", confidence=0.9, mode="deep", verified=True)
    si.record_win("Solve  x", "math", answer="1", confidence=0.9, mode="deep", verified=True)
    assert si.pending_count() == 1  # normalized key collides


def test_disabled_by_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_REASONING_SELF_IMPROVEMENT", "0")
    s = ReasoningSelfImprovement(tmp_path / "t.json")
    assert s.record_win("p", "math", answer="ans", confidence=0.9, mode="deep", verified=True) is False


def test_maybe_improve_insufficient(si):
    si.record_win("p", "math", answer="ans", confidence=0.9, mode="deep", verified=True)
    result = asyncio.run(si.maybe_improve(min_traces=64))
    assert result["status"] == "insufficient_traces"


def test_maybe_improve_feeds_and_marks(si, monkeypatch):
    # Avoid touching the real LoRA governor / psutil.
    import core.adaptation.online_lora_governor as gov

    class _FakeGov:
        def active_lora_processes(self):
            return []

    monkeypatch.setattr(gov, "get_online_lora_governor", lambda: _FakeGov())

    for i in range(3):
        si.record_win(f"problem {i}", "math", answer=str(i), confidence=0.9, mode="deep", verified=True)

    fed = {}

    async def _feed(examples):
        fed["n"] = len(examples)
        return {"ok": True, "fed": len(examples)}

    result = asyncio.run(si.maybe_improve(min_traces=2, feed_fn=_feed))
    assert result["status"] == "fed"
    assert result["count"] == 3
    assert fed["n"] == 3
    assert si.pending_count() == 0  # all marked fed


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "persist.json"
    s1 = ReasoningSelfImprovement(path)
    s1.record_win("durable", "code", answer="def f(): return 1", confidence=0.9, mode="deep", verified=True)
    s2 = ReasoningSelfImprovement(path)
    assert s2.pending_count() == 1


def test_max_traces_bounded(tmp_path):
    s = ReasoningSelfImprovement(tmp_path / "t.json", max_traces=64)
    for i in range(200):
        s.record_win(f"p{i}", "math", answer=str(i), confidence=0.9, mode="fast", verified=True)
    assert len(s._traces) <= 64
    assert s.stats()["evicted"] > 0
