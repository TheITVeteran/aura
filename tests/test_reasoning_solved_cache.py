"""Tests for the reasoning solved-cache (memoized verifier-clean derivations)."""
from __future__ import annotations

import time

import pytest

from core.brain.reasoning_solved_cache import (
    DEFAULT_CACHEABLE_TASK_TYPES,
    ReasoningSolvedCache,
    SolvedEntry,
)


@pytest.fixture
def cache(tmp_path):
    return ReasoningSolvedCache(tmp_path / "solved.json")


def test_miss_then_store_then_hit(cache):
    assert cache.get("what is 17 * 23", "math") is None
    stored = cache.put(
        "what is 17 * 23", "math", answer="391", confidence=0.95, mode="deep", verified=True
    )
    assert stored is True
    hit = cache.get("what is 17 * 23", "math")
    assert hit is not None
    assert hit.answer == "391"
    assert hit.hits == 1


def test_normalization_collides_phrasings(cache):
    cache.put("Compute  2 + 2", "math", answer="4", confidence=0.9, mode="fast", verified=True)
    # different whitespace + case must hit the same entry
    hit = cache.get("compute 2 + 2", "math")
    assert hit is not None and hit.answer == "4"


def test_refuses_unverified(cache):
    assert cache.put("q", "math", answer="42", confidence=0.99, mode="deep", verified=False) is False
    assert cache.get("q", "math") is None


def test_refuses_source_dependent_task_types(cache):
    # repo_audit/architecture/factual depend on mutable state — never cached.
    for tt in ("repo_audit", "architecture", "factual", "planning", "generic"):
        assert cache.is_cacheable(tt) is False
        assert cache.put("x", tt, answer="some answer", confidence=0.9, mode="deep", verified=True) is False
        assert cache.get("x", tt) is None


def test_cacheable_set_is_source_independent():
    assert DEFAULT_CACHEABLE_TASK_TYPES == frozenset({"math", "code", "logic"})


def test_refuses_empty_answer(cache):
    assert cache.put("q", "code", answer="   ", confidence=0.9, mode="deep", verified=True) is False


def test_low_confidence_rejected(cache):
    assert cache.put("q", "math", answer="answer", confidence=0.1, mode="deep", verified=True) is False


def test_ttl_expiry(tmp_path):
    c = ReasoningSolvedCache(tmp_path / "c.json", ttl_s=60.0)
    c.put("q", "math", answer="ans", confidence=0.9, mode="deep", verified=True)
    # Force the stored timestamp into the distant past.
    key = next(iter(c._entries))
    c._entries[key].stored_at = time.time() - 120.0
    assert c.get("q", "math") is None  # expired -> miss + dropped


def test_max_entries_eviction(tmp_path):
    c = ReasoningSolvedCache(tmp_path / "c.json", max_entries=16)
    for i in range(40):
        c.put(f"problem number {i}", "math", answer=f"ans{i}", confidence=0.9, mode="fast", verified=True)
    assert len(c._entries) <= 16
    assert c.stats()["evictions"] > 0


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "persist.json"
    c1 = ReasoningSolvedCache(path)
    c1.put("durable problem", "code", answer="def f(): return 1", confidence=0.9, mode="deep", verified=True)
    # New instance loads from disk.
    c2 = ReasoningSolvedCache(path)
    hit = c2.get("durable problem", "code")
    assert hit is not None and hit.answer == "def f(): return 1"


def test_entry_serialization_round_trip():
    e = SolvedEntry(answer="a", confidence=0.8, mode="deep", task_type="math", verifiers_run=["math_engine"])
    e2 = SolvedEntry.from_dict(e.to_dict())
    assert e2.answer == "a" and e2.task_type == "math" and e2.verifiers_run == ["math_engine"]


def test_stats_hit_rate(cache):
    cache.put("p", "math", answer="x", confidence=0.9, mode="fast", verified=True)
    cache.get("p", "math")  # hit
    cache.get("nope", "math")  # miss
    s = cache.stats()
    assert s["hits"] == 1 and s["misses"] == 1
    assert s["hit_rate"] == pytest.approx(0.5)
