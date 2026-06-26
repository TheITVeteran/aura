"""The dedup gate must stop identical-fact pollution even for high-importance writes."""
from __future__ import annotations

from core.memory.semantic_dedup import SemanticDedupGate


def test_exact_duplicate_rejected_even_high_importance():
    gate = SemanticDedupGate()
    fact = "Bryan prefers Python for tooling and scripting work."
    assert gate.should_store(fact, importance=0.95) is True
    # The same canonical fact again must NOT be stored, despite high importance.
    assert gate.should_store(fact, importance=0.95) is False
    assert gate.should_store(fact, importance=0.99) is False


def test_high_importance_still_allows_distinct_facts():
    gate = SemanticDedupGate()
    assert gate.should_store("Bryan prefers Python for tooling.", importance=0.9) is True
    assert gate.should_store("Bryan dislikes heavyweight Java frameworks.", importance=0.9) is True


def test_durable_exact_survives_window_prune():
    gate = SemanticDedupGate()
    fact = "Aura runs on Apple Silicon with a 64GB unified memory budget."
    assert gate.should_store(fact, importance=0.5) is True
    # Simulate the 1-hour recent window expiring (clears _recent + _exact_hashes)…
    gate._recent.clear()
    gate._exact_hashes.clear()
    # …the durable LRU still catches the exact duplicate.
    assert gate.should_store(fact, importance=0.5) is False
    assert gate.should_store(fact, importance=0.95) is False


def test_normalization_catches_punctuation_variants():
    gate = SemanticDedupGate()
    assert gate.should_store("Bryan prefers Python!!!", importance=0.9) is True
    # Same content modulo punctuation/case → exact-normalized duplicate.
    assert gate.should_store("bryan prefers python", importance=0.9) is False


def test_trivial_text_rejected():
    gate = SemanticDedupGate()
    assert gate.should_store("ok", importance=0.99) is False
