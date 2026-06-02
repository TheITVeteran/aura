from __future__ import annotations

import time

from core.memory.long_term_memory_engine import LongTermMemoryEngine, TaggedMemory
from core.memory.scar_formation import _MAX_SCARS
from core.memory.sovereign_pruner import SovereignPruner
from core.memory.retention_policy import (
    MemoryRetentionPolicy,
    episodic_retention_policy,
    hybrid_memory_retention_policy,
    long_term_retention_policy,
    sovereign_pruner_target_retention,
    working_history_retention_policy,
)


def test_long_term_retention_policy_scales_above_legacy_cap(monkeypatch) -> None:
    monkeypatch.delenv("AURA_LTM_MAX_MEMORIES", raising=False)

    assert long_term_retention_policy(ram_gb=64).max_items == 75_000
    assert long_term_retention_policy(ram_gb=16).max_items == 30_000
    assert long_term_retention_policy(ram_gb=4).max_items == 10_000


def test_episodic_retention_policy_scales_above_legacy_cap(monkeypatch) -> None:
    monkeypatch.delenv("AURA_EPISODIC_MAX_EPISODES", raising=False)

    assert episodic_retention_policy(ram_gb=64).max_items == 100_000
    assert episodic_retention_policy(ram_gb=16).max_items == 50_000
    assert episodic_retention_policy(ram_gb=4).max_items == 10_000


def test_hybrid_memory_retention_policy_scales_above_legacy_cap(monkeypatch) -> None:
    monkeypatch.delenv("AURA_HYBRID_MEMORY_MAX_ENTRIES", raising=False)

    assert hybrid_memory_retention_policy(ram_gb=64).max_items == 100_000
    assert hybrid_memory_retention_policy(ram_gb=16).max_items == 50_000
    assert hybrid_memory_retention_policy(ram_gb=4).max_items == 10_000


def test_sovereign_pruner_default_retention_is_not_legacy_aggressive(monkeypatch) -> None:
    monkeypatch.delenv("AURA_SOVEREIGN_PRUNER_TARGET_RETENTION", raising=False)

    assert sovereign_pruner_target_retention() == 0.75
    assert SovereignPruner().target_retention == 0.75
    assert SovereignPruner(target_retention=0.0).target_retention == 0.0


def test_working_history_retention_policy_exceeds_hidden_legacy_cap(monkeypatch) -> None:
    monkeypatch.delenv("AURA_EXECUTIVE_DECISION_HISTORY_MAX", raising=False)

    assert working_history_retention_policy("AURA_EXECUTIVE_DECISION_HISTORY_MAX", ram_gb=64).max_items == 10_000
    assert working_history_retention_policy("AURA_EXECUTIVE_DECISION_HISTORY_MAX", ram_gb=16).max_items == 5_000
    assert working_history_retention_policy("AURA_EXECUTIVE_DECISION_HISTORY_MAX", ram_gb=4).max_items == 1_000


def test_working_history_retention_policy_honors_env_override(monkeypatch) -> None:
    monkeypatch.setenv("AURA_COGNITIVE_THOUGHT_HISTORY_MAX", "12000")

    policy = working_history_retention_policy("AURA_COGNITIVE_THOUGHT_HISTORY_MAX", ram_gb=4)

    assert policy.max_items == 12_000
    assert policy.basis == "env:AURA_COGNITIVE_THOUGHT_HISTORY_MAX"


def test_behavioral_scar_cap_exceeds_legacy_floor() -> None:
    assert _MAX_SCARS >= 2_000


def test_long_term_retention_cap_preserves_important_emotional_memory() -> None:
    engine = LongTermMemoryEngine.__new__(LongTermMemoryEngine)
    engine._retention_policy = MemoryRetentionPolicy(
        max_items=4,
        prune_keep_fraction=0.90,
        basis="test",
    )
    now = time.time()
    important = TaggedMemory(
        id="important",
        content="important identity memory",
        timestamp=now - 10_000,
        emotional_valence=0.9,
        importance=1.0,
        decay_rate=0.001,
        last_rehearsed=now,
    )
    engine.memories = [
        TaggedMemory(
            id=f"noise-{idx}",
            content=f"noise {idx}",
            timestamp=now + idx,
            emotional_valence=0.0,
            importance=0.0,
            decay_rate=0.02,
            last_rehearsed=now,
        )
        for idx in range(8)
    ] + [important]

    engine._enforce_retention_cap()

    assert len(engine.memories) == 4
    assert important in engine.memories
