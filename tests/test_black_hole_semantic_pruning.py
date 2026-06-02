from __future__ import annotations

import time

from core.memory.black_hole_vault import BlackHoleVault
from core.memory.retention_policy import MemoryRetentionPolicy, black_hole_retention_policy


def test_black_hole_vault_keeps_semantically_central_memory_over_recent_noise() -> None:
    vault = BlackHoleVault.__new__(BlackHoleVault)
    now_ms = int(time.time() * 1000)
    critical = {
        "text": "core identity commitment",
        "created": now_ms - 90 * 86_400_000,
        "access_count": 0,
        "metadata": {"conceptual_centrality": 0.98, "core_identity": True},
    }
    noise = [
        {
            "text": f"recent noise {idx}",
            "created": now_ms - idx,
            "access_count": 0,
            "metadata": {"conceptual_centrality": 0.0, "affect_intensity": 0.0},
        }
        for idx in range(10)
    ]

    kept = vault._select_semantically_important([*noise, critical], keep_count=3)

    assert critical in kept
    assert vault._memory_importance(critical, now_ms=now_ms) == float("inf")


def test_black_hole_retention_policy_scales_above_legacy_cap(monkeypatch) -> None:
    monkeypatch.delenv("AURA_BLACK_HOLE_MAX_MEMORIES", raising=False)

    assert black_hole_retention_policy(ram_gb=64).max_items == 100_000
    assert black_hole_retention_policy(ram_gb=16).max_items == 50_000
    assert black_hole_retention_policy(ram_gb=4).max_items == 10_000


def test_black_hole_retention_policy_allows_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("AURA_BLACK_HOLE_MAX_MEMORIES", "250000")
    monkeypatch.setenv("AURA_BLACK_HOLE_PRUNE_KEEP_FRACTION", "0.95")

    policy = black_hole_retention_policy(ram_gb=4)

    assert policy.max_items == 250_000
    assert policy.prune_keep_fraction == 0.95
    assert policy.basis == "env:AURA_BLACK_HOLE_MAX_MEMORIES"


def test_black_hole_retention_cap_uses_semantic_importance_not_recency() -> None:
    vault = BlackHoleVault.__new__(BlackHoleVault)
    vault._retention_policy = MemoryRetentionPolicy(
        max_items=5,
        prune_keep_fraction=0.90,
        basis="test",
    )
    now_ms = int(time.time() * 1000)
    core_memory = {
        "text": "old important",
        "created": now_ms - 100 * 86_400_000,
        "access_count": 0,
        "metadata": {"conceptual_centrality": 1.0},
    }
    vault.memories = [
        {
            "text": f"recent low-value {idx}",
            "created": now_ms - idx,
            "access_count": 0,
            "metadata": {"conceptual_centrality": 0.0, "affect_intensity": 0.0},
        }
        for idx in range(8)
    ] + [core_memory]

    vault._enforce_retention_cap()

    assert len(vault.memories) == 5
    assert core_memory in vault.memories
    assert BlackHoleVault.get_stats(vault)["max_vectors"] == 5
