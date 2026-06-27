"""Tests for Φ-gated test-time compute allocation in the reasoning amplifier.

The contract under test: cognitive *demand* (free energy, uncertainty, stuck
valence, frustration) buys reasoning depth; Φ-*capacity* (how integrated the mind
is right now) gates how much of that depth it is allowed to spend.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain.reasoning_amplifier_v2 import (
    PHI_DELIBERATE,
    PHI_DORMANT,
    ReasoningAmplifierV2 as R,
    ReasoningMode as M,
    _MAX_SAMPLES,
    _MODE_BUDGET,
)


# ── pure budget resolver ────────────────────────────────────────────────────
def test_neutral_signals_keep_base_mode():
    mode, samples, tmult = R._resolve_compute_budget(M.NORMAL, {})
    assert mode is M.NORMAL
    assert samples == _MODE_BUDGET[M.NORMAL]
    assert tmult == 1.0


def test_high_demand_high_phi_deepens_and_spends_more():
    mode, samples, tmult = R._resolve_compute_budget(
        M.NORMAL, {"phi": 0.8, "free_energy": 0.9, "uncertainty": 0.8}
    )
    assert mode in (M.DEEP, M.EXTREME)          # demand bought depth
    assert samples > _MODE_BUDGET[M.NORMAL]       # more candidates
    assert tmult > 1.0                            # more wall-clock


def test_high_demand_low_phi_is_capped():
    # A fragmented mind (Φ below DORMANT) cannot sustain deep deliberation even
    # under maximal demand — the Φ-gate caps it at FAST.
    mode, samples, _ = R._resolve_compute_budget(
        M.NORMAL, {"phi": 0.02, "free_energy": 1.0, "uncertainty": 1.0}
    )
    assert mode is M.FAST
    assert samples == 1


def test_mid_low_phi_caps_at_normal():
    mode, _, _ = R._resolve_compute_budget(
        M.DEEP, {"phi": 0.10, "free_energy": 1.0, "uncertainty": 1.0}
    )
    assert mode is M.NORMAL  # below REACTIVE -> cap at NORMAL


def test_proof_mode_never_downgraded_by_low_phi():
    mode, _, _ = R._resolve_compute_budget(M.PROOF, {"phi": 0.0})
    assert mode is M.PROOF


def test_stuck_mind_deepens_even_without_free_energy():
    # Legacy hard trigger preserved: negative valence / high frustration deepens once.
    mode, _, _ = R._resolve_compute_budget(M.NORMAL, {"phi": 0.4, "valence": -0.6, "frustration": 0.7})
    assert mode is M.DEEP


def test_sample_budget_never_exceeds_max():
    mode, samples, _ = R._resolve_compute_budget(
        M.EXTREME, {"phi": 1.0, "free_energy": 1.0, "uncertainty": 1.0, "frustration": 1.0}
    )
    assert samples <= _MAX_SAMPLES


# ── component functions ─────────────────────────────────────────────────────
def test_phi_capacity_is_monotonic():
    lo = R._phi_capacity(PHI_DORMANT)
    mid = R._phi_capacity((PHI_DORMANT + PHI_DELIBERATE) / 2)
    hi = R._phi_capacity(PHI_DELIBERATE)
    assert lo < mid < hi
    assert hi == 1.0


def test_cognitive_demand_responds_to_each_signal():
    assert R._cognitive_demand({}) == 0.0
    assert R._cognitive_demand({"free_energy": 1.0}) > 0.0
    assert R._cognitive_demand({"uncertainty": 1.0}) > 0.0
    assert R._cognitive_demand({"valence": -1.0}) > 0.0
    assert R._cognitive_demand({"frustration": 1.0}) > 0.0
    # clamped to [0,1] even when every signal is maxed
    assert R._cognitive_demand(
        {"free_energy": 1.0, "uncertainty": 1.0, "valence": -1.0, "frustration": 1.0}
    ) == 1.0


def test_phi_mode_cap_passes_through_when_integrated():
    assert R._phi_mode_cap(M.DEEP, 0.9) is M.DEEP


# ── amplify() cache-hit short circuit ───────────────────────────────────────
def test_amplify_cache_hit_bypasses_generation(tmp_path, monkeypatch):
    from core.brain import reasoning_solved_cache as rsc
    from core.brain.reasoning_amplifier_v2 import AmplificationRequest

    # Temp-backed cache, pre-populated with a verifier-clean math answer.
    cache = rsc.ReasoningSolvedCache(tmp_path / "c.json")
    cache.put("what is 6 times 7", "math", answer="42", confidence=0.97, mode="deep", verified=True)
    monkeypatch.setattr(rsc, "get_reasoning_solved_cache", lambda: cache)

    calls = {"n": 0}

    async def _generate(prompt, temperature):
        calls["n"] += 1
        return "should never be called"

    amp = R(_generate)
    req = AmplificationRequest(
        objective="what is 6 times 7",
        task_type="math",
        context={"skip_evidence": True},
    )
    result = asyncio.run(amp.amplify(req))
    assert result.answer == "42"
    assert result.verified is True
    assert result.receipt.strategy_used == "solved_cache"
    assert "solved_cache_hit" in result.receipt.fallbacks_used
    assert calls["n"] == 0  # generation was fully bypassed
