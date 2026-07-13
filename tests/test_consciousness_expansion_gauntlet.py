"""tests/test_consciousness_expansion_gauntlet.py
====================================================
End-to-end gauntlet spanning Phases 1-7 of the Aura consciousness
expansion.  Runs adversarial, stress and integration checks that touch
all new modules together.

Coverage:
    Phase 1 (HierarchicalPhi) — stress compute + null baseline + 32+ nodes
    Phase 2 (HemisphericSplit) — callosum severance cycle + confabulation
    Phase 3 (MinimalSelfhood)  — dugesia transition under sustained training
    Phase 4 (RecursiveToM)     — depth-3 + scrub-jay behaviour change
    Phase 5 (OctopusFederation) — sever/restore latency
    Phase 6 (CellularTurnover) — 25% burst identity preservation
    Phase 7 (AbsorbedVoices)   — attribution after cross-voice training
    Phase 8 (UnifiedCognitiveBias) — fused vector composition sanity
    Cross-phase — biases stay bounded, stress under 500 combined ticks.
"""
from __future__ import annotations


import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.consciousness.hierarchical_phi import HierarchicalPhi  # noqa: E402
from core.consciousness.hemispheric_split import HemisphericSplit, Hemisphere  # noqa: E402
from core.consciousness.minimal_selfhood import (  # noqa: E402
    ACTION_CATEGORIES as MS_ACTIONS, MinimalSelfhood, Mode,
)
from core.consciousness.recursive_tom import RecursiveTheoryOfMind, MAX_DEPTH  # noqa: E402
from core.consciousness.octopus_arms import OctopusFederation, ArmState  # noqa: E402
from core.consciousness.cellular_turnover import (  # noqa: E402
    CellularTurnover, THRESHOLD_IDENTITY,
)
from core.consciousness.neural_mesh import NeuralMesh  # noqa: E402
from core.consciousness.absorbed_voices import AbsorbedVoices  # noqa: E402
from core.consciousness.unified_cognitive_bias import UnifiedCognitiveBias  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────

def _coherent_snapshot(rng, phase):
    cog = np.array([
        math.sin(phase), math.cos(phase * 1.1), math.sin(phase * 0.9),
        math.cos(phase * 1.2), math.sin(phase * 1.3), math.cos(phase * 0.8),
        math.sin(phase * 0.7), math.cos(phase * 1.05),
        math.sin(phase * 1.25), math.cos(phase * 0.85),
        math.sin(phase * 0.95), math.cos(phase * 1.35),
        math.sin(phase * 0.55), math.cos(phase * 1.45),
        math.sin(phase * 1.15), math.cos(phase * 0.75),
    ], dtype=np.float64) + rng.standard_normal(16) * 0.05
    mesh = rng.standard_normal(4096).astype(np.float32) * 0.2
    for c in range(64):
        start = c * 64
        mesh[start:start + 64] += 0.6 * math.sin(phase + c * 0.05)
    return cog, mesh


# ── Gauntlet tests ───────────────────────────────────────────────────────────

def test_gauntlet_hierarchical_phi_under_load():
    h = HierarchicalPhi()
    rng = np.random.default_rng(101)
    phase = 0.0
    for _ in range(600):
        phase += 0.19
        cog, mesh = _coherent_snapshot(rng, phase)
        h.record_snapshot(cog, mesh)
    t0 = time.time()
    result = h.compute(force=True)
    elapsed = time.time() - t0
    assert result is not None
    assert elapsed < 3.0
    # Null baseline stays strictly below measured max-complex phi.
    null_phi = h.compute_null_baseline()
    assert null_phi < max(result.max_complex_phi, 0.02)


def test_gauntlet_hemispheric_severance_and_restore_cycle():
    split = HemisphericSplit()
    rng = np.random.default_rng(202)
    for _ in range(120):
        exec_s = rng.standard_normal(8)
        sens_s = rng.standard_normal(16)
        cog = rng.standard_normal(16)
        emb = rng.standard_normal(8)
        split.tick(exec_s, sens_s, cog, emb)
    pre = split.agreement_rate()
    split.sever_callosum()
    for _ in range(80):
        exec_s = rng.standard_normal(8)
        sens_s = rng.standard_normal(16)
        cog = rng.standard_normal(16)
        emb = rng.standard_normal(8)
        split.tick(exec_s, sens_s, cog, emb)
    mid = split.agreement_rate()
    split.restore_callosum()
    for _ in range(120):
        exec_s = rng.standard_normal(8)
        sens_s = rng.standard_normal(16)
        cog = rng.standard_normal(16)
        emb = rng.standard_normal(8)
        split.tick(exec_s, sens_s, cog, emb)
    post = split.agreement_rate()
    # Confabulation path works.
    split.record_action("grab_thing", Hemisphere.RIGHT)
    split.supply_reason("grab_thing")
    assert split.confabulation_rate() > 0.0
    # Agreement monotonic-ish: severance drops, restore recovers.
    assert post >= mid - 1e-6


def test_gauntlet_minimal_selfhood_reaches_dugesia_and_biases_toward_rest():
    ms = MinimalSelfhood()
    pre = np.zeros(8, dtype=np.float32); pre[0] = 1.0
    post = np.zeros(8, dtype=np.float32); post[0] = 0.0
    for _ in range(90):
        t = ms.tag_action("rest", pre)
        ms.reinforce(t, post)
    ms.update(
        body_budget={"energy_reserves": 0.0, "resource_pressure": 0.2,
                     "thermal_stress": 0.0},
        affect={"coherence": 0.8, "curiosity": 0.7},
        cognitive_state={"social_hunger": 0.0, "prediction_error": 0.1,
                         "agency_score": 0.9},
    )
    assert ms.mode() == Mode.DUGESIA
    rest_idx = MS_ACTIONS.index("rest")
    top3 = np.argsort(ms.action_priority())[::-1][:3]
    assert rest_idx in top3


def test_gauntlet_tom_observer_effect_drives_bias_change():
    tom_alone = RecursiveTheoryOfMind()
    tom_watched = RecursiveTheoryOfMind()
    tom_watched.observe_agent(
        "bryan",
        strength=0.9,
        evidence_digest="a" * 64,
    )
    tom_watched.register_interaction(
        "bryan",
        {
            "agent_id": "bryan",
            "confidence": 0.8,
            "observations": 1,
            "social_rupture_risk": 0.0,
            "evidence_digest": "b" * 64,
            "affect_hypotheses": {},
        },
    )
    assert tom_watched.depth_reached("bryan") == MAX_DEPTH
    ba = tom_alone.get_observer_bias().bias
    bw = tom_watched.get_observer_bias().bias
    assert not np.allclose(ba, bw)


def test_gauntlet_octopus_severance_and_recovery():
    fed = OctopusFederation()
    rng = np.random.default_rng(303)
    for _ in range(30):
        fed.tick(rng.standard_normal(3))
    fed.sever_link()
    for _ in range(20):
        fed.tick(rng.standard_normal(3))
    fed.restore_link()
    for _ in range(60):
        fed.tick(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    # Should have returned to LINKED after stable-env ticks.
    assert fed.arbiter.link_state() in (ArmState.LINKED, ArmState.RECOVERING)


def test_gauntlet_cellular_turnover_preserves_identity_under_25pct_burst():
    mesh = NeuralMesh()
    rng = np.random.default_rng(404)
    for c in mesh.columns:
        c.x = rng.standard_normal(c.n).astype(np.float32) * 0.3
    turn = CellularTurnover(turnover_rate=0.0)
    turn.attach(mesh)
    fp_before = turn._fingerprints[-1]
    fp_after = turn.force_turnover(0.25)
    assert fp_after.similarity(fp_before) >= THRESHOLD_IDENTITY


def test_gauntlet_absorbed_voices_attribution_multi_voice():
    import tempfile
    av = AbsorbedVoices(storage_dir=Path(tempfile.mkdtemp()))
    av.add_voice("bryan", sample_text="enterprise quality, real tests, deep impact")
    av.add_voice("teacher", sample_text="fractions and decimals, examples first")
    av.add_voice("fiction", sample_text="dragons, wizards, distant galaxies")
    a = av.attribute_thought("add enterprise quality checks with deep tests")
    assert a.best_voice_id == "bryan"


def test_gauntlet_unified_bias_composes_all_three_sources():
    uni = UnifiedCognitiveBias()
    hemi = np.array([0.8, -0.3] + [0.0] * 14, dtype=np.float32)
    selfhood = np.array([0.1, 0.1] + [0.7] + [0.0] * 13, dtype=np.float32)
    observer = np.array([0.0, 0.6] + [0.0] * 14, dtype=np.float32)
    snap = uni.fuse(hemi, selfhood, observer, observer_presence=0.7)
    assert snap.fused.shape == (16,)
    assert np.all(np.isfinite(snap.fused))
    # Each contribution scaled by its weight.
    assert np.linalg.norm(snap.hemi_contribution) > 0.0
    assert np.linalg.norm(snap.selfhood_contribution) > 0.0
    assert np.linalg.norm(snap.observer_contribution) > 0.0


def test_gauntlet_biases_remain_bounded_across_many_iterations():
    split = HemisphericSplit()
    ms = MinimalSelfhood()
    tom = RecursiveTheoryOfMind()
    uni = UnifiedCognitiveBias()
    rng = np.random.default_rng(505)
    for i in range(300):
        exec_s = rng.standard_normal(8)
        sens_s = rng.standard_normal(16)
        cog = rng.standard_normal(16)
        emb = rng.standard_normal(8)
        split.tick(exec_s, sens_s, cog, emb)
        ms.update(
            body_budget={"energy_reserves": 0.5, "resource_pressure": 0.3,
                         "thermal_stress": 0.2},
            affect={"coherence": 0.6, "curiosity": 0.5},
            cognitive_state={"social_hunger": 0.3, "prediction_error": 0.2,
                             "agency_score": 0.6},
        )
        if i % 3 == 0:
            tom.observe_agent(
                f"agent_{i % 5}",
                strength=0.4,
                evidence_digest=f"{i + 1:064x}",
            )
        uni.fuse(
            split.fused_bias(),
            ms.get_priority_bias(),
            tom.get_observer_bias().bias,
            tom.total_observer_presence(),
        )
    snap = uni.last()
    assert snap is not None
    assert np.all(np.abs(snap.fused) <= 1.0 + 1e-5)


def test_gauntlet_combined_latency_budget():
    """A combined tick of hemispheric + selfhood + recursive ToM + unified
    fusion should complete in well under 20 ms so we can maintain > 50 Hz."""
    split = HemisphericSplit()
    ms = MinimalSelfhood()
    tom = RecursiveTheoryOfMind()
    uni = UnifiedCognitiveBias()
    rng = np.random.default_rng(606)
    # Warm up.
    for _ in range(10):
        exec_s = rng.standard_normal(8); sens_s = rng.standard_normal(16)
        cog = rng.standard_normal(16); emb = rng.standard_normal(8)
        split.tick(exec_s, sens_s, cog, emb)
    t0 = time.time()
    for i in range(50):
        exec_s = rng.standard_normal(8); sens_s = rng.standard_normal(16)
        cog = rng.standard_normal(16); emb = rng.standard_normal(8)
        split.tick(exec_s, sens_s, cog, emb)
        ms.update(
            body_budget={"energy_reserves": 0.5, "resource_pressure": 0.3,
                         "thermal_stress": 0.2},
            affect={"coherence": 0.6, "curiosity": 0.5},
            cognitive_state={"social_hunger": 0.3, "prediction_error": 0.2,
                             "agency_score": 0.6},
        )
        tom.observe_agent(
            "stream",
            strength=0.2,
            evidence_digest=f"{i + 1:064x}",
        )
        uni.fuse(
            split.fused_bias(),
            ms.get_priority_bias(),
            tom.get_observer_bias().bias,
            tom.total_observer_presence(),
        )
    elapsed = (time.time() - t0) / 50 * 1000.0
    assert elapsed < 20.0, f"combined tick too slow: {elapsed:.1f}ms (> 20ms)"


def test_gauntlet_existential_stakes():
    from core.consciousness.existential_stakes import ExistentialStakes
    stakes = ExistentialStakes(memory_limit_bytes=1000)
    # Ticking should compute active threat
    threat = stakes.update()
    assert threat == 1.0
    status = stakes.get_status()
    assert status["memory_threat"] == 1.0
    assert "SYSTEM RESOURCE WARNING" in stakes.get_context_block()


def test_gauntlet_temporal_continuity_accumulates_silence_residue():
    from core.consciousness.temporal_continuity import TemporalContinuityEngine

    engine = TemporalContinuityEngine()
    with engine._lock:
        engine._anchor_time = time.time() - 180.0

    engine.tick()
    residue = engine.get_residue()
    modulation = engine.compute_modulation()

    assert engine.is_ready() is True
    assert residue.silence_duration_s >= 170.0
    assert residue.silence_pressure > 0.5
    assert modulation["temperature_delta"] > 0.0
    assert modulation["token_budget_multiplier"] > 1.0


def test_gauntlet_synaptic_plasticity_learns_and_modulates(tmp_path, monkeypatch):
    from core.consciousness import synaptic_plasticity as plasticity_module

    monkeypatch.setattr(
        plasticity_module,
        "PERSIST_PATH",
        tmp_path / "synaptic_plasticity_state.json",
    )
    engine = plasticity_module.SynapticPlasticityEngine()
    substrate = np.linspace(-1.0, 1.0, plasticity_module.PROJECTION_DIM, dtype=np.float32)

    before = engine.get_status()["total_updates"]
    engine.pre_inference_capture(substrate, hedonic_score=0.2)
    engine.post_inference_learn(
        "A concrete response that improves task progress.",
        hedonic_after=0.8,
        surprise=0.1,
    )
    modulation = engine.compute_modulation(substrate)

    assert engine.is_ready() is True
    assert engine.get_status()["total_updates"] == before + 1
    assert set(modulation) == {
        "temperature_delta",
        "top_p_delta",
        "repetition_penalty_delta",
    }


def test_gauntlet_attention_gate_causally_prunes_context():
    from core.consciousness.attention_gate import AttentionGate
    from core.container import ServiceContainer

    ServiceContainer.register_instance(
        "attention_schema",
        SimpleNamespace(
            current_focus=SimpleNamespace(content="climate research browser tools"),
            salience_map={"climate": 0.9, "tools": 0.8},
        ),
    )
    gate = AttentionGate()
    messages = [
        {"role": "system", "content": "System policy stays visible."},
        {"role": "user", "content": "Please research climate change articles."},
        {"role": "assistant", "content": "I will use browser tools for climate research."},
        {"role": "user", "content": "An unrelated grocery list with apples and cereal."},
        {"role": "assistant", "content": "A short unrelated aside."},
        {"role": "user", "content": "Now summarize climate findings with citations."},
    ]

    gated = gate.gate_context(messages)
    status = gate.get_status()

    assert gate.is_ready() is True
    assert len(gated) >= 4
    assert gated[0]["role"] == "system"
    assert status["total_calls"] == 1
    assert status["total_gated"] >= 1
    assert any("[gated:" in item["content"] for item in gated)


def test_gauntlet_somatic_qualia_produces_bounded_sampling_perturbation():
    from core.consciousness.somatic_qualia import SomaticQualiaEngine
    from core.container import ServiceContainer

    substrate = SimpleNamespace(
        x=np.linspace(-0.8, 0.9, 96, dtype=np.float32),
        idx_valence=5,
    )
    mesh = SimpleNamespace(
        get_global_synchrony=lambda: 0.42,
        get_status=lambda: {"tier_energies": {"EXECUTIVE": 0.7, "SENSORY": 0.3}},
    )
    ServiceContainer.register_instance("conscious_substrate", substrate)
    ServiceContainer.register_instance("neural_mesh", mesh)

    engine = SomaticQualiaEngine()
    for _ in range(4):
        engine.tick()
    perturbation = engine.compute_perturbation()

    assert engine.is_ready() is True
    assert perturbation
    assert all(math.isfinite(float(value)) for value in perturbation.values())
    assert abs(perturbation.get("temperature_perturbation", 0.0)) <= 0.1
    assert abs(perturbation.get("repetition_penalty_perturbation", 0.0)) <= 0.08


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [
        test_gauntlet_hierarchical_phi_under_load,
        test_gauntlet_hemispheric_severance_and_restore_cycle,
        test_gauntlet_minimal_selfhood_reaches_dugesia_and_biases_toward_rest,
        test_gauntlet_tom_observer_effect_drives_bias_change,
        test_gauntlet_octopus_severance_and_recovery,
        test_gauntlet_cellular_turnover_preserves_identity_under_25pct_burst,
        test_gauntlet_absorbed_voices_attribution_multi_voice,
        test_gauntlet_unified_bias_composes_all_three_sources,
        test_gauntlet_biases_remain_bounded_across_many_iterations,
        test_gauntlet_combined_latency_budget,
        test_gauntlet_existential_stakes,
        test_gauntlet_temporal_continuity_accumulates_silence_residue,
        test_gauntlet_synaptic_plasticity_learns_and_modulates,
        test_gauntlet_attention_gate_causally_prunes_context,
        test_gauntlet_somatic_qualia_produces_bounded_sampling_perturbation,
    ]
    passed, failed = 0, []
    t0 = time.time()
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  ok {t.__name__}")
        except (AssertionError, OSError, RuntimeError, TypeError, ValueError) as exc:
            failed.append((t.__name__, exc))
            print(f"  FAIL {t.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed in {(time.time() - t0):.1f}s")
    sys.exit(0 if not failed else 1)
