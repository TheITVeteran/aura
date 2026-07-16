"""Contract tests: the falsification harness itself.

Two layers of honesty are tested here:
1. Task generators are exact, deterministic, self-verifying, and their
   depth knob really controls compositional depth.
2. The graders cannot be gamed: underpowered evidence grades CONJECTURE,
   losses grade REFUTED, compute-mismatched width comparisons are voided,
   and the engine hookup (slot ablation) works end to end on a real model.
"""
from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.experiments import (
    CONJECTURE,
    PROVEN,
    REFUTED,
    SUPPORTED,
    ArmResult,
    grade_treatment_vs_control,
    khop_reachability,
    modular_chain,
    nested_boolean,
    run_depth_extrapolation,
    run_latent_opt_control,
    run_recurrence_sweep,
    run_slot_causality,
    run_virtual_width,
    task_battery,
)


# ── Task generators ─────────────────────────────────────────────────────


def test_generators_are_deterministic_and_self_verifying():
    for gen in (khop_reachability, nested_boolean, modular_chain):
        a, b = gen(4, seed=9), gen(4, seed=9)
        assert a.prompt == b.prompt and a.answer == b.answer
        assert a.verify(f"thinking... the answer is {a.answer}")
        assert not a.verify("the answer is definitely 99999")
        assert a.depth == 4


def test_khop_answer_is_actually_reachable():
    task = khop_reachability(6, seed=3, n_nodes=8)
    edges = dict(
        tuple(map(int, pair.split("->")))
        for pair in task.prompt.split(": ")[1].split(". ")[0].split(", ")
    )
    start = int(task.prompt.split("Start at node ")[1].split(" ")[0])
    node = start
    for _ in range(6):
        node = edges[node]
    assert str(node) == task.answer


def test_modular_chain_answer_in_range():
    task = modular_chain(10, seed=5, mod=17)
    assert 0 <= int(task.answer) < 17


def test_verify_takes_last_claim_not_first():
    task = modular_chain(3, seed=1, mod=17)
    wrong = str((int(task.answer) + 1) % 17)
    assert task.verify(f"maybe {wrong}... no wait, it is {task.answer}")
    assert not task.verify(f"it is {task.answer}. Actually no: {wrong}")


def test_battery_covers_families_and_depths():
    tasks = task_battery(["khop", "boolean"], [2, 4], per_cell=3, seed=1)
    assert len(tasks) == 12
    assert {t.family for t in tasks} == {"khop", "boolean"}


# ── Graders ─────────────────────────────────────────────────────────────


def _arm(name, n, k):
    return ArmResult(name=name, n=n, successes=k)


def test_grader_underpowered_is_conjecture():
    claim = grade_treatment_vs_control(
        "x", "s",
        {"khop": _arm("t", 5, 5)},
        {"khop": _arm("c", 5, 0)},
    )
    assert claim.tier == CONJECTURE


def test_grader_single_family_separation_is_supported():
    claim = grade_treatment_vs_control(
        "x", "s",
        {"khop": _arm("t", 40, 36)},
        {"khop": _arm("c", 40, 10)},
    )
    assert claim.tier == SUPPORTED


def test_grader_two_family_separation_is_proven():
    claim = grade_treatment_vs_control(
        "x", "s",
        {"khop": _arm("t", 40, 36), "modular": _arm("t", 40, 34)},
        {"khop": _arm("c", 40, 10), "modular": _arm("c", 40, 12)},
    )
    assert claim.tier == PROVEN


def test_grader_losses_are_refuted():
    claim = grade_treatment_vs_control(
        "x", "s",
        {"khop": _arm("t", 40, 12)},
        {"khop": _arm("c", 40, 20)},
    )
    assert claim.tier == REFUTED


def test_recurrence_sweep_grades_monotone_gain():
    # Synthetic solver: succeeds iff steps >= task depth (perfect scaling).
    tasks = task_battery(["modular"], [1, 2, 4], per_cell=8, seed=2)
    result = run_recurrence_sweep(lambda t, s: s >= t.depth, tasks, [1, 2, 4])
    assert result["monotone_gain"] is True
    assert result["claim"]["tier"] == SUPPORTED
    flat = run_recurrence_sweep(lambda t, s: t.seed % 2 == 0, tasks, [1, 2, 4])
    assert flat["claim"]["tier"] in (REFUTED, CONJECTURE)


def test_depth_extrapolation_detects_scaling_and_flat():
    scaling = run_depth_extrapolation(
        lambda t, s: s >= t.depth, "modular", [2, 4, 8], [2, 4, 8], per_depth=8
    )
    assert scaling["claim"]["tier"] == SUPPORTED
    assert scaling["t_required"] == {2: 2, 4: 4, 8: 8}
    flat = run_depth_extrapolation(
        lambda t, s: True, "modular", [2, 4, 8], [2, 4, 8], per_depth=8
    )
    assert flat["claim"]["tier"] == REFUTED  # solvable but T does not scale


def test_slot_causality_requires_real_damage():
    tasks = task_battery(["boolean"], [3], per_cell=24, seed=3)
    hit = run_slot_causality(
        lambda t, slot: slot != 1,  # slot 1 is load-bearing
        tasks,
        slot_indices=[0, 1, 2],
    )
    assert hit["causally_necessary_slots"] == [1]
    assert hit["claim"]["tier"] == SUPPORTED
    miss = run_slot_causality(lambda t, slot: True, tasks, slot_indices=[0, 1])
    assert miss["claim"]["tier"] == REFUTED


def test_virtual_width_voids_unequal_compute():
    tasks = {"khop": task_battery(["khop"], [3], per_cell=24, seed=4)}
    honest = run_virtual_width(
        lambda t, k: (True, 1000),
        lambda t, k: (t.seed % 3 == 0, 1000),
        tasks,
        k=4,
    )
    assert honest["claim"]["tier"] == SUPPORTED
    cheat = run_virtual_width(
        lambda t, k: (True, 5000),  # branches quietly spent 5× compute
        lambda t, k: (t.seed % 3 == 0, 1000),
        tasks,
        k=4,
    )
    assert cheat["claim"]["tier"] == CONJECTURE
    assert "compute mismatch" in cheat["claim"]["evidence"]["voided"]


def test_latent_opt_control_only_rewards_direction():
    tasks = {"modular": task_battery(["modular"], [3], per_cell=24, seed=5)}
    directional = run_latent_opt_control(
        lambda t, arm: arm == "gradient" or (arm == "control" and t.seed % 4 == 0),
        tasks,
    )
    assert directional["claim"]["tier"] == SUPPORTED
    indiscriminate = run_latent_opt_control(
        lambda t, arm: arm in ("gradient", "control"), tasks
    )
    assert indiscriminate["claim"]["tier"] in (REFUTED, CONJECTURE)


# ── End-to-end engine hookup (real tiny model) ──────────────────────────


def test_engine_ablation_hook_runs_and_flags():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        CortexConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    args = ModelArgs(
        model_type="qwen2", hidden_size=64, num_hidden_layers=8,
        intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=128, num_key_value_heads=2, max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    engine = LatentCortexEngine(
        model,
        config=CortexConfig(
            workspace=WorkspaceConfig(n_slots=4, seed=3),
            recurrence=RecurrenceConfig(max_steps=4, min_steps=2),
            branches=BranchConfig(n_branches=1),
            decode_max_tokens=6,
        ),
    )
    intact = engine.reason(token_ids=[5, 9, 17, 3, 42])
    ablated = engine.reason(token_ids=[5, 9, 17, 3, 42], ablate_slot=1)
    assert intact.ok and ablated.ok
    assert "slot_ablated:1:zero" in ablated.receipt.honest_flags
    assert intact.receipt.first_logits_digest != ablated.receipt.first_logits_digest, (
        "slot ablation must causally move the answer distribution"
    )
