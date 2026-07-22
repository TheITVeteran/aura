"""Contract tests: the falsification harness itself.

Two layers of honesty are tested here:
1. Task generators are exact, deterministic, self-verifying, and their
   depth knob really controls compositional depth.
2. The graders cannot be gamed: underpowered evidence grades CONJECTURE,
   losses grade REFUTED, compute-mismatched width comparisons are voided,
   and the engine hookup (slot ablation) works end to end on a real model.
"""
from __future__ import annotations

import argparse

import pytest

from core.brain.llm.latent_cortex.experiments import (
    CONJECTURE,
    PROVEN,
    REFUTED,
    SUPPORTED,
    ArmResult,
    PairedObservation,
    grade_paired_treatment_vs_control,
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
from tools.latent_cortex_lab import _positive_float

# ── Task generators ─────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_lab_deadline_parser_rejects_nonfinite_or_nonpositive_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_float(value)


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


def test_aggregate_grader_cannot_claim_proven_without_pairs():
    claim = grade_treatment_vs_control(
        "x", "s",
        {"khop": _arm("t", 40, 36), "modular": _arm("t", 40, 34)},
        {"khop": _arm("c", 40, 10), "modular": _arm("c", 40, 12)},
    )
    assert claim.tier == SUPPORTED
    assert claim.evidence["aggregate_only"] is True


def test_paired_grader_can_prove_replicated_compute_matched_gain():
    observations = {
        family: [
            PairedObservation(
                task_id=f"{family}-{index}",
                family=family,
                treatment_success=True,
                control_success=index % 4 == 0,
                treatment_layer_apps=1000,
                control_layer_apps=1000,
            )
            for index in range(40)
        ]
        for family in ("khop", "modular")
    }
    claim = grade_paired_treatment_vs_control("x", "s", observations)
    assert claim.tier == PROVEN
    assert set(claim.evidence["positive_families"]) == {"khop", "modular"}


def test_paired_grader_voids_missing_or_mismatched_compute():
    missing = {
        "khop": [
            PairedObservation(f"t-{i}", "khop", True, False) for i in range(30)
        ]
    }
    assert grade_paired_treatment_vs_control("x", "s", missing).tier == CONJECTURE
    mismatch = {
        "khop": [
            PairedObservation(f"t-{i}", "khop", True, False, 2000, 1000)
            for i in range(30)
        ]
    }
    claim = grade_paired_treatment_vs_control("x", "s", mismatch)
    assert claim.tier == CONJECTURE
    assert claim.evidence["invalid_compute_families"] == ["khop"]

    zero = {
        "khop": [
            PairedObservation(f"z-{i}", "khop", True, False, 0, 0)
            for i in range(30)
        ]
    }
    zero_claim = grade_paired_treatment_vs_control("x", "s", zero)
    assert zero_claim.tier == CONJECTURE
    assert zero_claim.evidence["invalid_compute_families"] == ["khop"]


def test_paired_grader_rejects_duplicate_or_malformed_rows():
    duplicate = PairedObservation("same", "khop", True, False, 100, 100)
    with pytest.raises(ValueError, match="unique"):
        grade_paired_treatment_vs_control(
            "x", "s", {"khop": [duplicate, duplicate]}, require_compute=False
        )
    malformed = PairedObservation("bad", "khop", 1, False, 100, 100)
    with pytest.raises(ValueError, match="boolean"):
        grade_paired_treatment_vs_control(
            "x", "s", {"khop": [malformed]}, require_compute=False
        )
    with pytest.raises(ValueError, match="alpha"):
        grade_paired_treatment_vs_control("x", "s", {}, alpha=float("nan"))


def test_paired_effect_bound_tracks_alpha_and_blocks_weak_strict_claims():
    observations = {
        "khop": [
            PairedObservation(
                f"task-{index}",
                "khop",
                index < 21,
                False,
                1000,
                1000,
            )
            for index in range(30)
        ]
    }
    ordinary = grade_paired_treatment_vs_control(
        "x",
        "s",
        observations,
        alpha=0.05,
        minimum_effect=0.53,
    )
    strict = grade_paired_treatment_vs_control(
        "x",
        "s",
        observations,
        alpha=0.01,
        minimum_effect=0.53,
    )

    ordinary_stats = ordinary.evidence["families"]["khop"]
    strict_stats = strict.evidence["families"]["khop"]
    assert ordinary_stats["effect_bound_alpha"] == 0.05
    assert strict_stats["effect_bound_alpha"] == 0.01
    assert strict_stats["effect_interval"][0] <= ordinary_stats["effect_interval"][0]
    assert ordinary.evidence["positive_families"] == ["khop"]
    assert strict.evidence["positive_families"] == []


def test_paired_proven_requires_two_thirds_domain_breadth():
    observations = {
        family: [
            PairedObservation(
                f"{family}-{index}",
                family,
                family in {"a", "b"},
                False,
                1000,
                1000,
            )
            for index in range(40)
        ]
        for family in ("a", "b", "c", "d")
    }
    claim = grade_paired_treatment_vs_control("x", "s", observations)
    assert claim.tier == SUPPORTED
    assert claim.evidence["required_positive_families"] == 3


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
    result = run_recurrence_sweep(
        lambda t, s: (s >= t.depth, 1000),
        tasks,
        [1, 2, 4],
        baseline=lambda t: (t.depth == 1, 1000),
    )
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
    assert cheat["claim"]["evidence"]["invalid_compute_families"] == ["khop"]


def test_latent_opt_control_only_rewards_direction():
    tasks = {"modular": task_battery(["modular"], [3], per_cell=24, seed=5)}
    directional = run_latent_opt_control(
        lambda t, arm: (
            arm == "gradient" or (arm == "control" and t.seed % 4 == 0),
            1000,
        ),
        tasks,
    )
    assert directional["claim"]["tier"] == SUPPORTED
    indiscriminate = run_latent_opt_control(
        lambda t, arm: (arm in ("gradient", "control"), 1000), tasks
    )
    assert indiscriminate["claim"]["tier"] in (REFUTED, CONJECTURE)


def test_latent_opt_control_counterbalances_and_reports_execution_order():
    tasks = {"modular": task_battery(["modular"], [3], per_cell=24, seed=55)}
    observed_calls: list[tuple[int, str]] = []

    def solve(task, arm):
        observed_calls.append((task.seed, arm))
        return arm == "gradient", 1000

    result = run_latent_opt_control(solve, tasks)
    reported = result["execution_order"]
    flattened = [
        (tasks["modular"][index].seed, arm)
        for index, row in enumerate(reported)
        for arm in row["arms"]
    ]
    assert observed_calls == flattened
    gradient_first = sum(
        row["arms"].index("gradient") < row["arms"].index("control")
        for row in reported
    )
    assert abs(gradient_first - (len(reported) - gradient_first)) <= 1
    assert {tuple(row["arms"]) for row in reported} != {
        ("off", "gradient", "control")
    }


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


# ── Answer extraction + self-consistency voting (Experiment 4 control) ──


def test_extract_final_numeric_claim_uses_last_claim():
    from core.brain.llm.latent_cortex.experiments import extract_final_numeric_claim

    assert extract_final_numeric_claim("Thinking... 3 then 7, final answer: 5.") == "5"
    assert extract_final_numeric_claim("answer is -4") == "-4"
    assert extract_final_numeric_claim("no numbers here") == ""


def test_majority_answer_reports_no_majority_instead_of_breaking_ties():
    """CP126: a tie is the ABSENCE of a majority, not a decision.

    This test previously pinned the defect — it asserted the lexicographic
    winner. Manufacturing a definite answer from an undecided sample means a
    tie can be graded correct by luck of alphabetical order, inflating the
    self-consistency baseline this helper feeds.
    """
    from core.brain.llm.latent_cortex.experiments import majority_answer

    assert majority_answer(["5", "5", "7"]) == "5"
    assert majority_answer(["", "", "9"]) == "9"
    assert majority_answer([]) == ""
    # Tie ⇒ no majority, regardless of ordering.
    assert majority_answer(["3", "7"]) == ""
    assert majority_answer(["7", "3"]) == ""
    assert majority_answer(["3", "3", "7", "7"]) == ""
    # A genuine plurality still wins.
    assert majority_answer(["3", "3", "7"]) == "3"


# ── Factorial ablations: mechanism attribution ───────────────────────────


def test_factorial_ablations_attribute_gain_to_the_right_mechanism():
    from core.brain.llm.latent_cortex.experiments import (
        FACTORIAL_ARMS,
        run_factorial_ablations,
        task_battery,
    )

    tasks = task_battery(["modular"], [2], 24, seed=5)
    by_family = {"modular": tasks}

    def solve_arm(task, arm):
        # Deterministic synthetic world: recurrence-bearing arms solve
        # everything; vanilla and the others solve nothing.
        winners = {"recurrence_only", "recurrence_branches", "full_stack"}
        return (arm in winners), 100
    result = run_factorial_ablations(solve_arm, by_family)

    assert set(result["claims"]) == set(FACTORIAL_ARMS)
    assert set(result["attribution"]) == {
        "recurrence_only",
        "recurrence_branches",
        "full_stack",
    }
    losing = result["claims"]["branches_only"]
    assert losing["tier"] in {"CONJECTURE", "REFUTED"}
    vanilla_arm = result["arms"]["vanilla"]["modular"]
    assert vanilla_arm["n"] == 24 and vanilla_arm["successes"] == 0


def test_factorial_ablations_underpowered_stays_conjecture():
    from core.brain.llm.latent_cortex.experiments import (
        run_factorial_ablations,
        task_battery,
    )

    tasks = task_battery(["modular"], [2], 4, seed=6)  # n=4 << MIN_N
    result = run_factorial_ablations(
        lambda task, arm: (arm != "vanilla", 10),
        {"modular": tasks},
        arms=("full_stack",),
    )
    assert result["claims"]["full_stack"]["tier"] == "CONJECTURE"


# ── Experiment R: role lesion/swap causality ─────────────────────────────


@pytest.fixture(scope="module")
def tiny_model():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    args = ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=8,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def test_role_override_controls_branch_roles(tiny_model):
    from core.brain.llm.latent_cortex.engine import LatentCortexEngine
    from core.brain.llm.latent_cortex.types import (
        BranchConfig,
        ComputeBudget,
        CortexConfig,
        RecurrenceConfig,
        WorkspaceConfig,
    )

    engine = LatentCortexEngine(
        tiny_model,
        config=CortexConfig(
            workspace=WorkspaceConfig(n_slots=4, seed=7),
            recurrence=RecurrenceConfig(max_steps=2, min_steps=1),
            branches=BranchConfig(
                n_branches=2, roles=("analogy", "analogy")
            ),
            decode_max_tokens=4,
        ),
    )
    result = engine.reason(token_ids=[5, 9, 17, 3], budget=ComputeBudget())
    assert result.ok


def test_role_override_must_match_branch_count(tiny_model):
    from core.brain.llm.latent_cortex.branches import BranchEnsemble
    from core.brain.llm.latent_cortex.types import BranchConfig

    with pytest.raises(ValueError, match="exactly n_branches"):
        BranchEnsemble.seed(
            None,
            None,
            BranchConfig(n_branches=3, roles=("analogy",)),
            None,
            None,
            None,
            0,
        )


def test_run_role_lesion_grades_diversity_and_divergence():
    from core.brain.llm.latent_cortex.experiments import (
        ROLE_ARMS,
        run_role_lesion,
        task_battery,
    )

    battery = task_battery(["boolean"], [2], 20, seed=3)
    by_family = {"boolean": battery}
    lesioned_calls = {"n": 0}

    def solve_arm(task, arm):
        # Distinct/swapped roles solve everything with high divergence;
        # the lesioned ensemble collapses (low divergence, half the wins).
        if arm in {"distinct_roles", "swapped_roles", "restored_roles"}:
            return True, 100, 0.30
        lesioned_calls["n"] += 1
        return (lesioned_calls["n"] % 2 == 0), 100, 0.05

    report = run_role_lesion(solve_arm, by_family)
    assert set(report["arms"]) == set(ROLE_ARMS)
    assert report["behavioral_claim"]["tier"] in {"PROVEN", "SUPPORTED"}
    assert report["restoration_claim"]["tier"] in {"PROVEN", "SUPPORTED"}
    assert report["divergence_claim"]["tier"] == "SUPPORTED"
    assert report["role_causality"]["tier"] == "SUPPORTED"
    assert report["role_causality"]["compute_parity"] is True
    parity = report["swap_parity"]["boolean"]
    assert parity["distinct_accuracy"] == parity["swapped_accuracy"] == 1.0
    assert parity["task_compute_matched"] is True


def test_run_role_lesion_refutes_when_lesion_changes_nothing():
    from core.brain.llm.latent_cortex.experiments import (
        run_role_lesion,
        task_battery,
    )

    battery = task_battery(["boolean"], [2], 20, seed=5)
    by_family = {"boolean": battery}

    def solve_arm(task, arm):
        return True, 100, 0.10  # identical everywhere: roles carry nothing

    report = run_role_lesion(solve_arm, by_family)
    assert report["behavioral_claim"]["tier"] in {"CONJECTURE", "REFUTED"}
    assert report["divergence_claim"]["tier"] == "REFUTED"
    assert report["role_causality"]["tier"] == "CONJECTURE"


def test_run_role_lesion_conjectures_without_telemetry():
    from core.brain.llm.latent_cortex.experiments import (
        run_role_lesion,
        task_battery,
    )

    battery = task_battery(["boolean"], [2], 6, seed=9)

    def solve_arm(task, arm):
        return True, 100, float("nan")  # no exchange telemetry recorded

    report = run_role_lesion(solve_arm, {"boolean": battery})
    assert report["divergence_claim"]["tier"] == "CONJECTURE"


def test_run_role_lesion_cannot_claim_causality_with_compute_mismatch():
    from core.brain.llm.latent_cortex.experiments import (
        run_role_lesion,
        task_battery,
    )

    battery = task_battery(["boolean"], [2], 20, seed=11)
    calls = {"lesioned_uniform_role": 0}

    def solve_arm(task, arm):
        if arm == "lesioned_uniform_role":
            calls[arm] += 1
            return calls[arm] % 2 == 0, 100, 0.05
        return True, 101 if arm == "distinct_roles" else 100, 0.30

    report = run_role_lesion(solve_arm, {"boolean": battery})
    assert report["role_causality"]["compute_parity"] is False
    assert report["role_causality"]["tier"] == "CONJECTURE"
