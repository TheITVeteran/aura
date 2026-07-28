"""SPARK-061 contracts: a descending loss curve is not evidence by itself.

The tests that matter here are the ones that hand the instrument a *perfect*
improvement curve produced by a dead operator and require it to say no. Every
other check in the file is subordinate to that one, because the identity-
collapse solution is the failure the objective invites and the only one that
looks like success all the way to the end of a campaign.

Where the model matters the tests run a real MLX Qwen2 through the real live
path — the same `_prepare_live_path` / `_advance_recurrent_states` /
`_persist_and_score` chain the trainer uses — so displacement, necessity, and
state gradients are measured on genuine recurrence rather than a stub.
"""

from __future__ import annotations

import copy
import math

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.learning.progressive_recurrent_objective import (  # noqa: E402
    DEFAULT_DISPLACEMENT_FLOOR,
    MEASURED_1P5B_COLLAPSE_SOLUTION,
    PROGRESSIVE_OBJECTIVE_SCHEMA,
    PROGRESSIVE_TRAJECTORY_SET_SCHEMA,
    ProgressiveObjectiveError,
    ProgressiveTrajectory,
    ProgressiveTrajectorySet,
    build_progressive_report,
    canonical_sha256,
    displacement_floor_penalty,
    measure_progressive_trajectories,
    measure_progressive_trajectory,
    progressive_objective_loss,
    solve_collapse_barrier,
    state_gradient_norms,
    step_necessity,
    validate_progressive_report,
    validate_progressive_trajectory_set,
)

# The exact per-alpha means measured on the untrained Qwen2.5-1.5B at depth 4
# over khop + modular (6 tasks). Pinned here so the barrier analysis is a
# regression test rather than a one-off script result: (alpha, v4 loss, mean
# min displacement).
MEASURED_1P5B_SWEEP = (
    (0.002, 3.352137, 0.002404),
    (0.01, 3.343501, 0.011747),
    (0.05, 3.488149, 0.052329),
    (0.1, 3.53367, 0.089603),
    (0.25, 3.014076, 0.145112),
    (0.5, 2.980107, 0.185419),
)

PROMPT = [5, 9, 17, 3, 42]
ANSWER = [7, 11, 23]


def _model() -> Model:
    args = ModelArgs(
        model_type="qwen2",
        hidden_size=32,
        num_hidden_layers=4,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def tiny_model() -> Model:
    return _model()


def _spec(**overrides) -> RLCExecutionSpec:
    base = dict(
        n_slots=4,
        branch_roles=("constructive_solution",),
        recurrent_steps=2,
        prelude_frac=0.25,
        coda_frac=0.25,
    )
    base.update(overrides)
    return RLCExecutionSpec(**base)


def _trajectory(
    losses: tuple[float, ...],
    displacements: tuple[float, ...],
    *,
    answer_tokens: int = 3,
) -> ProgressiveTrajectory:
    return ProgressiveTrajectory(
        depth=len(losses),
        probe_steps=tuple(range(1, len(losses) + 1)),
        step_losses=losses,
        displacements=displacements,
        anchor_drifts=tuple(0.5 for _ in losses),
        answer_token_count=answer_tokens,
    )


# ── The central contract ────────────────────────────────────────────────


def test_perfect_improvement_from_a_dead_operator_is_refused():
    """The failure the whole module exists for.

    Losses fall monotonically and substantially — an unimpeachable curve. The
    operator never moved the state. Without the displacement measurement this
    reads as textbook success; the instrument must call it collapse and must
    withhold training support.
    """
    collapsed = _trajectory(
        losses=(2.40, 1.80, 1.20, 0.60),
        displacements=(1e-7, 1e-7, 1e-7, 1e-7),
    )
    report = build_progressive_report([collapsed])

    assert report["verdict"] == "degenerate_identity_collapse"
    assert report["supports_training"] is False
    # The improvement is real arithmetic; it just does not belong to recurrence.
    assert report["mean_improvement"] == pytest.approx(1.8, abs=1e-6)
    validate_progressive_report(report)


def test_a_collapsed_report_cannot_be_relabelled_as_progress():
    """Forging the verdict past the collapse evidence must not validate."""
    collapsed = _trajectory(
        losses=(2.40, 1.20),
        displacements=(1e-7, 1e-7),
    )
    report = build_progressive_report([collapsed])
    forged = {key: value for key, value in report.items() if key != "receipt_sha256"}
    forged["verdict"] = "real_progress"
    forged["supports_training"] = True
    forged["receipt_sha256"] = canonical_sha256(forged)

    # Caught by the full rebuild: the rows still say collapsed.
    with pytest.raises(ProgressiveObjectiveError, match="does not replay"):
        validate_progressive_report(forged)


# ── The other three degeneracies ────────────────────────────────────────


def test_early_step_sabotage_is_named():
    """Improvement bought by making step 1 worse than an honest depth-1 run."""
    sabotaged = _trajectory(losses=(3.00, 1.50), displacements=(0.4, 0.3))
    report = build_progressive_report([sabotaged], depth_one_reference=1.90)

    assert report["verdict"] == "degenerate_early_sabotage"
    assert report["supports_training"] is False
    validate_progressive_report(report)

    # The same curve against an honest reference is not sabotage.
    clean = build_progressive_report([sabotaged], depth_one_reference=3.10)
    assert clean["verdict"] == "real_progress"


def test_improvement_that_tracks_answer_length_is_confounded():
    """'Not merely imitate long solutions', measured rather than asserted."""
    trajectories = [
        _trajectory(
            losses=(2.0, 2.0 - 0.1 * index),
            displacements=(0.3, 0.3),
            answer_tokens=4 + 4 * index,
        )
        for index in range(1, 6)
    ]
    report = build_progressive_report(trajectories)

    assert report["verdict"] == "degenerate_length_confound"
    assert report["length_correlation"] is not None
    assert abs(report["length_correlation"]) > 0.9
    validate_progressive_report(report)


def test_steps_that_cost_nothing_to_remove_are_causally_idle():
    trajectories = [_trajectory(losses=(2.0, 1.5), displacements=(0.3, 0.3))]
    idle = build_progressive_report(trajectories, necessity={1: 0.0, 2: 1e-9})

    assert idle["verdict"] == "causally_idle"
    assert idle["idle_steps"] == [1, 2]
    validate_progressive_report(idle)

    # Removing step 2 costs 0.4 nats => that step was doing work.
    working = build_progressive_report(trajectories, necessity={1: 0.0, 2: -0.4})
    assert working["verdict"] == "real_progress"
    assert working["idle_steps"] == [1]


def test_no_improvement_is_reported_as_vacuous_not_as_progress():
    flat = _trajectory(losses=(2.0, 2.0), displacements=(0.3, 0.3))
    report = build_progressive_report([flat])
    assert report["verdict"] == "vacuous_no_improvement"
    assert report["supports_training"] is False


def test_gradient_health_classifies_the_unrolled_chain():
    trajectories = [_trajectory(losses=(2.0, 1.5), displacements=(0.3, 0.3))]
    healthy = build_progressive_report(trajectories, gradient_norms={1: 0.9, 2: 1.1})
    assert healthy["gradient_health"] == "healthy"

    vanishing = build_progressive_report(trajectories, gradient_norms={1: 1e-9, 2: 1.0})
    assert vanishing["gradient_health"] == "ill_conditioned"
    assert vanishing["gradient_ratio"] > 100.0

    dead = build_progressive_report(trajectories, gradient_norms={1: 0.0, 2: 1.0})
    assert dead["gradient_health"] == "dead_gradient"


def test_report_refuses_input_it_cannot_reason_about():
    with pytest.raises(ProgressiveObjectiveError, match="at least one"):
        build_progressive_report([])
    with pytest.raises(ProgressiveObjectiveError, match="two probed steps"):
        build_progressive_report([_trajectory(losses=(2.0,), displacements=(0.3,))])
    with pytest.raises(ProgressiveObjectiveError):
        build_progressive_report(
            [_trajectory(losses=(2.0, 1.0), displacements=(0.3, 0.3))],
            displacement_floor=float("nan"),
        )


def test_report_commitment_and_field_set_are_pinned():
    report = build_progressive_report([_trajectory(losses=(2.0, 1.5), displacements=(0.3, 0.3))])
    assert report["schema"] == PROGRESSIVE_OBJECTIVE_SCHEMA
    validate_progressive_report(report)

    tampered = dict(report)
    tampered["mean_improvement"] = 99.0
    with pytest.raises(ProgressiveObjectiveError, match="commitment"):
        validate_progressive_report(tampered)

    extra = {**report, "unexpected": 1}
    with pytest.raises(ProgressiveObjectiveError, match="fields"):
        validate_progressive_report(extra)


# ── The displacement term: it must actually push ────────────────────────


def test_displacement_penalty_is_silent_above_the_floor_and_fires_below():
    moving = [
        mx.zeros((1, 4, 8)),
        mx.ones((1, 4, 8)),
    ]
    penalty, measured = displacement_floor_penalty(moving, floor=0.01)
    assert float(penalty) == 0.0
    assert measured and measured[0] > 0.01

    base = mx.random.normal((1, 4, 8), key=mx.random.key(3))
    still = [base, base + 1e-9 * mx.ones((1, 4, 8))]
    penalty, measured = displacement_floor_penalty(still, floor=0.01)
    assert float(penalty) > 0.0
    assert measured[0] < 0.01


def test_displacement_penalty_gradient_does_not_vanish_at_collapse():
    """The reason the hinge is linear.

    v4 replaced a quadratic diversity penalty precisely because its gradient
    died in the regime that needed pressure. The collapse hinge must not
    repeat that: the gradient at a fully collapsed state has to be at least as
    strong as the gradient near the floor, or the optimizer will settle into
    collapse and never feel it.
    """
    base = mx.random.normal((1, 4, 8), key=mx.random.key(11))

    def penalty_at(scale: float) -> float:
        def loss(delta):
            return displacement_floor_penalty([base, base + delta], floor=0.05)[0]

        direction = scale * mx.random.normal((1, 4, 8), key=mx.random.key(13))
        gradient = mx.grad(loss)(direction)
        mx.eval(gradient)
        return float(mx.linalg.norm(mx.reshape(gradient, (-1,))))

    deep_collapse = penalty_at(1e-6)
    near_floor = penalty_at(1e-2)
    assert deep_collapse > 0.0
    assert deep_collapse >= 0.5 * near_floor


# ── Real-model measurement ──────────────────────────────────────────────


def test_trajectory_is_measured_on_a_real_recurrent_unroll(tiny_model):
    trajectory = measure_progressive_trajectory(tiny_model, PROMPT, ANSWER, spec=_spec(), depth=3)
    assert trajectory.probe_steps == (1, 2, 3)
    assert len(trajectory.step_losses) == 3
    assert len(trajectory.displacements) == 3
    assert all(math.isfinite(value) for value in trajectory.step_losses)
    # A genuine untrained operator moves the state; this is the measurement
    # that the collapse detector is calibrated against.
    assert trajectory.min_displacement > 0.0
    assert trajectory.answer_token_count == len(ANSWER)


def test_measurement_refuses_a_multi_branch_spec(tiny_model):
    wide = _spec(branch_roles=("constructive_solution", "counterexample_search"))
    with pytest.raises(ProgressiveObjectiveError, match="single-branch"):
        measure_progressive_trajectory(tiny_model, PROMPT, ANSWER, spec=wide, depth=2)


def test_branch_complete_measurement_preserves_the_exchange_coupled_graph(tiny_model):
    wide = _spec(
        branch_roles=("constructive_solution", "counterexample_search"),
        exchange_interval=1,
    )
    measured = measure_progressive_trajectories(
        tiny_model,
        PROMPT,
        ANSWER,
        spec=wide,
        depth=3,
        probe_steps=(1, 2, 3),
    )
    receipt = measured.receipt()

    assert isinstance(measured, ProgressiveTrajectorySet)
    assert receipt["schema"] == PROGRESSIVE_TRAJECTORY_SET_SCHEMA
    assert receipt["branch_roles"] == list(wide.branch_roles)
    assert receipt["branch_count"] == 2
    assert [row["branch_index"] for row in receipt["trajectories"]] == [0, 1]
    assert all(
        len(trajectory.step_losses) == 3 and trajectory.min_displacement > 0.0
        for trajectory in measured.trajectories
    )
    assert validate_progressive_trajectory_set(receipt) == receipt


def test_branch_complete_receipt_replays_child_summaries(tiny_model):
    measured = measure_progressive_trajectories(
        tiny_model,
        PROMPT,
        ANSWER,
        spec=_spec(branch_roles=("constructive_solution", "counterexample_search")),
        depth=2,
    )
    attacked = copy.deepcopy(measured.receipt())
    attacked["trajectories"][0]["improvement"] += 1.0
    payload = {key: value for key, value in attacked.items() if key != "receipt_sha256"}
    attacked["receipt_sha256"] = canonical_sha256(payload)

    with pytest.raises(ProgressiveObjectiveError, match="summary|replay"):
        validate_progressive_trajectory_set(attacked)


def test_step_necessity_lesions_a_real_step(tiny_model):
    deltas = step_necessity(
        tiny_model,
        PROMPT,
        ANSWER,
        spec=_spec(),
        depth=3,
        lesion_steps=(1, 3),
    )
    assert set(deltas) == {1, 3}
    assert all(math.isfinite(value) for value in deltas.values())
    # Removing a real window pass from an untrained model must change the
    # answer distribution; a zero here would mean the lesion did nothing.
    assert any(abs(value) > 1e-6 for value in deltas.values())


def test_state_gradients_reach_every_probed_depth(tiny_model):
    norms = state_gradient_norms(
        tiny_model, PROMPT, ANSWER, spec=_spec(), depth=3, probe_steps=(1, 2, 3)
    )
    assert set(norms) == {1, 2, 3}
    assert all(math.isfinite(value) and value > 0.0 for value in norms.values())


def test_progressive_loss_is_differentiable_and_reports_motion(tiny_model):
    loss, telemetry = progressive_objective_loss(tiny_model, PROMPT, ANSWER, spec=_spec(), depth=3)
    mx.eval(loss)
    assert math.isfinite(float(loss))
    assert telemetry["schema"] == PROGRESSIVE_OBJECTIVE_SCHEMA
    assert telemetry["depth"] == 3
    assert len(telemetry["step_losses"]) == 3
    assert len(telemetry["displacements"]) == 3
    assert telemetry["min_displacement"] > 0.0
    assert telemetry["collapse_pressure_active"] is False
    assert telemetry["displacement_floor"] == DEFAULT_DISPLACEMENT_FLOOR

    from mlx.utils import tree_flatten

    def objective(parameters):
        tiny_model.update(parameters)
        value, _ = progressive_objective_loss(tiny_model, PROMPT, ANSWER, spec=_spec(), depth=2)
        return value

    gradients = mx.grad(objective)(tiny_model.parameters())
    mx.eval(gradients)
    norms = [
        float(mx.linalg.norm(mx.reshape(value, (-1,))))
        for _name, value in tree_flatten(gradients)
        if isinstance(value, mx.array)
    ]
    # The objective must reach the weights: an objective whose gradient is
    # everywhere zero trains nothing regardless of what its curve does.
    assert norms, "no parameter gradients were produced"
    assert all(math.isfinite(norm) for norm in norms)
    assert any(norm > 0.0 for norm in norms)


def test_progressive_loss_raises_collapse_pressure_when_the_floor_is_high(
    tiny_model,
):
    """With the floor above the operator's real motion, the term must engage.

    This is the training-time counterpart of the collapse detector: the
    telemetry has to announce that the collapse term is carrying load, so a
    campaign can see the pressure in every step record instead of discovering
    it afterwards.
    """
    _loss, telemetry = progressive_objective_loss(
        tiny_model,
        PROMPT,
        ANSWER,
        spec=_spec(),
        depth=2,
        displacement_floor=0.99,
    )
    assert telemetry["collapse_pressure_active"] is True
    assert telemetry["displacement_penalty"] > 0.0


# ── The measured collapse landscape (SPARK-061's evidence leg) ───────────


def test_measured_sweep_shows_collapse_is_a_local_basin_not_the_optimum():
    """The finding that refined the inherited claim, pinned as a regression.

    v4's docstring says collapse is the cheapest solution. On the untrained
    1.5B at depth 4 that is not what happens: the GLOBAL minimum sits at the
    honest end. What is real is a local basin — moving out of low motion costs
    loss twice before it pays — and a basin is what actually traps an
    optimizer. The distinction matters because the two claims call for
    different fixes, and only the measured one is falsifiable.
    """
    solved = solve_collapse_barrier(MEASURED_1P5B_SWEEP)

    assert solved["collapse_is_global_optimum"] is False
    assert solved["collapse_is_local_basin"] is True
    assert [(row["from_alpha"], row["to_alpha"]) for row in solved["barriers"]] == [
        (0.01, 0.05),
        (0.05, 0.1),
    ]
    assert solved["barrier_total"] == pytest.approx(0.190169, abs=1e-6)


def test_the_detector_floor_is_too_low_to_price_the_measured_basin():
    """Why the module carries two constants instead of one.

    At the basin the measured displacement is 0.0117, ABOVE the 0.01 detector
    floor, so a penalty built on the detector floor is exactly zero precisely
    where the pressure is needed. This test exists so nobody 'simplifies' the
    two constants back into one.
    """
    basin_displacement = 0.011747
    assert basin_displacement > DEFAULT_DISPLACEMENT_FLOOR
    penalty = max(DEFAULT_DISPLACEMENT_FLOOR - basin_displacement, 0.0)
    assert penalty == 0.0

    solution = solve_collapse_barrier(MEASURED_1P5B_SWEEP)["solution"]
    assert solution["floor"] > basin_displacement


def test_solved_constants_actually_remove_the_basin():
    """The solver's output is checked by replaying it, not trusted."""
    solved = solve_collapse_barrier(MEASURED_1P5B_SWEEP)
    assert solved["solved"] is True
    weight = solved["solution"]["weight"]
    floor = solved["solution"]["floor"]

    penalized = [
        loss + weight * max(floor - displacement, 0.0)
        for _alpha, loss, displacement in MEASURED_1P5B_SWEEP
    ]
    assert all(
        later < earlier for earlier, later in zip(penalized[:-1], penalized[1:], strict=True)
    ), f"solved constants left a basin: {penalized}"

    # And the recorded reference constants are the ones the solver returns.
    assert MEASURED_1P5B_COLLAPSE_SOLUTION["weight"] == weight
    assert MEASURED_1P5B_COLLAPSE_SOLUTION["floor"] == pytest.approx(floor, abs=1e-6)


def test_barrier_solver_refuses_input_it_cannot_analyze():
    with pytest.raises(ProgressiveObjectiveError, match="three measured"):
        solve_collapse_barrier(MEASURED_1P5B_SWEEP[:2])
    with pytest.raises(ProgressiveObjectiveError, match="sorted"):
        solve_collapse_barrier(tuple(reversed(MEASURED_1P5B_SWEEP)))


def test_a_landscape_with_no_barrier_needs_no_pressure():
    """A monotonically-decreasing base objective is already safe."""
    clean = ((0.01, 3.5, 0.01), (0.1, 3.2, 0.09), (0.5, 3.0, 0.18))
    solved = solve_collapse_barrier(clean)
    assert solved["collapse_is_local_basin"] is False
    assert solved["barriers"] == []
    assert solved["solved"] is True


def test_a_forged_summary_cannot_outvote_its_own_trajectory_rows():
    """Adversarial self-review finding: the aggregates replayed, the atoms did not.

    The first version of this validator checked only summary fields, so a
    forger who edited the trajectories AND the summary consistently, then
    resealed, passed every aggregate check while the rows underneath said
    "collapsed". A commitment proves nobody altered the bytes after signing;
    it proves nothing about whether the signer's arithmetic was honest.
    """
    collapsed = _trajectory(losses=(2.4, 1.2), displacements=(1e-7, 1e-7))
    report = build_progressive_report([collapsed])

    forged = {key: value for key, value in report.items() if key != "receipt_sha256"}
    forged["min_displacement"] = 0.5
    forged["mean_displacement"] = 0.5
    forged["verdict"] = "real_progress"
    forged["supports_training"] = True
    forged["receipt_sha256"] = canonical_sha256(forged)

    with pytest.raises(ProgressiveObjectiveError, match="does not replay"):
        validate_progressive_report(forged)


def test_an_unmeasured_necessity_step_still_round_trips():
    """A recorded None is an unmeasured value, not an absent row."""
    trajectories = [_trajectory(losses=(2.0, 1.5), displacements=(0.3, 0.3))]
    report = build_progressive_report(trajectories, necessity={1: float("nan"), 2: -0.4})
    assert report["necessity"][0]["delta"] is None
    validate_progressive_report(report)
