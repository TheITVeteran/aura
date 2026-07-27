"""SPARK-065: findings come out of measurements, not out of a caller's opinion.

The depth observations are derived from real `trajectory_dynamics` reports --
produced here by running a real recurrent loop, not by writing dicts -- and the
operator observations from real `value_of_computation` action transitions.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.epistemic_state import OperationKind  # noqa: E402
from core.brain.llm.latent_cortex.value_of_computation import (  # noqa: E402
    ACTION_TRANSITION_SCHEMA,
)
from core.learning.architecture_meta_controller import (  # noqa: E402
    ArchitectureControlError,
    architecture_findings,
    propose_architecture_change,
)
from core.learning.architecture_observations import (  # noqa: E402
    ArchitectureObservationError,
    depth_observations,
    observe_architecture,
    operator_observations,
)
from core.learning.intrinsic_recurrence import (  # noqa: E402
    INTRINSIC_RECURRENCE_SCHEMA,
    RecurrentDepthPlan,
    recurrent_hidden_states,
    trajectory_dynamics,
)

# --- real dynamics reports from a real loop ---------------------------------


def _real_dynamics(iterations: int, *, renormalize: bool = True) -> dict:
    mx.random.seed(3)
    args = ModelArgs(
        model_type="qwen2", hidden_size=32, num_hidden_layers=6,
        intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=64, num_key_value_heads=2, max_position_embeddings=128,
        rope_theta=10000.0,
    )
    model = Model(args)
    model.freeze()
    mx.eval(model.parameters())
    plan = RecurrentDepthPlan(
        prelude_end=1, coda_start=4, iterations=iterations, renormalize=renormalize
    )
    _, trajectory = recurrent_hidden_states(model, mx.array([[1, 2, 3]]), plan)
    return trajectory_dynamics(trajectory)


def test_a_real_loop_produces_a_usable_dynamics_report():
    report = _real_dynamics(4)
    assert report["schema"] == INTRINSIC_RECURRENCE_SCHEMA
    assert report["measurable"] is True
    assert report["diverged"] is False
    assert isinstance(report["final_delta"], float)


def test_depth_observations_are_derived_from_real_reports():
    reports = [_real_dynamics(4) for _ in range(80)]
    observations = depth_observations(reports)
    assert [row["failure_mode"] for row in observations] == [
        "depth_saturation",
        "depth_starvation",
    ]
    # The episode count is the number of reports, not an argument.
    assert all(row["episodes"] == 80 for row in observations)
    assert all(0.0 <= row["statistic"] <= 1.0 for row in observations)


def test_both_depth_failures_are_always_reported_together():
    # They are opposite failures of one knob. Emitting only the one over
    # threshold would hide a depth change that fixes one by causing the other.
    reports = [_real_dynamics(2) for _ in range(70)]
    modes = {row["failure_mode"] for row in depth_observations(reports)}
    assert modes == {"depth_saturation", "depth_starvation"}


def _synthetic_dynamics(final_delta: float, *, at_fixed_point: bool) -> dict:
    return {
        "schema": INTRINSIC_RECURRENCE_SCHEMA,
        "iterations": 4,
        "measurable": True,
        "diverged": False,
        "relative_deltas": [0.4, 0.2, final_delta],
        "final_delta": final_delta,
        "contracting": True,
        "at_fixed_point": at_fixed_point,
        "oscillating": False,
    }


def test_a_loop_that_stopped_moving_reads_as_saturation():
    reports = [_synthetic_dynamics(0.001, at_fixed_point=True) for _ in range(100)]
    saturation = depth_observations(reports)[0]
    assert saturation["statistic"] == 1.0
    assert saturation["statistic"] > saturation["threshold"]


def test_a_loop_still_moving_hard_reads_as_starvation():
    reports = [_synthetic_dynamics(0.4, at_fixed_point=False) for _ in range(100)]
    starvation = depth_observations(reports)[1]
    assert starvation["statistic"] == 1.0
    assert depth_observations(reports)[0]["statistic"] == 0.0


def test_a_diverged_loop_stops_the_derivation_instead_of_averaging_into_it():
    reports = [_synthetic_dynamics(0.05, at_fixed_point=False) for _ in range(20)]
    reports.append(
        {
            "schema": INTRINSIC_RECURRENCE_SCHEMA,
            "iterations": 4,
            "measurable": False,
            "diverged": True,
            "reason": "non-finite hidden state: the loop overflowed",
        }
    )
    with pytest.raises(ArchitectureObservationError) as excinfo:
        depth_observations(reports)
    assert "diverged" in str(excinfo.value)


def test_an_unmeasurable_report_is_refused_not_dropped():
    # Dropping it would shrink the denominator and inflate every rate.
    reports = [_synthetic_dynamics(0.05, at_fixed_point=False) for _ in range(20)]
    reports.append(
        {
            "schema": INTRINSIC_RECURRENCE_SCHEMA,
            "iterations": 1,
            "measurable": False,
            "diverged": False,
            "reason": "need at least two iterations to measure motion",
        }
    )
    with pytest.raises(ArchitectureObservationError) as excinfo:
        depth_observations(reports)
    assert "not_measurable" in str(excinfo.value)


def test_a_foreign_report_is_not_a_dynamics_report():
    with pytest.raises(ArchitectureObservationError):
        depth_observations([{"schema": "aura.something.v1", "final_delta": 0.1}])


# --- real action transitions ------------------------------------------------


def _transition(action: OperationKind) -> dict:
    return {
        "schema": ACTION_TRANSITION_SCHEMA,
        "action": action.value,
        "step_index": 0,
        "bucket": "b",
        "snapshot_sha256": "0" * 64,
        "decision_sha256": "1" * 64,
        "before": {},
        "after": {},
        "checked": [],
        "metrics": {},
    }


def _uniform_transitions(per_action: int = 8) -> list[dict]:
    return [
        _transition(action)
        for action in OperationKind
        for _ in range(per_action)
    ]


def test_a_uniform_policy_has_no_router_collapse():
    rows = {row["failure_mode"]: row for row in operator_observations(_uniform_transitions())}
    assert rows["router_collapse"]["statistic"] == 0.0
    assert rows["dead_expert"]["statistic"] == 0.0
    assert rows["overloaded_expert"]["statistic"] < 0.2


def test_a_policy_that_always_picks_one_operator_is_a_collapsed_router():
    only = list(OperationKind)[0]
    rows = {
        row["failure_mode"]: row
        for row in operator_observations([_transition(only) for _ in range(200)])
    }
    assert rows["router_collapse"]["statistic"] == 1.0
    assert rows["overloaded_expert"]["statistic"] == 1.0
    # Every other vocabulary entry is dead.
    assert rows["dead_expert"]["statistic"] > 0.9


def test_never_selected_operators_are_dead_experts():
    actions = list(OperationKind)
    half = actions[: len(actions) // 2]
    transitions = [_transition(action) for action in half for _ in range(10)]
    rows = {row["failure_mode"]: row for row in operator_observations(transitions)}
    assert rows["dead_expert"]["statistic"] == pytest.approx(
        (len(actions) - len(half)) / len(actions), abs=1e-9
    )


def test_the_transition_count_is_derived_from_the_evidence():
    transitions = _uniform_transitions(per_action=3)
    rows = operator_observations(transitions)
    assert all(row["episodes"] == len(transitions) for row in rows)


def test_a_transition_that_is_not_an_action_transition_is_refused():
    with pytest.raises(ArchitectureObservationError):
        operator_observations([{"schema": "aura.other.v1", "action": "ANSWER"}])


def test_an_unknown_action_is_refused_rather_than_bucketed():
    bad = _transition(list(OperationKind)[0])
    bad["action"] = "TELEPATHY"
    with pytest.raises(ArchitectureObservationError) as excinfo:
        operator_observations([bad])
    assert "action_unknown" in str(excinfo.value)


# --- the produced observations feed the controller --------------------------


def test_produced_observations_drive_a_real_proposal():
    reports = [_synthetic_dynamics(0.001, at_fixed_point=True) for _ in range(100)]
    findings = architecture_findings(
        observe_architecture(
            dynamics_reports=reports,
            action_transitions=_uniform_transitions(),
        )
    )
    assert "depth_saturation" in {row["failure_mode"] for row in findings["findings"]}
    proposal = propose_architecture_change(
        findings=findings,
        failure_mode="depth_saturation",
        current_value=8.0,
        proposed_value=6.0,
        proposer_identity="aura.architecture_proposer",
    )
    assert proposal["knob"] == "recurrence_max_depth"
    assert proposal["finding_evidence_sha256"] == next(
        row["evidence_sha256"]
        for row in findings["findings"]
        if row["failure_mode"] == "depth_saturation"
    )


def test_an_unmeasured_surface_is_named_rather_than_scored_zero():
    # Nothing measures a routing decision against a known-correct one, so
    # `router_misroute` is absent. It must show up as unmeasured, not as a
    # healthy zero.
    findings = architecture_findings(
        observe_architecture(
            dynamics_reports=[
                _synthetic_dynamics(0.05, at_fixed_point=False) for _ in range(100)
            ],
            action_transitions=_uniform_transitions(),
        )
    )
    produced = {row["failure_mode"] for row in findings["findings"]} | {
        row["failure_mode"] for row in findings["clean"]
    }
    assert "router_misroute" not in produced


def test_a_healthy_run_produces_findings_that_authorize_nothing():
    findings = architecture_findings(
        observe_architecture(
            dynamics_reports=[
                _synthetic_dynamics(0.05, at_fixed_point=False) for _ in range(100)
            ],
            action_transitions=_uniform_transitions(),
        )
    )
    assert findings["findings"] == []
    with pytest.raises(ArchitectureControlError) as excinfo:
        propose_architecture_change(
            findings=findings,
            failure_mode="depth_saturation",
            current_value=8.0,
            proposed_value=6.0,
            proposer_identity="aura.architecture_proposer",
        )
    assert "without_finding" in str(excinfo.value)
