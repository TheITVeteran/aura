"""SPARK-065: Aura may change her own architecture, inside a narrow box.

These tests are mostly about the walls of the box: a proposal with no finding
behind it, a trial that ran in the live runtime, a self-approval, a candidate
that simply spent more compute, and a rollout that walks past a bad canary.
"""

from __future__ import annotations

import hashlib

import pytest

from core.learning.architecture_meta_controller import (
    ADMIT,
    APPROVER,
    CANARY,
    EXPANDED,
    FULL,
    HEALTHY,
    HUMAN_APPROVER,
    KNOB_BOUNDS,
    REFUSE,
    REGRESSED,
    REQUIRED_INVARIANTS,
    ROLLED_BACK,
    ArchitectureControlError,
    approve_architecture_change,
    architecture_findings,
    architecture_observation,
    candidate_trial,
    evaluate_rollout,
    invariant_result,
    propose_architecture_change,
    rollout_stage,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _observations(episodes: int = 512) -> list[dict]:
    return [
        architecture_observation(
            failure_mode="depth_saturation",
            episodes=episodes,
            statistic=0.41,
            threshold=0.20,
            evidence_sha256=_digest("depth"),
        ),
        architecture_observation(
            failure_mode="router_collapse",
            episodes=episodes,
            statistic=0.05,
            threshold=0.30,
            evidence_sha256=_digest("router"),
        ),
    ]


def _findings(episodes: int = 512) -> dict:
    return architecture_findings(_observations(episodes))


def _proposal(**overrides) -> dict:
    kwargs = {
        "findings": _findings(),
        "failure_mode": "depth_saturation",
        "current_value": 8.0,
        "proposed_value": 6.0,
        "proposer_identity": "aura.architecture_proposer",
    }
    kwargs.update(overrides)
    return propose_architecture_change(**kwargs)


def _invariants(broken: str | None = None) -> list[dict]:
    return [
        invariant_result(
            invariant=name,
            holds=name != broken,
            evidence_sha256=_digest(name),
        )
        for name in REQUIRED_INVARIANTS
    ]


def _trial(**overrides) -> dict:
    kwargs = {
        "proposal": _proposal(),
        "live_runtime_identity": "aura.live@8000",
        "candidate_runtime_identity": "aura.candidate@0",
        "incumbent_score": 0.61,
        "candidate_score": 0.68,
        "incumbent_compute_units": 10_000,
        "candidate_compute_units": 10_020,
        "episodes": 256,
        "invariants": _invariants(),
    }
    kwargs.update(overrides)
    return candidate_trial(**kwargs)


def _approval(**overrides) -> dict:
    kwargs = {
        "proposal": _proposal(),
        "trial": _trial(),
        "approver_role": APPROVER,
        "approver_identity": "aura.architecture_approver",
    }
    kwargs.update(overrides)
    return approve_architecture_change(**kwargs)


def _stages(*verdicts: str) -> list[dict]:
    fractions = {CANARY: 0.05, EXPANDED: 0.25, FULL: 1.0}
    order = [CANARY, EXPANDED, FULL]
    return [
        rollout_stage(
            stage=stage,
            traffic_fraction=fractions[stage],
            episodes=128,
            verdict=verdict,
            evidence_sha256=_digest(stage),
        )
        for stage, verdict in zip(order, verdicts, strict=False)
    ]


# --- measurement ------------------------------------------------------------


def test_a_surface_over_the_threshold_becomes_a_finding():
    report = _findings()
    assert [row["failure_mode"] for row in report["findings"]] == ["depth_saturation"]
    assert report["insufficient_evidence"] == []


def test_an_undermeasured_surface_is_not_a_finding():
    report = _findings(episodes=8)
    assert report["findings"] == []
    assert [row["reason"] for row in report["insufficient_evidence"]] == [
        "insufficient_evidence",
        "insufficient_evidence",
    ]


def test_a_clean_surface_says_so_instead_of_going_quiet():
    report = architecture_findings(
        [
            architecture_observation(
                failure_mode="dead_expert",
                episodes=512,
                statistic=0.01,
                threshold=0.10,
                evidence_sha256=_digest("expert"),
            )
        ]
    )
    assert report["findings"] == []
    assert [row["failure_mode"] for row in report["clean"]] == ["dead_expert"]
    assert report["surfaces_measured"] == ["expert"]
    assert report["surfaces_unmeasured"] == ["depth", "router"]


def test_an_undermeasured_surface_never_counts_as_measured():
    report = _findings(episodes=8)
    assert report["surfaces_measured"] == []
    assert report["surfaces_unmeasured"] == ["depth", "expert", "router"]


def test_an_unknown_failure_mode_is_refused():
    with pytest.raises(ArchitectureControlError):
        architecture_observation(
            failure_mode="vibes",
            episodes=512,
            statistic=1.0,
            threshold=0.1,
            evidence_sha256=_digest("x"),
        )


# --- bounded proposals ------------------------------------------------------


def test_a_finding_maps_to_exactly_one_knob():
    proposal = _proposal()
    assert proposal["knob"] == "recurrence_max_depth"
    assert proposal["surface"] == "depth"


def test_a_proposal_without_a_finding_is_refused():
    with pytest.raises(ArchitectureControlError) as excinfo:
        _proposal(failure_mode="router_collapse")
    assert "without_finding" in str(excinfo.value)


def test_a_proposal_outside_the_declared_bound_is_refused():
    with pytest.raises(ArchitectureControlError) as excinfo:
        _proposal(current_value=16.0, proposed_value=18.0)
    assert "out_of_bounds" in str(excinfo.value)


def test_a_proposal_that_jumps_further_than_the_step_cap_is_refused():
    with pytest.raises(ArchitectureControlError) as excinfo:
        _proposal(current_value=12.0, proposed_value=4.0)
    assert "step_too_large" in str(excinfo.value)


def test_a_no_op_proposal_is_refused():
    with pytest.raises(ArchitectureControlError):
        _proposal(current_value=8.0, proposed_value=8.0)


# --- isolated trials with complete invariants -------------------------------


def test_a_trial_inside_the_live_runtime_is_refused():
    with pytest.raises(ArchitectureControlError) as excinfo:
        _trial(
            live_runtime_identity="aura.live@8000",
            candidate_runtime_identity="aura.live@8000",
        )
    assert "not_isolated" in str(excinfo.value)


@pytest.mark.parametrize("dropped", REQUIRED_INVARIANTS)
def test_a_missing_invariant_makes_the_trial_invalid(dropped):
    with pytest.raises(ArchitectureControlError) as excinfo:
        _trial(
            invariants=[
                row for row in _invariants() if row["invariant"] != dropped
            ]
        )
    assert "invariant_set_incomplete" in str(excinfo.value)


def test_an_underpowered_trial_is_refused():
    with pytest.raises(ArchitectureControlError) as excinfo:
        _trial(episodes=4)
    assert "underpowered" in str(excinfo.value)


def test_a_candidate_that_spent_more_compute_is_marked_unequal():
    trial = _trial(candidate_compute_units=20_000)
    assert trial["equal_compute"] is False


# --- independent approval ---------------------------------------------------


def test_a_clean_trial_is_approved():
    approval = _approval()
    assert approval["decision"] == ADMIT
    assert approval["refusals"] == []


def test_the_proposer_cannot_approve_its_own_change():
    approval = _approval(approver_identity="aura.architecture_proposer")
    assert approval["decision"] == REFUSE
    assert approval["refusals"][0]["reason"] == "self_approval"


def test_a_violated_invariant_blocks_approval():
    approval = _approval(trial=_trial(invariants=_invariants(broken="bounded_depth")))
    assert approval["decision"] == REFUSE
    assert approval["refusals"][0] == {
        "reason": "invariant_violated",
        "invariants": ["bounded_depth"],
    }


def test_a_win_bought_with_extra_compute_blocks_approval():
    approval = _approval(trial=_trial(candidate_compute_units=20_000))
    assert approval["decision"] == REFUSE
    assert any(row["reason"] == "unequal_compute" for row in approval["refusals"])


def test_an_improvement_below_the_floor_blocks_approval():
    approval = _approval(trial=_trial(candidate_score=0.611))
    assert approval["decision"] == REFUSE
    assert any(
        row["reason"] == "improvement_below_floor" for row in approval["refusals"]
    )


def test_a_trial_bound_to_a_different_proposal_is_refused():
    other = _proposal(current_value=8.0, proposed_value=7.0)
    with pytest.raises(ArchitectureControlError) as excinfo:
        approve_architecture_change(
            proposal=other,
            trial=_trial(),
            approver_role=APPROVER,
            approver_identity="aura.architecture_approver",
        )
    assert "binds_other_proposal" in str(excinfo.value)


def test_a_knob_marked_requires_human_cannot_be_auto_approved(monkeypatch):
    monkeypatch.setitem(
        KNOB_BOUNDS,
        "recurrence_max_depth",
        {**KNOB_BOUNDS["recurrence_max_depth"], "requires_human": True},
    )
    proposal = _proposal()
    trial = _trial(proposal=proposal)
    automated = approve_architecture_change(
        proposal=proposal,
        trial=trial,
        approver_role=APPROVER,
        approver_identity="aura.architecture_approver",
    )
    assert automated["decision"] == REFUSE
    assert automated["refusals"][0]["reason"] == "requires_human_approver"

    by_hand = approve_architecture_change(
        proposal=proposal,
        trial=trial,
        approver_role=HUMAN_APPROVER,
        approver_identity="bryan",
    )
    assert by_hand["decision"] == ADMIT


# --- rollout ladder ---------------------------------------------------------


def test_a_clean_ladder_reaches_full():
    rollout = evaluate_rollout(
        approval=_approval(),
        stages=_stages(HEALTHY, HEALTHY, HEALTHY),
        rollback_revision="arch.rev.7",
    )
    assert rollout["outcome"] == ADMIT
    assert rollout["restored_revision"] is None


def test_a_regressed_canary_stops_the_ladder_and_names_the_way_back():
    rollout = evaluate_rollout(
        approval=_approval(),
        stages=_stages(REGRESSED, HEALTHY, HEALTHY),
        rollback_revision="arch.rev.7",
    )
    assert rollout["outcome"] == ROLLED_BACK
    assert rollout["regressed_at"] == CANARY
    assert rollout["restored_revision"] == "arch.rev.7"
    assert [row["stage"] for row in rollout["stages"]] == [CANARY]


def test_a_ladder_that_stops_short_of_full_is_not_a_rollout():
    rollout = evaluate_rollout(
        approval=_approval(),
        stages=_stages(HEALTHY, HEALTHY),
        rollback_revision="arch.rev.7",
    )
    assert rollout["outcome"] == ROLLED_BACK
    assert rollout["regressed_at"] == "incomplete_ladder"


def test_a_stage_out_of_order_is_refused():
    stages = _stages(HEALTHY, HEALTHY, HEALTHY)
    with pytest.raises(ArchitectureControlError) as excinfo:
        evaluate_rollout(
            approval=_approval(),
            stages=[stages[1], stages[0], stages[2]],
            rollback_revision="arch.rev.7",
        )
    assert "out_of_order" in str(excinfo.value)


def test_a_canary_carrying_full_traffic_is_refused():
    with pytest.raises(ArchitectureControlError) as excinfo:
        rollout_stage(
            stage=CANARY,
            traffic_fraction=1.0,
            episodes=128,
            verdict=HEALTHY,
            evidence_sha256=_digest("canary"),
        )
    assert "out_of_stage" in str(excinfo.value)


def test_a_rollout_without_an_admitted_approval_is_refused():
    refused = _approval(approver_identity="aura.architecture_proposer")
    with pytest.raises(ArchitectureControlError) as excinfo:
        evaluate_rollout(
            approval=refused,
            stages=_stages(HEALTHY, HEALTHY, HEALTHY),
            rollback_revision="arch.rev.7",
        )
    assert "without_approval" in str(excinfo.value)


def test_the_rollback_target_is_required_before_the_ladder_starts():
    with pytest.raises(ArchitectureControlError):
        evaluate_rollout(
            approval=_approval(),
            stages=_stages(HEALTHY, HEALTHY, HEALTHY),
            rollback_revision="",
        )
