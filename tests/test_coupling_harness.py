"""SPARK-067: the harness decides what kind of evidence it found.

The interesting case is the one that looks fine from the inside: a seam that
really is passing a field, whose receipt really does show it, and whose
downstream behavior is byte-identical whether the seam is open or closed. A
caller measuring that would write `behavioral` in good faith. The harness
compares outcomes and writes `metadata`.
"""

from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.coupling_harness import (
    CouplingHarnessError,
    measure_coupling_seam,
    measure_direction,
    measure_lesion,
)
from core.brain.llm.latent_cortex.coupling_matrix import (
    BEHAVIORAL,
    COUPLED,
    COUPLED_SUBSYSTEMS,
    FORWARD,
    METADATA,
    REFUSED,
    REVERSE,
    coupling_matrix,
)

_TRIALS = 64


# A tiny stand-in subsystem: a decision that may or may not consult the seam.
def _closed(index: int) -> dict:
    return {"decision": "default", "score": 0.40, "seam_field": None}


def _behavioral_open(index: int) -> dict:
    # The seam genuinely changes what gets decided on half the trials.
    if index % 2:
        return {"decision": "revised", "score": 0.62, "seam_field": index}
    return {"decision": "default", "score": 0.62, "seam_field": index}


def _metadata_open(index: int) -> dict:
    # The field is really passed and really recorded. Nothing downstream of
    # the decision can tell the difference.
    return {"decision": "default", "score": 0.40, "seam_field": index}


def _lesioned(index: int) -> dict:
    # Seam cut: back to the baseline behavior.
    return {"decision": "default", "score": 0.41, "seam_field": None}


def _metric(outcome: dict) -> float:
    return float(outcome["score"])


def _identity(outcome: dict) -> str:
    # What an observer downstream can actually see. Deliberately excludes
    # `seam_field`: a field nobody acts on is not an observable outcome.
    return str(outcome["decision"])


# --- the classification is measured, not accepted ---------------------------


def test_a_seam_that_changes_the_decision_is_behavioral():
    effect = measure_direction(
        direction=FORWARD,
        trials=_TRIALS,
        seam_closed=_closed,
        seam_open=_behavioral_open,
        outcome_metric=_metric,
        outcome_identity=_identity,
    )
    assert effect["kind"] == BEHAVIORAL
    assert effect["baseline_statistic"] == 0.40
    assert effect["observed_statistic"] == 0.62
    assert effect["observations"] == _TRIALS


def test_a_seam_that_only_moves_a_field_is_metadata_however_it_is_labelled():
    effect = measure_direction(
        direction=FORWARD,
        trials=_TRIALS,
        seam_closed=_closed,
        seam_open=_metadata_open,
        outcome_metric=_metric,
        outcome_identity=_identity,
    )
    # The field IS passed on every trial. Nothing observable changed.
    assert effect["kind"] == METADATA


def test_one_changed_trial_out_of_many_is_still_behavioral():
    def barely(index: int) -> dict:
        if index == 7:
            return {"decision": "revised", "score": 0.41, "seam_field": index}
        return _metadata_open(index)

    effect = measure_direction(
        direction=FORWARD,
        trials=_TRIALS,
        seam_closed=_closed,
        seam_open=barely,
        outcome_metric=_metric,
        outcome_identity=_identity,
    )
    assert effect["kind"] == BEHAVIORAL


def test_a_metadata_direction_is_refused_by_the_matrix():
    seam = measure_coupling_seam(
        subsystem="memory",
        trials=_TRIALS,
        forward_closed=_closed,
        forward_open=_metadata_open,
        forward_metric=_metric,
        forward_identity=_identity,
        reverse_closed=_closed,
        reverse_open=_behavioral_open,
        reverse_metric=_metric,
        reverse_identity=_identity,
        lesioned=_lesioned,
    )
    seams = [
        seam if name == "memory" else _healthy_seam(name)
        for name in COUPLED_SUBSYSTEMS
    ]
    matrix = coupling_matrix(seams)
    assert matrix["verdict"] == REFUSED
    assert matrix["uncoupled_subsystems"] == ["memory"]
    row = next(r for r in matrix["seams"] if r["subsystem"] == "memory")
    assert row["refusals"][0]["reason"] == "metadata_only_coupling"


# --- the lesion answers the other question ----------------------------------


def test_a_lesion_that_restores_the_baseline_removes_the_effect():
    lesion = measure_lesion(
        trials=_TRIALS,
        seam_closed=_closed,
        seam_open=_behavioral_open,
        seam_lesioned=_lesioned,
        outcome_metric=_metric,
        outcome_identity=_identity,
    )
    assert lesion["baseline_statistic"] == 0.40
    assert lesion["intact_statistic"] == 0.62
    assert lesion["lesioned_statistic"] == 0.41


def test_an_effect_that_survives_the_cut_is_refused_by_the_matrix():
    def ineffective_lesion(index: int) -> dict:
        # The "cut" changed nothing: the effect was not flowing through here.
        return _behavioral_open(index)

    seam = measure_coupling_seam(
        subsystem="tools",
        trials=_TRIALS,
        forward_closed=_closed,
        forward_open=_behavioral_open,
        forward_metric=_metric,
        forward_identity=_identity,
        reverse_closed=_closed,
        reverse_open=_behavioral_open,
        reverse_metric=_metric,
        reverse_identity=_identity,
        lesioned=ineffective_lesion,
    )
    matrix = coupling_matrix(
        [seam if name == "tools" else _healthy_seam(name) for name in COUPLED_SUBSYSTEMS]
    )
    row = next(r for r in matrix["seams"] if r["subsystem"] == "tools")
    assert row["verdict"] == REFUSED
    assert any(
        entry["reason"] == "lesion_did_not_remove_effect" for entry in row["refusals"]
    )


# --- a fully measured healthy seam passes -----------------------------------


def _healthy_seam(subsystem: str) -> dict:
    return measure_coupling_seam(
        subsystem=subsystem,
        trials=_TRIALS,
        forward_closed=_closed,
        forward_open=_behavioral_open,
        forward_metric=_metric,
        forward_identity=_identity,
        reverse_closed=_closed,
        reverse_open=_behavioral_open,
        reverse_metric=_metric,
        reverse_identity=_identity,
        lesioned=_lesioned,
    )


def test_nine_measured_seams_make_a_coupled_organism():
    matrix = coupling_matrix([_healthy_seam(name) for name in COUPLED_SUBSYSTEMS])
    assert matrix["verdict"] == COUPLED
    assert matrix["uncoupled_subsystems"] == []


def test_a_measured_seam_replays_identically():
    first = _healthy_seam("goals")
    second = _healthy_seam("goals")
    assert first == second


# --- refusals -----------------------------------------------------------


def test_too_few_trials_are_refused_where_they_can_still_be_fixed():
    with pytest.raises(CouplingHarnessError) as excinfo:
        measure_direction(
            direction=FORWARD,
            trials=8,
            seam_closed=_closed,
            seam_open=_behavioral_open,
            outcome_metric=_metric,
            outcome_identity=_identity,
        )
    assert "trials_below_floor" in str(excinfo.value)


def test_a_seam_that_raises_is_a_finding_not_a_weak_measurement():
    def explodes(index: int) -> dict:
        if index == 40:
            raise RuntimeError("the seam is broken")
        return _behavioral_open(index)

    with pytest.raises(CouplingHarnessError) as excinfo:
        measure_direction(
            direction=REVERSE,
            trials=_TRIALS,
            seam_closed=_closed,
            seam_open=explodes,
            outcome_metric=_metric,
            outcome_identity=_identity,
        )
    assert "trial_failed" in str(excinfo.value)


def test_a_non_finite_metric_is_refused():
    def nan_metric(outcome: dict) -> float:
        return float("nan")

    with pytest.raises(CouplingHarnessError):
        measure_direction(
            direction=FORWARD,
            trials=_TRIALS,
            seam_closed=_closed,
            seam_open=_behavioral_open,
            outcome_metric=nan_metric,
            outcome_identity=_identity,
        )


def test_an_unknown_direction_is_refused():
    with pytest.raises(CouplingHarnessError):
        measure_direction(
            direction="sideways",
            trials=_TRIALS,
            seam_closed=_closed,
            seam_open=_behavioral_open,
            outcome_metric=_metric,
            outcome_identity=_identity,
        )


def test_the_evidence_digest_distinguishes_two_different_measurements():
    behavioral = measure_direction(
        direction=FORWARD, trials=_TRIALS, seam_closed=_closed,
        seam_open=_behavioral_open, outcome_metric=_metric,
        outcome_identity=_identity,
    )
    metadata = measure_direction(
        direction=FORWARD, trials=_TRIALS, seam_closed=_closed,
        seam_open=_metadata_open, outcome_metric=_metric,
        outcome_identity=_identity,
    )
    assert behavioral["evidence_sha256"] != metadata["evidence_sha256"]
