"""SPARK-067: a field that was copied is not a subsystem that was coupled.

The matrix exists to refuse three things that look like coupling in a receipt:
metadata passed across a boundary, a one-way read, and an effect that survives
cutting the seam it supposedly flows through.
"""

from __future__ import annotations

import hashlib

import pytest

from core.brain.llm.latent_cortex.coupling_matrix import (
    BEHAVIORAL,
    COUPLED,
    COUPLED_SUBSYSTEMS,
    FORWARD,
    METADATA,
    REFUSED,
    REVERSE,
    CouplingMatrixError,
    coupling_effect,
    coupling_matrix,
    coupling_seam,
    lesion_result,
)


def _d(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _effect(direction: str, subsystem: str, *, kind: str = BEHAVIORAL, moved: bool = True, n: int = 256):
    return coupling_effect(
        direction=direction,
        kind=kind,
        baseline_statistic=0.40,
        observed_statistic=0.62 if moved else 0.40,
        observations=n,
        evidence_sha256=_d(f"{subsystem}-{direction}"),
    )


def _lesion(subsystem: str, *, removed: bool = True, n: int = 256):
    return lesion_result(
        baseline_statistic=0.40,
        intact_statistic=0.62,
        lesioned_statistic=0.41 if removed else 0.61,
        observations=n,
        evidence_sha256=_d(f"{subsystem}-lesion"),
    )


def _seam(subsystem: str, **overrides):
    return coupling_seam(
        subsystem=subsystem,
        forward=overrides.get("forward", _effect(FORWARD, subsystem)),
        reverse=overrides.get("reverse", _effect(REVERSE, subsystem)),
        lesion=overrides.get("lesion", _lesion(subsystem)),
    )


def _matrix(**per_subsystem):
    return coupling_matrix(
        [_seam(name, **per_subsystem.get(name, {})) for name in COUPLED_SUBSYSTEMS]
    )


# --- the healthy case -------------------------------------------------------


def test_a_fully_coupled_organism_is_reported_coupled():
    matrix = _matrix()
    assert matrix["verdict"] == COUPLED
    assert matrix["uncoupled_subsystems"] == []
    assert len(matrix["seams"]) == len(COUPLED_SUBSYSTEMS)


# --- metadata is not coupling -----------------------------------------------


def test_a_field_copied_across_the_boundary_is_refused_as_metadata():
    matrix = _matrix(memory={"forward": _effect(FORWARD, "memory", kind=METADATA)})
    assert matrix["verdict"] == REFUSED
    assert matrix["uncoupled_subsystems"] == ["memory"]
    seam = next(row for row in matrix["seams"] if row["subsystem"] == "memory")
    assert seam["refusals"][0] == {
        "reason": "metadata_only_coupling",
        "direction": "forward",
        "kind": METADATA,
    }


def test_metadata_in_either_direction_is_enough_to_refuse():
    matrix = _matrix(goals={"reverse": _effect(REVERSE, "goals", kind=METADATA)})
    seam = next(row for row in matrix["seams"] if row["subsystem"] == "goals")
    assert seam["verdict"] == REFUSED
    assert seam["refusals"][0]["direction"] == "reverse"


def test_a_metadata_seam_is_still_recordable():
    # Recording it is legal; counting it is not. The seam builds fine and the
    # matrix is what refuses to call it coupling.
    seam = _seam("tools", forward=_effect(FORWARD, "tools", kind=METADATA))
    assert seam["forward"]["kind"] == METADATA


# --- both directions --------------------------------------------------------


def test_a_direction_that_measured_nothing_is_refused():
    matrix = _matrix(
        affect_body={"reverse": _effect(REVERSE, "affect_body", moved=False)}
    )
    seam = next(row for row in matrix["seams"] if row["subsystem"] == "affect_body")
    assert seam["verdict"] == REFUSED
    assert any(row["reason"] == "no_measured_effect" for row in seam["refusals"])


def test_a_direction_label_that_disagrees_with_its_slot_is_refused():
    with pytest.raises(CouplingMatrixError) as excinfo:
        coupling_seam(
            subsystem="tools",
            forward=_effect(REVERSE, "tools"),
            reverse=_effect(REVERSE, "tools"),
            lesion=_lesion("tools"),
        )
    assert "direction_mismatch" in str(excinfo.value)


def test_an_underpowered_direction_is_refused():
    matrix = _matrix(learning={"forward": _effect(FORWARD, "learning", n=4)})
    seam = next(row for row in matrix["seams"] if row["subsystem"] == "learning")
    assert any(
        row["reason"] == "insufficient_observations" for row in seam["refusals"]
    )


# --- the lesion has to remove something -------------------------------------


def test_an_effect_that_survives_the_lesion_is_not_flowing_through_the_seam():
    matrix = _matrix(agency_will={"lesion": _lesion("agency_will", removed=False)})
    seam = next(row for row in matrix["seams"] if row["subsystem"] == "agency_will")
    assert seam["verdict"] == REFUSED
    refusal = next(
        row for row in seam["refusals"] if row["reason"] == "lesion_did_not_remove_effect"
    )
    assert refusal["required_removal_fraction"] == 0.5


def test_a_lesion_with_no_intact_effect_to_remove_is_refused():
    flat = lesion_result(
        baseline_statistic=0.40,
        intact_statistic=0.40,
        lesioned_statistic=0.40,
        observations=256,
        evidence_sha256=_d("flat"),
    )
    matrix = _matrix(global_workspace={"lesion": flat})
    seam = next(
        row for row in matrix["seams"] if row["subsystem"] == "global_workspace"
    )
    assert any(
        row["reason"] == "lesion_had_no_intact_effect_to_remove"
        for row in seam["refusals"]
    )


def test_a_partial_lesion_that_removes_most_of_the_effect_passes():
    partial = lesion_result(
        baseline_statistic=0.40,
        intact_statistic=0.62,
        lesioned_statistic=0.45,
        observations=256,
        evidence_sha256=_d("partial"),
    )
    matrix = _matrix(memory={"lesion": partial})
    seam = next(row for row in matrix["seams"] if row["subsystem"] == "memory")
    assert seam["verdict"] == COUPLED


# --- the subsystem set is complete by declaration ---------------------------


@pytest.mark.parametrize("dropped", COUPLED_SUBSYSTEMS)
def test_a_matrix_missing_a_subsystem_is_invalid_not_partial(dropped):
    seams = [_seam(name) for name in COUPLED_SUBSYSTEMS if name != dropped]
    with pytest.raises(CouplingMatrixError) as excinfo:
        coupling_matrix(seams)
    assert "incomplete" in str(excinfo.value)


def test_a_duplicate_seam_is_refused():
    seams = [_seam(name) for name in COUPLED_SUBSYSTEMS]
    seams.append(_seam("memory"))
    with pytest.raises(CouplingMatrixError) as excinfo:
        coupling_matrix(seams)
    assert "duplicate" in str(excinfo.value)


def test_an_unknown_subsystem_cannot_stand_in_for_a_required_one():
    with pytest.raises(CouplingMatrixError):
        coupling_seam(
            subsystem="vibes",
            forward=_effect(FORWARD, "vibes"),
            reverse=_effect(REVERSE, "vibes"),
            lesion=_lesion("vibes"),
        )


def test_an_edited_seam_breaks_its_digest():
    seams = [dict(_seam(name)) for name in COUPLED_SUBSYSTEMS]
    forward = dict(seams[0]["forward"])
    forward["observed_statistic"] = 0.99
    seams[0]["forward"] = forward
    with pytest.raises(CouplingMatrixError) as excinfo:
        coupling_matrix(seams)
    assert "seam_differs" in str(excinfo.value)


def test_every_uncoupled_subsystem_is_named_in_the_matrix_verdict():
    matrix = _matrix(
        tools={"forward": _effect(FORWARD, "tools", kind=METADATA)},
        goals={"lesion": _lesion("goals", removed=False)},
    )
    assert matrix["verdict"] == REFUSED
    assert matrix["uncoupled_subsystems"] == ["tools", "goals"]
