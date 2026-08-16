"""CP126 3f561b15: a verdict nobody can re-derive is not a result.

A Claim carried an experiment name, a statement, a tier, evidence and a
wall-clock grading time. Nothing recorded which tasks ran, which weights they
ran against, which schedule, which verifier, or which environment — and the
runners take opaque callbacks, so none of it could be recovered afterwards. The
tier looked identical whether the experiment was reproducible or not.

Provenance is now part of the claim, and a claim missing any of it is capped at
CONJECTURE. REFUTED survives the cap: a failure that cannot be reproduced is
still a failure observed, and downgrading it would turn missing paperwork into
good news.
"""
from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.experiments import (
    CONJECTURE,
    REFUTED,
    SUPPORTED,
    ExperimentProvenance,
    PairedObservation,
    experiments_implementation_sha256,
    grade_paired_treatment_vs_control,
)

_COMPLETE = ExperimentProvenance(
    task_manifest_sha256="a" * 64,
    checkpoint_fingerprint="b" * 64,
    schedule_sha256="c" * 64,
    verifier_version="verifier-1.0.0",
    environment_sha256="d" * 64,
)


def _observations(count: int = 40, *, treatment_wins: bool = True):
    return {
        "math": [
            PairedObservation(
                task_id=f"task-{index}",
                family="math",
                treatment_success=treatment_wins or index % 5 == 0,
                control_success=not treatment_wins or index % 5 == 0,
                treatment_layer_apps=100,
                control_layer_apps=100,
            )
            for index in range(count)
        ]
    }


class TestModuleDigest:
    def test_the_grader_measures_its_own_source(self):
        digest = experiments_implementation_sha256()
        assert len(digest) == 64
        assert digest == experiments_implementation_sha256()

    def test_the_digest_travels_with_every_claim(self):
        claim = grade_paired_treatment_vs_control(
            "x", "s", _observations(), provenance=_COMPLETE
        )
        assert claim.provenance["implementation_sha256"] == (
            experiments_implementation_sha256()
        )


class TestProvenanceGaps:
    def test_a_complete_record_has_no_gaps(self):
        assert _COMPLETE.gaps() == ()

    @pytest.mark.parametrize(
        "field",
        [
            "task_manifest_sha256",
            "checkpoint_fingerprint",
            "schedule_sha256",
            "verifier_version",
            "environment_sha256",
        ],
    )
    def test_each_missing_field_is_named(self, field):
        partial = ExperimentProvenance(
            **{**{f: getattr(_COMPLETE, f) for f in _COMPLETE.__slots__}, field: ""}
        )
        assert partial.gaps() == (field,)

    def test_whitespace_is_not_a_value(self):
        partial = ExperimentProvenance(
            task_manifest_sha256="   ",
            checkpoint_fingerprint="b" * 64,
            schedule_sha256="c" * 64,
            verifier_version="verifier-1.0.0",
            environment_sha256="d" * 64,
        )
        assert partial.gaps() == ("task_manifest_sha256",)


class TestTierCap:
    def test_a_reproducible_win_keeps_its_tier(self):
        """One family reaches SUPPORTED; PROVEN needs breadth this fixture
        does not have. Either way the cap leaves the tier alone."""
        claim = grade_paired_treatment_vs_control(
            "x", "s", _observations(), provenance=_COMPLETE
        )
        assert claim.tier == SUPPORTED
        assert claim.evidence["reproducible"] is True
        assert claim.evidence["provenance_gaps"] == []

    def test_the_same_evidence_without_provenance_is_conjecture(self):
        """Identical numbers, unreproducible verdict."""
        claim = grade_paired_treatment_vs_control("x", "s", _observations())
        assert claim.tier == CONJECTURE
        assert claim.evidence["reproducible"] is False
        assert set(claim.evidence["provenance_gaps"]) == {
            "task_manifest_sha256",
            "checkpoint_fingerprint",
            "schedule_sha256",
            "verifier_version",
            "environment_sha256",
        }

    def test_one_missing_field_is_enough_to_cap(self):
        partial = ExperimentProvenance(
            task_manifest_sha256="a" * 64,
            checkpoint_fingerprint="b" * 64,
            schedule_sha256="c" * 64,
            verifier_version="verifier-1.0.0",
            environment_sha256="",
        )
        claim = grade_paired_treatment_vs_control(
            "x", "s", _observations(), provenance=partial
        )
        assert claim.tier == CONJECTURE
        assert claim.evidence["provenance_gaps"] == ["environment_sha256"]

    def test_a_refutation_is_not_softened_by_missing_paperwork(self):
        """Downgrading a refutation would make absent provenance good news."""
        claim = grade_paired_treatment_vs_control(
            "x", "s", _observations(treatment_wins=False)
        )
        assert claim.tier == REFUTED
        assert claim.evidence["reproducible"] is False

    def test_the_serialized_claim_carries_the_record(self):
        payload = grade_paired_treatment_vs_control(
            "x", "s", _observations(), provenance=_COMPLETE
        ).to_dict()
        assert payload["provenance"]["task_manifest_sha256"] == "a" * 64
        assert payload["provenance"]["verifier_version"] == "verifier-1.0.0"
