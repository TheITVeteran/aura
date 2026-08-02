"""A release is what it is known to survive, not what happened to be green.

"The current checkout worked during this run" and "this release survives
its declared operating envelope" are different claims, and only the second
is worth shipping on. A green suite says nothing about the 64GB profile,
long conversations, model death and reload, or upgrade and rollback.

The defining behaviour under test is REFUSAL. A requirement with no
evidence must block, never quietly count as satisfied — this codebase's
recurring defect is the absence of a check reported as a passed check, and
a release certificate is where that would do the most damage.
"""
from __future__ import annotations

import pytest

from core.runtime.release_certificate import (
    REQUIREMENTS,
    CertificateBuilder,
    RequirementStatus,
)


def _complete(commit: str = "abc123def456") -> CertificateBuilder:
    builder = CertificateBuilder(commit=commit)
    for requirement in REQUIREMENTS:
        builder.submit(requirement.key, passed=True, produced_by="harness")
    return builder


# --------------------------------------------------------------- the refusal


def test_a_certificate_with_no_evidence_refuses():
    certificate = CertificateBuilder(commit="abc123").build()
    assert not certificate.certified
    assert all(
        result.status is RequirementStatus.MISSING for result in certificate.results
    )


def test_missing_evidence_is_never_silently_satisfied():
    """Every blocking requirement must be individually accounted for."""
    certificate = CertificateBuilder(commit="abc123").build()
    blocking = {r.key for r in REQUIREMENTS if r.blocking}
    named = {r.requirement.key for r in certificate.blocking_failures}
    assert named == blocking


@pytest.mark.parametrize("omitted", [r.key for r in REQUIREMENTS if r.blocking])
def test_omitting_any_single_requirement_blocks_the_release(omitted):
    """No requirement is inferred from its neighbours passing."""
    builder = CertificateBuilder(commit="abc123")
    for requirement in REQUIREMENTS:
        if requirement.key != omitted:
            builder.submit(requirement.key, passed=True, produced_by="harness")
    certificate = builder.build()
    assert not certificate.certified
    assert omitted in {r.requirement.key for r in certificate.blocking_failures}


def test_a_fully_evidenced_build_certifies():
    """The control: certification must be reachable, or this only ever says no."""
    certificate = _complete().build()
    assert certificate.certified
    assert certificate.summary().startswith("CERTIFIED")


def test_a_failed_requirement_blocks_just_as_hard_as_a_missing_one():
    builder = _complete()
    builder.submit("memory_ceiling", passed=False, produced_by="soak")
    assert not builder.build().certified


def test_failed_and_missing_stay_distinguishable():
    """'We looked and it broke' and 'we did not look' need different responses."""
    builder = CertificateBuilder(commit="abc123")
    builder.submit("memory_ceiling", passed=False, produced_by="soak")
    statuses = {r.requirement.key: r.status for r in builder.build().results}
    assert statuses["memory_ceiling"] is RequirementStatus.FAILED
    assert statuses["chaos_injection"] is RequirementStatus.MISSING


# ---------------------------------------------------------------- staleness


def test_evidence_from_another_commit_is_stale_not_valid():
    """A shard result from forty commits ago is not evidence about this build."""
    builder = CertificateBuilder(commit="new-commit")
    for requirement in REQUIREMENTS:
        builder.submit(
            requirement.key, passed=True, commit="old-commit", produced_by="harness"
        )
    certificate = builder.build()
    assert not certificate.certified
    assert all(r.status is RequirementStatus.STALE for r in certificate.results)


def test_an_unidentifiable_build_can_never_be_certified():
    """Unknown commit means nothing can be tied to it."""
    builder = CertificateBuilder(commit="unknown")
    for requirement in REQUIREMENTS:
        builder.submit(requirement.key, passed=True, produced_by="harness")
    assert not builder.build().certified


def test_the_stale_note_names_both_commits():
    # Real 40-char hashes: the note shows the first 12 of each, so a test
    # using short stand-ins would be asserting against the truncation
    # rather than against the message.
    new_commit = "a" * 40
    old_commit = "b" * 40
    builder = CertificateBuilder(commit=new_commit)
    builder.submit("hermetic_shards", passed=True, commit=old_commit)
    result = next(
        r for r in builder.build().results if r.requirement.key == "hermetic_shards"
    )
    assert old_commit[:12] in result.note
    assert new_commit[:12] in result.note


# ------------------------------------------------------------------ waivers


def test_a_waiver_requires_a_reason_and_a_named_person():
    """An anonymous waiver is a requirement quietly deleted."""
    builder = _complete()
    with pytest.raises(ValueError, match="reason"):
        builder.waive("chaos_injection", reason="", waived_by="bryan")
    with pytest.raises(ValueError, match="named person"):
        builder.waive("chaos_injection", reason="no chaos harness yet", waived_by="")


def test_a_properly_attributed_waiver_unblocks_and_is_recorded():
    builder = CertificateBuilder(commit="abc123")
    for requirement in REQUIREMENTS:
        if requirement.key != "chaos_injection":
            builder.submit(requirement.key, passed=True, produced_by="harness")
    builder.waive("chaos_injection", reason="no chaos harness yet", waived_by="bryan")
    certificate = builder.build()
    assert certificate.certified
    waived = next(
        r for r in certificate.results if r.requirement.key == "chaos_injection"
    )
    assert waived.status is RequirementStatus.WAIVED
    assert "bryan" in waived.note


# ------------------------------------------------------------------- misuse


def test_evidence_for_an_undeclared_requirement_is_refused():
    """A certificate paddable with irrelevant evidence certifies nothing."""
    with pytest.raises(KeyError):
        CertificateBuilder(commit="abc123").submit("something_i_made_up", passed=True)


def test_a_truthy_string_is_not_a_result():
    with pytest.raises(TypeError):
        CertificateBuilder(commit="abc123").submit("hermetic_shards", passed="yes")


def test_waiving_an_undeclared_requirement_is_refused():
    with pytest.raises(KeyError):
        CertificateBuilder(commit="abc123").waive(
            "invented", reason="r", waived_by="bryan"
        )


# ------------------------------------------------------------------ the artifact


def test_the_certificate_says_why_it_refused():
    certificate = CertificateBuilder(commit="abc123").build()
    summary = certificate.summary()
    assert summary.startswith("NOT CERTIFIED")
    assert "missing" in summary


def test_the_certificate_names_the_runtime_that_built_it():
    payload = _complete().build().to_dict()
    assert payload["runtime"]["runtime_instance_id"]
    assert payload["runtime"]["runtime_profile"] == "test"


def test_the_certificate_is_serializable_and_carries_its_reasoning():
    import json

    payload = _complete().build().to_dict()
    json.dumps(payload)
    assert len(payload["requirements"]) == len(REQUIREMENTS)
    assert all("status" in entry for entry in payload["requirements"])
    assert all("produced_by" in entry for entry in payload["requirements"])


def test_the_envelope_covers_more_than_the_test_suite():
    """Raw test count was never the remaining maturity problem."""
    keys = {requirement.key for requirement in REQUIREMENTS}
    for expected in (
        "conversation_soak",
        "contention_soak",
        "memory_ceiling",
        "worker_cancellation",
        "model_death_reload",
        "install_upgrade_rollback",
        "no_blank_responses",
        "learning_evidence",
    ):
        assert expected in keys
