"""CP126 contract tests for the architecture evolution governor."""
from __future__ import annotations

import hashlib

import pytest

from core.advanced_cognition.architecture_evolution import (
    ArchitectureEvolutionGovernor,
    FrozenMap,
    MutationTier,
    ProofObligation,
)


@pytest.fixture()
def repo(tmp_path):
    """A miniature repository the governor may operate on."""
    (tmp_path / "core" / "runtime").mkdir(parents=True)
    (tmp_path / "core" / "security").mkdir(parents=True)
    (tmp_path / "core" / "identity").mkdir(parents=True)
    (tmp_path / "adapters").mkdir()
    (tmp_path / "core" / "will.py").write_text("SEALED = True\n")
    (tmp_path / "core" / "widget.py").write_text("VALUE = 1\n")
    (tmp_path / "core" / "runtime" / "loop.py").write_text("TICK = 1\n")
    (tmp_path / "core" / "identity" / "self.py").write_text("NAME = 'aura'\n")
    (tmp_path / "adapters" / "plugin.py").write_text("PLUGIN = 1\n")
    (tmp_path / "artifact.json").write_text("{}\n")
    return tmp_path


def _governor(repo):
    return ArchitectureEvolutionGovernor(repo_root=repo)


def _evidence(plan, *, artifact="artifact.json", trust_root="aura-trust-root"):
    return {
        obligation.name: {
            "passed": True,
            "artifact": artifact,
            "verifier": "pytest",
            "subject_digest": plan.subject_digest,
            "trust_root": trust_root,
        }
        for obligation in plan.obligations
    }


# --- 65ecb1f1: obligations must not trust a truthy 'passed' -----------------


def test_truthy_non_boolean_passed_does_not_satisfy(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/widget.py"], summary="tweak")
    evidence = _evidence(plan)
    for value in evidence.values():
        value["passed"] = "yes"

    certified = governor.evaluate_promotion(plan, evidence)

    assert not certified.promotable
    assert any("not the boolean True" in reason for reason in certified.blocking_reasons)


def test_caller_constructed_evidence_without_an_artifact_is_refused(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/widget.py"], summary="tweak")
    evidence = {o.name: {"passed": True} for o in plan.obligations}

    certified = governor.evaluate_promotion(plan, evidence)

    assert not certified.promotable
    reasons = " ".join(certified.blocking_reasons)
    assert "artifact" in reasons and "verifier" in reasons and "subject_digest" in reasons


def test_unresolvable_artifact_is_refused(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/widget.py"], summary="tweak")

    certified = governor.evaluate_promotion(plan, _evidence(plan, artifact="does/not/exist.json"))

    assert not certified.promotable
    assert any("does not resolve" in reason for reason in certified.blocking_reasons)


def test_artifact_outside_the_repository_is_refused(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/widget.py"], summary="tweak")

    certified = governor.evaluate_promotion(plan, _evidence(plan, artifact="/etc/hosts"))

    assert not certified.promotable
    assert any("escapes the repository" in reason for reason in certified.blocking_reasons)


def test_content_digest_is_an_acceptable_artifact(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/widget.py"], summary="tweak")
    digest = hashlib.sha256(b"report").hexdigest()

    certified = governor.evaluate_promotion(plan, _evidence(plan, artifact=digest))

    assert certified.promotable


def test_evidence_about_other_code_is_refused(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/widget.py"], summary="tweak")
    evidence = _evidence(plan)
    for value in evidence.values():
        value["subject_digest"] = hashlib.sha256(b"some other tree").hexdigest()

    certified = governor.evaluate_promotion(plan, evidence)

    assert not certified.promotable
    assert any("does not match the planned targets" in r for r in certified.blocking_reasons)


def test_governance_tier_requires_a_trust_root(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/identity/self.py"], summary="identity")
    assert plan.tier is MutationTier.GOVERNANCE_OR_IDENTITY

    without = _evidence(plan)
    for value in without.values():
        value.pop("trust_root")
    assert not governor.evaluate_promotion(plan, without).promotable
    assert governor.evaluate_promotion(plan, _evidence(plan)).promotable


def test_optional_obligation_is_satisfied_without_evidence():
    obligation = ProofObligation("optional", "not required", required=False)
    assert obligation.satisfied
    assert obligation.defects == ()


# --- aedd615c: path classification must be canonical -----------------------


def test_traversal_and_dot_segments_still_classify_as_sealed(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(
        target_paths=["core/../core/./will.py"], summary="sneak"
    )
    assert plan.target_paths == ("core/will.py",)
    assert plan.tier is MutationTier.SEALED_CORE
    assert plan.sealed


def test_case_variants_classify_as_sealed(repo):
    plan = _governor(repo).plan_mutation(target_paths=["Core/Will.py"], summary="case")
    assert plan.tier is MutationTier.SEALED_CORE


def test_backslash_separators_classify_as_sealed(repo):
    plan = _governor(repo).plan_mutation(target_paths=["core\\will.py"], summary="sep")
    assert plan.tier is MutationTier.SEALED_CORE


def test_symlink_out_of_the_repository_is_rejected(repo, tmp_path):
    outside = tmp_path.parent / "outside_secret.py"
    outside.write_text("SECRET = 1\n")
    (repo / "core" / "link.py").symlink_to(outside)

    plan = _governor(repo).plan_mutation(target_paths=["core/link.py"], summary="link")

    assert plan.target_paths == ()
    assert plan.rejected_targets
    assert "escapes the repository" in plan.rejected_targets[0]["reason"]
    assert not plan.promotable


def test_absolute_path_outside_the_repository_is_rejected(repo):
    plan = _governor(repo).plan_mutation(target_paths=["/etc/passwd"], summary="nope")
    assert plan.target_paths == ()
    assert plan.rejected_targets


def test_misleading_substring_does_not_over_classify(repo):
    (repo / "docs").mkdir()
    (repo / "docs" / "core_security_notes.md").write_text("notes\n")

    plan = _governor(repo).plan_mutation(
        target_paths=["docs/core_security_notes.md"], summary="notes"
    )

    assert not plan.sealed
    assert plan.tier is MutationTier.ADAPTER


def test_rejected_target_escalates_the_tier(repo):
    plan = _governor(repo).plan_mutation(
        target_paths=["adapters/plugin.py", "/etc/passwd"], summary="mixed"
    )
    assert plan.tier >= MutationTier.GOVERNANCE_OR_IDENTITY
    assert not plan.promotable


# --- 33a4241b: SEALED_CORE must be reachable -------------------------------


def test_sealed_core_tier_is_reachable(repo):
    plan = _governor(repo).plan_mutation(target_paths=["core/will.py"], summary="x")
    assert plan.tier is MutationTier.SEALED_CORE
    assert plan.sealed
    assert not plan.promotable


def test_sealed_directory_prefix_matches(repo):
    (repo / "core" / "security" / "guard.py").write_text("GUARD = 1\n")
    plan = _governor(repo).plan_mutation(target_paths=["core/security/guard.py"], summary="x")
    assert plan.tier is MutationTier.SEALED_CORE


def test_tier_ladder_is_distinguishable(repo):
    governor = _governor(repo)
    assert governor.plan_mutation(
        target_paths=["adapters/plugin.py"], summary="x"
    ).tier is MutationTier.ADAPTER
    assert governor.plan_mutation(
        target_paths=["core/widget.py"], summary="x"
    ).tier is MutationTier.FEATURE_MODULE
    assert governor.plan_mutation(
        target_paths=["core/runtime/loop.py"], summary="x"
    ).tier is MutationTier.SHARED_RUNTIME
    assert governor.plan_mutation(
        target_paths=["core/identity/self.py"], summary="x"
    ).tier is MutationTier.GOVERNANCE_OR_IDENTITY
    assert governor.plan_mutation(
        target_paths=["core/will.py"], summary="x"
    ).tier is MutationTier.SEALED_CORE


def test_strongest_tier_wins_across_targets(repo):
    plan = _governor(repo).plan_mutation(
        target_paths=["adapters/plugin.py", "core/will.py"], summary="x"
    )
    assert plan.tier is MutationTier.SEALED_CORE


# --- a8481160: identity must bind code and evidence -------------------------


def test_plan_id_changes_when_the_target_content_changes(repo):
    governor = _governor(repo)
    before = governor.plan_mutation(target_paths=["core/widget.py"], summary="x")
    (repo / "core" / "widget.py").write_text("VALUE = 2\n")
    after = governor.plan_mutation(target_paths=["core/widget.py"], summary="x")

    assert before.plan_id != after.plan_id
    assert before.subject_digest != after.subject_digest


def test_certificate_id_moves_with_the_evidence_while_plan_id_holds(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/widget.py"], summary="x")
    certified = governor.evaluate_promotion(plan, _evidence(plan))

    assert certified.plan_id == plan.plan_id
    assert certified.certificate_id != plan.certificate_id
    assert certified.promotable and not plan.promotable


def test_absent_target_records_an_absent_digest(repo):
    plan = _governor(repo).plan_mutation(target_paths=["core/new_module.py"], summary="new")
    assert plan.source_digests["core/new_module.py"] == "absent"


# --- 8ca18b5b: records are immutable after the decision ---------------------


def test_plan_and_obligations_cannot_be_mutated(repo):
    plan = _governor(repo).plan_mutation(target_paths=["core/widget.py"], summary="x")

    with pytest.raises((AttributeError, TypeError)):
        plan.sealed = False
    with pytest.raises((AttributeError, TypeError)):
        plan.obligations[0].required = False


def test_nested_evidence_cannot_be_edited_after_certification(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/widget.py"], summary="x")
    evidence = _evidence(plan)
    for value in evidence.values():
        value["passed"] = False
    certified = governor.evaluate_promotion(plan, evidence)
    assert not certified.promotable

    with pytest.raises((AttributeError, TypeError)):
        certified.obligations[0].evidence["passed"] = True

    assert not certified.promotable


def test_source_evidence_dict_is_copied_not_aliased(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/widget.py"], summary="x")
    evidence = _evidence(plan)
    certified = governor.evaluate_promotion(plan, evidence)
    assert certified.promotable

    for value in evidence.values():
        value["passed"] = False

    assert certified.promotable  # the decision did not move under it


def test_frozen_map_round_trips_through_to_dict(repo):
    plan = _governor(repo).plan_mutation(target_paths=["core/widget.py"], summary="x")
    payload = plan.to_dict()
    assert isinstance(payload["source_digests"], dict)
    assert isinstance(payload["obligations"][0]["evidence"], dict)
    assert payload["promotable"] is False
    assert payload["blocking_reasons"]


def test_frozen_map_is_a_read_only_mapping():
    frozen = FrozenMap({"a": {"b": [1, 2]}})
    assert frozen["a"]["b"] == (1, 2)
    with pytest.raises(TypeError):
        frozen["a"] = 1


# --- d8af3bdc: the governor must actually govern the mutation ---------------


def test_promote_refuses_an_uncertified_plan(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/widget.py"], summary="x")
    applied = []

    receipt = governor.promote(plan, apply=lambda targets: applied.append(targets))

    assert receipt["status"] == "refused"
    assert receipt["promoted"] is False
    assert applied == []


def test_promote_refuses_a_sealed_plan_even_with_full_evidence(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/will.py"], summary="x")
    applied = []

    receipt = governor.promote(plan, _evidence(plan), apply=lambda t: applied.append(t))

    assert receipt["status"] == "refused"
    assert applied == []
    assert any("sealed core" in reason for reason in receipt["reasons"])


def test_promote_refuses_when_the_target_moved_since_certification(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/widget.py"], summary="x")
    evidence = _evidence(plan)
    (repo / "core" / "widget.py").write_text("VALUE = 999\n")
    applied = []

    receipt = governor.promote(plan, evidence, apply=lambda t: applied.append(t))

    assert receipt["status"] == "refused"
    assert applied == []
    assert any("changed since certification" in r for r in receipt["reasons"])


def test_promote_applies_a_certified_mutation(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/widget.py"], summary="x")

    def apply(targets):
        (repo / "core" / "widget.py").write_text("VALUE = 42\n")

    receipt = governor.promote(plan, _evidence(plan), apply=apply)

    assert receipt["status"] == "promoted"
    assert receipt["promoted"] is True
    assert (repo / "core" / "widget.py").read_text() == "VALUE = 42\n"


def test_failed_verification_rolls_the_mutation_back(repo):
    governor = _governor(repo)
    original = (repo / "core" / "widget.py").read_text()
    plan = governor.plan_mutation(target_paths=["core/widget.py"], summary="x")

    receipt = governor.promote(
        plan,
        _evidence(plan),
        apply=lambda t: (repo / "core" / "widget.py").write_text("BROKEN\n"),
        verify=lambda t: False,
    )

    assert receipt["status"] == "rolled_back"
    assert receipt["rolled_back"] is True
    assert (repo / "core" / "widget.py").read_text() == original


def test_raising_apply_rolls_the_mutation_back(repo):
    governor = _governor(repo)
    original = (repo / "core" / "widget.py").read_text()
    plan = governor.plan_mutation(target_paths=["core/widget.py"], summary="x")

    def apply(targets):
        (repo / "core" / "widget.py").write_text("HALF WRITTEN\n")
        raise RuntimeError("apply blew up")

    receipt = governor.promote(plan, _evidence(plan), apply=apply)

    assert receipt["status"] == "rolled_back"
    assert (repo / "core" / "widget.py").read_text() == original
    assert any("apply blew up" in r for r in receipt["reasons"])


def test_rollback_removes_a_file_that_did_not_exist_before(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/new_module.py"], summary="new")

    def apply(targets):
        (repo / "core" / "new_module.py").write_text("NEW = 1\n")
        raise RuntimeError("fail after create")

    receipt = governor.promote(plan, _evidence(plan), apply=apply)

    assert receipt["rolled_back"] is True
    assert not (repo / "core" / "new_module.py").exists()


def test_a_no_op_promotion_is_reported_as_such(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/widget.py"], summary="x")

    receipt = governor.promote(plan, _evidence(plan), apply=lambda t: None)

    assert receipt["status"] == "no_op"
    assert receipt["promoted"] is False


def test_certification_without_apply_reports_certified(repo):
    governor = _governor(repo)
    plan = governor.plan_mutation(target_paths=["core/widget.py"], summary="x")

    receipt = governor.promote(plan, _evidence(plan))

    assert receipt["status"] == "certified"
    assert receipt["promoted"] is False
    assert receipt["reasons"] == []
