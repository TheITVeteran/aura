"""CP126 eb6bed71 + 2cbeadca + a70d12ad: three labels doing a proof's work.

Breadth was "two unique non-empty domain strings". ``["math", "maths"]``
satisfied it, and so did ``["gsm8k", "gsm8k-hard"]`` — a claim about general
frontier capability resting on whether two labels happened to differ as text.

Episode and request ids only had to be non-empty and globally unique. They were
free strings, unrelated to the task that ran, the output produced, or the worker
that served it, so two trials could swap identifiers or one could be minted
afterwards to make a lineage look clean.

And the arms' information receipts had to match each other, with nothing tying
either to the preregistration. Both arms could be handed the same undeclared
retrieval context and decoded under the same undeclared policy, and the parity
check would pass.
"""
from __future__ import annotations

import copy

import pytest

from core.brain.llm.latent_cortex.frontier_certification import (
    MINIMUM_CAPABILITY_CLASSES,
    _expected_arm_identifier,
    canonical_sha256,
)
from tests.fixtures.latent_frontier import (
    _DOMAIN_TAXONOMY,
    _bundle,
    _certify,
    _refresh_task_commitment,
)


class TestDomainBreadth:
    def test_the_certificate_reports_capability_classes(self):
        certificate = _certify(_bundle())
        assert certificate["accepted"] is True, certificate["reasons"]
        assert len(certificate["capability_classes"]) >= MINIMUM_CAPABILITY_CLASSES
        assert certificate["required_capability_classes"] == (
            MINIMUM_CAPABILITY_CLASSES
        )

    def test_a_bundle_without_a_taxonomy_is_refused(self):
        bundle = _bundle()
        del bundle["domain_taxonomy"]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "domain_taxonomy_missing" in certificate["reasons"]

    def test_the_taxonomy_must_be_the_preregistered_one(self):
        """Rewriting the ontology after the results are in is not breadth."""
        bundle = _bundle()
        taxonomy = copy.deepcopy(_DOMAIN_TAXONOMY)
        taxonomy["ontology_id"] = "convenient-ontology-v2"
        bundle["domain_taxonomy"] = taxonomy
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "domain_taxonomy_not_preregistered" in certificate["reasons"]

    def test_three_labels_for_one_capability_do_not_make_breadth(self):
        """The exact substitution a string count cannot see."""
        bundle = _bundle()
        taxonomy = copy.deepcopy(_DOMAIN_TAXONOMY)
        for entry in taxonomy["domains"].values():
            entry["capability_class"] = "formal_quantitative_reasoning"
        bundle["domain_taxonomy"] = taxonomy
        bundle["preregistration"]["domain_taxonomy_sha256"] = canonical_sha256(taxonomy)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "domain_coverage_too_narrow" in certificate["reasons"]
        assert certificate["capability_classes"] == [
            "formal_quantitative_reasoning"
        ]

    def test_a_domain_outside_the_taxonomy_is_named(self):
        bundle = _bundle()
        taxonomy = copy.deepcopy(_DOMAIN_TAXONOMY)
        del taxonomy["domains"]["science"]
        bundle["domain_taxonomy"] = taxonomy
        bundle["preregistration"]["domain_taxonomy_sha256"] = canonical_sha256(taxonomy)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "science:domain_outside_taxonomy" in certificate["reasons"]

    def test_an_unattributed_taxonomy_is_refused(self):
        bundle = _bundle()
        taxonomy = copy.deepcopy(_DOMAIN_TAXONOMY)
        taxonomy["ontology_id"] = ""
        bundle["domain_taxonomy"] = taxonomy
        bundle["preregistration"]["domain_taxonomy_sha256"] = canonical_sha256(taxonomy)
        certificate = _certify(bundle)
        assert "domain_taxonomy_unattributed" in certificate["reasons"]

    def test_an_entry_with_no_capability_class_is_named(self):
        bundle = _bundle()
        taxonomy = copy.deepcopy(_DOMAIN_TAXONOMY)
        taxonomy["domains"]["coding"] = {"capability_class": "  "}
        bundle["domain_taxonomy"] = taxonomy
        bundle["preregistration"]["domain_taxonomy_sha256"] = canonical_sha256(taxonomy)
        certificate = _certify(bundle)
        assert "coding:domain_capability_class_missing" in certificate["reasons"]


class TestArmIdentifierBinding:
    def test_the_identifier_changes_with_every_part_of_the_record(self):
        base = {
            "trial_id": "math-0",
            "task_id": "heldout-math-0",
            "task_payload_sha256": "a" * 64,
            "treatment_output_sha256": "b" * 64,
            "control_output_sha256": "c" * 64,
            "evaluation_started_at": 1201.0,
        }
        original = _expected_arm_identifier(base, "treatment", "d" * 32)
        for field, value in (
            ("trial_id", "math-1"),
            ("task_id", "heldout-math-1"),
            ("task_payload_sha256", "e" * 64),
            ("treatment_output_sha256", "f" * 64),
            ("evaluation_started_at", 1202.0),
        ):
            assert _expected_arm_identifier(
                {**base, field: value}, "treatment", "d" * 32
            ) != original, field
        assert (
            _expected_arm_identifier(base, "treatment", "0" * 32) != original
        )

    def test_the_two_arms_of_one_trial_get_different_identifiers(self):
        base = {
            "trial_id": "math-0",
            "task_id": "heldout-math-0",
            "task_payload_sha256": "a" * 64,
            "treatment_output_sha256": "b" * 64,
            "control_output_sha256": "b" * 64,
            "evaluation_started_at": 1201.0,
        }
        assert _expected_arm_identifier(
            base, "treatment", "d" * 32
        ) != _expected_arm_identifier(base, "control", "d" * 32)

    def test_a_free_string_episode_id_is_refused(self):
        bundle = _bundle()
        trial = copy.deepcopy(bundle["trials"][0])
        trial["treatment_receipt"]["episode_id"] = "episode-math-0"
        bundle["trials"][0] = trial
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert any(
            reason.endswith(":treatment_episode_id_unbound")
            for reason in certificate["reasons"]
        )

    def test_a_free_string_request_id_is_refused(self):
        bundle = _bundle()
        trial = copy.deepcopy(bundle["trials"][0])
        trial["control_receipt"]["request_id"] = "control-math-0"
        bundle["trials"][0] = trial
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert any(
            reason.endswith(":control_request_id_unbound")
            for reason in certificate["reasons"]
        )

    def test_swapping_identifiers_between_trials_is_caught(self):
        """Both remain globally unique, which is all the old check asked."""
        bundle = _bundle()
        first = copy.deepcopy(bundle["trials"][0])
        second = copy.deepcopy(bundle["trials"][1])
        first["control_receipt"]["request_id"], second["control_receipt"][
            "request_id"
        ] = (
            second["control_receipt"]["request_id"],
            first["control_receipt"]["request_id"],
        )
        bundle["trials"][0] = first
        bundle["trials"][1] = second
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        unbound = [
            reason
            for reason in certificate["reasons"]
            if reason.endswith(":control_request_id_unbound")
        ]
        assert len(unbound) == 2
        assert "duplicate_control_request" not in " ".join(certificate["reasons"])


class TestArmVisiblePolicies:
    @pytest.mark.parametrize(
        ("policy", "prereg_key"),
        [
            ("decode", "decode_policy_sha256"),
            ("tool", "tool_policy_sha256"),
            ("verifier", "scorer_implementation_sha256"),
        ],
    )
    def test_each_policy_is_pinned_to_the_preregistration(self, policy, prereg_key):
        """Both arms matching each other is not the same as being declared."""
        bundle = _bundle()
        bundle["preregistration"][prereg_key] = "7" * 64
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert any(
            reason.endswith(f":treatment_{policy}_policy_not_preregistered")
            for reason in certificate["reasons"]
        )
        assert any(
            reason.endswith(f":control_{policy}_policy_not_preregistered")
            for reason in certificate["reasons"]
        )
