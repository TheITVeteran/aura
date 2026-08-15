"""CP126 a52cbfbf + 6b65ffd0: distinct task IDs are not distinct tasks.

Deduplication ran on ``task_payload_sha256``, which catches a copy-paste and
nothing else. Forty paraphrases of one problem carry forty distinct hashes and
each counted as an independent trial — toward the per-domain minimum and toward
the paired test. Sample size inflated while the evidence stood still.

The same commitment had a second hole. It bound WHICH tasks would run and
nothing about HOW: arm order, scorer configuration, decoding seeds and tool
policy were all outside the issuer's signature, so any of them could be chosen
after the outputs were known.
"""
from __future__ import annotations

import copy

import pytest

from core.brain.llm.latent_cortex.frontier_certification import (
    _task_manifest_sha256,
    _validate_task_diversity,
)
from tests.fixtures.latent_frontier import _bundle, _certify, _refresh_task_commitment


def _receipt(**overrides):
    receipt = {
        "method": "minhash_jaccard_13gram",
        "similarity_threshold": 0.6,
        "max_pairwise_similarity": 0.11,
        "task_families": {"t1": "family-a", "t2": "family-b"},
    }
    receipt.update(overrides)
    return receipt


class TestDiversityReceipt:
    def test_a_complete_receipt_yields_the_family_map(self):
        reasons: list[str] = []
        families, digest = _validate_task_diversity(
            {"task_diversity": _receipt()}, reasons
        )
        assert reasons == []
        assert families == {"t1": "family-a", "t2": "family-b"}
        assert len(digest) == 64

    def test_an_absent_receipt_is_named(self):
        reasons: list[str] = []
        families, digest = _validate_task_diversity({}, reasons)
        assert reasons == ["task_diversity_receipt_missing"]
        assert families == {}
        assert digest == ""

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({"method": ""}, "task_diversity_method_missing"),
            ({"similarity_threshold": 0.0}, "task_diversity_threshold_invalid"),
            ({"similarity_threshold": 1.5}, "task_diversity_threshold_invalid"),
            ({"max_pairwise_similarity": None}, "task_diversity_similarity_unmeasured"),
            ({"max_pairwise_similarity": 2.0}, "task_diversity_similarity_unmeasured"),
            ({"task_families": {}}, "task_families_missing"),
            ({"task_families": {"t1": 4}}, "task_families_malformed"),
            ({"task_families": {"": "family-a"}}, "task_families_malformed"),
        ],
    )
    def test_each_missing_element_is_named(self, overrides, expected):
        reasons: list[str] = []
        _validate_task_diversity({"task_diversity": _receipt(**overrides)}, reasons)
        assert expected in reasons

    def test_measured_similarity_above_the_ceiling_is_refused(self):
        reasons: list[str] = []
        _validate_task_diversity(
            {
                "task_diversity": _receipt(
                    similarity_threshold=0.6, max_pairwise_similarity=0.95
                )
            },
            reasons,
        )
        assert reasons == ["task_similarity_exceeds_threshold"]


class TestEffectiveSampleSize:
    def test_the_fixture_reports_one_family_per_task(self):
        certificate = _certify(_bundle())
        assert certificate["accepted"] is True, certificate["reasons"]
        assert certificate["effective_sample_by_domain"] == {
            "math": 40,
            "coding": 40,
            "science": 40,
        }

    def test_paraphrases_of_one_task_collapse_into_one_family(self):
        """The bundle still holds 40 math trials. It holds 4 math problems.

        Every count in the old certificate is unchanged by this mutation —
        distinct trial ids, distinct task ids, distinct payload hashes — and
        the domain would have passed its per-domain minimum on all of them.
        """
        bundle = _bundle()
        families = bundle["task_diversity"]["task_families"]
        for trial in bundle["trials"]:
            if trial["domain"] != "math":
                continue
            index = int(trial["trial_id"].rsplit("-", 1)[1])
            families[trial["task_id"]] = f"family-math-{index % 4}"
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert certificate["domain_counts"]["math"] == 40
        assert certificate["effective_sample_by_domain"]["math"] == 4
        assert "math:effective_sample_below_minimum" in certificate["reasons"]

    def test_correlated_tasks_cannot_inflate_the_paired_test(self):
        """A family contributes at most one discordant pair.

        Power here is computed on four independent problems, not forty
        correlated rows, so the domain can no longer buy significance by
        rephrasing.
        """
        bundle = _bundle()
        families = bundle["task_diversity"]["task_families"]
        for trial in bundle["trials"]:
            if trial["domain"] != "math":
                continue
            index = int(trial["trial_id"].rsplit("-", 1)[1])
            families[trial["task_id"]] = f"family-math-{index % 4}"
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["achieved_power_by_domain"]["math"] == 0.0
        assert "math:achieved_power_below_target" in certificate["reasons"]

    def test_a_task_outside_the_family_map_is_named(self):
        bundle = _bundle()
        victim = bundle["trials"][0]["task_id"]
        del bundle["task_diversity"]["task_families"][victim]
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert any(
            reason.endswith(":task_family_unassigned")
            for reason in certificate["reasons"]
        )


class TestCommitmentCoverage:
    def test_the_clustering_is_inside_the_issuer_signature(self):
        """Reclustering after the fact is choosing n once results are in."""
        bundle = _bundle()
        bundle["task_diversity"] = copy.deepcopy(bundle["task_diversity"])
        bundle["task_diversity"]["max_pairwise_similarity"] = 0.05
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "task_commitment_invalid" in certificate["reasons"]

    @pytest.mark.parametrize(
        "field",
        [
            "run_order",
            "scorer_config_sha256",
            "treatment_tool_policy_sha256",
            "control_tool_policy_sha256",
            "treatment_decode_policy_sha256",
            "control_decode_policy_sha256",
        ],
    )
    def test_experimental_design_moves_the_manifest_digest(self, field):
        """Each of these could previously be chosen after outputs were known."""
        bundle = _bundle()
        before = _task_manifest_sha256(bundle["trials"])
        bundle["trials"][0][field] = "changed"
        assert _task_manifest_sha256(bundle["trials"]) != before

    def test_changing_arm_order_after_issuance_breaks_the_commitment(self):
        bundle = _bundle()
        for trial in bundle["trials"]:
            trial["run_order"] = "treatment_first"
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "task_commitment_invalid" in certificate["reasons"]
