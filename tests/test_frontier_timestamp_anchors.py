"""CP126 89ca4271 + 9cccfcb5: a signature never says when.

frozen_at, task_generated_at, evaluation_started_at and committed_at were
ordinary numbers inside signed payloads. A signature proves who wrote a value;
it says nothing about when the event happened. A producer and a task issuer
holding their own keys could assemble a flawless chronology after the results
were in, sign it, and every ordering check in the certificate would pass.

An append-only log breaks that. The event's digest goes in as a leaf, the log
returns an index and an audit path, and anyone can recompute the root those
imply and compare it against a root pinned out of band. Backdating then means
forging a hash chain against a root the producer does not control.
"""
from __future__ import annotations

import copy
import hashlib

import pytest

from core.brain.llm.latent_cortex.frontier_certification import (
    _merkle_leaf_hash,
    _merkle_root_from_proof,
)
from tests.fixtures.latent_frontier import _bundle, _certify


def _reference_tree(count: int) -> tuple[str, list[list[str]]]:
    """An RFC 6962 tree built the long way, for checking the proof verifier."""
    leaves = [hashlib.sha256(f"leaf-{i}".encode()).digest() for i in range(count)]
    nodes = [_merkle_leaf_hash(leaf) for leaf in leaves]
    paths: list[list[str]] = [[] for _ in range(count)]
    membership = [[i] for i in range(count)]
    while len(nodes) > 1:
        parents: list[str] = []
        parent_membership: list[list[int]] = []
        for index in range(0, len(nodes), 2):
            if index + 1 < len(nodes):
                for leaf in membership[index]:
                    paths[leaf].append(nodes[index + 1])
                for leaf in membership[index + 1]:
                    paths[leaf].append(nodes[index])
                parents.append(
                    hashlib.sha256(
                        b"\x01"
                        + bytes.fromhex(nodes[index])
                        + bytes.fromhex(nodes[index + 1])
                    ).hexdigest()
                )
                parent_membership.append(membership[index] + membership[index + 1])
            else:
                parents.append(nodes[index])
                parent_membership.append(membership[index])
        nodes, membership = parents, parent_membership
    return nodes[0], paths


class TestInclusionProof:
    @pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 7, 8, 11, 16])
    def test_every_leaf_of_every_tree_shape_reaches_the_root(self, size):
        """Unbalanced trees are where hand-rolled Merkle code goes wrong."""
        root, paths = _reference_tree(size)
        for index in range(size):
            leaf = _merkle_leaf_hash(hashlib.sha256(f"leaf-{index}".encode()).digest())
            assert (
                _merkle_root_from_proof(
                    leaf, leaf_index=index, tree_size=size, proof=paths[index]
                )
                == root
            )

    def test_a_leaf_cannot_pose_as_an_interior_node(self):
        """The RFC 6962 prefixes exist for this."""
        data = hashlib.sha256(b"leaf-0").digest()
        assert _merkle_leaf_hash(data) != hashlib.sha256(data).hexdigest()

    @pytest.mark.parametrize(
        ("index", "size"),
        [(-1, 4), (4, 4), (0, 0), (0, -1)],
    )
    def test_indices_outside_the_tree_are_rejected(self, index, size):
        with pytest.raises(ValueError):
            _merkle_root_from_proof(
                "a" * 64, leaf_index=index, tree_size=size, proof=[]
            )

    def test_a_proof_longer_than_the_tree_is_rejected(self):
        with pytest.raises(ValueError):
            _merkle_root_from_proof(
                "a" * 64, leaf_index=0, tree_size=2, proof=["b" * 64, "c" * 64]
            )

    def test_a_proof_shorter_than_the_tree_is_rejected(self):
        with pytest.raises(ValueError):
            _merkle_root_from_proof(
                "a" * 64, leaf_index=0, tree_size=4, proof=["b" * 64]
            )

    def test_a_non_digest_in_the_path_is_rejected(self):
        with pytest.raises(ValueError):
            _merkle_root_from_proof(
                "a" * 64, leaf_index=0, tree_size=2, proof=["not-a-hash"]
            )

    def test_a_tampered_sibling_changes_the_root(self):
        root, paths = _reference_tree(4)
        leaf = _merkle_leaf_hash(hashlib.sha256(b"leaf-0").digest())
        tampered = ["f" * 64] + paths[0][1:]
        assert (
            _merkle_root_from_proof(leaf, leaf_index=0, tree_size=4, proof=tampered)
            != root
        )


class TestCertificateAnchors:
    def test_the_certificate_reports_when_the_log_saw_each_event(self):
        certificate = _certify(_bundle())
        assert certificate["accepted"] is True, certificate["reasons"]
        assert certificate["preregistration_logged_at"] == 900.0
        assert certificate["task_commitment_logged_at"] == 1200.5

    @pytest.mark.parametrize("event", ["preregistration", "task_commitment"])
    def test_a_missing_anchor_is_refused(self, event):
        bundle = _bundle()
        del bundle["timestamp_anchors"][event]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert f"{event}_timestamp_anchor_missing" in certificate["reasons"]

    def test_a_bundle_with_no_anchors_at_all_is_refused(self):
        bundle = _bundle()
        del bundle["timestamp_anchors"]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "preregistration_timestamp_anchor_missing" in certificate["reasons"]
        assert "task_commitment_timestamp_anchor_missing" in certificate["reasons"]

    def test_an_unpinned_log_proves_nothing(self):
        """A root the producer chose is a root the producer can forge to."""
        bundle = _bundle()
        anchors = copy.deepcopy(bundle["timestamp_anchors"])
        anchors["preregistration"]["log_id"] = "producers-own-log"
        bundle["timestamp_anchors"] = anchors
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "preregistration_transparency_log_untrusted" in certificate["reasons"]

    def test_an_anchor_for_other_data_is_refused(self):
        bundle = _bundle()
        anchors = copy.deepcopy(bundle["timestamp_anchors"])
        anchors["preregistration"]["leaf_data_sha256"] = "4" * 64
        bundle["timestamp_anchors"] = anchors
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "preregistration_anchor_commits_other_data" in certificate["reasons"]

    def test_a_forged_audit_path_cannot_reach_the_pinned_root(self):
        bundle = _bundle()
        anchors = copy.deepcopy(bundle["timestamp_anchors"])
        anchors["task_commitment"]["inclusion_proof"] = ["e" * 64]
        bundle["timestamp_anchors"] = anchors
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert (
            "task_commitment_inclusion_proof_does_not_reach_pinned_root"
            in certificate["reasons"]
        )

    def test_a_proof_against_another_tree_size_is_refused(self):
        bundle = _bundle()
        anchors = copy.deepcopy(bundle["timestamp_anchors"])
        anchors["task_commitment"]["tree_size"] = 4
        bundle["timestamp_anchors"] = anchors
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "task_commitment_anchor_tree_size_unpinned" in certificate["reasons"]

    @pytest.mark.parametrize(
        "mutation",
        [
            {"leaf_index": "first"},
            {"tree_size": None},
            {"inclusion_proof": "not-a-list"},
        ],
    )
    def test_a_malformed_proof_is_named(self, mutation):
        bundle = _bundle()
        anchors = copy.deepcopy(bundle["timestamp_anchors"])
        anchors["preregistration"].update(mutation)
        bundle["timestamp_anchors"] = anchors
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "preregistration_inclusion_proof_malformed" in certificate["reasons"]

    def test_an_anchor_without_a_time_is_refused(self):
        bundle = _bundle()
        anchors = copy.deepcopy(bundle["timestamp_anchors"])
        del anchors["preregistration"]["logged_at"]
        bundle["timestamp_anchors"] = anchors
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "preregistration_anchor_time_missing" in certificate["reasons"]


class TestOrdering:
    def test_a_preregistration_logged_after_the_tasks_is_refused(self):
        """Written to fit the tasks is exactly what preregistration excludes."""
        bundle = _bundle()
        anchors = copy.deepcopy(bundle["timestamp_anchors"])
        earliest = min(trial["task_generated_at"] for trial in bundle["trials"])
        anchors["preregistration"]["logged_at"] = earliest + 1.0
        bundle["timestamp_anchors"] = anchors
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert (
            "preregistration_logged_after_task_generation" in certificate["reasons"]
        )

    def test_a_commitment_logged_after_evaluation_began_is_refused(self):
        bundle = _bundle()
        anchors = copy.deepcopy(bundle["timestamp_anchors"])
        earliest = min(trial["evaluation_started_at"] for trial in bundle["trials"])
        anchors["task_commitment"]["logged_at"] = earliest + 1.0
        bundle["timestamp_anchors"] = anchors
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert (
            "task_commitment_logged_after_evaluation_started"
            in certificate["reasons"]
        )
