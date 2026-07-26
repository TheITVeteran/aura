"""SPARK-054 complete public causal-receipt contracts."""

from __future__ import annotations

import copy
import json

import pytest

from core.brain.llm.latent_cortex.causal_receipt import (
    build_causal_receipt,
    validate_causal_receipt,
)
from core.brain.llm.latent_cortex.loop_core import canonical_sha256

STAGES = (
    "identity_and_ingress",
    "state_lineage",
    "cognitive_operators",
    "branch_isolation_and_exchange",
    "tool_memory_and_external_evidence",
    "verification",
    "accepted_and_rejected_updates",
    "compute_accounting",
    "temporary_and_durable_adaptation",
    "stopping",
    "final_synthesis",
    "runtime_and_model_integrity",
)


def _complete_worker_receipt(*, private_marker: str = "PRIVATE-THOUGHT-DO-NOT-LEAK"):
    digest = "e" * 64
    return {
        "episode_id": "episode-spark-054",
        "request_payload_sha256": "1" * 64,
        "input_tokens_sha256": "2" * 64,
        "input_token_count": 16,
        "runtime_identity": {"identity_bound": True, "source_verified": True},
        "recurrent_grounding": {"receipt_sha256": "3" * 64},
        "loop_stability": {"all_finite": True},
        "kv_state_tree": {"root_sha256": "4" * 64},
        "cognitive_action_trace": [
            {"action": "falsify", "private_diagnostic": private_marker}
        ],
        "branch_isolation": {"certified": True},
        "selected_branch": 0,
        "verifier_fusion": {"selected_branch": 0},
        "neural_uncertainty": {"uncertainty": 0.1},
        "mistake_locator": {"mistake_found": False},
        "update_acceptance": {"accepted": 1, "rejected": 1},
        "verified_best_state": {"selected_branch": 0},
        "budget": {"spent_layer_apps": 128, "max_layer_apps": 1024},
        "latent_opt_applied": False,
        "fast_weights_applied": False,
        "fast_weight_learning": {
            "schema": "aura.rlc.fast_weight_learning.v1",
            "disposition": "not_admitted_high_confidence_evidence_absent",
            "receipt_sha256": "8" * 64,
        },
        "halting_reason": "converged",
        "halting": {"receipt_sha256": "5" * 64},
        "terminal_disposition": {
            "reason": "verified_convergence",
            "disposition": "answer",
            "language": {
                "output_text_sha256": "6" * 64,
                "source": "resident_model_decode",
            },
        },
        "decode_generated_tokens": 24,
        "decode_termination": "eos",
        "checkpoint_fingerprint": digest,
        "worker_boot_id": "boot-spark-054",
        "worker_pid": 42,
        "worker_model_path": "/models/resident-32b",
        "worker_source_sha256": "7" * 64,
        "params_unchanged": True,
        "fast_weights_erased": True,
        "weight_integrity": {
            "params_before": digest,
            "params_after": digest,
            "canary_before": digest,
            "canary_after": digest,
        },
        "integrity_verdicts": {
            "params_unchanged": {"verdict": "proven"},
            "fast_weights_erased": {"verdict": "proven"},
            "contradictions": [],
        },
    }


def _rehash_envelope(value: dict) -> None:
    previous = ""
    for node in value["nodes"]:
        node["prior_node_sha256"] = previous
        body = {key: item for key, item in node.items() if key != "node_sha256"}
        node["node_sha256"] = canonical_sha256(body)
        previous = node["node_sha256"]
    value["root_node_sha256"] = previous
    payload = {key: item for key, item in value.items() if key != "receipt_sha256"}
    value["receipt_sha256"] = canonical_sha256(payload)


def test_complete_envelope_orders_and_hash_links_every_causal_stage():
    worker = _complete_worker_receipt()
    receipt = build_causal_receipt(worker)

    assert receipt["stage_order"] == list(STAGES)
    assert [node["stage"] for node in receipt["nodes"]] == list(STAGES)
    assert receipt["required_stages_complete"] is True
    assert receipt["missing_required_stages"] == []
    assert receipt["integrity_proven"] is True
    assert receipt["final_output_text_sha256"] == "6" * 64
    assert receipt["root_node_sha256"] == receipt["nodes"][-1]["node_sha256"]
    assert receipt["nodes"][0]["prior_node_sha256"] == ""
    for prior, current in zip(receipt["nodes"], receipt["nodes"][1:], strict=False):
        assert current["prior_node_sha256"] == prior["node_sha256"]
    assert validate_causal_receipt(
        receipt,
        worker_receipt=worker,
        require_complete=True,
    ) == receipt


def test_optional_evidence_stage_distinguishes_absent_from_observed():
    worker = _complete_worker_receipt()
    absent = build_causal_receipt(worker)
    node = absent["nodes"][STAGES.index("tool_memory_and_external_evidence")]
    assert node["required"] is False
    assert node["status"] == "not_applicable"

    worker["nonparametric_memory"] = {"retrievals": 2}
    observed = build_causal_receipt(worker)
    node = observed["nodes"][STAGES.index("tool_memory_and_external_evidence")]
    assert node["status"] == "observed"


def test_causal_envelope_commits_query_scoped_fast_weight_contract():
    worker = _complete_worker_receipt()
    receipt = build_causal_receipt(worker)
    adaptation = receipt["nodes"][STAGES.index("temporary_and_durable_adaptation")]
    assert "fast_weight_learning" in {
        row["field"] for row in adaptation["source_commitments"]
    }

    worker["fast_weight_learning"]["disposition"] = (
        "accepted_causal_improvement"
    )
    with pytest.raises(ValueError, match="independently reconstructed"):
        validate_causal_receipt(
            receipt,
            worker_receipt=worker,
            require_complete=True,
        )


def test_public_envelope_never_copies_private_reasoning_or_tool_values():
    marker = "PRIVATE-THOUGHT-8e8cf8e9"
    worker = _complete_worker_receipt(private_marker=marker)
    worker["nonparametric_memory"] = {
        "retrieved_secret": "API-TOKEN-PRIVATE-842da1f0"
    }
    rendered = json.dumps(build_causal_receipt(worker), sort_keys=True)

    assert marker not in rendered
    assert "API-TOKEN-PRIVATE-842da1f0" not in rendered
    assert "private_diagnostic" not in rendered
    assert "retrieved_secret" not in rendered
    assert "cognitive_action_trace" in rendered
    assert "nonparametric_memory" in rendered


@pytest.mark.parametrize(
    "mutation",
    (
        "node_status",
        "stage_order",
        "privacy_contract",
        "final_output_hash",
    ),
)
def test_independent_reconstruction_rejects_self_consistent_envelope_tampering(
    mutation: str,
):
    worker = _complete_worker_receipt()
    receipt = copy.deepcopy(build_causal_receipt(worker))
    if mutation == "node_status":
        receipt["nodes"][2]["status"] = "incomplete"
    elif mutation == "stage_order":
        receipt["stage_order"][0], receipt["stage_order"][1] = (
            receipt["stage_order"][1],
            receipt["stage_order"][0],
        )
    elif mutation == "privacy_contract":
        receipt["privacy_contract"]["hidden_state_values_included"] = True
    else:
        receipt["final_output_text_sha256"] = "f" * 64
    _rehash_envelope(receipt)

    with pytest.raises(ValueError, match="independently reconstructed"):
        validate_causal_receipt(
            receipt,
            worker_receipt=worker,
            require_complete=True,
        )


def test_source_tampering_is_rejected_without_copying_source_values():
    worker = _complete_worker_receipt()
    receipt = build_causal_receipt(worker)
    worker["budget"]["spent_layer_apps"] += 1

    with pytest.raises(ValueError, match="independently reconstructed"):
        validate_causal_receipt(
            receipt,
            worker_receipt=worker,
            require_complete=True,
        )


def test_missing_stage_and_unproven_integrity_remain_honest_partial_receipts():
    incomplete_worker = _complete_worker_receipt()
    incomplete_worker.pop("kv_state_tree")
    incomplete = build_causal_receipt(incomplete_worker)
    assert incomplete["required_stages_complete"] is False
    assert incomplete["missing_required_stages"] == ["state_lineage"]
    assert validate_causal_receipt(
        incomplete,
        worker_receipt=incomplete_worker,
        require_complete=False,
    ) == incomplete
    with pytest.raises(ValueError, match="incomplete"):
        validate_causal_receipt(
            incomplete,
            worker_receipt=incomplete_worker,
            require_complete=True,
        )

    unproven_worker = _complete_worker_receipt()
    unproven_worker["integrity_verdicts"]["params_unchanged"]["verdict"] = "unproven"
    unproven = build_causal_receipt(unproven_worker)
    assert unproven["required_stages_complete"] is True
    assert unproven["integrity_proven"] is False
    with pytest.raises(ValueError, match="integrity is unproven"):
        validate_causal_receipt(
            unproven,
            worker_receipt=unproven_worker,
            require_complete=True,
        )
