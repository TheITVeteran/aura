from __future__ import annotations

import copy
import hashlib

import pytest

from core.brain.llm.latent_cortex.atomic_decomposition import (
    build_atomic_decomposition,
)
from core.brain.llm.latent_cortex.diagnostic_action_selector import (
    build_candidate_routes,
    build_diagnostic_action_selector_receipt,
    validate_diagnostic_action_selector_receipt,
)
from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.brain.llm.latent_cortex.value_of_computation import (
    action_cost_estimate,
    build_evidence_snapshot,
)
from tests.fixtures.action_calibration import certified_action_snapshot


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _graph(
    *,
    dispute_kind: str = "causal_transition",
    decompositions: dict | None = None,
) -> dict:
    candidate_available = dispute_kind != "causal_transition"
    return {
        "receipt_sha256": _digest("graph"),
        "candidate_decompositions": decompositions or {},
        "pairwise": [
            {
                "left": 0,
                "right": 1,
                "localized": True,
                "causal_divergence": {
                    "available": True,
                    "kind": "causal_transition",
                    "action_step": 2,
                },
                "candidate_divergence": (
                    {
                        "available": True,
                        "kind": dispute_kind,
                        "atom_ordinal": 0,
                        "left": {
                            "atom_id": "a000",
                            "text_sha256": _digest("left"),
                        },
                        "right": {
                            "atom_id": "a000",
                            "text_sha256": _digest("right"),
                        },
                    }
                    if candidate_available
                    else {
                        "available": False,
                        "reason": "decoded_candidates_unavailable",
                    }
                ),
            }
        ],
    }


def _policy(snapshot: dict, executors: tuple[OperationKind, ...]) -> dict:
    return {
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "executors": [operation.value for operation in executors],
    }


def _trace(
    *,
    memory: bool = False,
    evidence: bool = False,
    verifier: bool = True,
    savepoint: bool = True,
) -> list[dict]:
    return [
        {
            "state_signal": {
                "has_memory": memory,
                "has_evidence": evidence,
                "has_verifier": verifier,
                "has_savepoint": savepoint,
            }
        },
        {
            "state_signal": {
                "has_memory": memory,
                "has_evidence": evidence,
                "has_verifier": verifier,
                "has_savepoint": savepoint,
            }
        },
    ]


def _build(
    *,
    graph: dict | None = None,
    routes: dict | None = None,
    cells: dict | None = None,
    snapshot: dict | None = None,
    executors: tuple[OperationKind, ...] = (
        OperationKind.SIMULATE,
        OperationKind.FALSIFY,
        OperationKind.CHECK_ASSUMPTION,
        OperationKind.REGENERATE_FROM_PREFIX,
    ),
    trace: list[dict] | None = None,
) -> dict:
    snapshot = snapshot or build_evidence_snapshot(
        bucket="diagnostic",
        cells=cells or {},
    )
    return build_diagnostic_action_selector_receipt(
        disagreement_graph=graph or _graph(),
        candidate_routes=routes or {},
        action_policy_evidence=snapshot,
        value_policy=_policy(snapshot, executors),
        action_trace=trace or _trace(),
    )


def test_causal_divergence_selects_prefix_regeneration_when_available():
    receipt = _build()

    selected = receipt["plans"][0]["selected"]
    assert selected["status"] == "diagnostic_operation_selected"
    assert selected["method"] == "regenerate_from_prefix"
    assert selected["operation_kind"] == "regenerate_from_prefix"
    assert selected["cost_basis"] == "declared_bootstrap_cost"
    assert receipt["execution_effect"] == "none"


def test_exact_arithmetic_route_is_recorded_as_already_executed_resolution():
    candidates = {0: "2 + 2 = 4", 1: "2 + 2 = 5"}
    objective = "Check both arithmetic claims."
    decompositions = {
        str(index): build_atomic_decomposition(text, objective=objective)
        for index, text in candidates.items()
    }
    routes = build_candidate_routes(
        candidates,
        objective=objective,
        candidate_decompositions=decompositions,
    )
    receipt = _build(
        graph=_graph(dispute_kind="claim", decompositions=decompositions),
        routes=routes,
    )

    plan = receipt["plans"][0]
    assert {row["outcome"] for row in plan["route_rows"]} == {
        "verified",
        "refuted",
    }
    assert plan["selected"] == {
        "status": "resolved_by_existing_exact_route",
        "method": "execute",
        "operation_kind": "execute",
        "applicability": 1.0,
        "applicability_basis": "deterministic_route_already_executed",
        "expected_resolution_score": 1.0,
        "expected_resolution_basis": "deterministic_exact",
        "cost_upper_bound": 0.0,
        "cost_basis": "already_executed_deterministic_route",
        "already_executed": True,
    }


def test_two_individually_valid_exact_claims_do_not_fake_dispute_resolution():
    candidates = {0: "2 + 2 = 4", 1: "2 + 3 = 5"}
    objective = "Determine which arithmetic claim answers the objective."
    decompositions = {
        str(index): build_atomic_decomposition(text, objective=objective)
        for index, text in candidates.items()
    }
    routes = build_candidate_routes(
        candidates,
        objective=objective,
        candidate_decompositions=decompositions,
    )
    receipt = _build(
        graph=_graph(dispute_kind="claim", decompositions=decompositions),
        routes=routes,
    )

    execute = next(row for row in receipt["plans"][0]["candidates"] if row["method"] == "execute")
    assert execute["already_executed"] is False
    assert execute["available"] is False
    assert receipt["plans"][0]["selected"]["method"] == "specialized_verifier"


def test_source_route_selects_real_available_evidence_reinspection():
    candidates = {
        0: "According to source A, the value is 4.",
        1: "According to source B, the value is 5.",
    }
    objective = "Resolve the sourced factual disagreement."
    decompositions = {
        str(index): build_atomic_decomposition(text, objective=objective)
        for index, text in candidates.items()
    }
    routes = build_candidate_routes(
        candidates,
        objective=objective,
        candidate_decompositions=decompositions,
    )
    receipt = _build(
        graph=_graph(dispute_kind="claim", decompositions=decompositions),
        routes=routes,
        executors=(
            OperationKind.RETRIEVE_EVIDENCE,
            OperationKind.FALSIFY,
            OperationKind.CHECK_ASSUMPTION,
        ),
        trace=_trace(evidence=True),
    )

    selected = receipt["plans"][0]["selected"]
    assert selected["method"] == "retrieve"
    assert selected["operation_kind"] == "retrieve_evidence"
    assert selected["applicability_basis"] == ("disputed_atom_requested_this_verifier_class")


def test_memory_only_retrieval_uses_search_memory_not_evidence_operation():
    receipt = _build(
        graph=_graph(dispute_kind="claim"),
        executors=(OperationKind.SEARCH_MEMORY,),
        trace=_trace(memory=True, verifier=False, savepoint=False),
    )

    selected = receipt["plans"][0]["selected"]
    assert selected["method"] == "retrieve"
    assert selected["operation_kind"] == "search_memory"


def test_measured_cost_breaks_equal_capability_band_conservatively(
    tmp_path,
    monkeypatch,
):
    snapshot, root_pem = certified_action_snapshot(
        bucket="diagnostic",
        cells={
            OperationKind.CHECK_ASSUMPTION: (0.7, 0.8, 0.9, 0.9),
            OperationKind.FALSIFY: (0.7, 0.8, 0.9, 0.05),
        },
    )
    root_path = tmp_path / "action-calibration-root.pem"
    root_path.write_bytes(root_pem)
    monkeypatch.setenv(
        "AURA_RLC_ACTION_CALIBRATION_TRUST_ROOT",
        str(root_path),
    )
    receipt = _build(
        graph=_graph(dispute_kind="assumption"),
        snapshot=snapshot,
        executors=(OperationKind.FALSIFY, OperationKind.CHECK_ASSUMPTION),
        trace=_trace(savepoint=False),
    )

    selected = receipt["plans"][0]["selected"]
    assert selected["method"] == "falsify"
    assert selected["cost_basis"] == "measured_cost_ucb"
    assert selected["cost_upper_bound"] == pytest.approx(0.05)


def test_no_available_capable_executor_abstains_instead_of_inventing_one():
    receipt = _build(
        executors=(OperationKind.ANSWER,),
        trace=_trace(verifier=False, savepoint=False),
    )

    assert receipt["selected_plan_count"] == 0
    assert receipt["plans"][0]["selected"]["status"] == ("no_admissible_diagnostic_operation")
    assert all(row["available"] is False for row in receipt["plans"][0]["candidates"])


def test_nonpositive_measured_gain_cannot_win_as_expected_resolution(
    tmp_path,
    monkeypatch,
):
    snapshot, root_pem = certified_action_snapshot(
        bucket="diagnostic",
        cells={
            OperationKind.CHECK_ASSUMPTION: (-0.9, -0.8, -0.7, 0.1),
        },
    )
    root_path = tmp_path / "action-calibration-root.pem"
    root_path.write_bytes(root_pem)
    monkeypatch.setenv(
        "AURA_RLC_ACTION_CALIBRATION_TRUST_ROOT",
        str(root_path),
    )
    receipt = _build(
        graph=_graph(dispute_kind="assumption"),
        snapshot=snapshot,
        executors=(OperationKind.CHECK_ASSUMPTION,),
        trace=_trace(savepoint=False),
    )

    candidate = next(
        row for row in receipt["plans"][0]["candidates"] if row["method"] == "specialized_verifier"
    )
    assert candidate["available"] is True
    assert candidate["expected_resolution_score"] == 0.0
    assert receipt["plans"][0]["selected"]["status"] == ("no_admissible_diagnostic_operation")


def test_formalization_is_not_misrepresented_as_a_proof_executor():
    receipt = _build(
        graph=_graph(dispute_kind="dependency_transition"),
        executors=(OperationKind.FORMALIZE, OperationKind.CHECK_ASSUMPTION),
        trace=_trace(savepoint=False),
    )

    prove = next(row for row in receipt["plans"][0]["candidates"] if row["method"] == "prove")
    assert prove["available"] is False
    assert receipt["plans"][0]["selected"]["method"] == "specialized_verifier"


def test_cost_estimate_rejects_unknown_action_and_binds_snapshot():
    snapshot = build_evidence_snapshot(bucket="diagnostic", cells={})
    estimate = action_cost_estimate(snapshot, OperationKind.FALSIFY)
    assert estimate["basis"] == "declared_bootstrap_cost"
    assert estimate["gain_lower_bound"] is None
    assert estimate["evidence_snapshot_sha256"] == snapshot["snapshot_sha256"]
    with pytest.raises(ValueError, match="unknown cognitive action"):
        action_cost_estimate(snapshot, "invent_answer")


def test_validator_reconstructs_selection_and_rejects_authority_tampering():
    receipt = _build()
    validate_diagnostic_action_selector_receipt(
        receipt,
        disagreement_graph=_graph(),
        value_policy={
            "snapshot_sha256": receipt["action_policy_evidence"]["snapshot_sha256"],
            "executors": receipt["capabilities"]["executors"],
        },
        action_trace=_trace(),
    )

    tampered = copy.deepcopy(receipt)
    tampered["execution_effect"] = "executed"
    with pytest.raises(ValueError, match="differs from reconstruction"):
        validate_diagnostic_action_selector_receipt(
            tampered,
            disagreement_graph=_graph(),
            value_policy={
                "snapshot_sha256": receipt["action_policy_evidence"]["snapshot_sha256"],
                "executors": receipt["capabilities"]["executors"],
            },
            action_trace=_trace(),
        )


def test_inconsistent_capability_flags_fail_closed():
    trace = _trace()
    trace[1]["state_signal"]["has_verifier"] = False
    with pytest.raises(ValueError, match="has_verifier is inconsistent"):
        _build(trace=trace)
