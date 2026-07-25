"""Evidence-bound diagnostic operation selection for localized disputes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from core.brain.llm.latent_cortex.deterministic_verifier_router import (
    validate_deterministic_router_envelope,
)
from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.brain.llm.latent_cortex.value_of_computation import (
    action_cost_estimate,
    validate_evidence_snapshot,
)

DIAGNOSTIC_ACTION_SELECTOR_SCHEMA = "aura.rlc.diagnostic_action_selector.v1"

_METHOD_OPERATION = {
    "execute": OperationKind.EXECUTE,
    "retrieve": OperationKind.RETRIEVE_EVIDENCE,
    "prove": OperationKind.FORMALIZE,
    "simulate": OperationKind.SIMULATE,
    "falsify": OperationKind.FALSIFY,
    "regenerate_from_prefix": OperationKind.REGENERATE_FROM_PREFIX,
    "specialized_verifier": OperationKind.CHECK_ASSUMPTION,
}
_EXACT_VERIFIERS = {"exact_integer_arithmetic", "python_ast", "json_parser"}
_ROUTE_METHOD = {
    "formal_solver": "prove",
    "source_retrieval": "retrieve",
    "simulation": "simulate",
    "planning": "simulate",
}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build_candidate_routes(
    candidates: Mapping[int, str],
    *,
    objective: str,
    candidate_decompositions: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Run the pure deterministic router over every decoded branch candidate."""

    from core.brain.llm.latent_cortex.deterministic_verifier_router import (
        build_deterministic_router_receipt,
    )

    if sorted(candidates) != list(range(len(candidates))):
        raise ValueError("diagnostic route candidates must be contiguous")
    if set(candidate_decompositions) != {str(index) for index in candidates}:
        raise ValueError("diagnostic route decomposition coverage differs")
    return {
        str(index): build_deterministic_router_receipt(
            candidates[index],
            objective=objective,
            atomic_receipt=candidate_decompositions[str(index)],
        )
        for index in sorted(candidates)
    }


def _validated_routes(
    routes: Any,
    *,
    decompositions: Mapping[str, Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    if routes in (None, {}):
        if decompositions:
            raise ValueError("decoded candidate routes are missing")
        return {}
    if not isinstance(routes, Mapping) or set(routes) != set(decompositions):
        raise ValueError("diagnostic route coverage differs")
    return {
        int(index): validate_deterministic_router_envelope(
            routes[index],
            atomic_receipt=decompositions[index],
        )
        for index in sorted(routes, key=int)
    }


def _capabilities(
    *,
    value_policy: Any,
    action_trace: Any,
    evidence_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value_policy, Mapping):
        raise ValueError("value policy receipt is missing")
    raw_executors = value_policy.get("executors")
    if not isinstance(raw_executors, list):
        raise ValueError("value policy executor inventory is invalid")
    try:
        executors = tuple(OperationKind(value) for value in raw_executors)
    except (TypeError, ValueError) as exc:
        raise ValueError("value policy executor inventory is invalid") from exc
    if len(executors) != len(set(executors)):
        raise ValueError("value policy executor inventory contains duplicates")
    if value_policy.get("snapshot_sha256") != evidence_snapshot["snapshot_sha256"]:
        raise ValueError("diagnostic evidence differs from value policy")
    if not isinstance(action_trace, list) or not action_trace:
        raise ValueError("diagnostic selection requires action-state evidence")
    signals: list[Mapping[str, Any]] = []
    for row in action_trace:
        signal = row.get("state_signal") if isinstance(row, Mapping) else None
        if not isinstance(signal, Mapping):
            raise ValueError("diagnostic action-state evidence is invalid")
        signals.append(signal)

    def stable_flag(name: str) -> bool:
        values = {signal.get(name) for signal in signals}
        if not values <= {True, False} or len(values) != 1:
            raise ValueError(f"diagnostic capability {name} is inconsistent")
        return bool(next(iter(values)))

    has_memory = stable_flag("has_memory")
    has_evidence = stable_flag("has_evidence")
    has_verifier = stable_flag("has_verifier")
    has_savepoint = all(signal.get("has_savepoint") is True for signal in signals)
    available = set(executors)
    retrieve_operation = (
        OperationKind.RETRIEVE_EVIDENCE.value
        if OperationKind.RETRIEVE_EVIDENCE in available and has_evidence
        else OperationKind.SEARCH_MEMORY.value
        if OperationKind.SEARCH_MEMORY in available and has_memory
        else ""
    )
    return {
        "executors": [operation.value for operation in executors],
        "has_memory": has_memory,
        "has_context_evidence": has_evidence,
        "has_task_verifier": has_verifier,
        "has_prefix_savepoint": has_savepoint,
        "retrieve": bool(retrieve_operation),
        "retrieve_operation": retrieve_operation,
        "simulate": OperationKind.SIMULATE in available,
        "falsify": OperationKind.FALSIFY in available and has_verifier,
        "regenerate_from_prefix": (
            OperationKind.REGENERATE_FROM_PREFIX in available and has_savepoint
        ),
        "specialized_verifier": (
            OperationKind.CHECK_ASSUMPTION in available and has_verifier
        ),
    }


def _route_rows(
    pair: Mapping[str, Any],
    routes: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidate = pair["candidate_divergence"]
    if candidate.get("available") is not True:
        return []
    rows: list[dict[str, Any]] = []
    for side in ("left", "right"):
        atom = candidate.get(side)
        if not isinstance(atom, Mapping):
            continue
        branch = int(pair[side])
        route = routes.get(branch)
        if route is None:
            continue
        match = next(
            (row for row in route["routes"] if row["atom_id"] == atom["atom_id"]),
            None,
        )
        if match is not None:
            rows.append(
                {
                    "side": side,
                    "branch": branch,
                    "atom_id": atom["atom_id"],
                    "verifier": match["verifier"],
                    "outcome": match["outcome"],
                    "route_sha256": match["route_sha256"],
                }
            )
    return rows


def _applicability(
    method: str,
    *,
    dispute_kind: str,
    route_rows: list[dict[str, Any]],
    has_candidate: bool,
) -> tuple[float, str, bool]:
    exact = [
        row
        for row in route_rows
        if row["verifier"] in _EXACT_VERIFIERS
        and row["outcome"] in {"verified", "refuted"}
    ]
    exact_outcomes = {row["outcome"] for row in exact}
    if method == "execute" and "refuted" in exact_outcomes:
        return 1.0, "deterministic_route_already_executed", True
    routed_methods = {
        _ROUTE_METHOD[row["verifier"]]
        for row in route_rows
        if row["verifier"] in _ROUTE_METHOD
    }
    if method in routed_methods:
        return 0.98, "disputed_atom_requested_this_verifier_class", False
    if dispute_kind == "assumption":
        values = {
            "specialized_verifier": (0.95, "direct_assumption_test"),
            "falsify": (0.90, "counterexample_can_refute_assumption"),
            "regenerate_from_prefix": (0.78, "restart_before_assumption"),
            "simulate": (0.72, "latent_counterfactual_probe"),
            "prove": (0.70, "formalize_assumption_obligation"),
            "retrieve": (0.62, "inspect_admitted_evidence"),
        }
    elif dispute_kind == "dependency_transition":
        values = {
            "prove": (0.95, "derive_disputed_transition"),
            "specialized_verifier": (0.88, "verify_disputed_transition"),
            "regenerate_from_prefix": (0.84, "restart_before_transition"),
            "falsify": (0.78, "counterexample_to_transition"),
            "simulate": (0.75, "simulate_transition_consequence"),
            "retrieve": (0.60, "inspect_transition_evidence"),
        }
    elif dispute_kind == "claim":
        values = {
            "specialized_verifier": (0.90, "grade_disputed_claim"),
            "falsify": (0.84, "seek_claim_counterexample"),
            "retrieve": (0.74, "inspect_claim_evidence"),
            "simulate": (0.65, "simulate_claim_consequence"),
            "regenerate_from_prefix": (0.62, "restart_before_claim"),
        }
    else:
        values = {
            "regenerate_from_prefix": (0.90, "restart_before_causal_divergence"),
            "specialized_verifier": (0.82, "compare_divergent_programs"),
            "falsify": (0.76, "challenge_divergent_program"),
            "simulate": (0.70, "simulate_divergent_program"),
        }
    score, reason = values.get(method, (0.0, "not_applicable"))
    if not has_candidate and method in {"retrieve", "prove", "execute"}:
        return 0.0, "decoded_dispute_unavailable", False
    return score, reason, False


def build_diagnostic_action_selector_receipt(
    *,
    disagreement_graph: Any,
    candidate_routes: Any,
    action_policy_evidence: Any,
    value_policy: Any,
    action_trace: Any,
) -> dict[str, Any]:
    """Select one cheapest high-applicability diagnostic per branch pair."""

    if not isinstance(disagreement_graph, Mapping):
        raise ValueError("disagreement graph is missing")
    decompositions = disagreement_graph.get("candidate_decompositions")
    if not isinstance(decompositions, Mapping):
        raise ValueError("disagreement candidate decompositions are invalid")
    evidence = validate_evidence_snapshot(action_policy_evidence)
    routes = _validated_routes(candidate_routes, decompositions=decompositions)
    capabilities = _capabilities(
        value_policy=value_policy,
        action_trace=action_trace,
        evidence_snapshot=evidence,
    )
    plans: list[dict[str, Any]] = []
    for pair in disagreement_graph.get("pairwise", []):
        if not isinstance(pair, Mapping) or pair.get("localized") is not True:
            continue
        route_rows = _route_rows(pair, routes)
        candidate = pair["candidate_divergence"]
        has_candidate = candidate.get("available") is True
        dispute_kind = (
            str(candidate["kind"])
            if has_candidate
            else "causal_transition"
        )
        candidates: list[dict[str, Any]] = []
        for method, declared_operation in _METHOD_OPERATION.items():
            operation = (
                OperationKind(capabilities["retrieve_operation"])
                if method == "retrieve" and capabilities["retrieve_operation"]
                else declared_operation
            )
            score, reason, already_executed = _applicability(
                method,
                dispute_kind=dispute_kind,
                route_rows=route_rows,
                has_candidate=has_candidate,
            )
            available = (
                bool(already_executed)
                if method == "execute"
                else False
                if method == "prove"
                else bool(capabilities.get(method))
            )
            cost = (
                {
                    "action": operation.value,
                    "n": 1,
                    "measured": True,
                    "basis": "already_executed_deterministic_route",
                    "gain_basis": "deterministic_exact",
                    "gain_lower_bound": 1.0,
                    "cost_upper_bound": 0.0,
                    "evidence_snapshot_sha256": evidence["snapshot_sha256"],
                }
                if already_executed
                else action_cost_estimate(evidence, operation)
            )
            gain_lcb = cost["gain_lower_bound"]
            expected_score = (
                1.0
                if already_executed
                else score
                if gain_lcb is None
                else score * max(0.0, float(gain_lcb))
            )
            candidates.append(
                {
                    "method": method,
                    "operation_kind": operation.value,
                    "available": available,
                    "applicability": round(score, 6),
                    "applicability_basis": reason,
                    "expected_resolution_score": round(expected_score, 8),
                    "expected_resolution_basis": (
                        "deterministic_exact"
                        if already_executed
                        else "measured_verified_gain_lcb"
                        if gain_lcb is not None
                        else "preregistered_structural_bootstrap_prior"
                    ),
                    "already_executed": already_executed,
                    "cost": cost,
                }
            )
        feasible = [
            row
            for row in candidates
            if row["available"] and row["expected_resolution_score"] > 0.0
        ]
        if feasible:
            best_applicability = max(
                float(row["expected_resolution_score"]) for row in feasible
            )
            capable = [
                row
                for row in feasible
                if float(row["expected_resolution_score"])
                >= best_applicability - 0.05
            ]
            selected = min(
                capable,
                key=lambda row: (
                    float(row["cost"]["cost_upper_bound"]),
                    -float(row["expected_resolution_score"]),
                    str(row["method"]),
                ),
            )
            selected_summary = {
                "status": (
                    "resolved_by_existing_exact_route"
                    if selected["already_executed"]
                    else "diagnostic_operation_selected"
                ),
                "method": selected["method"],
                "operation_kind": selected["operation_kind"],
                "applicability": selected["applicability"],
                "applicability_basis": selected["applicability_basis"],
                "expected_resolution_score": selected[
                    "expected_resolution_score"
                ],
                "expected_resolution_basis": selected[
                    "expected_resolution_basis"
                ],
                "cost_upper_bound": selected["cost"]["cost_upper_bound"],
                "cost_basis": selected["cost"]["basis"],
                "already_executed": selected["already_executed"],
            }
        else:
            selected_summary = {
                "status": "no_admissible_diagnostic_operation",
                "method": "",
                "operation_kind": "",
                "applicability": 0.0,
                "applicability_basis": "no_available_capable_executor",
                "expected_resolution_score": 0.0,
                "expected_resolution_basis": "no_positive_admissible_evidence",
                "cost_upper_bound": None,
                "cost_basis": "unavailable",
                "already_executed": False,
            }
        plans.append(
            {
                "left": int(pair["left"]),
                "right": int(pair["right"]),
                "dispute_kind": dispute_kind,
                "dispute_sha256": _sha(
                    {
                        "causal": pair["causal_divergence"],
                        "candidate": pair["candidate_divergence"],
                    }
                ),
                "route_rows": route_rows,
                "candidates": candidates,
                "selected": selected_summary,
            }
        )
    payload = {
        "schema": DIAGNOSTIC_ACTION_SELECTOR_SCHEMA,
        "disagreement_graph_sha256": disagreement_graph.get("receipt_sha256"),
        "action_policy_evidence": evidence,
        "candidate_routes": {str(index): routes[index] for index in sorted(routes)},
        "capabilities": capabilities,
        "plans": plans,
        "localized_plan_count": len(plans),
        "selected_plan_count": sum(
            row["selected"]["status"] != "no_admissible_diagnostic_operation"
            for row in plans
        ),
        "authority": "diagnostic_recommendation_only",
        "branch_selection_effect": "none",
        "repair_effect": "none",
        "execution_effect": "none",
    }
    return {**payload, "receipt_sha256": _sha(payload)}


def validate_diagnostic_action_selector_receipt(
    value: Any,
    *,
    disagreement_graph: Any,
    value_policy: Any,
    action_trace: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("diagnostic action selector receipt is missing")
    expected = build_diagnostic_action_selector_receipt(
        disagreement_graph=disagreement_graph,
        candidate_routes=value.get("candidate_routes"),
        action_policy_evidence=value.get("action_policy_evidence"),
        value_policy=value_policy,
        action_trace=action_trace,
    )
    if dict(value) != expected:
        raise ValueError("diagnostic action selector differs from reconstruction")
    if (
        value.get("authority") != "diagnostic_recommendation_only"
        or value.get("branch_selection_effect") != "none"
        or value.get("repair_effect") != "none"
        or value.get("execution_effect") != "none"
    ):
        raise ValueError("diagnostic action selector exceeded its authority")
    return dict(value)


__all__ = [
    "DIAGNOSTIC_ACTION_SELECTOR_SCHEMA",
    "build_candidate_routes",
    "build_diagnostic_action_selector_receipt",
    "validate_diagnostic_action_selector_receipt",
]
