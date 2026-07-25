"""Bounded, source-private local repair for exactly refuted branch claims.

The worker owns candidate prose.  The service receives only atomic
decompositions, deterministic-verifier receipts, and commitments proving that
the prefix before a failed atom stayed unchanged.  A repaired candidate is
additive: it cannot replace the accepted answer until the separate confidence
gate grants that authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from core.brain.llm.latent_cortex.atomic_decomposition import (
    build_atomic_decomposition,
    validate_atomic_decomposition,
    validate_atomic_decomposition_envelope,
)
from core.brain.llm.latent_cortex.deterministic_verifier_router import (
    build_deterministic_router_receipt,
    validate_deterministic_router_envelope,
)

LOCAL_REPAIR_SCHEMA = "aura.rlc.local_repair.v1"
MAX_REPAIR_REQUESTS = 8
MAX_REPAIR_GENERATION_TOKENS = 576
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_EXACT_VERIFIERS = {"exact_integer_arithmetic", "python_ast", "json_parser"}
_ALLOWED_FAILURES = {
    "budget_unavailable",
    "generation_failed",
    "generation_contract_invalid",
    "replacement_invalid",
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


def _descendant_closure(
    decomposition: Mapping[str, Any],
    failed_atom_id: str,
) -> tuple[list[str], list[str]]:
    atoms = {str(row["atom_id"]) for row in decomposition["atoms"]}
    if failed_atom_id not in atoms:
        raise ValueError("failed repair atom is absent from its decomposition")
    descendants = {failed_atom_id}
    invalidated_transitions: set[str] = set()
    changed = True
    while changed:
        changed = False
        for transition in decomposition["transitions"]:
            premises = {str(value) for value in transition["premise_ids"]}
            if not premises & descendants:
                continue
            invalidated_transitions.add(str(transition["transition_id"]))
            output_id = str(transition["output_id"])
            if output_id not in descendants:
                descendants.add(output_id)
                changed = True
    ordered_atoms = [
        str(row["atom_id"])
        for row in decomposition["atoms"]
        if row["atom_id"] in descendants
    ]
    ordered_transitions = [
        str(row["transition_id"])
        for row in decomposition["transitions"]
        if row["transition_id"] in invalidated_transitions
    ]
    return ordered_atoms, ordered_transitions


def _route_inventory(
    selector: Mapping[str, Any],
    decompositions: Mapping[str, Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    raw = selector.get("candidate_routes")
    if not isinstance(raw, Mapping) or set(raw) != set(decompositions):
        raise ValueError("local repair route coverage differs")
    return {
        int(index): validate_deterministic_router_envelope(
            raw[index],
            atomic_receipt=decompositions[index],
        )
        for index in sorted(raw, key=int)
    }


def _repair_requests(
    *,
    disagreement_graph: Any,
    diagnostic_selection: Any,
    max_requests: int,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    if not isinstance(disagreement_graph, Mapping):
        raise ValueError("local repair disagreement graph is missing")
    if not isinstance(diagnostic_selection, Mapping):
        raise ValueError("local repair diagnostic selection is missing")
    if (
        type(max_requests) is not int
        or not 0 <= max_requests <= MAX_REPAIR_REQUESTS
    ):
        raise ValueError("local repair request budget is invalid")
    graph_sha = disagreement_graph.get("receipt_sha256")
    selector_sha = diagnostic_selection.get("receipt_sha256")
    if (
        not isinstance(graph_sha, str)
        or _SHA256_RE.fullmatch(graph_sha) is None
        or not isinstance(selector_sha, str)
        or _SHA256_RE.fullmatch(selector_sha) is None
        or diagnostic_selection.get("disagreement_graph_sha256") != graph_sha
    ):
        raise ValueError("local repair upstream binding is invalid")
    raw_decompositions = disagreement_graph.get("candidate_decompositions")
    if not isinstance(raw_decompositions, Mapping):
        raise ValueError("local repair candidate inventory is invalid")
    decompositions = {
        str(index): validate_atomic_decomposition_envelope(value)
        for index, value in raw_decompositions.items()
    }
    routes = _route_inventory(diagnostic_selection, decompositions)
    plans = diagnostic_selection.get("plans")
    if not isinstance(plans, list):
        raise ValueError("local repair diagnostic plans are invalid")

    repair_requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for plan in plans:
        if not isinstance(plan, Mapping):
            raise ValueError("local repair diagnostic plan is invalid")
        selected = plan.get("selected")
        route_rows = plan.get("route_rows")
        if not isinstance(selected, Mapping) or not isinstance(route_rows, list):
            raise ValueError("local repair diagnostic plan is incomplete")
        if (
            selected.get("status") != "resolved_by_existing_exact_route"
            or selected.get("already_executed") is not True
            or selected.get("method") != "execute"
        ):
            continue
        for route_row in route_rows:
            if (
                not isinstance(route_row, Mapping)
                or route_row.get("outcome") != "refuted"
                or route_row.get("verifier") not in _EXACT_VERIFIERS
            ):
                continue
            branch = route_row.get("branch")
            atom_id = route_row.get("atom_id")
            if (
                type(branch) is not int
                or branch not in routes
                or not isinstance(atom_id, str)
            ):
                raise ValueError("local repair refutation binding is invalid")
            route = next(
                (
                    row
                    for row in routes[branch]["routes"]
                    if row["atom_id"] == atom_id
                ),
                None,
            )
            if (
                route is None
                or route["route_sha256"] != route_row.get("route_sha256")
                or route["verifier"] != route_row.get("verifier")
                or route["outcome"] != "refuted"
            ):
                raise ValueError("local repair route differs from primary evidence")
            decomposition = decompositions[str(branch)]
            atom_index = next(
                (
                    index
                    for index, atom in enumerate(decomposition["atoms"])
                    if atom["atom_id"] == atom_id
                ),
                None,
            )
            if atom_index is None:
                raise ValueError("local repair atom differs from primary evidence")
            invalidated_atoms, invalidated_transitions = _descendant_closure(
                decomposition,
                atom_id,
            )
            preserved_atoms = [
                {
                    "atom_id": atom["atom_id"],
                    "atom_sha256": atom["atom_sha256"],
                    "text_sha256": atom["text_sha256"],
                }
                for atom in decomposition["atoms"][:atom_index]
            ]
            verified_ancestor_routes = [
                {
                    "atom_id": row["atom_id"],
                    "verifier": row["verifier"],
                    "route_sha256": row["route_sha256"],
                }
                for row in routes[branch]["routes"][:atom_index]
                if row["verifier"] in _EXACT_VERIFIERS
                and row["outcome"] == "verified"
            ]
            invalidated_set = set(invalidated_atoms)
            preserved_unrelated_atoms = [
                {
                    "ordinal": index,
                    "atom_id": atom["atom_id"],
                    "kind": atom["kind"],
                    "text_sha256": atom["text_sha256"],
                    "dependency_cues": list(atom["dependency_cues"]),
                }
                for index, atom in enumerate(decomposition["atoms"])
                if atom["atom_id"] not in invalidated_set
            ]
            request_payload = {
                "pair": {
                    "left": int(plan["left"]),
                    "right": int(plan["right"]),
                },
                "dispute_sha256": str(plan["dispute_sha256"]),
                "branch": branch,
                "failed_atom_id": atom_id,
                "failed_atom_ordinal": atom_index,
                "failed_route_sha256": route["route_sha256"],
                "required_verifier": route["verifier"],
                "original_decomposition_sha256": decomposition["receipt_sha256"],
                "original_source_sha256": decomposition["source_sha256"],
                "preserved_prefix_atoms": preserved_atoms,
                "verified_ancestor_routes": verified_ancestor_routes,
                "preserved_unrelated_atoms": preserved_unrelated_atoms,
                "last_valid_atom_id": (
                    str(decomposition["atoms"][atom_index - 1]["atom_id"])
                    if atom_index
                    else ""
                ),
                "invalidated_atom_ids": invalidated_atoms,
                "invalidated_transition_ids": invalidated_transitions,
            }
            request_id = _sha(request_payload)
            if request_id in seen:
                raise ValueError("local repair request is duplicated")
            seen.add(request_id)
            repair_requests.append({**request_payload, "request_id": request_id})
    repair_requests.sort(
        key=lambda row: (
            row["branch"],
            row["failed_atom_ordinal"],
            row["request_id"],
        )
    )
    return repair_requests[:max_requests], routes


def _request_prefix(
    request: Mapping[str, Any],
    decomposition: Mapping[str, Any],
    candidate: str,
) -> str:
    ordinal = int(request["failed_atom_ordinal"])
    if ordinal >= len(decomposition["atoms"]):
        raise ValueError("local repair frontier is outside its candidate")
    start = int(decomposition["atoms"][ordinal]["start"])
    return candidate[:start]


def prepare_local_repair_requests(
    *,
    disagreement_graph: Any,
    diagnostic_selection: Any,
    branch_candidates: Mapping[int, str],
    objective: str,
    max_requests: int = 1,
) -> list[dict[str, Any]]:
    """Return bounded worker-private prompts for exact-refutation repairs."""

    if not isinstance(branch_candidates, Mapping) or not isinstance(objective, str):
        raise TypeError("local repair private sources are invalid")
    repair_requests, routes = _repair_requests(
        disagreement_graph=disagreement_graph,
        diagnostic_selection=diagnostic_selection,
        max_requests=max_requests,
    )
    decompositions = disagreement_graph["candidate_decompositions"]
    prepared: list[dict[str, Any]] = []
    for request in repair_requests:
        branch = int(request["branch"])
        candidate = branch_candidates.get(branch)
        if not isinstance(candidate, str):
            raise ValueError("local repair candidate source is missing")
        decomposition = validate_atomic_decomposition(
            decompositions[str(branch)],
            candidate=candidate,
            objective=objective,
        )
        route = next(
            row
            for row in routes[branch]["routes"]
            if row["route_sha256"] == request["failed_route_sha256"]
        )
        prefix = _request_prefix(request, decomposition, candidate)
        prompt = (
            "Repair only the invalid suffix of a candidate answer. Preserve the "
            "provided prefix byte-for-byte and replace the failed claim plus every "
            "dependent claim. Do not repeat the prefix. The exact verifier "
            f"{request['required_verifier']} refuted the failed claim with evidence "
            f"{json.dumps(route['detail'], sort_keys=True, ensure_ascii=True)}. "
            "Every original atom outside the invalidation set must remain "
            "byte-identical and in the same order. "
            f"Objective: {objective}\n"
            f"INVALIDATED_ATOM_IDS: {request['invalidated_atom_ids']}\n"
            f"ORIGINAL_CANDIDATE:\n{candidate}\n"
            f"PRESERVED_PREFIX:\n{prefix}\n"
            'Return exactly FINAL_ANSWER: {"replacement_suffix":"..."} with no '
            "other keys or trailing text."
        )
        prepared.append(
            {
                **request,
                "prefix": prefix,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
    return prepared


def parse_local_repair_generation(value: Any, *, prefix: str) -> str:
    """Parse a fresh-context repair response and restore its preserved prefix."""

    if not isinstance(value, str) or not isinstance(prefix, str):
        raise TypeError("local repair generation must be text")
    marker = "FINAL_ANSWER:"
    if value.count(marker) != 1:
        raise ValueError("local repair generation contract marker is invalid")
    before, encoded = value.split(marker, 1)
    if before.strip() or not encoded.strip():
        raise ValueError("local repair generation contains extra material")
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError("local repair generation JSON is invalid") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"replacement_suffix"}
        or not isinstance(payload["replacement_suffix"], str)
        or not payload["replacement_suffix"].strip()
        or len(payload["replacement_suffix"]) > 32_768
    ):
        raise ValueError("local repair generation payload is invalid")
    return prefix + payload["replacement_suffix"]


def _validate_generation_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("local repair generation context is missing")
    required = {
        "prompt_sha256",
        "generated_token_count",
        "termination",
        "initial_cache_offsets",
        "final_cache_offsets",
        "all_initial_offsets_zero",
        "solver_context_imported",
        "parameter_relation",
    }
    if set(value) != required:
        raise ValueError("local repair generation context fields differ")
    initial = value["initial_cache_offsets"]
    final = value["final_cache_offsets"]
    if (
        _SHA256_RE.fullmatch(str(value["prompt_sha256"])) is None
        or type(value["generated_token_count"]) is not int
        or not 1 <= value["generated_token_count"] <= MAX_REPAIR_GENERATION_TOKENS
        or value["termination"] != "contract_complete"
        or not isinstance(initial, list)
        or not initial
        or len(initial) > 256
        or any(type(offset) is not int or offset != 0 for offset in initial)
        or not isinstance(final, list)
        or len(final) != len(initial)
        or any(type(offset) is not int or offset <= 0 for offset in final)
        or value["all_initial_offsets_zero"] is not True
        or value["solver_context_imported"] is not False
        or value["parameter_relation"] != "shared_resident_checkpoint"
    ):
        raise ValueError("local repair generation context is invalid")
    return dict(value)


def _admitted_transaction(
    *,
    request: Mapping[str, Any],
    candidate: str,
    objective: str,
    generation_context: Any,
) -> dict[str, Any]:
    context = _validate_generation_context(generation_context)
    if context["prompt_sha256"] != request["prompt_sha256"]:
        raise ValueError("local repair generation used a different prompt")
    decomposition = build_atomic_decomposition(candidate, objective=objective)
    route = build_deterministic_router_receipt(
        candidate,
        objective=objective,
        atomic_receipt=decomposition,
    )
    prefix_count = len(request["preserved_prefix_atoms"])
    observed_prefix = [
        {
            "atom_id": atom["atom_id"],
            "atom_sha256": atom["atom_sha256"],
            "text_sha256": atom["text_sha256"],
        }
        for atom in decomposition["atoms"][:prefix_count]
    ]
    prefix_unchanged = observed_prefix == request["preserved_prefix_atoms"]
    unrelated_unchanged = all(
        preserved["ordinal"] < len(decomposition["atoms"])
        and {
            "ordinal": preserved["ordinal"],
            "atom_id": decomposition["atoms"][preserved["ordinal"]]["atom_id"],
            "kind": decomposition["atoms"][preserved["ordinal"]]["kind"],
            "text_sha256": decomposition["atoms"][preserved["ordinal"]][
                "text_sha256"
            ],
            "dependency_cues": list(
                decomposition["atoms"][preserved["ordinal"]]["dependency_cues"]
            ),
        }
        == preserved
        for preserved in request["preserved_unrelated_atoms"]
    )
    ordinal = int(request["failed_atom_ordinal"])
    replacement_route = route["routes"][ordinal] if ordinal < len(route["routes"]) else None
    failed_verifier_rechecked = bool(
        replacement_route
        and replacement_route["verifier"] == request["required_verifier"]
    )
    failed_verifier_passed = bool(
        failed_verifier_rechecked and replacement_route["outcome"] == "verified"
    )
    no_exact_refutations = not any(
        row["verifier"] in _EXACT_VERIFIERS and row["outcome"] == "refuted"
        for row in route["routes"]
    )
    admitted = (
        prefix_unchanged
        and unrelated_unchanged
        and failed_verifier_passed
        and no_exact_refutations
        and decomposition["grade_admissible"] is True
    )
    reason = (
        "exact_refutation_removed_with_prefix_preserved"
        if admitted
        else "preserved_prefix_changed"
        if not prefix_unchanged
        else "unrelated_atom_changed"
        if not unrelated_unchanged
        else "failed_verifier_not_rechecked"
        if not failed_verifier_rechecked
        else "failed_verifier_still_not_verified"
        if not failed_verifier_passed
        else "another_exact_refutation_remains"
        if not no_exact_refutations
        else "replacement_decomposition_not_admissible"
    )
    payload = {
        "request_id": request["request_id"],
        "status": (
            "repaired_candidate_admitted"
            if admitted
            else "repaired_candidate_rejected"
        ),
        "reason": reason,
        "generation_context": context,
        "replacement_decomposition": decomposition,
        "replacement_routes": route,
        "preserved_prefix_unchanged": prefix_unchanged,
        "unrelated_work_unchanged": unrelated_unchanged,
        "failed_verifier_rechecked": failed_verifier_rechecked,
        "failed_verifier_passed": failed_verifier_passed,
        "no_exact_refutations": no_exact_refutations,
        "invalidated_atom_ids": list(request["invalidated_atom_ids"]),
        "invalidated_transition_ids": list(request["invalidated_transition_ids"]),
        "repair_candidate_effect": "candidate_pool_addition" if admitted else "none",
        "answer_selection_effect": "none",
        "latent_state_effect": "none",
    }
    return {**payload, "transaction_sha256": _sha(payload)}


def build_local_repair_receipt(
    *,
    disagreement_graph: Any,
    diagnostic_selection: Any,
    branch_candidates: Mapping[int, str],
    objective: str,
    generated_repairs: Mapping[str, Mapping[str, Any]] | None = None,
    execution_failures: Mapping[str, str] | None = None,
    max_requests: int = 1,
) -> dict[str, Any]:
    """Build a text-free receipt for bounded local repair attempts."""

    prepared = prepare_local_repair_requests(
        disagreement_graph=disagreement_graph,
        diagnostic_selection=diagnostic_selection,
        branch_candidates=branch_candidates,
        objective=objective,
        max_requests=max_requests,
    )
    generated = dict(generated_repairs or {})
    failures = dict(execution_failures or {})
    request_ids = {row["request_id"] for row in prepared}
    if set(generated) - request_ids or set(failures) - request_ids:
        raise ValueError("local repair result names an unknown request")
    if set(generated) & set(failures):
        raise ValueError("local repair request has conflicting outcomes")
    if any(reason not in _ALLOWED_FAILURES for reason in failures.values()):
        raise ValueError("local repair failure reason is invalid")
    transactions: list[dict[str, Any]] = []
    for request in prepared:
        request_id = request["request_id"]
        if request_id in generated:
            result = generated[request_id]
            if not isinstance(result, Mapping) or set(result) != {
                "candidate",
                "generation_context",
            }:
                raise ValueError("local repair generated result is invalid")
            candidate = result["candidate"]
            if not isinstance(candidate, str):
                raise ValueError("local repair generated candidate is invalid")
            transaction = _admitted_transaction(
                request=request,
                candidate=candidate,
                objective=objective,
                generation_context=result["generation_context"],
            )
        else:
            reason = failures.get(request_id, "generation_failed")
            transaction_payload = {
                "request_id": request_id,
                "status": "repair_not_executed",
                "reason": reason,
                "generation_context": {},
                "replacement_decomposition": {},
                "replacement_routes": {},
                "preserved_prefix_unchanged": False,
                "unrelated_work_unchanged": False,
                "failed_verifier_rechecked": False,
                "failed_verifier_passed": False,
                "no_exact_refutations": False,
                "invalidated_atom_ids": list(request["invalidated_atom_ids"]),
                "invalidated_transition_ids": list(
                    request["invalidated_transition_ids"]
                ),
                "repair_candidate_effect": "none",
                "answer_selection_effect": "none",
                "latent_state_effect": "none",
            }
            transaction = {
                **transaction_payload,
                "transaction_sha256": _sha(transaction_payload),
            }
        transactions.append(transaction)
    public_requests = [
        {key: value for key, value in request.items() if key not in {"prefix", "prompt"}}
        for request in prepared
    ]
    branches = [dict(row) for row in disagreement_graph.get("branches", [])]
    admitted = sum(
        row["status"] == "repaired_candidate_admitted" for row in transactions
    )
    payload = {
        "schema": LOCAL_REPAIR_SCHEMA,
        "disagreement_graph_sha256": disagreement_graph.get("receipt_sha256"),
        "diagnostic_selection_sha256": diagnostic_selection.get("receipt_sha256"),
        "max_requests": max_requests,
        "requests": public_requests,
        "transactions": transactions,
        "request_count": len(public_requests),
        "attempted_count": len(generated),
        "admitted_count": admitted,
        "original_branch_commitments_before": branches,
        "original_branch_commitments_after": branches,
        "unrelated_original_work_unchanged": True,
        "worker_source_boundary": (
            "worker_reconstructs_private_source_service_validates_commitments"
        ),
        "authority": "bounded_epistemic_candidate_repair",
        "repair_effect": "candidate_pool_addition" if admitted else "none",
        "answer_selection_effect": "none",
        "accepted_answer_effect": "none",
        "latent_state_effect": "none",
    }
    return {**payload, "receipt_sha256": _sha(payload)}


def validate_local_repair_receipt(
    value: Any,
    *,
    disagreement_graph: Any,
    diagnostic_selection: Any,
) -> dict[str, Any]:
    """Independently reconstruct invalidation and replacement admission."""

    if not isinstance(value, Mapping):
        raise ValueError("local repair receipt is missing")
    fields = {
        "schema",
        "disagreement_graph_sha256",
        "diagnostic_selection_sha256",
        "max_requests",
        "requests",
        "transactions",
        "request_count",
        "attempted_count",
        "admitted_count",
        "original_branch_commitments_before",
        "original_branch_commitments_after",
        "unrelated_original_work_unchanged",
        "worker_source_boundary",
        "authority",
        "repair_effect",
        "answer_selection_effect",
        "accepted_answer_effect",
        "latent_state_effect",
        "receipt_sha256",
    }
    if set(value) != fields:
        raise ValueError("local repair receipt fields differ")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != _sha(payload):
        raise ValueError("local repair receipt commitment mismatch")
    expected_requests, _routes = _repair_requests(
        disagreement_graph=disagreement_graph,
        diagnostic_selection=diagnostic_selection,
        max_requests=value["max_requests"],
    )
    repair_requests = value["requests"]
    if not isinstance(repair_requests, list) or len(repair_requests) != len(
        expected_requests
    ):
        raise ValueError("local repair request coverage differs")
    public_expected: list[dict[str, Any]] = []
    for request, request_row in zip(
        expected_requests,
        repair_requests,
        strict=True,
    ):
        if (
            not isinstance(request_row, Mapping)
            or set(request_row) != set(request) | {"prompt_sha256"}
            or _SHA256_RE.fullmatch(str(request_row.get("prompt_sha256"))) is None
        ):
            raise ValueError("local repair request fields differ")
        public_expected.append(
            {**request, "prompt_sha256": request_row["prompt_sha256"]}
        )
    if (
        value["schema"] != LOCAL_REPAIR_SCHEMA
        or repair_requests != public_expected
    ):
        raise ValueError("local repair request reconstruction differs")
    transactions = value["transactions"]
    if not isinstance(transactions, list) or len(transactions) != len(
        repair_requests
    ):
        raise ValueError("local repair transaction coverage differs")
    admitted = 0
    attempted = 0
    for request, transaction in zip(
        repair_requests,
        transactions,
        strict=True,
    ):
        if not isinstance(transaction, Mapping):
            raise ValueError("local repair transaction is invalid")
        transaction_fields = {
            "request_id",
            "status",
            "reason",
            "generation_context",
            "replacement_decomposition",
            "replacement_routes",
            "preserved_prefix_unchanged",
            "unrelated_work_unchanged",
            "failed_verifier_rechecked",
            "failed_verifier_passed",
            "no_exact_refutations",
            "invalidated_atom_ids",
            "invalidated_transition_ids",
            "repair_candidate_effect",
            "answer_selection_effect",
            "latent_state_effect",
            "transaction_sha256",
        }
        if set(transaction) != transaction_fields:
            raise ValueError("local repair transaction fields differ")
        transaction_payload = {
            key: transaction[key] for key in transaction_fields - {"transaction_sha256"}
        }
        if (
            transaction["transaction_sha256"] != _sha(transaction_payload)
            or transaction["request_id"] != request["request_id"]
            or transaction["invalidated_atom_ids"] != request["invalidated_atom_ids"]
            or transaction["invalidated_transition_ids"]
            != request["invalidated_transition_ids"]
            or transaction["answer_selection_effect"] != "none"
            or transaction["latent_state_effect"] != "none"
        ):
            raise ValueError("local repair transaction binding is invalid")
        if transaction["status"] == "repair_not_executed":
            if (
                transaction["reason"] not in _ALLOWED_FAILURES
                or transaction["generation_context"] != {}
                or transaction["replacement_decomposition"] != {}
                or transaction["replacement_routes"] != {}
                or transaction["preserved_prefix_unchanged"] is not False
                or transaction["unrelated_work_unchanged"] is not False
                or transaction["failed_verifier_rechecked"] is not False
                or transaction["failed_verifier_passed"] is not False
                or transaction["no_exact_refutations"] is not False
                or transaction["repair_candidate_effect"] != "none"
            ):
                raise ValueError("local repair non-execution claim is invalid")
            continue
        attempted += 1
        context = _validate_generation_context(transaction["generation_context"])
        if context["prompt_sha256"] != request["prompt_sha256"]:
            raise ValueError("local repair generation prompt binding differs")
        decomposition = validate_atomic_decomposition_envelope(
            transaction["replacement_decomposition"]
        )
        routes = validate_deterministic_router_envelope(
            transaction["replacement_routes"],
            atomic_receipt=decomposition,
        )
        prefix_count = len(request["preserved_prefix_atoms"])
        observed_prefix = [
            {
                "atom_id": atom["atom_id"],
                "atom_sha256": atom["atom_sha256"],
                "text_sha256": atom["text_sha256"],
            }
            for atom in decomposition["atoms"][:prefix_count]
        ]
        prefix_unchanged = observed_prefix == request["preserved_prefix_atoms"]
        unrelated_unchanged = all(
            preserved["ordinal"] < len(decomposition["atoms"])
            and {
                "ordinal": preserved["ordinal"],
                "atom_id": decomposition["atoms"][preserved["ordinal"]][
                    "atom_id"
                ],
                "kind": decomposition["atoms"][preserved["ordinal"]]["kind"],
                "text_sha256": decomposition["atoms"][preserved["ordinal"]][
                    "text_sha256"
                ],
                "dependency_cues": list(
                    decomposition["atoms"][preserved["ordinal"]][
                        "dependency_cues"
                    ]
                ),
            }
            == preserved
            for preserved in request["preserved_unrelated_atoms"]
        )
        ordinal = int(request["failed_atom_ordinal"])
        replacement_route = (
            routes["routes"][ordinal] if ordinal < len(routes["routes"]) else None
        )
        rechecked = bool(
            replacement_route
            and replacement_route["verifier"] == request["required_verifier"]
        )
        passed = bool(rechecked and replacement_route["outcome"] == "verified")
        no_refutations = not any(
            row["verifier"] in _EXACT_VERIFIERS and row["outcome"] == "refuted"
            for row in routes["routes"]
        )
        should_admit = (
            prefix_unchanged
            and unrelated_unchanged
            and passed
            and no_refutations
            and decomposition["grade_admissible"] is True
        )
        expected_reason = (
            "exact_refutation_removed_with_prefix_preserved"
            if should_admit
            else "preserved_prefix_changed"
            if not prefix_unchanged
            else "unrelated_atom_changed"
            if not unrelated_unchanged
            else "failed_verifier_not_rechecked"
            if not rechecked
            else "failed_verifier_still_not_verified"
            if not passed
            else "another_exact_refutation_remains"
            if not no_refutations
            else "replacement_decomposition_not_admissible"
        )
        if (
            context != transaction["generation_context"]
            or transaction["status"]
            != (
                "repaired_candidate_admitted"
                if should_admit
                else "repaired_candidate_rejected"
            )
            or transaction["reason"] != expected_reason
            or transaction["preserved_prefix_unchanged"] is not prefix_unchanged
            or transaction["unrelated_work_unchanged"] is not unrelated_unchanged
            or transaction["failed_verifier_rechecked"] is not rechecked
            or transaction["failed_verifier_passed"] is not passed
            or transaction["no_exact_refutations"] is not no_refutations
            or transaction["repair_candidate_effect"]
            != ("candidate_pool_addition" if should_admit else "none")
        ):
            raise ValueError("local repair admission differs from reconstruction")
        admitted += int(should_admit)
    branches = [dict(row) for row in disagreement_graph.get("branches", [])]
    if (
        value["request_count"] != len(repair_requests)
        or value["attempted_count"] != attempted
        or value["admitted_count"] != admitted
        or value["original_branch_commitments_before"] != branches
        or value["original_branch_commitments_after"] != branches
        or value["unrelated_original_work_unchanged"] is not True
        or value["worker_source_boundary"]
        != "worker_reconstructs_private_source_service_validates_commitments"
        or value["authority"] != "bounded_epistemic_candidate_repair"
        or value["repair_effect"]
        != ("candidate_pool_addition" if admitted else "none")
        or value["answer_selection_effect"] != "none"
        or value["accepted_answer_effect"] != "none"
        or value["latent_state_effect"] != "none"
    ):
        raise ValueError("local repair summary or authority is invalid")
    return dict(value)


__all__ = [
    "LOCAL_REPAIR_SCHEMA",
    "MAX_REPAIR_GENERATION_TOKENS",
    "MAX_REPAIR_REQUESTS",
    "build_local_repair_receipt",
    "parse_local_repair_generation",
    "prepare_local_repair_requests",
    "validate_local_repair_receipt",
]
