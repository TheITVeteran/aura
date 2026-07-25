"""Confidence-bound authority for promoting a locally repaired answer.

The authority object is deliberately narrow: full-span validity of atomic
claims handled by semantic deterministic verifiers. Syntax-only parsers can
refute malformed code/JSON, but cannot certify semantic correctness. Unknown
or partially covered prose keeps the interval at [0, 1].

Private candidate text crosses the worker/service IPC boundary only long enough
for the service to rebuild decomposition and verifier evidence. Public receipts
contain commitments, intervals, and decisions, never hidden candidate prose.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from core.brain.llm.latent_cortex.atomic_decomposition import (
    build_atomic_decomposition,
    validate_atomic_decomposition_envelope,
)
from core.brain.llm.latent_cortex.deterministic_verifier_router import (
    build_deterministic_router_receipt,
    validate_deterministic_router_envelope,
)
from core.brain.llm.latent_cortex.local_repair import (
    validate_local_repair_receipt,
)

ANSWER_REPLACEMENT_SCHEMA = "aura.rlc.answer_replacement.v2"
ANSWER_REPLACEMENT_PRIVATE_SCHEMA = "aura.rlc.answer_replacement_private.v2"
DEFAULT_REPLACEMENT_MARGIN = 0.05
MAX_REPLACEMENT_OUTPUT_TOKENS = 1024
_REFUTATION_VERIFIERS = {"exact_integer_arithmetic", "python_ast", "json_parser"}
_SEMANTIC_EXACT_VERIFIERS = {"exact_integer_arithmetic"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FULL_INTEGER_ARITHMETIC_RE = re.compile(
    r"\s*-?\d{1,12}\s*[+\-*/x×]\s*-?\d{1,12}\s*=\s*-?\d{1,12}\s*[.!?]?\s*\Z"
)


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


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _token_sha(tokens: Sequence[int]) -> str:
    return _sha(list(tokens))


def _margin(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) < 1.0
    ):
        raise ValueError("answer replacement margin must be finite in [0, 1)")
    return round(float(value), 10)


def _output_limit(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= MAX_REPLACEMENT_OUTPUT_TOKENS:
        raise ValueError("answer replacement output limit is invalid")
    return value


def _quality_interval(
    decomposition: Mapping[str, Any],
    routes: Mapping[str, Any],
    *,
    candidate: str,
) -> dict[str, Any]:
    atomic = validate_atomic_decomposition_envelope(decomposition)
    routed = validate_deterministic_router_envelope(
        routes,
        atomic_receipt=atomic,
    )
    if _text_sha(candidate) != atomic["source_sha256"]:
        raise ValueError("answer replacement candidate source differs")
    refuted = [
        row
        for row in routed["routes"]
        if row["verifier"] in _REFUTATION_VERIFIERS
        and row["outcome"] == "refuted"
    ]
    semantic_verified = 0
    partial_or_nonsemantic = 0
    for atom, route in zip(atomic["atoms"], routed["routes"], strict=True):
        fragment = candidate[int(atom["start"]) : int(atom["end"])]
        full_span_semantic = bool(
            route["verifier"] in _SEMANTIC_EXACT_VERIFIERS
            and route["outcome"] == "verified"
            and _FULL_INTEGER_ARITHMETIC_RE.fullmatch(fragment)
        )
        semantic_verified += int(full_span_semantic)
        partial_or_nonsemantic += int(not full_span_semantic)
    every_atom_semantically_verified = bool(atomic["atoms"]) and (
        semantic_verified == len(atomic["atoms"])
    )
    if refuted or atomic["grade_admissible"] is not True:
        lower = upper = 0.0
        basis = (
            "deterministic_exact_refutation"
            if refuted
            else "structural_grade_refutation"
        )
    elif every_atom_semantically_verified:
        lower = upper = 1.0
        basis = "full_span_semantic_exact_complete"
    else:
        lower, upper = 0.0, 1.0
        basis = "incomplete_semantic_exact_coverage"
    payload = {
        "object": "conjunctive_full_span_exact_claim_validity",
        "lower_bound": lower,
        "upper_bound": upper,
        "basis": basis,
        "atom_count": len(atomic["atoms"]),
        "semantic_exact_verified_count": semantic_verified,
        "exact_refuted_count": len(refuted),
        "partial_or_nonsemantic_count": partial_or_nonsemantic,
        "decomposition_sha256": atomic["receipt_sha256"],
        "routes_sha256": routed["receipt_sha256"],
    }
    return {**payload, "interval_sha256": _sha(payload)}


def _normalize_private_evidence(value: Any) -> dict[str, Any]:
    fields = {
        "schema",
        "objective",
        "branch_candidates",
        "generated_repairs",
        "baseline_text",
        "baseline_tokens",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("answer replacement private evidence fields differ")
    branches = value["branch_candidates"]
    repairs = value["generated_repairs"]
    baseline_tokens = value["baseline_tokens"]
    if (
        value["schema"] != ANSWER_REPLACEMENT_PRIVATE_SCHEMA
        or not isinstance(value["objective"], str)
        or not isinstance(value["baseline_text"], str)
        or not isinstance(baseline_tokens, list)
        or len(baseline_tokens) > MAX_REPLACEMENT_OUTPUT_TOKENS
        or any(type(token) is not int or token < 0 for token in baseline_tokens)
        or not isinstance(branches, Mapping)
        or len(branches) > 64
        or any(
            not isinstance(key, str)
            or not key.isdigit()
            or not isinstance(text, str)
            or len(text) > 131_072
            for key, text in branches.items()
        )
        or not isinstance(repairs, Mapping)
        or len(repairs) > 8
        or any(
            _SHA256_RE.fullmatch(str(key)) is None
            or not isinstance(text, str)
            or len(text) > 131_072
            for key, text in repairs.items()
        )
    ):
        raise ValueError("answer replacement private evidence is invalid")
    return {
        "schema": ANSWER_REPLACEMENT_PRIVATE_SCHEMA,
        "objective": value["objective"],
        "branch_candidates": {
            str(key): str(text) for key, text in sorted(branches.items())
        },
        "generated_repairs": {
            str(key): str(text) for key, text in sorted(repairs.items())
        },
        "baseline_text": value["baseline_text"],
        "baseline_tokens": list(baseline_tokens),
    }


def _candidate_inventory(
    *,
    disagreement_graph: Mapping[str, Any],
    diagnostic_selection: Mapping[str, Any],
    local_repair: Mapping[str, Any],
    private_evidence: Mapping[str, Any],
    selected_branch: int,
    baseline_quality: Mapping[str, Any],
    margin: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decompositions = disagreement_graph.get("candidate_decompositions")
    candidate_routes = diagnostic_selection.get("candidate_routes")
    if not isinstance(decompositions, Mapping) or not isinstance(
        candidate_routes,
        Mapping,
    ):
        raise ValueError("answer replacement candidate inventory is missing")
    branch_texts = private_evidence["branch_candidates"]
    if set(branch_texts) != set(decompositions):
        raise ValueError("answer replacement private branch coverage differs")
    branch_quality: dict[int, dict[str, Any]] = {}
    for index in sorted(decompositions, key=int):
        text = branch_texts[index]
        branch_quality[int(index)] = _quality_interval(
            decompositions[index],
            candidate_routes[index],
            candidate=text,
        )
    if selected_branch in branch_quality:
        selected_quality = branch_quality[selected_branch]
    else:
        unavailable_payload = {
            "object": "conjunctive_full_span_exact_claim_validity",
            "lower_bound": 0.0,
            "upper_bound": 1.0,
            "basis": "candidate_probe_unavailable",
            "atom_count": 0,
            "semantic_exact_verified_count": 0,
            "exact_refuted_count": 0,
            "partial_or_nonsemantic_count": 0,
            "decomposition_sha256": "",
            "routes_sha256": "",
        }
        selected_quality = {
            **unavailable_payload,
            "interval_sha256": _sha(unavailable_payload),
        }

    repairs = private_evidence["generated_repairs"]
    rows: list[dict[str, Any]] = []
    for request, transaction in zip(
        local_repair["requests"],
        local_repair["transactions"],
        strict=True,
    ):
        branch = int(request["branch"])
        replacement_available = (
            transaction["status"] == "repaired_candidate_admitted"
        )
        replacement_text = repairs.get(request["request_id"])
        if replacement_available and not isinstance(replacement_text, str):
            raise ValueError("admitted replacement private source is absent")
        if not replacement_available and replacement_text is not None:
            raise ValueError("rejected replacement retained private authority")
        replacement_quality = (
            _quality_interval(
                transaction["replacement_decomposition"],
                transaction["replacement_routes"],
                candidate=replacement_text,
            )
            if replacement_available
            else None
        )
        original_routes = validate_deterministic_router_envelope(
            candidate_routes[str(branch)],
            atomic_receipt=decompositions[str(branch)],
        )
        failed_ordinal = int(request["failed_atom_ordinal"])
        original_failed_route = original_routes["routes"][failed_ordinal]
        replacement_failed_route = (
            transaction["replacement_routes"]["routes"][failed_ordinal]
            if replacement_available
            else None
        )
        same_verifier_class = bool(
            replacement_failed_route
            and original_failed_route["verifier"] == request["required_verifier"]
            and replacement_failed_route["verifier"] == request["required_verifier"]
            and original_failed_route["outcome"] == "refuted"
            and replacement_failed_route["outcome"] == "verified"
        )
        dominates = bool(
            branch == selected_branch
            and replacement_quality is not None
            and same_verifier_class
            and float(replacement_quality["lower_bound"])
            > float(baseline_quality["upper_bound"]) + margin
        )
        payload = {
            "request_id": request["request_id"],
            "branch": branch,
            "transaction_sha256": transaction["transaction_sha256"],
            "transaction_status": transaction["status"],
            "required_verifier": request["required_verifier"],
            "same_verifier_class": same_verifier_class,
            "source_branch_quality": branch_quality[branch],
            "replacement_quality": replacement_quality,
            "dominance_margin": margin,
            "compared_against": "actual_final_decode",
            "dominates": dominates,
        }
        rows.append({**payload, "candidate_decision_sha256": _sha(payload)})
    return rows, selected_quality


def _intended_decision(
    rows: list[dict[str, Any]],
    *,
    enabled: bool,
    selected_branch_quality: Mapping[str, Any],
    baseline_quality: Mapping[str, Any],
) -> tuple[str, str, str]:
    if not enabled:
        return "retain", "answer_replacement_disabled", ""
    dominant = [row for row in rows if row["dominates"]]
    if dominant:
        winner = sorted(
            dominant,
            key=lambda row: (
                -float(row["replacement_quality"]["lower_bound"]),
                row["request_id"],
            ),
        )[0]
        return (
            "replace",
            "replacement_lower_bound_exceeds_final_decode_upper_bound_plus_margin",
            str(winner["request_id"]),
        )
    if baseline_quality["basis"] == "full_span_semantic_exact_complete":
        return "retain", "final_decode_already_exactly_verified", ""
    if (
        baseline_quality["basis"] == "deterministic_exact_refutation"
        or selected_branch_quality["basis"] == "deterministic_exact_refutation"
    ):
        return "abstain", "known_refutation_has_no_dominant_repair", ""
    return "retain", "no_proven_dominance_or_known_refutation", ""


def build_answer_replacement_receipt(
    *,
    disagreement_graph: Any,
    diagnostic_selection: Any,
    local_repair: Any,
    selected_branch: int,
    branch_candidates: Mapping[int, str],
    generated_repairs: Mapping[str, Mapping[str, Any]],
    objective: str,
    baseline_text: str,
    baseline_tokens: Sequence[int],
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[Sequence[int]], str],
    enabled: bool = True,
    margin: float = DEFAULT_REPLACEMENT_MARGIN,
    max_output_tokens: int,
) -> tuple[dict[str, Any], list[int], dict[str, Any]]:
    """Select and bind output, returning private evidence for service replay."""

    if type(enabled) is not bool:
        raise ValueError("answer replacement enabled flag must be boolean")
    if type(selected_branch) is not int or selected_branch < 0:
        raise ValueError("answer replacement selected branch is invalid")
    if not isinstance(disagreement_graph, Mapping) or not isinstance(
        diagnostic_selection,
        Mapping,
    ):
        raise ValueError("answer replacement upstream evidence is missing")
    if not isinstance(local_repair, Mapping):
        raise ValueError("answer replacement local repair is missing")
    if not isinstance(objective, str) or not isinstance(baseline_text, str):
        raise ValueError("answer replacement text inputs are invalid")
    baseline = list(baseline_tokens)
    if any(type(token) is not int or token < 0 for token in baseline):
        raise ValueError("answer replacement baseline tokens are invalid")
    normalized_margin = _margin(margin)
    output_limit = _output_limit(max_output_tokens)
    validate_local_repair_receipt(
        local_repair,
        disagreement_graph=disagreement_graph,
        diagnostic_selection=diagnostic_selection,
    )
    admitted_request_ids = {
        str(transaction["request_id"])
        for transaction in local_repair["transactions"]
        if transaction["status"] == "repaired_candidate_admitted"
    }
    private_evidence = _normalize_private_evidence(
        {
            "schema": ANSWER_REPLACEMENT_PRIVATE_SCHEMA,
            "objective": objective,
            "branch_candidates": {
                str(index): text for index, text in branch_candidates.items()
            },
            "generated_repairs": {
                request_id: str(result["candidate"])
                for request_id, result in generated_repairs.items()
                if request_id in admitted_request_ids
                if isinstance(result, Mapping)
                and isinstance(result.get("candidate"), str)
            },
            "baseline_text": baseline_text,
            "baseline_tokens": baseline,
        }
    )
    baseline_decomposition = build_atomic_decomposition(
        baseline_text,
        objective=objective,
    )
    baseline_routes = build_deterministic_router_receipt(
        baseline_text,
        objective=objective,
        atomic_receipt=baseline_decomposition,
    )
    baseline_quality = _quality_interval(
        baseline_decomposition,
        baseline_routes,
        candidate=baseline_text,
    )
    rows, selected_quality = _candidate_inventory(
        disagreement_graph=disagreement_graph,
        diagnostic_selection=diagnostic_selection,
        local_repair=local_repair,
        private_evidence=private_evidence,
        selected_branch=selected_branch,
        baseline_quality=baseline_quality,
        margin=normalized_margin,
    )
    intended, reason, selected_request_id = _intended_decision(
        rows,
        enabled=enabled,
        selected_branch_quality=selected_quality,
        baseline_quality=baseline_quality,
    )
    decision = intended
    binding_status = "not_required"
    accepted_text = baseline_text
    accepted_tokens = baseline
    if intended == "replace":
        candidate = private_evidence["generated_repairs"].get(selected_request_id)
        try:
            if not isinstance(candidate, str):
                raise ValueError("replacement private source is absent")
            encoded = list(encode(candidate))
            if (
                not encoded
                or any(type(token) is not int or token < 0 for token in encoded)
                or len(encoded) > output_limit
                or decode(encoded) != candidate
            ):
                raise ValueError("replacement output binding failed")
        except (AttributeError, KeyError, TypeError, ValueError):
            decision = "abstain"
            reason = "dominant_repair_output_binding_failed"
            binding_status = "failed_closed"
            accepted_text = ""
            accepted_tokens = []
        else:
            binding_status = "exact_text_token_roundtrip"
            accepted_text = candidate
            accepted_tokens = encoded
    elif intended == "abstain":
        accepted_text = ""
        accepted_tokens = []
    baseline_binding = {
        "text_sha256": _text_sha(baseline_text),
        "token_count": len(baseline),
        "tokens_sha256": _token_sha(baseline),
    }
    output_binding = {
        "source": (
            "repaired_candidate"
            if decision == "replace"
            else "baseline_decode"
            if decision == "retain"
            else "none"
        ),
        "text_sha256": _text_sha(accepted_text) if decision != "abstain" else "",
        "token_count": len(accepted_tokens),
        "tokens_sha256": (
            _token_sha(accepted_tokens) if decision != "abstain" else ""
        ),
        "binding_status": binding_status,
    }
    policy = {
        "enabled": enabled,
        "margin": normalized_margin,
        "max_output_tokens": output_limit,
        "interval_object": "conjunctive_full_span_exact_claim_validity",
        "replacement_rule": "new_lower_gt_final_decode_upper_plus_margin",
        "syntax_only_verifier_policy": "refutation_only",
        "unknown_claim_policy": "interval_zero_to_one_no_authority",
        "objective_completion_gate": "parent_service_output_quality",
    }
    payload = {
        "schema": ANSWER_REPLACEMENT_SCHEMA,
        "disagreement_graph_sha256": disagreement_graph["receipt_sha256"],
        "diagnostic_selection_sha256": diagnostic_selection["receipt_sha256"],
        "local_repair_sha256": local_repair["receipt_sha256"],
        "private_evidence_sha256": _sha(private_evidence),
        "selected_branch": selected_branch,
        "policy": policy,
        "baseline_decomposition": baseline_decomposition,
        "baseline_routes": baseline_routes,
        "baseline_quality": baseline_quality,
        "selected_branch_quality": selected_quality,
        "candidates": rows,
        "intended_decision": intended,
        "decision": decision,
        "reason": reason,
        "selected_request_id": selected_request_id,
        "baseline_decode": baseline_binding,
        "accepted_output": output_binding,
        "answer_selection_effect": (
            "replaced"
            if decision == "replace"
            else "retained"
            if decision == "retain"
            else "abstained"
        ),
        "latent_state_effect": "none",
        "authority": "confidence_bound_answer_replacement",
    }
    receipt = {**payload, "receipt_sha256": _sha(payload)}
    validate_answer_replacement_receipt(
        receipt,
        disagreement_graph=disagreement_graph,
        diagnostic_selection=diagnostic_selection,
        local_repair=local_repair,
        private_evidence=private_evidence,
        expected_objective=objective,
        expected_selected_branch=selected_branch,
        expected_enabled=enabled,
        expected_margin=normalized_margin,
        expected_max_output_tokens=output_limit,
        expected_output_text=accepted_text,
        expected_output_tokens=accepted_tokens,
    )
    return receipt, accepted_tokens, private_evidence


def validate_answer_replacement_receipt(
    value: Any,
    *,
    disagreement_graph: Any,
    diagnostic_selection: Any,
    local_repair: Any,
    private_evidence: Any,
    expected_objective: str,
    expected_selected_branch: int,
    expected_enabled: bool,
    expected_margin: float,
    expected_max_output_tokens: int,
    expected_output_text: str | None = None,
    expected_output_tokens: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Rebuild baseline and repair evidence in the validating trust domain."""

    if not isinstance(value, Mapping):
        raise ValueError("answer replacement receipt is missing")
    fields = {
        "schema",
        "disagreement_graph_sha256",
        "diagnostic_selection_sha256",
        "local_repair_sha256",
        "private_evidence_sha256",
        "selected_branch",
        "policy",
        "baseline_decomposition",
        "baseline_routes",
        "baseline_quality",
        "selected_branch_quality",
        "candidates",
        "intended_decision",
        "decision",
        "reason",
        "selected_request_id",
        "baseline_decode",
        "accepted_output",
        "answer_selection_effect",
        "latent_state_effect",
        "authority",
        "receipt_sha256",
    }
    if set(value) != fields:
        raise ValueError("answer replacement receipt fields differ")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != _sha(payload):
        raise ValueError("answer replacement receipt commitment mismatch")
    if type(expected_enabled) is not bool or not isinstance(
        expected_objective,
        str,
    ):
        raise ValueError("answer replacement expected policy is invalid")
    margin = _margin(expected_margin)
    output_limit = _output_limit(expected_max_output_tokens)
    private = _normalize_private_evidence(private_evidence)
    if (
        private["objective"] != expected_objective
        or value["private_evidence_sha256"] != _sha(private)
    ):
        raise ValueError("answer replacement private evidence binding differs")
    validate_local_repair_receipt(
        local_repair,
        disagreement_graph=disagreement_graph,
        diagnostic_selection=diagnostic_selection,
    )
    baseline_text = private["baseline_text"]
    baseline_tokens = private["baseline_tokens"]
    baseline_decomposition = build_atomic_decomposition(
        baseline_text,
        objective=expected_objective,
    )
    baseline_routes = build_deterministic_router_receipt(
        baseline_text,
        objective=expected_objective,
        atomic_receipt=baseline_decomposition,
    )
    baseline_quality = _quality_interval(
        baseline_decomposition,
        baseline_routes,
        candidate=baseline_text,
    )
    expected_rows, selected_quality = _candidate_inventory(
        disagreement_graph=disagreement_graph,
        diagnostic_selection=diagnostic_selection,
        local_repair=local_repair,
        private_evidence=private,
        selected_branch=expected_selected_branch,
        baseline_quality=baseline_quality,
        margin=margin,
    )
    intended, expected_reason, selected_request_id = _intended_decision(
        expected_rows,
        enabled=expected_enabled,
        selected_branch_quality=selected_quality,
        baseline_quality=baseline_quality,
    )
    policy = {
        "enabled": expected_enabled,
        "margin": margin,
        "max_output_tokens": output_limit,
        "interval_object": "conjunctive_full_span_exact_claim_validity",
        "replacement_rule": "new_lower_gt_final_decode_upper_plus_margin",
        "syntax_only_verifier_policy": "refutation_only",
        "unknown_claim_policy": "interval_zero_to_one_no_authority",
        "objective_completion_gate": "parent_service_output_quality",
    }
    baseline = value["baseline_decode"]
    binding = value["accepted_output"]
    if (
        value["schema"] != ANSWER_REPLACEMENT_SCHEMA
        or value["disagreement_graph_sha256"]
        != disagreement_graph.get("receipt_sha256")
        or value["diagnostic_selection_sha256"]
        != diagnostic_selection.get("receipt_sha256")
        or value["local_repair_sha256"] != local_repair.get("receipt_sha256")
        or value["selected_branch"] != expected_selected_branch
        or value["policy"] != policy
        or value["baseline_decomposition"] != baseline_decomposition
        or value["baseline_routes"] != baseline_routes
        or value["baseline_quality"] != baseline_quality
        or value["selected_branch_quality"] != selected_quality
        or value["candidates"] != expected_rows
        or value["intended_decision"] != intended
        or value["selected_request_id"] != selected_request_id
        or baseline
        != {
            "text_sha256": _text_sha(baseline_text),
            "token_count": len(baseline_tokens),
            "tokens_sha256": _token_sha(baseline_tokens),
        }
        or not isinstance(binding, Mapping)
        or set(binding)
        != {
            "source",
            "text_sha256",
            "token_count",
            "tokens_sha256",
            "binding_status",
        }
        or value["latent_state_effect"] != "none"
        or value["authority"] != "confidence_bound_answer_replacement"
    ):
        raise ValueError("answer replacement reconstruction differs")
    decision = value["decision"]
    if decision == "replace":
        candidate = private["generated_repairs"].get(selected_request_id)
        expected_effect = "replaced"
        if (
            intended != "replace"
            or value["reason"] != expected_reason
            or not isinstance(candidate, str)
            or binding["source"] != "repaired_candidate"
            or binding["binding_status"] != "exact_text_token_roundtrip"
            or binding["text_sha256"] != _text_sha(candidate)
            or not 0 < binding["token_count"] <= output_limit
            or _SHA256_RE.fullmatch(str(binding["tokens_sha256"])) is None
        ):
            raise ValueError("answer replacement authority is invalid")
    elif decision == "retain":
        expected_effect = "retained"
        if (
            intended != "retain"
            or value["reason"] != expected_reason
            or binding["source"] != "baseline_decode"
            or binding["binding_status"] != "not_required"
            or binding["text_sha256"] != _text_sha(baseline_text)
            or _SHA256_RE.fullmatch(str(binding["tokens_sha256"])) is None
        ):
            raise ValueError("answer retention binding is invalid")
    elif decision == "abstain":
        expected_effect = "abstained"
        binding_failure = (
            intended == "replace"
            and value["reason"] == "dominant_repair_output_binding_failed"
            and binding["binding_status"] == "failed_closed"
        )
        if (
            not (
                (intended == "abstain" and value["reason"] == expected_reason)
                or binding_failure
            )
            or binding["source"] != "none"
            or binding["text_sha256"] != ""
            or binding["token_count"] != 0
            or binding["tokens_sha256"] != ""
            or binding["binding_status"] not in {"not_required", "failed_closed"}
        ):
            raise ValueError("answer abstention binding is invalid")
    else:
        raise ValueError("answer replacement decision is invalid")
    if (
        value["answer_selection_effect"] != expected_effect
        or (
            expected_output_text is not None
            and binding["text_sha256"]
            != (_text_sha(expected_output_text) if decision != "abstain" else "")
        )
        or (
            expected_output_tokens is not None
            and (
                binding["token_count"] != len(expected_output_tokens)
                or binding["tokens_sha256"]
                != (
                    _token_sha(expected_output_tokens)
                    if decision != "abstain"
                    else ""
                )
            )
        )
    ):
        raise ValueError("answer replacement output binding differs")
    return dict(value)


__all__ = [
    "ANSWER_REPLACEMENT_PRIVATE_SCHEMA",
    "ANSWER_REPLACEMENT_SCHEMA",
    "DEFAULT_REPLACEMENT_MARGIN",
    "MAX_REPLACEMENT_OUTPUT_TOKENS",
    "build_answer_replacement_receipt",
    "validate_answer_replacement_receipt",
]
