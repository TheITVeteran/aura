"""Deterministic per-atom verifier routing with explicit non-verdicts."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from core.brain.llm.latent_cortex.atomic_decomposition import (
    validate_atomic_decomposition,
    validate_atomic_decomposition_envelope,
)

DETERMINISTIC_ROUTER_SCHEMA = "aura.rlc.deterministic_verifier_router.v1"
_ARITH_RE = re.compile(
    r"(?<![\d.])(-?\d{1,12})\s*([+\-*/x×])\s*(-?\d{1,12})\s*=\s*(-?\d{1,12})(?!\d)(?!\.\d)"
)
_FORMAL_RE = re.compile(r"\b(?:sat|smt|z3|theorem|lemma|proof|prove)\b", re.I)
_RETRIEVAL_RE = re.compile(r"\b(?:citation|source|according\s+to|retriev|url|doi)\w*\b", re.I)
_SIMULATION_RE = re.compile(
    r"\b(?:simulate|simulation|physics|trajectory|counterfactual)\w*\b", re.I
)
_PLANNING_RE = re.compile(r"\b(?:plan|schedule|route|workflow|step\s+\d+)\w*\b", re.I)


class RouteOutcome(StrEnum):
    VERIFIED = "verified"
    REFUTED = "refuted"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


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


def _strip_fence(fragment: str) -> str:
    if not fragment.startswith("```"):
        return fragment
    first = fragment.find("\n")
    if first < 0:
        return ""
    body = fragment[first + 1 :]
    return body[:-3] if body.endswith("```") else body


def _arithmetic_verdict(fragment: str) -> tuple[RouteOutcome, str, dict[str, Any]] | None:
    matches = list(_ARITH_RE.finditer(fragment))
    if not matches:
        return None
    failures: list[str] = []
    for match in matches:
        left, operator, right, claimed = match.groups()
        a, b, expected = int(left), int(right), int(claimed)
        operator = "*" if operator in {"x", "×"} else operator
        if operator == "/":
            if b == 0 or a % b:
                failures.append("non_integral_or_zero_division")
                continue
            actual = a // b
        elif operator == "+":
            actual = a + b
        elif operator == "-":
            actual = a - b
        else:
            actual = a * b
        if actual != expected:
            failures.append(f"claim_{match.start()}_mismatch")
    outcome = RouteOutcome.REFUTED if failures else RouteOutcome.VERIFIED
    return (
        outcome,
        "exact_integer_arithmetic",
        {
            "claims_checked": len(matches),
            "failure_codes": failures,
        },
    )


def _route_atom(
    fragment: str,
    atom: Mapping[str, Any],
    *,
    code_atom_count: int,
) -> tuple[str, str, dict[str, Any]]:
    if atom.get("kind") == "code":
        if code_atom_count > 1:
            return (
                RouteOutcome.UNSUPPORTED.value,
                "python_ast",
                {"reason": "chunked_code_requires_bundle_compilation"},
            )
        try:
            ast.parse(_strip_fence(fragment))
        except SyntaxError as exc:
            return (
                RouteOutcome.REFUTED.value,
                "python_ast",
                {
                    "failure_code": "syntax_error",
                    "line": int(exc.lineno or 0),
                },
            )
        return RouteOutcome.VERIFIED.value, "python_ast", {"compiled": True}

    arithmetic = _arithmetic_verdict(fragment)
    if arithmetic is not None:
        outcome, verifier, detail = arithmetic
        return outcome.value, verifier, detail

    stripped = fragment.strip()
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
        except json.JSONDecodeError as exc:
            closing = "}" if stripped.startswith("{") else "]"
            if not stripped.endswith(closing):
                return RouteOutcome.UNKNOWN.value, "json_parser", {"reason": "partial_json_atom"}
            return (
                RouteOutcome.REFUTED.value,
                "json_parser",
                {
                    "failure_code": "invalid_json",
                    "position": exc.pos,
                },
            )
        return RouteOutcome.VERIFIED.value, "json_parser", {"parsed": True}

    for pattern, verifier, reason in (
        (_FORMAL_RE, "formal_solver", "machine_form_not_supplied"),
        (_RETRIEVAL_RE, "source_retrieval", "source_bound_observation_required"),
        (_SIMULATION_RE, "simulation", "governed_simulator_context_required"),
        (_PLANNING_RE, "planning", "typed_world_state_required"),
    ):
        if pattern.search(fragment):
            return RouteOutcome.UNSUPPORTED.value, verifier, {"reason": reason}
    return RouteOutcome.UNKNOWN.value, "none", {"reason": "no_sound_deterministic_route"}


def build_deterministic_router_receipt(
    candidate: str,
    *,
    objective: str,
    atomic_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    atomic = validate_atomic_decomposition(
        atomic_receipt,
        candidate=candidate,
        objective=objective,
    )
    routes: list[dict[str, Any]] = []
    code_atom_count = sum(atom["kind"] == "code" for atom in atomic["atoms"])
    for atom in atomic["atoms"]:
        fragment = candidate[atom["start"] : atom["end"]]
        outcome, verifier, detail = _route_atom(
            fragment,
            atom,
            code_atom_count=code_atom_count,
        )
        tool_payload = {
            "tool_id": verifier,
            "execution_mode": "pure_local_read_only",
            "input_sha256": atom["text_sha256"],
            "output_sha256": _sha(detail),
        }
        row_payload = {
            "atom_id": atom["atom_id"],
            "atom_sha256": atom["atom_sha256"],
            "verifier": verifier,
            "outcome": outcome,
            "detail": detail,
            "tool_receipt": tool_payload,
        }
        routes.append({**row_payload, "route_sha256": _sha(row_payload)})
    counts = {
        outcome.value: sum(row["outcome"] == outcome.value for row in routes)
        for outcome in RouteOutcome
    }
    payload = {
        "schema": DETERMINISTIC_ROUTER_SCHEMA,
        "source_sha256": atomic["source_sha256"],
        "objective_sha256": atomic["objective_sha256"],
        "atomic_receipt_sha256": atomic["receipt_sha256"],
        "routes": routes,
        "counts": counts,
        "checked": counts["verified"] + counts["refuted"] > 0,
        "hard_pass": counts["refuted"] == 0,
    }
    return {**payload, "receipt_sha256": _sha(payload)}


def validate_deterministic_router_envelope(
    value: Mapping[str, Any],
    *,
    atomic_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently validate text-free route and tool commitments."""

    atomic = validate_atomic_decomposition_envelope(atomic_receipt)
    fields = {
        "schema",
        "source_sha256",
        "objective_sha256",
        "atomic_receipt_sha256",
        "routes",
        "counts",
        "checked",
        "hard_pass",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("deterministic router fields do not match schema")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != _sha(payload):
        raise ValueError("deterministic router commitment mismatch")
    if (
        value["schema"] != DETERMINISTIC_ROUTER_SCHEMA
        or value["source_sha256"] != atomic["source_sha256"]
        or value["objective_sha256"] != atomic["objective_sha256"]
        or value["atomic_receipt_sha256"] != atomic["receipt_sha256"]
    ):
        raise ValueError("deterministic router source binding mismatch")
    routes = value["routes"]
    if not isinstance(routes, list) or len(routes) != len(atomic["atoms"]):
        raise ValueError("deterministic router route inventory mismatch")
    outcomes = {outcome.value for outcome in RouteOutcome}
    for atom, route in zip(atomic["atoms"], routes, strict=True):
        route_fields = {
            "atom_id",
            "atom_sha256",
            "verifier",
            "outcome",
            "detail",
            "tool_receipt",
            "route_sha256",
        }
        if not isinstance(route, Mapping) or set(route) != route_fields:
            raise ValueError("deterministic route fields do not match schema")
        route_payload = {key: route[key] for key in route_fields - {"route_sha256"}}
        tool = route["tool_receipt"]
        if (
            route["route_sha256"] != _sha(route_payload)
            or route["atom_id"] != atom["atom_id"]
            or route["atom_sha256"] != atom["atom_sha256"]
            or route["outcome"] not in outcomes
            or not isinstance(route["verifier"], str)
            or not route["verifier"]
            or not isinstance(route["detail"], Mapping)
            or not isinstance(tool, Mapping)
            or set(tool) != {"tool_id", "execution_mode", "input_sha256", "output_sha256"}
            or tool["tool_id"] != route["verifier"]
            or tool["execution_mode"] != "pure_local_read_only"
            or tool["input_sha256"] != atom["text_sha256"]
            or tool["output_sha256"] != _sha(route["detail"])
        ):
            raise ValueError("deterministic route commitment is invalid")
    counts = {
        outcome.value: sum(route["outcome"] == outcome.value for route in routes)
        for outcome in RouteOutcome
    }
    expected_checked = counts["verified"] + counts["refuted"] > 0
    expected_hard_pass = counts["refuted"] == 0
    if (
        value["counts"] != counts
        or value["checked"] is not expected_checked
        or value["hard_pass"] is not expected_hard_pass
    ):
        raise ValueError("deterministic router reconstructed verdict mismatch")
    return dict(value)


def router_check(
    candidate: str,
    *,
    objective: str,
    atomic_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        receipt = build_deterministic_router_receipt(
            candidate,
            objective=objective,
            atomic_receipt=atomic_receipt,
        )
    except (TypeError, ValueError) as exc:
        return {
            "applicable": True,
            "valid": False,
            "score": 0.0,
            "failures": [f"deterministic_router:{type(exc).__name__}:{exc}"],
            "receipt": None,
        }
    failures = [
        f"refuted:{row['atom_id']}:{row['verifier']}"
        for row in receipt["routes"]
        if row["outcome"] == RouteOutcome.REFUTED.value
    ]
    score = None
    if receipt["checked"]:
        checked = receipt["counts"]["verified"] + receipt["counts"]["refuted"]
        score = receipt["counts"]["verified"] / checked
    return {
        "applicable": receipt["checked"],
        "valid": receipt["hard_pass"],
        "score": score,
        "failures": failures,
        "receipt": receipt,
    }
