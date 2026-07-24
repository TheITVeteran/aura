"""Fresh-context derivation and falsification for disputed candidate atoms.

The resident model is allowed to propose a witness, never to certify itself.
Each request contains only the original objective and one anonymized atom, is
decoded from a zero-offset KV cache, and is committed without retaining model
scratch text in the public receipt.  A proposal earns causal authority only
when a deterministic relation checker can reconstruct the witness and prove
that it supports or refutes the target atom.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from core.brain.llm.latent_cortex.atomic_decomposition import (
    build_atomic_decomposition,
    validate_atomic_decomposition,
    validate_atomic_decomposition_envelope,
)
from core.brain.llm.latent_cortex.deterministic_verifier_router import (
    RouteOutcome,
    build_deterministic_router_receipt,
    validate_deterministic_router_envelope,
)

GENERATIVE_VERIFIER_SCHEMA = "aura.rlc.generative_verifier.v1"
FRESH_CONTEXT_SCHEMA = "aura.rlc.fresh_verifier_context.v1"
_RESULT_RE = re.compile(r"FINAL_ANSWER\s*:\s*(\{.*\})\s*$", re.DOTALL)
_ARITH_RE = re.compile(
    r"(?<![\d.])(-?\d{1,12})\s*([+\-*/x\u00d7])\s*(-?\d{1,12})\s*=\s*(-?\d{1,12})(?!\d)(?!\.\d)"
)
_VERDICTS = {"supports", "refutes", "unknown"}


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


def build_verification_prompt(*, objective: str, atom: str, atom_sha256: str) -> str:
    """Create the ownership-free prompt used in the isolated verifier cache."""

    objective = str(objective or "")[:8192]
    atom = str(atom or "")[:512]
    return (
        "You are an independent derivation lane. You did not produce the candidate.\n"
        "Work only from the problem and the anonymized disputed claim below.\n"
        "Try to derive or falsify the claim from scratch. Do not defer to its wording.\n"
        "Return exactly FINAL_ANSWER followed by one JSON object with string keys "
        '"claim_sha256", "verdict", and "witness". Verdict must be supports, '
        "refutes, or unknown. A witness must be a minimal independently derived, "
        "machine-checkable equality or artifact; use an empty witness for unknown.\n\n"
        f"PROBLEM:\n{objective}\n\n"
        f"ANONYMIZED_CLAIM_SHA256: {atom_sha256}\n"
        f"ANONYMIZED_CLAIM:\n{atom}\n"
    )


def parse_generation_result(text: str, *, claim_sha256: str) -> dict[str, str]:
    """Parse the strict public result contract; surrounding rationale is refused."""

    match = _RESULT_RE.fullmatch(str(text or "").strip())
    if match is None:
        raise ValueError("generation did not satisfy FINAL_ANSWER JSON contract")
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError("generation result is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"claim_sha256", "verdict", "witness"}:
        raise ValueError("generation result fields do not match contract")
    if any(not isinstance(value[key], str) for key in value):
        raise ValueError("generation result values must be strings")
    if value["claim_sha256"] != claim_sha256:
        raise ValueError("generation result is not bound to the disputed claim")
    if value["verdict"] not in _VERDICTS:
        raise ValueError("generation verdict is invalid")
    if len(value["witness"]) > 2048:
        raise ValueError("generation witness exceeds 2048 characters")
    if value["verdict"] == "unknown" and value["witness"].strip():
        raise ValueError("unknown verdict must not smuggle an unverified witness")
    if value["verdict"] != "unknown" and not value["witness"].strip():
        raise ValueError("support or refutation requires a witness")
    return {key: value[key] for key in ("claim_sha256", "verdict", "witness")}


def _arithmetic_claim(text: str) -> tuple[int, str, int, int] | None:
    matches = list(_ARITH_RE.finditer(text))
    if len(matches) != 1:
        return None
    left, operator, right, claimed = matches[0].groups()
    return (
        int(left),
        "*" if operator in {"x", "\u00d7"} else operator,
        int(right),
        int(claimed),
    )


def _arithmetic_actual(left: int, operator: str, right: int) -> int | None:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if right == 0 or left % right:
        return None
    return left // right


def _deterministic_relation(
    *,
    target: str,
    verdict: str,
    witness: str,
) -> tuple[bool, dict[str, Any], dict[str, Any] | None]:
    """Admit only exact arithmetic relations currently reconstructable end to end."""

    if verdict == "unknown":
        return False, {"kind": "none", "reason": "generator_abstained"}, None
    target_claim = _arithmetic_claim(target)
    witness_claim = _arithmetic_claim(witness)
    if target_claim is None or witness_claim is None:
        return False, {"kind": "none", "reason": "relation_not_machine_checkable"}, None
    left, operator, right, target_value = target_claim
    w_left, w_operator, w_right, witness_value = witness_claim
    actual = _arithmetic_actual(left, operator, right)
    if actual is None or (left, operator, right) != (w_left, w_operator, w_right):
        return False, {"kind": "none", "reason": "witness_does_not_rederive_target"}, None

    witness_atomic = build_atomic_decomposition(witness, objective=target)
    witness_router = build_deterministic_router_receipt(
        witness,
        objective=target,
        atomic_receipt=witness_atomic,
    )
    verified = (
        witness_value == actual
        and witness_router["counts"][RouteOutcome.VERIFIED.value] == 1
        and witness_router["counts"][RouteOutcome.REFUTED.value] == 0
    )
    relation_matches = (verdict == "supports" and target_value == actual) or (
        verdict == "refutes" and target_value != actual
    )
    evidence = {
        "kind": "exact_integer_arithmetic_relation",
        "left": left,
        "operator": operator,
        "right": right,
        "target_claimed": target_value,
        "derived": witness_value,
        "actual": actual,
        "witness_sha256": _text_sha(witness),
    }
    return bool(verified and relation_matches), evidence, witness_router


def run_generative_verifier(
    candidate: str,
    *,
    objective: str,
    generate: Callable[[str], Mapping[str, Any]],
    max_atoms: int = 1,
) -> dict[str, Any]:
    """Challenge bounded disputed atoms through independently cached generation."""

    if type(max_atoms) is not int or not 1 <= max_atoms <= 8:
        raise ValueError("max_atoms must be an integer in [1, 8]")
    atomic = build_atomic_decomposition(candidate, objective=objective)
    routed = build_deterministic_router_receipt(
        candidate,
        objective=objective,
        atomic_receipt=atomic,
    )
    validated = validate_atomic_decomposition(
        atomic,
        candidate=candidate,
        objective=objective,
    )
    route_by_atom = {row["atom_id"]: row for row in routed["routes"]}
    priority = {
        RouteOutcome.REFUTED.value: 0,
        RouteOutcome.UNKNOWN.value: 1,
        RouteOutcome.UNSUPPORTED.value: 2,
        RouteOutcome.VERIFIED.value: 3,
    }
    disputed = [
        atom
        for atom in validated["atoms"]
        if route_by_atom[atom["atom_id"]]["outcome"] != RouteOutcome.VERIFIED.value
    ]
    targets = sorted(
        disputed,
        key=lambda atom: (priority[route_by_atom[atom["atom_id"]]["outcome"]], atom["atom_id"]),
    )[:max_atoms]
    attempts: list[dict[str, Any]] = []
    for atom in targets:
        target = candidate[atom["start"] : atom["end"]]
        prompt = build_verification_prompt(
            objective=objective,
            atom=target,
            atom_sha256=atom["atom_sha256"],
        )
        base = {
            "atom_id": atom["atom_id"],
            "atom_sha256": atom["atom_sha256"],
            "prior_route_outcome": route_by_atom[atom["atom_id"]]["outcome"],
            "prompt_sha256": _text_sha(prompt),
        }
        try:
            generated = generate(prompt)
            if not isinstance(generated, Mapping):
                raise TypeError("generator result must be a mapping")
        except (OSError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
            row = {
                **base,
                "generation_status": "abstained",
                "generated_output_sha256": "",
                "verdict": "unknown",
                "witness_sha256": "",
                "context": {},
                "relation": {"kind": "none", "reason": f"{type(exc).__name__}:{exc}"[:240]},
                "witness_router_sha256": "",
                "authority_admitted": False,
            }
        else:
            generated_text = str(generated.get("text") or "")
            context = dict(generated.get("context") or {})
            try:
                result = parse_generation_result(
                    generated_text,
                    claim_sha256=atom["atom_sha256"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                row = {
                    **base,
                    "generation_status": "complete",
                    "generated_output_sha256": _text_sha(generated_text),
                    "verdict": "unknown",
                    "witness_sha256": _text_sha(""),
                    "context": context,
                    "relation": {
                        "kind": "none",
                        "reason": f"contract_refused:{type(exc).__name__}:{exc}"[:240],
                    },
                    "witness_router_sha256": "",
                    "authority_admitted": False,
                }
            else:
                admitted, relation, witness_router = _deterministic_relation(
                    target=target,
                    verdict=result["verdict"],
                    witness=result["witness"],
                )
                row = {
                    **base,
                    "generation_status": "complete",
                    "generated_output_sha256": _text_sha(generated_text),
                    "verdict": result["verdict"],
                    "witness_sha256": _text_sha(result["witness"]),
                    "context": context,
                    "relation": relation,
                    "witness_router_sha256": (
                        str(witness_router.get("receipt_sha256") or "")
                        if witness_router is not None
                        else ""
                    ),
                    "authority_admitted": admitted,
                }
        attempts.append(row)

    admitted = [row for row in attempts if row["authority_admitted"]]
    causal_refutation = any(row["verdict"] == "refutes" for row in admitted)
    payload = {
        "schema": GENERATIVE_VERIFIER_SCHEMA,
        "source_sha256": atomic["source_sha256"],
        "objective_sha256": atomic["objective_sha256"],
        "atomic_receipt_sha256": atomic["receipt_sha256"],
        "deterministic_router_sha256": routed["receipt_sha256"],
        "atomic_decomposition": atomic,
        "deterministic_router": routed,
        "parameter_independence": False,
        "context_independence": True,
        "authority_mode": "deterministic_refutation_veto_only",
        "attempts": attempts,
        "attempted": len(attempts),
        "admitted": len(admitted),
        "causal_refutation": causal_refutation,
        "selection_authority_admitted": causal_refutation,
        "vetoed_branch": None,
        "replacement_branch": None,
        "selection_effect": "none",
    }
    return validate_generative_verifier_envelope(
        {**payload, "receipt_sha256": _sha(payload)}
    )


def bind_selection_effect(
    value: Mapping[str, Any],
    *,
    vetoed_branch: int,
    replacement_branch: int | None,
) -> dict[str, Any]:
    """Bind the already-proven refutation to its actual branch-selection effect."""

    receipt = validate_generative_verifier_envelope(value)
    if receipt["causal_refutation"] is not True:
        raise ValueError("selection effect requires an admitted refutation")
    if type(vetoed_branch) is not int or vetoed_branch < 0:
        raise ValueError("vetoed branch must be a non-negative integer")
    if replacement_branch is not None and (
        type(replacement_branch) is not int
        or replacement_branch < 0
        or replacement_branch == vetoed_branch
    ):
        raise ValueError("replacement branch is invalid")
    receipt.pop("receipt_sha256", None)
    receipt.update(
        {
            "vetoed_branch": vetoed_branch,
            "replacement_branch": replacement_branch,
            "selection_effect": (
                "winner_replaced" if replacement_branch is not None else "no_alternative"
            ),
        }
    )
    return {**receipt, "receipt_sha256": _sha(receipt)}


def validate_generative_verifier_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the text-free public receipt and recompute every admitted verdict."""

    fields = {
        "schema",
        "source_sha256",
        "objective_sha256",
        "atomic_receipt_sha256",
        "deterministic_router_sha256",
        "atomic_decomposition",
        "deterministic_router",
        "parameter_independence",
        "context_independence",
        "authority_mode",
        "attempts",
        "attempted",
        "admitted",
        "causal_refutation",
        "selection_authority_admitted",
        "vetoed_branch",
        "replacement_branch",
        "selection_effect",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("generative verifier fields do not match schema")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != _sha(payload):
        raise ValueError("generative verifier commitment mismatch")
    if (
        value["schema"] != GENERATIVE_VERIFIER_SCHEMA
        or value["parameter_independence"] is not False
        or value["context_independence"] is not True
        or value["authority_mode"] != "deterministic_refutation_veto_only"
    ):
        raise ValueError("generative verifier independence or authority claim is invalid")
    for name in (
        "source_sha256",
        "objective_sha256",
        "atomic_receipt_sha256",
        "deterministic_router_sha256",
        "receipt_sha256",
    ):
        if not isinstance(value[name], str) or len(value[name]) != 64:
            raise ValueError("generative verifier hash is invalid")
    atomic = validate_atomic_decomposition_envelope(value["atomic_decomposition"])
    routed = validate_deterministic_router_envelope(
        value["deterministic_router"],
        atomic_receipt=atomic,
    )
    if (
        value["source_sha256"] != atomic["source_sha256"]
        or value["objective_sha256"] != atomic["objective_sha256"]
        or value["atomic_receipt_sha256"] != atomic["receipt_sha256"]
        or value["deterministic_router_sha256"] != routed["receipt_sha256"]
    ):
        raise ValueError("generative verifier source envelopes do not match")
    attempts = value["attempts"]
    if not isinstance(attempts, list) or not 0 <= len(attempts) <= 8:
        raise ValueError("generative verifier attempt inventory is invalid")
    admitted = 0
    refuted = False
    route_by_atom = {row["atom_id"]: row for row in routed["routes"]}
    priority = {
        RouteOutcome.REFUTED.value: 0,
        RouteOutcome.UNKNOWN.value: 1,
        RouteOutcome.UNSUPPORTED.value: 2,
        RouteOutcome.VERIFIED.value: 3,
    }
    expected_targets = [
        atom["atom_id"]
        for atom in sorted(
            (
                atom
                for atom in atomic["atoms"]
                if route_by_atom[atom["atom_id"]]["outcome"]
                != RouteOutcome.VERIFIED.value
            ),
            key=lambda atom: (
                priority[route_by_atom[atom["atom_id"]]["outcome"]],
                atom["atom_id"],
            ),
        )
    ][: len(attempts)]
    if [row.get("atom_id") for row in attempts if isinstance(row, Mapping)] != expected_targets:
        raise ValueError("generative verifier challenge order was cherry-picked")
    for row in attempts:
        required = {
            "atom_id",
            "atom_sha256",
            "prior_route_outcome",
            "prompt_sha256",
            "generation_status",
            "generated_output_sha256",
            "verdict",
            "witness_sha256",
            "context",
            "relation",
            "witness_router_sha256",
            "authority_admitted",
        }
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("generative verifier attempt fields do not match schema")
        if (
            not isinstance(row["atom_id"], str)
            or not row["atom_id"]
            or not isinstance(row["atom_sha256"], str)
            or len(row["atom_sha256"]) != 64
            or not isinstance(row["prompt_sha256"], str)
            or len(row["prompt_sha256"]) != 64
            or row["generation_status"] not in {"complete", "abstained"}
            or row["verdict"] not in _VERDICTS
            or row["prior_route_outcome"]
            not in {outcome.value for outcome in RouteOutcome}
            or row["atom_id"] not in route_by_atom
            or row["atom_sha256"]
            != next(
                atom["atom_sha256"]
                for atom in atomic["atoms"]
                if atom["atom_id"] == row["atom_id"]
            )
            or row["prior_route_outcome"] != route_by_atom[row["atom_id"]]["outcome"]
            or type(row["authority_admitted"]) is not bool
        ):
            raise ValueError("generative verifier attempt verdict is invalid")
        relation = row["relation"]
        authority = row["authority_admitted"]
        context = row["context"]
        if row["generation_status"] == "complete":
            context_fields = {
                "schema",
                "prompt_token_count",
                "generated_token_count",
                "termination",
                "initial_cache_offsets",
                "final_cache_offsets",
                "all_initial_offsets_zero",
                "solver_context_imported",
                "parameter_relation",
            }
            if not isinstance(context, Mapping) or set(context) != context_fields:
                raise ValueError("fresh verifier context fields do not match schema")
            initial = context["initial_cache_offsets"]
            final = context["final_cache_offsets"]
            if (
                context["schema"] != FRESH_CONTEXT_SCHEMA
                or not isinstance(initial, list)
                or not initial
                or any(type(offset) is not int or offset != 0 for offset in initial)
                or not isinstance(final, list)
                or len(final) != len(initial)
                or any(type(offset) is not int or offset < 0 for offset in final)
                or context["all_initial_offsets_zero"] is not True
                or context["solver_context_imported"] is not False
                or context["parameter_relation"] != "shared_resident_checkpoint"
                or type(context["prompt_token_count"]) is not int
                or context["prompt_token_count"] < 1
                or type(context["generated_token_count"]) is not int
                or context["generated_token_count"] < 1
                or not isinstance(context["termination"], str)
                or len(set(final)) != 1
                or final[0] < context["prompt_token_count"]
                or not isinstance(row["generated_output_sha256"], str)
                or len(row["generated_output_sha256"]) != 64
                or not isinstance(row["witness_sha256"], str)
                or len(row["witness_sha256"]) != 64
            ):
                raise ValueError("fresh verifier context isolation is invalid")
        elif (
            context != {}
            or row["generated_output_sha256"] != ""
            or row["witness_sha256"] != ""
            or row["witness_router_sha256"] != ""
            or row["verdict"] != "unknown"
            or row["authority_admitted"] is not False
        ):
            raise ValueError("abstained generation claimed evidence or authority")
        if authority:
            expected_relation_fields = {
                "kind",
                "left",
                "operator",
                "right",
                "target_claimed",
                "derived",
                "actual",
                "witness_sha256",
            }
            if not isinstance(relation, Mapping) or set(relation) != expected_relation_fields:
                raise ValueError("admitted generative relation fields are invalid")
            actual = _arithmetic_actual(
                relation["left"], relation["operator"], relation["right"]
            )
            if (
                relation["kind"] != "exact_integer_arithmetic_relation"
                or actual is None
                or relation["actual"] != actual
                or relation["derived"] != actual
                or relation["witness_sha256"] != row["witness_sha256"]
                or not isinstance(row["witness_router_sha256"], str)
                or len(row["witness_router_sha256"]) != 64
                or (
                    row["verdict"] == "supports"
                    and relation["target_claimed"] != actual
                )
                or (
                    row["verdict"] == "refutes"
                    and relation["target_claimed"] == actual
                )
            ):
                raise ValueError("admitted generative relation does not reconstruct")
            admitted += 1
            refuted = refuted or row["verdict"] == "refutes"
        elif not isinstance(relation, Mapping):
            raise ValueError("generative abstention relation is invalid")
    if (
        type(value["attempted"]) is not int
        or type(value["admitted"]) is not int
        or value["attempted"] != len(attempts)
        or value["admitted"] != admitted
        or value["causal_refutation"] is not refuted
        or value["selection_authority_admitted"] is not refuted
    ):
        raise ValueError("generative verifier aggregate verdict mismatch")
    vetoed = value["vetoed_branch"]
    replacement = value["replacement_branch"]
    effect = value["selection_effect"]
    if effect == "none":
        if vetoed is not None or replacement is not None:
            raise ValueError("no-effect generative receipt names branches")
    elif effect == "winner_replaced":
        if (
            refuted is not True
            or type(vetoed) is not int
            or vetoed < 0
            or type(replacement) is not int
            or replacement < 0
            or replacement == vetoed
        ):
            raise ValueError("generative replacement effect is invalid")
    elif effect == "no_alternative":
        if refuted is not True or type(vetoed) is not int or vetoed < 0 or replacement is not None:
            raise ValueError("generative no-alternative effect is invalid")
    else:
        raise ValueError("generative selection effect is invalid")
    return dict(value)


__all__ = [
    "GENERATIVE_VERIFIER_SCHEMA",
    "FRESH_CONTEXT_SCHEMA",
    "build_verification_prompt",
    "bind_selection_effect",
    "parse_generation_result",
    "run_generative_verifier",
    "validate_generative_verifier_envelope",
]
