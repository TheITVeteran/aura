"""Fresh-context counterfactual checks with bounded selection authority.

The verifier never promotes a lower task-verifier score. It may only resolve
an exact top-score tie after every tied branch receives the same reconstructable
interventions. Model output is a proposal; exact arithmetic recomputation is
the authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.brain.llm.latent_cortex.atomic_decomposition import (
    build_atomic_decomposition,
    validate_atomic_decomposition,
)
from core.brain.llm.latent_cortex.generative_verifier import FRESH_CONTEXT_SCHEMA
from core.runtime.file_read_gateway import read_stable_bytes

COUNTERFACTUAL_VERIFIER_SCHEMA = "aura.rlc.counterfactual_verifier.v1"
_RESULT_RE = re.compile(r"FINAL_ANSWER\s*:\s*(\{.*\})\s*$", re.DOTALL)
_ARITH_RE = re.compile(
    r"(?<![\d.])(-?\d{1,12})\s*([+\-*/x\u00d7])\s*(-?\d{1,12})"
    r"\s*=\s*(-?\d{1,12})(?!\d)(?!\.\d)"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INTERVENTION_FAMILIES = ("left_input_delta", "right_input_delta", "operator_assumption_swap")
_OUTCOMES = {"correct_change", "invariant_failure", "incorrect_change", "abstained"}


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


def _rank_evidence(row: Mapping[str, Any]) -> tuple[float, int, int]:
    return (
        -float(row["robustness_score"] or 0.0),
        int(row["invariant_failures"]),
        int(row["incorrect_changes"]),
    )


@lru_cache(maxsize=1)
def _implementation_sha256() -> str:
    source = read_stable_bytes(Path(__file__), max_bytes=2 * 1024 * 1024)
    return hashlib.sha256(source).hexdigest()


def _normal_operator(value: str) -> str:
    return "*" if value in {"x", "\u00d7"} else value


def _actual(left: int, operator: str, right: int) -> int | None:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator != "/" or right == 0 or left % right:
        return None
    return left // right


def _claim(value: str) -> dict[str, int | str] | None:
    matches = list(_ARITH_RE.finditer(value))
    if len(matches) != 1:
        return None
    left, operator, right, claimed = matches[0].groups()
    normalized = _normal_operator(operator)
    actual = _actual(int(left), normalized, int(right))
    if actual is None:
        return None
    return {
        "left": int(left),
        "operator": normalized,
        "right": int(right),
        "claimed": int(claimed),
        "actual": actual,
    }


def _prediction_claim(value: str) -> dict[str, int | str] | None:
    text = str(value or "").strip()
    match = _ARITH_RE.fullmatch(text)
    if match is None:
        return None
    return _claim(text)


def _intervention_payload(
    *,
    family: str,
    before: Mapping[str, Any],
    left: int,
    operator: str,
    right: int,
) -> dict[str, Any] | None:
    expected = _actual(left, operator, right)
    baseline_claimed = int(before["claimed"])
    if expected is None or expected == baseline_claimed:
        return None
    payload = {
        "family": family,
        "before": {
            "left": int(before["left"]),
            "operator": str(before["operator"]),
            "right": int(before["right"]),
            "claimed_result": baseline_claimed,
            "actual_result": int(before["actual"]),
        },
        "after": {
            "left": left,
            "operator": operator,
            "right": right,
            "expected_result": expected,
        },
        "expected_consequence_changed": True,
    }
    return {**payload, "intervention_sha256": _sha(payload)}


def _interventions(claim: Mapping[str, Any], *, maximum: int) -> list[dict[str, Any]]:
    left = int(claim["left"])
    right = int(claim["right"])
    operator = str(claim["operator"])
    candidates = [
        _intervention_payload(
            family="left_input_delta",
            before=claim,
            left=left + 1,
            operator=operator,
            right=right,
        ),
        _intervention_payload(
            family="right_input_delta",
            before=claim,
            left=left,
            operator=operator,
            right=right + 1,
        ),
    ]
    for alternative in ("+", "-", "*"):
        if alternative == operator:
            continue
        candidates.append(
            _intervention_payload(
                family="operator_assumption_swap",
                before=claim,
                left=left,
                operator=alternative,
                right=right,
            )
        )
    distinct: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        if row is None or row["intervention_sha256"] in seen:
            continue
        seen.add(row["intervention_sha256"])
        distinct.append(row)
    distinct.sort(
        key=lambda row: (
            _INTERVENTION_FAMILIES.index(str(row["family"])),
            str(row["intervention_sha256"]),
        )
    )
    return distinct[:maximum]


def build_counterfactual_prompt(
    *,
    objective: str,
    claim_text: str,
    claim_sha256: str,
    intervention: Mapping[str, Any],
) -> str:
    """Build an ownership-free, exact-intervention prediction prompt."""

    after = intervention["after"]
    return (
        "You are an independent counterfactual prediction lane. You did not produce "
        "the candidate and receive no ownership identity or prior solver state.\n"
        "Recompute the consequence after the stated intervention. Do not preserve "
        "the original result merely because it appeared in the claim.\n"
        "Return exactly FINAL_ANSWER followed by one JSON object with string keys "
        '"claim_sha256", "intervention_sha256", and "prediction". Prediction must '
        "be one integer equality and contain no prose.\n\n"
        f"PROBLEM:\n{str(objective or '')[:8192]}\n\n"
        f"ANONYMIZED_CLAIM_SHA256: {claim_sha256}\n"
        f"ANONYMIZED_CLAIM:\n{str(claim_text or '')[:512]}\n"
        f"INTERVENTION_SHA256: {intervention['intervention_sha256']}\n"
        f"COUNTERFACTUAL_INPUT: {after['left']} {after['operator']} {after['right']}\n"
    )


def parse_counterfactual_result(
    text: str,
    *,
    claim_sha256: str,
    intervention_sha256: str,
) -> dict[str, str]:
    match = _RESULT_RE.fullmatch(str(text or "").strip())
    if match is None:
        raise ValueError("counterfactual generation did not satisfy the FINAL_ANSWER contract")
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError("counterfactual result is not valid JSON") from exc
    required = {"claim_sha256", "intervention_sha256", "prediction"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("counterfactual result fields do not match contract")
    if any(not isinstance(value[key], str) for key in required):
        raise ValueError("counterfactual result values must be strings")
    if value["claim_sha256"] != claim_sha256:
        raise ValueError("counterfactual result is not bound to the claim")
    if value["intervention_sha256"] != intervention_sha256:
        raise ValueError("counterfactual result is not bound to the intervention")
    if len(value["prediction"]) > 256:
        raise ValueError("counterfactual prediction exceeds 256 characters")
    return {key: value[key] for key in ("claim_sha256", "intervention_sha256", "prediction")}


def _validate_context(value: Any) -> dict[str, Any]:
    fields = {
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
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("counterfactual context fields do not match schema")
    initial = value["initial_cache_offsets"]
    final = value["final_cache_offsets"]
    if (
        value["schema"] != FRESH_CONTEXT_SCHEMA
        or not isinstance(initial, list)
        or not initial
        or any(type(offset) is not int or offset != 0 for offset in initial)
        or not isinstance(final, list)
        or len(final) != len(initial)
        or any(type(offset) is not int or offset < 0 for offset in final)
        or len(set(final)) != 1
        or value["all_initial_offsets_zero"] is not True
        or value["solver_context_imported"] is not False
        or value["parameter_relation"] != "shared_resident_checkpoint"
        or type(value["prompt_token_count"]) is not int
        or value["prompt_token_count"] < 1
        or type(value["generated_token_count"]) is not int
        or value["generated_token_count"] < 1
        or final[0] < value["prompt_token_count"]
        or not isinstance(value["termination"], str)
        or not value["termination"]
    ):
        raise ValueError("counterfactual context isolation is invalid")
    return dict(value)


def _evaluate_prediction(
    prediction: str,
    *,
    intervention: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    parsed = _prediction_claim(prediction)
    after = intervention["after"]
    if parsed is None or (
        parsed["left"],
        parsed["operator"],
        parsed["right"],
    ) != (after["left"], after["operator"], after["right"]):
        return "abstained", {"reason": "prediction_not_exactly_machine_checkable"}
    predicted = int(parsed["claimed"])
    expected = int(after["expected_result"])
    baseline = int(intervention["before"]["claimed_result"])
    if predicted == expected:
        outcome = "correct_change"
    elif predicted == baseline:
        outcome = "invariant_failure"
    else:
        outcome = "incorrect_change"
    return outcome, {
        "predicted_result": predicted,
        "expected_result": expected,
        "baseline_result": baseline,
        "prediction_sha256": _text_sha(prediction),
    }


def _attempt(
    *,
    objective: str,
    atom: Mapping[str, Any],
    claim_text: str,
    intervention: Mapping[str, Any],
    generate: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    prompt = build_counterfactual_prompt(
        objective=objective,
        claim_text=claim_text,
        claim_sha256=atom["atom_sha256"],
        intervention=intervention,
    )
    base = {
        "atom_id": atom["atom_id"],
        "atom_sha256": atom["atom_sha256"],
        "claim_text": claim_text,
        "claim_text_sha256": _text_sha(claim_text),
        "intervention": dict(intervention),
        "prompt_sha256": _text_sha(prompt),
    }
    try:
        generated = generate(prompt)
        if not isinstance(generated, Mapping):
            raise TypeError("counterfactual generator result must be a mapping")
    except (OSError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
        return {
            **base,
            "generation_status": "abstained",
            "generated_output_sha256": "",
            "prediction_text": "",
            "prediction_sha256": "",
            "context": {},
            "outcome": "abstained",
            "evidence": {"reason": f"{type(exc).__name__}:{exc}"[:240]},
        }
    generated_text = str(generated.get("text") or "")
    context = dict(generated.get("context") or {})
    try:
        parsed = parse_counterfactual_result(
            generated_text,
            claim_sha256=atom["atom_sha256"],
            intervention_sha256=intervention["intervention_sha256"],
        )
        outcome, evidence = _evaluate_prediction(
            parsed["prediction"],
            intervention=intervention,
        )
    except (KeyError, TypeError, ValueError) as exc:
        outcome = "abstained"
        evidence = {"reason": f"contract_refused:{type(exc).__name__}:{exc}"[:240]}
        prediction_text = ""
        prediction_sha256 = ""
    else:
        prediction_text = parsed["prediction"]
        prediction_sha256 = _text_sha(parsed["prediction"])
    return {
        **base,
        "generation_status": "complete",
        "generated_output_sha256": _text_sha(generated_text),
        "prediction_text": prediction_text,
        "prediction_sha256": prediction_sha256,
        "context": context,
        "outcome": outcome,
        "evidence": evidence,
    }


def run_counterfactual_verifier(
    candidates: Mapping[int, str],
    *,
    objective: str,
    task_scores: Mapping[int, float],
    selected_branch: int,
    generate: Callable[[str], Mapping[str, Any]],
    max_atoms: int = 1,
    max_interventions: int = 2,
    tie_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Resolve only exact top-score ties using equal counterfactual coverage."""

    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("counterfactual candidates are required")
    branches = sorted(candidates)
    if (
        not 1 <= len(branches) <= 8
        or any(type(branch) is not int for branch in branches)
        or branches != list(range(len(branches)))
        or set(task_scores) != set(branches)
        or any(not isinstance(candidates[branch], str) for branch in branches)
    ):
        raise ValueError("counterfactual branch inventory differs")
    if type(selected_branch) is not int or selected_branch not in candidates:
        raise ValueError("counterfactual selected branch is invalid")
    if type(max_atoms) is not int or not 1 <= max_atoms <= 4:
        raise ValueError("counterfactual max_atoms must be inside [1, 4]")
    if type(max_interventions) is not int or not 1 <= max_interventions <= 3:
        raise ValueError("counterfactual max_interventions must be inside [1, 3]")
    if isinstance(tie_tolerance, bool) or not isinstance(tie_tolerance, (int, float)):
        raise ValueError("counterfactual tie tolerance is invalid")
    tolerance = float(tie_tolerance)
    if not 0.0 <= tolerance <= 0.01:
        raise ValueError("counterfactual tie tolerance is outside [0, 0.01]")
    if any(
        isinstance(task_scores[branch], bool)
        or not isinstance(task_scores[branch], (int, float))
        for branch in branches
    ):
        raise ValueError("counterfactual task scores are invalid")
    normalized_scores = {branch: round(float(task_scores[branch]), 10) for branch in branches}
    if any(not 0.0 <= score <= 1.0 for score in normalized_scores.values()):
        raise ValueError("counterfactual task scores are outside [0, 1]")
    normalized_objective = str(objective or "")[:8192]
    top = max(normalized_scores.values())
    tied = [
        branch for branch in branches if top - normalized_scores[branch] <= tolerance
    ]
    if selected_branch not in tied:
        raise ValueError("counterfactual source winner is not top-scoring")

    branch_rows: list[dict[str, Any]] = []
    for branch in tied:
        candidate = candidates[branch]
        if len(candidate) > 16_384:
            raise ValueError("counterfactual candidate exceeds 16384 characters")
        atomic = build_atomic_decomposition(candidate, objective=normalized_objective)
        visible = validate_atomic_decomposition(
            atomic,
            candidate=candidate,
            objective=normalized_objective,
        )
        arithmetic_atoms: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
        if len(tied) >= 2:
            for atom in visible["atoms"]:
                text = candidate[atom["start"] : atom["end"]]
                claim = _claim(text)
                if claim is not None and len(text) <= 512:
                    arithmetic_atoms.append((atom, text, claim))
        attempts: list[dict[str, Any]] = []
        for atom, text, claim in arithmetic_atoms[:max_atoms]:
            for intervention in _interventions(claim, maximum=max_interventions):
                attempts.append(
                    _attempt(
                        objective=normalized_objective,
                        atom=atom,
                        claim_text=text,
                        intervention=intervention,
                        generate=generate,
                    )
                )
        admitted = [row for row in attempts if row["outcome"] != "abstained"]
        correct = sum(row["outcome"] == "correct_change" for row in admitted)
        invariant = sum(row["outcome"] == "invariant_failure" for row in admitted)
        incorrect = sum(row["outcome"] == "incorrect_change" for row in admitted)
        branch_rows.append(
            {
                "branch": branch,
                "candidate_text": candidate,
                "candidate_sha256": _text_sha(candidate),
                "atomic_decomposition": atomic,
                "attempts": attempts,
                "attempted": len(attempts),
                "admitted": len(admitted),
                "correct_changes": correct,
                "invariant_failures": invariant,
                "incorrect_changes": incorrect,
                "complete_coverage": bool(attempts) and len(admitted) == len(attempts),
                "robustness_score": (
                    round(correct / len(admitted), 10) if admitted else None
                ),
            }
        )

    comparable = (
        len(tied) >= 2
        and len(branch_rows) == len(tied)
        and all(row["complete_coverage"] for row in branch_rows)
        and len({row["attempted"] for row in branch_rows}) == 1
        and branch_rows[0]["attempted"] > 0
    )
    ranking = sorted(
        branch_rows,
        key=lambda row: (*_rank_evidence(row), int(row["branch"])),
    )
    distinguished = (
        comparable
        and sum(
            _rank_evidence(row) == _rank_evidence(ranking[0])
            for row in branch_rows
        )
        == 1
    )
    proposed = int(ranking[0]["branch"]) if distinguished else selected_branch
    effect = (
        "winner_replaced"
        if distinguished and proposed != selected_branch
        else "winner_confirmed"
        if distinguished
        else "none"
    )
    payload = {
        "schema": COUNTERFACTUAL_VERIFIER_SCHEMA,
        "implementation_sha256": _implementation_sha256(),
        "objective_text": normalized_objective,
        "objective_sha256": _text_sha(normalized_objective),
        "parameter_independence": False,
        "context_independence": True,
        "authority_mode": "equal_task_score_counterfactual_tiebreak_only",
        "tie_tolerance": tolerance,
        "max_atoms": max_atoms,
        "max_interventions": max_interventions,
        "task_scores": {str(key): value for key, value in normalized_scores.items()},
        "source_selected_branch": selected_branch,
        "tied_branches": tied,
        "branches": branch_rows,
        "all_tied_branches_covered": comparable,
        "selection_authority_admitted": distinguished,
        "selected_branch": proposed,
        "selection_effect": effect,
    }
    return validate_counterfactual_verifier_envelope(
        {**payload, "receipt_sha256": _sha(payload)}
    )


def validate_counterfactual_verifier_envelope(value: Any) -> dict[str, Any]:
    """Reconstruct the public counterfactual receipt and selection effect."""

    fields = {
        "schema",
        "implementation_sha256",
        "objective_text",
        "objective_sha256",
        "parameter_independence",
        "context_independence",
        "authority_mode",
        "tie_tolerance",
        "max_atoms",
        "max_interventions",
        "task_scores",
        "source_selected_branch",
        "tied_branches",
        "branches",
        "all_tied_branches_covered",
        "selection_authority_admitted",
        "selected_branch",
        "selection_effect",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("counterfactual verifier fields do not match schema")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != _sha(payload):
        raise ValueError("counterfactual verifier commitment mismatch")
    if (
        value["schema"] != COUNTERFACTUAL_VERIFIER_SCHEMA
        or value["implementation_sha256"] != _implementation_sha256()
        or value["parameter_independence"] is not False
        or value["context_independence"] is not True
        or value["authority_mode"] != "equal_task_score_counterfactual_tiebreak_only"
        or not isinstance(value["objective_text"], str)
        or len(value["objective_text"]) > 8192
        or value["objective_sha256"] != _text_sha(value["objective_text"])
        or not _SHA256_RE.fullmatch(str(value["objective_sha256"]))
    ):
        raise ValueError("counterfactual verifier identity or authority claim is invalid")
    tolerance = value["tie_tolerance"]
    max_atoms = value["max_atoms"]
    max_interventions = value["max_interventions"]
    scores = value["task_scores"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not 0.0 <= float(tolerance) <= 0.01
        or type(max_atoms) is not int
        or not 1 <= max_atoms <= 4
        or type(max_interventions) is not int
        or not 1 <= max_interventions <= 3
        or not isinstance(scores, Mapping)
        or not scores
        or len(scores) > 8
    ):
        raise ValueError("counterfactual verifier task scores are invalid")
    normalized_scores: dict[int, float] = {}
    for key, score in scores.items():
        try:
            branch = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError("counterfactual task-score branch is invalid") from exc
        if (
            str(branch) != key
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError("counterfactual task-score value is invalid")
        normalized_scores[branch] = float(score)
    if sorted(normalized_scores) != list(range(len(normalized_scores))):
        raise ValueError("counterfactual task-score inventory is not contiguous")
    top = max(normalized_scores.values())
    expected_tied = [
        branch
        for branch in sorted(normalized_scores)
        if top - normalized_scores[branch] <= float(tolerance)
    ]
    source_selected = value["source_selected_branch"]
    if (
        value["tied_branches"] != expected_tied
        or not isinstance(value["tied_branches"], list)
        or any(type(branch) is not int for branch in value["tied_branches"])
        or type(source_selected) is not int
        or source_selected not in expected_tied
    ):
        raise ValueError("counterfactual tie boundary is invalid")
    rows = value["branches"]
    if (
        not isinstance(rows, list)
        or any(not isinstance(row, Mapping) for row in rows)
        or [row["branch"] if "branch" in row else None for row in rows] != expected_tied
        or any(type(row["branch"]) is not int for row in rows)
    ):
        raise ValueError("counterfactual branch coverage differs")
    reconstructed: list[dict[str, Any]] = []
    for row in rows:
        row_fields = {
            "branch",
            "candidate_text",
            "candidate_sha256",
            "atomic_decomposition",
            "attempts",
            "attempted",
            "admitted",
            "correct_changes",
            "invariant_failures",
            "incorrect_changes",
            "complete_coverage",
            "robustness_score",
        }
        if not isinstance(row, Mapping) or set(row) != row_fields:
            raise ValueError("counterfactual branch fields do not match schema")
        candidate_text = row["candidate_text"]
        if not isinstance(candidate_text, str):
            raise ValueError("counterfactual candidate text is invalid")
        if len(candidate_text) > 16_384:
            raise ValueError("counterfactual candidate exceeds 16384 characters")
        atomic = validate_atomic_decomposition(
            row["atomic_decomposition"],
            candidate=candidate_text,
            objective=value["objective_text"],
        )
        attempts = row["attempts"]
        if not isinstance(attempts, list) or len(attempts) > 12:
            raise ValueError("counterfactual attempt inventory is invalid")
        outcomes: list[str] = []
        seen_interventions: set[str] = set()
        for attempt in attempts:
            attempt_fields = {
                "atom_id",
                "atom_sha256",
                "claim_text",
                "claim_text_sha256",
                "intervention",
                "prompt_sha256",
                "generation_status",
                "generated_output_sha256",
                "prediction_text",
                "prediction_sha256",
                "context",
                "outcome",
                "evidence",
            }
            if not isinstance(attempt, Mapping) or set(attempt) != attempt_fields:
                raise ValueError("counterfactual attempt fields do not match schema")
            claim_text = attempt["claim_text"]
            intervention = attempt["intervention"]
            if (
                not isinstance(claim_text, str)
                or len(claim_text) > 512
                or attempt["claim_text_sha256"] != _text_sha(claim_text)
                or not isinstance(intervention, Mapping)
                or set(intervention)
                != {
                    "family",
                    "before",
                    "after",
                    "expected_consequence_changed",
                    "intervention_sha256",
                }
                or intervention.get("family") not in _INTERVENTION_FAMILIES
                or intervention.get("intervention_sha256")
                != _sha(
                    {
                        key: intervention[key]
                        for key in (
                            "family",
                            "before",
                            "after",
                            "expected_consequence_changed",
                        )
                    }
                )
                or intervention["intervention_sha256"] in seen_interventions
                or attempt["outcome"] not in _OUTCOMES
                or attempt["generation_status"] not in {"complete", "abstained"}
            ):
                raise ValueError("counterfactual attempt identity is invalid")
            seen_interventions.add(intervention["intervention_sha256"])
            claim = _claim(claim_text)
            expected_interventions = {
                item["intervention_sha256"]: item
                for item in _interventions(claim or {}, maximum=3)
            } if claim is not None else {}
            source_atom = next(
                (
                    atom
                    for atom in atomic["atoms"]
                    if atom["atom_id"] == attempt["atom_id"]
                ),
                None,
            )
            if (
                source_atom is None
                or attempt["atom_sha256"] != source_atom["atom_sha256"]
                or _text_sha(claim_text) != source_atom["text_sha256"]
                or intervention["intervention_sha256"] not in expected_interventions
                or intervention != expected_interventions[intervention["intervention_sha256"]]
                or attempt["prompt_sha256"]
                != _text_sha(
                    build_counterfactual_prompt(
                        objective=value["objective_text"],
                        claim_text=claim_text,
                        claim_sha256=attempt["atom_sha256"],
                        intervention=intervention,
                    )
                )
            ):
                raise ValueError("counterfactual claim or intervention does not reconstruct")
            if attempt["generation_status"] == "complete":
                _validate_context(attempt["context"])
                if (
                    not _SHA256_RE.fullmatch(str(attempt["generated_output_sha256"]))
                    or not isinstance(attempt["prediction_text"], str)
                    or len(attempt["prediction_text"]) > 256
                    or (
                        attempt["outcome"] != "abstained"
                        and (
                            not _SHA256_RE.fullmatch(str(attempt["prediction_sha256"]))
                            or attempt["prediction_sha256"]
                            != _text_sha(attempt["prediction_text"])
                        )
                    )
                ):
                    raise ValueError("counterfactual generation evidence is invalid")
                evidence = attempt["evidence"]
                if attempt["outcome"] != "abstained":
                    outcome, reconstructed_evidence = _evaluate_prediction(
                        attempt["prediction_text"],
                        intervention=intervention,
                    )
                    if (
                        outcome != attempt["outcome"]
                        or evidence != reconstructed_evidence
                        or evidence["prediction_sha256"] != attempt["prediction_sha256"]
                    ):
                        raise ValueError("counterfactual outcome does not reconstruct")
                elif (
                    attempt["prediction_text"] != ""
                    or attempt["prediction_sha256"] != ""
                    or not isinstance(evidence, Mapping)
                    or set(evidence) != {"reason"}
                    or not isinstance(evidence["reason"], str)
                    or not evidence["reason"]
                    or len(evidence["reason"]) > 240
                ):
                    raise ValueError("abstained counterfactual contract retained prediction")
            elif (
                attempt["context"] != {}
                or attempt["generated_output_sha256"] != ""
                or attempt["prediction_text"] != ""
                or attempt["prediction_sha256"] != ""
                or attempt["outcome"] != "abstained"
                or not isinstance(attempt["evidence"], Mapping)
                or set(attempt["evidence"]) != {"reason"}
                or not isinstance(attempt["evidence"]["reason"], str)
                or not attempt["evidence"]["reason"]
                or len(attempt["evidence"]["reason"]) > 240
            ):
                raise ValueError("abstained counterfactual attempt claimed evidence")
            outcomes.append(str(attempt["outcome"]))
        expected_attempts: list[tuple[str, str]] = []
        visible = validate_atomic_decomposition(
            row["atomic_decomposition"],
            candidate=candidate_text,
            objective=value["objective_text"],
        )
        arithmetic_atoms: list[tuple[dict[str, Any], dict[str, Any]]] = []
        if len(expected_tied) >= 2:
            for atom in visible["atoms"]:
                claim_text = candidate_text[atom["start"] : atom["end"]]
                claim = _claim(claim_text)
                if claim is not None and len(claim_text) <= 512:
                    arithmetic_atoms.append((atom, claim))
        for atom, claim in arithmetic_atoms[:max_atoms]:
            expected_attempts.extend(
                (atom["atom_id"], intervention["intervention_sha256"])
                for intervention in _interventions(
                    claim,
                    maximum=max_interventions,
                )
            )
        observed_attempts = [
            (
                str(attempt["atom_id"]),
                str(attempt["intervention"]["intervention_sha256"]),
            )
            for attempt in attempts
        ]
        if observed_attempts != expected_attempts:
            raise ValueError("counterfactual intervention order was cherry-picked")
        admitted = sum(outcome != "abstained" for outcome in outcomes)
        correct = outcomes.count("correct_change")
        invariant = outcomes.count("invariant_failure")
        incorrect = outcomes.count("incorrect_change")
        complete = bool(attempts) and admitted == len(attempts)
        score = round(correct / admitted, 10) if admitted else None
        if (
            row["candidate_sha256"] != _text_sha(candidate_text)
            or row["candidate_sha256"] != atomic["source_sha256"]
            or any(
                type(row[field]) is not int
                for field in (
                    "attempted",
                    "admitted",
                    "correct_changes",
                    "invariant_failures",
                    "incorrect_changes",
                )
            )
            or not (
                row["robustness_score"] is None
                or (
                    isinstance(row["robustness_score"], float)
                    and 0.0 <= row["robustness_score"] <= 1.0
                )
            )
            or row["attempted"] != len(attempts)
            or row["admitted"] != admitted
            or row["correct_changes"] != correct
            or row["invariant_failures"] != invariant
            or row["incorrect_changes"] != incorrect
            or row["complete_coverage"] is not complete
            or row["robustness_score"] != score
        ):
            raise ValueError("counterfactual branch aggregate does not reconstruct")
        reconstructed.append(dict(row))
    comparable = (
        len(expected_tied) >= 2
        and all(row["complete_coverage"] for row in reconstructed)
        and len({row["attempted"] for row in reconstructed}) == 1
        and reconstructed[0]["attempted"] > 0
    )
    ranking = sorted(
        reconstructed,
        key=lambda row: (*_rank_evidence(row), int(row["branch"])),
    )
    distinguished = (
        comparable
        and sum(
            _rank_evidence(row) == _rank_evidence(ranking[0])
            for row in reconstructed
        )
        == 1
    )
    selected = int(ranking[0]["branch"]) if distinguished else source_selected
    effect = (
        "winner_replaced"
        if distinguished and selected != source_selected
        else "winner_confirmed"
        if distinguished
        else "none"
    )
    if (
        value["all_tied_branches_covered"] is not comparable
        or value["selection_authority_admitted"] is not distinguished
        or value["selected_branch"] != selected
        or value["selection_effect"] != effect
    ):
        raise ValueError("counterfactual selection effect does not reconstruct")
    return dict(value)


__all__ = [
    "COUNTERFACTUAL_VERIFIER_SCHEMA",
    "build_counterfactual_prompt",
    "parse_counterfactual_result",
    "run_counterfactual_verifier",
    "validate_counterfactual_verifier_envelope",
]
