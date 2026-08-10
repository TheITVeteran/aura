"""Verifier-authorized teaching signals for bounded neural plasticity.

A teaching event is not evidence that the neural system solved a task.  It is
an authority bridge from an independently verified correction to a private,
query-scoped learning target.  Public receipts commit to the teacher and target
without carrying answer text; only a fresh decode after teacher removal may
establish a causal mechanism result.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from core.brain.llm.latent_cortex.atomic_decomposition import (
    build_atomic_decomposition,
)
from core.brain.llm.latent_cortex.deterministic_verifier_router import (
    build_deterministic_router_receipt,
)
from core.brain.llm.latent_cortex.objective_program_verifier import (
    solve_objective_program,
    validate_objective_program_solution,
    verify_objective_program,
)

TEACHING_EVENT_SCHEMA = "aura.rlc.teaching_event.v1"
_ALLOWED_PLASTICITY_SCOPES = (
    "activation_state",
    "episodic_fast_weights",
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_exact_objective_teaching_event(
    *,
    objective: str,
    incumbent_candidate: str,
    source_state_sha256: str,
    tokenizer: Any,
    structural_diversity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    """Create one private target from a public-objective exact solution.

    The returned target tokens remain worker-private.  The event and admission
    receipts contain commitments only.  A caller must decode again from the
    original query state without adding the solution to model context.
    """

    if not isinstance(objective, str) or not objective:
        raise ValueError("teaching event objective is empty")
    if not isinstance(incumbent_candidate, str):
        raise TypeError("teaching event incumbent candidate must be text")
    if not _is_sha256(source_state_sha256):
        raise ValueError("teaching event source state commitment is invalid")
    solved = solve_objective_program(objective)
    if solved is None:
        raise ValueError("exact objective teacher is unavailable")
    full_solution, solution_receipt = solved
    validate_objective_program_solution(
        solution_receipt,
        objective=objective,
        candidate=full_solution,
    )
    teacher_candidate = full_solution.rsplit("\n", 1)[-1]
    teacher_verdict = verify_objective_program(
        teacher_candidate,
        objective=objective,
    )
    if teacher_verdict is None or teacher_verdict["outcome"] != "verified":
        raise RuntimeError("exact objective teacher did not verify")
    incumbent_verdict = verify_objective_program(
        incumbent_candidate,
        objective=objective,
    )
    if incumbent_verdict is not None and incumbent_verdict["outcome"] == "verified":
        raise ValueError("teaching event incumbent is already objective-verified")
    if _text_sha256(incumbent_candidate) == _text_sha256(teacher_candidate):
        raise ValueError("teaching event does not contain a correction")

    atomic = build_atomic_decomposition(teacher_candidate, objective=objective)
    router = build_deterministic_router_receipt(
        teacher_candidate,
        objective=objective,
        atomic_receipt=atomic,
    )
    from core.brain.llm.latent_cortex.fast_weight_learning import (
        build_fast_weight_admission,
    )

    admission, target_tokens = build_fast_weight_admission(
        {
            "checks": {
                "atomic_decomposition": {"receipt": atomic},
                "deterministic_router": {"receipt": router},
            }
        },
        candidate=teacher_candidate,
        objective=objective,
        evaluation_index=0,
        tokenizer=tokenizer,
        structural_diversity=structural_diversity,
    )
    if admission["admitted"] is not True or not target_tokens:
        raise RuntimeError("exact objective teacher was not admitted for plasticity")
    payload = {
        "schema": TEACHING_EVENT_SCHEMA,
        "objective_sha256": _text_sha256(objective),
        "source_state_sha256": source_state_sha256,
        "incumbent_candidate_sha256": _text_sha256(incumbent_candidate),
        "teacher_candidate_sha256": _text_sha256(teacher_candidate),
        "producer_solution_receipt_sha256": solution_receipt["receipt_sha256"],
        "teacher_verifier_receipt_sha256": teacher_verdict["receipt_sha256"],
        "incumbent_verifier_receipt_sha256": (
            incumbent_verdict["receipt_sha256"] if incumbent_verdict is not None else ""
        ),
        "incumbent_outcome": (
            incumbent_verdict["outcome"] if incumbent_verdict is not None else "unrecognized"
        ),
        "verifier_family": "exact_objective_program",
        "critic_recalibration_sha256": admission["critic_recalibration"][
            "receipt_sha256"
        ],
        "pseudo_label_admission_sha256": admission["pseudo_label_admission"][
            "receipt_sha256"
        ],
        "fast_weight_admission_sha256": admission["receipt_sha256"],
        "target_tokens_sha256": admission["target_tokens_sha256"],
        "target_token_count": admission["target_token_count"],
        "confidence_lower_95": admission["pseudo_label_admission"][
            "confidence_lower_95"
        ],
        "correction_type": "verified_exact_objective_replacement",
        "allowed_plasticity_scopes": list(_ALLOWED_PLASTICITY_SCOPES),
        "lifetime": "single_query_erase_required",
        "teacher_context_policy": "private_target_only_never_model_context",
        "teacher_removed_before_causal_probe_required": True,
        "capability_claim_authority": False,
        "claim_boundary": "mechanism_diagnostic_not_teacher_free_capability_evidence",
    }
    receipt = {**payload, "receipt_sha256": _canonical_sha256(payload)}
    validate_teaching_event(
        receipt,
        admission=admission,
        expected_objective_sha256=_text_sha256(objective),
        expected_source_state_sha256=source_state_sha256,
    )
    return receipt, admission, target_tokens


def validate_teaching_event(
    value: Any,
    *,
    admission: Mapping[str, Any],
    expected_objective_sha256: str | None = None,
    expected_source_state_sha256: str | None = None,
) -> dict[str, Any]:
    fields = {
        "schema",
        "objective_sha256",
        "source_state_sha256",
        "incumbent_candidate_sha256",
        "teacher_candidate_sha256",
        "producer_solution_receipt_sha256",
        "teacher_verifier_receipt_sha256",
        "incumbent_verifier_receipt_sha256",
        "incumbent_outcome",
        "verifier_family",
        "critic_recalibration_sha256",
        "pseudo_label_admission_sha256",
        "fast_weight_admission_sha256",
        "target_tokens_sha256",
        "target_token_count",
        "confidence_lower_95",
        "correction_type",
        "allowed_plasticity_scopes",
        "lifetime",
        "teacher_context_policy",
        "teacher_removed_before_causal_probe_required",
        "capability_claim_authority",
        "claim_boundary",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("teaching event fields differ")
    payload = {name: value[name] for name in fields - {"receipt_sha256"}}
    if value["receipt_sha256"] != _canonical_sha256(payload):
        raise ValueError("teaching event commitment mismatch")
    from core.brain.llm.latent_cortex.fast_weight_learning import (
        validate_fast_weight_admission,
    )

    admitted = validate_fast_weight_admission(admission)
    digest_fields = (
        "objective_sha256",
        "source_state_sha256",
        "incumbent_candidate_sha256",
        "teacher_candidate_sha256",
        "producer_solution_receipt_sha256",
        "teacher_verifier_receipt_sha256",
        "critic_recalibration_sha256",
        "pseudo_label_admission_sha256",
        "fast_weight_admission_sha256",
        "target_tokens_sha256",
    )
    if any(not _is_sha256(value[name]) for name in digest_fields):
        raise ValueError("teaching event contains an invalid commitment")
    incumbent_receipt_sha = value["incumbent_verifier_receipt_sha256"]
    if incumbent_receipt_sha and not _is_sha256(incumbent_receipt_sha):
        raise ValueError("teaching event incumbent verifier commitment is invalid")
    if (
        value["schema"] != TEACHING_EVENT_SCHEMA
        or value["verifier_family"] != "exact_objective_program"
        or value["correction_type"] != "verified_exact_objective_replacement"
        or value["allowed_plasticity_scopes"] != list(_ALLOWED_PLASTICITY_SCOPES)
        or value["lifetime"] != "single_query_erase_required"
        or value["teacher_context_policy"]
        != "private_target_only_never_model_context"
        or value["teacher_removed_before_causal_probe_required"] is not True
        or value["capability_claim_authority"] is not False
        or value["claim_boundary"]
        != "mechanism_diagnostic_not_teacher_free_capability_evidence"
        or value["incumbent_outcome"] not in {"refuted", "unrecognized"}
        or type(value["target_token_count"]) is not int
        or value["target_token_count"] <= 0
        or not isinstance(value["confidence_lower_95"], (int, float))
        or isinstance(value["confidence_lower_95"], bool)
    ):
        raise ValueError("teaching event policy is invalid")
    if expected_objective_sha256 is not None and (
        value["objective_sha256"] != expected_objective_sha256
    ):
        raise ValueError("teaching event objective differs")
    if expected_source_state_sha256 is not None and (
        value["source_state_sha256"] != expected_source_state_sha256
    ):
        raise ValueError("teaching event source state differs")
    if (
        admitted["admitted"] is not True
        or admitted["critic_recalibration"]["verifier_family"]
        != "exact_objective_program"
        or value["critic_recalibration_sha256"]
        != admitted["critic_recalibration"]["receipt_sha256"]
        or value["pseudo_label_admission_sha256"]
        != admitted["pseudo_label_admission"]["receipt_sha256"]
        or value["fast_weight_admission_sha256"] != admitted["receipt_sha256"]
        or value["target_tokens_sha256"] != admitted["target_tokens_sha256"]
        or value["target_token_count"] != admitted["target_token_count"]
        or value["confidence_lower_95"]
        != admitted["pseudo_label_admission"]["confidence_lower_95"]
    ):
        raise ValueError("teaching event admission binding differs")
    return dict(value)


__all__ = [
    "TEACHING_EVENT_SCHEMA",
    "build_exact_objective_teaching_event",
    "validate_teaching_event",
]
