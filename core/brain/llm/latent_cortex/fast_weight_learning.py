"""Proof contract for query-scoped fast-weight learning.

Temporary weights are useful only when three claims can be reconstructed:
the learning target came from exact evidence, attachment was functionally
identity, and the adapted function improved a matched query probe before it
was allowed to influence the answer. This module keeps those claims separate
from private latent values and independently validates the public envelope.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from core.brain.llm.latent_cortex.atomic_decomposition import (
    validate_atomic_decomposition,
    validate_atomic_decomposition_envelope,
)
from core.brain.llm.latent_cortex.deterministic_verifier_router import (
    validate_deterministic_router_envelope,
)
from core.brain.llm.latent_cortex.test_time_training import (
    build_critic_recalibration_receipt,
    build_pseudo_label_admission,
    build_test_time_training_receipt,
    validate_critic_recalibration_receipt,
    validate_pseudo_label_admission,
    validate_test_time_training_receipt,
)

ADMISSION_SCHEMA = "aura.rlc.fast_weight_admission.v1"
LEARNING_SCHEMA = "aura.rlc.fast_weight_learning.v1"
LEASE_SCHEMA = "aura.rlc.fast_weight_model_lease.v1"
MAX_TARGET_TOKENS = 256

_ADMISSION_POLICY = {
    "source": "same_query_pre_attach_probe",
    "evidence": "atomic_exact_local_verified_atoms_only",
    "refuted_atoms_allowed": False,
    "unsupported_atoms_allowed": False,
    "unknown_atoms_are_excluded_from_target": True,
    "max_target_tokens": MAX_TARGET_TOKENS,
    "pseudo_label_authority": (
        "held_out_recalibration_lower_bound_above_0.90_exact_verifier_only"
    ),
}
_LEARNING_POLICY = {
    "attach_identity": "measured_exact_full_stack_probe",
    "model_mutation": "exclusive_process_local_query_lease",
    "acceptance": "strict_matched_probe_improvement_and_token_change",
    "protected_behavior": "capability_canary_nonregression",
    "cleanup": "exact_detach_erase_and_lease_release",
}

_ADMISSION_FIELDS = {
    "schema",
    "policy_sha256",
    "candidate_checked",
    "admitted",
    "reason",
    "source_sha256",
    "objective_sha256",
    "evaluation_index",
    "atomic_decomposition",
    "deterministic_router",
    "evidence_atom_ids",
    "evidence_atom_sha256s",
    "evidence_text_sha256",
    "target_tokens_sha256",
    "target_token_count",
    "critic_recalibration",
    "pseudo_label_admission",
    "receipt_sha256",
}
_LEARNING_FIELDS = {
    "schema",
    "policy_sha256",
    "episode_id",
    "input_tokens_sha256",
    "selected_branch",
    "winner_state_sha256",
    "admission",
    "lease",
    "attach_identity",
    "optimization",
    "controls",
    "causal_probe",
    "final_answer",
    "cleanup",
    "disposition",
    "receipt_sha256",
}
_DISPOSITIONS = {
    "not_admitted_high_confidence_evidence_absent",
    "rejected_no_accepted_step",
    "rejected_capability_regression",
    "rejected_no_causal_effect",
    "rejected_non_improvement",
    "rejected_matched_control",
    "rejected_verifier_unavailable",
    "rejected_state_lineage_changed",
    "accepted_causal_improvement",
    "accepted_probe_not_output_under_incumbent_policy",
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def token_sequence_sha256(tokens: Sequence[int]) -> str:
    normalized = list(tokens)
    if any(type(token) is not int or token < 0 for token in normalized):
        raise ValueError("token sequence contains an invalid token")
    return _canonical_sha256(normalized)


def _admission_reason(
    *,
    atomic_admissible: bool,
    verified: int,
    refuted: int,
    unsupported: int,
    target_token_count: int,
    pseudo_label_admitted: bool,
    pseudo_label_reason: str,
) -> str:
    if not atomic_admissible:
        return "atomic_decomposition_unproven"
    if refuted:
        return "deterministic_evidence_refuted"
    if unsupported:
        return "unsupported_evidence_dependency"
    if not verified:
        return "no_exact_local_evidence"
    if target_token_count <= 0:
        return "evidence_target_tokenization_empty"
    if target_token_count > MAX_TARGET_TOKENS:
        return "evidence_target_exceeds_bound"
    if not pseudo_label_admitted:
        return pseudo_label_reason
    return "admitted_exact_local_evidence"


def unavailable_admission(
    *,
    source_sha256: str,
    objective_sha256: str,
    reason: str,
) -> dict[str, Any]:
    if not _is_sha256(source_sha256) or not _is_sha256(objective_sha256):
        raise ValueError("unavailable admission requires source commitments")
    if reason not in {
        "verifier_unavailable",
        "verifier_provider_untrusted",
        "candidate_evaluation_unavailable",
    }:
        raise ValueError("unavailable admission reason is invalid")
    payload = {
        "schema": ADMISSION_SCHEMA,
        "policy_sha256": _canonical_sha256(_ADMISSION_POLICY),
        "candidate_checked": False,
        "admitted": False,
        "reason": reason,
        "source_sha256": source_sha256,
        "objective_sha256": objective_sha256,
        "evaluation_index": -1,
        "atomic_decomposition": {},
        "deterministic_router": {},
        "evidence_atom_ids": [],
        "evidence_atom_sha256s": [],
        "evidence_text_sha256": _text_sha256(""),
        "target_tokens_sha256": token_sequence_sha256([]),
        "target_token_count": 0,
        "critic_recalibration": {},
        "pseudo_label_admission": {},
    }
    return {**payload, "receipt_sha256": _canonical_sha256(payload)}


def build_fast_weight_admission(
    evaluation: Mapping[str, Any],
    *,
    candidate: str,
    objective: str,
    evaluation_index: int,
    tokenizer: Any,
    structural_diversity: Mapping[str, Any],
) -> tuple[dict[str, Any], list[int]]:
    """Extract a bounded private target from exact verified candidate atoms."""

    if not isinstance(evaluation, Mapping):
        raise TypeError("fast-weight admission evaluation must be a mapping")
    if type(evaluation_index) is not int or evaluation_index < 0:
        raise ValueError("fast-weight evaluation index is invalid")
    checks = evaluation.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("fast-weight evaluation lacks verifier checks")
    atomic_check = checks.get("atomic_decomposition")
    router_check = checks.get("deterministic_router")
    if not isinstance(atomic_check, Mapping) or not isinstance(router_check, Mapping):
        raise ValueError("fast-weight evaluation lacks exact evidence surfaces")
    atomic_receipt = atomic_check.get("receipt")
    router_receipt = router_check.get("receipt")
    atomic = validate_atomic_decomposition(
        atomic_receipt,
        candidate=candidate,
        objective=objective,
    )
    router = validate_deterministic_router_envelope(
        router_receipt,
        atomic_receipt=atomic,
    )
    verified_rows = [row for row in router["routes"] if row["outcome"] == "verified"]
    atom_by_id = {row["atom_id"]: row for row in atomic["atoms"]}
    evidence_fragments = [
        candidate[
            int(atom_by_id[row["atom_id"]]["start"]) : int(
                atom_by_id[row["atom_id"]]["end"]
            )
        ]
        for row in verified_rows
    ]
    evidence_text = "\n".join(evidence_fragments)
    raw_tokens = tokenizer.encode(evidence_text, add_special_tokens=False)
    if not isinstance(raw_tokens, (list, tuple)):
        raise TypeError("fast-weight evidence tokenizer returned a non-sequence")
    target_tokens = [int(token) for token in raw_tokens]
    if any(type(token) is not int or token < 0 for token in raw_tokens):
        raise ValueError("fast-weight evidence tokenizer returned an invalid token")
    counts = router["counts"]
    critic_recalibration = build_critic_recalibration_receipt()
    pseudo_label_admission = build_pseudo_label_admission(
        router_receipt=router,
        atomic_receipt=atomic,
        source_sha256=atomic["source_sha256"],
        structural_diversity=structural_diversity,
        critic_recalibration=critic_recalibration,
    )
    reason = _admission_reason(
        atomic_admissible=bool(atomic["grade_admissible"]),
        verified=int(counts["verified"]),
        refuted=int(counts["refuted"]),
        unsupported=int(counts["unsupported"]),
        target_token_count=len(target_tokens),
        pseudo_label_admitted=bool(pseudo_label_admission["admitted"]),
        pseudo_label_reason=str(pseudo_label_admission["reason"]),
    )
    admitted = reason == "admitted_exact_local_evidence"
    evidence_atom_ids = [row["atom_id"] for row in verified_rows]
    evidence_atom_sha256s = [row["atom_sha256"] for row in verified_rows]
    payload = {
        "schema": ADMISSION_SCHEMA,
        "policy_sha256": _canonical_sha256(_ADMISSION_POLICY),
        "candidate_checked": True,
        "admitted": admitted,
        "reason": reason,
        "source_sha256": atomic["source_sha256"],
        "objective_sha256": atomic["objective_sha256"],
        "evaluation_index": evaluation_index,
        "atomic_decomposition": atomic,
        "deterministic_router": router,
        "evidence_atom_ids": evidence_atom_ids,
        "evidence_atom_sha256s": evidence_atom_sha256s,
        "evidence_text_sha256": _text_sha256(evidence_text),
        "target_tokens_sha256": token_sequence_sha256(target_tokens),
        "target_token_count": len(target_tokens),
        "critic_recalibration": critic_recalibration,
        "pseudo_label_admission": pseudo_label_admission,
    }
    receipt = {**payload, "receipt_sha256": _canonical_sha256(payload)}
    validate_fast_weight_admission(
        receipt,
        expected_source_sha256=_text_sha256(candidate),
        expected_objective_sha256=_text_sha256(objective),
    )
    return receipt, target_tokens if admitted else []


def validate_fast_weight_admission(
    value: Mapping[str, Any],
    *,
    expected_source_sha256: str | None = None,
    expected_objective_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ADMISSION_FIELDS:
        raise ValueError("fast-weight admission fields do not match schema")
    payload = {key: value[key] for key in _ADMISSION_FIELDS - {"receipt_sha256"}}
    if value["receipt_sha256"] != _canonical_sha256(payload):
        raise ValueError("fast-weight admission commitment mismatch")
    if value["schema"] != ADMISSION_SCHEMA or value["policy_sha256"] != _canonical_sha256(
        _ADMISSION_POLICY
    ):
        raise ValueError("fast-weight admission policy mismatch")
    for field in ("source_sha256", "objective_sha256", "evidence_text_sha256", "target_tokens_sha256"):
        if not _is_sha256(value[field]):
            raise ValueError(f"fast-weight admission {field} is invalid")
    if expected_source_sha256 is not None and value["source_sha256"] != expected_source_sha256:
        raise ValueError("fast-weight admission source does not match the probe")
    if (
        expected_objective_sha256 is not None
        and value["objective_sha256"] != expected_objective_sha256
    ):
        raise ValueError("fast-weight admission objective does not match the query")
    if type(value["candidate_checked"]) is not bool or type(value["admitted"]) is not bool:
        raise ValueError("fast-weight admission verdict types are invalid")
    if type(value["evaluation_index"]) is not int or type(value["target_token_count"]) is not int:
        raise ValueError("fast-weight admission count types are invalid")
    if not value["candidate_checked"]:
        if (
            value["admitted"]
            or value["evaluation_index"] != -1
            or value["atomic_decomposition"]
            or value["deterministic_router"]
            or value["evidence_atom_ids"]
            or value["evidence_atom_sha256s"]
            or value["target_token_count"] != 0
            or value["critic_recalibration"]
            or value["pseudo_label_admission"]
            or value["evidence_text_sha256"] != _text_sha256("")
            or value["target_tokens_sha256"] != token_sequence_sha256([])
            or value["reason"]
            not in {
                "verifier_unavailable",
                "verifier_provider_untrusted",
                "candidate_evaluation_unavailable",
            }
        ):
            raise ValueError("unavailable fast-weight admission is contradictory")
        return dict(value)
    if value["evaluation_index"] < 0 or value["target_token_count"] < 0:
        raise ValueError("fast-weight admission count is outside bounds")
    atomic = validate_atomic_decomposition_envelope(value["atomic_decomposition"])
    router = validate_deterministic_router_envelope(
        value["deterministic_router"],
        atomic_receipt=atomic,
    )
    if (
        value["source_sha256"] != atomic["source_sha256"]
        or value["objective_sha256"] != atomic["objective_sha256"]
    ):
        raise ValueError("fast-weight admission evidence source mismatch")
    verified_rows = [row for row in router["routes"] if row["outcome"] == "verified"]
    expected_ids = [row["atom_id"] for row in verified_rows]
    expected_hashes = [row["atom_sha256"] for row in verified_rows]
    if value["evidence_atom_ids"] != expected_ids or value["evidence_atom_sha256s"] != expected_hashes:
        raise ValueError("fast-weight admission exact-evidence inventory mismatch")
    counts = router["counts"]
    critic = validate_critic_recalibration_receipt(
        value["critic_recalibration"]
    )
    pseudo = validate_pseudo_label_admission(
        value["pseudo_label_admission"],
        router_receipt=router,
        atomic_receipt=atomic,
        structural_diversity={
            "certified": value["pseudo_label_admission"].get(
                "structural_diversity_certified"
            ),
            "receipt_sha256": value["pseudo_label_admission"].get(
                "structural_diversity_sha256"
            ),
        },
        critic_recalibration=critic,
    )
    expected_reason = _admission_reason(
        atomic_admissible=bool(atomic["grade_admissible"]),
        verified=int(counts["verified"]),
        refuted=int(counts["refuted"]),
        unsupported=int(counts["unsupported"]),
        target_token_count=value["target_token_count"],
        pseudo_label_admitted=bool(pseudo["admitted"]),
        pseudo_label_reason=str(pseudo["reason"]),
    )
    if value["reason"] != expected_reason or value["admitted"] is not (
        expected_reason == "admitted_exact_local_evidence"
    ):
        raise ValueError("fast-weight admission verdict does not reconstruct")
    return dict(value)


def empty_learning_state(
    *,
    episode_id: str,
    input_tokens_sha256: str,
    selected_branch: int,
    winner_state_sha256: str,
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    validate_fast_weight_admission(admission)
    return {
        "schema": LEARNING_SCHEMA,
        "policy_sha256": _canonical_sha256(_LEARNING_POLICY),
        "episode_id": episode_id,
        "input_tokens_sha256": input_tokens_sha256,
        "selected_branch": selected_branch,
        "winner_state_sha256": winner_state_sha256,
        "admission": dict(admission),
        "lease": {
            "schema": LEASE_SCHEMA,
            "owner_sha256": "",
            "model_sha256": "",
            "acquired": False,
            "released": False,
            "conflicts": 0,
        },
        "attach_identity": {
            "measured": False,
            "pre_probe_sha256": "",
            "post_probe_sha256": "",
            "exact": False,
            "winner_state_before_sha256": winner_state_sha256,
            "winner_state_after_sha256": winner_state_sha256,
        },
        "optimization": {
            "optimizer": "",
            "attempts": 0,
            "accepted_steps": 0,
            "rejected_steps": 0,
            "budget_exhausted": False,
            "loss_trail": [],
            "gradient_norm_trail": [],
            "accepted_step_sizes": [],
            "line_search_backtracks": 0,
        },
        "controls": {
            "decision": "not_run",
            "capability_canaries": {},
            "test_time_training": (
                build_test_time_training_receipt(
                    critic_recalibration=admission[
                        "critic_recalibration"
                    ],
                    pseudo_label_admission=admission[
                        "pseudo_label_admission"
                    ],
                    matched_compute=None,
                )
                if admission["candidate_checked"]
                else {}
            ),
        },
        "causal_probe": {
            "evaluated": False,
            "pre_tokens_sha256": "",
            "post_tokens_sha256": "",
            "pre_text_sha256": "",
            "post_text_sha256": "",
            "pre_score": None,
            "post_score": None,
            "token_sequence_changed": False,
            "strict_improvement": False,
            "winner_state_before_sha256": winner_state_sha256,
            "winner_state_after_sha256": winner_state_sha256,
        },
        "final_answer": {
            "decoded_under_adaptation": False,
            "tokens_sha256": token_sequence_sha256([]),
            "text_sha256": _text_sha256(""),
            "token_count": 0,
        },
        "cleanup": {
            "required": False,
            "detached": False,
            "erase_proven": None,
            "lease_released": False,
            "conflicts": 0,
            "pre_probe_sha256": "",
            "post_probe_sha256": "",
            "erased_layer_ids": [],
        },
        "disposition": "not_admitted_high_confidence_evidence_absent",
    }


def finalize_fast_weight_learning_receipt(state: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(state, Mapping) or set(state) != _LEARNING_FIELDS - {"receipt_sha256"}:
        raise ValueError("fast-weight learning state fields do not match schema")
    payload = dict(state)
    receipt = {**payload, "receipt_sha256": _canonical_sha256(payload)}
    return validate_fast_weight_learning_receipt(receipt)


def validate_fast_weight_learning_receipt(
    value: Mapping[str, Any],
    *,
    expected_episode_id: str | None = None,
    expected_input_tokens_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _LEARNING_FIELDS:
        raise ValueError("fast-weight learning fields do not match schema")
    payload = {key: value[key] for key in _LEARNING_FIELDS - {"receipt_sha256"}}
    if value["receipt_sha256"] != _canonical_sha256(payload):
        raise ValueError("fast-weight learning commitment mismatch")
    if value["schema"] != LEARNING_SCHEMA or value["policy_sha256"] != _canonical_sha256(
        _LEARNING_POLICY
    ):
        raise ValueError("fast-weight learning policy mismatch")
    if expected_episode_id is not None and value["episode_id"] != expected_episode_id:
        raise ValueError("fast-weight learning episode mismatch")
    if (
        expected_input_tokens_sha256 is not None
        and value["input_tokens_sha256"] != expected_input_tokens_sha256
    ):
        raise ValueError("fast-weight learning input mismatch")
    if (
        not isinstance(value["episode_id"], str)
        or not value["episode_id"]
        or not _is_sha256(value["input_tokens_sha256"])
        or type(value["selected_branch"]) is not int
        or value["selected_branch"] < 0
        or not _is_sha256(value["winner_state_sha256"])
        or value["disposition"] not in _DISPOSITIONS
    ):
        raise ValueError("fast-weight learning identity fields are invalid")
    admission = validate_fast_weight_admission(value["admission"])
    lease = value["lease"]
    attach = value["attach_identity"]
    optimization = value["optimization"]
    controls = value["controls"]
    causal = value["causal_probe"]
    final = value["final_answer"]
    cleanup = value["cleanup"]
    if not isinstance(lease, Mapping) or set(lease) != {
        "schema", "owner_sha256", "model_sha256", "acquired", "released", "conflicts"
    } or lease["schema"] != LEASE_SCHEMA:
        raise ValueError("fast-weight model lease receipt is invalid")
    if not isinstance(attach, Mapping) or set(attach) != {
        "measured", "pre_probe_sha256", "post_probe_sha256", "exact",
        "winner_state_before_sha256", "winner_state_after_sha256",
    }:
        raise ValueError("fast-weight attach identity receipt is invalid")
    if not isinstance(optimization, Mapping) or set(optimization) != {
        "optimizer", "attempts", "accepted_steps", "rejected_steps", "budget_exhausted",
        "loss_trail", "gradient_norm_trail", "accepted_step_sizes", "line_search_backtracks",
    }:
        raise ValueError("fast-weight optimization receipt is invalid")
    if not isinstance(controls, Mapping) or set(controls) != {
        "decision",
        "capability_canaries",
        "test_time_training",
    }:
        raise ValueError("fast-weight control receipt is invalid")
    if not isinstance(causal, Mapping) or set(causal) != {
        "evaluated", "pre_tokens_sha256", "post_tokens_sha256", "pre_text_sha256",
        "post_text_sha256", "pre_score", "post_score", "token_sequence_changed",
        "strict_improvement", "winner_state_before_sha256", "winner_state_after_sha256",
    }:
        raise ValueError("fast-weight causal probe receipt is invalid")
    if not isinstance(final, Mapping) or set(final) != {
        "decoded_under_adaptation", "tokens_sha256", "text_sha256", "token_count"
    }:
        raise ValueError("fast-weight final-answer binding is invalid")
    if not isinstance(cleanup, Mapping) or set(cleanup) != {
        "required", "detached", "erase_proven", "lease_released", "conflicts",
        "pre_probe_sha256", "post_probe_sha256", "erased_layer_ids",
    }:
        raise ValueError("fast-weight cleanup receipt is invalid")
    if (
        type(lease["acquired"]) is not bool
        or type(lease["released"]) is not bool
        or type(attach["measured"]) is not bool
        or type(attach["exact"]) is not bool
        or type(optimization["budget_exhausted"]) is not bool
        or type(causal["evaluated"]) is not bool
        or type(causal["token_sequence_changed"]) is not bool
        or type(causal["strict_improvement"]) is not bool
        or type(final["decoded_under_adaptation"]) is not bool
        or type(cleanup["required"]) is not bool
        or type(cleanup["detached"]) is not bool
        or type(cleanup["lease_released"]) is not bool
        or not isinstance(cleanup["erased_layer_ids"], list)
        or any(
            not isinstance(item, str) or not item
            for item in cleanup["erased_layer_ids"]
        )
        or (
            cleanup["erase_proven"] is not None
            and type(cleanup["erase_proven"]) is not bool
        )
    ):
        raise ValueError("fast-weight receipt boolean types are invalid")
    if (
        not isinstance(optimization["optimizer"], str)
        or not isinstance(controls["decision"], str)
        or controls["decision"]
        not in {"not_run", "accepted", "rescaled", "erased", "identity_no_check"}
        or not isinstance(controls["capability_canaries"], Mapping)
    ):
        raise ValueError("fast-weight optimizer or control receipt is invalid")
    test_time_training = controls["test_time_training"]
    if admission["candidate_checked"]:
        validate_test_time_training_receipt(
            test_time_training,
            fast_weight_admission=admission,
        )
    elif test_time_training:
        raise ValueError(
            "unavailable fast-weight admission claimed test-time training"
        )
    integer_fields = (
        optimization["attempts"], optimization["accepted_steps"],
        optimization["rejected_steps"], optimization["line_search_backtracks"],
        lease["conflicts"], cleanup["conflicts"], final["token_count"],
    )
    if any(type(item) is not int or item < 0 for item in integer_fields):
        raise ValueError("fast-weight receipt count is invalid")
    for trail_name in ("loss_trail", "gradient_norm_trail", "accepted_step_sizes"):
        trail = optimization[trail_name]
        if not isinstance(trail, list) or any(
            not _is_finite_number(item)
            for item in trail
        ):
            raise ValueError("fast-weight optimization trail is invalid")
    if optimization["attempts"] != optimization["accepted_steps"] + optimization["rejected_steps"]:
        raise ValueError("fast-weight optimization accounting mismatch")
    if (
        len(optimization["gradient_norm_trail"]) != optimization["attempts"]
        or len(optimization["accepted_step_sizes"]) != optimization["accepted_steps"]
        or any(float(item) <= 0.0 for item in optimization["gradient_norm_trail"])
        or any(float(item) <= 0.0 for item in optimization["accepted_step_sizes"])
        or (
            optimization["attempts"] > 0
            and len(optimization["loss_trail"])
            != optimization["accepted_steps"] + 1
        )
        or any(
            float(later) >= float(earlier)
            for earlier, later in zip(
                optimization["loss_trail"],
                optimization["loss_trail"][1:],
                strict=False,
            )
        )
    ):
        raise ValueError("fast-weight optimization evidence is inconsistent")
    winner_hashes = {
        value["winner_state_sha256"],
        attach["winner_state_before_sha256"],
        attach["winner_state_after_sha256"],
        causal["winner_state_before_sha256"],
        causal["winner_state_after_sha256"],
    }
    if len(winner_hashes) != 1 or not all(_is_sha256(item) for item in winner_hashes):
        raise ValueError("fast-weight learning changed or lost the winner state")
    disposition = value["disposition"]
    not_admitted = disposition == "not_admitted_high_confidence_evidence_absent"
    accepted = disposition == "accepted_causal_improvement"
    accepted_probe_only = (
        disposition == "accepted_probe_not_output_under_incumbent_policy"
    )
    attached = bool(lease["acquired"])
    if causal["evaluated"]:
        if (
            not _is_sha256(causal["pre_tokens_sha256"])
            or not _is_sha256(causal["post_tokens_sha256"])
            or not _is_sha256(causal["pre_text_sha256"])
            or not _is_sha256(causal["post_text_sha256"])
            or not _is_finite_number(causal["pre_score"])
            or (
                causal["post_score"] is not None
                and not _is_finite_number(causal["post_score"])
            )
            or causal["token_sequence_changed"]
            is not (
                causal["pre_tokens_sha256"]
                != causal["post_tokens_sha256"]
            )
        ):
            raise ValueError("fast-weight causal probe evidence is inconsistent")
        reconstructed_improvement = bool(
            causal["post_score"] is not None
            and float(causal["post_score"]) > float(causal["pre_score"]) + 1e-6
            and causal["token_sequence_changed"]
        )
        if causal["strict_improvement"] is not reconstructed_improvement:
            raise ValueError("fast-weight causal improvement does not reconstruct")
    elif (
        causal["pre_tokens_sha256"]
        or causal["post_tokens_sha256"]
        or causal["pre_text_sha256"]
        or causal["post_text_sha256"]
        or causal["pre_score"] is not None
        or causal["post_score"] is not None
        or causal["token_sequence_changed"]
        or causal["strict_improvement"]
    ):
        raise ValueError("unevaluated fast-weight causal probe contains evidence")
    if not_admitted:
        if (
            admission["admitted"]
            or attached
            or lease["owner_sha256"]
            or lease["model_sha256"]
            or lease["released"]
            or lease["conflicts"]
            or attach["measured"]
            or optimization["attempts"]
            or optimization["optimizer"]
            or controls["decision"] != "not_run"
            or controls["capability_canaries"]
            or causal["evaluated"]
            or final["decoded_under_adaptation"]
            or cleanup["required"]
            or cleanup["detached"]
            or cleanup["erase_proven"] is not None
            or cleanup["lease_released"]
            or cleanup["conflicts"]
            or cleanup["pre_probe_sha256"]
            or cleanup["post_probe_sha256"]
            or cleanup["erased_layer_ids"]
        ):
            raise ValueError("non-admitted fast-weight learning performed adaptation")
    else:
        if (
            admission["admitted"] is not True
            or not attached
            or lease["released"] is not True
            or lease["conflicts"] != 0
            or not _is_sha256(lease["owner_sha256"])
            or not _is_sha256(lease["model_sha256"])
            or attach["measured"] is not True
            or attach["exact"] is not True
            or attach["pre_probe_sha256"] != attach["post_probe_sha256"]
            or not _is_sha256(attach["pre_probe_sha256"])
            or cleanup["required"] is not True
            or cleanup["detached"] is not True
            or cleanup["erase_proven"] is not True
            or cleanup["lease_released"] is not True
            or cleanup["conflicts"] != 0
            or not _is_sha256(cleanup["pre_probe_sha256"])
            or cleanup["pre_probe_sha256"]
            != cleanup["post_probe_sha256"]
            or not cleanup["erased_layer_ids"]
        ):
            raise ValueError("admitted fast-weight lifecycle proof is incomplete")
        if optimization["optimizer"] != "rms_normalized_sgd_backtracking_v1":
            raise ValueError("admitted fast-weight optimizer identity is invalid")
    if accepted:
        if (
            optimization["accepted_steps"] <= 0
            or optimization["budget_exhausted"] is not False
            or controls["decision"] not in {"accepted", "rescaled"}
            or causal["evaluated"] is not True
            or causal["token_sequence_changed"] is not True
            or causal["strict_improvement"] is not True
            or not isinstance(causal["pre_score"], (int, float))
            or not isinstance(causal["post_score"], (int, float))
            or float(causal["post_score"]) <= float(causal["pre_score"]) + 1e-6
            or causal["pre_text_sha256"] != admission["source_sha256"]
            or final["decoded_under_adaptation"] is not True
            or test_time_training["decision"]
            != "accepted_bounded_refinement"
        ):
            raise ValueError("accepted fast-weight adaptation lacks causal improvement")
    elif accepted_probe_only:
        if (
            optimization["accepted_steps"] <= 0
            or optimization["budget_exhausted"] is not False
            or controls["decision"] not in {"accepted", "rescaled"}
            or causal["evaluated"] is not True
            or causal["token_sequence_changed"] is not True
            or causal["strict_improvement"] is not True
            or not isinstance(causal["pre_score"], (int, float))
            or not isinstance(causal["post_score"], (int, float))
            or float(causal["post_score"]) <= float(causal["pre_score"]) + 1e-6
            or causal["pre_text_sha256"] != admission["source_sha256"]
            or final["decoded_under_adaptation"] is not False
            or test_time_training["decision"]
            != "accepted_bounded_refinement"
        ):
            raise ValueError(
                "probe-only fast-weight adaptation lacks causal improvement"
            )
    elif not not_admitted and final["decoded_under_adaptation"] is not False:
        raise ValueError("rejected fast-weight adaptation influenced the final answer")
    if disposition == "rejected_no_accepted_step" and (
        optimization["accepted_steps"] != 0
        or causal["evaluated"]
    ):
        raise ValueError("no-step fast-weight rejection is contradictory")
    if disposition == "rejected_capability_regression" and (
        controls["decision"] != "erased"
        or causal["evaluated"]
    ):
        raise ValueError("capability-regression fast-weight rejection is contradictory")
    if disposition == "rejected_verifier_unavailable" and causal["evaluated"]:
        raise ValueError("unavailable-verifier fast-weight rejection is contradictory")
    if disposition == "rejected_no_causal_effect" and (
        causal["evaluated"] is not True
        or causal["token_sequence_changed"] is not False
        or causal["strict_improvement"] is not False
    ):
        raise ValueError("no-effect fast-weight rejection is contradictory")
    if disposition == "rejected_non_improvement" and (
        causal["evaluated"] is not True
        or causal["strict_improvement"] is not False
    ):
        raise ValueError("non-improving fast-weight rejection is contradictory")
    if disposition == "rejected_state_lineage_changed" and (
        causal["evaluated"] is not True
        or causal["strict_improvement"] is not True
    ):
        raise ValueError("state-lineage fast-weight rejection is contradictory")
    if disposition == "rejected_matched_control" and (
        causal["evaluated"] is not True
        or causal["strict_improvement"] is not True
        or test_time_training["decision"] != "rejected_matched_control"
    ):
        raise ValueError(
            "matched-control fast-weight rejection is contradictory"
        )
    if (
        not _is_sha256(final["tokens_sha256"])
        or not _is_sha256(final["text_sha256"])
    ):
        raise ValueError("fast-weight final answer commitment is invalid")
    return dict(value)


__all__ = [
    "ADMISSION_SCHEMA",
    "LEARNING_SCHEMA",
    "MAX_TARGET_TOKENS",
    "build_fast_weight_admission",
    "empty_learning_state",
    "finalize_fast_weight_learning_receipt",
    "token_sequence_sha256",
    "unavailable_admission",
    "validate_fast_weight_admission",
    "validate_fast_weight_learning_receipt",
]
