"""Runtime evidence summaries for RLC reconciliation treatments."""

from __future__ import annotations

from typing import Any


def full_stack_evidence(receipt: dict[str, Any]) -> dict[str, Any]:
    """Compact, public proof that the named treatment actually ran its stack.

    A configuration test proves only that switches were set. This summary binds
    the runtime episode to the mechanisms that executed, including conservative
    non-admission (which is a valid outcome) versus unavailable infrastructure
    (which is not a measurement of that mechanism).
    """

    verifier_preflight = receipt.get("verifier_preflight") or {}
    branch_exchange = receipt.get("branch_exchange") or {}
    fast_weight_learning = receipt.get("fast_weight_learning") or {}
    local_repair = receipt.get("local_repair") or {}
    answer_replacement = receipt.get("answer_replacement") or {}
    incumbent = receipt.get("incumbent_artifact") or {}
    incumbent_binding = incumbent.get("binding") or {}
    incumbent_output = incumbent.get("output") or {}
    baseline_decode = answer_replacement.get("baseline_decode") or {}
    issues: list[str] = []

    try:
        from core.brain.llm.latent_cortex.incumbent_artifact import (
            validate_incumbent_receipt,
        )

        validate_incumbent_receipt(
            incumbent,
            checkpoint_fingerprint=str(receipt.get("checkpoint_fingerprint") or ""),
            checkpoint_fingerprint_method=str(receipt.get("checkpoint_fingerprint_method") or ""),
        )
        incumbent_valid = True
    except (KeyError, TypeError, ValueError):
        incumbent_valid = False

    try:
        from core.brain.llm.latent_cortex.causal_receipt import (
            validate_causal_receipt,
        )

        validate_causal_receipt(
            receipt.get("causal_receipt"),
            worker_receipt=receipt,
            require_complete=True,
        )
        causal_identity_valid = True
    except (KeyError, TypeError, ValueError):
        causal_identity_valid = False

    def require(condition: bool, issue: str) -> None:
        if not condition:
            issues.append(issue)

    require(int(receipt.get("n_slots") or 0) > 0, "workspace_not_measured")
    require(int(receipt.get("steps_taken") or 0) >= 2, "recurrence_not_executed")
    require(int(receipt.get("n_branches") or 0) >= 2, "virtual_width_not_executed")
    isolation = receipt.get("branch_isolation") or {}
    require(isolation.get("certified") is True, "branch_isolation_not_certified")
    candidates = isolation.get("candidates") or []
    require(
        len(candidates) == int(receipt.get("n_branches") or 0)
        and all(isinstance(row.get("role"), str) and row["role"] for row in candidates),
        "branch_roles_not_measured",
    )
    require(
        branch_exchange.get("schema") == "aura.rlc.branch_exchange_trace.v1",
        "branch_exchange_not_measured",
    )
    require(
        verifier_preflight.get("verifier_admitted") is True,
        "task_verifier_not_admitted",
    )
    require(
        str(receipt.get("latent_opt_mode") or "") in {"gradient", "control"},
        "latent_optimization_not_executed",
    )
    require(int(receipt.get("latent_opt_attempts") or 0) > 0, "latent_optimization_no_attempt")
    require(bool(fast_weight_learning), "fast_weight_policy_not_measured")
    require(
        fast_weight_learning.get("disposition") != "rejected_verifier_unavailable",
        "fast_weight_verifier_unavailable",
    )
    require(bool(receipt.get("value_of_computation")), "adaptive_controller_not_measured")
    require(bool(receipt.get("cognitive_action_trace")), "cognitive_actions_not_measured")
    require(bool(receipt.get("diagnostic_action_selection")), "diagnostics_not_measured")
    require(bool(local_repair), "local_repair_policy_not_measured")
    require(bool(answer_replacement), "incumbent_promotion_gate_not_measured")
    require(causal_identity_valid, "causal_runtime_identity_not_measured")
    require(
        (receipt.get("runtime_identity") or {}).get("identity_bound") is True,
        "source_runtime_identity_not_bound",
    )
    require(incumbent_valid, "canonical_incumbent_not_measured")
    require(
        incumbent_binding.get("checkpoint_fingerprint") == receipt.get("checkpoint_fingerprint")
        and receipt.get("checkpoint_fingerprint_method") == "sha256"
        and int(receipt.get("checkpoint_file_count") or 0) > 0,
        "cryptographic_checkpoint_binding_not_measured",
    )
    require(
        incumbent_output.get("text_sha256") == baseline_decode.get("text_sha256")
        and incumbent_output.get("tokens_sha256") == baseline_decode.get("tokens_sha256")
        and incumbent_output.get("token_count") == baseline_decode.get("token_count"),
        "promotion_baseline_differs_from_canonical_incumbent",
    )
    require(receipt.get("params_unchanged") is True, "base_parameters_not_proven_unchanged")
    if receipt.get("fast_weights_applied") is True:
        require(receipt.get("fast_weights_erased") is True, "fast_weights_not_proven_erased")
        require(bool(receipt.get("fast_weight_cleanup")), "fast_weight_cleanup_not_measured")

    return {
        "valid": not issues,
        "issues": sorted(set(issues)),
        "n_slots": int(receipt.get("n_slots") or 0),
        "cognitive_slot_count": len(receipt.get("cognitive_slots") or []),
        "steps_taken": int(receipt.get("steps_taken") or 0),
        "n_branches": int(receipt.get("n_branches") or 0),
        "exchange_count": len(branch_exchange.get("exchanges") or []),
        "verifier_admitted": verifier_preflight.get("verifier_admitted") is True,
        "latent_opt_mode": str(receipt.get("latent_opt_mode") or ""),
        "latent_opt_attempts": int(receipt.get("latent_opt_attempts") or 0),
        "latent_opt_accepted_steps": int(receipt.get("latent_opt_steps") or 0),
        "fast_weight_disposition": str(fast_weight_learning.get("disposition") or ""),
        "fast_weights_applied": receipt.get("fast_weights_applied") is True,
        "fast_weights_erased": receipt.get("fast_weights_erased"),
        "controller_decisions": len(receipt.get("cognitive_action_trace") or []),
        "repair_requests": len(local_repair.get("requests") or []),
        "replacement_decision": str(answer_replacement.get("decision") or ""),
        "incumbent_receipt_sha256": str(incumbent.get("receipt_sha256") or ""),
        "checkpoint_fingerprint": str(receipt.get("checkpoint_fingerprint") or ""),
        "params_unchanged": receipt.get("params_unchanged") is True,
    }


__all__ = ["full_stack_evidence"]
