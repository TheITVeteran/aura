"""SPARK-003 failure/threat model with executable mitigations.

One typed registry binds every named Spark threat class to the concrete
mitigating mechanisms in this package and to the exact tests that prove
each mitigation fires.  The registry is executable, not narrative:
``validate_threat_model`` fails closed when a required threat class is
missing, a mitigation module has moved, or a bound check no longer exists
in the test suite — so the threat model cannot silently rot while the
code drifts.  The bound tests themselves run inside the ordinary offline
gates; this module proves the binding, the suite proves the behavior.

Residual-risk lines are part of the contract: a threat model that claims
zero residual risk is the first thing to distrust.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never

from core.brain.llm.latent_cortex.epistemic_state import canonical_sha256

THREAT_MODEL_SCHEMA = "aura.latent_cortex.threat_model.v1"

REQUIRED_THREAT_IDS = (
    "anchoring",
    "verifier_collusion",
    "fake_branch_diversity",
    "reward_hacking",
    "answer_leakage",
    "right_to_wrong_correction",
    "context_contamination",
    "state_corruption",
    "budget_abuse",
    "stale_tools",
    "adaptation_leakage",
    "unsafe_self_modification",
    # Added after the SPARK-061 sweep measured it (F8, 2026-07-27). The
    # ledger's original enumeration is a floor, not a ceiling: a threat model
    # that cannot grow when a new failure is measured is a document, not a
    # registry.
    "recurrence_inertness",
)


class ThreatModelError(ValueError):
    """Stable fail-closed threat-model error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ThreatModelError(code)


@dataclass(frozen=True, slots=True)
class MitigationCheck:
    """One executable proof that a mitigation fires: a real suite test."""

    test_file: str
    test_name: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.test_file, str)
            or not self.test_file.startswith("tests/test_")
            or not self.test_file.endswith(".py")
            or not isinstance(self.test_name, str)
            or not self.test_name.startswith("test_")
        ):
            _fail("threat_model_check_invalid")

    def to_dict(self) -> dict[str, str]:
        return {"test_file": self.test_file, "test_name": self.test_name}


@dataclass(frozen=True, slots=True)
class ThreatEntry:
    """One named failure mode, its mechanisms, proofs, and honest residue."""

    threat_id: str
    name: str
    failure_mode: str
    mitigations: tuple[str, ...]
    checks: tuple[MitigationCheck, ...]
    residual_risk: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.threat_id, str)
            or not self.threat_id
            or not isinstance(self.name, str)
            or not self.name
            or not isinstance(self.failure_mode, str)
            or len(self.failure_mode) < 40
            or not self.mitigations
            or any(
                not isinstance(path, str) or not path.endswith(".py") for path in self.mitigations
            )
            or not self.checks
            or any(not isinstance(check, MitigationCheck) for check in self.checks)
            or not isinstance(self.residual_risk, str)
            or len(self.residual_risk) < 20
        ):
            _fail("threat_model_entry_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "threat_id": self.threat_id,
            "name": self.name,
            "failure_mode": self.failure_mode,
            "mitigations": list(self.mitigations),
            "checks": [check.to_dict() for check in self.checks],
            "residual_risk": self.residual_risk,
        }


_LC = "core/brain/llm/latent_cortex"

THREATS: tuple[ThreatEntry, ...] = (
    ThreatEntry(
        threat_id="anchoring",
        name="Anchoring on the first drafted answer",
        failure_mode=(
            "A reviewer or later pass conditions on the first candidate's "
            "text or provenance and rationalizes toward it instead of "
            "re-deriving from the problem; rejected reasoning stays in the "
            "context and keeps steering attention."
        ),
        mitigations=(
            f"{_LC}/blind_review.py",
            f"{_LC}/kv_state_tree.py",
        ),
        checks=(
            MitigationCheck(
                "tests/test_rlc_blind_review.py",
                "test_reviewer_sees_only_deranged_origin_free_candidate_text",
            ),
            MitigationCheck(
                "tests/test_kv_state_tree.py",
                "test_rejected_child_cannot_become_a_later_parent",
            ),
            MitigationCheck(
                "tests/test_kv_state_tree.py",
                "test_real_qwen_rejected_work_cannot_change_regenerated_window",
            ),
        ),
        residual_risk=(
            "The ordinary non-RLC response lane still drafts and revises in "
            "one context; anchoring is only removed inside latent episodes."
        ),
    ),
    ThreatEntry(
        threat_id="verifier_collusion",
        name="Verifier collusion with the generator",
        failure_mode=(
            "A critic that shares the generator's weights, imports, or "
            "runtime state inherits its blind spots and approves what the "
            "generator prefers; identity tampering lets a colluding critic "
            "impersonate an independent one."
        ),
        mitigations=(
            f"{_LC}/critic_identity.py",
            f"{_LC}/blind_review.py",
        ),
        checks=(
            MitigationCheck(
                "tests/test_rlc_critic_identity.py",
                "test_symbolic_critic_identity_is_disjoint_from_neural_generator",
            ),
            MitigationCheck(
                "tests/test_rlc_critic_identity.py",
                "test_dependency_audit_rejects_generator_runtime_imports",
            ),
            MitigationCheck(
                "tests/test_rlc_critic_identity.py",
                "test_source_and_generator_identity_tampering_is_rejected",
            ),
        ),
        residual_risk=(
            "Learned process/generative verifiers still share base "
            "pretraining with the generator; SPARK-041..046 own narrowing "
            "that correlation."
        ),
    ),
    ThreatEntry(
        threat_id="fake_branch_diversity",
        name="Fake branch diversity",
        failure_mode=(
            "Paraphrase-level variation masquerades as independent "
            "hypotheses, so N branches carry one error structure and "
            "majority signals count the same mistake N times."
        ),
        mitigations=(
            f"{_LC}/structural_diversity.py",
            f"{_LC}/correlated_support.py",
        ),
        checks=(
            MitigationCheck(
                "tests/test_rlc_structural_diversity.py",
                "test_surface_paraphrase_cannot_create_or_change_structural_support",
            ),
            MitigationCheck(
                "tests/test_rlc_structural_diversity.py",
                "test_distinct_causal_structures_create_independent_support_classes",
            ),
            MitigationCheck(
                "tests/test_rlc_correlated_support.py",
                "test_correlated_paths_reduce_effective_support_and_exchange_weight",
            ),
        ),
        residual_risk=(
            "Structural fingerprints are heuristic; two branches can share "
            "a deep failure while differing structurally."
        ),
    ),
    ThreatEntry(
        threat_id="reward_hacking",
        name="Reward hacking of review and verification",
        failure_mode=(
            "A verifier that always votes yes, tracks labels instead of "
            "content, or is tuned on its own grading passes candidates "
            "without discriminating; training then optimizes the exploit "
            "instead of the task."
        ),
        mitigations=(
            f"{_LC}/blind_review.py",
            f"{_LC}/task_verifiers.py",
        ),
        checks=(
            MitigationCheck(
                "tests/test_rlc_blind_review.py",
                "test_decoy_balanced_review_admits_a_discriminating_stable_verifier",
            ),
            MitigationCheck(
                "tests/test_rlc_blind_review.py",
                "test_decoy_validator_rejects_order_label_and_verdict_tampering",
            ),
            MitigationCheck(
                "tests/test_latent_cortex_update_acceptance.py",
                "test_train_and_calibration_examples_must_be_disjoint",
            ),
        ),
        residual_risk=(
            "Delta-reward RLVR with an explicit error-introduction-rate "
            "penalty (SPARK-060) is not implemented yet; training-side "
            "reward surfaces remain wider than review-side ones."
        ),
    ),
    ThreatEntry(
        threat_id="answer_leakage",
        name="Answer leakage into tasks or training",
        failure_mode=(
            "Task manifests, seeds, or nonces leak the expected answer; "
            "training families overlap evaluation families, so measured "
            "gains are memorization instead of capability."
        ),
        mitigations=(
            f"{_LC}/frontier_tasks.py",
            f"{_LC}/answer_contract.py",
            "core/learning/recurrence_curriculum.py",
        ),
        checks=(
            MitigationCheck(
                "tests/test_latent_cortex_frontier_tasks.py",
                "test_public_task_and_manifest_do_not_leak_seed_nonce_or_answer_payload",
            ),
            MitigationCheck(
                "tests/test_latent_cortex_frontier_tasks.py",
                "test_current_registry_truthfully_declares_full_training_lineage",
            ),
            MitigationCheck(
                "tests/test_answer_channel_curriculum.py",
                "test_answer_channel_split_is_disjoint",
            ),
        ),
        residual_risk=(
            "Pretraining-corpus contamination of the base model is only "
            "bounded by the signed contamination audit, not eliminated."
        ),
    ),
    ThreatEntry(
        threat_id="right_to_wrong_correction",
        name="Right-to-wrong correction",
        failure_mode=(
            "A later pass replaces a correct verified answer because a "
            "persuasive critique or an unverified higher score appeared; "
            "review deletes value instead of adding it."
        ),
        mitigations=(
            f"{_LC}/verified_best.py",
            f"{_LC}/update_gate.py",
        ),
        checks=(
            MitigationCheck(
                "tests/test_latent_cortex_verified_best.py",
                "test_scalar_verifier_can_rank_but_cannot_certify_state",
            ),
            MitigationCheck(
                "tests/test_latent_cortex_verified_best.py",
                "test_deterministic_exact_observation_and_final_reversion_are_explicit",
            ),
            MitigationCheck(
                "tests/test_latent_cortex_verified_best.py",
                "test_confidence_dominance_promotes_and_overlap_preserves_branch_locally",
            ),
        ),
        residual_risk=(
            "Confidence-bound replacement depends on verifier calibration; "
            "a miscalibrated verifier can still under-protect the incumbent."
        ),
    ),
    ThreatEntry(
        threat_id="context_contamination",
        name="Context contamination across branches and episodes",
        failure_mode=(
            "One branch or a previous episode leaks tokens, hidden state, "
            "cache slices, or RNG position into another, so 'independent' "
            "solutions share causes and evidence enters without a typed "
            "ancestor."
        ),
        mitigations=(
            f"{_LC}/branches.py",
            f"{_LC}/blind_review.py",
            f"{_LC}/state_causality.py",
        ),
        checks=(
            MitigationCheck(
                "tests/test_rlc_blind_review.py",
                "test_blind_review_requires_fresh_context_isolation",
            ),
            MitigationCheck(
                "tests/test_latent_cortex_wiring.py",
                "test_service_reconstructs_and_rejects_branch_isolation_tampering",
            ),
            MitigationCheck(
                "tests/test_spark_state_causality.py",
                "test_unknown_evidence_refused",
            ),
        ),
        residual_risk=(
            "Cross-episode contamination through consolidated memory is "
            "governed by the memory bridge, not physically impossible."
        ),
    ),
    ThreatEntry(
        threat_id="state_corruption",
        name="Epistemic state corruption",
        failure_mode=(
            "In-place mutation, non-canonical serialization, journal "
            "truncation, or envelope tampering rewrites what the system "
            "believed without leaving a trace, so later passes reason from "
            "a forged past."
        ),
        mitigations=(
            f"{_LC}/epistemic_state.py",
            f"{_LC}/epistemic_journal.py",
        ),
        checks=(
            MitigationCheck(
                "tests/test_rlc_epistemic_state.py",
                "test_genesis_is_canonical_deeply_immutable_and_content_addressed",
            ),
            MitigationCheck(
                "tests/test_rlc_epistemic_journal.py",
                "test_journal_initializes_recovers_and_continues_exact_state",
            ),
            MitigationCheck(
                "tests/test_rlc_epistemic_journal.py",
                "test_gateway_write_if_absent_never_replaces_the_winner",
            ),
        ),
        residual_risk=(
            "Digest chains prove tampering happened; they cannot recover "
            "state that was corrupted before its first durable append."
        ),
    ),
    ThreatEntry(
        threat_id="budget_abuse",
        name="Compute budget abuse",
        failure_mode=(
            "Unmetered side computation (probes, retrieval, verifier "
            "passes) grants one arm hidden compute, so measured wins are "
            "resource asymmetries, and a runaway episode starves the "
            "resident."
        ),
        mitigations=(
            f"{_LC}/resource_accounting.py",
            f"{_LC}/virtual_quanta.py",
            f"{_LC}/types.py",
        ),
        checks=(
            MitigationCheck(
                "tests/test_latent_cortex_resource_accounting.py",
                "test_resource_receipt_reconstructs_totals_and_rejects_tampering",
            ),
            MitigationCheck(
                "tests/test_latent_cortex_virtual_quanta.py",
                "test_guided_quantum_requires_measured_win_applies_once_and_erases",
            ),
            MitigationCheck(
                "tests/test_latent_cortex_virtual_quanta.py",
                "test_resource_mismatch_refuses_even_when_guided_score_wins",
            ),
        ),
        residual_risk=(
            "Wall-clock variance under host load is observable but not "
            "attributable to a single episode's accounting."
        ),
    ),
    ThreatEntry(
        threat_id="stale_tools",
        name="Stale tool results and expired evidence",
        failure_mode=(
            "A cached tool observation or expired calibration keeps "
            "supporting claims after the world moved, so answers cite "
            "evidence that was true when observed and false when used."
        ),
        mitigations=(
            f"{_LC}/epistemic_state.py",
            f"{_LC}/epistemic_calibration.py",
        ),
        checks=(
            MitigationCheck(
                "tests/test_rlc_epistemic_state.py",
                "test_answer_requires_fresh_transitive_evidence_closure",
            ),
            MitigationCheck(
                "tests/test_rlc_epistemic_state.py",
                "test_answer_rejects_expired_profile_and_inflated_confidence",
            ),
        ),
        residual_risk=(
            "Freshness windows are declared by evidence producers; a "
            "producer that over-declares validity defeats the expiry gate."
        ),
    ),
    ThreatEntry(
        threat_id="adaptation_leakage",
        name="Query-scoped adaptation leakage",
        failure_mode=(
            "Fast weights or a per-query LoRA delta survive the episode "
            "that created them, silently changing the resident model for "
            "every later user and invalidating identity proofs."
        ),
        mitigations=(
            f"{_LC}/fast_weights.py",
            f"{_LC}/governance.py",
        ),
        checks=(
            MitigationCheck(
                "tests/test_latent_cortex_opt_fastweights.py",
                "test_fast_weights_compose_with_real_lora_and_restore_exact_module",
            ),
            MitigationCheck(
                "tests/test_latent_cortex_opt_fastweights.py",
                "test_fast_weights_identity_at_attach",
            ),
            MitigationCheck(
                "tests/test_latent_cortex_wiring.py",
                "test_client_latent_reason_integrity_failure_recycles_resident",
            ),
        ),
        residual_risk=(
            "Erasure is proven per-episode; a worker crash between attach "
            "and erase relies on the recycle path, not on proof."
        ),
    ),
    ThreatEntry(
        threat_id="unsafe_self_modification",
        name="Unsafe self-modification",
        failure_mode=(
            "A learned head, adapter, or update path changes live behavior "
            "without admission evidence, or an episode mutates resident "
            "parameters and the change is discovered only by drift."
        ),
        mitigations=(
            f"{_LC}/governance.py",
            f"{_LC}/update_gate.py",
            f"{_LC}/adapter_identity.py",
        ),
        checks=(
            MitigationCheck(
                "tests/test_latent_cortex_update_acceptance.py",
                "test_head_is_actually_fitted_calibrated_and_round_trips",
            ),
            MitigationCheck(
                "tests/test_latent_cortex_proof_integrity.py",
                "test_recycle_requires_an_explicit_pass",
            ),
            MitigationCheck(
                "tests/test_latent_cortex_proof_integrity.py",
                "test_absent_proof_is_reported",
            ),
        ),
        residual_risk=(
            "Admission gates cover the latent-cortex package; organism-wide "
            "self-modification authority is owned by the Will/governance "
            "stack outside this module."
        ),
    ),
    ThreatEntry(
        threat_id="recurrence_inertness",
        name="Training the recurrent operator inert",
        failure_mode=(
            "A monotone-improvement objective is unwinnable on families "
            "where depth is destructive, and the identity operator "
            "satisfies it perfectly: every step loss is equal, the hinge "
            "is silent, and the loss curve still descends because the "
            "answer head is learning. Recurrence is dead while the "
            "campaign reports success. Measured on the untrained 1.5B, "
            "low motion is a local basin behind a ~0.19-nat barrier, and "
            "an auxiliary term that is declared, weighted and inert hides "
            "the same way: the composite descends regardless."
        ),
        mitigations=(
            "core/learning/progressive_recurrent_objective.py",
            "core/learning/auxiliary_objective_curriculum.py",
        ),
        checks=(
            MitigationCheck(
                "tests/test_progressive_recurrent_objective.py",
                "test_perfect_improvement_from_a_dead_operator_is_refused",
            ),
            MitigationCheck(
                "tests/test_progressive_recurrent_objective.py",
                "test_measured_sweep_shows_collapse_is_a_local_basin_not_the_optimum",
            ),
            MitigationCheck(
                "tests/test_progressive_recurrent_objective.py",
                "test_steps_that_cost_nothing_to_remove_are_causally_idle",
            ),
            MitigationCheck(
                "tests/test_auxiliary_objective_curriculum.py",
                "test_a_declared_term_with_no_gradient_path_is_inert_and_refuses",
            ),
            MitigationCheck(
                "tests/test_auxiliary_objective_curriculum.py",
                "test_a_head_term_that_reached_the_base_weights_is_misdeclared",
            ),
        ),
        residual_risk=(
            "The detectors are not yet mandatory on the training path: "
            "tools/train_grpo.py does not consult them, so a campaign can "
            "still be launched without a progressive report. The pricing "
            "constants are also operating-point specific and have only "
            "been solved on the 1.5B, not the resident 32B."
        ),
    ),
)


def validate_threat_model(repo_root: Path | None = None) -> dict[str, Any]:
    """Fail closed unless every threat, mitigation, and check resolves."""

    root = (
        Path(__file__).resolve().parents[4]
        if repo_root is None
        else Path(repo_root).expanduser().resolve()
    )
    seen: set[str] = set()
    check_count = 0
    for entry in THREATS:
        if entry.threat_id in seen:
            _fail("threat_model_duplicate_threat")
        seen.add(entry.threat_id)
        for module in entry.mitigations:
            if not (root / module).is_file():
                _fail("threat_model_mitigation_missing")
        for check in entry.checks:
            test_path = root / check.test_file
            if not test_path.is_file():
                _fail("threat_model_check_file_missing")
            if f"def {check.test_name}(" not in test_path.read_text(encoding="utf-8"):
                _fail("threat_model_check_test_missing")
            check_count += 1
    if seen != set(REQUIRED_THREAT_IDS):
        _fail("threat_model_coverage_incomplete")
    body = {
        "schema": THREAT_MODEL_SCHEMA,
        "threat_count": len(THREATS),
        "check_count": check_count,
        "threats": [entry.to_dict() for entry in THREATS],
    }
    return {**body, "registry_sha256": canonical_sha256(body)}


__all__ = [
    "REQUIRED_THREAT_IDS",
    "THREATS",
    "THREAT_MODEL_SCHEMA",
    "MitigationCheck",
    "ThreatEntry",
    "ThreatModelError",
    "validate_threat_model",
]
