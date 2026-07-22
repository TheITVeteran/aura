"""Shared cryptographic fixture for Latent Cortex frontier-verifier tests."""

from __future__ import annotations

import base64

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from core.brain.frontier_evidence_v5 import canonical_json_bytes
from core.brain.llm.latent_cortex.frontier_certification import (
    INDEPENDENT_ATTESTATION_SCHEMA,
    SCHEMA,
    TASK_COMMITMENT_ATTESTATION_SCHEMA,
    canonical_sha256,
    evidence_payload_sha256,
    verify_frontier_gain_bundle,
)
from core.brain.llm.latent_cortex.frontier_artifacts import (
    ARTIFACT_VERIFICATION_SCHEMA,
)
from core.brain.llm.latent_cortex.frontier_verifier import (
    TRUST_CONFIG_SCHEMA,
    verifier_implementation_sha256,
)

_VERIFIER_KEY = Ed25519PrivateKey.generate()
_TASK_ISSUER_KEY = Ed25519PrivateKey.generate()
_VERIFIER_ID = "independent-proof-kernel"
_TASK_ISSUER_ID = "independent-task-issuer"
_VERIFIER_IMPLEMENTATION = "d" * 64
_VERIFIER_RELEASE = "0" * 64
_TASK_ISSUER_IMPLEMENTATION = "2" * 64
_TASK_ISSUER_RELEASE = "3" * 64
_VERIFIER_PUBLIC_KEY = base64.b64encode(
    _VERIFIER_KEY.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
).decode("ascii")
_TRUSTED_VERIFIERS = {
    _VERIFIER_ID: {
        "public_key_b64": _VERIFIER_PUBLIC_KEY,
        "implementation_sha256": _VERIFIER_IMPLEMENTATION,
        "release_sha256": _VERIFIER_RELEASE,
    }
}
_TASK_ISSUER_PUBLIC_KEY = base64.b64encode(
    _TASK_ISSUER_KEY.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
).decode("ascii")
_TRUSTED_TASK_ISSUERS = {
    _TASK_ISSUER_ID: {
        "public_key_b64": _TASK_ISSUER_PUBLIC_KEY,
        "implementation_sha256": _TASK_ISSUER_IMPLEMENTATION,
        "release_sha256": _TASK_ISSUER_RELEASE,
    }
}


def _trust_config() -> dict:
    return {
        "schema": TRUST_CONFIG_SCHEMA,
        "verification_kernel_sha256": verifier_implementation_sha256(),
        "task_issuers": _TRUSTED_TASK_ISSUERS,
        "verifiers": _TRUSTED_VERIFIERS,
    }


def _refresh_task_commitment(bundle: dict) -> None:
    manifest = [
        {
            "trial_id": trial["trial_id"],
            "task_id": trial["task_id"],
            "domain": trial["domain"],
            "task_payload_sha256": trial["task_payload_sha256"],
            "task_generated_at": trial["task_generated_at"],
        }
        for trial in bundle["trials"]
    ]
    manifest.sort(key=lambda row: row["trial_id"])
    manifest_sha256 = canonical_sha256(manifest)
    task_commitment_sha256 = canonical_sha256(
        {
            "architecture_freeze_sha256": bundle["preregistration"]["architecture_freeze_sha256"],
            "preregistration_sha256": bundle["preregistration_sha256"],
            "task_count": len(manifest),
            "task_manifest_sha256": manifest_sha256,
        }
    )
    payload = {
        "architecture_freeze_sha256": bundle["preregistration"]["architecture_freeze_sha256"],
        "preregistration_sha256": bundle["preregistration_sha256"],
        "task_commitment_sha256": task_commitment_sha256,
        "task_manifest_sha256": manifest_sha256,
        "task_count": len(manifest),
        "issuer_implementation_sha256": _TASK_ISSUER_IMPLEMENTATION,
        "issuer_release_sha256": _TASK_ISSUER_RELEASE,
        "committed_at": 1200.0,
    }
    bundle["task_commitment_sha256"] = task_commitment_sha256
    bundle["task_commitment"] = {
        "schema": TASK_COMMITMENT_ATTESTATION_SCHEMA,
        "signed_payload": payload,
        "signer": {
            "algorithm": "Ed25519",
            "signer_id": _TASK_ISSUER_ID,
            "public_key_b64": _TASK_ISSUER_PUBLIC_KEY,
            "signature_b64": base64.b64encode(
                _TASK_ISSUER_KEY.sign(canonical_json_bytes(payload))
            ).decode("ascii"),
        },
    }


def _refresh_attestation(bundle: dict, *, verified_at: float = 2000.0) -> None:
    payload = {
        "accepted": True,
        "claim_tier": "PROVEN",
        "producer_id": bundle["producer_id"],
        "evidence_payload_sha256": evidence_payload_sha256(bundle),
        "preregistration_sha256": bundle["preregistration_sha256"],
        "implementation_sha256": _VERIFIER_IMPLEMENTATION,
        "verifier_release_sha256": _VERIFIER_RELEASE,
        "raw_artifact_manifest_sha256": bundle["raw_artifact_manifest_sha256"],
        "task_commitment_sha256": bundle["task_commitment_sha256"],
        "verified_at": verified_at,
    }
    bundle["independent_verifier"] = {
        "schema": INDEPENDENT_ATTESTATION_SCHEMA,
        "signed_payload": payload,
        "signer": {
            "algorithm": "Ed25519",
            "signer_id": _VERIFIER_ID,
            "public_key_b64": _VERIFIER_PUBLIC_KEY,
            "signature_b64": base64.b64encode(
                _VERIFIER_KEY.sign(canonical_json_bytes(payload))
            ).decode("ascii"),
        },
    }


def _raw_artifact_receipt(bundle: dict) -> dict:
    """A stand-in for the receipt ``verify_raw_artifact_package`` returns.

    Certification requires a real artifact receipt bound to the bundle; the
    on-disk artifact package itself is exercised by the frontier_artifacts
    tests, so this fixture supplies an equivalently-bound receipt.
    """
    receipt = {
        "schema": ARTIFACT_VERIFICATION_SCHEMA,
        "accepted": True,
        "manifest_sha256": bundle["raw_artifact_manifest_sha256"],
        "artifact_store_count": 3,
        "artifact_bytes": 4096,
        "trial_count": len(bundle["trials"]),
        "lineage_sha256": canonical_sha256(["lineage", bundle["producer_id"]]),
        "store_receipts": {},
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _certify(bundle: dict, *, raw_artifact_receipt: dict | None = None) -> dict:
    return verify_frontier_gain_bundle(
        bundle,
        trusted_verifiers=_TRUSTED_VERIFIERS,
        trusted_task_issuers=_TRUSTED_TASK_ISSUERS,
        raw_artifact_receipt=(
            _raw_artifact_receipt(bundle)
            if raw_artifact_receipt is None
            else raw_artifact_receipt
        ),
    )


# A control that is genuinely vanilla declares every enhancement OFF and
# pins its decoding, so "unenhanced" is a checked claim rather than an
# unmentioned one.
_VANILLA_CONTROL_MANIFEST = {
    "fast_weights_applied": False,
    "latent_opt_applied": False,
    "recurrence_adapter_applied": False,
    "retrieval_applied": False,
    "nonparametric_memory_applied": False,
    "contrastive_decoding_applied": False,
    "speculative_decoding_applied": False,
    "expert_adapter_applied": False,
    "affective_steering_active": False,
    "prompt_cache_reused": False,
    "decode_temperature": 0.0,
    "decode_top_p": 1.0,
    "decode_repetition_penalty_applied": 1.0,
}

_CONTROL_DECODE_SPEC = {
    "decode_temperature": 0.0,
    "decode_top_p": 1.0,
    "decode_repetition_penalty_applied": 1.0,
}


def _bundle(
    *,
    include_attestation: bool = True,
    comparison_kind: str = "resident_32b_vs_vanilla_same_checkpoint",
) -> dict:
    checkpoint = "8" * 64
    app_build = "4" * 64
    external = comparison_kind == "resident_32b_vs_external_frontier"
    prereg = {
        "protocol_id": "rlc-frontier-v1",
        "comparison_kind": comparison_kind,
        # The decoding the control is preregistered to run. A control run
        # hotter or wider than the treatment is not a control of the same
        # thing, so this is declared up front and checked per trial.
        "control_decode_spec": dict(_CONTROL_DECODE_SPEC),
        "architecture_freeze_sha256": "a" * 64,
        "tool_policy_sha256": "c" * 64,
        "compute_estimator_sha256": "e" * 64,
        "treatment_checkpoint_fingerprint": checkpoint,
        "frozen_at": 1000.0,
        "domains": ["math", "coding", "science"],
        "min_trials_per_domain": 40,
        "alpha": 0.05,
        "minimum_effect": 0.05,
        "compute_tolerance": 0.05,
        "compute_metric": "estimated_flops",
        # Scorer, decoding, and the absolute capability floor are frozen with
        # everything else so they cannot be chosen after outputs are known.
        "scorer_implementation_sha256": "a" * 64,
        "decode_policy_sha256": "d" * 64,
        "min_treatment_success_rate": 0.6,
    }
    if external:
        prereg["control_model_id"] = "frontier-model-x"
        prereg["control_model_build_fingerprint"] = "build-2026-06"
        prereg["control_provider"] = "frontier-provider"
    else:
        prereg["control_checkpoint_fingerprint"] = checkpoint
    trials = []
    index = 0
    for domain in prereg["domains"]:
        for cell in range(40):
            trial_id = f"{domain}-{cell}"
            trials.append(
                {
                    "trial_id": trial_id,
                    "task_id": f"heldout-{trial_id}",
                    "domain": domain,
                    "held_out": True,
                    "contamination_scan_passed": True,
                    "task_generated_at": 1001.0 + index,
                    "evaluation_started_at": 1201.0 + index,
                    "verifier_blinded": True,
                    "verifier_receipt_sha256": canonical_sha256(["verifier", trial_id]),
                    "task_payload_sha256": canonical_sha256(["task", trial_id]),
                    "treatment_output_sha256": canonical_sha256(["treatment", trial_id]),
                    "control_output_sha256": canonical_sha256(["control", trial_id]),
                    "scorer_config_sha256": "f" * 64,
                    "scorer_implementation_sha256": "a" * 64,
                    "treatment_information_sha256": "1" * 64,
                    "control_information_sha256": "1" * 64,
                    "treatment_tool_policy_sha256": "c" * 64,
                    "control_tool_policy_sha256": "c" * 64,
                    "treatment_decode_policy_sha256": "d" * 64,
                    "control_decode_policy_sha256": "d" * 64,
                    "run_order": "treatment_first" if index % 2 == 0 else "control_first",
                    "treatment_success": True,
                    "control_success": cell % 4 == 0,
                    "treatment_compute": {
                        "estimated_flops": 1_000_000.0,
                        "layer_apps": 1000,
                        "estimator_sha256": "e" * 64,
                    },
                    "control_compute": {
                        "estimated_flops": 1_000_000.0,
                        "layer_apps": 1000,
                        "estimator_sha256": "e" * 64,
                    },
                    "treatment_receipt": {
                        "episode_id": f"episode-{trial_id}",
                        "checkpoint_fingerprint": checkpoint,
                        "checkpoint_fingerprint_method": "sha256",
                        "checkpoint_file_count": 8,
                        "worker_boot_id": "boot-live-1",
                        "installed_app_build_sha256": app_build,
                        "schedule_hash": "3" * 64,
                        "params_unchanged": True,
                        "latent_opt_applied": True,
                        "latent_opt_mode": "gradient",
                        "latent_opt_attempts": 2,
                        "latent_opt_steps": 2,
                        "latent_opt_rejected": 0,
                        "latent_opt_budget_exhausted": False,
                        "fast_weights_applied": True,
                        "fast_weights_erased": True,
                        "fast_weights_layers": 4,
                        "fast_weight_optimization_attempts": 2,
                        "fast_weight_optimized_steps": 2,
                        "fast_weight_rejected_steps": 0,
                        "fast_weight_budget_exhausted": False,
                        "n_slots": 8,
                        "n_branches": 2,
                        "steps_taken": 4,
                        "honest_flags": [],
                    },
                    "control_receipt": (
                        {
                            "request_id": f"control-{trial_id}",
                            "model_id": "frontier-model-x",
                            "model_build_fingerprint": "build-2026-06",
                            "provider": "frontier-provider",
                            "provider_receipt_sha256": canonical_sha256(
                                ["provider", trial_id]
                            ),
                        }
                        if external
                        else {
                            "request_id": f"control-{trial_id}",
                            "mode": "vanilla",
                            "latent_cortex_enabled": False,
                            "params_unchanged": True,
                            # CP126 6e1ef7be: a published claim needs the
                            # digest evidence, not the boolean above.
                            "weight_integrity": {
                                "algorithm": "sha256",
                                "version": 1,
                                "params_before": canonical_sha256(
                                    ["control-params", trial_id]
                                ),
                                "params_after": canonical_sha256(
                                    ["control-params", trial_id]
                                ),
                            },
                            "checkpoint_fingerprint": checkpoint,
                            "checkpoint_fingerprint_method": "sha256",
                            "checkpoint_file_count": 8,
                            "worker_boot_id": "boot-live-1",
                            "installed_app_build_sha256": app_build,
                            # CP126 869a0ce4: a control is vanilla only when
                            # every enhancement is positively declared OFF.
                            # An absent field is undeclared, not disabled.
                            **_VANILLA_CONTROL_MANIFEST,
                        }
                    ),
                }
            )
            index += 1
    bundle = {
        "schema": SCHEMA,
        "producer_id": "aura-lab",
        "preregistration": prereg,
        "preregistration_sha256": canonical_sha256(prereg),
        "resident_model": {
            "parameter_count": 32_000_000_000,
            # CP126 8923b135: a count inside the 30-40B window does not say
            # WHICH 32B-class model this is; quantization and architecture
            # change what the weights compute.
            "architecture": "qwen2",
            "quantization_bits": 4,
            "quantization_group_size": 64,
            # The class claim must be derived from the hashed checkpoint
            # manifest, not asserted as a bare integer.
            "parameter_count_source": "checkpoint_manifest",
            "checkpoint_manifest_sha256": checkpoint,
            "checkpoint_fingerprint": checkpoint,
            "checkpoint_fingerprint_method": "sha256",
            "checkpoint_file_count": 8,
            "worker_boot_id": "boot-live-1",
            "loaded_in_installed_app": True,
            "installed_app_bundle_id": "com.aura.desktop",
            "installed_app_build_sha256": app_build,
            "worker_binary_sha256": "5" * 64,
            "build_provenance_sha256": "6" * 64,
            "source_commit": "7" * 40,
            "latent_cortex_architecture_sha256": "a" * 64,
        },
        "raw_artifact_manifest_sha256": "9" * 64,
        "trials": trials,
    }
    _refresh_task_commitment(bundle)
    if include_attestation:
        _refresh_attestation(bundle)
    return bundle


__all__ = [
    "_TASK_ISSUER_ID",
    "_TASK_ISSUER_PUBLIC_KEY",
    "_TRUSTED_TASK_ISSUERS",
    "_TRUSTED_VERIFIERS",
    "_VERIFIER_ID",
    "_VERIFIER_KEY",
    "_VERIFIER_PUBLIC_KEY",
    "_bundle",
    "_certify",
    "_raw_artifact_receipt",
    "_refresh_attestation",
    "_refresh_task_commitment",
    "_trust_config",
]
