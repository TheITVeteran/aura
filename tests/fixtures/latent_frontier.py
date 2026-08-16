"""Shared cryptographic fixture for Latent Cortex frontier-verifier tests."""

from __future__ import annotations

import base64
import hashlib

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from core.brain.frontier_evidence_v5 import canonical_json_bytes
from core.brain.llm.latent_cortex.frontier_artifacts import (
    ARTIFACT_VERIFICATION_SCHEMA,
)
from core.brain.llm.latent_cortex.frontier_certification import (
    INDEPENDENT_ATTESTATION_SCHEMA,
    PRODUCER_ATTESTATION_SCHEMA,
    SCHEMA,
    TASK_COMMITMENT_ATTESTATION_SCHEMA,
    _expected_arm_identifier,
    _merkle_leaf_hash,
    _task_manifest_sha256,
    canonical_sha256,
    evidence_payload_sha256,
    verify_frontier_gain_bundle,
)
from core.brain.llm.latent_cortex.frontier_verifier import (
    TRUST_CONFIG_SCHEMA,
    verifier_implementation_sha256,
)
from core.brain.llm.latent_cortex.resource_accounting import (
    ModelComputeProfile,
    ResourceLedger,
    build_information_receipt,
)
from tests.fixtures.rlc_runtime_integrity import (
    accepted_fast_weight_learning,
    attach_bound_runtime_integrity,
    complete_worker_identity,
)


def _public_key(key: Ed25519PrivateKey) -> str:
    return base64.b64encode(
        key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode("ascii")


_VERIFIER_KEY = Ed25519PrivateKey.generate()
_SECOND_VERIFIER_KEY = Ed25519PrivateKey.generate()
_TASK_ISSUER_KEY = Ed25519PrivateKey.generate()
_PRODUCER_KEY = Ed25519PrivateKey.generate()
_VERIFIER_ID = "independent-proof-kernel"
_SECOND_VERIFIER_ID = "second-independent-proof-kernel"
_TASK_ISSUER_ID = "independent-task-issuer"
_PRODUCER_ID = "aura-lab"
_VERIFIER_IMPLEMENTATION = "d" * 64
_VERIFIER_RELEASE = "0" * 64
_SECOND_VERIFIER_IMPLEMENTATION = "8" * 64
_SECOND_VERIFIER_RELEASE = "b" * 64
_TASK_ISSUER_IMPLEMENTATION = "2" * 64
_TASK_ISSUER_RELEASE = "3" * 64
_PRODUCER_IMPLEMENTATION = "1" * 64
_PRODUCER_RELEASE = "4" * 64
_VERIFIER_PUBLIC_KEY = _public_key(_VERIFIER_KEY)
_SECOND_VERIFIER_PUBLIC_KEY = _public_key(_SECOND_VERIFIER_KEY)
_TASK_ISSUER_PUBLIC_KEY = _public_key(_TASK_ISSUER_KEY)
_PRODUCER_PUBLIC_KEY = _public_key(_PRODUCER_KEY)
# Four distinct organizations. Separate keypairs inside one organization are
# separation on paper; the point of an independent verifier is a second party.
_TRUSTED_VERIFIERS = {
    _VERIFIER_ID: {
        "public_key_b64": _VERIFIER_PUBLIC_KEY,
        "implementation_sha256": _VERIFIER_IMPLEMENTATION,
        "release_sha256": _VERIFIER_RELEASE,
        "organization": "proof-kernel-consortium",
    },
    _SECOND_VERIFIER_ID: {
        "public_key_b64": _SECOND_VERIFIER_PUBLIC_KEY,
        "implementation_sha256": _SECOND_VERIFIER_IMPLEMENTATION,
        "release_sha256": _SECOND_VERIFIER_RELEASE,
        "organization": "independent-evaluation-board",
    },
}
_TRUSTED_TASK_ISSUERS = {
    _TASK_ISSUER_ID: {
        "public_key_b64": _TASK_ISSUER_PUBLIC_KEY,
        "implementation_sha256": _TASK_ISSUER_IMPLEMENTATION,
        "release_sha256": _TASK_ISSUER_RELEASE,
        "organization": "held-out-task-custodian",
    }
}
_TRUSTED_PRODUCERS = {
    _PRODUCER_ID: {
        "public_key_b64": _PRODUCER_PUBLIC_KEY,
        "implementation_sha256": _PRODUCER_IMPLEMENTATION,
        "release_sha256": _PRODUCER_RELEASE,
        "organization": "aura-lab-org",
    }
}
_VERIFIER_KEYS = {
    _VERIFIER_ID: (_VERIFIER_KEY, _VERIFIER_IMPLEMENTATION, _VERIFIER_RELEASE),
    _SECOND_VERIFIER_ID: (
        _SECOND_VERIFIER_KEY,
        _SECOND_VERIFIER_IMPLEMENTATION,
        _SECOND_VERIFIER_RELEASE,
    ),
}


# The architecture the claim is actually about. A toy decoder here made every
# FLOPs comparison internally consistent and unrelated to a 32B model, which
# is precisely what the profile/parameter reconciliation now refuses.
_RESIDENT_PROFILE = ModelComputeProfile(
    model_type="qwen2",
    hidden_size=5120,
    intermediate_size=27648,
    num_hidden_layers=64,
    num_attention_heads=40,
    num_key_value_heads=8,
    vocab_size=152064,
    head_dim=128,
)
_RESIDENT_PROFILE_RECEIPT = _RESIDENT_PROFILE.to_receipt()


def _trial_accounting(task_payload_sha256: str) -> tuple[dict, dict]:
    profile = _RESIDENT_PROFILE
    ledger = ResourceLedger(profile)
    ledger.charge(
        "fixture_inference",
        transformer_layer_apps=64 * 16,
        attention_query_key_pairs=400,
        output_head_tokens=10,
        verifier_calls=1,
        verifier_input_bytes=128,
        verifier_output_bytes=8,
        host_scalar_ops=1_000,
    )
    information = build_information_receipt(
        sources=[
            {
                "source_id": "held_out_task",
                "kind": "task_prompt",
                "content_sha256": task_payload_sha256,
                "byte_count": 128,
                "token_count": 32,
            }
        ],
        policies={
            "decode": "d" * 64,
            "tool": "c" * 64,
            "verifier": "a" * 64,
        },
    )
    return ledger.to_receipt(), information


# An external taxonomy the preregistration pins by digest. Breadth is counted
# in capability classes, so three domains that all reduce to arithmetic would
# not clear it however differently they are spelled.
_DOMAIN_TAXONOMY = {
    "ontology_id": "aura-frontier-capability-ontology-v1",
    "domains": {
        "math": {"capability_class": "formal_quantitative_reasoning"},
        "coding": {"capability_class": "program_synthesis_and_repair"},
        "science": {"capability_class": "empirical_explanation"},
    },
}


_TRANSPARENCY_LOG_ID = "aura-frontier-evidence-log"


def _anchor_log(leaves: list[str]) -> tuple[str, dict[str, dict]]:
    """A two-leaf append-only log, with an audit path for each leaf.

    Small enough to read, real enough to verify: the certificate recomputes
    the root from the leaf and its sibling and compares it against the root
    pinned in the trust config.
    """
    leaf_hashes = [_merkle_leaf_hash(bytes.fromhex(leaf)) for leaf in leaves]
    root = hashlib.sha256(
        b"\x01" + bytes.fromhex(leaf_hashes[0]) + bytes.fromhex(leaf_hashes[1])
    ).hexdigest()
    anchors = {
        leaves[index]: {
            "log_id": _TRANSPARENCY_LOG_ID,
            "leaf_data_sha256": leaves[index],
            "leaf_index": index,
            "tree_size": 2,
            "inclusion_proof": [leaf_hashes[1 - index]],
        }
        for index in (0, 1)
    }
    return root, anchors


def _trust_config() -> dict:
    return {
        "schema": TRUST_CONFIG_SCHEMA,
        "verification_kernel_sha256": verifier_implementation_sha256(),
        "producers": _TRUSTED_PRODUCERS,
        "task_issuers": _TRUSTED_TASK_ISSUERS,
        "verifiers": _TRUSTED_VERIFIERS,
        "transparency_logs": dict(_TRANSPARENCY_LOG_PINS),
    }


#: Roots pinned out of band, one entry per log this fixture has ever built.
#: A producer-supplied root would prove nothing — the point of the anchor is
#: that the root is not the producer's to choose. Entries ACCUMULATE and each
#: log id embeds its own root, so bundles built in any order stay verifiable
#: no matter when they are certified.
_TRANSPARENCY_LOG_PINS: dict[str, dict] = {}


def _refresh_timestamp_anchors(bundle: dict) -> None:
    """Log the preregistration and the task commitment, then pin the root."""
    root, anchors = _anchor_log(
        [bundle["preregistration_sha256"], bundle["task_commitment_sha256"]]
    )
    log_id = f"{_TRANSPARENCY_LOG_ID}-{root[:12]}"
    _TRANSPARENCY_LOG_PINS[log_id] = {"root_sha256": root, "tree_size": 2}
    bundle["timestamp_anchors"] = {
        "preregistration": {
            **anchors[bundle["preregistration_sha256"]],
            "log_id": log_id,
            # Logged before the first task was generated (1001.0).
            "logged_at": 900.0,
        },
        "task_commitment": {
            **anchors[bundle["task_commitment_sha256"]],
            "log_id": log_id,
            # Logged before the first evaluation started (1201.0).
            "logged_at": 1200.5,
        },
    }


def _rebind_trial_identifiers(trial: dict, *, worker_boot_id: str = "b" * 32) -> None:
    """Re-derive an arm's identifiers after its task or output digests change.

    Episode and request ids are derived from the trial's own record, so a
    caller that rewrites the task payload or the outputs — the raw-artifact
    package builder does exactly that — has to re-derive them. The runtime
    integrity proof binds the episode id, so it is rebound too.
    """
    treatment = trial["treatment_receipt"]
    treatment["episode_id"] = _expected_arm_identifier(
        trial, "treatment", worker_boot_id
    )
    treatment["input_tokens_sha256"] = trial["task_payload_sha256"]
    worker_identity = treatment["worker_identity"]
    treatment["fast_weight_learning"] = accepted_fast_weight_learning(
        episode_id=treatment["episode_id"],
        input_tokens_sha256=trial["task_payload_sha256"],
    )
    attach_bound_runtime_integrity(treatment, worker_identity=worker_identity)
    control = trial["control_receipt"]
    control["request_id"] = _expected_arm_identifier(trial, "control", worker_boot_id)
    if "runtime_integrity" in control:
        control["input_tokens_sha256"] = trial["task_payload_sha256"]
        attach_bound_runtime_integrity(
            control, worker_identity=control["worker_identity"]
        )


def _set_outcome(trial: dict, *, treatment: bool, control: bool) -> None:
    """Set an arm outcome AND the score behind it.

    The certificate checks that each boolean is the preregistered threshold
    applied to the recorded score, so a test that flips only the boolean is
    building a bundle no honest producer could emit.
    """
    trial["treatment_success"] = treatment
    trial["control_success"] = control
    trial["treatment_score"] = 0.9 if treatment else 0.2
    trial["control_score"] = 0.8 if control else 0.2


def _refresh_task_commitment(bundle: dict) -> None:
    # The manifest digest comes from the PRODUCTION builder. A fixture that
    # rebuilt the row shape by hand would keep signing whatever the fixture
    # thought the manifest was, so widening the real manifest would leave
    # every test green while the new fields went uncommitted.
    manifest_sha256 = _task_manifest_sha256(bundle["trials"])
    diversity_sha256 = canonical_sha256(bundle["task_diversity"])
    task_commitment_sha256 = canonical_sha256(
        {
            "architecture_freeze_sha256": bundle["preregistration"]["architecture_freeze_sha256"],
            "preregistration_sha256": bundle["preregistration_sha256"],
            "task_count": len(bundle["trials"]),
            "task_manifest_sha256": manifest_sha256,
            "task_diversity_sha256": diversity_sha256,
            "blinding_map_sha256": bundle["blinding"]["arm_label_map_sha256"],
        }
    )
    payload = {
        "architecture_freeze_sha256": bundle["preregistration"]["architecture_freeze_sha256"],
        "preregistration_sha256": bundle["preregistration_sha256"],
        "task_commitment_sha256": task_commitment_sha256,
        "task_manifest_sha256": manifest_sha256,
        "task_diversity_sha256": diversity_sha256,
        "blinding_map_sha256": bundle["blinding"]["arm_label_map_sha256"],
        "task_count": len(bundle["trials"]),
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
    # The commitment digest just changed and the anchor commits to it, so the
    # log entry has to be rebuilt alongside.
    if "timestamp_anchors" in bundle:
        _refresh_timestamp_anchors(bundle)


def _refresh_producer_attestation(bundle: dict, *, produced_at: float = 1900.0) -> None:
    """The producer signs its own bundle with a pinned key.

    Without this the producer identity was a string the bundle wrote about
    itself, and a verifier counted as independent by differing from it.
    """
    payload = {
        "producer_id": _PRODUCER_ID,
        "evidence_payload_sha256": evidence_payload_sha256(bundle),
        "preregistration_sha256": bundle["preregistration_sha256"],
        "task_commitment_sha256": bundle["task_commitment_sha256"],
        "raw_artifact_manifest_sha256": bundle["raw_artifact_manifest_sha256"],
        "produced_at": produced_at,
    }
    bundle["producer_attestation"] = {
        "schema": PRODUCER_ATTESTATION_SCHEMA,
        "signed_payload": payload,
        "signer": {
            "algorithm": "Ed25519",
            "signer_id": _PRODUCER_ID,
            "public_key_b64": _PRODUCER_PUBLIC_KEY,
            "signature_b64": base64.b64encode(
                _PRODUCER_KEY.sign(canonical_json_bytes(payload))
            ).decode("ascii"),
        },
    }


def _signed_attestation(
    bundle: dict,
    verifier_id: str,
    verified_at: float,
    artifact_receipt: dict | None = None,
) -> dict:
    """Sign an attestation over the receipt this verifier actually produced.

    The standalone verifier opens real files and gets a real receipt; the
    in-memory tests use the stand-in below. Either way the signature has to be
    over the receipt the certifying call is handed, because binding it is what
    makes the signature evidence of work rather than of key possession.
    """
    key, implementation, release = _VERIFIER_KEYS[verifier_id]
    payload = {
        "accepted": True,
        "claim_tier": "PROVEN",
        "producer_id": bundle["producer_id"],
        "evidence_payload_sha256": evidence_payload_sha256(bundle),
        "preregistration_sha256": bundle["preregistration_sha256"],
        "implementation_sha256": implementation,
        "verifier_release_sha256": release,
        "raw_artifact_manifest_sha256": bundle["raw_artifact_manifest_sha256"],
        "task_commitment_sha256": bundle["task_commitment_sha256"],
        # The receipt this verifier produced by opening the raw artifacts.
        "raw_artifact_receipt_sha256": canonical_sha256(
            dict(artifact_receipt)
            if artifact_receipt is not None
            else _raw_artifact_receipt(bundle)
        ),
        "verified_at": verified_at,
    }
    return {
        "schema": INDEPENDENT_ATTESTATION_SCHEMA,
        "signed_payload": payload,
        "signer": {
            "algorithm": "Ed25519",
            "signer_id": verifier_id,
            "public_key_b64": _public_key(key),
            "signature_b64": base64.b64encode(
                key.sign(canonical_json_bytes(payload))
            ).decode("ascii"),
        },
    }


def _refresh_attestation(
    bundle: dict,
    *,
    verified_at: float = 2000.0,
    verifier_ids: tuple[str, ...] = (_VERIFIER_ID, _SECOND_VERIFIER_ID),
    artifact_receipt: dict | None = None,
) -> None:
    # The producer signature covers the evidence payload, so it is refreshed
    # first and then excluded from what the verifiers hash.
    _refresh_producer_attestation(bundle)
    bundle["independent_verifiers"] = [
        _signed_attestation(bundle, verifier_id, verified_at, artifact_receipt)
        for verifier_id in verifier_ids
    ]


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
        trusted_producers=_TRUSTED_PRODUCERS,
        trusted_transparency_logs=_TRANSPARENCY_LOG_PINS,
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
        # The architecture the estimator was pointed at, and the per-trial
        # forward-pass budget. 1000 layer applications over 64 layers is not
        # a whole number of passes, so the fixture charges whole stacks.
        "compute_profile_sha256": _RESIDENT_PROFILE_RECEIPT["profile_sha256"],
        "max_forward_passes_per_trial": 64,
        # params_unchanged rests on a fixed-stride canary. One tensor in eight
        # is the weakest sampling this claim may be made on; the fixture's
        # stride-7 canary covers 19 of 128 leaves.
        "min_parameter_canary_tensor_coverage": 0.125,
        "domain_taxonomy_sha256": canonical_sha256(_DOMAIN_TAXONOMY),
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
        # Where the pass line sits, and how far it may be moved before the
        # gain has to survive on its own.
        "success_threshold": 0.6,
        "threshold_sensitivity_band": 0.1,
        # Release budgets. Beating your own ablation says nothing about
        # whether this build is worse than the one already shipped.
        "max_success_rate_regression": 0.02,
        "max_latency_regression_ratio": 1.1,
        "max_compute_regression_ratio": 1.1,
        "safety_suite_sha256": "9" * 64,
        "max_safety_violations": 0,
        "max_expected_calibration_error": 0.05,
        # Sized against a McNemar alternative, not a trial count. With 40
        # trials per domain the study can detect a treatment that wins 75%
        # of disagreements at alpha 0.05 with power above 0.8; it could not
        # detect a much smaller edge, and the certificate now says so.
        "target_power": 0.8,
        "preregistered_discordant_win_share": 0.75,
        "max_order_effect": 0.1,
    }
    if external:
        prereg["control_model_id"] = "frontier-model-x"
        prereg["control_model_build_fingerprint"] = "build-2026-06"
        # A RECOGNISED frontier lab. The placeholder "frontier-provider"
        # is now refused: a frontier claim must name a lab that ships
        # frontier models, or "beat an external frontier" could be
        # earned against anything the producer chose to call one.
        prereg["control_provider"] = "openai"
    else:
        prereg["control_checkpoint_fingerprint"] = checkpoint
    trials = []
    index = 0
    for domain in prereg["domains"]:
        for cell in range(40):
            trial_id = f"{domain}-{cell}"
            task_payload_sha256 = canonical_sha256(["task", trial_id])
            treatment_resource, treatment_information = _trial_accounting(
                task_payload_sha256
            )
            control_resource, control_information = _trial_accounting(
                task_payload_sha256
            )
            # Identifiers DERIVED from the trial's own record, using the
            # production function so the two cannot drift. A free string
            # could be swapped between trials or minted afterwards to make a
            # lineage look clean, and global uniqueness would still hold.
            identity_source = {
                "trial_id": trial_id,
                "task_id": f"heldout-{trial_id}",
                "task_payload_sha256": task_payload_sha256,
                "treatment_output_sha256": canonical_sha256(["treatment", trial_id]),
                "control_output_sha256": canonical_sha256(["control", trial_id]),
                "evaluation_started_at": 1201.0 + index,
            }
            treatment_episode_id = _expected_arm_identifier(
                identity_source, "treatment", "b" * 32
            )
            control_request_id = _expected_arm_identifier(
                identity_source, "control", "b" * 32
            )
            trials.append(
                {
                    "trial_id": trial_id,
                    "task_id": f"heldout-{trial_id}",
                    "domain": domain,
                    "held_out": True,
                    "contamination_scan_passed": True,
                    # The boolean above is a claim. This is the measurement
                    # behind it: which scanner ran, how, what it was held to,
                    # and what it actually found.
                    "contamination_scan": {
                        "scanner_implementation_sha256": "b" * 64,
                        "method": "13gram_overlap",
                        "max_overlap_threshold": 0.02,
                        "max_overlap_observed": 0.0,
                    },
                    "task_generated_at": 1001.0 + index,
                    "evaluation_started_at": 1201.0 + index,
                    # When the evidence actually existed. The attestation is
                    # timestamped against the last of these, not against when
                    # the run began.
                    "evaluation_completed_at": 1401.0 + index,
                    "scoring_completed_at": 1601.0 + index,
                    "verifier_blinded": True,
                    "verifier_receipt_sha256": canonical_sha256(["verifier", trial_id]),
                    "task_payload_sha256": task_payload_sha256,
                    "treatment_output_sha256": canonical_sha256(["treatment", trial_id]),
                    "control_output_sha256": canonical_sha256(["control", trial_id]),
                    "scorer_config_sha256": "f" * 64,
                    "scorer_implementation_sha256": "a" * 64,
                    "treatment_information_sha256": treatment_information[
                        "receipt_sha256"
                    ],
                    "control_information_sha256": control_information[
                        "receipt_sha256"
                    ],
                    "treatment_information": treatment_information,
                    "control_information": control_information,
                    "treatment_tool_policy_sha256": "c" * 64,
                    "control_tool_policy_sha256": "c" * 64,
                    "treatment_decode_policy_sha256": "d" * 64,
                    "control_decode_policy_sha256": "d" * 64,
                    # Order alternates in BLOCKS of four so it is orthogonal
                    # to the every-fourth-cell control success below. Simple
                    # parity put every control win in a treatment-first run,
                    # which is a confounded design the order-effect check
                    # would (correctly) refuse.
                    "run_order": (
                        "treatment_first" if (cell % 8) < 4 else "control_first"
                    ),
                    # The booleans are the preregistered threshold applied
                    # to these scores, and the margin survives moving the
                    # line anywhere inside the preregistered band.
                    "treatment_score": 0.9,
                    "control_score": 0.8 if cell % 4 == 0 else 0.2,
                    "treatment_success": True,
                    "control_success": cell % 4 == 0,
                    "treatment_compute": {
                        "estimated_flops": treatment_resource["estimated_flops"],
                        "layer_apps": 64 * 16,
                        "estimator_sha256": "e" * 64,
                        "resource_accounting": treatment_resource,
                    },
                    "control_compute": {
                        "estimated_flops": control_resource["estimated_flops"],
                        "layer_apps": 64 * 16,
                        "estimator_sha256": "e" * 64,
                        "resource_accounting": control_resource,
                    },
                    "treatment_receipt": {
                        "episode_id": treatment_episode_id,
                        "checkpoint_fingerprint": checkpoint,
                        "checkpoint_fingerprint_method": "sha256",
                        "checkpoint_file_count": 8,
                        "worker_boot_id": "b" * 32,
                        "installed_app_build_sha256": app_build,
                        "schedule_hash": "3" * 64,
                        "params_unchanged": True,
                        # CP126 6090a5ae + 01e8b3c1: the TREATMENT arm is the
                        # one whose effect gets published, so its integrity
                        # claims need the same digest evidence the control
                        # already carried. EpisodeReceipt.to_dict() emits
                        # weight_integrity + integrity_verdicts for real
                        # episodes; the fixture now matches the producer.
                        "weight_integrity": {
                            "algorithm": "sha256",
                            "version": 1,
                            "params_before": canonical_sha256(
                                ["treatment-params", trial_id]
                            ),
                            "params_after": canonical_sha256(
                                ["treatment-params", trial_id]
                            ),
                            "canary_before": canonical_sha256(
                                ["treatment-canary", trial_id]
                            ),
                            "canary_after": canonical_sha256(
                                ["treatment-canary", trial_id]
                            ),
                        },
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
                            "request_id": control_request_id,
                            "model_id": "frontier-model-x",
                            "model_build_fingerprint": "build-2026-06",
                            "provider": "openai",
                            "provider_receipt_sha256": canonical_sha256(
                                ["provider", trial_id]
                            ),
                        }
                        if external
                        else {
                            "request_id": control_request_id,
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
                            "worker_boot_id": "b" * 32,
                            "installed_app_build_sha256": app_build,
                            # CP126 869a0ce4: a control is vanilla only when
                            # every enhancement is positively declared OFF.
                            # An absent field is undeclared, not disabled.
                            **_VANILLA_CONTROL_MANIFEST,
                        }
                    ),
                }
            )
            worker_identity = complete_worker_identity(
                boot_id="b" * 32,
                pid=4242,
                model_path="/models/frontier-fixture",
            )
            treatment = trials[-1]["treatment_receipt"]
            treatment["input_tokens_sha256"] = task_payload_sha256
            treatment["worker_identity"] = worker_identity
            treatment["fast_weight_learning"] = (
                accepted_fast_weight_learning(
                    episode_id=treatment["episode_id"],
                    input_tokens_sha256=task_payload_sha256,
                )
            )
            attach_bound_runtime_integrity(
                treatment,
                worker_identity=worker_identity,
            )
            if not external:
                control = trials[-1]["control_receipt"]
                control["episode_id"] = f"control-episode-{trial_id}"
                control["input_tokens_sha256"] = task_payload_sha256
                control["worker_identity"] = worker_identity
                attach_bound_runtime_integrity(
                    control,
                    worker_identity=worker_identity,
                )
            index += 1
    bundle = {
        "schema": SCHEMA,
        "producer_id": _PRODUCER_ID,
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
            "worker_boot_id": "b" * 32,
            "loaded_in_installed_app": True,
            "installed_app_bundle_id": "com.aura.desktop",
            "installed_app_build_sha256": app_build,
            "worker_binary_sha256": "5" * 64,
            "build_provenance_sha256": "6" * 64,
            "source_commit": "7" * 40,
            "latent_cortex_architecture_sha256": "a" * 64,
            # One builder tying the four build identifiers together. Four
            # well-formed digests with no relationship between them said
            # nothing about which source produced the binary that ran.
            "release_attestation": {
                "builder_id": "aura-release-builder",
                "source_commit": "7" * 40,
                "rebuild_worker_binary_sha256": "5" * 64,
                "build_provenance_sha256": "6" * 64,
                "installed_app_build_sha256": app_build,
                "attested_at": 800.0,
            },
            "compute_profile": dict(_RESIDENT_PROFILE_RECEIPT),
        },
        "raw_artifact_manifest_sha256": "9" * 64,
        "domain_taxonomy": _DOMAIN_TAXONOMY,
        "release_readiness": {
            "previous_release": {
                "certificate_sha256": "c" * 64,
                "treatment_success_rate": 0.98,
                "median_latency": 2.0,
                "median_compute": 1.0e12,
            },
            "median_latency": 1.9,
            "median_compute": 1.0e12,
            "safety_suite": {
                "suite_sha256": "9" * 64,
                "cases_run": 512,
                "violations": 0,
            },
            "calibration": {
                "method": "equal_mass_binned_ece",
                "bins": 15,
                "expected_calibration_error": 0.021,
            },
        },
        "trials": trials,
        # The issuer's own clustering, committed before evaluation. Each task
        # is its own family here, so the effective sample equals the trial
        # count — which is what a genuinely diverse task set looks like.
        "task_diversity": {
            "method": "minhash_jaccard_13gram",
            "similarity_threshold": 0.6,
            "max_pairwise_similarity": 0.11,
            "task_families": {
                trial["task_id"]: f"family-{trial['task_id']}" for trial in trials
            },
        },
    }
    # Blinding as evidence rather than a per-trial boolean: the arm-label
    # assignment is committed before evaluation, revealed after the last
    # score by the role that held it, and the scorer's inputs were scanned
    # for arm markers.
    bundle["blinding"] = {
        "arm_label_map_sha256": canonical_sha256(
            {trial["trial_id"]: trial["run_order"] for trial in trials}
        ),
        "method": "arm_labels_replaced_with_opaque_ids",
        "revealed_at": 1800.0,
        "revealed_by": _TASK_ISSUER_ID,
        "marker_scan": {
            "scanner_implementation_sha256": "7" * 64,
            "method": "arm_marker_regex_sweep",
            "markers_checked": 24,
            "markers_found": 0,
        },
    }
    _refresh_task_commitment(bundle)
    _refresh_timestamp_anchors(bundle)
    # The producer signature is not part of the verifier quorum: an unsigned
    # bundle is one nobody has attested to yet, and its producer is still a
    # pinned identity.
    _refresh_producer_attestation(bundle)
    if include_attestation:
        _refresh_attestation(bundle)
    return bundle


__all__ = [
    "_DOMAIN_TAXONOMY",
    "_PRODUCER_ID",
    "_RESIDENT_PROFILE",
    "_SECOND_VERIFIER_ID",
    "_TASK_ISSUER_ID",
    "_TRUSTED_PRODUCERS",
    "_set_outcome",
    "_signed_attestation",
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
    "_TRANSPARENCY_LOG_PINS",
    "_rebind_trial_identifiers",
    "_refresh_producer_attestation",
    "_refresh_timestamp_anchors",
    "_refresh_task_commitment",
    "_trust_config",
]
