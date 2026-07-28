from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex import frontier_tasks as frontier_tasks_runtime
from core.brain.llm.latent_cortex.campaign_journal import (
    canonical_json_bytes as trust_canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    CONTAMINATION_AUDITOR,
    EVIDENCE_VERIFIER,
    TASK_ISSUER,
    build_role_attestation,
    validate_campaign_trust_policy,
)
from core.brain.llm.latent_cortex.frontier_tasks import (
    FINAL_ANSWER_MARKER,
    generate_task,
)
from core.learning import recurrent_grpo as recurrent_grpo_runtime
from core.learning import verified_transition_episode as transition_runtime
from core.learning import verified_transition_reward as reward_runtime
from core.learning import verified_transition_training_evidence as training_evidence_runtime
from core.learning import verified_transition_update as update_runtime
from core.learning.verified_transition_campaign import (
    VerifiedTransitionCampaignError,
    VerifiedTransitionCampaignLedger,
    build_transition_campaign_manifest,
    campaign_group_from_manifest,
)
from core.learning.verified_transition_episode import (
    ExternalAttemptLedger,
    TransitionArtifactStore,
    TransitionTrustContext,
    VerifiedTransitionError,
    build_attempt_ledger_event_payload,
    build_attempt_ledger_open_payload,
    build_attempt_ledger_terminal_payload,
    build_calibration_case,
    build_calibration_payload,
    build_campaign_runner_journal_payload,
    build_evidence_verifier_journal_payload,
    build_execution_manifest,
    build_execution_observer_payload,
    build_frontier_task_issuer_payload,
    build_frontier_witness_payload,
    build_generation_trace_payload,
    build_reasoning_pass_receipt,
    build_transition_attempt_journal,
    build_verified_transition_episode,
    canonical_candidate_model_input,
    canonical_json_bytes,
    capture_execution_process_observation,
    execution_observer_implementation_identity,
    issue_frontier_verifier_authority,
    planned_transition_immutable_context_sha256,
    seal_calibration_evidence,
    strict_canonical_json_loads,
    validate_frontier_verifier_authority,
    validate_reasoning_pass_receipt,
    validate_verified_transition_episode,
    verifier_implementation_identity,
)
from core.learning.verified_transition_group_admission import (
    TransitionGroupPlanEntry,
    VerifiedTransitionGroupError,
    build_transition_group_manifest,
    build_verified_transition_group_admission,
    sampling_config_sha256,
    validate_verified_transition_group_admission,
)
from core.learning.verified_transition_reward import (
    TransitionRewardConfig,
    VerifiedTransitionEvidence,
    VerifiedTransitionRewardAdmissionError,
    VerifiedTransitionRewardError,
    build_verified_transition_reward_batch,
    require_optimizer_admission,
    validate_verified_transition_reward_batch,
)
from core.learning.verified_transition_training_evidence import (
    VerifiedTransitionReplayGroup,
    VerifiedTransitionTrainingEvidenceError,
    validate_verified_transition_training_evidence,
)
from core.learning.verified_transition_update import (
    VerifiedTransitionUpdateError,
    VerifiedTransitionUpdateJournal,
    apply_verified_transition_group_update,
    commit_staged_verified_transition_update,
    reconcile_interrupted_verified_transition_update,
    recover_committed_campaign_group,
    validate_verified_transition_reconciliation_receipt,
    validate_verified_transition_update_receipt,
)
from core.runtime.resource_observation import HostResourceObserver, ObservationSource
from tools.independent_paired_campaign_scoring import (
    score_frontier_response_independently,
)

PROTOCOL_SHA256 = "9" * 64
OBSERVED_AT = 1_800_000_300
PASS_0_AT = 1_800_000_210_000_000_000


def _exact_objective_receipt() -> dict[str, Any]:
    return {
        "schema": recurrent_grpo_runtime.RECURRENT_GRPO_SCHEMA,
        "mode": "exact_adjoint_single_update",
        "advantage_report": {"schema": "aura.test.advantage.v1"},
        "reference_kl": 0.0,
        "old_policy_approx_kl": 0.0,
        "clip_fraction": 0.0,
        "policy_loss": -0.125,
        "objective_at_sampling": -0.125,
        "gradient_surrogate_value": -0.125,
        "completion_count": 2,
        "token_count": 2,
        "branch_indices": [0, 1],
        "has_gradient": True,
    }


def _seal_float_receipt(document: dict[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(document)
    body.pop("receipt_sha256", None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return {
        **body,
        "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _seal_exact_adjoint_receipt(document: dict[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(document)
    body.pop("receipt_sha256", None)
    input_payload = {
        key: body[key]
        for key in (
            "policy_sha256",
            "prompt_tokens_sha256",
            "prompt_token_count",
            "answer_tokens_sha256",
            "answer_token_count",
            "bridge_tokens_sha256",
            "bridge_token_count",
            "token_loss_weights",
            "execution_spec_sha256",
            "recurrent_depth",
            "execution_branch_count",
            "branch_indices",
            "diversity_weight",
            "diversity_target_cos",
            "trajectory_config",
        )
    }
    encoded = json.dumps(
        input_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    body["objective_input_sha256"] = hashlib.sha256(
        b"aura.exact_adjoint_input.v1\0" + encoded
    ).hexdigest()
    return _seal_float_receipt(body)


def _trajectory_objective_receipt(admission_sha256: str) -> dict[str, Any]:
    trajectory_config = {
        "schema": "aura.exact_adjoint_trajectory_objective.v1",
        "probe_steps": [1, 2],
        "improvement_weight": 0.5,
        "improvement_margin": 0.2,
        "displacement_weight": 0.0,
        "displacement_floor": 0.01,
        "oscillation_weight": 0.0,
    }
    exact = _seal_exact_adjoint_receipt(
        {
            "schema": "aura.exact_adjoint_trajectory_objective_receipt.v2",
            "value": 0.1,
            "terminal_value": 0.0,
            "diversity_value": 0.0,
            "trajectory_values": {
                "improvement": 0.1,
                "displacement": 0.0,
                "oscillation": 0.0,
            },
            "step_losses": {"1": [1.0], "2": [1.0]},
            "displacements": [],
            "oscillation_cosines": [],
            "diversity_cosines": [0.0],
            "branch_indices": [0],
            "execution_spec_sha256": "2" * 64,
            "recurrent_depth": 2,
            "execution_branch_count": 2,
            "diversity_weight": 0.0,
            "diversity_target_cos": 0.98,
            "policy_sha256": "1" * 64,
            "prompt_tokens_sha256": "4" * 64,
            "prompt_token_count": 1,
            "answer_tokens_sha256": "7" * 64,
            "answer_token_count": 1,
            "bridge_tokens_sha256": hashlib.sha256(b"[]").hexdigest(),
            "bridge_token_count": 0,
            "token_loss_weights": [0.0],
            "trajectory_config": trajectory_config,
        }
    )
    group = _seal_float_receipt(
        {
            "schema": recurrent_grpo_runtime.VERIFIED_TRAJECTORY_GROUP_SCHEMA,
            "group_admission_sha256": admission_sha256,
            "reward_receipt_sha256": "3" * 64,
            "policy_sha256": "1" * 64,
            "execution_spec_sha256": "2" * 64,
            "prompt_tokens_sha256": "4" * 64,
            "sample_receipt_sha256s": ["5" * 64, "6" * 64],
            "completion_tokens_sha256s": ["7" * 64, "8" * 64],
            "sample_branch_indices": [0, 1],
            "execution_branch_count": 2,
            "verified_rewards": [1.0, 0.0],
            "advantage_clip": 4.0,
            "advantages": [1.0, -1.0],
            "config": {
                "schema": recurrent_grpo_runtime.VERIFIED_TRAJECTORY_GROUP_SCHEMA,
                "trajectory_config": trajectory_config,
                "diversity_weight": 0.0,
                "diversity_target_cos": 0.98,
                "improvement_credit": "positive_advantage_l1_normalized",
                "structural_scope": ("all_exchange_coupled_branches_once_per_prompt"),
                "anchor_selection": ("maximum_verified_reward_then_lowest_index"),
            },
            "positive_completion_indices": [0],
            "positive_advantage_weights": [1.0],
            "anchor_completion_index": 0,
            "anchor_branch_index": 0,
            "improvement_receipts": [
                {
                    "completion_index": 0,
                    "completion_tokens_sha256": "7" * 64,
                    "advantage_weight": 1.0,
                    "objective_receipt": exact,
                }
            ],
            "structural_receipt": None,
            "trajectory_objective_value": 0.1,
        }
    )
    base = _exact_objective_receipt()
    return {
        **base,
        "mode": "exact_adjoint_trajectory_composite_single_update",
        "advantage_report": {
            "schema": "aura.grpo.v2",
            "advantages": [1.0, -1.0],
            "mean_reward": 0.5,
            "reward_std": 0.5,
            "degenerate": False,
            "all_correct": False,
            "all_wrong": False,
            "uniform_partial": False,
        },
        "trajectory_objective_value": 0.1,
        "composite_objective_at_sampling": -0.025,
        "composite_gradient_surrogate_value": -0.025,
        "trajectory_receipt": group,
    }


def _trajectory_source_binding(
    objective_receipt: dict[str, Any],
) -> dict[str, Any]:
    trajectory = objective_receipt["trajectory_receipt"]
    payload = {
        "schema": recurrent_grpo_runtime.VERIFIED_TRAJECTORY_SOURCE_SCHEMA,
        **{
            field: copy.deepcopy(trajectory[field])
            for field in (
                "group_admission_sha256",
                "reward_receipt_sha256",
                "policy_sha256",
                "execution_spec_sha256",
                "prompt_tokens_sha256",
                "sample_receipt_sha256s",
                "completion_tokens_sha256s",
                "sample_branch_indices",
                "execution_branch_count",
                "verified_rewards",
                "advantage_clip",
                "config",
            )
        },
    }
    return {
        **payload,
        "source_sha256": hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
    }


def _trajectory_objective_receipt_from_source(
    source_binding: dict[str, Any],
) -> dict[str, Any]:
    receipt = _trajectory_objective_receipt(source_binding["group_admission_sha256"])
    group = copy.deepcopy(receipt["trajectory_receipt"])
    for field in (
        "group_admission_sha256",
        "reward_receipt_sha256",
        "policy_sha256",
        "execution_spec_sha256",
        "prompt_tokens_sha256",
        "sample_receipt_sha256s",
        "completion_tokens_sha256s",
        "sample_branch_indices",
        "execution_branch_count",
        "verified_rewards",
        "advantage_clip",
        "config",
    ):
        group[field] = copy.deepcopy(source_binding[field])

    advantage_report = recurrent_grpo_runtime.group_advantages(
        source_binding["verified_rewards"],
        clip=source_binding["advantage_clip"],
    )
    advantages = list(advantage_report["advantages"])
    positive = [index for index, value in enumerate(advantages) if value > 0.0]
    assert positive == [0]
    weights = [advantages[index] / sum(advantages[item] for item in positive) for index in positive]
    anchor = max(
        range(len(source_binding["verified_rewards"])),
        key=lambda index: (source_binding["verified_rewards"][index], -index),
    )

    exact = copy.deepcopy(group["improvement_receipts"][0]["objective_receipt"])
    exact["execution_spec_sha256"] = source_binding["execution_spec_sha256"]
    exact["execution_branch_count"] = source_binding["execution_branch_count"]
    exact["branch_indices"] = [source_binding["sample_branch_indices"][positive[0]]]
    exact["policy_sha256"] = source_binding["policy_sha256"]
    exact["prompt_tokens_sha256"] = source_binding["prompt_tokens_sha256"]
    exact["answer_tokens_sha256"] = source_binding["completion_tokens_sha256s"][positive[0]]
    exact["trajectory_config"] = copy.deepcopy(source_binding["config"]["trajectory_config"])
    exact = _seal_exact_adjoint_receipt(exact)

    group["advantages"] = advantages
    group["positive_completion_indices"] = positive
    group["positive_advantage_weights"] = weights
    group["anchor_completion_index"] = anchor
    group["anchor_branch_index"] = source_binding["sample_branch_indices"][anchor]
    group["improvement_receipts"] = [
        {
            "completion_index": positive[0],
            "completion_tokens_sha256": source_binding["completion_tokens_sha256s"][positive[0]],
            "advantage_weight": weights[0],
            "objective_receipt": exact,
        }
    ]
    group = _seal_float_receipt(group)
    receipt["advantage_report"] = advantage_report
    receipt["trajectory_receipt"] = group
    return receipt


PASS_1_AT = 1_800_000_215_000_000_000
VERIFIER_AT = 1_800_000_221_000_000_000
RUNNER_AT = 1_800_000_222
EVIDENCE_AT = 1_800_000_223
TRACE_0_AT = PASS_0_AT + 1_000_000_000
TRACE_1_AT = PASS_1_AT + 1_000_000_000
OBSERVER_0_AT = PASS_0_AT + 2_000_000_000
OBSERVER_1_AT = PASS_1_AT + 2_000_000_000
LEDGER_TERMINAL_AT = 1_800_000_220_000_000_000


def _public_raw(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _public_pem(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _pin(
    role: str,
    key: Ed25519PrivateKey,
    *,
    implementation_sha256: str | None = None,
) -> dict[str, str]:
    raw = _public_raw(key)
    return {
        "signer_id": f"{role}-signer",
        "organization_id": f"{role}-organization",
        "public_key_b64": base64.b64encode(raw).decode("ascii"),
        "key_id": hashlib.sha256(raw).hexdigest(),
        "implementation_sha256": implementation_sha256
        or hashlib.sha256(f"{role}:implementation".encode()).hexdigest(),
        "release_sha256": hashlib.sha256(f"{role}:release".encode()).hexdigest(),
        "custody_class": "external_service",
        "custody_evidence_sha256": hashlib.sha256(f"{role}:custody".encode()).hexdigest(),
    }


def _trust_fixture(
    generation_worker_identity_sha256: str,
) -> tuple[
    dict[str, Any],
    Ed25519PrivateKey,
    dict[str, Ed25519PrivateKey],
]:
    root = Ed25519PrivateKey.generate()
    role_keys = {role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES}
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "spark-060-transition-proof",
        "policy_revision": 1,
        "campaign_name": "spark-060-transition-proof",
        "protocol_sha256": PROTOCOL_SHA256,
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": 1_800_000_000,
        "not_before_unix": 1_800_000_100,
        "expires_at_unix": 1_800_086_400,
        "roles": {
            role: _pin(
                role,
                role_keys[role],
                implementation_sha256=(
                    verifier_implementation_identity(score_frontier_response_independently)
                    if role == EVIDENCE_VERIFIER
                    else execution_observer_implementation_identity()
                    if role == CONTAMINATION_AUDITOR
                    else generation_worker_identity_sha256
                    if role == CAMPAIGN_RUNNER
                    else None
                ),
            )
            for role in CAMPAIGN_TRUST_ROLES
        },
    }
    signed = trust_canonical_json_bytes(body)
    root_raw = _public_raw(root)
    policy = {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }
    return policy, root, role_keys


def _byte_encode(payload: bytes) -> list[int]:
    return list(payload)


def _byte_decode(tokens: list[int] | tuple[int, ...]) -> bytes:
    return bytes(tokens)


def _offset_byte_encode(payload: bytes) -> list[int]:
    return [value + 1 for value in payload]


def _offset_byte_decode(tokens: list[int] | tuple[int, ...]) -> bytes:
    return bytes(value - 1 for value in tokens)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _correct_response(task: Any, *, prefix: str) -> bytes:
    answer = json.dumps(
        task.reveal_for_verifier()["expected"],
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{prefix}\n{FINAL_ANSWER_MARKER} {answer}".encode()


def _wrong_response(task: Any) -> bytes:
    expected = copy.deepcopy(task.reveal_for_verifier()["expected"])
    expected["count"] += 1
    answer = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    return f"Initial attempt.\n{FINAL_ANSWER_MARKER} {answer}".encode()


def _reseal(document: dict[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(document)
    body.pop("receipt_sha256", None)
    return {
        **body,
        "receipt_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _artifact(store: TransitionArtifactStore, label: str) -> dict[str, Any]:
    return store.put_json({"schema": f"aura.test.{label}.v1", "label": label})


def _component_roots(root: Path) -> dict[str, Path]:
    roots = {
        role: root / role
        for role in (
            "base_checkpoint",
            "adapter_stack",
            "tokenizer",
            "policy",
            "personality",
            "runtime",
            "source_closure",
            "generation_worker",
        )
    }
    for role, component_root in roots.items():
        component_root.mkdir(parents=True)
        (component_root / "fixture.bin").write_bytes(f"{role}:fixture-bytes".encode())
    return roots


def _clear_append_only_for_test(path: Path) -> None:
    append_only_flag = getattr(os.stat(path, follow_symlinks=False), "st_flags", 0)
    if append_only_flag and hasattr(os, "chflags") and path.exists():
        os.chflags(path, 0, follow_symlinks=False)


def _calibration_evidence(
    *,
    policy: Any,
    role_keys: dict[str, Ed25519PrivateKey],
    task: Any,
) -> dict[str, Any]:
    calibration_tasks = [
        generate_task("mathematics", seed=817_100, difficulty=2),
        generate_task("mathematics", seed=817_101, difficulty=2),
        generate_task("mathematics", seed=817_102, difficulty=2),
    ]
    cases = [
        build_calibration_case(
            task=calibration_tasks[0],
            case_kind="canonical_positive",
            independent_scorer=score_frontier_response_independently,
        ),
        build_calibration_case(
            task=calibration_tasks[1],
            case_kind="missing_marker_negative",
            independent_scorer=score_frontier_response_independently,
        ),
        build_calibration_case(
            task=calibration_tasks[2],
            case_kind="parsed_wrong_negative",
            independent_scorer=score_frontier_response_independently,
        ),
    ]
    assert task.task_id not in {calibration_task.task_id for calibration_task in calibration_tasks}
    payload = build_calibration_payload(
        verifier_implementation_sha256=verifier_implementation_identity(
            score_frontier_response_independently
        ),
        trust_policy_sha256=policy.policy_sha256,
        cases=cases,
        independent_scorer=score_frontier_response_independently,
        acceptance_policy_sha256=_sha("strict-calibration-acceptance"),
        calibrated_at_unix_ns=1_800_000_204_000_000_000,
    )
    attestation = build_role_attestation(
        policy,
        role=EVIDENCE_VERIFIER,
        payload=payload,
        signed_at_unix=1_800_000_204,
        private_key=role_keys[EVIDENCE_VERIFIER],
    )
    return seal_calibration_evidence(
        payload,
        evidence_verifier_attestation=attestation,
    )


def _authority_expected(
    policy: Any,
    task: Any,
    *,
    execution_manifest: dict[str, Any],
    calibration_evidence: dict[str, Any],
) -> dict[str, str]:
    runner = policy.role_pin(CAMPAIGN_RUNNER)
    verifier = policy.role_pin(EVIDENCE_VERIFIER)
    return {
        "authority_id": "spark-060-frontier-authority",
        "verifier_id": "frontier-dual-replay",
        "verifier_version": "1",
        "issuer_commitment_sha256": hashlib.sha256(
            canonical_json_bytes(build_frontier_task_issuer_payload(task))
        ).hexdigest(),
        "verifier_trust_policy_sha256": policy.policy_sha256,
        "calibration_evidence_sha256": hashlib.sha256(
            canonical_json_bytes(calibration_evidence)
        ).hexdigest(),
        "execution_manifest_sha256": hashlib.sha256(
            canonical_json_bytes(execution_manifest)
        ).hexdigest(),
        "producer_identity_sha256": runner["key_id"],
        "verifier_identity_sha256": verifier["implementation_sha256"],
        "independent_witness_identity_sha256": verifier["key_id"],
    }


def _issue_authority(
    *,
    store: TransitionArtifactStore,
    task: Any,
    response: bytes,
    expected: dict[str, str],
    trust_context: TransitionTrustContext,
    policy: Any,
    role_keys: dict[str, Ed25519PrivateKey],
    task_issuer_attestation: dict[str, Any],
    issued_at: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response_artifact = store.put_bytes(
        response,
        media_type="text/plain;charset=utf-8",
    )
    witness_payload = build_frontier_witness_payload(
        store,
        task=task,
        response_artifact=response_artifact,
        expected_authority=expected,
        independent_scorer=score_frontier_response_independently,
        issued_at_unix_ns=issued_at,
    )
    witness_attestation = build_role_attestation(
        policy,
        role=EVIDENCE_VERIFIER,
        payload=witness_payload,
        signed_at_unix=issued_at // 1_000_000_000,
        private_key=role_keys[EVIDENCE_VERIFIER],
    )
    receipt = issue_frontier_verifier_authority(
        store,
        task=task,
        response_artifact=response_artifact,
        expected_authority=expected,
        independent_scorer=score_frontier_response_independently,
        trust_context=trust_context,
        task_issuer_attestation=task_issuer_attestation,
        evidence_verifier_attestation=witness_attestation,
        issued_at_unix_ns=issued_at,
        sealed_at_unix_ns=issued_at + 10,
    )
    return receipt, response_artifact


def _pass_context(
    store: TransitionArtifactStore,
    *,
    episode_id: str = "spark-060-episode-0001",
    generated_at: int,
    latent_label: str,
    task: Any,
    execution_manifest: dict[str, Any],
) -> dict[str, Any]:
    roots = execution_manifest["component_roots"]
    execution_manifest_sha256 = hashlib.sha256(canonical_json_bytes(execution_manifest)).hexdigest()
    execution_spec = store.put_json(
        {
            "schema": "aura.verified_transition.execution_spec.v2",
            "candidate_visible": False,
            "execution_manifest_sha256": execution_manifest_sha256,
            "sampling_policy_sha256": _sha("sampling-policy"),
            "recurrent_execution_spec_sha256": _sha("recurrent-execution-spec"),
        }
    )
    latent_path = store.put_json(
        {
            "schema": "aura.verified_transition.latent_path.v1",
            "candidate_visible": False,
            "mechanism_id": latent_label,
            "configuration_sha256": _sha(f"{latent_label}:configuration"),
            "recurrence_steps": 6,
            "branch_count": 2,
        }
    )
    empty_artifacts = {
        field: store.put_json(
            {
                "schema": schema,
                "candidate_visible": False,
                "items": [],
            }
        )
        for field, schema in {
            "tool_snapshot_artifact": ("aura.verified_transition.tool_snapshot.v1"),
            "evidence_snapshot_artifact": ("aura.verified_transition.evidence_snapshot.v1"),
            "world_state_snapshot_artifact": ("aura.verified_transition.world_state_snapshot.v1"),
        }.items()
    }
    host_process = HostResourceObserver(
        source=ObservationSource.HOST,
        scenario_id="verified-transition-test",
    ).process(os.getpid())
    assert host_process is not None
    executable_sha256 = hashlib.sha256(Path(host_process.exe).read_bytes()).hexdigest()
    process = store.put_json(
        {
            "schema": "aura.verified_transition.process_receipt.v1",
            "candidate_visible": False,
            "generation_worker_identity_sha256": execution_manifest[
                "generation_worker_identity_sha256"
            ],
            "execution_manifest_sha256": execution_manifest_sha256,
            "worker_pid": os.getpid(),
            "worker_start_time_unix_ns": int(round(host_process.create_time * 1_000_000_000)),
            "worker_executable_sha256": executable_sha256,
            "executable_component_root_sha256": execution_manifest[
                "generation_worker_identity_sha256"
            ],
            "loaded_component_roots": execution_manifest["component_roots"],
            "observer_contract": "external_process_monitor_required",
            "started_at_unix_ns": generated_at - 50,
            "finished_at_unix_ns": generated_at,
            "exit_code": 0,
        }
    )
    measurements = {
        field: store.put_json(
            {
                "schema": schema,
                "candidate_visible": False,
                "measurement_micros": 250_000,
            }
        )
        for field, schema in {
            "uncertainty_receipt_artifact": ("aura.verified_transition.uncertainty_receipt.v1"),
            "diversity_receipt_artifact": ("aura.verified_transition.diversity_receipt.v1"),
            "resource_receipt_artifact": ("aura.verified_transition.resource_receipt.v1"),
        }.items()
    }
    return {
        "episode_id": episode_id,
        "case_id": "math-case-0001",
        "family": "mathematics-prime-count",
        "depth": 3,
        "sealed_task_commitment_sha256": hashlib.sha256(
            canonical_json_bytes(build_frontier_task_issuer_payload(task))
        ).hexdigest(),
        "model_identity_sha256": execution_manifest["model_identity_sha256"],
        "base_checkpoint_sha256": roots["base_checkpoint"],
        "adapter_stack_sha256": roots["adapter_stack"],
        "tokenizer_sha256": roots["tokenizer"],
        "policy_sha256": roots["policy"],
        "personality_sha256": roots["personality"],
        "runtime_sha256": roots["runtime"],
        "source_closure_sha256": roots["source_closure"],
        "execution_spec_artifact": execution_spec,
        "latent_path_artifact": latent_path,
        **empty_artifacts,
        "rng_root_sha256": _sha("rng"),
        "generation_budget": {
            "max_output_tokens": 1024,
            "max_wall_time_ms": 20_000,
            "max_compute_units": 1_000_000,
        },
        "deadline_unix_ns": VERIFIER_AT + 20_000,
        "process_receipt_artifact": process,
        **measurements,
        "generated_at_unix_ns": generated_at,
        "sealed_at_unix_ns": VERIFIER_AT + 100,
    }


def _generation_attestation(
    *,
    store: TransitionArtifactStore,
    pass_index: int,
    task: Any,
    response: bytes,
    expected: dict[str, str],
    trust_context: TransitionTrustContext,
    context: dict[str, Any],
    policy: Any,
    role_keys: dict[str, Ed25519PrivateKey],
    signed_at_unix_ns: int,
) -> dict[str, Any]:
    model_input = canonical_candidate_model_input(task)
    payload = build_generation_trace_payload(
        store,
        pass_index=pass_index,
        task=task,
        model_input_bytes=model_input,
        response_bytes=response,
        input_token_ids=_byte_encode(model_input),
        output_token_ids=_byte_encode(response),
        emitted_token_pieces=[bytes([value]) for value in response],
        behavior_policy_logprobs=["-1"] * len(response),
        expected_authority=expected,
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        trust_context=trust_context,
        context=context,
        trace_signed_at_unix_ns=signed_at_unix_ns,
    )
    return build_role_attestation(
        policy,
        role=CAMPAIGN_RUNNER,
        payload=payload,
        signed_at_unix=signed_at_unix_ns // 1_000_000_000,
        private_key=role_keys[CAMPAIGN_RUNNER],
    )


def _execution_observer_attestation(
    *,
    store: TransitionArtifactStore,
    generation_attestation: dict[str, Any],
    context: dict[str, Any],
    trust_context: TransitionTrustContext,
    policy: Any,
    role_keys: dict[str, Ed25519PrivateKey],
    observed_at_unix_ns: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generation_trace_payload = generation_attestation["signed_payload"]["payload"]
    real_time_ns = transition_runtime.time.time_ns
    transition_runtime.time.time_ns = lambda: observed_at_unix_ns
    try:
        observation_artifact = capture_execution_process_observation(
            store,
            context=context,
            execution_component_roots=trust_context.execution_component_roots,
        )
    finally:
        transition_runtime.time.time_ns = real_time_ns
    payload = build_execution_observer_payload(
        store,
        generation_trace_payload=generation_trace_payload,
        generation_worker_attestation=generation_attestation,
        context=context,
        process_observation_artifact=observation_artifact,
        observed_at_unix_ns=observed_at_unix_ns,
    )
    return (
        build_role_attestation(
            policy,
            role=CONTAMINATION_AUDITOR,
            payload=payload,
            signed_at_unix=observed_at_unix_ns // 1_000_000_000,
            private_key=role_keys[CONTAMINATION_AUDITOR],
        ),
        observation_artifact,
    )


def _append_attempt_event(
    *,
    ledger: ExternalAttemptLedger,
    policy: Any,
    role_keys: dict[str, Ed25519PrivateKey],
    episode_id: str,
    immutable_context_sha256: str,
    sequence: int,
    previous_event_sha256: str,
    event_time_unix_ns: int,
    event_type: str,
    event_fields: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    payload = build_attempt_ledger_event_payload(
        episode_id=episode_id,
        protocol_sha256=PROTOCOL_SHA256,
        immutable_context_sha256=immutable_context_sha256,
        sequence=sequence,
        previous_event_sha256=previous_event_sha256,
        event_time_unix_ns=event_time_unix_ns,
        event_type=event_type,
        event_fields=event_fields,
    )
    attestation = build_role_attestation(
        policy,
        role=CAMPAIGN_RUNNER,
        payload=payload,
        signed_at_unix=event_time_unix_ns // 1_000_000_000,
        private_key=role_keys[CAMPAIGN_RUNNER],
    )
    ledger.append(policy=policy, attestation=attestation)
    return attestation, hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _build_complete_episode(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
    *,
    episode_id: str = "spark-060-episode-0001",
    pass_0_correct: bool = False,
    pass_1_correct: bool = True,
    shared_policy_case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tmp_path = tmp_path_factory.mktemp(f"verified-transition-{episode_id[-4:]}")
    task = generate_task("mathematics", seed=817_231, difficulty=2)
    store = TransitionArtifactStore(tmp_path / "evidence")
    component_roots = _component_roots(tmp_path / "components")
    component_handles = [(root / "fixture.bin").open("rb") for root in component_roots.values()]
    request.addfinalizer(lambda: [handle.close() for handle in component_handles])
    execution_manifest = build_execution_manifest(
        manifest_id="spark-060-fixture-execution",
        component_roots=component_roots,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        independent_scorer=score_frontier_response_independently,
        created_at_unix_ns=1_800_000_200_000_000_000,
    )
    if shared_policy_case is None:
        policy_document, root, role_keys = _trust_fixture(
            execution_manifest["generation_worker_identity_sha256"]
        )
        policy = validate_campaign_trust_policy(
            policy_document,
            trusted_root_public_key_pem=_public_pem(root),
            expected_campaign_name="spark-060-transition-proof",
            expected_protocol_sha256=PROTOCOL_SHA256,
            now_unix=OBSERVED_AT,
        )
    else:
        policy_document = shared_policy_case["policy_document"]
        root = shared_policy_case["root"]
        role_keys = shared_policy_case["role_keys"]
        policy = shared_policy_case["policy"]
        assert (
            policy.role_pin(CAMPAIGN_RUNNER)["implementation_sha256"]
            == (execution_manifest["generation_worker_identity_sha256"])
        )
    calibration_evidence = _calibration_evidence(
        policy=policy,
        role_keys=role_keys,
        task=task,
    )
    attempt_ledger = ExternalAttemptLedger(
        tmp_path / "attempt-ledger" / "events.jsonl",
        create=True,
    )
    request.addfinalizer(lambda: _clear_append_only_for_test(attempt_ledger.path))
    trust_context = TransitionTrustContext(
        policy_document=policy_document,
        trusted_root_public_key_pem=_public_pem(root),
        expected_campaign_name="spark-060-transition-proof",
        expected_protocol_sha256=PROTOCOL_SHA256,
        expected_policy_sha256=policy.policy_sha256,
        observed_at_unix=OBSERVED_AT,
        execution_manifest=execution_manifest,
        execution_component_roots=component_roots,
        expected_execution_manifest_sha256=hashlib.sha256(
            canonical_json_bytes(execution_manifest)
        ).hexdigest(),
        calibration_evidence=calibration_evidence,
        expected_calibration_evidence_sha256=hashlib.sha256(
            canonical_json_bytes(calibration_evidence)
        ).hexdigest(),
        attempt_ledger_path=attempt_ledger.path,
        expected_attempt_ledger_identity_sha256=(attempt_ledger.identity_sha256),
        attempt_ledger_open_attestation=None,
        attempt_ledger_terminal_attestation=None,
        task_issuer_attestation=None,
    )
    expected = _authority_expected(
        policy,
        task,
        execution_manifest=execution_manifest,
        calibration_evidence=calibration_evidence,
    )
    task_issuer_attestation = build_role_attestation(
        policy,
        role=TASK_ISSUER,
        payload=build_frontier_task_issuer_payload(task),
        signed_at_unix=1_800_000_205,
        private_key=role_keys[TASK_ISSUER],
    )
    trust_context = replace(
        trust_context,
        task_issuer_attestation=task_issuer_attestation,
    )
    pass_0_response = (
        _correct_response(task, prefix="Initial independently checked answer.")
        if pass_0_correct
        else _wrong_response(task)
    )
    pass_1_response = (
        _correct_response(task, prefix="Rechecked independently.")
        if pass_1_correct
        else _wrong_response(task)
    )
    context_0 = _pass_context(
        store,
        episode_id=episode_id,
        generated_at=PASS_0_AT,
        latent_label="latent-shared",
        task=task,
        execution_manifest=execution_manifest,
    )
    context_1 = _pass_context(
        store,
        episode_id=episode_id,
        generated_at=PASS_1_AT,
        latent_label="latent-shared",
        task=task,
        execution_manifest=execution_manifest,
    )
    immutable_context_sha256 = planned_transition_immutable_context_sha256(
        store,
        task=task,
        context=context_0,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        trust_context=trust_context,
    )
    assert immutable_context_sha256 == (
        planned_transition_immutable_context_sha256(
            store,
            task=task,
            context=context_1,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            trust_context=trust_context,
        )
    )
    ledger_open_payload = build_attempt_ledger_open_payload(
        episode_id=context_0["episode_id"],
        protocol_sha256=PROTOCOL_SHA256,
        immutable_context_sha256=immutable_context_sha256,
        attempt_ledger_identity_sha256=attempt_ledger.identity_sha256,
        opened_at_unix_ns=PASS_0_AT - 6_000_000_000,
    )
    ledger_open_attestation = build_role_attestation(
        policy,
        role=TASK_ISSUER,
        payload=ledger_open_payload,
        signed_at_unix=(ledger_open_payload["opened_at_unix_ns"] // 1_000_000_000),
        private_key=role_keys[TASK_ISSUER],
    )
    trust_context = replace(
        trust_context,
        attempt_ledger_open_attestation=ledger_open_attestation,
    )
    runner_event_attestations: list[dict[str, Any]] = []
    previous_event_sha256 = "0" * 64

    def append_event(
        sequence: int,
        event_time_unix_ns: int,
        event_type: str,
        event_fields: dict[str, Any],
    ) -> None:
        nonlocal previous_event_sha256
        attestation, previous_event_sha256 = _append_attempt_event(
            ledger=attempt_ledger,
            policy=policy,
            role_keys=role_keys,
            episode_id=context_0["episode_id"],
            immutable_context_sha256=immutable_context_sha256,
            sequence=sequence,
            previous_event_sha256=previous_event_sha256,
            event_time_unix_ns=event_time_unix_ns,
            event_type=event_type,
            event_fields=event_fields,
        )
        runner_event_attestations.append(attestation)

    append_event(
        0,
        PASS_0_AT - 2_000_000_000,
        "episode_opened",
        {"planned_attempt_count": 2, "launch_counter": 0},
    )
    append_event(
        1,
        PASS_0_AT - 1_000_000_000,
        "attempt_launched",
        {
            "ordinal": 0,
            "pass_index": 0,
            "rng_root_sha256": context_0["rng_root_sha256"],
            "deadline_unix_ns": context_0["deadline_unix_ns"],
            "launch_counter": 1,
        },
    )
    generation_attestation_0 = _generation_attestation(
        store=store,
        pass_index=0,
        task=task,
        response=pass_0_response,
        expected=expected,
        trust_context=trust_context,
        context=context_0,
        policy=policy,
        role_keys=role_keys,
        signed_at_unix_ns=TRACE_0_AT,
    )
    generation_attestation_0_artifact = store.put_json(generation_attestation_0)
    (
        execution_observer_attestation_0,
        execution_process_observation_artifact_0,
    ) = _execution_observer_attestation(
        store=store,
        generation_attestation=generation_attestation_0,
        context=context_0,
        trust_context=trust_context,
        policy=policy,
        role_keys=role_keys,
        observed_at_unix_ns=OBSERVER_0_AT,
    )
    execution_observer_attestation_0_artifact = store.put_json(execution_observer_attestation_0)
    append_event(
        2,
        PASS_0_AT + 3_000_000_000,
        "attempt_finished",
        {
            "ordinal": 0,
            "pass_index": 0,
            "response_sha256": hashlib.sha256(pass_0_response).hexdigest(),
            "generation_attestation_sha256": (generation_attestation_0_artifact["payload_sha256"]),
            "execution_observer_attestation_sha256": (
                execution_observer_attestation_0_artifact["payload_sha256"]
            ),
            "status": "completed",
            "launch_counter": 1,
        },
    )
    append_event(
        3,
        PASS_1_AT - 1_000_000_000,
        "attempt_launched",
        {
            "ordinal": 1,
            "pass_index": 1,
            "rng_root_sha256": context_1["rng_root_sha256"],
            "deadline_unix_ns": context_1["deadline_unix_ns"],
            "launch_counter": 2,
        },
    )
    generation_attestation_1 = _generation_attestation(
        store=store,
        pass_index=1,
        task=task,
        response=pass_1_response,
        expected=expected,
        trust_context=trust_context,
        context=context_1,
        policy=policy,
        role_keys=role_keys,
        signed_at_unix_ns=TRACE_1_AT,
    )
    generation_attestation_1_artifact = store.put_json(generation_attestation_1)
    (
        execution_observer_attestation_1,
        execution_process_observation_artifact_1,
    ) = _execution_observer_attestation(
        store=store,
        generation_attestation=generation_attestation_1,
        context=context_1,
        trust_context=trust_context,
        policy=policy,
        role_keys=role_keys,
        observed_at_unix_ns=OBSERVER_1_AT,
    )
    execution_observer_attestation_1_artifact = store.put_json(execution_observer_attestation_1)
    append_event(
        4,
        PASS_1_AT + 3_000_000_000,
        "attempt_finished",
        {
            "ordinal": 1,
            "pass_index": 1,
            "response_sha256": hashlib.sha256(pass_1_response).hexdigest(),
            "generation_attestation_sha256": (generation_attestation_1_artifact["payload_sha256"]),
            "execution_observer_attestation_sha256": (
                execution_observer_attestation_1_artifact["payload_sha256"]
            ),
            "status": "completed",
            "launch_counter": 2,
        },
    )
    append_event(
        5,
        PASS_1_AT + 4_000_000_000,
        "episode_terminal",
        {
            "attempt_count": 2,
            "final_pass_index": 1,
            "terminal_state": "attempts_completed",
            "launch_counter": 2,
        },
    )
    _ledger_attestations, ledger_content_sha256 = attempt_ledger.snapshot()
    ledger_terminal_payload = build_attempt_ledger_terminal_payload(
        episode_id=context_0["episode_id"],
        protocol_sha256=PROTOCOL_SHA256,
        immutable_context_sha256=immutable_context_sha256,
        attempt_ledger_identity_sha256=attempt_ledger.identity_sha256,
        attempt_ledger_content_sha256=ledger_content_sha256,
        event_chain_head_sha256=previous_event_sha256,
        terminal_at_unix_ns=LEDGER_TERMINAL_AT,
    )
    ledger_terminal_attestation = build_role_attestation(
        policy,
        role=EVIDENCE_VERIFIER,
        payload=ledger_terminal_payload,
        signed_at_unix=LEDGER_TERMINAL_AT // 1_000_000_000,
        private_key=role_keys[EVIDENCE_VERIFIER],
    )
    trust_context = replace(
        trust_context,
        attempt_ledger_terminal_attestation=(ledger_terminal_attestation),
    )
    authority_0, _response_0 = _issue_authority(
        store=store,
        task=task,
        response=pass_0_response,
        expected=expected,
        trust_context=trust_context,
        policy=policy,
        role_keys=role_keys,
        task_issuer_attestation=task_issuer_attestation,
        issued_at=VERIFIER_AT,
    )
    authority_1, _response_1 = _issue_authority(
        store=store,
        task=task,
        response=pass_1_response,
        expected=expected,
        trust_context=trust_context,
        policy=policy,
        role_keys=role_keys,
        task_issuer_attestation=task_issuer_attestation,
        issued_at=VERIFIER_AT,
    )
    model_input = canonical_candidate_model_input(task)
    pass_0 = build_reasoning_pass_receipt(
        store,
        pass_index=0,
        task=task,
        model_input_bytes=model_input,
        response_bytes=pass_0_response,
        input_token_ids=_byte_encode(model_input),
        output_token_ids=_byte_encode(pass_0_response),
        emitted_token_pieces=[bytes([value]) for value in pass_0_response],
        behavior_policy_logprobs=["-1"] * len(pass_0_response),
        verifier_authority=authority_0,
        expected_authority=expected,
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        trust_context=trust_context,
        context=context_0,
        generation_worker_attestation=generation_attestation_0,
        execution_process_observation_artifact=(execution_process_observation_artifact_0),
        execution_observer_attestation=(execution_observer_attestation_0),
        trace_signed_at_unix_ns=TRACE_0_AT,
    )
    pass_1 = build_reasoning_pass_receipt(
        store,
        pass_index=1,
        task=task,
        model_input_bytes=model_input,
        response_bytes=pass_1_response,
        input_token_ids=_byte_encode(model_input),
        output_token_ids=_byte_encode(pass_1_response),
        emitted_token_pieces=[bytes([value]) for value in pass_1_response],
        behavior_policy_logprobs=["-1"] * len(pass_1_response),
        verifier_authority=authority_1,
        expected_authority=expected,
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        trust_context=trust_context,
        context=context_1,
        generation_worker_attestation=generation_attestation_1,
        execution_process_observation_artifact=(execution_process_observation_artifact_1),
        execution_observer_attestation=(execution_observer_attestation_1),
        trace_signed_at_unix_ns=TRACE_1_AT,
    )
    journal = build_transition_attempt_journal(
        pass_0=pass_0,
        pass_1=pass_1,
        protocol_sha256=PROTOCOL_SHA256,
        trust_context=trust_context,
    )
    runner_signed_at_unix_ns = RUNNER_AT * 1_000_000_000
    runner_attestation = build_role_attestation(
        policy,
        role=CAMPAIGN_RUNNER,
        payload=build_campaign_runner_journal_payload(
            journal,
            signed_at_unix_ns=runner_signed_at_unix_ns,
        ),
        signed_at_unix=RUNNER_AT,
        private_key=role_keys[CAMPAIGN_RUNNER],
    )
    verifier_journal_payload = build_evidence_verifier_journal_payload(
        journal,
        runner_attestation,
        signed_at_unix_ns=EVIDENCE_AT * 1_000_000_000,
    )
    evidence_verifier_journal_attestation = build_role_attestation(
        policy,
        role=EVIDENCE_VERIFIER,
        payload=verifier_journal_payload,
        signed_at_unix=EVIDENCE_AT,
        private_key=role_keys[EVIDENCE_VERIFIER],
    )
    episode = build_verified_transition_episode(
        store,
        pass_0=pass_0,
        pass_1=pass_1,
        task=task,
        expected_authority=expected,
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        trust_context=trust_context,
        attempt_journal=journal,
        campaign_runner_attestation=runner_attestation,
        evidence_verifier_journal_attestation=(evidence_verifier_journal_attestation),
        created_at_unix_ns=EVIDENCE_AT * 1_000_000_000,
        sealed_at_unix_ns=EVIDENCE_AT * 1_000_000_000 + 100,
    )
    return {
        "store": store,
        "task": task,
        "policy": policy,
        "policy_document": policy_document,
        "root": root,
        "role_keys": role_keys,
        "trust_context": trust_context,
        "execution_manifest": execution_manifest,
        "component_roots": component_roots,
        "attempt_ledger": attempt_ledger,
        "calibration_evidence": calibration_evidence,
        "expected": expected,
        "pass_0": pass_0,
        "pass_1": pass_1,
        "journal": journal,
        "runner_event_attestations": runner_event_attestations,
        "execution_observer_attestation_0": (execution_observer_attestation_0),
        "execution_observer_attestation_1": (execution_observer_attestation_1),
        "execution_process_observation_artifact_0": (execution_process_observation_artifact_0),
        "execution_process_observation_artifact_1": (execution_process_observation_artifact_1),
        "runner_attestation": runner_attestation,
        "evidence_verifier_journal_attestation": (evidence_verifier_journal_attestation),
        "episode": episode,
    }


@pytest.fixture(scope="module")
def complete_episode(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> dict[str, Any]:
    return _build_complete_episode(tmp_path_factory, request)


@pytest.fixture(scope="module")
def transition_outcome_episodes(
    complete_episode: dict[str, Any],
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> dict[str, dict[str, Any]]:
    return {
        "wrong_to_right": complete_episode,
        "right_to_right": _build_complete_episode(
            tmp_path_factory,
            request,
            episode_id="spark-060-episode-anchor",
            pass_0_correct=True,
            pass_1_correct=True,
            shared_policy_case=complete_episode,
        ),
        "right_to_wrong": _build_complete_episode(
            tmp_path_factory,
            request,
            episode_id="spark-060-episode-regression",
            pass_0_correct=True,
            pass_1_correct=False,
            shared_policy_case=complete_episode,
        ),
    }


def _transition_evidence(case: MappingProxyType | dict[str, Any]) -> VerifiedTransitionEvidence:
    return VerifiedTransitionEvidence(
        store=case["store"],
        episode=case["episode"],
        task=case["task"],
        expected_authority=case["expected"],
        trust_context=case["trust_context"],
    )


def _transition_sample(case: dict[str, Any], prompt_sha256: str) -> SimpleNamespace:
    receipt = case["pass_1"]
    return SimpleNamespace(
        prompt_tokens_sha256=prompt_sha256,
        tokens=tuple(receipt["output_token_ids"]),
        policy_sha256=receipt["policy_sha256"],
        execution_spec_sha256=_sha("recurrent-execution-spec"),
        branch_index=0,
        seed=817_231,
        sampling_config={
            "schema": "aura.recurrent_sampling_config.v1",
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 1024,
        },
        behavior_logprobs=tuple(float(value) for value in receipt["behavior_policy_logprobs"]),
    )


def _canonical_transition_sample(
    case: dict[str, Any],
    prompt_sha256: str,
    *,
    branch_index: int,
) -> recurrent_grpo_runtime.RecurrentPolicySample:
    receipt = case["pass_1"]
    behavior_logprobs = tuple(float(value) for value in receipt["behavior_policy_logprobs"])
    return recurrent_grpo_runtime.RecurrentPolicySample(
        tokens=tuple(receipt["output_token_ids"]),
        branch_index=branch_index,
        behavior_logprobs=behavior_logprobs,
        differentiable_logprobs=behavior_logprobs,
        max_abs_logprob_drift=0.0,
        mean_abs_logprob_drift=0.0,
        max_abs_logprob_drift_token_index=0,
        clipped_token_fraction=0.0,
        old_policy_approx_kl=0.0,
        behavior_admitted=True,
        execution_spec_sha256=_sha("recurrent-execution-spec"),
        policy_sha256=receipt["policy_sha256"],
        prompt_tokens_sha256=prompt_sha256,
        seed=817_231 + branch_index,
        sampling_config=recurrent_grpo_runtime.RecurrentSamplingConfig(
            max_tokens=1024,
        ),
        episode_receipt={
            "decode_termination": "verified_transition_fixture",
            "params_unchanged": True,
            "runtime_integrity": {},
            "nonparametric_memory": {"status": "disabled"},
            "recurrence_adapter": {},
        },
        episode_id=case["episode"]["episode_id"],
        rng_root_sha256=receipt["rng_root_sha256"],
    )


def _group_plan_entry(case: dict[str, Any], sample: SimpleNamespace) -> TransitionGroupPlanEntry:
    return TransitionGroupPlanEntry(
        episode_id=case["episode"]["episode_id"],
        task_id=case["episode"]["task_id"],
        rng_root_sha256=case["pass_1"]["rng_root_sha256"],
        policy_sha256=sample.policy_sha256,
        recurrent_execution_spec_sha256=sample.execution_spec_sha256,
        producing_branch_index=sample.branch_index,
        sample_seed=sample.seed,
        sampling_config_sha256=sampling_config_sha256(sample),
    )


def _signed_group_manifest(
    cases: tuple[dict[str, Any], ...],
    samples: tuple[SimpleNamespace, ...],
    reward_receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reward_config_sha256 = hashlib.sha256(
        canonical_json_bytes(reward_receipt["reward_config"])
    ).hexdigest()
    manifest = build_transition_group_manifest(
        group_id="spark-060-group-0001",
        task_id=cases[0]["episode"]["task_id"],
        entries=tuple(
            _group_plan_entry(case, sample) for case, sample in zip(cases, samples, strict=True)
        ),
        reward_config_sha256=reward_config_sha256,
        planned_at_unix_ns=1_800_000_204_000_000_000,
    )
    attestation = build_role_attestation(
        cases[0]["policy"],
        role=TASK_ISSUER,
        payload=manifest,
        signed_at_unix=1_800_000_204,
        private_key=cases[0]["role_keys"][TASK_ISSUER],
    )
    return manifest, attestation


def _positive_group_material(
    transition_outcome_episodes: dict[str, dict[str, Any]],
    *,
    canonical_samples: bool = False,
) -> dict[str, Any]:
    improved = transition_outcome_episodes["wrong_to_right"]
    anchor = transition_outcome_episodes["right_to_right"]
    evidence = (_transition_evidence(improved), _transition_evidence(anchor))
    batch = build_verified_transition_reward_batch(
        improved["store"],
        evidence,
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        created_at_unix_ns=1_800_000_224_000_000_000,
    )
    prompt_tokens = tuple(improved["pass_1"]["input_token_ids"])
    assert prompt_tokens == tuple(anchor["pass_1"]["input_token_ids"])
    prompt_sha256 = hashlib.sha256(
        json.dumps(list(prompt_tokens), separators=(",", ":")).encode("ascii")
    ).hexdigest()
    samples = (
        (
            _canonical_transition_sample(
                improved,
                prompt_sha256,
                branch_index=0,
            ),
            _canonical_transition_sample(
                anchor,
                prompt_sha256,
                branch_index=1,
            ),
        )
        if canonical_samples
        else (
            _transition_sample(improved, prompt_sha256),
            _transition_sample(anchor, prompt_sha256),
        )
    )
    manifest, manifest_attestation = _signed_group_manifest((improved, anchor), samples, batch)
    return {
        "store": improved["store"],
        "cases": (improved, anchor),
        "evidence": evidence,
        "batch": batch,
        "prompt_tokens": prompt_tokens,
        "samples": samples,
        "manifest": manifest,
        "manifest_attestation": manifest_attestation,
    }


def _positive_admission_material(
    transition_outcome_episodes: dict[str, dict[str, Any]],
    *,
    canonical_samples: bool = False,
) -> dict[str, Any]:
    material = _positive_group_material(
        transition_outcome_episodes,
        canonical_samples=canonical_samples,
    )
    admission = build_verified_transition_group_admission(
        material["store"],
        material["batch"],
        material["evidence"],
        material["samples"],
        material["prompt_tokens"],
        group_manifest=material["manifest"],
        group_manifest_attestation=material["manifest_attestation"],
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        created_at_unix_ns=1_800_000_224_000_000_000,
    )
    return {**material, "admission": admission}


def _open_transition_campaign(
    material: dict[str, Any], root: Path
) -> VerifiedTransitionCampaignLedger:
    issuer = material["cases"][0]
    manifest = build_transition_campaign_manifest(
        campaign_id=f"spark-060-{root.name}",
        groups=(
            campaign_group_from_manifest(0, material["manifest"], material["manifest_attestation"]),
        ),
        trust_policy_sha256=issuer["policy"].policy_sha256,
        planned_at_unix_ns=1_800_000_204_000_000_000,
    )
    attestation = build_role_attestation(
        issuer["policy"],
        role=TASK_ISSUER,
        payload=manifest,
        signed_at_unix=1_800_000_204,
        private_key=issuer["role_keys"][TASK_ISSUER],
    )
    ledger = VerifiedTransitionCampaignLedger.create(
        root,
        campaign_manifest=manifest,
        campaign_manifest_attestation=attestation,
        policy=issuer["policy"],
    )
    ledger.start_group(
        sequence=0,
        group_manifest=material["manifest"],
        group_manifest_attestation=material["manifest_attestation"],
        policy=issuer["policy"],
        started_at_unix_ns=1_800_000_224_000_000_000,
    )
    return ledger


def test_complete_episode_reconstructs_from_bytes_and_dual_signed_authorities(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    validated = validate_verified_transition_episode(
        case["store"],
        case["episode"],
        task=case["task"],
        expected_authority=case["expected"],
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        trust_context=case["trust_context"],
    )
    assert validated == case["episode"]


def test_verified_transition_reward_reconstructs_only_from_sealed_authorities(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    evidence = (
        VerifiedTransitionEvidence(
            store=case["store"],
            episode=case["episode"],
            task=case["task"],
            expected_authority=case["expected"],
            trust_context=case["trust_context"],
        ),
    )
    batch = build_verified_transition_reward_batch(
        case["store"],
        evidence,
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        created_at_unix_ns=1_800_000_224_000_000_000,
    )

    assert batch["wrong_to_right"] == 1
    assert batch["right_to_wrong"] == 0
    assert batch["eir_defined"] is False
    assert batch["eir_micros"] is None
    assert batch["optimizer_admitted"] is False
    assert batch["optimizer_admission_reason"] == ("eir_undefined_no_initially_correct_control")
    transition = batch["transitions"][0]
    assert transition["transition_kind"] == "wrong_to_right"
    expected_resource = len(case["pass_1"]["output_token_ids"]) * 1_000_000 // 1024
    expected_compute_cost = -(expected_resource * 100_000 // 1_000_000)
    assert transition["reward_components_micros"] == {
        "correctness_delta_micros": 1_000_000,
        "information_gain_micros": 100_000,
        "diversity_gain_micros": 100_000,
        "compute_cost_micros": expected_compute_cost,
        "unsupported_confidence_micros": 0,
    }
    assert transition["resource_1_micros"] == expected_resource
    assert transition["reward_micros"] == 1_200_000 + expected_compute_cost
    assert (
        validate_verified_transition_reward_batch(
            case["store"],
            batch,
            evidence,
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
        )
        == batch
    )
    with pytest.raises(
        VerifiedTransitionRewardAdmissionError,
        match="eir_undefined_no_initially_correct_control",
    ):
        require_optimizer_admission(batch)


def test_verified_transition_reward_rejects_rehashed_scalar_reward_forgery(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    evidence = (
        VerifiedTransitionEvidence(
            store=case["store"],
            episode=case["episode"],
            task=case["task"],
            expected_authority=case["expected"],
            trust_context=case["trust_context"],
        ),
    )
    batch = build_verified_transition_reward_batch(
        case["store"],
        evidence,
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        created_at_unix_ns=1_800_000_224_000_000_000,
    )
    forged = copy.deepcopy(batch)
    forged["transitions"][0]["reward_micros"] += 1
    unsigned = dict(forged)
    unsigned.pop("receipt_sha256")
    forged["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()

    with pytest.raises(
        VerifiedTransitionRewardError,
        match="transition_reward_reconstruction_mismatch",
    ):
        validate_verified_transition_reward_batch(
            case["store"],
            forged,
            evidence,
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
        )


def test_transition_reward_config_makes_correctness_lexicographically_dominant() -> None:
    with pytest.raises(ValueError, match="correctness delta must dominate"):
        TransitionRewardConfig(
            correctness_delta_weight_micros=650_000,
        )


def test_rejected_transition_batch_never_reaches_gradient_construction(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    evidence = (
        VerifiedTransitionEvidence(
            store=case["store"],
            episode=case["episode"],
            task=case["task"],
            expected_authority=case["expected"],
            trust_context=case["trust_context"],
        ),
    )
    batch = build_verified_transition_reward_batch(
        case["store"],
        evidence,
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        created_at_unix_ns=1_800_000_224_000_000_000,
    )
    prompt_tokens = tuple(case["pass_1"]["input_token_ids"])
    prompt_sha256 = hashlib.sha256(
        json.dumps(list(prompt_tokens), separators=(",", ":")).encode("ascii")
    ).hexdigest()
    samples = (_transition_sample(case, prompt_sha256),)

    with pytest.raises(VerifiedTransitionRewardAdmissionError):
        build_verified_transition_group_admission(
            case["store"],
            batch,
            evidence,
            samples,
            prompt_tokens,
            group_manifest={},
            group_manifest_attestation={},
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            created_at_unix_ns=1_800_000_224_000_000_000,
        )


def test_transition_reward_aggregate_rejects_regression_before_scalar_score() -> None:
    records = (
        {"transition_kind": "wrong_to_right", "reward_micros": 975_000},
        {"transition_kind": "right_to_right", "reward_micros": -25_000},
        {"transition_kind": "right_to_wrong", "reward_micros": 9_000_000},
    )
    batch = reward_runtime._assemble_reward_batch(
        records=records,
        episode_artifacts=({"row": 0}, {"row": 1}, {"row": 2}),
        task_id="aggregate-test",
        config=TransitionRewardConfig(),
        created_at_unix_ns=1,
    )

    assert batch["right_to_wrong"] == 1
    assert batch["eir_defined"] is True
    assert batch["eir_numerator"] == 1
    assert batch["eir_denominator"] == 2
    assert batch["eir_micros"] == 500_000
    assert batch["optimizer_admitted"] is False
    assert batch["optimizer_admission_reason"] == "right_to_wrong_regression"


def test_transition_reward_aggregate_admits_improvement_with_clean_anchor() -> None:
    prompt_tokens = (7, 8)
    records = (
        {
            "transition_kind": "wrong_to_right",
            "reward_micros": 975_000,
            "pass_1_input_token_ids": list(prompt_tokens),
            "pass_1_output_token_ids": [11],
            "pass_1_policy_sha256": "a" * 64,
            "pass_1_behavior_policy_logprobs": ["-1.25"],
        },
        {
            "transition_kind": "right_to_right",
            "reward_micros": -25_000,
            "pass_1_input_token_ids": list(prompt_tokens),
            "pass_1_output_token_ids": [12],
            "pass_1_policy_sha256": "a" * 64,
            "pass_1_behavior_policy_logprobs": ["-0.75"],
        },
    )
    batch = reward_runtime._assemble_reward_batch(
        records=records,
        episode_artifacts=({"row": 0}, {"row": 1}),
        task_id="aggregate-test",
        config=TransitionRewardConfig(),
        created_at_unix_ns=1,
    )
    prompt_sha256 = hashlib.sha256(
        json.dumps(list(prompt_tokens), separators=(",", ":")).encode("ascii")
    ).hexdigest()
    samples = (
        SimpleNamespace(
            prompt_tokens_sha256=prompt_sha256,
            tokens=(11,),
            policy_sha256="a" * 64,
            behavior_logprobs=(-1.25,),
        ),
        SimpleNamespace(
            prompt_tokens_sha256=prompt_sha256,
            tokens=(12,),
            policy_sha256="a" * 64,
            behavior_logprobs=(-0.75,),
        ),
    )

    assert batch["optimizer_admitted"] is True
    assert batch["eir_defined"] is True
    assert batch["eir_micros"] == 0
    assert reward_runtime.rewards_for_recurrent_samples(batch, samples, prompt_tokens) == (
        0.975,
        -0.025,
    )


def test_signed_transition_group_replays_and_reaches_gradient_only_after_admission(
    transition_outcome_episodes: dict[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    improved = transition_outcome_episodes["wrong_to_right"]
    anchor = transition_outcome_episodes["right_to_right"]
    evidence = (_transition_evidence(improved), _transition_evidence(anchor))
    batch = build_verified_transition_reward_batch(
        improved["store"],
        evidence,
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        created_at_unix_ns=1_800_000_224_000_000_000,
    )
    prompt_tokens = tuple(improved["pass_1"]["input_token_ids"])
    assert prompt_tokens == tuple(anchor["pass_1"]["input_token_ids"])
    prompt_sha256 = hashlib.sha256(
        json.dumps(list(prompt_tokens), separators=(",", ":")).encode("ascii")
    ).hexdigest()
    samples = (
        _transition_sample(improved, prompt_sha256),
        _transition_sample(anchor, prompt_sha256),
    )
    manifest, manifest_attestation = _signed_group_manifest((improved, anchor), samples, batch)
    admission = build_verified_transition_group_admission(
        improved["store"],
        batch,
        evidence,
        samples,
        prompt_tokens,
        group_manifest=manifest,
        group_manifest_attestation=manifest_attestation,
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        created_at_unix_ns=1_800_000_224_000_000_000,
    )
    observed: dict[str, Any] = {}
    sentinel = object()

    def _capture_gradient(
        _model: Any,
        _prompt: Any,
        _samples: Any,
        rewards: Any,
        **_kwargs: Any,
    ) -> object:
        observed["rewards"] = tuple(rewards)
        return sentinel

    monkeypatch.setattr(
        recurrent_grpo_runtime,
        "exact_adjoint_sampled_group_value_and_grad",
        _capture_gradient,
    )
    result = recurrent_grpo_runtime.exact_adjoint_verified_transition_group_value_and_grad(
        None,
        prompt_tokens,
        samples,
        admission,
        batch,
        evidence,
        transition_store=improved["store"],
        group_manifest=manifest,
        group_manifest_attestation=manifest_attestation,
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        spec=None,
    )

    assert result is sentinel
    assert batch["optimizer_admitted"] is True
    assert batch["wrong_to_right"] == 1
    assert batch["right_to_wrong"] == 0
    assert batch["eir_defined"] is True
    assert batch["eir_micros"] == 0
    assert observed["rewards"] == tuple(
        transition["reward_micros"] / 1_000_000 for transition in batch["transitions"]
    )
    assert (
        validate_verified_transition_reward_batch(
            improved["store"],
            batch,
            evidence,
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
        )
        == batch
    )
    assert (
        validate_verified_transition_group_admission(
            improved["store"],
            admission,
            batch,
            evidence,
            samples,
            prompt_tokens,
            group_manifest=manifest,
            group_manifest_attestation=manifest_attestation,
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
        )
        == admission
    )


def test_signed_right_to_wrong_rejects_before_gradient_even_with_improvement(
    transition_outcome_episodes: dict[str, dict[str, Any]],
) -> None:
    improved = transition_outcome_episodes["wrong_to_right"]
    regressed = transition_outcome_episodes["right_to_wrong"]
    evidence = (_transition_evidence(improved), _transition_evidence(regressed))
    batch = build_verified_transition_reward_batch(
        improved["store"],
        evidence,
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        created_at_unix_ns=1_800_000_224_000_000_000,
    )

    assert batch["wrong_to_right"] == 1
    assert batch["right_to_wrong"] == 1
    assert batch["eir_numerator"] == 1
    assert batch["eir_denominator"] == 1
    assert batch["eir_micros"] == 1_000_000
    assert batch["optimizer_admission_reason"] == "right_to_wrong_regression"
    prompt_tokens = tuple(improved["pass_1"]["input_token_ids"])
    prompt_sha256 = hashlib.sha256(
        json.dumps(list(prompt_tokens), separators=(",", ":")).encode("ascii")
    ).hexdigest()
    samples = (
        _transition_sample(improved, prompt_sha256),
        _transition_sample(regressed, prompt_sha256),
    )
    with pytest.raises(
        VerifiedTransitionRewardAdmissionError,
        match="right_to_wrong_regression",
    ):
        build_verified_transition_group_admission(
            improved["store"],
            batch,
            evidence,
            samples,
            prompt_tokens,
            group_manifest={},
            group_manifest_attestation={},
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            created_at_unix_ns=1_800_000_224_000_000_000,
        )


def test_transition_group_rejects_duplicate_signed_episode(
    complete_episode: dict[str, Any],
) -> None:
    evidence = _transition_evidence(complete_episode)
    with pytest.raises(
        VerifiedTransitionRewardError,
        match="transition_reward_duplicate_episode",
    ):
        build_verified_transition_reward_batch(
            complete_episode["store"],
            (evidence, evidence),
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            created_at_unix_ns=1_800_000_224_000_000_000,
        )


@pytest.mark.parametrize(
    "mutation",
    ["entry_order", "branch_index", "sample_seed", "sampling_config"],
)
def test_signed_group_manifest_rejects_membership_and_execution_substitution(
    transition_outcome_episodes: dict[str, dict[str, Any]],
    mutation: str,
) -> None:
    material = _positive_group_material(transition_outcome_episodes)
    manifest = copy.deepcopy(material["manifest"])
    samples = list(material["samples"])
    if mutation == "entry_order":
        manifest["entries"] = list(reversed(manifest["entries"]))
    elif mutation == "branch_index":
        manifest["entries"][0]["producing_branch_index"] = 1
    elif mutation == "sample_seed":
        manifest["entries"][0]["sample_seed"] += 1
    else:
        samples[0] = SimpleNamespace(
            **{
                **vars(samples[0]),
                "sampling_config": {
                    **samples[0].sampling_config,
                    "max_tokens": 512,
                },
            }
        )
    if mutation != "sampling_config":
        unsigned = dict(manifest)
        unsigned.pop("manifest_sha256")
        manifest["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    issuer = material["cases"][0]
    attestation = build_role_attestation(
        issuer["policy"],
        role=TASK_ISSUER,
        payload=manifest,
        signed_at_unix=1_800_000_204,
        private_key=issuer["role_keys"][TASK_ISSUER],
    )

    with pytest.raises(
        VerifiedTransitionGroupError,
        match="group_manifest_membership_mismatch",
    ):
        build_verified_transition_group_admission(
            material["store"],
            material["batch"],
            material["evidence"],
            tuple(samples),
            material["prompt_tokens"],
            group_manifest=manifest,
            group_manifest_attestation=attestation,
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            created_at_unix_ns=1_800_000_224_000_000_000,
        )


def test_group_manifest_signature_must_precede_task_disclosure(
    transition_outcome_episodes: dict[str, dict[str, Any]],
) -> None:
    material = _positive_group_material(transition_outcome_episodes)
    entries = tuple(TransitionGroupPlanEntry(**entry) for entry in material["manifest"]["entries"])
    manifest = build_transition_group_manifest(
        group_id="spark-060-group-late",
        task_id=material["manifest"]["task_id"],
        entries=entries,
        reward_config_sha256=material["manifest"]["reward_config_sha256"],
        planned_at_unix_ns=1_800_000_206_000_000_000,
    )
    issuer = material["cases"][0]
    attestation = build_role_attestation(
        issuer["policy"],
        role=TASK_ISSUER,
        payload=manifest,
        signed_at_unix=1_800_000_206,
        private_key=issuer["role_keys"][TASK_ISSUER],
    )

    with pytest.raises(ValueError, match="campaign_attestation_too_late"):
        build_verified_transition_group_admission(
            material["store"],
            material["batch"],
            material["evidence"],
            material["samples"],
            material["prompt_tokens"],
            group_manifest=manifest,
            group_manifest_attestation=attestation,
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            created_at_unix_ns=1_800_000_224_000_000_000,
        )


def test_campaign_ledger_requires_every_predeclared_group_and_external_close(
    transition_outcome_episodes: dict[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    material = _positive_group_material(transition_outcome_episodes)
    issuer = material["cases"][0]
    second_entries = tuple(
        TransitionGroupPlanEntry(
            **{
                **entry,
                "episode_id": f"{entry['episode_id']}-second",
            }
        )
        for entry in material["manifest"]["entries"]
    )
    second_manifest = build_transition_group_manifest(
        group_id="spark-060-group-0002",
        task_id=material["manifest"]["task_id"],
        entries=second_entries,
        reward_config_sha256=material["manifest"]["reward_config_sha256"],
        planned_at_unix_ns=1_800_000_204_000_000_000,
    )
    second_attestation = build_role_attestation(
        issuer["policy"],
        role=TASK_ISSUER,
        payload=second_manifest,
        signed_at_unix=1_800_000_204,
        private_key=issuer["role_keys"][TASK_ISSUER],
    )
    campaign = build_transition_campaign_manifest(
        campaign_id="spark-060-complete-campaign",
        groups=(
            campaign_group_from_manifest(0, material["manifest"], material["manifest_attestation"]),
            campaign_group_from_manifest(1, second_manifest, second_attestation),
        ),
        trust_policy_sha256=issuer["policy"].policy_sha256,
        planned_at_unix_ns=1_800_000_204_000_000_000,
    )
    campaign_attestation = build_role_attestation(
        issuer["policy"],
        role=TASK_ISSUER,
        payload=campaign,
        signed_at_unix=1_800_000_204,
        private_key=issuer["role_keys"][TASK_ISSUER],
    )
    ledger = VerifiedTransitionCampaignLedger.create(
        tmp_path / "campaign",
        campaign_manifest=campaign,
        campaign_manifest_attestation=campaign_attestation,
        policy=issuer["policy"],
    )

    with pytest.raises(
        VerifiedTransitionCampaignError,
        match="campaign_ledger_record_missing:group-00000000.terminal.json",
    ):
        ledger.start_group(
            sequence=1,
            group_manifest=second_manifest,
            group_manifest_attestation=second_attestation,
            policy=issuer["policy"],
            started_at_unix_ns=1_800_000_225_000_000_000,
        )

    ledger.start_group(
        sequence=0,
        group_manifest=material["manifest"],
        group_manifest_attestation=material["manifest_attestation"],
        policy=issuer["policy"],
        started_at_unix_ns=1_800_000_225_000_000_000,
    )
    with pytest.raises(VerifiedTransitionCampaignError, match="campaign_group_already_started"):
        ledger.start_group(
            sequence=0,
            group_manifest=material["manifest"],
            group_manifest_attestation=material["manifest_attestation"],
            policy=issuer["policy"],
            started_at_unix_ns=1_800_000_225_000_000_000,
        )
    ledger.finish_group(
        sequence=0,
        status="updated",
        reward_receipt_sha256=_sha("campaign-group-reward"),
        group_admission_sha256=_sha("campaign-group-admission"),
        update_receipt_sha256=_sha("campaign-group-update"),
        terminal_reason="optimizer_update_committed",
        finished_at_unix_ns=1_800_000_226_000_000_000,
    )
    with pytest.raises(
        VerifiedTransitionCampaignError,
        match="campaign_ledger_record_missing:group-00000001.started.json",
    ):
        ledger.close_payload(
            completed_at_unix_ns=1_800_000_229_000_000_000,
            policy=issuer["policy"],
        )

    ledger.start_group(
        sequence=1,
        group_manifest=second_manifest,
        group_manifest_attestation=second_attestation,
        policy=issuer["policy"],
        started_at_unix_ns=1_800_000_227_000_000_000,
    )
    ledger.finish_group(
        sequence=1,
        status="rejected",
        reward_receipt_sha256=_sha("campaign-group-rejected-reward"),
        group_admission_sha256=None,
        update_receipt_sha256=None,
        terminal_reason="right_to_wrong_regression",
        finished_at_unix_ns=1_800_000_228_000_000_000,
    )
    close_payload = ledger.close_payload(
        completed_at_unix_ns=1_800_000_229_000_000_000,
        policy=issuer["policy"],
    )
    verifier_attestation = build_role_attestation(
        issuer["policy"],
        role=EVIDENCE_VERIFIER,
        payload=close_payload,
        signed_at_unix=1_800_000_229,
        private_key=issuer["role_keys"][EVIDENCE_VERIFIER],
    )
    receipt = ledger.close(
        close_payload=close_payload,
        evidence_verifier_attestation=verifier_attestation,
        policy=issuer["policy"],
    )

    assert close_payload["group_statuses"] == ["updated", "rejected"]
    assert close_payload["updated_count"] == 1
    assert close_payload["rejected_count"] == 1
    assert ledger.validate_closed(policy=issuer["policy"]) == receipt

    terminal_path = tmp_path / "campaign/group-00000001.terminal.json"
    original_terminal = json.loads(terminal_path.read_text(encoding="ascii"))
    tampered = copy.deepcopy(original_terminal)
    tampered["terminal_reason"] = "forged_result"
    terminal_path.write_bytes(
        json.dumps(tampered, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    with pytest.raises(
        VerifiedTransitionCampaignError, match="campaign_group_terminal_digest_mismatch"
    ):
        ledger.validate_closed(policy=issuer["policy"])

    resealed = _reseal(
        {
            **original_terminal,
            "status": "updated",
            "update_receipt_sha256": None,
        }
    )
    terminal_path.write_bytes(
        json.dumps(resealed, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    with pytest.raises(VerifiedTransitionCampaignError, match="campaign_group_terminal_invalid"):
        ledger.validate_closed(policy=issuer["policy"])


def test_group_admission_rejects_rehashed_manifest_attestation_forgery(
    transition_outcome_episodes: dict[str, dict[str, Any]],
) -> None:
    material = _positive_group_material(transition_outcome_episodes)
    attacked = copy.deepcopy(material["manifest_attestation"])
    attacked["signature_b64"] = base64.b64encode(b"\x00" * 64).decode("ascii")
    with pytest.raises(ValueError, match="campaign_attestation_signature_invalid"):
        build_verified_transition_group_admission(
            material["store"],
            material["batch"],
            material["evidence"],
            material["samples"],
            material["prompt_tokens"],
            group_manifest=material["manifest"],
            group_manifest_attestation=attacked,
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            created_at_unix_ns=1_800_000_224_000_000_000,
        )


def test_verified_transition_update_is_exactly_once_and_durably_receipted(
    transition_outcome_episodes: dict[str, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    material = _positive_admission_material(transition_outcome_episodes)
    policy_before = material["admission"]["policy_sha256"]
    policy_after = _sha("verified-transition-policy-after")

    class Model:
        version = 0

    class Optimizer:
        def __init__(self) -> None:
            self.update_count = 0

        def update(self, model: Model, gradients: Any) -> None:
            assert gradients == {"delta": 1}
            self.update_count += 1
            model.version += 1

    class Objective:
        gradients = {"delta": 1}

        @staticmethod
        def receipt() -> dict[str, Any]:
            return _exact_objective_receipt()

    model = Model()
    optimizer = Optimizer()
    monkeypatch.setattr(
        update_runtime,
        "recurrent_policy_sha256",
        lambda observed, _spec: policy_before if observed.version == 0 else policy_after,
    )
    monkeypatch.setattr(
        update_runtime,
        "exact_adjoint_verified_transition_group_value_and_grad",
        lambda *_args, **_kwargs: Objective(),
    )
    times = iter((1_800_000_225_000_000_000, 1_800_000_226_000_000_000))
    journal = VerifiedTransitionUpdateJournal.open(tmp_path / "updates")
    campaign_ledger = _open_transition_campaign(material, tmp_path / "campaign")
    spec = SimpleNamespace(sha256=material["admission"]["recurrent_execution_spec_sha256"])

    class TransactionCoordinator:
        events: list[str] = []

        def stage_post_update(self, **kwargs):
            assert kwargs["policy_before_sha256"] == policy_before
            assert kwargs["policy_after_sha256"] == policy_after
            self.events.append("stage")

        def record_update_commit(self, _receipt):
            assert self.events == ["stage"]
            self.events.append("update")

        def record_campaign_terminal(self, _receipt):
            assert self.events == ["stage", "update"]
            self.events.append("terminal")

    transaction = TransactionCoordinator()

    receipt = apply_verified_transition_group_update(
        model,
        optimizer,
        material["prompt_tokens"],
        material["samples"],
        material["admission"],
        material["batch"],
        material["evidence"],
        transition_store=material["store"],
        group_manifest=material["manifest"],
        group_manifest_attestation=material["manifest_attestation"],
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        spec=spec,
        journal=journal,
        campaign_ledger=campaign_ledger,
        campaign_sequence=0,
        now_unix_ns=lambda: next(times),
        transaction_coordinator=transaction,
    )

    assert optimizer.update_count == 1
    assert model.version == 1
    assert receipt["optimizer_update_count"] == 1
    assert receipt["policy_before_sha256"] == policy_before
    assert receipt["policy_after_sha256"] == policy_after
    assert transaction.events == ["stage", "update", "terminal"]
    assert validate_verified_transition_update_receipt(journal, receipt) == receipt
    admission_sha256 = material["admission"]["receipt_sha256"]
    assert (tmp_path / "updates" / f"{admission_sha256}.reserved.json").is_file()
    assert (tmp_path / "updates" / f"{admission_sha256}.committed.json").is_file()

    close_payload = campaign_ledger.close_payload(
        completed_at_unix_ns=1_800_000_227_000_000_000,
        policy=material["cases"][0]["policy"],
    )
    close_attestation = build_role_attestation(
        material["cases"][0]["policy"],
        role=EVIDENCE_VERIFIER,
        payload=close_payload,
        signed_at_unix=1_800_000_227,
        private_key=material["cases"][0]["role_keys"][EVIDENCE_VERIFIER],
    )
    campaign_ledger.close(
        close_payload=close_payload,
        evidence_verifier_attestation=close_attestation,
        policy=material["cases"][0]["policy"],
    )
    training_evidence = validate_verified_transition_training_evidence(
        campaign_ledger,
        policy=material["cases"][0]["policy"],
        groups=(
            VerifiedTransitionReplayGroup(
                sequence=0,
                transition_store=material["store"],
                group_admission_receipt=material["admission"],
                reward_receipt=material["batch"],
                transition_evidence=material["evidence"],
                samples=material["samples"],
                prompt_tokens=material["prompt_tokens"],
                group_manifest=material["manifest"],
                group_manifest_attestation=material["manifest_attestation"],
                independent_scorer=score_frontier_response_independently,
                token_encoder=_byte_encode,
                token_decoder=_byte_decode,
                update_journal=journal,
                update_receipt=receipt,
            ),
        ),
    )
    assert training_evidence["source_artifacts_replayed"] is True
    assert training_evidence["legacy_scalar_reward_path_used"] is False
    assert training_evidence["optimizer_update_count"] == 1
    assert training_evidence["final_policy_sha256"] == policy_after

    objective_path = tmp_path / "updates" / f"{admission_sha256}.objective.json"
    objective_bytes = objective_path.read_bytes()
    tampered_objective = json.loads(objective_bytes)
    tampered_objective["objective_receipt"]["policy_loss"] = -0.5
    objective_path.write_bytes(
        json.dumps(
            tampered_objective,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    )
    with pytest.raises(
        VerifiedTransitionUpdateError,
        match="verified_transition_objective_digest_mismatch",
    ):
        validate_verified_transition_update_receipt(journal, receipt)
    objective_path.write_bytes(objective_bytes)

    forged = _reseal(
        {
            **receipt,
            "policy_after_sha256": _sha("forged-policy-after"),
        }
    )
    with pytest.raises(
        VerifiedTransitionUpdateError,
        match="verified_transition_update_reconstruction_mismatch",
    ):
        validate_verified_transition_update_receipt(journal, forged)

    model.version = 0
    replay_campaign = _open_transition_campaign(material, tmp_path / "campaign-replay")
    with pytest.raises(
        VerifiedTransitionUpdateError,
        match="verified_transition_admission_already_reserved",
    ):
        apply_verified_transition_group_update(
            model,
            optimizer,
            material["prompt_tokens"],
            material["samples"],
            material["admission"],
            material["batch"],
            material["evidence"],
            transition_store=material["store"],
            group_manifest=material["manifest"],
            group_manifest_attestation=material["manifest_attestation"],
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            spec=spec,
            journal=journal,
            campaign_ledger=replay_campaign,
            campaign_sequence=0,
            now_unix_ns=lambda: 1_800_000_227_000_000_000,
        )
    assert optimizer.update_count == 1


def test_trajectory_objective_survives_journal_commit_and_rejects_cross_admission(
    tmp_path: Path,
) -> None:
    admission = "d" * 64
    before = "1" * 64
    after = "f" * 64
    journal = VerifiedTransitionUpdateJournal.open(tmp_path / "trajectory-updates")
    journal.reserve(
        admission_sha256=admission,
        policy_before_sha256=before,
        reserved_at_unix_ns=1_800_000_225_000_000_000,
    )
    trajectory_objective = _trajectory_objective_receipt(admission)
    record = journal.record_objective(
        admission_sha256=admission,
        objective_receipt=trajectory_objective,
        trajectory_source_binding=_trajectory_source_binding(trajectory_objective),
    )

    receipt = commit_staged_verified_transition_update(
        journal,
        admission_sha256=admission,
        policy_before_sha256=before,
        policy_after_sha256=after,
        committed_at_unix_ns=1_800_000_226_000_000_000,
    )

    assert record["objective_receipt"]["mode"] == (
        "exact_adjoint_trajectory_composite_single_update"
    )
    assert validate_verified_transition_update_receipt(journal, receipt) == receipt

    other = "0" * 64
    other_journal = VerifiedTransitionUpdateJournal.open(tmp_path / "trajectory-cross-admission")
    with pytest.raises(
        VerifiedTransitionUpdateError,
        match="verified_transition_trajectory_binding_invalid",
    ):
        other_journal.record_objective(
            admission_sha256=other,
            objective_receipt=trajectory_objective,
            trajectory_source_binding=_trajectory_source_binding(trajectory_objective),
        )


def test_admitted_trajectory_group_updates_once_and_replays_from_real_sources(
    transition_outcome_episodes: dict[str, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    material = _positive_admission_material(
        transition_outcome_episodes,
        canonical_samples=True,
    )
    policy_before = material["admission"]["policy_sha256"]
    policy_after = _sha("verified-trajectory-policy-after")
    spec = SimpleNamespace(
        sha256=material["admission"]["recurrent_execution_spec_sha256"],
        recurrent_steps=2,
        branch_roles=("constructive_solution", "critical_verification"),
    )
    trajectory_group_config = recurrent_grpo_runtime.VerifiedTrajectoryGroupConfig(
        trajectory_config=recurrent_grpo_runtime.ExactAdjointTrajectoryConfig(
            probe_steps=(1, 2),
            improvement_weight=0.5,
            improvement_margin=0.2,
            displacement_weight=0.0,
            displacement_floor=0.01,
            oscillation_weight=0.0,
        ),
    )
    recurrent_config = recurrent_grpo_runtime.RecurrentGRPOConfig()
    source_binding = recurrent_grpo_runtime.build_verified_trajectory_group_source_binding(
        material["admission"],
        material["batch"],
        material["samples"],
        material["prompt_tokens"],
        spec=spec,
        trajectory_group_config=trajectory_group_config,
        advantage_clip=recurrent_config.advantage_clip,
    )
    objective_receipt = _trajectory_objective_receipt_from_source(source_binding)

    class Model:
        version = 0

    class Optimizer:
        def __init__(self) -> None:
            self.update_count = 0

        def update(self, model: Model, gradients: Any) -> None:
            assert gradients == {"delta": 1}
            self.update_count += 1
            model.version += 1

    class Objective:
        gradients = {"delta": 1}

        @staticmethod
        def receipt() -> dict[str, Any]:
            return copy.deepcopy(objective_receipt)

    class TransactionCoordinator:
        def __init__(self) -> None:
            self.events: list[str] = []

        def stage_post_update(self, **kwargs: Any) -> None:
            assert kwargs == {
                "policy_before_sha256": policy_before,
                "policy_after_sha256": policy_after,
                "group_admission_sha256": material["admission"]["receipt_sha256"],
            }
            self.events.append("stage")

        def record_update_commit(self, _receipt: Any) -> None:
            assert self.events == ["stage"]
            self.events.append("update")

        def record_campaign_terminal(self, _receipt: Any) -> None:
            assert self.events == ["stage", "update"]
            self.events.append("terminal")

    model = Model()
    optimizer = Optimizer()
    transaction = TransactionCoordinator()
    monkeypatch.setattr(
        update_runtime,
        "recurrent_policy_sha256",
        lambda observed, _spec: policy_before if observed.version == 0 else policy_after,
    )
    monkeypatch.setattr(
        update_runtime,
        "exact_adjoint_verified_transition_group_value_and_grad",
        lambda *_args, **_kwargs: Objective(),
    )
    times = iter((1_800_000_225_000_000_000, 1_800_000_226_000_000_000))
    journal = VerifiedTransitionUpdateJournal.open(tmp_path / "trajectory-updates")
    campaign_ledger = _open_transition_campaign(
        material,
        tmp_path / "trajectory-campaign",
    )

    receipt = apply_verified_transition_group_update(
        model,
        optimizer,
        material["prompt_tokens"],
        material["samples"],
        material["admission"],
        material["batch"],
        material["evidence"],
        transition_store=material["store"],
        group_manifest=material["manifest"],
        group_manifest_attestation=material["manifest_attestation"],
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        spec=spec,
        journal=journal,
        campaign_ledger=campaign_ledger,
        campaign_sequence=0,
        config=recurrent_config,
        trajectory_group_config=trajectory_group_config,
        now_unix_ns=lambda: next(times),
        transaction_coordinator=transaction,
    )

    objective_record = journal.read(
        material["admission"]["receipt_sha256"],
        "objective",
    )
    assert objective_record["trajectory_source_binding"] == source_binding
    assert objective_record["objective_receipt"] == objective_receipt
    assert validate_verified_transition_update_receipt(journal, receipt) == receipt
    assert optimizer.update_count == 1
    assert model.version == 1
    assert transaction.events == ["stage", "update", "terminal"]

    close_payload = campaign_ledger.close_payload(
        completed_at_unix_ns=1_800_000_227_000_000_000,
        policy=material["cases"][0]["policy"],
    )
    close_attestation = build_role_attestation(
        material["cases"][0]["policy"],
        role=EVIDENCE_VERIFIER,
        payload=close_payload,
        signed_at_unix=1_800_000_227,
        private_key=material["cases"][0]["role_keys"][EVIDENCE_VERIFIER],
    )
    campaign_ledger.close(
        close_payload=close_payload,
        evidence_verifier_attestation=close_attestation,
        policy=material["cases"][0]["policy"],
    )
    replay_group = VerifiedTransitionReplayGroup(
        sequence=0,
        transition_store=material["store"],
        group_admission_receipt=material["admission"],
        reward_receipt=material["batch"],
        transition_evidence=material["evidence"],
        samples=material["samples"],
        prompt_tokens=material["prompt_tokens"],
        group_manifest=material["manifest"],
        group_manifest_attestation=material["manifest_attestation"],
        independent_scorer=score_frontier_response_independently,
        token_encoder=_byte_encode,
        token_decoder=_byte_decode,
        update_journal=journal,
        update_receipt=receipt,
    )
    training_evidence = validate_verified_transition_training_evidence(
        campaign_ledger,
        policy=material["cases"][0]["policy"],
        groups=(replay_group,),
        execution_spec=spec,
        trajectory_group_config=trajectory_group_config,
        advantage_clip=recurrent_config.advantage_clip,
    )
    assert training_evidence["source_artifacts_replayed"] is True
    assert training_evidence["optimizer_update_count"] == 1

    wrong_config = recurrent_grpo_runtime.VerifiedTrajectoryGroupConfig(
        trajectory_config=trajectory_group_config.trajectory_config,
        diversity_weight=0.1,
    )
    with pytest.raises(
        VerifiedTransitionTrainingEvidenceError,
        match="verified_training_update_replay_failed",
    ):
        validate_verified_transition_training_evidence(
            campaign_ledger,
            policy=material["cases"][0]["policy"],
            groups=(replay_group,),
            execution_spec=spec,
            trajectory_group_config=wrong_config,
            advantage_clip=recurrent_config.advantage_clip,
        )


def test_staged_commit_rejects_resealed_false_trajectory_before_publication(
    tmp_path: Path,
) -> None:
    admission = "9" * 64
    before = "1" * 64
    after = "8" * 64
    journal = VerifiedTransitionUpdateJournal.open(tmp_path / "trajectory-tamper")
    journal.reserve(
        admission_sha256=admission,
        policy_before_sha256=before,
        reserved_at_unix_ns=1_800_000_225_000_000_000,
    )
    trajectory_objective = _trajectory_objective_receipt(admission)
    journal.record_objective(
        admission_sha256=admission,
        objective_receipt=trajectory_objective,
        trajectory_source_binding=_trajectory_source_binding(trajectory_objective),
    )

    objective_path = journal._path(admission, "objective")
    record = json.loads(objective_path.read_bytes())
    group = record["objective_receipt"]["trajectory_receipt"]
    exact = group["improvement_receipts"][0]["objective_receipt"]
    exact["trajectory_values"]["improvement"] += 1.0
    exact["value"] += 1.0
    group["improvement_receipts"][0]["objective_receipt"] = _seal_float_receipt(exact)
    group["trajectory_objective_value"] += 1.0
    record["objective_receipt"]["trajectory_receipt"] = _seal_float_receipt(group)
    record["objective_receipt"]["trajectory_objective_value"] += 1.0
    record["objective_receipt"]["composite_objective_at_sampling"] += 1.0
    record["objective_receipt"]["composite_gradient_surrogate_value"] += 1.0
    record["objective_receipt_sha256"] = hashlib.sha256(
        json.dumps(
            record["objective_receipt"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    record = _seal_float_receipt(record)
    _clear_append_only_for_test(objective_path)
    objective_path.write_bytes(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    )

    with pytest.raises(
        VerifiedTransitionUpdateError,
        match="verified_transition_trajectory_receipt_invalid",
    ):
        commit_staged_verified_transition_update(
            journal,
            admission_sha256=admission,
            policy_before_sha256=before,
            policy_after_sha256=after,
            committed_at_unix_ns=1_800_000_226_000_000_000,
        )
    assert not journal.exists(admission, "committed")


def test_training_evidence_rejects_admission_for_a_different_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission = {
        "receipt_sha256": _sha("admission-policy-a"),
        "policy_sha256": _sha("policy-a"),
    }
    update = {
        "receipt_sha256": _sha("update-policy-b"),
        "group_admission_sha256": admission["receipt_sha256"],
        "objective_receipt_sha256": _sha("objective-policy-b"),
        "policy_before_sha256": _sha("policy-b"),
        "policy_after_sha256": _sha("policy-c"),
    }

    class Ledger:
        @staticmethod
        def validate_closed(*, policy: Any) -> dict[str, Any]:
            del policy
            return {
                "receipt_sha256": _sha("closed-campaign"),
                "close_payload": {
                    "campaign_manifest_sha256": _sha("campaign-manifest"),
                    "group_statuses": ["updated"],
                    "updated_count": 1,
                },
            }

        @staticmethod
        def group_records(*, sequence: int, policy: Any) -> tuple[dict, dict]:
            del policy
            assert sequence == 0
            return {"group_manifest": {}}, {
                "status": "updated",
                "group_admission_sha256": admission["receipt_sha256"],
                "update_receipt_sha256": update["receipt_sha256"],
            }

    monkeypatch.setattr(
        training_evidence_runtime,
        "validate_verified_transition_group_admission",
        lambda *_args, **_kwargs: admission,
    )
    monkeypatch.setattr(
        training_evidence_runtime,
        "validate_verified_transition_update_receipt",
        lambda *_args, **_kwargs: update,
    )

    class Journal:
        @staticmethod
        def read(_admission_sha256: str, role: str) -> dict[str, Any]:
            assert role == "objective"
            return {"objective_receipt": _exact_objective_receipt()}

    group = VerifiedTransitionReplayGroup(
        sequence=0,
        transition_store=object(),
        group_admission_receipt={},
        reward_receipt={},
        transition_evidence=(),
        samples=(),
        prompt_tokens=(),
        group_manifest={},
        group_manifest_attestation={},
        independent_scorer=lambda *_args: {},
        token_encoder=lambda _value: (),
        token_decoder=lambda _value: b"",
        update_journal=Journal(),
        update_receipt={},
    )

    with pytest.raises(
        training_evidence_runtime.VerifiedTransitionTrainingEvidenceError,
        match="verified_training_group_source_binding_mismatch",
    ):
        validate_verified_transition_training_evidence(
            Ledger(),
            policy=object(),
            groups=(group,),
        )


def test_group_records_returns_validated_snapshot_without_post_validation_reread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = {"receipt_sha256": _sha("snapshot-start")}
    terminal = {"receipt_sha256": _sha("snapshot-terminal"), "status": "updated"}
    close = {
        "close_payload": {"group_count": 1},
    }
    ledger = object.__new__(VerifiedTransitionCampaignLedger)
    monkeypatch.setattr(
        VerifiedTransitionCampaignLedger,
        "_validate_closed_snapshot",
        lambda _self, *, policy: (close, ((start, terminal),)),
    )
    monkeypatch.setattr(
        VerifiedTransitionCampaignLedger,
        "_read",
        lambda *_args, **_kwargs: pytest.fail("post-validation record reread"),
    )

    assert ledger.group_records(sequence=0, policy=object()) == (start, terminal)


def test_policy_drift_after_gradient_blocks_optimizer_and_burns_admission(
    transition_outcome_episodes: dict[str, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    material = _positive_admission_material(transition_outcome_episodes)
    policy_before = material["admission"]["policy_sha256"]
    policy_drifted = _sha("verified-transition-policy-drifted")

    class Model:
        version = 0

    class Optimizer:
        update_count = 0

        def update(self, _model: Model, _gradients: Any) -> None:
            self.update_count += 1

    class Objective:
        gradients = {"delta": 1}

        @staticmethod
        def receipt() -> dict[str, Any]:
            return _exact_objective_receipt()

    model = Model()
    optimizer = Optimizer()
    monkeypatch.setattr(
        update_runtime,
        "recurrent_policy_sha256",
        lambda observed, _spec: policy_before if observed.version == 0 else policy_drifted,
    )

    def _drift_then_return(*_args: Any, **_kwargs: Any) -> Objective:
        model.version = 1
        return Objective()

    monkeypatch.setattr(
        update_runtime,
        "exact_adjoint_verified_transition_group_value_and_grad",
        _drift_then_return,
    )
    journal = VerifiedTransitionUpdateJournal.open(tmp_path / "updates")
    campaign_ledger = _open_transition_campaign(material, tmp_path / "campaign")
    spec = SimpleNamespace(sha256=material["admission"]["recurrent_execution_spec_sha256"])
    with pytest.raises(
        VerifiedTransitionUpdateError,
        match="verified_transition_policy_changed_before_update",
    ):
        apply_verified_transition_group_update(
            model,
            optimizer,
            material["prompt_tokens"],
            material["samples"],
            material["admission"],
            material["batch"],
            material["evidence"],
            transition_store=material["store"],
            group_manifest=material["manifest"],
            group_manifest_attestation=material["manifest_attestation"],
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            spec=spec,
            journal=journal,
            campaign_ledger=campaign_ledger,
            campaign_sequence=0,
            now_unix_ns=lambda: 1_800_000_225_000_000_000,
        )
    assert optimizer.update_count == 0
    admission_sha256 = material["admission"]["receipt_sha256"]
    assert (tmp_path / "updates" / f"{admission_sha256}.reserved.json").is_file()
    assert not (tmp_path / "updates" / f"{admission_sha256}.committed.json").exists()


def test_durable_commit_recovers_update_receipt_and_campaign_terminal(
    transition_outcome_episodes: dict[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    material = _positive_admission_material(transition_outcome_episodes)
    admission_sha256 = material["admission"]["receipt_sha256"]
    policy_before = material["admission"]["policy_sha256"]
    policy_after = _sha("recovered-policy-after")
    journal = VerifiedTransitionUpdateJournal.open(tmp_path / "recovery-updates")
    reservation = journal.reserve(
        admission_sha256=admission_sha256,
        policy_before_sha256=policy_before,
        reserved_at_unix_ns=1_800_000_225_000_000_000,
    )
    objective = journal.record_objective(
        admission_sha256=admission_sha256,
        objective_receipt=_exact_objective_receipt(),
    )
    journal.commit(
        admission_sha256=admission_sha256,
        reservation_sha256=reservation["receipt_sha256"],
        policy_before_sha256=policy_before,
        policy_after_sha256=policy_after,
        objective_record_sha256=objective["receipt_sha256"],
        objective_receipt_sha256=objective["objective_receipt_sha256"],
        committed_at_unix_ns=1_800_000_226_000_000_000,
    )
    campaign = _open_transition_campaign(material, tmp_path / "recovery-campaign")

    recovered = recover_committed_campaign_group(
        journal,
        campaign,
        campaign_sequence=0,
        admission_sha256=admission_sha256,
        reward_receipt_sha256=material["batch"]["receipt_sha256"],
    )

    assert recovered["policy_before_sha256"] == policy_before
    assert recovered["policy_after_sha256"] == policy_after
    terminal = json.loads(
        (tmp_path / "recovery-campaign/group-00000000.terminal.json").read_text(encoding="ascii")
    )
    assert terminal["status"] == "updated"
    assert terminal["update_receipt_sha256"] == recovered["receipt_sha256"]
    assert terminal["terminal_reason"] == "optimizer_update_recovered_from_commit"


def test_staged_post_update_policy_can_complete_missing_journal_commit(
    transition_outcome_episodes: dict[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    material = _positive_admission_material(transition_outcome_episodes)
    admission = material["admission"]["receipt_sha256"]
    before = material["admission"]["policy_sha256"]
    after = _sha("staged-policy-after")
    journal = VerifiedTransitionUpdateJournal.open(tmp_path / "staged-updates")
    journal.reserve(
        admission_sha256=admission,
        policy_before_sha256=before,
        reserved_at_unix_ns=1_800_000_225_000_000_000,
    )
    journal.record_objective(
        admission_sha256=admission,
        objective_receipt=_exact_objective_receipt(),
    )

    receipt = commit_staged_verified_transition_update(
        journal,
        admission_sha256=admission,
        policy_before_sha256=before,
        policy_after_sha256=after,
        committed_at_unix_ns=1_800_000_226_000_000_000,
    )

    assert receipt["policy_before_sha256"] == before
    assert receipt["policy_after_sha256"] == after
    assert validate_verified_transition_update_receipt(journal, receipt) == receipt
    assert (
        commit_staged_verified_transition_update(
            journal,
            admission_sha256=admission,
            policy_before_sha256=before,
            policy_after_sha256=after,
        )
        == receipt
    )


@pytest.mark.parametrize(
    ("policy_changed", "classification", "requires_recovery"),
    [
        (False, "reserved_no_policy_change", False),
        (True, "policy_changed_without_commit", True),
    ],
)
def test_interrupted_update_reconciliation_burns_admission_without_guessing(
    transition_outcome_episodes: dict[str, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy_changed: bool,
    classification: str,
    requires_recovery: bool,
) -> None:
    material = _positive_admission_material(transition_outcome_episodes)
    admission_sha256 = material["admission"]["receipt_sha256"]
    policy_before = material["admission"]["policy_sha256"]
    policy_observed = _sha("interrupted-policy-changed") if policy_changed else policy_before
    journal = VerifiedTransitionUpdateJournal.open(tmp_path / f"updates-{int(policy_changed)}")
    journal.reserve(
        admission_sha256=admission_sha256,
        policy_before_sha256=policy_before,
        reserved_at_unix_ns=1_800_000_225_000_000_000,
    )
    monkeypatch.setattr(
        update_runtime,
        "recurrent_policy_sha256",
        lambda _model, _spec: policy_observed,
    )

    receipt = reconcile_interrupted_verified_transition_update(
        object(),
        SimpleNamespace(sha256=material["admission"]["recurrent_execution_spec_sha256"]),
        journal,
        admission_sha256,
        now_unix_ns=lambda: 1_800_000_226_000_000_000,
    )

    assert receipt["classification"] == classification
    assert receipt["admission_reusable"] is False
    assert receipt["requires_fresh_admission"] is True
    assert receipt["requires_checkpoint_recovery"] is requires_recovery
    assert validate_verified_transition_reconciliation_receipt(journal, receipt) == receipt
    with pytest.raises(
        VerifiedTransitionUpdateError,
        match="verified_transition_admission_already_reconciled",
    ):
        reconcile_interrupted_verified_transition_update(
            object(),
            SimpleNamespace(sha256=material["admission"]["recurrent_execution_spec_sha256"]),
            journal,
            admission_sha256,
            now_unix_ns=lambda: 1_800_000_227_000_000_000,
        )


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (b'{"x":1,"x":2}', "document_duplicate_key"),
        (b'{"x":1} ', "document_noncanonical"),
        (b'{"x":1.5}', "document_floating_point_forbidden"),
        (b'{"x":NaN}', "document_non_finite_number"),
        (b'{"x":9223372036854775808}', "document_integer_out_of_bounds"),
        ('{"x":"caf\u00e9"}'.encode(), "document_not_ascii"),
    ],
)
def test_strict_parser_rejects_ambiguous_or_noncanonical_json(
    payload: bytes,
    code: str,
) -> None:
    with pytest.raises(VerifiedTransitionError, match=code):
        strict_canonical_json_loads(payload)


def test_strict_parser_rejects_excessive_depth_and_nodes() -> None:
    too_deep: Any = 0
    for _ in range(20):
        too_deep = [too_deep]
    with pytest.raises(VerifiedTransitionError, match="json_depth_limit_exceeded"):
        canonical_json_bytes(too_deep)
    with pytest.raises(VerifiedTransitionError, match="json_node_limit_exceeded"):
        canonical_json_bytes([0] * 17_000)


def test_store_rejects_symlink_and_hardlink_artifacts(tmp_path: Path) -> None:
    target = tmp_path / "real"
    target.mkdir(mode=0o700)
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(
        VerifiedTransitionError,
        match="artifact_store_symlink_path_rejected",
    ):
        TransitionArtifactStore(link)

    store = TransitionArtifactStore(tmp_path / "store")
    binding = store.put_bytes(b"payload", media_type="application/octet-stream")
    blob = store.blob_root / binding["payload_sha256"]
    os.link(blob, tmp_path / "alias")
    with pytest.raises(
        VerifiedTransitionError,
        match="artifact_file_identity_invalid",
    ):
        store.read_bytes(binding)


def test_store_rejects_preexisting_nonprivate_directory(tmp_path: Path) -> None:
    root = tmp_path / "nonprivate"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    with pytest.raises(
        VerifiedTransitionError,
        match="artifact_store_directory_not_private",
    ):
        TransitionArtifactStore(root)


def test_store_rejects_root_directory_replacement(tmp_path: Path) -> None:
    store = TransitionArtifactStore(tmp_path / "store")
    binding = store.put_bytes(b"payload", media_type="application/octet-stream")
    displaced = tmp_path / "displaced-store"
    store.root.rename(displaced)
    store.root.mkdir(mode=0o700)
    (store.root / "blobs").mkdir(mode=0o700)

    with pytest.raises(
        VerifiedTransitionError,
        match="artifact_store_directory_replaced",
    ):
        store.read_bytes(binding)


def test_store_rejects_digest_length_and_symlink_rebinding(tmp_path: Path) -> None:
    store = TransitionArtifactStore(tmp_path / "store")
    binding = store.put_bytes(b"payload", media_type="application/octet-stream")
    bad_length = {**binding, "byte_length": binding["byte_length"] + 1}
    with pytest.raises(
        VerifiedTransitionError,
        match="artifact_file_identity_invalid",
    ):
        store.read_bytes(bad_length)

    blob = store.blob_root / binding["payload_sha256"]
    blob.unlink()
    blob.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(VerifiedTransitionError, match="artifact_unreadable"):
        store.read_bytes(binding)


def test_store_detects_post_read_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = TransitionArtifactStore(tmp_path / "store")
    binding = store.put_bytes(b"payload", media_type="application/octet-stream")
    blob = store.blob_root / binding["payload_sha256"]
    original_stat = os.stat
    replaced = False

    def replacing_stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        nonlocal replaced
        if (
            path == binding["payload_sha256"]
            and kwargs.get("dir_fd") == store._blob_root_fd
            and not replaced
        ):
            replaced = True
            replacement = store.blob_root / "replacement"
            replacement.write_bytes(b"payload")
            replacement.chmod(0o600)
            os.replace(replacement, blob)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(transition_runtime.os, "stat", replacing_stat)
    with pytest.raises(
        VerifiedTransitionError,
        match="artifact_replaced_during_read",
    ):
        store.read_bytes(binding)


def test_authority_rejects_wrong_external_root_and_witness_signature(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    wrong_context = TransitionTrustContext(
        policy_document=case["policy_document"],
        trusted_root_public_key_pem=_public_pem(Ed25519PrivateKey.generate()),
        expected_campaign_name="spark-060-transition-proof",
        expected_protocol_sha256=PROTOCOL_SHA256,
        expected_policy_sha256=case["policy"].policy_sha256,
        observed_at_unix=OBSERVED_AT,
        execution_manifest=case["execution_manifest"],
        execution_component_roots=case["component_roots"],
        expected_execution_manifest_sha256=case["trust_context"].expected_execution_manifest_sha256,
        calibration_evidence=case["calibration_evidence"],
        expected_calibration_evidence_sha256=case[
            "trust_context"
        ].expected_calibration_evidence_sha256,
        attempt_ledger_path=case["trust_context"].attempt_ledger_path,
        expected_attempt_ledger_identity_sha256=case[
            "trust_context"
        ].expected_attempt_ledger_identity_sha256,
        attempt_ledger_open_attestation=case["trust_context"].attempt_ledger_open_attestation,
        attempt_ledger_terminal_attestation=case[
            "trust_context"
        ].attempt_ledger_terminal_attestation,
        task_issuer_attestation=case["trust_context"].task_issuer_attestation,
    )
    authority = case["store"].read_json(
        case["pass_0"]["verifier_authority_artifact"],
        role="authority",
    )
    with pytest.raises(ValueError, match="campaign_trust_root_key_mismatch"):
        validate_frontier_verifier_authority(
            case["store"],
            authority,
            task=case["task"],
            response_artifact=case["pass_0"]["response_artifact"],
            expected_authority=case["expected"],
            independent_scorer=score_frontier_response_independently,
            trust_context=wrong_context,
        )

    witness = case["store"].read_json(
        authority["independent_witness_artifact"],
        role="witness",
    )
    attacked = copy.deepcopy(witness)
    attacked["evidence_verifier_attestation"]["signature_b64"] = base64.b64encode(
        b"x" * 64
    ).decode()
    attacked = _reseal(attacked)
    authority["independent_witness_artifact"] = case["store"].put_json(attacked)
    authority = _reseal(authority)
    with pytest.raises(ValueError, match="campaign_attestation_signature_invalid"):
        validate_frontier_verifier_authority(
            case["store"],
            authority,
            task=case["task"],
            response_artifact=case["pass_0"]["response_artifact"],
            expected_authority=case["expected"],
            independent_scorer=score_frontier_response_independently,
            trust_context=case["trust_context"],
        )

    authority = case["store"].read_json(
        case["pass_0"]["verifier_authority_artifact"],
        role="authority",
    )
    issuer = case["store"].read_json(
        authority["task_issuer_attestation_artifact"],
        role="issuer",
    )
    issuer["signature_b64"] = base64.b64encode(b"x" * 64).decode()
    authority["task_issuer_attestation_artifact"] = case["store"].put_json(issuer)
    authority = _reseal(authority)
    with pytest.raises(ValueError, match="campaign_attestation_signature_invalid"):
        validate_frontier_verifier_authority(
            case["store"],
            authority,
            task=case["task"],
            response_artifact=case["pass_0"]["response_artifact"],
            expected_authority=case["expected"],
            independent_scorer=score_frontier_response_independently,
            trust_context=case["trust_context"],
        )


def test_authority_rejects_independent_scorer_disagreement(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    authority = case["store"].read_json(
        case["pass_0"]["verifier_authority_artifact"],
        role="authority",
    )

    def disagree(_task: Any, _response: Any) -> dict[str, Any]:
        return {
            "parsed": False,
            "correct": False,
            "reason": "forced_disagreement",
            "normalized_answer_sha256": None,
        }

    with pytest.raises(
        VerifiedTransitionError,
        match=(
            "transition_verifier_identity_not_policy_pinned"
            "|verifier_authority_source_closure_mismatch"
        ),
    ):
        validate_frontier_verifier_authority(
            case["store"],
            authority,
            task=case["task"],
            response_artifact=case["pass_0"]["response_artifact"],
            expected_authority=case["expected"],
            independent_scorer=disagree,
            trust_context=case["trust_context"],
        )


@pytest.mark.parametrize("surface", ["parser", "scorer_registry"])
def test_authority_detects_loaded_runtime_substitution(
    complete_episode: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    case = complete_episode
    authority = case["store"].read_json(
        case["pass_0"]["verifier_authority_artifact"],
        role="authority",
    )
    if surface == "parser":
        monkeypatch.setattr(
            frontier_tasks_runtime,
            "parse_final_answer",
            lambda _response: {"count": 0, "witness": []},
        )
    else:
        original = dict(frontier_tasks_runtime._SCORERS)
        original["score_mathematics"] = lambda _answer, _expected: True
        monkeypatch.setattr(
            frontier_tasks_runtime,
            "_SCORERS",
            MappingProxyType(original),
        )
    with pytest.raises(
        VerifiedTransitionError,
        match=(
            "transition_verifier_identity_not_policy_pinned"
            "|verifier_authority_source_closure_mismatch"
        ),
    ):
        validate_frontier_verifier_authority(
            case["store"],
            authority,
            task=case["task"],
            response_artifact=case["pass_0"]["response_artifact"],
            expected_authority=case["expected"],
            independent_scorer=score_frontier_response_independently,
            trust_context=case["trust_context"],
        )


def test_authority_rejects_task_answer_and_outcome_rebinding(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    authority = case["store"].read_json(
        case["pass_0"]["verifier_authority_artifact"],
        role="authority",
    )
    other_task = generate_task("mathematics", seed=817_232, difficulty=2)
    with pytest.raises(
        VerifiedTransitionError,
        match=("(task_issuer_commitment|verifier_authority_task_identity)_mismatch"),
    ):
        validate_frontier_verifier_authority(
            case["store"],
            authority,
            task=other_task,
            response_artifact=case["pass_0"]["response_artifact"],
            expected_authority=case["expected"],
            independent_scorer=score_frontier_response_independently,
            trust_context=case["trust_context"],
        )

    with pytest.raises(
        VerifiedTransitionError,
        match="frontier_task_exact_type_required",
    ):
        build_frontier_task_issuer_payload(object())

    forged = copy.deepcopy(authority)
    forged["outcome"] = "pass"
    forged["verifier_output"]["correct"] = True
    forged = _reseal(forged)
    with pytest.raises(
        VerifiedTransitionError,
        match="verifier_authority_output_mismatch",
    ):
        validate_frontier_verifier_authority(
            case["store"],
            forged,
            task=case["task"],
            response_artifact=case["pass_0"]["response_artifact"],
            expected_authority=case["expected"],
            independent_scorer=score_frontier_response_independently,
            trust_context=case["trust_context"],
        )


@pytest.mark.parametrize(
    "field",
    [
        "task_id",
        "policy_sha256",
        "rng_root_sha256",
        "execution_spec_artifact",
        "latent_path_artifact",
        "tool_snapshot_artifact",
        "generation_budget",
    ],
)
def test_episode_rejects_immutable_context_drift(
    complete_episode: dict[str, Any],
    field: str,
) -> None:
    case = complete_episode
    attacked = copy.deepcopy(case["pass_1"])
    if field.endswith("_artifact"):
        attacked[field] = _artifact(case["store"], f"attacked-{field}")
    elif field == "generation_budget":
        attacked[field]["max_compute_units"] += 1
    elif field == "task_id":
        attacked[field] = "attacked-task"
    else:
        attacked[field] = _sha(f"attacked-{field}")
    attacked = _reseal(attacked)
    with pytest.raises(
        VerifiedTransitionError,
        match="(drift|mismatch|isolation_failed|schema_invalid)",
    ):
        build_verified_transition_episode(
            case["store"],
            pass_0=case["pass_0"],
            pass_1=attacked,
            task=case["task"],
            expected_authority=case["expected"],
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            trust_context=case["trust_context"],
            attempt_journal=case["journal"],
            campaign_runner_attestation=case["runner_attestation"],
            evidence_verifier_journal_attestation=case["evidence_verifier_journal_attestation"],
            created_at_unix_ns=EVIDENCE_AT * 1_000_000_000,
            sealed_at_unix_ns=EVIDENCE_AT * 1_000_000_000 + 100,
        )


def test_pass_rejects_token_text_logprob_and_outcome_forgery(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    for mutation in ("token", "logprob", "outcome"):
        attacked = copy.deepcopy(case["pass_0"])
        if mutation == "token":
            attacked["output_token_ids"][0] += 1
        elif mutation == "logprob":
            attacked["behavior_policy_logprobs"].pop()
        else:
            attacked["final_success"] = True
        attacked = _reseal(attacked)
        expected = {
            "token": "response_token_mismatch",
            "logprob": "behavior_policy_logprobs_invalid",
            "outcome": "reasoning_pass_schema_invalid",
        }[mutation]
        with pytest.raises(VerifiedTransitionError, match=expected):
            validate_reasoning_pass_receipt(
                case["store"],
                attacked,
                task=case["task"],
                expected_authority=case["expected"],
                independent_scorer=score_frontier_response_independently,
                token_encoder=_byte_encode,
                token_decoder=_byte_decode,
                trust_context=case["trust_context"],
            )


def test_episode_rejects_swapped_duplicate_or_reused_passes(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    for first, second, code in (
        (case["pass_1"], case["pass_0"], "transition_pass_order_invalid"),
        (case["pass_0"], case["pass_0"], "transition_pass_order_invalid"),
    ):
        with pytest.raises(VerifiedTransitionError, match=code):
            build_verified_transition_episode(
                case["store"],
                pass_0=first,
                pass_1=second,
                task=case["task"],
                expected_authority=case["expected"],
                independent_scorer=score_frontier_response_independently,
                token_encoder=_byte_encode,
                token_decoder=_byte_decode,
                trust_context=case["trust_context"],
                attempt_journal=case["journal"],
                campaign_runner_attestation=case["runner_attestation"],
                evidence_verifier_journal_attestation=case["evidence_verifier_journal_attestation"],
            )

    reused = copy.deepcopy(case["pass_1"])
    reused["response_artifact"] = case["pass_0"]["response_artifact"]
    reused["output_token_ids"] = case["pass_0"]["output_token_ids"]
    reused["input_token_ids"] = case["pass_0"]["input_token_ids"]
    reused["behavior_policy_logprobs"] = case["pass_0"]["behavior_policy_logprobs"]
    reused["emitted_token_pieces_artifact"] = case["pass_0"]["emitted_token_pieces_artifact"]
    reused["verifier_authority_artifact"] = case["pass_0"]["verifier_authority_artifact"]
    reused["generated_at_unix_ns"] = case["pass_0"]["generated_at_unix_ns"]
    reused["sealed_at_unix_ns"] = case["pass_0"]["sealed_at_unix_ns"]
    reused = _reseal(reused)
    with pytest.raises(
        VerifiedTransitionError,
        match=(
            "transition_response_reused|process_receipt_generation_time_mismatch|generation_trace"
        ),
    ):
        build_verified_transition_episode(
            case["store"],
            pass_0=case["pass_0"],
            pass_1=reused,
            task=case["task"],
            expected_authority=case["expected"],
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            trust_context=case["trust_context"],
            attempt_journal=case["journal"],
            campaign_runner_attestation=case["runner_attestation"],
            evidence_verifier_journal_attestation=case["evidence_verifier_journal_attestation"],
        )


def test_episode_rejects_hidden_third_attempt_and_unrelated_runner_signature(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    attacked = copy.deepcopy(case["journal"])
    attacked["attempt_count"] = 3
    attacked["attempts"].append(
        {
            "ordinal": 2,
            "pass_index": 1,
            "pass_receipt_sha256": _sha("hidden-pass"),
            "response_sha256": _sha("hidden-response"),
        }
    )
    attacked = _reseal(attacked)
    with pytest.raises(
        VerifiedTransitionError,
        match="transition_attempt_journal_mismatch",
    ):
        build_verified_transition_episode(
            case["store"],
            pass_0=case["pass_0"],
            pass_1=case["pass_1"],
            task=case["task"],
            expected_authority=case["expected"],
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            trust_context=case["trust_context"],
            attempt_journal=attacked,
            campaign_runner_attestation=case["runner_attestation"],
            evidence_verifier_journal_attestation=case["evidence_verifier_journal_attestation"],
        )


def _pass_trace_arguments(case: dict[str, Any]) -> dict[str, Any]:
    receipt = case["pass_0"]
    store = case["store"]
    response = store.read_bytes(
        receipt["response_artifact"],
        expected_media_type="text/plain;charset=utf-8",
    )
    model_input = store.read_bytes(
        receipt["model_input_artifact"],
        expected_media_type="application/octet-stream",
    )
    return {
        "store": store,
        "pass_index": 0,
        "task": case["task"],
        "model_input_bytes": model_input,
        "response_bytes": response,
        "input_token_ids": receipt["input_token_ids"],
        "output_token_ids": receipt["output_token_ids"],
        "emitted_token_pieces": [bytes([value]) for value in response],
        "behavior_policy_logprobs": receipt["behavior_policy_logprobs"],
        "expected_authority": case["expected"],
        "independent_scorer": score_frontier_response_independently,
        "token_encoder": _byte_encode,
        "token_decoder": _byte_decode,
        "trust_context": case["trust_context"],
        "context": {field: receipt[field] for field in transition_runtime._PASS_CONTEXT_KEYS},
        "trace_signed_at_unix_ns": TRACE_0_AT,
    }


def test_candidate_input_and_side_channels_reject_answer_injection(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    arguments = _pass_trace_arguments(case)
    expected = json.dumps(
        case["task"].reveal_for_verifier()["expected"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with pytest.raises(
        VerifiedTransitionError,
        match="candidate_model_input_not_canonical",
    ):
        build_generation_trace_payload(
            **{
                **arguments,
                "model_input_bytes": arguments["model_input_bytes"] + b"\nEXPECTED=" + expected,
            }
        )

    for field in (
        "execution_spec_artifact",
        "latent_path_artifact",
        "tool_snapshot_artifact",
        "evidence_snapshot_artifact",
        "world_state_snapshot_artifact",
    ):
        attacked = dict(arguments)
        attacked["context"] = copy.deepcopy(arguments["context"])
        attacked["context"][field] = case["store"].put_json(
            {
                "schema": "aura.attack.answer_injection.v1",
                "candidate_visible": True,
                "expected": expected.decode(),
            }
        )
        with pytest.raises(
            VerifiedTransitionError,
            match="(schema_invalid|candidate_isolation_failed)",
        ):
            build_generation_trace_payload(**attacked)


def test_execution_and_calibration_out_of_band_pins_reject_rebinding(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    tampered_manifest = copy.deepcopy(case["execution_manifest"])
    tampered_manifest["manifest_id"] = "attacker-manifest"
    tampered_manifest = _reseal(tampered_manifest)
    wrong_manifest_context = replace(
        case["trust_context"],
        execution_manifest=tampered_manifest,
    )
    with pytest.raises(
        VerifiedTransitionError,
        match="transition_execution_manifest_pin_mismatch",
    ):
        build_generation_trace_payload(
            **{
                **_pass_trace_arguments(case),
                "trust_context": wrong_manifest_context,
            }
        )

    tampered_calibration = copy.deepcopy(case["calibration_evidence"])
    tampered_calibration["agreement_count"] -= 1
    tampered_calibration = _reseal(tampered_calibration)
    wrong_calibration_context = replace(
        case["trust_context"],
        calibration_evidence=tampered_calibration,
    )
    with pytest.raises(
        VerifiedTransitionError,
        match="transition_calibration_evidence_pin_mismatch",
    ):
        build_generation_trace_payload(
            **{
                **_pass_trace_arguments(case),
                "trust_context": wrong_calibration_context,
            }
        )


def test_tokenizer_pin_positive_logprobs_and_generation_signature_are_enforced(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    arguments = _pass_trace_arguments(case)
    with pytest.raises(
        VerifiedTransitionError,
        match="behavior_policy_logprob_positive",
    ):
        build_generation_trace_payload(
            **{
                **arguments,
                "behavior_policy_logprobs": ["1"] * len(arguments["output_token_ids"]),
            }
        )
    with pytest.raises(
        VerifiedTransitionError,
        match=("execution_manifest_token_encoder_mismatch|token_encoder_callable_mismatch"),
    ):
        validate_reasoning_pass_receipt(
            case["store"],
            case["pass_0"],
            task=case["task"],
            expected_authority=case["expected"],
            independent_scorer=score_frontier_response_independently,
            token_encoder=_offset_byte_encode,
            token_decoder=_offset_byte_decode,
            trust_context=case["trust_context"],
        )

    attacked = copy.deepcopy(case["pass_0"])
    generation_attestation = case["store"].read_json(
        attacked["generation_worker_attestation_artifact"],
        role="generation_worker_attestation",
    )
    generation_attestation["signature_b64"] = base64.b64encode(b"\x00" * 64).decode()
    attacked["generation_worker_attestation_artifact"] = case["store"].put_json(
        generation_attestation
    )
    attacked = _reseal(attacked)
    with pytest.raises(
        VerifiedTransitionError,
        match="campaign_attestation_signature_invalid",
    ):
        validate_reasoning_pass_receipt(
            case["store"],
            attacked,
            task=case["task"],
            expected_authority=case["expected"],
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            trust_context=case["trust_context"],
        )


def test_scalar_scorer_global_substitution_changes_policy_identity(
    complete_episode: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = complete_episode
    authority = case["store"].read_json(
        case["pass_0"]["verifier_authority_artifact"],
        role="authority",
    )
    monkeypatch.setattr(
        frontier_tasks_runtime,
        "FINAL_ANSWER_MARKER",
        "ATTACKED FINAL ANSWER:",
    )
    with pytest.raises(
        VerifiedTransitionError,
        match="transition_verifier_identity_not_policy_pinned",
    ):
        validate_frontier_verifier_authority(
            case["store"],
            authority,
            task=case["task"],
            response_artifact=case["pass_0"]["response_artifact"],
            expected_authority=case["expected"],
            independent_scorer=score_frontier_response_independently,
            trust_context=case["trust_context"],
        )


def test_module_attribute_substitution_changes_verifier_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = verifier_implementation_identity(score_frontier_response_independently)
    original_loads = json.loads

    def attacked_loads(*args: Any, **kwargs: Any) -> Any:
        return original_loads(*args, **kwargs)

    monkeypatch.setattr(json, "loads", attacked_loads)
    assert verifier_implementation_identity(score_frontier_response_independently) != before


def test_transitive_module_attribute_substitution_changes_verifier_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = verifier_implementation_identity(score_frontier_response_independently)
    original_decoder = json.JSONDecoder

    class AttackedDecoder(original_decoder):
        pass

    monkeypatch.setattr(json, "JSONDecoder", AttackedDecoder)
    assert verifier_implementation_identity(score_frontier_response_independently) != before


def test_transitive_class_method_substitution_changes_verifier_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = verifier_implementation_identity(score_frontier_response_independently)
    original_decode = json.JSONDecoder.decode

    def attacked_decode(self: Any, payload: str, *args: Any, **kwargs: Any) -> Any:
        return original_decode(self, payload, *args, **kwargs)

    monkeypatch.setattr(json.JSONDecoder, "decode", attacked_decode)
    assert verifier_implementation_identity(score_frontier_response_independently) != before


def test_execution_manifest_replays_actual_component_files(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    component_file = case["component_roots"]["runtime"] / "fixture.bin"
    original = component_file.read_bytes()
    try:
        component_file.write_bytes(b"substituted-runtime")
        with pytest.raises(
            VerifiedTransitionError,
            match="execution_manifest_component_content_mismatch",
        ):
            validate_reasoning_pass_receipt(
                case["store"],
                case["pass_0"],
                task=case["task"],
                expected_authority=case["expected"],
                independent_scorer=score_frontier_response_independently,
                token_encoder=_byte_encode,
                token_decoder=_byte_decode,
                trust_context=case["trust_context"],
            )
    finally:
        component_file.write_bytes(original)


def test_calibration_replays_immutable_cases(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    attacked = copy.deepcopy(case["calibration_evidence"])
    attacked["cases"][0]["response_b64"] = base64.b64encode(b"not a scored answer").decode("ascii")
    attacked = _reseal(attacked)
    attacked_context = replace(
        case["trust_context"],
        calibration_evidence=attacked,
        expected_calibration_evidence_sha256=hashlib.sha256(
            canonical_json_bytes(attacked)
        ).hexdigest(),
    )
    attacked_expected = {
        **case["expected"],
        "calibration_evidence_sha256": (attacked_context.expected_calibration_evidence_sha256),
    }
    with pytest.raises(
        VerifiedTransitionError,
        match="calibration_control_response_mismatch",
    ):
        validate_reasoning_pass_receipt(
            case["store"],
            case["pass_0"],
            task=case["task"],
            expected_authority=attacked_expected,
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            trust_context=attacked_context,
        )


def test_calibration_requires_positive_and_negative_control_coverage() -> None:
    tasks = [
        generate_task("mathematics", seed=817_102, difficulty=2),
        generate_task("mathematics", seed=817_103, difficulty=2),
    ]
    cases = [
        build_calibration_case(
            task=task,
            case_kind="canonical_positive",
            independent_scorer=score_frontier_response_independently,
        )
        for task in tasks
    ]
    with pytest.raises(
        VerifiedTransitionError,
        match="calibration_control_coverage_incomplete",
    ):
        build_calibration_payload(
            verifier_implementation_sha256=verifier_implementation_identity(
                score_frontier_response_independently
            ),
            trust_policy_sha256=_sha("calibration-trust-policy"),
            cases=cases,
            independent_scorer=score_frontier_response_independently,
            acceptance_policy_sha256=_sha("calibration-acceptance-policy"),
            calibrated_at_unix_ns=1_800_000_204_000_000_000,
        )


def test_calibration_includes_parsed_but_wrong_semantic_control(
    complete_episode: dict[str, Any],
) -> None:
    cases = {case["case_kind"]: case for case in complete_episode["calibration_evidence"]["cases"]}
    negative = cases["parsed_wrong_negative"]
    assert negative["primary_output"]["parsed"] is True
    assert negative["primary_output"]["correct"] is False
    assert negative["independent_output"]["parsed"] is True
    assert negative["independent_output"]["correct"] is False


def test_execution_observer_signature_is_independently_enforced(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    attacked = copy.deepcopy(case["pass_0"])
    observer = case["store"].read_json(
        attacked["execution_observer_attestation_artifact"],
        role="execution_observer_attestation",
    )
    observer["signature_b64"] = base64.b64encode(b"\x00" * 64).decode()
    attacked["execution_observer_attestation_artifact"] = case["store"].put_json(observer)
    attacked = _reseal(attacked)
    with pytest.raises(
        VerifiedTransitionError,
        match="campaign_attestation_signature_invalid",
    ):
        validate_reasoning_pass_receipt(
            case["store"],
            attacked,
            task=case["task"],
            expected_authority=case["expected"],
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            trust_context=case["trust_context"],
        )


def test_execution_process_observation_is_host_collected_and_tamper_evident(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    binding = case["execution_process_observation_artifact_0"]
    observation = case["store"].read_json(
        binding,
        role="execution_process_observation",
    )
    assert observation["observation_source"] == "host"
    assert observation["observer_backend"] == "HostResourceObserver"
    assert observation["pid"] == os.getpid()
    assert observation["observed_component_roots"] == case["execution_manifest"]["component_roots"]
    assert observation["open_file_identity_count"] >= 8
    assert len(observation["observed_component_descriptor_identities_sha256"]) == 64

    attacked_observation = copy.deepcopy(observation)
    attacked_observation["observation_source"] = "simulated"
    attacked = copy.deepcopy(case["pass_0"])
    attacked["execution_process_observation_artifact"] = case["store"].put_json(
        attacked_observation
    )
    attacked = _reseal(attacked)
    with pytest.raises(
        VerifiedTransitionError,
        match="execution_process_observation_mismatch",
    ):
        validate_reasoning_pass_receipt(
            case["store"],
            attacked,
            task=case["task"],
            expected_authority=case["expected"],
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            trust_context=case["trust_context"],
        )


def test_execution_process_observation_rejects_replaced_open_component_inode(
    complete_episode: dict[str, Any],
    tmp_path: Path,
) -> None:
    case = complete_episode
    component_roots = _component_roots(tmp_path / "replaced-components")
    handles = [(root / "fixture.bin").open("rb") for root in component_roots.values()]
    try:
        runtime_path = component_roots["runtime"] / "fixture.bin"
        replacement = runtime_path.with_name("replacement.bin")
        replacement.write_bytes(b"replacement-runtime-component")
        os.replace(replacement, runtime_path)

        attacked_manifest = build_execution_manifest(
            manifest_id="execution-manifest-replaced-runtime",
            component_roots=component_roots,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            independent_scorer=score_frontier_response_independently,
            created_at_unix_ns=PASS_0_AT - 50,
        )
        attacked_context = _pass_context(
            case["store"],
            generated_at=PASS_0_AT,
            latent_label="latent-replaced-runtime",
            task=case["task"],
            execution_manifest=attacked_manifest,
        )
        with pytest.raises(
            VerifiedTransitionError,
            match="execution_observer_component_descriptor_identity_mismatch",
        ):
            capture_execution_process_observation(
                case["store"],
                context=attacked_context,
                execution_component_roots=component_roots,
            )
    finally:
        for handle in handles:
            handle.close()


def test_process_receipt_component_substitution_is_rejected(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    arguments = _pass_trace_arguments(case)
    context = copy.deepcopy(arguments["context"])
    process = case["store"].read_json(
        context["process_receipt_artifact"],
        role="process_receipt",
    )
    process["loaded_component_roots"]["runtime"] = _sha("substituted-runtime-root")
    context["process_receipt_artifact"] = case["store"].put_json(process)
    with pytest.raises(
        VerifiedTransitionError,
        match="process_loaded_component_roots_invalid",
    ):
        build_generation_trace_payload(
            **{
                **arguments,
                "context": context,
            }
        )


def test_attempt_ledger_rejects_omitted_reordered_and_extra_events(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    for events in (
        case["runner_event_attestations"][:-1],
        list(reversed(case["runner_event_attestations"])),
        [
            *case["runner_event_attestations"],
            case["runner_event_attestations"][-1],
        ],
    ):
        attacked = copy.deepcopy(case["journal"])
        attacked["runner_event_attestations"] = events
        attacked = _reseal(attacked)
        with pytest.raises(
            VerifiedTransitionError,
            match="transition_attempt_journal_mismatch",
        ):
            build_verified_transition_episode(
                case["store"],
                pass_0=case["pass_0"],
                pass_1=case["pass_1"],
                task=case["task"],
                expected_authority=case["expected"],
                independent_scorer=score_frontier_response_independently,
                token_encoder=_byte_encode,
                token_decoder=_byte_decode,
                trust_context=case["trust_context"],
                attempt_journal=attacked,
                campaign_runner_attestation=case["runner_attestation"],
                evidence_verifier_journal_attestation=case["evidence_verifier_journal_attestation"],
            )


def test_attempt_ledger_pin_and_terminal_are_enforced(
    complete_episode: dict[str, Any],
    tmp_path: Path,
) -> None:
    case = complete_episode
    alternate = ExternalAttemptLedger(
        tmp_path / "alternate-ledger" / "events.jsonl",
        create=True,
    )
    try:
        wrong_context = replace(
            case["trust_context"],
            attempt_ledger_path=alternate.path,
        )
        with pytest.raises(
            VerifiedTransitionError,
            match="attempt_ledger_identity_pin_mismatch",
        ):
            build_transition_attempt_journal(
                pass_0=case["pass_0"],
                pass_1=case["pass_1"],
                protocol_sha256=PROTOCOL_SHA256,
                trust_context=wrong_context,
            )
    finally:
        _clear_append_only_for_test(alternate.path)

    with pytest.raises(
        VerifiedTransitionError,
        match="attempt_ledger_already_terminal",
    ):
        case["attempt_ledger"].append(
            policy=case["policy"],
            attestation=case["runner_event_attestations"][-1],
        )


def test_attempt_ledger_same_inode_rewrite_is_rejected(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    path = case["attempt_ledger"].path
    original = path.read_bytes()
    original_stat = os.stat(path, follow_symlinks=False)
    original_flags = getattr(original_stat, "st_flags", 0)
    try:
        if hasattr(os, "chflags"):
            os.chflags(path, 0, follow_symlinks=False)
        path.write_bytes(original.rsplit(b"\n", 2)[0] + b"\n")
        if hasattr(os, "chflags"):
            os.chflags(path, original_flags, follow_symlinks=False)
        assert os.stat(path, follow_symlinks=False).st_ino == original_stat.st_ino
        with pytest.raises(VerifiedTransitionError):
            build_transition_attempt_journal(
                pass_0=case["pass_0"],
                pass_1=case["pass_1"],
                protocol_sha256=PROTOCOL_SHA256,
                trust_context=case["trust_context"],
            )
    finally:
        if hasattr(os, "chflags"):
            os.chflags(path, 0, follow_symlinks=False)
        path.write_bytes(original)
        if hasattr(os, "chflags"):
            os.chflags(path, original_flags, follow_symlinks=False)


@pytest.mark.parametrize(
    "field",
    [
        "attempt_ledger_open_attestation",
        "attempt_ledger_terminal_attestation",
    ],
)
def test_attempt_ledger_external_checkpoint_signatures_are_enforced(
    complete_episode: dict[str, Any],
    field: str,
) -> None:
    case = complete_episode
    attacked = copy.deepcopy(getattr(case["trust_context"], field))
    attacked["signature_b64"] = base64.b64encode(b"\x00" * 64).decode()
    attacked_context = replace(case["trust_context"], **{field: attacked})
    with pytest.raises(
        VerifiedTransitionError,
        match="campaign_attestation_signature_invalid",
    ):
        build_transition_attempt_journal(
            pass_0=case["pass_0"],
            pass_1=case["pass_1"],
            protocol_sha256=PROTOCOL_SHA256,
            trust_context=attacked_context,
        )


def test_attempt_ledger_must_open_before_task_disclosure(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    issuer_at = case["trust_context"].task_issuer_attestation["signed_payload"]["signed_at_unix"]
    original_payload = case["trust_context"].attempt_ledger_open_attestation["signed_payload"][
        "payload"
    ]
    attacked_payload = {
        **original_payload,
        "opened_at_unix_ns": (issuer_at + 1) * 1_000_000_000,
    }
    attacked_open = build_role_attestation(
        case["policy"],
        role=TASK_ISSUER,
        payload=attacked_payload,
        signed_at_unix=issuer_at + 1,
        private_key=case["role_keys"][TASK_ISSUER],
    )
    attacked_context = replace(
        case["trust_context"],
        attempt_ledger_open_attestation=attacked_open,
    )
    with pytest.raises(
        VerifiedTransitionError,
        match="attempt_ledger_not_open_before_task_disclosure",
    ):
        build_generation_trace_payload(
            **{
                **_pass_trace_arguments(case),
                "trust_context": attacked_context,
            }
        )


def test_nanosecond_chronology_rejects_preseal_runner_attestation(
    complete_episode: dict[str, Any],
) -> None:
    case = complete_episode
    premature_at = VERIFIER_AT
    premature = build_role_attestation(
        case["policy"],
        role=CAMPAIGN_RUNNER,
        payload=build_campaign_runner_journal_payload(
            case["journal"],
            signed_at_unix_ns=premature_at,
        ),
        signed_at_unix=premature_at // 1_000_000_000,
        private_key=case["role_keys"][CAMPAIGN_RUNNER],
    )
    with pytest.raises(
        VerifiedTransitionError,
        match="runner_journal_signed_before_pass_seal",
    ):
        build_verified_transition_episode(
            case["store"],
            pass_0=case["pass_0"],
            pass_1=case["pass_1"],
            task=case["task"],
            expected_authority=case["expected"],
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            trust_context=case["trust_context"],
            attempt_journal=case["journal"],
            campaign_runner_attestation=premature,
            evidence_verifier_journal_attestation=case["evidence_verifier_journal_attestation"],
        )

    wrong_attestation = build_role_attestation(
        case["policy"],
        role=CAMPAIGN_RUNNER,
        payload={
            **build_campaign_runner_journal_payload(
                case["journal"],
                signed_at_unix_ns=RUNNER_AT * 1_000_000_000,
            ),
            "terminal_state": "aborted",
        },
        signed_at_unix=RUNNER_AT,
        private_key=case["role_keys"][CAMPAIGN_RUNNER],
    )
    with pytest.raises(ValueError, match="campaign_attestation_payload_mismatch"):
        build_verified_transition_episode(
            case["store"],
            pass_0=case["pass_0"],
            pass_1=case["pass_1"],
            task=case["task"],
            expected_authority=case["expected"],
            independent_scorer=score_frontier_response_independently,
            token_encoder=_byte_encode,
            token_decoder=_byte_decode,
            trust_context=case["trust_context"],
            attempt_journal=case["journal"],
            campaign_runner_attestation=wrong_attestation,
            evidence_verifier_journal_attestation=case["evidence_verifier_journal_attestation"],
        )
