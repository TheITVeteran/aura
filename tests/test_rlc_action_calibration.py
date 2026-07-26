from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.action_calibration import (
    ACTION_CALIBRATION_RESULT_SCHEMA,
    ACTION_CALIBRATION_VERIFICATION_SCHEMA,
    ACTION_RESOURCE_DIMENSIONS,
    CONTROL_ARM,
    EXPECTED_ACTION_COUNT,
    MIN_EXECUTION_COUNT,
    MIN_PAIR_COUNT,
    TREATMENT_ARM,
    ActionCalibrationError,
    action_calibration_contamination_payload,
    action_calibration_final_verifier_payload,
    action_calibration_issuer_payload,
    action_calibration_output_seal_payload,
    action_calibration_runner_payload,
    action_calibration_starting_state_payload,
    action_calibration_verifier_payload,
    build_action_calibration_candidate,
    build_action_calibration_plan,
    certified_evidence_snapshot,
    finalize_action_calibration_certificate,
    verify_action_calibration_certificate,
)
from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignJournal,
    canonical_json_bytes,
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
from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.brain.llm.latent_cortex.execution_controller import (
    ExecutionController,
)
from core.brain.llm.latent_cortex.frontier_tasks import (
    FRONTIER_DOMAINS,
    FrontierTask,
    generate_task,
    reblind_frontier_task,
)
from core.brain.llm.latent_cortex.resource_accounting import (
    ModelComputeProfile,
    ResourceLedger,
    build_information_receipt,
)
from core.brain.llm.latent_cortex.value_of_computation import (
    ACTION_EVIDENCE_SCHEMA,
    validate_evidence_snapshot,
)
from tools.verify_rlc_action_calibration import verify_files

CAMPAIGN_NAME = "rlc-action-calibration-test"
MODEL_IDENTITY = {
    "model_path": "/sealed/resident-32b",
    "checkpoint_sha256": "a" * 64,
    "runtime_bundle_sha256": "b" * 64,
    "logical_parameter_count": 32_763_876_352,
}
EXECUTION_CONFIG = {
    "worker_task_material": "public_manifest_only",
    "answer_reveal_protocol": "sealed_outputs_then_issuer_reveal_v1",
    "answer_blind_nonce_policy": "external_issuer_csprng_256",
    "answer_blind_nonce_disclosure": "post_seal_answer_reveal",
    "answer_blind_nonce_count": MIN_PAIR_COUNT,
    "answer_blind_nonce_min_entropy_bits": 256,
    "generation_seed_policy": "external_issuer_uniform_63bit",
    "generation_seed_count": MIN_PAIR_COUNT,
    "generation_seed_min_entropy_bits": 60,
    "task_assignment_policy": "external_issuer_stratified_random_without_replacement_v1",
    "task_assignment_seed_sha256": "2" * 64,
    "action_cost_budget_estimated_flops": 10**12,
    "action_resource_caps": {
        name: 100 if name == "host_scalar_ops" else 10**12 for name in ACTION_RESOURCE_DIMENSIONS
    },
    "continuation_policy_sha256": "c" * 64,
    "budget_policy_sha256": "d" * 64,
    "rng_root_sha256": "e" * 64,
    "instrumentation_sha256": "f" * 64,
    "execute_fixture_policy_sha256": "1" * 64,
    "execute_calibration_effect_class": "deterministic_sandbox",
}
CALIBRATION_BUCKET = "general|none|short|s:mid|u:mid"


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


def _trust_fixture():
    root = Ed25519PrivateKey.generate()
    role_keys = {role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES}
    roles = {}
    for role, key in role_keys.items():
        public = _public_raw(key)
        roles[role] = {
            "signer_id": f"{role}-signer",
            "organization_id": f"{role}-organization",
            "public_key_b64": base64.b64encode(public).decode("ascii"),
            "key_id": hashlib.sha256(public).hexdigest(),
            "implementation_sha256": hashlib.sha256(f"{role}:implementation".encode()).hexdigest(),
            "release_sha256": hashlib.sha256(f"{role}:release".encode()).hexdigest(),
            "custody_class": "external_service",
            "custody_evidence_sha256": hashlib.sha256(f"{role}:custody".encode()).hexdigest(),
        }
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "action-calibration-test-policy",
        "policy_revision": 1,
        "campaign_name": CAMPAIGN_NAME,
        "protocol_sha256": "9" * 64,
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": 800,
        "not_before_unix": 900,
        "expires_at_unix": 2_000,
        "roles": roles,
    }
    signed = canonical_json_bytes(body)
    root_raw = _public_raw(root)
    document = {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }
    policy = validate_campaign_trust_policy(
        document,
        trusted_root_public_key_pem=_public_pem(root),
        expected_campaign_name=CAMPAIGN_NAME,
        now_unix=1_000,
    )
    return policy, role_keys, root


def _tasks_by_action(
    task_count: int = 8,
) -> dict[OperationKind, tuple[FrontierTask, ...]]:
    values: dict[OperationKind, tuple[FrontierTask, ...]] = {}
    for action_ordinal, action in enumerate(OperationKind):
        values[action] = tuple(
            reblind_frontier_task(
                generate_task(
                    FRONTIER_DOMAINS[(action_ordinal + task_ordinal) % 2],
                    seed=10_000 + action_ordinal * 100 + task_ordinal,
                    difficulty=2,
                ),
                blind_nonce=hashlib.sha256(
                    (f"external-issuer:{action.value}:{task_ordinal}").encode()
                ).digest(),
            )
            for task_ordinal in range(task_count)
        )
    return values


def _profile() -> ModelComputeProfile:
    return ModelComputeProfile(
        model_type="action-calibration-fixture",
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        head_dim=4,
    )


def _resource_receipt(*, action: bool, control: bool = False) -> dict:
    ledger = ResourceLedger(_profile())
    if not control:
        ledger.charge(
            "forced_action" if action else "whole_arm",
            transformer_layer_apps=4 if action else 20,
            attention_query_key_pairs=8 if action else 40,
            output_head_tokens=0 if action else 4,
            host_scalar_ops=100 if action else 500,
        )
    return ledger.to_receipt()


def _information_receipt(task_payload_sha256: str) -> dict:
    return build_information_receipt(
        sources=[
            {
                "source_id": "held_out_public_task",
                "kind": "task_prompt",
                "content_sha256": task_payload_sha256,
                "byte_count": 256,
                "token_count": 64,
            }
        ],
        policies={
            "continuation": EXECUTION_CONFIG["continuation_policy_sha256"],
            "budget": EXECUTION_CONFIG["budget_policy_sha256"],
        },
    )


def _consumed_information_receipt() -> dict:
    return build_information_receipt(
        sources=[],
        policies={
            "acquisition": hashlib.sha256(b"action-calibration-acquisition-policy").hexdigest(),
        },
    )


def _answer(task: FrontierTask, *, correct: bool) -> str:
    if not correct:
        return "intentionally invalid calibration control output"
    return "FINAL_ANSWER: " + json.dumps(
        task.reveal_for_verifier()["expected"],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _starting_state_receipts(
    tasks: dict[OperationKind, tuple[FrontierTask, ...]],
    *,
    policy,
    role_keys,
    execution_config: dict,
) -> dict[str, dict]:
    receipts: dict[str, dict] = {}
    for action, action_tasks in tasks.items():
        for task in action_tasks:
            component_hashes = {
                name: hashlib.sha256(
                    f"{task.task_id}:{name}:captured-runtime-bytes".encode()
                ).hexdigest()
                for name in (
                    "latent_slots_sha256",
                    "branch_state_sha256",
                    "kv_cache_sha256",
                    "evidence_state_sha256",
                    "memory_state_sha256",
                    "public_action_state_sha256",
                    "durable_state_sha256",
                    "rng_state_sha256",
                )
            }
            payload = action_calibration_starting_state_payload(
                campaign_name=CAMPAIGN_NAME,
                action=action,
                task=task.public,
                model_identity=MODEL_IDENTITY,
                execution_config=execution_config,
                calibration_bucket=CALIBRATION_BUCKET,
                capture_id=f"capture:{task.task_id}",
                captured_at_unix=1_025,
                bucket_classifier_sha256="3" * 64,
                bucket_evidence_sha256=hashlib.sha256(
                    f"{task.task_id}:{CALIBRATION_BUCKET}".encode()
                ).hexdigest(),
                state_component_sha256=component_hashes,
            )
            receipts[task.task_id] = {
                **payload,
                "capture_attestation": build_role_attestation(
                    policy,
                    role=CAMPAIGN_RUNNER,
                    payload=payload,
                    signed_at_unix=1_025,
                    private_key=role_keys[CAMPAIGN_RUNNER],
                ),
            }
    return receipts


def _plan(policy, role_keys, *, task_count: int = 8):
    tasks = _tasks_by_action(task_count)
    execution_config = {
        **EXECUTION_CONFIG,
        "answer_blind_nonce_count": EXPECTED_ACTION_COUNT * task_count,
        "generation_seed_count": EXPECTED_ACTION_COUNT * task_count,
    }
    starting_state_receipts = _starting_state_receipts(
        tasks,
        policy=policy,
        role_keys=role_keys,
        execution_config=execution_config,
    )
    plan = build_action_calibration_plan(
        CAMPAIGN_NAME,
        tasks,
        model_identity=MODEL_IDENTITY,
        execution_config=execution_config,
        calibration_bucket=CALIBRATION_BUCKET,
        starting_state_receipts=starting_state_receipts,
        campaign_trust={
            "prelaunch_verified": True,
            "externally_custodied": True,
            "policy_sha256": policy.policy_sha256,
        },
        claim_eligible=True,
    )
    return plan, tasks


def _result_core(
    plan,
    cell_id: str,
    attempt_id: str,
    task: FrontierTask,
) -> dict:
    definition = plan.cell_definition(cell_id)
    treatment = definition["arm"] == TREATMENT_ARM
    text = _answer(task, correct=treatment)
    telemetry_body = {
        "schema": "aura.rlc.action_calibration.host_telemetry.v1",
        "instrumentation_sha256": EXECUTION_CONFIG["instrumentation_sha256"],
        "monotonic_start_ns": 1_000_000,
        "monotonic_end_ns": 2_000_000,
        "cpu_time_ns": 500_000,
        "peak_resident_bytes": 1_048_576,
        "complete": True,
    }
    erasure_body = {
        "schema": "aura.rlc.action_calibration.mutation_erasure.v1",
        "pre_durable_state_sha256": definition["starting_state"]["durable_state_sha256"],
        "post_durable_state_sha256": definition["starting_state"]["durable_state_sha256"],
        "transient_state_erased": True,
    }
    return {
        "schema": ACTION_CALIBRATION_RESULT_SCHEMA,
        "arm": definition["arm"],
        "action": definition["action"],
        "pair_id": definition["pair_id"],
        "campaign_plan_sha256": plan.plan_sha256,
        "attempt_id": attempt_id,
        "starting_state_sha256": definition["starting_state_sha256"],
        "starting_state": definition["starting_state"],
        "action_execution": {
            "selection_mode": ("campaign_forced" if treatment else "matched_no_action_control"),
            "selected_action": definition["action"] if treatment else None,
            "campaign_authority_sha256": plan.plan_sha256,
        },
        "action_trace": {
            "schema": "aura.rlc.action_calibration.action_trace.v1",
            "action": definition["action"],
            "intervention_ordinal": 0,
            "selected_action_occurrences": 1 if treatment else 0,
            "action_excluded_at_intervention": not treatment,
            "pre_state_sha256": definition["starting_state_sha256"],
            "post_state_sha256": hashlib.sha256(f"{cell_id}:post".encode()).hexdigest(),
        },
        "text": text,
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "runtime_identity": MODEL_IDENTITY,
        "resource_accounting": _resource_receipt(action=False),
        "action_resource_accounting": _resource_receipt(
            action=treatment,
            control=not treatment,
        ),
        "available_information_accounting": _information_receipt(definition["task_payload_sha256"]),
        "consumed_information_accounting": (_consumed_information_receipt()),
        "host_telemetry": {
            **telemetry_body,
            "sample_sha256": hashlib.sha256(canonical_json_bytes(telemetry_body)).hexdigest(),
        },
        "mutation_erasure": {
            **erasure_body,
            "receipt_sha256": hashlib.sha256(canonical_json_bytes(erasure_body)).hexdigest(),
        },
    }


def _complete_campaign(
    tmp_path: Path,
    *,
    task_count: int = 8,
    claim_interventions: bool = False,
):
    policy, role_keys, root = _trust_fixture()
    plan, tasks_by_action = _plan(
        policy,
        role_keys,
        task_count=task_count,
    )
    tasks = {
        task.task_id: task for action_tasks in tasks_by_action.values() for task in action_tasks
    }
    attempts: dict[str, str] = {}
    results: dict[str, dict] = {}
    journal_path = tmp_path / "action-calibration.jsonl"
    with CampaignJournal(journal_path, plan) as journal:
        for cell_id in plan.cell_ids:
            definition = plan.cell_definition(cell_id)
            attempt_id = journal.start_cell(cell_id)
            if claim_interventions:
                snapshot = journal.resume()
                journal.claim_action_intervention(
                    cell_id,
                    attempt_id,
                    intervention_sha256=hashlib.sha256(
                        f"{cell_id}:intervention".encode()
                    ).hexdigest(),
                    request_payload_sha256=hashlib.sha256(
                        f"{cell_id}:request".encode()
                    ).hexdigest(),
                    expected_journal_head_sha256=snapshot.journal_head_sha256,
                    expected_journal_event_count=2 + len(attempts) * 3,
                )
            core = _result_core(
                plan,
                cell_id,
                attempt_id,
                tasks[definition["task_id"]],
            )
            runner_payload = action_calibration_runner_payload(
                plan=plan,
                cell_id=cell_id,
                attempt_id=attempt_id,
                result_core=core,
            )
            result = {
                **core,
                "runner_attestation": build_role_attestation(
                    policy,
                    role=CAMPAIGN_RUNNER,
                    payload=runner_payload,
                    signed_at_unix=1_100,
                    private_key=role_keys[CAMPAIGN_RUNNER],
                ),
            }
            journal.record_arm_result(cell_id, attempt_id, result)
            attempts[cell_id] = attempt_id
            results[cell_id] = result

        sealed_head = journal.resume().journal_head_sha256
        output_seal = action_calibration_output_seal_payload(
            plan,
            result_sha256_by_cell={
                cell_id: hashlib.sha256(canonical_json_bytes(result)).hexdigest()
                for cell_id, result in results.items()
            },
            journal_head_sha256=sealed_head,
            journal_event_count=1
            + len(plan.cell_ids) * (3 if claim_interventions else 2),
        )
        output_seal_attestation = build_role_attestation(
            policy,
            role=CAMPAIGN_RUNNER,
            payload=output_seal,
            signed_at_unix=1_200,
            private_key=role_keys[CAMPAIGN_RUNNER],
        )

        for cell_id in plan.cell_ids:
            definition = plan.cell_definition(cell_id)
            result = results[cell_id]
            task = tasks[definition["task_id"]]
            score = task.score(result["text"]).to_dict()
            result_sha256 = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
            verifier_payload = action_calibration_verifier_payload(
                plan=plan,
                cell_id=cell_id,
                result_sha256=result_sha256,
                score_receipt=score,
                answer_commitment_sha256=task.public.answer_commitment_sha256,
            )
            verification = {
                "schema": ACTION_CALIBRATION_VERIFICATION_SCHEMA,
                "correct": score["correct"],
                "score_receipt": score,
                "answer_commitment_sha256": (task.public.answer_commitment_sha256),
                "result_sha256": result_sha256,
                "verifier_attestation": build_role_attestation(
                    policy,
                    role=EVIDENCE_VERIFIER,
                    payload=verifier_payload,
                    signed_at_unix=1_300,
                    private_key=role_keys[EVIDENCE_VERIFIER],
                ),
            }
            journal.record_verified(
                cell_id,
                attempts[cell_id],
                verification,
            )
            journal.commit_cell(
                cell_id,
                attempts[cell_id],
                {
                    "result_sha256": result_sha256,
                    "verification_sha256": hashlib.sha256(
                        canonical_json_bytes(verification)
                    ).hexdigest(),
                },
            )
        records = journal.committed_records()
        manifest = journal.finalize(tmp_path / "manifest.json")
    journal_transcript = [
        json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]

    issuer_payload = action_calibration_issuer_payload(plan)
    contamination_payload = action_calibration_contamination_payload(
        plan,
        corpus_snapshot_sha256="7" * 64,
        methods=("exact_prompt", "normalized_prompt", "token_fivegram"),
    )
    candidate = build_action_calibration_candidate(
        records,
        plan=plan,
        issuer_tasks=tuple(tasks.values()),
        campaign_manifest=manifest,
        campaign_journal=journal_transcript,
        policy=policy,
        issuer_attestation=build_role_attestation(
            policy,
            role=TASK_ISSUER,
            payload=issuer_payload,
            signed_at_unix=1_000,
            private_key=role_keys[TASK_ISSUER],
        ),
        contamination_attestation=build_role_attestation(
            policy,
            role=CONTAMINATION_AUDITOR,
            payload=contamination_payload,
            signed_at_unix=1_000,
            private_key=role_keys[CONTAMINATION_AUDITOR],
        ),
        contamination_payload=contamination_payload,
        output_seal_payload=output_seal,
        output_seal_attestation=output_seal_attestation,
    )
    final_payload = action_calibration_final_verifier_payload(
        candidate,
        policy=policy,
    )
    certificate = finalize_action_calibration_certificate(
        candidate,
        policy=policy,
        final_verifier_attestation=build_role_attestation(
            policy,
            role=EVIDENCE_VERIFIER,
            payload=final_payload,
            signed_at_unix=1_400,
            private_key=role_keys[EVIDENCE_VERIFIER],
        ),
    )
    return policy, root, plan, candidate, certificate


def test_plan_freezes_exact_global_coverage_and_counterbalances_each_action():
    policy, role_keys, _root = _trust_fixture()
    plan, _tasks = _plan(policy, role_keys)
    metadata = plan.to_dict()["metadata"]

    assert metadata["action_count"] == EXPECTED_ACTION_COUNT
    assert metadata["pair_count"] == MIN_PAIR_COUNT
    assert metadata["execution_count"] == MIN_EXECUTION_COUNT
    assert len(plan.cell_ids) == MIN_EXECUTION_COUNT
    assert len({row["task_id"] for row in metadata["assignments"]}) == MIN_PAIR_COUNT
    assert (
        metadata["sampling_frame"]["assignment_policy"]
        == "external_issuer_stratified_random_without_replacement_v1"
    )
    assert len(metadata["sampling_frame"]["task_sampling_identities"]) == MIN_PAIR_COUNT
    for action in OperationKind:
        rows = [row for row in metadata["assignments"] if row["action"] == action.value]
        assert len(rows) == 8
        assert sum(row["arm_order"][0] == TREATMENT_ARM for row in rows) == 4
        assert sum(row["arm_order"][0] == CONTROL_ARM for row in rows) == 4
        for row in rows:
            cell = next(
                plan.cell_definition(cell_id)
                for cell_id in plan.cell_ids
                if plan.cell_definition(cell_id)["pair_id"] == row["pair_id"]
            )
            assert (
                cell["starting_state"]["capture_mode"]
                == "externally_captured_runtime_state_v1"
            )
            assert cell["starting_state"]["calibration_bucket"] == CALIBRATION_BUCKET


def test_plan_rejects_task_reuse_and_missing_action():
    policy, _role_keys, _root = _trust_fixture()
    tasks = _tasks_by_action()
    missing = dict(tasks)
    missing.pop(OperationKind.ABSTAIN)
    with pytest.raises(
        ActionCalibrationError,
        match="action_calibration_action_coverage_invalid",
    ):
        build_action_calibration_plan(
            CAMPAIGN_NAME,
            missing,
            model_identity=MODEL_IDENTITY,
            execution_config=EXECUTION_CONFIG,
            calibration_bucket=CALIBRATION_BUCKET,
        )

    reused = dict(tasks)
    reused[OperationKind.ABSTAIN] = tasks[OperationKind.ANSWER]
    with pytest.raises(
        ActionCalibrationError,
        match="action_calibration_task_reused",
    ):
        build_action_calibration_plan(
            CAMPAIGN_NAME,
            reused,
            model_identity=MODEL_IDENTITY,
            execution_config=EXECUTION_CONFIG,
            calibration_bucket=CALIBRATION_BUCKET,
            campaign_trust={
                "prelaunch_verified": True,
                "externally_custodied": True,
                "policy_sha256": policy.policy_sha256,
            },
            claim_eligible=True,
        )


def test_plan_rejects_reblinded_duplicates_and_unbalanced_sampling_frame():
    policy, _role_keys, _root = _trust_fixture()
    tasks = _tasks_by_action()
    base_tasks = (
        generate_task(FRONTIER_DOMAINS[0], seed=880_001, difficulty=2),
        generate_task(FRONTIER_DOMAINS[1], seed=880_002, difficulty=2),
    )
    duplicate_tasks = {
        action: tuple(
            reblind_frontier_task(
                base_tasks[ordinal % 2],
                blind_nonce=hashlib.sha256(
                    f"duplicate-reblind:{action.value}:{ordinal}".encode()
                ).digest(),
            )
            for ordinal in range(8)
        )
        for action in OperationKind
    }
    with pytest.raises(
        ActionCalibrationError,
        match="action_calibration_underlying_task_reused",
    ):
        build_action_calibration_plan(
            CAMPAIGN_NAME,
            duplicate_tasks,
            model_identity=MODEL_IDENTITY,
            execution_config=EXECUTION_CONFIG,
            calibration_bucket=CALIBRATION_BUCKET,
            campaign_trust={
                "prelaunch_verified": True,
                "externally_custodied": True,
                "policy_sha256": policy.policy_sha256,
            },
            claim_eligible=True,
        )

    unbalanced = _tasks_by_action()
    replacement_domain = next(
        domain
        for domain in FRONTIER_DOMAINS[:2]
        if domain != unbalanced[OperationKind.ABSTAIN][0].domain
    )
    replacement = reblind_frontier_task(
        generate_task(
            replacement_domain,
            seed=880_003,
            difficulty=2,
        ),
        blind_nonce=hashlib.sha256(b"unbalanced-sampling-frame").digest(),
    )
    unbalanced[OperationKind.ABSTAIN] = (
        replacement,
        *unbalanced[OperationKind.ABSTAIN][1:],
    )
    with pytest.raises(
        ActionCalibrationError,
        match="action_calibration_sampling_frame_unbalanced",
    ):
        build_action_calibration_plan(
            CAMPAIGN_NAME,
            unbalanced,
            model_identity=MODEL_IDENTITY,
            execution_config=EXECUTION_CONFIG,
            calibration_bucket=CALIBRATION_BUCKET,
            claim_eligible=False,
        )

    missing_cap = copy.deepcopy(EXECUTION_CONFIG)
    missing_cap["action_resource_caps"].pop("tool_calls")
    with pytest.raises(
        ActionCalibrationError,
        match="action_calibration_resource_caps_invalid",
    ):
        build_action_calibration_plan(
            CAMPAIGN_NAME,
            tasks,
            model_identity=MODEL_IDENTITY,
            execution_config=missing_cap,
            calibration_bucket=CALIBRATION_BUCKET,
        )

    unblinded = dict(tasks)
    unblinded[OperationKind.ABSTAIN] = (
        generate_task(
            tasks[OperationKind.ABSTAIN][0].domain,
            seed=999_001,
            difficulty=2,
        ),
        *tasks[OperationKind.ABSTAIN][1:],
    )
    with pytest.raises(
        ActionCalibrationError,
        match="action_calibration_external_blinding_required",
    ):
        build_action_calibration_plan(
            CAMPAIGN_NAME,
            unblinded,
            model_identity=MODEL_IDENTITY,
            execution_config=EXECUTION_CONFIG,
            calibration_bucket=CALIBRATION_BUCKET,
            campaign_trust={
                "prelaunch_verified": True,
                "externally_custodied": True,
                "policy_sha256": policy.policy_sha256,
            },
            claim_eligible=True,
        )


def test_plan_rejects_state_capture_for_a_different_bucket():
    policy, role_keys, _root = _trust_fixture()
    tasks = _tasks_by_action()
    receipts = _starting_state_receipts(
        tasks,
        policy=policy,
        role_keys=role_keys,
        execution_config=EXECUTION_CONFIG,
    )
    first_task_id = next(iter(receipts))
    receipts[first_task_id]["calibration_bucket"] = "different|bucket"
    with pytest.raises(
        ActionCalibrationError,
        match="action_calibration_state_capture_invalid",
    ):
        build_action_calibration_plan(
            CAMPAIGN_NAME,
            tasks,
            model_identity=MODEL_IDENTITY,
            execution_config=EXECUTION_CONFIG,
            calibration_bucket=CALIBRATION_BUCKET,
            starting_state_receipts=receipts,
            campaign_trust={
                "prelaunch_verified": True,
                "externally_custodied": True,
                "policy_sha256": policy.policy_sha256,
            },
            claim_eligible=True,
        )


def test_full_external_campaign_verifies_but_eight_is_not_promoted(tmp_path):
    policy, _root, plan, candidate, certificate = _complete_campaign(tmp_path)

    assert (
        verify_action_calibration_certificate(
            certificate,
            policy=policy,
        )
        == certificate
    )
    assert candidate["pair_count"] == MIN_PAIR_COUNT
    assert candidate["execution_count"] == MIN_EXECUTION_COUNT
    assert all(cell["n"] == 8 for cell in candidate["cells"].values())
    assert all(cell["measured"] is False for cell in candidate["cells"].values())
    assert all(cell["cost_mean"] == 1.0 for cell in candidate["cells"].values())
    snapshot = certified_evidence_snapshot(
        certificate,
        policy=policy,
        bucket=CALIBRATION_BUCKET,
    )
    assert snapshot["candidate_sha256"] == candidate["candidate_sha256"]
    assert snapshot["policy_sha256"] == policy.policy_sha256
    assert all(cell["measured"] is False for cell in snapshot["cells"].values())
    controller = ExecutionController(tmp_path / "controller")
    controller._certified_action_certificate = certificate
    controller._certified_action_policy = policy
    assert controller.action_evidence_snapshot(bucket=CALIBRATION_BUCKET) == snapshot
    assert (
        controller.action_evidence_snapshot(bucket="different|bucket")["schema"]
        == ACTION_EVIDENCE_SCHEMA
    )
    assert plan.plan_sha256 == candidate["plan_sha256"]


def test_final_certificate_replays_claimed_intervention_transitions(tmp_path):
    policy, _root, plan, candidate, certificate = _complete_campaign(
        tmp_path,
        claim_interventions=True,
    )

    claims = [
        event
        for event in candidate["campaign_journal"]
        if event["event"] == "ACTION_INTERVENTION_CLAIMED"
    ]
    assert len(claims) == len(plan.cell_ids)
    assert verify_action_calibration_certificate(certificate, policy=policy) == certificate


def test_certificate_rejects_statistic_and_final_attestation_tampering(
    tmp_path,
):
    policy, _root, _plan, _candidate, certificate = _complete_campaign(tmp_path)

    attacked = copy.deepcopy(certificate)
    attacked["candidate"]["cells"]["formalize"]["gain_lcb"] = 1.0
    with pytest.raises(ActionCalibrationError):
        verify_action_calibration_certificate(attacked, policy=policy)

    attacked = copy.deepcopy(certificate)
    attacked["final_verifier_payload"]["accepted"] = False
    with pytest.raises(ActionCalibrationError):
        verify_action_calibration_certificate(attacked, policy=policy)


def test_detached_verifier_reconstructs_outcomes_from_journal(tmp_path):
    policy, _root, _plan, candidate, _certificate = _complete_campaign(tmp_path)
    attacked = copy.deepcopy(candidate)
    for observation in attacked["observations"]:
        observation["treatment_success"] = not observation["treatment_success"]
        observation["control_success"] = not observation["control_success"]
    attacked["cells"] = copy.deepcopy(candidate["cells"])
    body = {name: value for name, value in attacked.items() if name != "candidate_sha256"}
    attacked["candidate_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()

    with pytest.raises(
        ActionCalibrationError,
        match="action_calibration_certificate_observation_binding_invalid",
    ):
        action_calibration_final_verifier_payload(attacked, policy=policy)


def test_candidate_rejects_resource_cap_and_output_seal_shape_tampering(
    tmp_path,
):
    policy, _root, _plan, candidate, _certificate = _complete_campaign(tmp_path)

    attacked = copy.deepcopy(candidate)
    attacked["observations"][0]["treatment_action_resources"]["host_scalar_ops"] = (
        EXECUTION_CONFIG["action_resource_caps"]["host_scalar_ops"] + 1
    )
    body = {name: value for name, value in attacked.items() if name != "candidate_sha256"}
    attacked["candidate_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    with pytest.raises(
        ActionCalibrationError,
        match="action_calibration_action_resource_vector_invalid",
    ):
        action_calibration_final_verifier_payload(attacked, policy=policy)

    attacked = copy.deepcopy(candidate)
    attacked["output_seal_payload"] = []
    body = {name: value for name, value in attacked.items() if name != "candidate_sha256"}
    attacked["candidate_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    with pytest.raises(
        ActionCalibrationError,
        match="action_calibration_candidate_invalid",
    ):
        action_calibration_final_verifier_payload(attacked, policy=policy)

    attacked = copy.deepcopy(candidate)
    attacked["campaign_journal"][1]["payload"]["attempt_number"] = 2
    body = {name: value for name, value in attacked.items() if name != "candidate_sha256"}
    attacked["candidate_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    with pytest.raises(
        ActionCalibrationError,
        match="action_calibration_journal_chain_invalid",
    ):
        action_calibration_final_verifier_payload(attacked, policy=policy)


def test_twenty_unique_pairs_promote_each_certified_action(tmp_path):
    policy, _root, _plan, candidate, certificate = _complete_campaign(
        tmp_path,
        task_count=20,
    )
    assert candidate["pair_count"] == EXPECTED_ACTION_COUNT * 20
    assert all(cell["n"] == 20 and cell["measured"] is True for cell in candidate["cells"].values())
    snapshot = certified_evidence_snapshot(
        certificate,
        policy=policy,
        bucket=CALIBRATION_BUCKET,
    )
    assert all(cell["measured"] for cell in snapshot["cells"].values())


def test_independent_cli_kernel_and_runtime_loader_use_external_root(
    tmp_path,
    monkeypatch,
):
    policy, root, _plan, _candidate, certificate = _complete_campaign(tmp_path)
    certificate_path = tmp_path / "certificate.json"
    policy_path = tmp_path / "policy.json"
    root_path = tmp_path / "trust-root.pem"
    certificate_path.write_bytes(canonical_json_bytes(certificate))
    policy_path.write_bytes(canonical_json_bytes(policy.document))
    root_path.write_bytes(_public_pem(root))

    verdict = verify_files(
        certificate_path=certificate_path,
        policy_path=policy_path,
        trusted_root_path=root_path,
        now_unix=1_000,
        bucket=CALIBRATION_BUCKET,
    )
    assert verdict["accepted"] is True
    assert verdict["frontier_claim_eligible"] is False
    assert verdict["evidence_snapshot"]["schema"].endswith("certified_evidence.v2")
    snapshot = verdict["evidence_snapshot"]

    monkeypatch.delenv(
        "AURA_RLC_ACTION_CALIBRATION_TRUST_ROOT",
        raising=False,
    )
    with pytest.raises(ValueError, match="trust root is not configured"):
        validate_evidence_snapshot(snapshot)

    wrong_root_path = tmp_path / "wrong-trust-root.pem"
    wrong_root_path.write_bytes(_public_pem(Ed25519PrivateKey.generate()))
    monkeypatch.setenv(
        "AURA_RLC_ACTION_CALIBRATION_TRUST_ROOT",
        str(wrong_root_path),
    )
    with pytest.raises(ValueError, match="trust admission failed"):
        validate_evidence_snapshot(snapshot)

    monkeypatch.setenv(
        "AURA_RLC_ACTION_CALIBRATION_CERTIFICATE",
        str(certificate_path),
    )
    monkeypatch.setenv(
        "AURA_RLC_ACTION_CALIBRATION_POLICY",
        str(policy_path),
    )
    monkeypatch.setenv(
        "AURA_RLC_ACTION_CALIBRATION_TRUST_ROOT",
        str(root_path),
    )
    monkeypatch.setattr(
        "core.brain.llm.latent_cortex.execution_controller.time.time",
        lambda: 1_000,
    )
    assert validate_evidence_snapshot(snapshot) == snapshot

    attacked = copy.deepcopy(snapshot)
    attacked["cells"]["formalize"]["cost_mean"] = 0.5
    attacked_body = {name: value for name, value in attacked.items() if name != "snapshot_sha256"}
    attacked["snapshot_sha256"] = hashlib.sha256(canonical_json_bytes(attacked_body)).hexdigest()
    with pytest.raises(ValueError, match="final verdict is invalid"):
        validate_evidence_snapshot(attacked)

    controller = ExecutionController(tmp_path / "runtime-controller")
    status = controller.status()["certified_action_evidence"]
    assert status["admitted"] is True
    assert status["load_error"] is None
    assert status["certificate_sha256"] == certificate["certificate_sha256"]


def test_malformed_optional_artifact_fails_closed_without_crashing_runtime(
    tmp_path,
    monkeypatch,
):
    policy, _role_keys, root = _trust_fixture()
    certificate_path = tmp_path / "malformed-certificate.json"
    policy_path = tmp_path / "policy.json"
    root_path = tmp_path / "trust-root.pem"
    certificate_path.write_text('{"candidate":[]}', encoding="utf-8")
    policy_path.write_bytes(canonical_json_bytes(policy.document))
    root_path.write_bytes(_public_pem(root))

    with pytest.raises(ValueError, match="candidate must be an object"):
        verify_files(
            certificate_path=certificate_path,
            policy_path=policy_path,
            trusted_root_path=root_path,
            now_unix=1_000,
            bucket=None,
        )

    monkeypatch.setenv(
        "AURA_RLC_ACTION_CALIBRATION_CERTIFICATE",
        str(certificate_path),
    )
    monkeypatch.setenv(
        "AURA_RLC_ACTION_CALIBRATION_POLICY",
        str(policy_path),
    )
    monkeypatch.setenv(
        "AURA_RLC_ACTION_CALIBRATION_TRUST_ROOT",
        str(root_path),
    )
    controller = ExecutionController(tmp_path / "malformed-controller")
    status = controller.status()["certified_action_evidence"]
    assert status["admitted"] is False
    assert "candidate must be an object" in status["load_error"]
