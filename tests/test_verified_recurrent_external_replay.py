"""End-to-end external replay for one causally verified recurrent update."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.qwen2 import Model, ModelArgs  # noqa: E402

from core.brain.llm.latent_cortex.campaign_trust import (  # noqa: E402
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    TASK_ISSUER,
    VerifiedCampaignTrustPolicy,
    build_role_attestation,
    policy_signed_payload,
    validate_campaign_trust_policy,
)
from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.learning.recurrence_curriculum import khop_reachability  # noqa: E402
from core.learning.recurrent_grpo import (  # noqa: E402
    RecurrentSamplingConfig,
    recurrent_policy_sample_from_causal_pair,
    recurrent_policy_sha256,
    sample_final_recurrent_transition_pair,
)
from core.learning.verified_recurrent_transition_evidence import (  # noqa: E402
    VerifiedRecurrentTransitionEvidenceError,
)
from core.learning.verified_recurrent_transition_repository import (  # noqa: E402
    VerifiedRecurrentTransitionRepositoryError,
    produce_verified_recurrent_transition_group,
    recurrent_trace_token_decoder,
    recurrent_trace_token_encoder,
    score_verified_recurrent_training_task,
    verify_recurrent_evidence_manifest_artifacts,
)
from core.learning.verified_token_trace import (  # noqa: E402
    build_tokenizer_bundle_identity,
    tokenizer_file_bindings_from_bytes,
)
from core.learning.verified_training_task import (  # noqa: E402
    build_verified_training_task,
)
from core.learning.verified_transition_causal_campaign import (  # noqa: E402
    CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA,
    CausalCampaignScheduleEntry,
    VerifiedTransitionCausalCampaignLedger,
    build_causal_campaign_manifest,
    validate_causal_campaign_evidence_manifest,
)
from core.learning.verified_transition_episode import (  # noqa: E402
    VerifiedTransitionError,
    canonical_json_bytes,
)
from core.learning.verified_transition_group_admission import (  # noqa: E402
    TransitionGroupPlanEntry,
    VerifiedTransitionGroupError,
    build_transition_group_manifest,
    sampling_config_sha256,
)
from core.learning.verified_transition_production_factory import (  # noqa: E402
    ProviderBoundTrainingTask,
)
from core.learning.verified_transition_rejection_transaction import (  # noqa: E402
    VerifiedTransitionRejectionTransactionCoordinator,
    VerifiedTransitionRejectionTransactionError,
    VerifiedTransitionRejectionTransactionStore,
    build_rejected_transaction_trainer_step,
)
from core.learning.verified_transition_reward import (  # noqa: E402
    TransitionRewardConfig,
    VerifiedTransitionRewardAdmissionError,
    VerifiedTransitionRewardError,
)
from core.learning.verified_transition_trainer import (  # noqa: E402
    apply_prepared_verified_transition_group,
    build_verified_transition_step_receipt,
    build_verified_transition_step_static,
)
from core.learning.verified_transition_transaction import (  # noqa: E402
    TrainerCheckpointEvidence,
    VerifiedTransitionTransactionCoordinator,
    VerifiedTransitionTransactionError,
    VerifiedTransitionTransactionStore,
    build_transaction_trainer_step,
)
from core.learning.verified_transition_update import (  # noqa: E402
    VerifiedTransitionUpdateError,
)
from tools.recurrence_native_train_v2 import _wrap_window_layers  # noqa: E402

BASE_SECOND = 1_800_000_000
PROMPT_TOKENS = (5, 9, 17)
REPLAY_FAILURES = (
    VerifiedRecurrentTransitionRepositoryError,
    VerifiedRecurrentTransitionEvidenceError,
    VerifiedTransitionError,
    VerifiedTransitionGroupError,
    VerifiedTransitionRewardAdmissionError,
    VerifiedTransitionRewardError,
    VerifiedTransitionUpdateError,
    VerifiedTransitionTransactionError,
    VerifiedTransitionRejectionTransactionError,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return payload + (b"\n" if newline else b"")


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


def _role_pin(role: str, key: Ed25519PrivateKey) -> dict[str, str]:
    public = _public_raw(key)
    return {
        "signer_id": f"{role}-external-replay-signer",
        "organization_id": f"{role}-external-replay-custodian",
        "public_key_b64": base64.b64encode(public).decode("ascii"),
        "key_id": hashlib.sha256(public).hexdigest(),
        "implementation_sha256": _sha(f"{role}-implementation"),
        "release_sha256": _sha(f"{role}-release"),
        "custody_class": "external_service",
        "custody_evidence_sha256": _sha(f"{role}-custody"),
    }


def _trust_material() -> tuple[
    VerifiedCampaignTrustPolicy,
    dict[str, Ed25519PrivateKey],
]:
    root = Ed25519PrivateKey.generate()
    role_keys = {
        role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES
    }
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "recurrent-external-replay-2026-07",
        "policy_revision": 1,
        "campaign_name": "recurrent-external-replay",
        "protocol_sha256": _sha("recurrent-external-replay-protocol"),
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": BASE_SECOND,
        "not_before_unix": BASE_SECOND + 100,
        "expires_at_unix": BASE_SECOND + 10_000,
        "roles": {
            role: _role_pin(role, role_keys[role])
            for role in CAMPAIGN_TRUST_ROLES
        },
    }
    signed = canonical_json_bytes(body)
    root_raw = _public_raw(root)
    document = {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode(
                "ascii"
            ),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }
    assert policy_signed_payload(document) == body
    policy = validate_campaign_trust_policy(
        document,
        trusted_root_public_key_pem=_public_pem(root),
        expected_campaign_name="recurrent-external-replay",
        expected_protocol_sha256=_sha(
            "recurrent-external-replay-protocol"
        ),
        now_unix=BASE_SECOND + 120,
    )
    return policy, role_keys


def _prepared_model() -> Model:
    mx.random.seed(929)
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=16,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=2,
            max_position_embeddings=64,
            rope_theta=10000.0,
        )
    )
    mx.eval(model.parameters())
    assert _wrap_window_layers(
        model,
        rank=2,
        targets=("o_proj",),
        prelude_frac=0.25,
        coda_frac=0.25,
    )
    return model


def _execution_spec() -> RLCExecutionSpec:
    return RLCExecutionSpec(
        n_slots=2,
        branch_roles=("constructive_solution", "critical_audit"),
        exchange_interval=1,
        recurrent_steps=2,
        alpha=0.35,
        prelude_frac=0.25,
        coda_frac=0.25,
    )


def _checkpoint_evidence(
    transaction: Any,
    trainer_step: dict[str, Any],
    *,
    execution_spec_sha256: str,
) -> TrainerCheckpointEvidence:
    document = {
        "schema": "aura.grpo_checkpoint.v2",
        "checkpoint_id": "step-00000001-" + "1" * 32,
        "created_unix": float(BASE_SECOND + 164),
        "protocol_sha256": _sha("checkpoint-protocol"),
        "dataset_sha256": _sha("checkpoint-dataset"),
        "step": 1,
        "curriculum": {},
        "telemetry": {},
        "last_step_committed": True,
        "history": [{"step": 0, "overall": 0.25}],
        "baseline_eval": None,
        "calibration": None,
        "elapsed_training_s": 1.0,
        "invocation_count": 1,
        "rng_strategy": "stateless_sha256_step_seeded_v1",
        "optimizer_updates": 1,
        "last_step_kind": "verified_optimizer_update",
        "execution_mode": "recurrent",
        "execution_spec_sha256": execution_spec_sha256,
        "step_receipts": [trainer_step],
        "adapter": {
            "path": "adapter.safetensors",
            "sha256": transaction.stage["adapter"]["sha256"],
            "size_bytes": transaction.stage["adapter"]["size_bytes"],
        },
        "optimizer": {
            "path": "optimizer.safetensors",
            "sha256": transaction.stage["optimizer"]["sha256"],
            "size_bytes": transaction.stage["optimizer"]["size_bytes"],
        },
    }
    return TrainerCheckpointEvidence(
        document=document,
        artifact_sha256=hashlib.sha256(
            _canonical(document, newline=True)
        ).hexdigest(),
    )


def _rejection_checkpoint_directory(
    root: Path,
    trainer_step: Mapping[str, Any],
    *,
    execution_spec_sha256: str,
) -> Path:
    directory = root / ("step-00000001-" + "2" * 32)
    directory.mkdir(mode=0o700)
    document = {
        "schema": "aura.grpo_checkpoint.v2",
        "checkpoint_id": directory.name,
        "created_unix": float(BASE_SECOND + 164),
        "protocol_sha256": _sha("checkpoint-protocol"),
        "dataset_sha256": _sha("checkpoint-dataset"),
        "step": 1,
        "curriculum": {},
        "telemetry": {},
        "last_step_committed": True,
        "history": [{"step": 0, "overall": 1.0}],
        "baseline_eval": None,
        "calibration": None,
        "elapsed_training_s": 1.0,
        "invocation_count": 1,
        "rng_strategy": "stateless_sha256_step_seeded_v1",
        "optimizer_updates": 0,
        "last_step_kind": "verified_rejected_group",
        "execution_mode": "recurrent",
        "execution_spec_sha256": execution_spec_sha256,
        "step_receipts": [dict(trainer_step)],
        "adapter": {
            "path": "adapter.safetensors",
            "sha256": _sha("rejected-adapter"),
            "size_bytes": 1,
        },
        "optimizer": {
            "path": "optimizer.safetensors",
            "sha256": _sha("rejected-optimizer"),
            "size_bytes": 1,
        },
    }
    (directory / "complete.json").write_bytes(
        _canonical(document, newline=True)
    )
    (directory / "complete.json").chmod(0o600)
    return directory


def _build_replay(root: Path, *, admitted: bool) -> dict[str, Any]:
    root.chmod(0o700)
    model = _prepared_model()
    spec = _execution_spec()
    sampling = RecurrentSamplingConfig(max_tokens=2)
    samples = tuple(
        recurrent_policy_sample_from_causal_pair(
            sample_final_recurrent_transition_pair(
                model,
                PROMPT_TOKENS,
                spec=spec,
                branch_index=0,
                seed=seed,
                sampling=sampling,
                episode_id=f"updated-{seed}",
            )
        )
        for seed in (9, 1)
    )
    assert tuple(sample.tokens for sample in samples) == (
        (21, 3),
        (9, 11),
    )

    source_task = khop_reachability(1, 929)
    public_task, _sealed_task = build_verified_training_task(
        source_task,
        answer_nonce=b"external-replay-answer-nonce-32b",
    )
    task_commitment = public_task.to_dict()
    task = ProviderBoundTrainingTask(source_task, task_commitment)
    bundle = build_tokenizer_bundle_identity(
        tokenizer_class="test.RecurrentExternalReplayTokenizer",
        tokenizer_files=tokenizer_file_bindings_from_bytes(
            {
                "tokenizer.json": b'{"kind":"external-replay"}',
                "tokenizer_config.json": b'{"streaming":"prefix"}',
            }
        ),
        chat_template=None,
        special_token_map={},
        encode_options={},
        decode_options={},
        implementation_source_sha256=_sha(
            "external-replay-tokenizer-implementation"
        ),
    )

    class Adapter:
        bundle_identity = bundle

        @staticmethod
        def encode_prompt(text: str) -> tuple[int, ...]:
            assert text == source_task.prompt
            return PROMPT_TOKENS

        @staticmethod
        def decode_output(tokens: Any) -> str:
            token_ids = tuple(tokens)
            if len(token_ids) == 1:
                return 'FINAL_ANSWER: {"node":'
            if len(token_ids) != 2:
                raise ValueError("unexpected test token sequence")
            node = 4 if admitted and token_ids[0] == 7 else 5
            return f'FINAL_ANSWER: {{"node":{node}}}'

        @classmethod
        def stream_decode_deltas(cls, tokens: Any) -> tuple[str, ...]:
            token_ids = tuple(tokens)
            rendered = tuple(
                cls.decode_output(token_ids[:index])
                for index in range(1, len(token_ids) + 1)
            )
            return tuple(
                value
                if index == 0
                else value[len(rendered[index - 1]) :]
                for index, value in enumerate(rendered)
            )

    policy, role_keys = _trust_material()
    contract_sha256 = _sha("external-replay-provider-contract")
    schedule_root_sha256 = _sha("external-replay-schedule-root")
    initial_policy_sha256 = recurrent_policy_sha256(model, spec)
    assert all(
        sample.policy_sha256 == initial_policy_sha256
        for sample in samples
    )
    campaign_planned_second = BASE_SECOND + 150
    campaign_manifest = build_causal_campaign_manifest(
        campaign_id="recurrent-external-replay-campaign",
        provider_contract_sha256=contract_sha256,
        campaign_schedule_root_sha256=schedule_root_sha256,
        trust_policy_sha256=policy.policy_sha256,
        initial_policy_sha256=initial_policy_sha256,
        schedule=(
            CausalCampaignScheduleEntry(
                sequence=0,
                task_id=task.task_id,
                task_commitment_sha256=public_task.task_commitment_sha256,
            ),
        ),
        planned_at_unix_ns=campaign_planned_second * 1_000_000_000,
    )
    campaign_attestation = build_role_attestation(
        policy,
        role=TASK_ISSUER,
        payload=campaign_manifest,
        signed_at_unix=campaign_planned_second,
        private_key=role_keys[TASK_ISSUER],
    )
    campaign_root = (root / "campaign").resolve()
    campaign_ledger = VerifiedTransitionCausalCampaignLedger.create(
        campaign_root,
        campaign_manifest=campaign_manifest,
        campaign_manifest_attestation=campaign_attestation,
        policy=policy,
    )

    group_planned_second = BASE_SECOND + 160
    reward_config_sha256 = hashlib.sha256(
        canonical_json_bytes(TransitionRewardConfig().to_dict())
    ).hexdigest()
    group_manifest = build_transition_group_manifest(
        group_id="recurrent-external-replay-group-0",
        task_id=task.task_id,
        entries=tuple(
            TransitionGroupPlanEntry(
                episode_id=sample.episode_id,
                task_id=task.task_id,
                rng_root_sha256=sample.rng_root_sha256,
                policy_sha256=sample.policy_sha256,
                recurrent_execution_spec_sha256=(
                    sample.execution_spec_sha256
                ),
                producing_branch_index=sample.branch_index,
                sample_seed=sample.seed,
                sampling_config_sha256=sampling_config_sha256(sample),
            )
            for sample in samples
        ),
        reward_config_sha256=reward_config_sha256,
        planned_at_unix_ns=group_planned_second * 1_000_000_000,
    )
    lineage = {
        "schema": "aura.verified_transition.lineage_plan.v1",
        "contract_sha256": contract_sha256,
        "campaign_id": campaign_manifest["campaign_id"],
        "campaign_schedule_root_sha256": schedule_root_sha256,
        "sequence": 0,
        "task_commitment_sha256": public_task.task_commitment_sha256,
        "policy_before_sha256": initial_policy_sha256,
        "group_manifest_sha256": group_manifest["manifest_sha256"],
    }
    group_attestation = build_role_attestation(
        policy,
        role=TASK_ISSUER,
        payload=group_manifest,
        signed_at_unix=group_planned_second,
        private_key=role_keys[TASK_ISSUER],
    )
    lineage_attestation = build_role_attestation(
        policy,
        role=TASK_ISSUER,
        payload=lineage,
        signed_at_unix=group_planned_second,
        private_key=role_keys[TASK_ISSUER],
    )
    campaign_ledger.admit_group_plan(
        sequence=0,
        campaign_id=campaign_manifest["campaign_id"],
        campaign_schedule_root_sha256=schedule_root_sha256,
        policy_before_sha256=initial_policy_sha256,
        group_manifest=group_manifest,
        group_manifest_attestation=group_attestation,
        lineage_plan=lineage,
        lineage_attestation=lineage_attestation,
        policy=policy,
        admitted_at_unix_ns=(group_planned_second + 1)
        * 1_000_000_000,
    )

    roots = {
        name: str((root / name).resolve())
        for name in (
            "transition_artifacts",
            "updates",
            "replay_artifacts",
            "transactions",
        )
    }
    request = SimpleNamespace(
        schema="aura.verified_transition.production_request.v2",
        contract_sha256=contract_sha256,
        campaign_schedule_root_sha256=schedule_root_sha256,
        sequence=0,
        task=task,
        prompt_text=source_task.prompt,
        prompt_tokens=PROMPT_TOKENS,
        samples=samples,
        completions=tuple(
            Adapter.decode_output(sample.tokens) for sample in samples
        ),
        group_manifest=group_manifest,
        group_manifest_attestation=group_attestation,
        provider_config={},
        ledger_roots={
            "campaign": str(campaign_root),
            **roots,
        },
        campaign_ledger=campaign_ledger,
        campaign_trust_policy=policy,
        tokenizer_bundle_sha256=bundle["bundle_sha256"],
        tokenizer_trace_adapter=Adapter(),
        independent_scorer=score_verified_recurrent_training_task,
        token_encoder=recurrent_trace_token_encoder,
        token_decoder=recurrent_trace_token_decoder,
    )
    prepared = produce_verified_recurrent_transition_group(request)
    assert prepared.reward_receipt["optimizer_admitted"] is admitted
    transition_kinds = [
        transition["transition_kind"]
        for transition in prepared.reward_receipt["transitions"]
    ]
    assert transition_kinds == (
        ["wrong_to_right", "right_to_right"]
        if admitted
        else ["right_to_right", "right_to_right"]
    )
    assert (prepared.group_admission_receipt is not None) is admitted
    assert (prepared.update_journal is not None) is admitted

    answer_channel = {
        "correct_fraction": 1.0,
        "completion_count": len(samples),
    }
    trainer_step_static = build_verified_transition_step_static(
        samples=samples,
        reward_receipt=prepared.reward_receipt,
        answer_channel=answer_channel,
    )
    optimizer = optim.Adam(learning_rate=0.01)
    if admitted:
        transaction_store = VerifiedTransitionTransactionStore.open(
            roots["transactions"]
        )
        transaction_coordinator = VerifiedTransitionTransactionCoordinator(
            store=transaction_store,
            sequence=0,
            trainer_step=1,
            task_id=task.task_id,
            trainer_sample_seed=929,
            execution_spec_sha256=spec.sha256,
            campaign_manifest_sha256=campaign_manifest["manifest_sha256"],
            campaign_schedule_root_sha256=schedule_root_sha256,
            group_manifest_sha256=group_manifest["manifest_sha256"],
            reward_receipt_sha256=prepared.reward_receipt[
                "receipt_sha256"
            ],
            trainer_step_static=trainer_step_static,
            adapter_tensors=lambda: dict(
                tree_flatten(model.trainable_parameters())
            ),
            optimizer_tensors=lambda: dict(tree_flatten(optimizer.state)),
        )
        update_times = iter(
            (
                (BASE_SECOND + 162) * 1_000_000_000,
                (BASE_SECOND + 163) * 1_000_000_000,
            )
        )
        mutation = apply_prepared_verified_transition_group(
            model,
            optimizer,
            PROMPT_TOKENS,
            samples,
            prepared,
            spec=spec,
            now_unix_ns=lambda: next(update_times),
            transaction_coordinator=transaction_coordinator,
        )
        assert mutation.optimizer_updated is True
        assert mutation.policy_before_sha256 == initial_policy_sha256
        assert mutation.policy_after_sha256 != initial_policy_sha256
        assert mutation.replay_group is not None
        trainer_step = build_verified_transition_step_receipt(
            step_number=1,
            task_id=task.task_id,
            sample_seed=929,
            execution_spec_sha256=spec.sha256,
            samples=samples,
            answer_channel=answer_channel,
            mutation=mutation,
        )
        assert prepared.group_admission_receipt is not None
        admission_sha256 = prepared.group_admission_receipt[
            "receipt_sha256"
        ]
        transaction = transaction_store.load(
            sequence=0,
            admission_sha256=admission_sha256,
            load_tensors=True,
        )
        assert transaction is not None
        assert build_transaction_trainer_step(transaction) == trainer_step
        transaction_store.record_trainer_checkpoint(
            sequence=0,
            admission_sha256=admission_sha256,
            checkpoint=_checkpoint_evidence(
                transaction,
                trainer_step,
                execution_spec_sha256=spec.sha256,
            ),
        )
        transaction = transaction_store.load(
            sequence=0,
            admission_sha256=admission_sha256,
            load_tensors=True,
        )
        assert transaction is not None
        assert tuple(event["kind"] for event in transaction.events) == (
            "update_commit",
            "campaign_terminal",
            "trainer_checkpoint",
        )
        assert transaction.adapter_tensors
        assert transaction.optimizer_tensors
        rejection = None
    else:
        rejection_store = VerifiedTransitionRejectionTransactionStore.open(
            roots["transactions"]
        )
        rejection_coordinator = (
            VerifiedTransitionRejectionTransactionCoordinator(
                store=rejection_store,
                sequence=0,
                trainer_step=1,
                task_id=task.task_id,
                trainer_sample_seed=929,
                execution_spec_sha256=spec.sha256,
                campaign_manifest_sha256=campaign_manifest[
                    "manifest_sha256"
                ],
                campaign_schedule_root_sha256=schedule_root_sha256,
                group_manifest_sha256=group_manifest["manifest_sha256"],
                reward_receipt_sha256=prepared.reward_receipt[
                    "receipt_sha256"
                ],
                trainer_step_static=trainer_step_static,
            )
        )
        mutation = apply_prepared_verified_transition_group(
            model,
            optimizer,
            PROMPT_TOKENS,
            samples,
            prepared,
            spec=spec,
            now_unix_ns=lambda: (BASE_SECOND + 163) * 1_000_000_000,
            rejection_transaction_coordinator=rejection_coordinator,
        )
        assert mutation.optimizer_updated is False
        assert mutation.policy_before_sha256 == initial_policy_sha256
        assert mutation.policy_after_sha256 == initial_policy_sha256
        assert mutation.replay_group is None
        trainer_step = build_verified_transition_step_receipt(
            step_number=1,
            task_id=task.task_id,
            sample_seed=929,
            execution_spec_sha256=spec.sha256,
            samples=samples,
            answer_channel=answer_channel,
            mutation=mutation,
        )
        rejection = rejection_store.load(
            sequence=0,
            reward_sha256=prepared.reward_receipt["receipt_sha256"],
        )
        assert rejection is not None
        assert build_rejected_transaction_trainer_step(
            rejection
        ) == trainer_step
        rejection_store.record_trainer_checkpoint(
            sequence=0,
            reward_sha256=prepared.reward_receipt["receipt_sha256"],
            checkpoint_dir=_rejection_checkpoint_directory(
                root,
                trainer_step,
                execution_spec_sha256=spec.sha256,
            ),
        )
        rejection = rejection_store.load(
            sequence=0,
            reward_sha256=prepared.reward_receipt["receipt_sha256"],
        )
        assert rejection is not None
        assert tuple(event["kind"] for event in rejection.events) == (
            "campaign_terminal",
            "trainer_checkpoint",
        )
        transaction = None
        admission_sha256 = None

    package_path = (
        Path(roots["replay_artifacts"])
        / "group-00000000.prepared.json"
    ).resolve(strict=True)
    package_bytes = package_path.read_bytes()
    package = json.loads(package_bytes)
    package_row = {
        "sequence": 0,
        "status": "updated" if admitted else "rejected",
        "package_artifact": {
            "path": str(package_path),
            "sha256": hashlib.sha256(package_bytes).hexdigest(),
            "size_bytes": len(package_bytes),
        },
        "package_receipt_sha256": package["receipt_sha256"],
        "group_manifest_sha256": group_manifest["manifest_sha256"],
        "reward_receipt_sha256": prepared.reward_receipt[
            "receipt_sha256"
        ],
        "group_admission_sha256": admission_sha256,
        "update_receipt_sha256": mutation.update_receipt_sha256,
        "trainer_step_receipt_sha256": trainer_step["receipt_sha256"],
        "sample_receipt_sha256s": package["sample_receipt_sha256s"],
        "evidence_receipt_sha256s": package[
            "evidence_receipt_sha256s"
        ],
    }
    evidence_body = {
        "schema": CAUSAL_CAMPAIGN_EVIDENCE_MANIFEST_SCHEMA,
        "contract_sha256": contract_sha256,
        "campaign_schedule_root_sha256": schedule_root_sha256,
        "trust_policy_sha256": policy.policy_sha256,
        "campaign_ledger_root": str(campaign_root),
        "transition_artifact_root": roots["transition_artifacts"],
        "update_journal_root": roots["updates"],
        "transaction_root": roots["transactions"],
        "completed_groups": 1,
        "halt_reason": "max_steps",
        "group_packages": [package_row],
        "updated_replay_sequences": [0] if admitted else [],
        "created_at_unix_ns": (BASE_SECOND + 165) * 1_000_000_000,
    }
    evidence_manifest = validate_causal_campaign_evidence_manifest(
        {
            **evidence_body,
            "manifest_sha256": hashlib.sha256(
                canonical_json_bytes(evidence_body)
            ).hexdigest(),
        }
    )
    transition_blob_root = (
        Path(roots["transition_artifacts"]) / "blobs"
    )
    prepared.transition_store.close()
    paths = {
        "evidence": transition_blob_root
        / package["evidence_artifacts"][0]["payload_sha256"],
        "reward": transition_blob_root
        / package["reward_artifact"]["payload_sha256"],
    }
    if admitted:
        assert admission_sha256 is not None
        assert transaction is not None
        assert package["group_admission_artifact"] is not None
        paths.update(
            {
                "admission": transition_blob_root
                / package["group_admission_artifact"]["payload_sha256"],
                "update_journal": Path(roots["updates"])
                / f"{admission_sha256}.committed.json",
                "transaction_event": (
                    transaction.transaction_dir
                    / "generations"
                    / "00000003-trainer-checkpoint"
                    / "evidence.json"
                ),
                "adapter_tensor": (
                    transaction.transaction_dir
                    / "generations"
                    / "00000000-staged"
                    / "adapter.safetensors"
                ),
            }
        )
    else:
        assert rejection is not None
        paths["rejection_event"] = (
            rejection.transaction_dir / "00000002-trainer-checkpoint.json"
        )
    return {
        "manifest": evidence_manifest,
        "policy": policy,
        "paths": paths,
    }


@pytest.fixture(scope="module")
def updated_replay(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _build_replay(
        tmp_path_factory.mktemp("verified-recurrent-external-update"),
        admitted=True,
    )


@pytest.fixture(scope="module")
def rejected_replay(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _build_replay(
        tmp_path_factory.mktemp("verified-recurrent-external-rejection"),
        admitted=False,
    )


def _verify(material: dict[str, Any]) -> dict[str, Any]:
    return verify_recurrent_evidence_manifest_artifacts(
        material["manifest"],
        campaign_trust_policy=material["policy"],
        verifier_identity="independent-recurrent-replay-verifier",
        verified_at_unix=BASE_SECOND + 200,
    )


def _replace_with_corrupt_bytes(path: Path) -> tuple[bytes, int]:
    original = path.read_bytes()
    original_mode = stat.S_IMODE(path.stat().st_mode)
    attacked = bytearray(original)
    attacked[len(attacked) // 2] ^= 0x01
    os.chmod(path, 0o600)
    path.write_bytes(attacked)
    os.chmod(path, original_mode)
    return original, original_mode


def _restore_bytes(path: Path, payload: bytes, mode: int) -> None:
    os.chmod(path, 0o600)
    path.write_bytes(payload)
    os.chmod(path, mode)


def test_updated_group_full_external_replay_is_green(
    updated_replay: dict[str, Any],
) -> None:
    receipt = _verify(updated_replay)

    assert receipt["verified_package_count"] == 1
    assert receipt["validation_profile"] == (
        "recurrent_transition_causal_replay.v2"
    )
    assert receipt["evidence_manifest_sha256"] == (
        updated_replay["manifest"]["manifest_sha256"]
    )


@pytest.mark.parametrize(
    "artifact_role",
    (
        "evidence",
        "reward",
        "admission",
        "update_journal",
        "transaction_event",
        "adapter_tensor",
    ),
)
def test_updated_group_external_replay_fails_closed_on_corruption(
    updated_replay: dict[str, Any],
    artifact_role: str,
) -> None:
    path = updated_replay["paths"][artifact_role]
    original, mode = _replace_with_corrupt_bytes(path)
    try:
        with pytest.raises(REPLAY_FAILURES):
            _verify(updated_replay)
    finally:
        _restore_bytes(path, original, mode)

    assert _verify(updated_replay)["verified_package_count"] == 1


def test_rejected_group_full_external_replay_is_green(
    rejected_replay: dict[str, Any],
) -> None:
    receipt = _verify(rejected_replay)

    assert receipt["verified_package_count"] == 1
    assert receipt["validation_profile"] == (
        "recurrent_transition_causal_replay.v2"
    )
    assert receipt["evidence_manifest_sha256"] == (
        rejected_replay["manifest"]["manifest_sha256"]
    )


def test_rejected_group_external_replay_fails_closed_on_chain_corruption(
    rejected_replay: dict[str, Any],
) -> None:
    path = rejected_replay["paths"]["rejection_event"]
    original, mode = _replace_with_corrupt_bytes(path)
    try:
        with pytest.raises(REPLAY_FAILURES):
            _verify(rejected_replay)
    finally:
        _restore_bytes(path, original, mode)

    assert _verify(rejected_replay)["verified_package_count"] == 1
