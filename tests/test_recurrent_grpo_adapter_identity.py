"""Identity, tamper, and freeze contracts for recurrent-GRPO adapters."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.brain.llm.latent_cortex import (
    recurrent_grpo_adapter_identity as identity_runtime,
)
from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_launch_bundle import (
    CampaignLaunchBundleError,
    build_adapter_freeze_certificate,
    verify_adapter_freeze,
)
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
    IDENTITY_RECEIPT_SCHEMA,
    MANIFEST_FILE,
    VERIFIED_IDENTITY_RECEIPT_SCHEMA,
    RecurrentGRPOAdapterIdentityError,
    recurrent_policy_sha256_from_safetensors,
    sha256_bytes,
    validate_recurrent_grpo_adapter_identity_with_verified_transitions,
    validate_verified_recurrent_grpo_adapter_identity_receipt,
)
from core.learning import (
    verified_transition_training_evidence as transition_evidence_runtime,
)
from core.learning.grpo import GRPOConfig, group_advantages
from core.learning.grpo_training_state import canonical_json_bytes as training_json_bytes
from core.learning.recurrent_grpo import recurrent_policy_sha256
from core.learning.verified_transition_trainer import (
    VerifiedTransitionMutationResult,
    build_verified_transition_step_receipt,
)
from tests.fixtures.rlc_runtime_integrity import engine_runtime_integrity
from tools import prepare_latent_cortex_campaign as preparation
from tools.train_grpo import (
    GRPO_DATASET_SCHEMA,
    GRPO_PROTOCOL_SCHEMA,
    GRPO_TRAIN_SCHEMA,
    _advantage_report_with_verifier_rate,
    _build_recurrent_step_receipt,
    _publish_recurrent_adapter_bundle,
    _shape_recurrent_rewards_from_ce_trails,
    _validate_published_recurrent_bundle,
)

SOURCE_ROLES = {
    "trainer",
    "grpo",
    "curriculum",
    "tasks",
    "checkpoint",
    "artifact_schema",
    "adapter",
    "recurrent_grpo",
    "recurrent_objective",
    "execution_spec",
    "latent_engine",
    "recurrence",
    "verified_trainer",
    "transition_campaign",
    "transition_episode",
    "transition_reward",
    "transition_admission",
    "transition_update",
    "transition_training_evidence",
    "campaign_trust",
    "transition_provider",
    "transition_provider_factory",
    "transition_launch_bundle",
    "transition_transaction",
    "transition_rejection_transaction",
    "transition_causal_campaign",
    "verified_training_task",
    "verified_token_trace",
}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _verified_training_evidence(base_identity: dict) -> dict:
    material = {
        "schema": "aura.verified_transition.training_evidence.v2",
        "campaign_receipt_sha256": "a" * 64,
        "campaign_manifest_sha256": "b" * 64,
        "updated_sequences": [0],
        "reward_receipt_sha256s": ["7" * 64],
        "group_admission_sha256s": ["8" * 64],
        "update_receipt_sha256s": ["9" * 64],
        "objective_receipt_sha256s": ["e" * 64],
        "optimizer_update_count": 1,
        "initial_policy_sha256": "4" * 64,
        "final_policy_sha256": base_identity["final_policy_sha256"],
        "source_artifacts_replayed": True,
        "legacy_scalar_reward_path_used": False,
    }
    return {
        **material,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(material)),
    }


def _fixture(
    tmp_path: Path,
    *,
    mutate_receipt=None,
    trajectory_credit: bool = False,
    verified_step: bool = False,
) -> dict:
    mx = pytest.importorskip("mlx.core")
    out = tmp_path / "training"
    out.mkdir()
    adapter_id = "recurrent-grpo-test"
    spec = RLCExecutionSpec(
        n_slots=2,
        branch_roles=("constructive_solution",),
        recurrent_steps=2,
        adaptive_halting=False,
        latent_opt_mode="disabled",
        fast_weights_mode="disabled",
        decode_bridge_policy="none",
    )
    projection = "model.layers.1.self_attn.o_proj"
    mx.save_safetensors(
        str(out / "grpo_adapters.safetensors"),
        {
            f"{projection}.lora_a": mx.zeros((4, 2)),
            f"{projection}.lora_b": mx.ones((2, 4)),
        },
    )
    frozen_policy_sha256 = recurrent_policy_sha256_from_safetensors(
        (out / "grpo_adapters.safetensors").read_bytes(),
        execution_spec_sha256=spec.sha256,
    )
    sources = {}
    source_paths = {}
    snapshot_dir = out / "source_snapshots"
    snapshot_dir.mkdir()
    for role in sorted(SOURCE_ROLES):
        payload = f"# frozen {role}\n".encode("ascii")
        path = snapshot_dir / f"{role}.py"
        path.write_bytes(payload)
        source_paths[role] = path
        sources[role] = {
            "path": f"source/{role}.py",
            "sha256": _sha(payload),
            "size_bytes": len(payload),
        }
    base = {"method": "sha256", "fingerprint": "1" * 64, "files": 1}
    behavior = {"bundle_sha256": "2" * 64, "file_count": 1, "files": []}
    personality = {
        "present": False,
        "bundle_sha256": "",
        "file_count": 0,
        "files": [],
    }
    runtime = {
        "python": "3.12.0",
        "dependencies": {},
        "platform_system": "Darwin",
        "platform_release": "test",
        "platform_machine": "arm64",
        "identity_sha256": "3" * 64,
    }
    dataset = {
        "schema": GRPO_DATASET_SCHEMA,
        "seed": 7,
        "train": [{"task_id": "train-1"}],
        "holdout": [{"task_id": "holdout-1"}],
    }
    dataset_bytes = training_json_bytes(dataset)
    training = {
        "execution_mode": "recurrent",
        "execution_spec": spec.to_dict(),
        "execution_spec_sha256": spec.sha256,
        "verified_transition_provider_contract_sha256": "f" * 64,
        "domains": ["logic"],
        "depths": [2],
        "train_per_cell": 1,
        "holdout_per_cell": 1,
        "group_size": 2,
        "temperature": 1.0,
        "max_tokens": 32,
        "kl_coefficient": 0.02,
        "format_credit": 0.0,
        "trajectory_credit": trajectory_credit,
        "trajectory_shaping_weight": 0.25,
        "lora_rank": 2,
        "lora_targets": "o_proj",
        "lora_layers": 1,
        "lora_initialization_seed": 71,
        "learning_rate": 1e-5,
        "max_steps": 1,
        "eval_every": 1,
        "checkpoint_every": 1,
        "min_signal_groups": 8,
        "calibrate": False,
        "calibrate_samples": 1,
        "calibrate_group": 2,
        "calibrate_tokens": 0,
        "calibrate_minutes": 1.0,
        "cot": True,
        "seed": 7,
        "memory_fraction": 0.4,
        "rng_strategy": "stateless_sha256_step_seeded_v1",
    }
    protocol = {
        "schema": GRPO_PROTOCOL_SCHEMA,
        "adapter_id": adapter_id,
        "model_path": "/model",
        "base_checkpoint": base,
        "model_behavior": behavior,
        "personality_adapter": personality,
        "runtime": runtime,
        "dataset_sha256": _sha(dataset_bytes),
        "sources": sources,
        "training": training,
    }
    protocol_bytes = training_json_bytes(protocol)
    prompt_tokens_sha256 = "6" * 64
    sample = {
        "schema": "aura.recurrent_sampling_behavior.v4",
        "episode_id": "adapter-identity-sample",
        "behavior_admitted": True,
        "execution_spec_sha256": spec.sha256,
        "prompt_tokens_sha256": prompt_tokens_sha256,
        "cached_params_unchanged": True,
        "cached_runtime_integrity": engine_runtime_integrity(
            episode_id="adapter-identity-sample",
            input_tokens_sha256=prompt_tokens_sha256,
            checkpoint_required=False,
        ),
        "cached_nonparametric_memory_status": "disabled_by_policy",
        "cached_recurrence_adapter": {
            "schema": "aura.recurrence_adapter_activation.v1",
            "active": True,
            "scope": "latent_slots_only",
            "calls": 1,
            "adapted_positions": 2,
            "observed_positions": 2,
        },
        "policy_sha256": "4" * 64,
    }
    verifier_rewards = [0.0, 0.0] if trajectory_credit else [1.0, 0.0]
    verifier_advantage = group_advantages(verifier_rewards)
    trajectory = (
        _shape_recurrent_rewards_from_ce_trails(
            verifier_rewards,
            [[2.0, 1.0, 0.5], [0.5, 1.0, 2.0]],
            shaping_weight=0.25,
        )
        if trajectory_credit
        else None
    )
    effective_rewards = (
        list(trajectory["shaped_rewards"]) if trajectory is not None else list(verifier_rewards)
    )
    advantage = (
        _advantage_report_with_verifier_rate(
            group_advantages(effective_rewards),
            verifier_advantage,
        )
        if trajectory is not None
        else verifier_advantage
    )
    answer_channel = {
        "completions": 2,
        "parseable": 2,
        "unparseable": 0,
        "correct": 0 if trajectory_credit else 1,
        "parseable_fraction": 1.0,
        "correct_fraction": 0.0 if trajectory_credit else 0.5,
        "grade_reasons": (
            {"incorrect": 2} if trajectory_credit else {"correct": 1, "incorrect": 1}
        ),
    }
    if verified_step:
        group_manifest_sha256 = "6" * 64
        group_admission_sha256 = "8" * 64
        update_receipt_sha256 = "9" * 64

        class Sample:
            @staticmethod
            def receipt():
                return dict(sample)

        step_receipt = build_verified_transition_step_receipt(
            step_number=1,
            task_id="train-1",
            sample_seed=41,
            execution_spec_sha256=spec.sha256,
            samples=(Sample(), Sample()),
            answer_channel=answer_channel,
            mutation=VerifiedTransitionMutationResult(
                campaign_sequence=0,
                group_manifest_sha256=group_manifest_sha256,
                optimizer_updated=True,
                structured_rewards=(1.1, -0.1),
                optimizer_admission_reason="admitted",
                reward_receipt_sha256="7" * 64,
                group_admission_sha256=group_admission_sha256,
                update_receipt_sha256=update_receipt_sha256,
                update_receipt={
                    "schema": "aura.verified_transition.update_receipt.v1",
                    "optimizer_update_count": 1,
                    "group_admission_sha256": group_admission_sha256,
                    "policy_before_sha256": "4" * 64,
                    "policy_after_sha256": frozen_policy_sha256,
                    "receipt_sha256": update_receipt_sha256,
                },
                terminal_receipt={
                    "schema": "aura.verified_transition.campaign_group_terminal.v2",
                    "sequence": 0,
                    "group_manifest_sha256": group_manifest_sha256,
                    "status": "updated",
                    "reward_receipt_sha256": "7" * 64,
                    "group_admission_sha256": group_admission_sha256,
                    "update_receipt_sha256": update_receipt_sha256,
                    "receipt_sha256": "a" * 64,
                },
                policy_before_sha256="4" * 64,
                policy_after_sha256=frozen_policy_sha256,
                replay_group=object(),
            ),
        )
    else:
        step_receipt = _build_recurrent_step_receipt(
            step_number=1,
            task_id="train-1",
            sample_seed=41,
            execution_spec_sha256=spec.sha256,
            samples=[dict(sample), dict(sample)],
            effective_rewards=effective_rewards,
            verifier_rewards=verifier_rewards,
            answer_channel=answer_channel,
            verifier_advantage_report=verifier_advantage,
            trajectory_credit=trajectory,
            advantage_report=advantage,
            step_kind="optimizer_update",
            update={"schema": "aura.recurrent_grpo.v1", "has_gradient": True},
            policy_after_sha256=frozen_policy_sha256,
            trajectory_credit_enabled=trajectory_credit,
            trajectory_shaping_weight=0.25,
            advantage_clip=4.0,
        )
    receipt = {
        "schema": GRPO_TRAIN_SCHEMA,
        "adapter_id": adapter_id,
        "protocol_sha256": _sha(protocol_bytes),
        "dataset_sha256": _sha(dataset_bytes),
        "model": {"path": "/model", "base_checkpoint": base, "behavior": behavior},
        "config": GRPOConfig(group_size=2, kl_coefficient=0.02).to_receipt(),
        "execution_mode": "recurrent",
        "execution_spec": spec.to_dict(),
        "execution_spec_sha256": spec.sha256,
        "domains": ["logic"],
        "depths": [2],
        "train_tasks": 1,
        "holdout_tasks": 1,
        "steps": 1,
        "optimizer_updates": 1,
        "invocation_count": 1,
        "termination": {"reason": "max_steps", "completed_budget": True, "signal": None},
        "learning_signal": {},
        "curriculum": {},
        "calibration": None,
        "baseline": {},
        "history": [],
        "step_receipts": [step_receipt],
        "final": {},
        "adapter_decode_delta": 0.0,
        "adapter_standard_decode_delta": None,
        "adapter_recurrent_decode_delta": 0.0,
        "checkpoint": "checkpoints/step-1",
        "verdict": {
            "had_signal": True,
            "point_estimate_improved": False,
            "causal_gain_proven": False,
            "causal_gain_blocker": "requires fresh powered base/adapter x standard/RLC factorial gate",
            "diagnosis": "healthy",
        },
        "elapsed_minutes": 1.0,
    }
    if mutate_receipt is not None:
        mutate_receipt(receipt)
    receipt_bytes = training_json_bytes(receipt)
    identity = _publish_recurrent_adapter_bundle(
        out,
        adapter_id=adapter_id,
        protocol=protocol,
        protocol_bytes=protocol_bytes,
        dataset_bytes=dataset_bytes,
        receipt=receipt,
        receipt_bytes=receipt_bytes,
        execution_spec=spec,
        source_roles=source_paths,
    )
    return {
        "out": out,
        "adapter_id": adapter_id,
        "base": base,
        "behavior": behavior,
        "personality": personality,
        "runtime": runtime,
        "identity": identity,
        "frozen_policy_sha256": frozen_policy_sha256,
    }


def _validate(fixture: dict) -> dict:
    return _validate_published_recurrent_bundle(
        fixture["out"],
        adapter_id=fixture["adapter_id"],
        base_identity=fixture["base"],
        behavior_identity=fixture["behavior"],
        personality_identity=fixture["personality"],
        runtime_identity=fixture["runtime"],
    )


def _verified_wrapper_inputs(
    fixture: dict,
) -> tuple[bytes, dict[str, bytes], object, tuple[object, ...]]:
    manifest_bytes = (fixture["out"] / MANIFEST_FILE).read_bytes()
    manifest = json.loads(manifest_bytes)
    artifacts = {
        binding["path"]: (fixture["out"] / binding["path"]).read_bytes()
        for role, binding in identity_runtime.declared_bindings(manifest)
    }
    receipt = json.loads(artifacts[manifest["training_receipt"]["path"]])
    step = receipt["step_receipts"][0]
    start = {
        "receipt_sha256": "b" * 64,
        "group_manifest": {
            "task_id": step["task_id"],
            "manifest_sha256": step["group_manifest_sha256"],
        },
    }
    terminal = dict(step["terminal"])

    class Ledger:
        def __init__(self):
            self.start = start
            self.terminal = terminal

        def validate_closed(self, *, policy):
            del policy
            return {"close_payload": {"group_count": 1}}

        def group_records(self, *, sequence, policy):
            del policy
            assert sequence == 0
            return self.start, self.terminal

    samples = tuple(
        SimpleNamespace(receipt=lambda value=dict(sample): dict(value))
        for sample in step["samples"]
    )
    replay = SimpleNamespace(
        sequence=0,
        reward_receipt={"receipt_sha256": step["reward_receipt_sha256"]},
        group_manifest={"manifest_sha256": step["group_manifest_sha256"]},
        group_admission_receipt={"receipt_sha256": step["group_admission_sha256"]},
        update_receipt=dict(step["update"]),
        samples=samples,
    )
    return manifest_bytes, artifacts, Ledger(), (replay,)


def _mock_verified_replay(monkeypatch, base_identity: dict) -> dict:
    evidence = _verified_training_evidence(base_identity)
    monkeypatch.setattr(
        identity_runtime,
        "validate_recurrent_grpo_adapter_identity",
        lambda *_args, **_kwargs: dict(base_identity),
    )
    monkeypatch.setattr(
        transition_evidence_runtime,
        "validate_verified_transition_training_evidence",
        lambda *_args, **_kwargs: dict(evidence),
    )
    return evidence


def _rebind_receipt(fixture: dict, mutate) -> None:
    out = fixture["out"]
    receipt_path = out / "campaign_adapter/grpo_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    mutate(receipt)
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    receipt_path.write_bytes(receipt_bytes)
    manifest_path = out / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["training_receipt"] = {
        "path": "campaign_adapter/grpo_receipt.json",
        "sha256": sha256_bytes(receipt_bytes),
        "size_bytes": len(receipt_bytes),
    }
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    manifest_path.write_bytes(manifest_bytes)
    completion_path = out / "training_completion.json"
    completion = json.loads(completion_path.read_text(encoding="ascii"))
    completion["receipt_sha256"] = manifest["training_receipt"]["sha256"]
    completion["manifest_sha256"] = sha256_bytes(manifest_bytes)
    completion_path.write_bytes(canonical_json_bytes(completion) + b"\n")


def test_recurrent_grpo_bundle_is_complete_distinct_and_idempotent(tmp_path):
    fixture = _fixture(tmp_path)
    identity = _validate(fixture)

    assert identity == fixture["identity"]
    assert identity["schema"] == IDENTITY_RECEIPT_SCHEMA
    assert identity["training_method"] == "recurrent_grpo"
    assert identity["optimizer_updates"] == 1
    assert identity["causal_gain_proven"] is False
    assert _validate(fixture) == identity


def test_recurrent_grpo_identity_accepts_verified_transition_step_format(tmp_path):
    fixture = _fixture(tmp_path, verified_step=True)
    identity = _validate(fixture)

    assert identity == fixture["identity"]
    assert identity["optimizer_updates"] == 1
    assert identity["final_policy_sha256"] == fixture["frozen_policy_sha256"]


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_frozen_policy_hash_matches_live_mlx_tensor_tree(tmp_path, dtype_name):
    mx = pytest.importorskip("mlx.core")
    from mlx.utils import tree_flatten

    spec = RLCExecutionSpec(
        n_slots=2,
        branch_roles=("constructive_solution",),
        recurrent_steps=2,
        adaptive_halting=False,
        latent_opt_mode="disabled",
        fast_weights_mode="disabled",
        decode_bridge_policy="none",
    )

    class Model:
        def __init__(self) -> None:
            dtype = getattr(mx, dtype_name)
            self.parameters = {
                "model": {
                    "layers": [
                        {
                            "self_attn": {
                                "o_proj": {
                                    "lora_a": mx.array([[0.25, -0.5]], dtype=dtype),
                                    "lora_b": mx.array([[1.5], [-2.0]], dtype=dtype),
                                }
                            }
                        }
                    ]
                }
            }

        def trainable_parameters(self):
            return self.parameters

    model = Model()
    path = tmp_path / f"{dtype_name}.safetensors"
    mx.save_safetensors(str(path), dict(tree_flatten(model.trainable_parameters())))

    assert recurrent_policy_sha256_from_safetensors(
        path.read_bytes(),
        execution_spec_sha256=spec.sha256,
    ) == recurrent_policy_sha256(model, spec)


def test_verified_identity_requires_source_replay_and_cross_binds_final_policy(
    monkeypatch,
    tmp_path,
):
    fixture = _fixture(tmp_path, verified_step=True)
    base_identity = fixture["identity"]
    evidence = _verified_training_evidence(base_identity)
    manifest, artifacts, ledger, groups = _verified_wrapper_inputs(fixture)
    monkeypatch.setattr(
        identity_runtime,
        "validate_recurrent_grpo_adapter_identity",
        lambda *_args, **_kwargs: dict(base_identity),
    )
    monkeypatch.setattr(
        transition_evidence_runtime,
        "validate_verified_transition_training_evidence",
        lambda *_args, **_kwargs: dict(evidence),
    )

    result = validate_recurrent_grpo_adapter_identity_with_verified_transitions(
        manifest,
        adapter_id="verified-adapter",
        actual_base_checkpoint={},
        actual_model_behavior_bundle={},
        actual_personality_adapter={},
        actual_runtime_environment={},
        artifacts=artifacts,
        tensor_metadata=(),
        transition_campaign_ledger=ledger,
        transition_policy=object(),
        transition_groups=groups,
    )

    assert result["proof_grade_mutation"] is True
    assert result["legacy_scalar_reward_path_used"] is False
    assert result["verified_transition_evidence_sha256"] == evidence["receipt_sha256"]
    assert result["base_identity"] == base_identity

    evidence = _verified_training_evidence(base_identity)
    evidence["final_policy_sha256"] = "4" * 64
    unsigned = dict(evidence)
    unsigned.pop("receipt_sha256")
    evidence["receipt_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    with pytest.raises(
        RecurrentGRPOAdapterIdentityError,
        match="verified_transition_identity_cross_binding_mismatch",
    ):
        validate_recurrent_grpo_adapter_identity_with_verified_transitions(
            manifest,
            adapter_id="verified-adapter",
            actual_base_checkpoint={},
            actual_model_behavior_bundle={},
            actual_personality_adapter={},
            actual_runtime_environment={},
            artifacts=artifacts,
            tensor_metadata=(),
            transition_campaign_ledger=ledger,
            transition_policy=object(),
            transition_groups=groups,
        )


def test_verified_identity_rejects_legacy_step_even_with_matching_campaign(
    monkeypatch,
    tmp_path,
):
    fixture = _fixture(tmp_path, verified_step=True)
    _mock_verified_replay(monkeypatch, fixture["identity"])
    manifest_bytes, artifacts, ledger, groups = _verified_wrapper_inputs(fixture)
    manifest = json.loads(manifest_bytes)
    receipt_path = manifest["training_receipt"]["path"]
    receipt = json.loads(artifacts[receipt_path])
    receipt["step_receipts"][0]["schema"] = "aura.recurrent_grpo.step.v3"
    rebound = training_json_bytes(receipt)
    artifacts[receipt_path] = rebound
    manifest["training_receipt"] = {
        **manifest["training_receipt"],
        "sha256": _sha(rebound),
        "size_bytes": len(rebound),
    }

    with pytest.raises(
        RecurrentGRPOAdapterIdentityError,
        match="verified_transition_identity_legacy_step_forbidden",
    ):
        validate_recurrent_grpo_adapter_identity_with_verified_transitions(
            manifest,
            adapter_id=fixture["adapter_id"],
            actual_base_checkpoint={},
            actual_model_behavior_bundle={},
            actual_personality_adapter={},
            actual_runtime_environment={},
            artifacts=artifacts,
            tensor_metadata=(),
            transition_campaign_ledger=ledger,
            transition_policy=object(),
            transition_groups=groups,
        )


def test_verified_identity_rejects_unrelated_campaign_with_same_count_and_policy(
    monkeypatch,
    tmp_path,
):
    fixture = _fixture(tmp_path, verified_step=True)
    _mock_verified_replay(monkeypatch, fixture["identity"])
    manifest, artifacts, ledger, groups = _verified_wrapper_inputs(fixture)
    ledger.start = {
        **ledger.start,
        "group_manifest": {
            **ledger.start["group_manifest"],
            "manifest_sha256": "0" * 64,
        },
    }

    with pytest.raises(
        RecurrentGRPOAdapterIdentityError,
        match="verified_transition_ordered_step_binding_mismatch",
    ):
        validate_recurrent_grpo_adapter_identity_with_verified_transitions(
            manifest,
            adapter_id=fixture["adapter_id"],
            actual_base_checkpoint={},
            actual_model_behavior_bundle={},
            actual_personality_adapter={},
            actual_runtime_environment={},
            artifacts=artifacts,
            tensor_metadata=(),
            transition_campaign_ledger=ledger,
            transition_policy=object(),
            transition_groups=groups,
        )


def test_frozen_verified_identity_reconstructs_step_chain_semantics(
    monkeypatch,
    tmp_path,
):
    fixture = _fixture(tmp_path, verified_step=True)
    _mock_verified_replay(monkeypatch, fixture["identity"])
    manifest, artifacts, ledger, groups = _verified_wrapper_inputs(fixture)
    identity = validate_recurrent_grpo_adapter_identity_with_verified_transitions(
        manifest,
        adapter_id=fixture["adapter_id"],
        actual_base_checkpoint={},
        actual_model_behavior_bundle={},
        actual_personality_adapter={},
        actual_runtime_environment={},
        artifacts=artifacts,
        tensor_metadata=(),
        transition_campaign_ledger=ledger,
        transition_policy=object(),
        transition_groups=groups,
    )
    attacked = copy.deepcopy(identity)
    attacked["verified_step_chain"][0]["reward_receipt_sha256"] = "0" * 64
    attacked["verified_step_chain_sha256"] = sha256_bytes(
        canonical_json_bytes(attacked["verified_step_chain"])
    )
    unsigned = dict(attacked)
    unsigned.pop("composite_identity_sha256")
    attacked["composite_identity_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))

    with pytest.raises(
        RecurrentGRPOAdapterIdentityError,
        match="verified_identity_receipt_reconstruction_mismatch",
    ):
        validate_verified_recurrent_grpo_adapter_identity_receipt(attacked)


def test_recurrent_grpo_shaped_reward_bundle_round_trips_real_producer_format(
    tmp_path,
):
    fixture = _fixture(tmp_path, trajectory_credit=True)
    identity = _validate(fixture)
    receipt = json.loads(
        (fixture["out"] / "campaign_adapter/grpo_receipt.json").read_text(encoding="ascii")
    )
    rewards = receipt["step_receipts"][0]["rewards"]

    assert identity == fixture["identity"]
    assert min(rewards) < 0.0
    assert receipt["step_receipts"][0]["trajectory_credit"] is not None


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda step: step["rewards"].__setitem__(0, step["rewards"][0] + 0.01),
            "step_trajectory_effective_reward_mismatch",
        ),
        (
            lambda step: step["verifier_rewards"].__setitem__(0, 1.1),
            "step_verifier_rewards_invalid",
        ),
        (
            lambda step: step["answer_channel"].update({"correct": 2}),
            "step_answer_channel_fraction_mismatch",
        ),
    ],
)
def test_recurrent_grpo_identity_rejects_reward_channel_rebinding(
    tmp_path,
    mutation,
    reason,
):
    fixture = _fixture(tmp_path, trajectory_credit=True)
    _rebind_receipt(
        fixture,
        lambda receipt: mutation(receipt["step_receipts"][0]),
    )

    with pytest.raises(RecurrentGRPOAdapterIdentityError, match=reason):
        _validate(fixture)


def test_recurrent_grpo_identity_rejects_config_protocol_rebinding(tmp_path):
    fixture = _fixture(tmp_path)
    _rebind_receipt(
        fixture,
        lambda receipt: receipt["config"].update({"kl_coefficient": 0.03}),
    )

    with pytest.raises(
        RecurrentGRPOAdapterIdentityError,
        match="receipt_config_cross_binding_mismatch",
    ):
        _validate(fixture)


def test_recurrent_grpo_identity_rejects_rebound_causal_gain_claim(tmp_path):
    fixture = _fixture(tmp_path)
    _rebind_receipt(
        fixture,
        lambda receipt: receipt["verdict"].update({"causal_gain_proven": True}),
    )

    with pytest.raises(
        RecurrentGRPOAdapterIdentityError,
        match="training_receipt_cross_binding_mismatch",
    ):
        _validate(fixture)


def test_recurrent_grpo_publication_preflights_before_completion_marker(tmp_path):
    with pytest.raises(
        RecurrentGRPOAdapterIdentityError,
        match="training_receipt_cross_binding_mismatch",
    ):
        _fixture(
            tmp_path,
            mutate_receipt=lambda receipt: receipt["verdict"].update({"causal_gain_proven": True}),
        )

    assert not (tmp_path / "training/training_completion.json").exists()


def test_recurrent_grpo_identity_rejects_rebound_unadmitted_behavior(tmp_path):
    fixture = _fixture(tmp_path)
    _rebind_receipt(
        fixture,
        lambda receipt: receipt["step_receipts"][0]["samples"][0].update(
            {"behavior_admitted": False}
        ),
    )

    with pytest.raises(RecurrentGRPOAdapterIdentityError, match="sample_behavior_not_admitted"):
        _validate(fixture)


def test_recurrent_grpo_identity_rejects_receipt_policy_not_in_frozen_tensors(tmp_path):
    fixture = _fixture(tmp_path)
    _rebind_receipt(
        fixture,
        lambda receipt: receipt["step_receipts"][0].update({"policy_after_sha256": "0" * 64}),
    )

    with pytest.raises(
        RecurrentGRPOAdapterIdentityError,
        match="final_policy_adapter_mismatch",
    ):
        _validate(fixture)


def test_recurrent_grpo_identity_rejects_runtime_or_model_substitution(tmp_path):
    fixture = _fixture(tmp_path)
    wrong_base = copy.deepcopy(fixture["base"])
    wrong_base["fingerprint"] = "9" * 64

    with pytest.raises(RecurrentGRPOAdapterIdentityError, match="base_checkpoint_mismatch"):
        _validate_published_recurrent_bundle(
            fixture["out"],
            adapter_id=fixture["adapter_id"],
            base_identity=wrong_base,
            behavior_identity=fixture["behavior"],
            personality_identity=fixture["personality"],
            runtime_identity=fixture["runtime"],
        )


def test_recurrent_grpo_bundle_freezes_and_verifies_without_unplanned_files(tmp_path):
    fixture = _fixture(tmp_path)
    source = fixture["out"]
    staging = tmp_path / ".freeze.staging"
    destination = tmp_path / "freeze"
    inventory = preparation.copy_adapter_snapshot(source, staging)
    certificate = build_adapter_freeze_certificate(
        adapter_id=fixture["adapter_id"],
        inventory=inventory,
        identity_receipt=fixture["identity"],
        model_identity={
            "fingerprint": "1" * 64,
            "files": 1,
            "model_behavior_bundle_sha256": "2" * 64,
            "runtime_bundle_sha256": "6" * 64,
            "runtime_environment_identity_sha256": "3" * 64,
            "personality_adapter_bundle_sha256": "",
            "effective_stack_sha256": "7" * 64,
        },
        validator_identity={"validator_sha256": "8" * 64},
    )
    preparation.seal_adapter_snapshot(staging, destination, certificate)

    verified = verify_adapter_freeze(destination)
    assert verified["identity_receipt"]["schema"] == IDENTITY_RECEIPT_SCHEMA
    assert not (destination / "grpo_adapters.safetensors").exists()
    assert (destination / "campaign_adapter/adapters.safetensors").exists()


def test_verified_recurrent_grpo_identity_freezes_only_with_proof_fields(
    tmp_path,
    monkeypatch,
):
    fixture = _fixture(tmp_path, verified_step=True)
    source = fixture["out"]
    staging = tmp_path / ".verified-freeze.staging"
    destination = tmp_path / "verified-freeze"
    inventory = preparation.copy_adapter_snapshot(source, staging)
    evidence = _verified_training_evidence(fixture["identity"])
    manifest, artifacts, ledger, groups = _verified_wrapper_inputs(fixture)
    monkeypatch.setattr(
        identity_runtime,
        "validate_recurrent_grpo_adapter_identity",
        lambda *_args, **_kwargs: dict(fixture["identity"]),
    )
    monkeypatch.setattr(
        transition_evidence_runtime,
        "validate_verified_transition_training_evidence",
        lambda *_args, **_kwargs: dict(evidence),
    )
    identity = validate_recurrent_grpo_adapter_identity_with_verified_transitions(
        manifest,
        adapter_id=fixture["adapter_id"],
        actual_base_checkpoint={},
        actual_model_behavior_bundle={},
        actual_personality_adapter={},
        actual_runtime_environment={},
        artifacts=artifacts,
        tensor_metadata=(),
        transition_campaign_ledger=ledger,
        transition_policy=object(),
        transition_groups=groups,
    )
    monkeypatch.undo()
    model_identity = {
        "fingerprint": "1" * 64,
        "files": 1,
        "model_behavior_bundle_sha256": "2" * 64,
        "runtime_bundle_sha256": "6" * 64,
        "runtime_environment_identity_sha256": "3" * 64,
        "personality_adapter_bundle_sha256": "",
        "effective_stack_sha256": "7" * 64,
    }
    certificate = build_adapter_freeze_certificate(
        adapter_id=fixture["adapter_id"],
        inventory=inventory,
        identity_receipt=identity,
        model_identity=model_identity,
        validator_identity={"validator_sha256": "8" * 64},
    )
    preparation.seal_adapter_snapshot(staging, destination, certificate)

    verified = verify_adapter_freeze(destination)
    assert verified["identity_receipt"]["schema"] == VERIFIED_IDENTITY_RECEIPT_SCHEMA
    assert verified["identity_receipt"]["proof_grade_mutation"] is True

    incomplete = copy.deepcopy(identity)
    incomplete["verified_transition_evidence"]["final_policy_sha256"] = "0" * 64
    with pytest.raises(CampaignLaunchBundleError, match="adapter_identity_receipt_invalid"):
        build_adapter_freeze_certificate(
            adapter_id=fixture["adapter_id"],
            inventory=inventory,
            identity_receipt=incomplete,
            model_identity=model_identity,
            validator_identity={"validator_sha256": "8" * 64},
        )
