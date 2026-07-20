"""Fail-closed identity contracts for recurrence-native v2 bundles."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
    SOURCE_ROLES,
    RecurrenceAdapterIdentityV2Error,
    canonical_json_bytes,
    personality_bundle_identity,
    runtime_environment_identity,
    validate_v2_adapter_identity,
)


def _encoded(value) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _binding(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _bundle():
    base = {"fingerprint": "a" * 64, "method": "sha256", "files": 2}
    behavior_files = [
        {"path": name, "sha256": character * 64, "size_bytes": 1}
        for name, character in (
            ("config.json", "d"),
            ("tokenizer.json", "e"),
            ("tokenizer_config.json", "f"),
        )
    ]
    model_behavior = {
        "bundle_sha256": hashlib.sha256(
            canonical_json_bytes(behavior_files)
        ).hexdigest(),
        "file_count": len(behavior_files),
        "files": behavior_files,
    }
    personality = personality_bundle_identity(None)
    training_runtime = runtime_environment_identity()
    spec = RLCExecutionSpec(recurrent_steps=4)
    sources: dict[str, dict[str, object]] = {}
    config_sources: dict[str, dict[str, object]] = {}
    artifacts: dict[str, bytes] = {}
    for role in sorted(SOURCE_ROLES):
        payload = f"# immutable {role} source\n".encode("ascii")
        digest = hashlib.sha256(payload).hexdigest()
        snapshot_path = f"source_snapshots/{role}.py"
        artifacts[snapshot_path] = payload
        origin_path = f"core/frozen/{role}.py"
        sources[role] = {
            "origin_path": origin_path,
            "snapshot_path": snapshot_path,
            "sha256": digest,
            "size_bytes": len(payload),
        }
        config_sources[role] = {
            "path": origin_path,
            "sha256": digest,
            "size_bytes": len(payload),
        }
    dataset = {
        "schema": "aura.recurrence_native_dataset.v2",
        "generator": config_sources["task_generator"],
        "train_seed": 1777,
        "families": ["khop"],
        "task_depths": [2],
        "per_cell": 1,
        "examples": [
            {
                "family": "khop",
                "depth": 2,
                "seed": 1,
                "prompt": "p",
                "answer": "a",
                "prompt_tokens": [1],
                "answer_tokens": [2],
            }
        ],
    }
    dataset_bytes = _encoded(dataset)
    dataset_sha = hashlib.sha256(dataset_bytes).hexdigest()
    lora_config = {
        "rank": 2,
        "targets": ["o_proj"],
        "wrapped_projections": ["model.layers.1.self_attn.o_proj"],
    }
    gradient_execution = {
        "schema": "aura.recurrence_streamed_depth_gradient.v1",
        "mode": "depth_serial_exact_sum",
        "concurrent_depth_graphs": 1,
        "optimizer_updates_per_sample": 1,
        "finite_loss_and_gradient_required_before_update": True,
    }
    config = {
        "schema": "aura.recurrence_native_training_config.v2",
        "model_path": "/models/frozen",
        "base_checkpoint": base,
        "model_behavior_bundle": model_behavior,
        "personality_adapter_path": "",
        "personality_adapter": personality,
        "training_runtime": training_runtime,
        "execution_spec": spec.to_dict(),
        "execution_spec_sha256": spec.sha256,
        "dataset_sha256": dataset_sha,
        "objective_schema": "aura.recurrence_native_objective.v2",
        "curriculum_depths": [1, 2, 4],
        "monotonicity_weight": 0.5,
        "lora": lora_config,
        "optimizer": {"name": "AdamW", "learning_rate": 0.0001, "weight_decay": 0.01},
        "gradient_execution": gradient_execution,
        "train_seed": 1777,
        "max_steps": 8,
        "sources": config_sources,
    }
    config_bytes = _encoded(config)
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    lora = {
        "rank": 2,
        "targets": ["o_proj"],
        "wrapped_projections": 1,
        "projection_paths": ["model.layers.1.self_attn.o_proj"],
        "trainable_params": 64,
    }
    receipt = {
        "schema": "aura.recurrence_native_train.v2",
        "objective_schema": "aura.recurrence_native_objective.v2",
        "objective_source_sha256": config_sources["objective"]["sha256"],
        "trainer_source_sha256": config_sources["trainer"]["sha256"],
        "config_sha256": config_sha,
        "dataset_sha256": dataset_sha,
        "execution_spec_sha256": spec.sha256,
        "base_checkpoint": base,
        "model_behavior_bundle": model_behavior,
        "personality_adapter": personality,
        "training_runtime": training_runtime,
        "lora": lora,
        "optimizer": config["optimizer"],
        "gradient_execution": gradient_execution,
        "steps": 8,
        "epoch": 1,
        "cursor": 2,
        "elapsed_training_s": 3.5,
        "invocation_count": 1,
        "halt_reason": "max_steps",
        "complete": True,
        "final_checkpoint": "step-00000008-frozen",
        "loss_trail": [{"step": 8, "mean_loss": 0.5}],
    }
    receipt_bytes = _encoded(receipt)
    spec_bytes = _encoded(spec.to_dict())
    adapter_bytes = b"frozen-adapter"
    loader_config = {
        "schema": "aura.recurrence_scoped_lora_config.v1",
        "fine_tune_type": "recurrence_scoped_lora",
        "loader": "aura_custom_loader_required",
        "model": "/models/frozen",
        "num_layers": 1,
        "wrapped_projection_count": 1,
        "lora_parameters": {
            "rank": 2,
            "scale": 20.0,
            "dropout": 0.0,
            "keys": ["o_proj"],
        },
        "execution_spec_sha256": spec.sha256,
    }
    loader_config_bytes = _encoded(loader_config)
    artifacts.update(
        {
            "adapters.safetensors": adapter_bytes,
            "adapter_final.safetensors": adapter_bytes,
            "adapter_config.json": loader_config_bytes,
            "receipt.json": receipt_bytes,
            "training_config.json": config_bytes,
            "dataset_manifest.json": dataset_bytes,
            "execution_spec.json": spec_bytes,
        }
    )
    tensors = [
        {
            "key": "model.layers.1.self_attn.o_proj.lora_a",
            "shape": [16, 2],
            "dtype": "float32",
        },
        {
            "key": "model.layers.1.self_attn.o_proj.lora_b",
            "shape": [2, 16],
            "dtype": "float32",
        },
    ]
    manifest = {
        "schema": "aura.recurrence_adapter_manifest.v2",
        "adapter_id": "test-v2",
        "base_checkpoint": base,
        "model_behavior_bundle": model_behavior,
        "personality_adapter": personality,
        "training_runtime": training_runtime,
        "adapter": _binding("adapters.safetensors", adapter_bytes),
        "adapter_alias": _binding("adapter_final.safetensors", adapter_bytes),
        "loader_config": _binding("adapter_config.json", loader_config_bytes),
        "training_receipt": _binding("receipt.json", receipt_bytes),
        "training_config": _binding("training_config.json", config_bytes),
        "dataset_manifest": _binding("dataset_manifest.json", dataset_bytes),
        "execution_spec": _binding("execution_spec.json", spec_bytes),
        "config_sha256": config_sha,
        "dataset_sha256": dataset_sha,
        "execution_spec_sha256": spec.sha256,
        "sources": sources,
        "lora": lora,
        "tensors": tensors,
    }
    manifest_bytes = _encoded(manifest)
    completion = {
        "schema": "aura.recurrence_native_training_completion.v1",
        "complete": True,
        "halt_reason": "max_steps",
        "step": 8,
        "adapter_sha256": manifest["adapter"]["sha256"],
        "receipt_sha256": manifest["training_receipt"]["sha256"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    artifacts["training_completion.json"] = _encoded(completion)
    return (
        manifest_bytes,
        artifacts,
        tensors,
        base,
        model_behavior,
        personality,
        training_runtime,
    )


def _validate(bundle=None, *, allow_bounded_partial: bool = False):
    (
        manifest,
        artifacts,
        tensors,
        base,
        model_behavior,
        personality,
        training_runtime,
    ) = bundle or _bundle()
    return validate_v2_adapter_identity(
        manifest,
        adapter_id="test-v2",
        actual_base_checkpoint=base,
        actual_model_behavior_bundle=model_behavior,
        actual_personality_adapter=personality,
        actual_runtime_environment=training_runtime,
        artifacts=artifacts,
        tensor_metadata=tensors,
        allow_bounded_partial=allow_bounded_partial,
    )


def _partialize_bundle(
    bundle=None,
    *,
    halt_reason: str = "wall_clock",
    steps: int = 7,
):
    (
        manifest_bytes,
        artifacts,
        tensors,
        base,
        model_behavior,
        personality,
        training_runtime,
    ) = copy.deepcopy(bundle or _bundle())
    manifest = json.loads(manifest_bytes)
    receipt = json.loads(artifacts["receipt.json"])
    receipt.update(complete=False, halt_reason=halt_reason, steps=steps)
    receipt_bytes = _encoded(receipt)
    artifacts["receipt.json"] = receipt_bytes
    manifest["training_receipt"] = _binding("receipt.json", receipt_bytes)
    manifest_bytes = _encoded(manifest)
    completion = json.loads(artifacts["training_completion.json"])
    completion.update(
        complete=False,
        halt_reason=halt_reason,
        step=steps,
        receipt_sha256=manifest["training_receipt"]["sha256"],
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    artifacts["training_completion.json"] = _encoded(completion)
    return (
        manifest_bytes,
        artifacts,
        tensors,
        base,
        model_behavior,
        personality,
        training_runtime,
    )


def test_complete_training_bundle_validates_with_training_time_provenance():
    receipt = _validate()

    assert receipt["complete"] is True
    assert receipt["objective_name"] == "aura.recurrence_native_objective.v2"
    assert receipt["objective_source_provenance"] == "training_time_archived_source"
    assert receipt["wrapped_projection_count"] == 1
    assert receipt["gradient_execution"]["mode"] == "depth_serial_exact_sum"
    assert "training_scope" not in receipt
    assert "load_eligible" not in receipt


def test_gradient_execution_must_match_exact_trainer_contract():
    manifest_bytes, artifacts, tensors, base, model_behavior, personality, training_runtime = _bundle()
    receipt = __import__("json").loads(artifacts["receipt.json"])
    receipt["gradient_execution"]["concurrent_depth_graphs"] = 2
    artifacts["receipt.json"] = _encoded(receipt)
    manifest = __import__("json").loads(manifest_bytes)
    manifest["training_receipt"] = _binding(
        "receipt.json", artifacts["receipt.json"]
    )
    manifest_bytes = _encoded(manifest)

    with pytest.raises(
        RecurrenceAdapterIdentityV2Error,
        match="gradient_execution_cross_binding_mismatch",
    ):
        _validate(
            (
                manifest_bytes,
                artifacts,
                tensors,
                base,
                model_behavior,
                personality,
                training_runtime,
            )
        )


def test_partial_training_bundle_is_never_load_eligible():
    with pytest.raises(RecurrenceAdapterIdentityV2Error, match="training_incomplete"):
        _validate(_partialize_bundle())


def test_bounded_partial_scope_is_explicit_and_never_load_eligible():
    receipt = _validate(
        _partialize_bundle(),
        allow_bounded_partial=True,
    )

    assert receipt["complete"] is False
    assert receipt["training_scope"] == "bounded_partial_training"
    assert receipt["training_halt_reason"] == "wall_clock"
    assert receipt["training_steps"] == 7
    assert receipt["training_max_steps"] == 8
    assert receipt["load_eligible"] is False


@pytest.mark.parametrize(
    ("halt_reason", "steps", "error"),
    [
        ("interrupted", 7, "bounded_partial_halt_reason_invalid"),
        ("non_finite_loss", 7, "bounded_partial_halt_reason_invalid"),
        ("wall_clock", 0, "bounded_partial_step_invalid"),
        ("wall_clock", 8, "bounded_partial_step_invalid"),
        ("wall_clock", 9, "bounded_partial_step_invalid"),
    ],
)
def test_bounded_partial_scope_rejects_nonadmissible_terminal_states(
    halt_reason,
    steps,
    error,
):
    with pytest.raises(RecurrenceAdapterIdentityV2Error, match=error):
        _validate(
            _partialize_bundle(halt_reason=halt_reason, steps=steps),
            allow_bounded_partial=True,
        )


def test_bounded_partial_completion_record_must_match_receipt():
    bundle = _partialize_bundle()
    completion = json.loads(bundle[1]["training_completion.json"])
    completion["step"] = 6
    bundle[1]["training_completion.json"] = _encoded(completion)

    with pytest.raises(
        RecurrenceAdapterIdentityV2Error,
        match="training_completion_mismatch",
    ):
        _validate(bundle, allow_bounded_partial=True)


def test_source_snapshot_tamper_fails_even_when_manifest_is_unchanged():
    bundle = _bundle()
    bundle[1]["source_snapshots/objective.py"] += b"# changed\n"

    with pytest.raises(
        RecurrenceAdapterIdentityV2Error, match="source_objective_size_mismatch"
    ):
        _validate(bundle)


def test_tensor_topology_and_effective_personality_are_identity_material():
    manifest, artifacts, tensors, base, model_behavior, personality, training_runtime = _bundle()
    changed_tensors = copy.deepcopy(tensors)
    changed_tensors[0]["shape"] = [17, 2]
    with pytest.raises(RecurrenceAdapterIdentityV2Error, match="tensor_metadata_mismatch"):
        _validate(
            (
                manifest,
                artifacts,
                changed_tensors,
                base,
                model_behavior,
                personality,
                training_runtime,
            )
        )

    personality_files = [{"path": "x", "sha256": "c" * 64, "size_bytes": 1}]
    with pytest.raises(RecurrenceAdapterIdentityV2Error, match="personality_adapter_mismatch"):
        validate_v2_adapter_identity(
            manifest,
            adapter_id="test-v2",
            actual_base_checkpoint=base,
            actual_model_behavior_bundle=model_behavior,
            actual_personality_adapter={
                "present": True,
                "bundle_sha256": hashlib.sha256(
                    canonical_json_bytes(personality_files)
                ).hexdigest(),
                "file_count": 1,
                "files": personality_files,
            },
            actual_runtime_environment=training_runtime,
            artifacts=artifacts,
            tensor_metadata=tensors,
        )


# ── CP181: v3 objective bundles inside the v2 bundle format ─────────────


def _upgrade_bundle_to_v3(*, strip_v3_evidence: bool = False):
    """Rebuild the frozen v2 bundle as a v3-objective bundle, re-hashing
    every dependent binding (dataset → config → receipt → manifest →
    completion) exactly as the trainer would."""
    (
        manifest_bytes,
        artifacts,
        tensors,
        base,
        model_behavior,
        personality,
        training_runtime,
    ) = _bundle()
    artifacts = dict(artifacts)
    manifest = json.loads(manifest_bytes)
    dataset = json.loads(artifacts["dataset_manifest.json"])
    config = json.loads(artifacts["training_config.json"])
    receipt = json.loads(artifacts["receipt.json"])

    dataset["holdout_per_cell"] = 0
    dataset["holdout_indices"] = []
    dataset_bytes = _encoded(dataset)
    dataset_sha = hashlib.sha256(dataset_bytes).hexdigest()

    v3_options = {
        "depth_margin": 0.05,
        "diversity_weight": 0.25,
        "diversity_target_cos": 0.98,
    }
    config["objective_schema"] = "aura.recurrence_native_objective.v3"
    config["dataset_sha256"] = dataset_sha
    config["objective_options"] = v3_options
    config["bridge"] = {
        "policy": "assistant_answer",
        "token_count": 3,
        "tokens_sha256": "b" * 64,
    }
    config["holdout"] = {
        "per_cell": 0,
        "count": 0,
        "eval_samples": 8,
        "indices_sha256": "c" * 64,
    }
    config_bytes = _encoded(config)
    config_sha = hashlib.sha256(config_bytes).hexdigest()

    receipt["objective_schema"] = "aura.recurrence_native_objective.v3"
    receipt["config_sha256"] = config_sha
    receipt["dataset_sha256"] = dataset_sha
    receipt["objective_options"] = v3_options
    receipt["holdout_trail"] = []
    if strip_v3_evidence:
        receipt.pop("objective_options")
        receipt.pop("holdout_trail")
    receipt_bytes = _encoded(receipt)

    artifacts["dataset_manifest.json"] = dataset_bytes
    artifacts["training_config.json"] = config_bytes
    artifacts["receipt.json"] = receipt_bytes
    manifest["dataset_manifest"] = _binding(
        "dataset_manifest.json", dataset_bytes
    )
    manifest["training_config"] = _binding("training_config.json", config_bytes)
    manifest["training_receipt"] = _binding("receipt.json", receipt_bytes)
    manifest["config_sha256"] = config_sha
    manifest["dataset_sha256"] = dataset_sha
    manifest_bytes = _encoded(manifest)
    completion = json.loads(artifacts["training_completion.json"])
    completion["receipt_sha256"] = manifest["training_receipt"]["sha256"]
    completion["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    artifacts["training_completion.json"] = _encoded(completion)
    return (
        manifest_bytes,
        artifacts,
        tensors,
        base,
        model_behavior,
        personality,
        training_runtime,
    )


def test_v3_objective_bundle_validates_and_names_v3():
    receipt = _validate(_upgrade_bundle_to_v3())
    assert receipt["complete"] is True
    assert receipt["objective_name"] == "aura.recurrence_native_objective.v3"


def test_v3_schema_without_v3_evidence_is_rejected():
    with pytest.raises(RecurrenceAdapterIdentityV2Error):
        _validate(_upgrade_bundle_to_v3(strip_v3_evidence=True))


def test_v2_bundle_with_v3_evidence_keys_is_rejected():
    (
        manifest_bytes,
        artifacts,
        tensors,
        base,
        model_behavior,
        personality,
        training_runtime,
    ) = _bundle()
    artifacts = dict(artifacts)
    manifest = json.loads(manifest_bytes)
    receipt = json.loads(artifacts["receipt.json"])
    receipt["objective_options"] = {"depth_margin": 0.05}  # v3 key, v2 schema
    receipt_bytes = _encoded(receipt)
    artifacts["receipt.json"] = receipt_bytes
    manifest["training_receipt"] = _binding("receipt.json", receipt_bytes)
    manifest_bytes = _encoded(manifest)
    completion = json.loads(artifacts["training_completion.json"])
    completion["receipt_sha256"] = manifest["training_receipt"]["sha256"]
    completion["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    artifacts["training_completion.json"] = _encoded(completion)
    with pytest.raises(RecurrenceAdapterIdentityV2Error):
        _validate(
            (
                manifest_bytes,
                artifacts,
                tensors,
                base,
                model_behavior,
                personality,
                training_runtime,
            )
        )


def test_receipt_config_objective_schemas_must_agree():
    (
        manifest_bytes,
        artifacts,
        tensors,
        base,
        model_behavior,
        personality,
        training_runtime,
    ) = _upgrade_bundle_to_v3()
    artifacts = dict(artifacts)
    manifest = json.loads(manifest_bytes)
    config = json.loads(artifacts["training_config.json"])
    config["objective_schema"] = "aura.recurrence_native_objective.v2"
    config_bytes = _encoded(config)
    artifacts["training_config.json"] = config_bytes
    manifest["training_config"] = _binding("training_config.json", config_bytes)
    manifest["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    receipt = json.loads(artifacts["receipt.json"])
    receipt["config_sha256"] = manifest["config_sha256"]
    receipt_bytes = _encoded(receipt)
    artifacts["receipt.json"] = receipt_bytes
    manifest["training_receipt"] = _binding("receipt.json", receipt_bytes)
    manifest_bytes = _encoded(manifest)
    completion = json.loads(artifacts["training_completion.json"])
    completion["receipt_sha256"] = manifest["training_receipt"]["sha256"]
    completion["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    artifacts["training_completion.json"] = _encoded(completion)
    with pytest.raises(RecurrenceAdapterIdentityV2Error):
        _validate(
            (
                manifest_bytes,
                artifacts,
                tensors,
                base,
                model_behavior,
                personality,
                training_runtime,
            )
        )


def _upgrade_bundle_to_migrated_v3():
    (
        manifest_bytes,
        artifacts,
        tensors,
        base,
        model_behavior,
        personality,
        training_runtime,
    ) = _upgrade_bundle_to_v3()
    artifacts = dict(artifacts)
    manifest = json.loads(manifest_bytes)
    config = json.loads(artifacts["training_config.json"])
    receipt = json.loads(artifacts["receipt.json"])
    gradient = {
        "schema": "aura.recurrence_streamed_depth_gradient.v5",
        "mode": "depth_serial_exact_sum",
        "concurrent_depth_graphs": 1,
        "optimizer_updates_per_sample": 1,
        "finite_loss_and_gradient_required_before_update": True,
        "activation_rematerialization": "transformer_layer_group_checkpoint",
        "layer_group_size": 4,
        "recurrent_transition_checkpointing": True,
    }
    config["gradient_execution"] = gradient
    receipt["gradient_execution"] = gradient
    migration_material = {
        "schema": "aura.recurrence_checkpoint_migration.v2",
        "source": {
            "checkpoint": "checkpoints/step-00000007-source",
            "step": 7,
            "config_sha256": "1" * 64,
            "dataset_sha256": config["dataset_sha256"],
            "execution_spec_sha256": config["execution_spec_sha256"],
        },
        "destination": {
            "complete": {"sha256": "2" * 64},
            "adapter": {"sha256": "3" * 64},
            "optimizer": {"sha256": "4" * 64},
        },
        "failure": {"tombstone": {"sha256": "5" * 64}},
        "recovery_attempts": [],
        "required_execution_change": {
            "activation_rematerialization": "transformer_layer_group_checkpoint",
            "layer_group_size": 4,
            "recurrent_transition_checkpointing": True,
        },
        "new_trainer": {"sha256": receipt["trainer_source_sha256"]},
    }
    migration = {
        **migration_material,
        "migration_sha256": hashlib.sha256(
            canonical_json_bytes(migration_material)
        ).hexdigest(),
    }
    resume_migration = {
        "schema": migration["schema"],
        "migration_sha256": migration["migration_sha256"],
        "source_checkpoint": migration["source"]["checkpoint"],
        "source_step": migration["source"]["step"],
        "source_config_sha256": migration["source"]["config_sha256"],
        "dataset_sha256": migration["source"]["dataset_sha256"],
        "execution_spec_sha256": migration["source"]["execution_spec_sha256"],
        "checkpoint_complete_sha256": migration["destination"]["complete"]["sha256"],
        "adapter_sha256": migration["destination"]["adapter"]["sha256"],
        "optimizer_sha256": migration["destination"]["optimizer"]["sha256"],
        "failure_tombstone_sha256": migration["failure"]["tombstone"]["sha256"],
        "recovery_attempt_count": 0,
        "recovery_attempts_sha256": hashlib.sha256(
            canonical_json_bytes(migration["recovery_attempts"])
        ).hexdigest(),
        "activation_rematerialization": "transformer_layer_group_checkpoint",
        "layer_group_size": 4,
        "recurrent_transition_checkpointing": True,
        "new_trainer_sha256": migration["new_trainer"]["sha256"],
    }
    config["resume_migration"] = resume_migration
    config_bytes = _encoded(config)
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    receipt["resume_migration"] = resume_migration
    receipt["config_sha256"] = config_sha
    receipt_bytes = _encoded(receipt)
    migration_bytes = _encoded(migration)
    artifacts["training_config.json"] = config_bytes
    artifacts["receipt.json"] = receipt_bytes
    artifacts["checkpoint_migration.json"] = migration_bytes
    manifest["training_config"] = _binding("training_config.json", config_bytes)
    manifest["training_receipt"] = _binding("receipt.json", receipt_bytes)
    manifest["checkpoint_migration"] = _binding(
        "checkpoint_migration.json", migration_bytes
    )
    manifest["config_sha256"] = config_sha
    manifest_bytes = _encoded(manifest)
    completion = json.loads(artifacts["training_completion.json"])
    completion["receipt_sha256"] = manifest["training_receipt"]["sha256"]
    completion["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    artifacts["training_completion.json"] = _encoded(completion)
    return (
        manifest_bytes,
        artifacts,
        tensors,
        base,
        model_behavior,
        personality,
        training_runtime,
    )


def test_migrated_v3_bundle_binds_checkpoint_provenance():
    receipt = _validate(_upgrade_bundle_to_migrated_v3())

    assert receipt["complete"] is True
    assert receipt["resume_migration"]["source_step"] == 7
    assert receipt["resume_migration"]["activation_rematerialization"] == (
        "transformer_layer_group_checkpoint"
    )
    assert receipt["resume_migration"]["layer_group_size"] == 4
    assert receipt["resume_migration"]["recurrent_transition_checkpointing"] is True
    assert receipt["checkpoint_migration_sha256"]


def test_migrated_v3_bundle_rejects_certificate_tamper():
    bundle = _upgrade_bundle_to_migrated_v3()
    certificate = json.loads(bundle[1]["checkpoint_migration.json"])
    certificate["source"]["step"] = 6
    bundle[1]["checkpoint_migration.json"] = _encoded(certificate)

    with pytest.raises(
        RecurrenceAdapterIdentityV2Error,
        match="checkpoint_migration_size_mismatch|checkpoint_migration_sha256_mismatch",
    ):
        _validate(bundle)


def test_migrated_v3_bundle_requires_all_three_binding_surfaces():
    bundle = _upgrade_bundle_to_migrated_v3()
    manifest = json.loads(bundle[0])
    manifest.pop("checkpoint_migration")
    bundle = (_encoded(manifest), *bundle[1:])

    with pytest.raises(
        RecurrenceAdapterIdentityV2Error,
        match="resume_migration_cross_binding_mismatch",
    ):
        _validate(bundle)
