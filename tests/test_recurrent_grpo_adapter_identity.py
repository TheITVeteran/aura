"""Identity, tamper, and freeze contracts for recurrent-GRPO adapters."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_launch_bundle import (
    build_adapter_freeze_certificate,
    verify_adapter_freeze,
)
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
    IDENTITY_RECEIPT_SCHEMA,
    MANIFEST_FILE,
    RecurrentGRPOAdapterIdentityError,
    sha256_bytes,
)
from core.learning.grpo_training_state import canonical_json_bytes as training_json_bytes
from tests.fixtures.rlc_runtime_integrity import engine_runtime_integrity
from tools import prepare_latent_cortex_campaign as preparation
from tools.train_grpo import (
    GRPO_DATASET_SCHEMA,
    GRPO_PROTOCOL_SCHEMA,
    GRPO_TRAIN_SCHEMA,
    _publish_recurrent_adapter_bundle,
    _validate_published_recurrent_bundle,
)

SOURCE_ROLES = {
    "trainer",
    "grpo",
    "curriculum",
    "tasks",
    "checkpoint",
    "adapter",
    "recurrent_grpo",
    "recurrent_objective",
    "execution_spec",
    "latent_engine",
    "recurrence",
}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path, *, mutate_receipt=None) -> dict:
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
        "domains": ["logic"],
        "depths": [2],
        "train_per_cell": 1,
        "holdout_per_cell": 1,
        "group_size": 2,
        "temperature": 1.0,
        "max_tokens": 32,
        "kl_coefficient": 0.02,
        "format_credit": 0.0,
        "lora_rank": 2,
        "lora_targets": "o_proj",
        "lora_layers": 1,
        "learning_rate": 1e-5,
        "max_steps": 1,
        "eval_every": 1,
        "checkpoint_every": 1,
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
        "schema": "aura.recurrent_sampling_behavior.v3",
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
    receipt = {
        "schema": GRPO_TRAIN_SCHEMA,
        "adapter_id": adapter_id,
        "protocol_sha256": _sha(protocol_bytes),
        "dataset_sha256": _sha(dataset_bytes),
        "model": {"path": "/model", "base_checkpoint": base, "behavior": behavior},
        "config": {},
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
        "step_receipts": [
            {
                "step": 1,
                "task_id": "train-1",
                "sample_seed": 41,
                "execution_spec_sha256": spec.sha256,
                "rewards": [1, 0],
                "advantage_report": {},
                "samples": [dict(sample), dict(sample)],
                "step_kind": "optimizer_update",
                "update": {"schema": "aura.recurrent_grpo.v1", "has_gradient": True},
                "policy_after_sha256": "5" * 64,
            }
        ],
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
            mutate_receipt=lambda receipt: receipt["verdict"].update(
                {"causal_gain_proven": True}
            ),
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

    with pytest.raises(
        RecurrentGRPOAdapterIdentityError, match="sample_behavior_not_admitted"
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
