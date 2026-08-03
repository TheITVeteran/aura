"""Adversarial contracts for resident recurrent-SFT adapter identity."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.resident_recurrent_sft_adapter_identity import (
    CONTROLLER_COMPLETION_SCHEMA,
    INVOCATION_SCHEMA,
    LEGACY_MANIFEST_SCHEMA,
    MANIFEST_SCHEMA,
    PACKAGE_COMPLETION_SCHEMA,
    ResidentRecurrentSFTAdapterIdentityError,
    topology_sha256,
    validate_resident_recurrent_sft_adapter_identity,
)
from core.learning.recurrence_curriculum import RECURRENCE_TRAINING_FAMILIES
from core.learning.resident_recurrent_sft_bootstrap_authority import (
    REQUIRED_SOURCE_ROLES,
    TRAINING_AUTHORITY,
    ResidentSFTBootstrapConfig,
    build_authority,
    build_dataset_commitment,
    canonical_dataset_payloads,
    sha256_json,
)
from core.learning.resident_recurrent_sft_bootstrap_state import (
    CHECKPOINT_SCHEMA,
    POINTER_SCHEMA,
    authority_state_bindings,
    order_sha256,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _binding(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": _sha(payload), "size_bytes": len(payload)}


def _json(value: Any) -> bytes:
    return canonical_json_bytes(value)


def _runtime(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_json(body)}


def test_topology_digest_preserves_bootstrap_mlx_dtype_spelling() -> None:
    tensors = [
        {
            "key": "model.layers.40.self_attn.q_proj.lora_a",
            "shape": [5120, 8],
            "dtype": "float32",
        }
    ]

    assert topology_sha256(tensors) == sha256_json(
        [
            {
                "name": "model.layers.40.self_attn.q_proj.lora_a",
                "shape": [5120, 8],
                "dtype": "mlx.core.float32",
            }
        ]
    )


@dataclass
class _Bundle:
    manifest: dict[str, Any]
    artifacts: dict[str, bytes]
    tensors: list[dict[str, Any]]
    base: dict[str, Any]
    behavior: dict[str, Any]
    personality: dict[str, Any]
    evaluation_runtime: dict[str, Any]

    def validate(self) -> dict[str, Any]:
        return validate_resident_recurrent_sft_adapter_identity(
            _json(self.manifest),
            adapter_id="resident-sft-test",
            actual_base_checkpoint=self.base,
            actual_model_behavior_bundle=self.behavior,
            actual_personality_adapter=self.personality,
            actual_runtime_environment=self.evaluation_runtime,
            artifacts=self.artifacts,
            tensor_metadata=self.tensors,
        )

    def replace_artifact(self, role: str, payload: bytes) -> None:
        bindings = self.manifest["bindings"]
        if role.startswith("source_"):
            binding = bindings["source_snapshots"][role.removeprefix("source_")]
        else:
            binding = bindings[role]
        self.artifacts[binding["path"]] = payload
        binding.update(_binding(binding["path"], payload))

    def sync_package_completion(self) -> None:
        bindings = self.manifest["bindings"]
        authority = json.loads(self.artifacts[bindings["authority"]["path"]])
        complete = json.loads(self.artifacts[bindings["checkpoint_complete"]["path"]])
        self.artifacts["training_completion.json"] = _json(
            {
                "schema": PACKAGE_COMPLETION_SCHEMA,
                "complete": True,
                "halt_reason": "max_steps",
                "step": complete["state"]["step"],
                "adapter_sha256": bindings["adapter"]["sha256"],
                "checkpoint_complete_sha256": bindings["checkpoint_complete"]["sha256"],
                "authority_sha256": authority["authority_sha256"],
                "manifest_sha256": _sha(_json(self.manifest)),
            }
        )


def _row(task_id: str, prompt: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "family": sorted(RECURRENCE_TRAINING_FAMILIES)[0],
        "depth": 2,
        "prompt": prompt,
        "answer": 'FINAL_ANSWER: {"value":1}',
    }


def _bundle(*, manifest_schema: str = MANIFEST_SCHEMA) -> _Bundle:
    spec = RLCExecutionSpec(recurrent_steps=2)
    spec_payload = _json(spec.to_dict())
    train_rows = [_row("train.1", "Solve the training recurrence.")]
    validation_rows = [_row("validation.1", "Solve the held-out recurrence.")]
    train_payload, validation_payload = canonical_dataset_payloads(
        train_rows,
        validation_rows,
    )
    dataset = build_dataset_commitment(train_rows, validation_rows)
    source_payloads = {role: f"# frozen {role}\n".encode("ascii") for role in REQUIRED_SOURCE_ROLES}
    trust_payload = _json({"policy": "test-only-frozen-trust"})
    base = {"fingerprint": "a" * 64, "method": "sha256", "files": 9}
    behavior = {"bundle_sha256": "b" * 64, "file_count": 3, "files": []}
    personality = {"identity_sha256": "c" * 64, "present": False}
    runtime_body = {"python": "3.12.0", "mlx": "0.test"}
    evaluation_runtime = _runtime(runtime_body)
    training_runtime_body = {**runtime_body, "interpreter": "/frozen/python"}
    training_runtime = _runtime(training_runtime_body)
    config = ResidentSFTBootstrapConfig(
        seed=2026080107,
        max_steps=8,
        max_invocation_steps=4,
        lora_rank=2,
        lora_targets=("o_proj",),
        lora_layers=1,
        evaluate_every=8,
        validation_examples=1,
    )
    authority = build_authority(
        campaign_id="resident-32b-recurrent-sft-bootstrap-cp-test",
        campaign_scope="full_bootstrap",
        committed_at="2026-08-01T01:00:00-07:00",
        expires_at="2026-08-08T01:00:00-07:00",
        model_path="training/fused-model/aura-32b",
        model_identity=base,
        behavior_identity=behavior,
        personality_identity=personality,
        tokenizer_identity={
            "identity_sha256": "d" * 64,
            "artifact_sha256": "e" * 64,
            "runtime_sha256": "f" * 64,
        },
        execution_spec={
            **_binding("config/execution-spec.json", spec_payload),
            "semantic_sha256": spec.sha256,
        },
        dataset=dataset,
        dataset_artifacts={
            "train": _binding("datasets/train.json", train_payload),
            "validation": _binding("datasets/validation.json", validation_payload),
        },
        sources={
            role: _binding(f"sources/{role}.py", payload)
            for role, payload in source_payloads.items()
        },
        runtime_identity=training_runtime,
        trust_policy={
            **_binding("trust/policy.json", trust_payload),
            "semantic_sha256": _sha(trust_payload),
        },
        artifact_root="artifacts/cp-test",
        artifact_root_identity={"st_dev": 1, "st_ino": 2},
        config=config,
    )
    tensors = [
        {
            "key": "model.layers.1.self_attn.o_proj.lora_a",
            "shape": [4, 2],
            "dtype": "float32",
        },
        {
            "key": "model.layers.1.self_attn.o_proj.lora_b",
            "shape": [2, 4],
            "dtype": "float32",
        },
    ]
    if manifest_schema == MANIFEST_SCHEMA:
        for depth in range(2):
            tensors.extend(
                [
                    {
                        "key": f"model.layers.1.self_attn.o_proj.depth_a.{depth}",
                        "shape": [4, 2],
                        "dtype": "float32",
                    },
                    {
                        "key": f"model.layers.1.self_attn.o_proj.depth_b.{depth}",
                        "shape": [2, 4],
                        "dtype": "float32",
                    },
                ]
            )
    adapter_payload = b"synthetic-adapter-safetensors"
    optimizer_payload = b"synthetic-optimizer-safetensors"
    state = {
        **authority_state_bindings(authority),
        "checkpoint_sequence": 9,
        "step": 8,
        "optimizer_updates": 8,
        "epoch": 8,
        "cursor": 0,
        "order": [0],
        "order_sha256": order_sha256(order=[0], seed=config.seed, epoch=8),
        "sampler": config.sampler,
        "seed": config.seed,
        "train_example_count": 1,
        "validation_example_count": 1,
        "elapsed_training_s": 10.0,
        "invocation_count": 2,
        "sample_history_sha256": "1" * 64,
        "initial_adapter_sha256": "2" * 64,
        "adapter_topology_sha256": topology_sha256(tensors),
        "loss_trail": [{"step": 8, "mean_loss": 1.0}],
        "validation_trail": [{"step": 8, "mean_loss": 1.25}],
        "pending_losses": [],
        "baseline_validation": {"mean_loss": 2.0, "examples": 1},
        "last_step_committed": True,
        "terminal": True,
        "halt_reason": "max_steps",
    }
    checkpoint_id = "sequence-00000009-step-00000008-test"
    complete = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_id": checkpoint_id,
        "created_at_unix": 1.0,
        "state": state,
        "adapter": _binding("adapter.safetensors", adapter_payload),
        "optimizer": _binding("optimizer.safetensors", optimizer_payload),
    }
    complete_payload = _json(complete)
    pointer = {
        "schema": POINTER_SCHEMA,
        "checkpoint": f"checkpoints/{checkpoint_id}",
        "checkpoint_sequence": 9,
        "complete_sha256": _sha(complete_payload),
    }
    controller_body = {
        "schema": CONTROLLER_COMPLETION_SCHEMA,
        "authority_sha256": authority["authority_sha256"],
        "bootstrap_complete": True,
        "base_checkpoint_immutable": True,
        "checkpoint": {"complete_sha256": _sha(complete_payload)},
    }
    controller = {
        **controller_body,
        "completion_sha256": sha256_json(controller_body),
    }
    invocation_body = {
        "schema": INVOCATION_SCHEMA,
        "authority_sha256": authority["authority_sha256"],
        "checkpoint_complete_sha256": _sha(complete_payload),
        "bootstrap_complete": True,
        "base_checkpoint_immutable": True,
        "base_checkpoint_before": base,
        "base_checkpoint_after": base,
    }
    invocation = {
        **invocation_body,
        "receipt_sha256": sha256_json(invocation_body),
    }
    artifacts = {
        "evidence/authority.json": _json(authority),
        "adapter.safetensors": adapter_payload,
        "optimizer.safetensors": optimizer_payload,
        "datasets/train.json": train_payload,
        "datasets/validation.json": validation_payload,
        "config/execution-spec.json": spec_payload,
        "trust/policy.json": trust_payload,
        "checkpoint/latest.json": _json(pointer),
        "checkpoint/complete.json": complete_payload,
        "evidence/controller-completion.json": _json(controller),
        "evidence/terminal-invocation.json": _json(invocation),
        **{f"sources/{role}.py": payload for role, payload in source_payloads.items()},
    }
    bindings = {
        "authority": _binding("evidence/authority.json", artifacts["evidence/authority.json"]),
        "adapter": _binding("adapter.safetensors", adapter_payload),
        "optimizer": _binding("optimizer.safetensors", optimizer_payload),
        "train_dataset": _binding("datasets/train.json", train_payload),
        "validation_dataset": _binding("datasets/validation.json", validation_payload),
        "execution_spec": _binding("config/execution-spec.json", spec_payload),
        "trust_policy": _binding("trust/policy.json", trust_payload),
        "checkpoint_pointer": _binding(
            "checkpoint/latest.json", artifacts["checkpoint/latest.json"]
        ),
        "checkpoint_complete": _binding("checkpoint/complete.json", complete_payload),
        "controller_completion": _binding(
            "evidence/controller-completion.json",
            artifacts["evidence/controller-completion.json"],
        ),
        "terminal_invocation": _binding(
            "evidence/terminal-invocation.json",
            artifacts["evidence/terminal-invocation.json"],
        ),
        "source_snapshots": {
            role: _binding(f"sources/{role}.py", payload)
            for role, payload in source_payloads.items()
        },
    }
    lora = {
        "rank": 2,
        "scale": 20.0,
        "dropout": 0.0,
        "layers": 1,
        "targets": ["o_proj"],
        "wrapped_projections": 1,
        "projection_paths": ["model.layers.1.self_attn.o_proj"],
        "trainable_params": sum(
            dimension
            for tensor in tensors
            for dimension in [tensor["shape"][0] * tensor["shape"][1]]
        ),
    }
    if manifest_schema == MANIFEST_SCHEMA:
        lora.update(
            {
                "conditioning_schema": "aura.depth_conditioned_lora.v1",
                "depth_bank_size": 2,
            }
        )
    manifest = {
        "schema": manifest_schema,
        "adapter_id": "resident-sft-test",
        "training_protocol": TRAINING_AUTHORITY,
        "base_checkpoint": base,
        "model_behavior_bundle": behavior,
        "personality_adapter": personality,
        "training_runtime": training_runtime,
        "bindings": bindings,
        "lora": lora,
        "tensors": copy.deepcopy(tensors),
        "claim_boundary": {
            "training_objective_learned": True,
            "reasoning_gain_proven": False,
            "causal_gain_proven": False,
            "frontier_level_proven": False,
            "promotion_allowed": False,
        },
    }
    bundle = _Bundle(
        manifest=manifest,
        artifacts=artifacts,
        tensors=tensors,
        base=base,
        behavior=behavior,
        personality=personality,
        evaluation_runtime=evaluation_runtime,
    )
    bundle.sync_package_completion()
    return bundle


def test_exact_resident_sft_package_is_accepted() -> None:
    receipt = _bundle().validate()

    assert receipt["adapter_id"] == "resident-sft-test"
    assert receipt["terminal_step"] == 8
    assert receipt["training_objective_learned"] is True
    assert receipt["reasoning_gain_proven"] is False
    assert receipt["promotion_allowed"] is False


def test_legacy_shared_operator_package_remains_verifiable() -> None:
    receipt = _bundle(manifest_schema=LEGACY_MANIFEST_SCHEMA).validate()

    assert receipt["complete"] is True
    assert receipt["reasoning_gain_proven"] is False


@pytest.mark.parametrize(
    "role",
    [
        "adapter",
        "optimizer",
        "authority",
        "checkpoint_pointer",
        "checkpoint_complete",
        "train_dataset",
        "validation_dataset",
        "execution_spec",
        *(f"source_{role}" for role in sorted(REQUIRED_SOURCE_ROLES)),
    ],
)
def test_bound_artifact_tampering_is_rejected(role: str) -> None:
    bundle = _bundle()
    bindings = bundle.manifest["bindings"]
    if role.startswith("source_"):
        path = bindings["source_snapshots"][role.removeprefix("source_")]["path"]
    else:
        path = bindings[role]["path"]
    payload = bundle.artifacts[path]
    bundle.artifacts[path] = bytes([payload[0] ^ 1]) + payload[1:]

    with pytest.raises(
        ResidentRecurrentSFTAdapterIdentityError,
        match=rf"resident_sft_adapter_{role}_sha256_mismatch",
    ):
        bundle.validate()


def test_controller_completion_digest_tampering_survives_outer_rebind_but_fails() -> None:
    bundle = _bundle()
    controller = copy.deepcopy(json.loads(bundle.artifacts["evidence/controller-completion.json"]))
    controller["bootstrap_complete"] = False
    bundle.replace_artifact("controller_completion", _json(controller))

    with pytest.raises(
        ResidentRecurrentSFTAdapterIdentityError,
        match="resident_sft_adapter_completion_evidence_invalid",
    ):
        bundle.validate()


def test_terminal_invocation_tampering_survives_outer_rebind_but_fails() -> None:
    bundle = _bundle()
    invocation = copy.deepcopy(json.loads(bundle.artifacts["evidence/terminal-invocation.json"]))
    invocation["base_checkpoint_after"]["fingerprint"] = "0" * 64
    bundle.replace_artifact("terminal_invocation", _json(invocation))

    with pytest.raises(
        ResidentRecurrentSFTAdapterIdentityError,
        match="resident_sft_adapter_completion_evidence_invalid",
    ):
        bundle.validate()


def test_runtime_compatibility_drift_is_rejected() -> None:
    bundle = _bundle()
    bundle.evaluation_runtime = _runtime({"python": "3.13.0", "mlx": "0.test"})

    with pytest.raises(
        ResidentRecurrentSFTAdapterIdentityError,
        match="resident_sft_adapter_effective_stack_mismatch",
    ):
        bundle.validate()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda bundle: bundle.manifest["lora"].update(rank=3),
        lambda bundle: bundle.manifest["lora"].update(
            projection_paths=["model.layers.2.self_attn.o_proj"]
        ),
    ],
)
def test_lora_topology_tampering_is_rejected(
    mutate: Callable[[_Bundle], None],
) -> None:
    bundle = _bundle()
    mutate(bundle)

    with pytest.raises(ResidentRecurrentSFTAdapterIdentityError):
        bundle.validate()


def test_tensor_inventory_tampering_is_rejected() -> None:
    bundle = _bundle()
    bundle.tensors[0]["shape"] = [8, 2]

    with pytest.raises(
        ResidentRecurrentSFTAdapterIdentityError,
        match="resident_sft_adapter_tensor_inventory_mismatch",
    ):
        bundle.validate()


def test_claim_boundary_cannot_be_strengthened() -> None:
    bundle = _bundle()
    bundle.manifest["claim_boundary"]["reasoning_gain_proven"] = True

    with pytest.raises(
        ResidentRecurrentSFTAdapterIdentityError,
        match="resident_sft_adapter_claim_boundary_invalid",
    ):
        bundle.validate()


def _refresh_checkpoint_chain(bundle: _Bundle, complete: dict[str, Any]) -> None:
    complete_payload = _json(complete)
    bundle.replace_artifact("checkpoint_complete", complete_payload)
    complete_sha = _sha(complete_payload)

    pointer = json.loads(bundle.artifacts["checkpoint/latest.json"])
    pointer["complete_sha256"] = complete_sha
    bundle.replace_artifact("checkpoint_pointer", _json(pointer))

    controller = json.loads(bundle.artifacts["evidence/controller-completion.json"])
    controller["checkpoint"]["complete_sha256"] = complete_sha
    controller_body = dict(controller)
    controller_body.pop("completion_sha256")
    controller["completion_sha256"] = sha256_json(controller_body)
    bundle.replace_artifact("controller_completion", _json(controller))

    invocation = json.loads(bundle.artifacts["evidence/terminal-invocation.json"])
    invocation["checkpoint_complete_sha256"] = complete_sha
    invocation_body = dict(invocation)
    invocation_body.pop("receipt_sha256")
    invocation["receipt_sha256"] = sha256_json(invocation_body)
    bundle.replace_artifact("terminal_invocation", _json(invocation))
    bundle.sync_package_completion()


def test_recommitted_nonterminal_state_is_rejected() -> None:
    bundle = _bundle()
    complete = json.loads(bundle.artifacts["checkpoint/complete.json"])
    complete["state"]["terminal"] = False
    complete["state"]["halt_reason"] = None
    _refresh_checkpoint_chain(bundle, complete)

    with pytest.raises(
        ResidentRecurrentSFTAdapterIdentityError,
        match="resident_sft_adapter_terminal_checkpoint_invalid",
    ):
        bundle.validate()


def test_recommitted_terminal_step_drift_is_rejected() -> None:
    bundle = _bundle()
    complete = json.loads(bundle.artifacts["checkpoint/complete.json"])
    complete["state"]["step"] = 7
    complete["state"]["optimizer_updates"] = 7
    complete["state"]["epoch"] = 7
    complete["state"]["order_sha256"] = order_sha256(
        order=[0], seed=complete["state"]["seed"], epoch=7
    )
    _refresh_checkpoint_chain(bundle, complete)

    with pytest.raises(
        ResidentRecurrentSFTAdapterIdentityError,
        match="resident_sft_adapter_terminal_checkpoint_invalid",
    ):
        bundle.validate()
