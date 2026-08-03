from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignJournal,
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
    full_weight_checkpoint_identity,
    model_behavior_bundle_identity,
)
from core.brain.llm.latent_cortex.resident_recurrent_sft_adapter_identity import (
    MANIFEST_SCHEMA,
    declared_bindings,
)
from core.learning.recurrence_curriculum import RECURRENCE_TRAINING_FAMILIES
from core.learning.recurrence_native_objective_v5 import (
    GeneratedRollinBranchEvidence,
    GeneratedRollinLivePathEvaluation,
    GeneratedRollinSelectionConfig,
)
from core.learning.recurrent_sft_execution import adapter_tensor_fingerprint
from core.learning.resident_recurrent_sft_bootstrap_authority import (
    REQUIRED_SOURCE_ROLES,
    ResidentSFTBootstrapConfig,
    build_authority,
    build_dataset_commitment,
    canonical_dataset_payloads,
    sha256_bytes,
    sha256_json,
)
from core.learning.resident_recurrent_sft_bootstrap_execution import (
    adapter_topology_sha256,
)
from core.learning.resident_recurrent_sft_bootstrap_state import (
    authority_state_bindings,
    order_sha256,
    save_checkpoint,
)
from tools import materialize_resident_recurrent_sft_adapter as materializer
from tools.resident_recurrent_sft_bootstrap_identity import absent_personality_identity


@dataclass(frozen=True)
class CampaignFixture:
    capsule: Path
    campaign: Path
    model: Path
    destination: Path
    evaluation_runtime: dict[str, Any]


def _objective_record() -> dict[str, Any]:
    evaluation = GeneratedRollinLivePathEvaluation(
        value=0.5 - 0.5 * math.log((1.0 + math.exp(-1.0)) / 2.0),
        branches=(
            GeneratedRollinBranchEvidence(
                branch_index=0,
                branch_seed=11,
                loss=0.5,
                selection_weight=0.7310585786300049,
                generated_tokens_sha256="1" * 64,
                effective_rollin_sha256="2" * 64,
                student_forced_positions=(0,),
            ),
            GeneratedRollinBranchEvidence(
                branch_index=1,
                branch_seed=12,
                loss=1.0,
                selection_weight=0.2689414213699951,
                generated_tokens_sha256="3" * 64,
                effective_rollin_sha256="4" * 64,
                student_forced_positions=(0,),
            ),
        ),
        answer_token_count=2,
        execution_spec_sha256="5" * 64,
        prompt_tokens_sha256="6" * 64,
        answer_tokens_sha256="7" * 64,
        bridge_tokens_sha256="8" * 64,
        config=GeneratedRollinSelectionConfig(
            branch_softmin_temperature=0.5,
        ),
        base_seed=10,
    )
    # Softmin(0.5, 1.0; tau=.5), not the weighted arithmetic mean.
    receipt = evaluation.receipt()
    return {
        "loss": receipt["value"],
        "branch_values": [branch["loss"] for branch in receipt["branches"]],
        "branch_weights": [
            branch["selection_weight"] for branch in receipt["branches"]
        ],
        "rollin_base_seed": receipt["base_seed"],
        "execution_spec_sha256": receipt["execution_spec_sha256"],
        "objective_receipt_sha256": receipt["receipt_sha256"],
        "objective_receipt": receipt,
    }


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _binding(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": sha256_bytes(payload), "size_bytes": len(payload)}


def _row(task_id: str, prompt: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "family": sorted(RECURRENCE_TRAINING_FAMILIES)[0],
        "depth": 2,
        "prompt": prompt,
        "answer": 'FINAL_ANSWER: {"value":1}',
    }


def _runtime_identities() -> tuple[dict[str, Any], dict[str, Any]]:
    evaluation_body = {
        "python": "3.12.test",
        "platform_system": "Darwin",
        "platform_release": "test",
        "platform_machine": "arm64",
        "dependencies": {},
    }
    evaluation = {**evaluation_body, "identity_sha256": sha256_json(evaluation_body)}
    training_body = {
        **evaluation_body,
        "interpreter": {
            "schema": "aura.resident_recurrent_sft_python.v1",
            "executable": "/test/python",
            "real_executable": "/test/python",
            "sys_prefix": "/test",
            "base_prefix": "/test",
            "sha256": "a" * 64,
            "size_bytes": 1,
        },
    }
    training = {**training_body, "identity_sha256": sha256_json(training_body)}
    return training, evaluation


def _adapter_tensors(value: float, *, depth_bank_size: int = 0) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for layer in range(40, 48):
        for target in ("q_proj", "v_proj", "o_proj"):
            projection = f"model.layers.{layer}.self_attn.{target}"
            tensors[f"{projection}.lora_a"] = mx.full((2, 8), value)
            tensors[f"{projection}.lora_b"] = mx.full((8, 2), value + 0.25)
            for depth in range(depth_bank_size):
                tensors[f"{projection}.depth_a.{depth}"] = mx.full((2, 8), value)
                tensors[f"{projection}.depth_b.{depth}"] = mx.full(
                    (8, 2), value + 0.25
                )
    return tensors


def _state(
    authority: dict[str, Any],
    *,
    sequence: int,
    step: int,
    invocation_count: int,
    initial_sha256: str,
    current_sha256: str,
    topology_sha256: str,
    terminal: bool,
) -> dict[str, Any]:
    loss_trail = []
    validation_trail = []
    if step:
        loss_trail = [
            {
                "step": 1,
                "epoch": 0,
                "cursor": 1,
                "example_id": "b" * 64,
                "loss": 1.0,
                "branch_values": [1.0, 1.0],
                "adapter_before_sha256": initial_sha256,
                "adapter_after_sha256": current_sha256,
            }
        ]
        validation_trail = [{"step": 1, "mean_loss": 1.0, "examples": 1}]
    return {
        **authority_state_bindings(authority),
        "checkpoint_sequence": sequence,
        "step": step,
        "optimizer_updates": step,
        "epoch": step,
        "cursor": 0,
        "order": [0],
        "order_sha256": order_sha256(order=[0], seed=authority["trainer"]["seed"], epoch=step),
        "sampler": authority["trainer"]["sampler"],
        "seed": authority["trainer"]["seed"],
        "train_example_count": 1,
        "validation_example_count": 1,
        "elapsed_training_s": float(step + 1),
        "invocation_count": invocation_count,
        "sample_history_sha256": ("0" * 64 if step == 0 else "c" * 64),
        "initial_adapter_sha256": initial_sha256,
        "adapter_topology_sha256": topology_sha256,
        "loss_trail": loss_trail,
        "validation_trail": validation_trail,
        "pending_losses": [],
        "baseline_validation": {"mean_loss": 2.0, "examples": 1},
        "last_step_committed": True,
        "terminal": terminal,
        "halt_reason": "max_steps" if terminal else None,
    }


def _build_campaign(tmp_path: Path, *, depth_bank_size: int = 0) -> CampaignFixture:
    capsule = tmp_path / "capsule"
    campaign = capsule / "artifacts" / "cp796"
    training = campaign / "training"
    controller = campaign / "controller"
    inputs = campaign / "inputs"
    model = capsule / "model"
    for path in (training, controller, inputs, model):
        path.mkdir(parents=True, exist_ok=True)

    model_config = {"num_hidden_layers": 64}
    _write(model / "config.json", canonical_json_bytes(model_config))
    _write(model / "tokenizer.json", b"{}")
    _write(model / "tokenizer_config.json", b"{}")
    _write(model / "model.safetensors", b"test-model-weights")
    base_identity = full_weight_checkpoint_identity(model)
    behavior_identity = model_behavior_bundle_identity(model)

    spec_source = Path("config/latent_cortex/resident_32b_recurrent_grpo_execution_spec.json")
    spec_payload = spec_source.read_bytes()
    spec = RLCExecutionSpec.from_dict(json.loads(spec_payload))
    _write(capsule / "execution-spec.json", spec_payload)
    trust_payload = canonical_json_bytes(
        {"schema": "aura.test.trust.v1", "decision": "training-only"}
    )
    _write(inputs / "trust-policy.json", trust_payload)
    train_rows = [_row("train.1", "Solve training recurrence.")]
    validation_rows = [_row("validation.1", "Solve heldout recurrence.")]
    train_payload, validation_payload = canonical_dataset_payloads(train_rows, validation_rows)
    _write(inputs / "train.json", train_payload)
    _write(inputs / "validation.json", validation_payload)
    source_payloads = {
        role: f"source snapshot: {role}\n".encode("ascii") for role in REQUIRED_SOURCE_ROLES
    }
    for role, payload in source_payloads.items():
        _write(capsule / "sources" / f"{role}.py", payload)
    training_runtime, evaluation_runtime = _runtime_identities()
    now = datetime.now(UTC)
    config = ResidentSFTBootstrapConfig(
        seed=7,
        lora_initialization_seed=11,
        max_steps=1,
        max_invocation_steps=1,
        evaluate_every=1,
        validation_examples=1,
        memory_fraction=0.42,
    )
    authority = build_authority(
        campaign_id="resident-32b-recurrent-sft-bootstrap-cp796-test",
        campaign_scope="full_bootstrap",
        committed_at=now.isoformat(),
        expires_at=(now + timedelta(days=1)).isoformat(),
        model_path="model",
        model_identity=base_identity,
        behavior_identity=behavior_identity,
        personality_identity=absent_personality_identity(),
        tokenizer_identity={
            "identity_sha256": "d" * 64,
            "artifact_sha256": "e" * 64,
            "runtime_sha256": "f" * 64,
        },
        execution_spec={
            **_binding("execution-spec.json", spec_payload),
            "semantic_sha256": spec.sha256,
        },
        dataset=build_dataset_commitment(train_rows, validation_rows),
        dataset_artifacts={
            "train": _binding("artifacts/cp796/inputs/train.json", train_payload),
            "validation": _binding("artifacts/cp796/inputs/validation.json", validation_payload),
        },
        sources={
            role: _binding(f"sources/{role}.py", payload)
            for role, payload in source_payloads.items()
        },
        runtime_identity=training_runtime,
        trust_policy={
            **_binding("artifacts/cp796/inputs/trust-policy.json", trust_payload),
            "semantic_sha256": sha256_bytes(trust_payload),
        },
        artifact_root="artifacts/cp796/training",
        artifact_root_identity={
            "st_dev": training.stat().st_dev,
            "st_ino": training.stat().st_ino,
        },
        config=config,
    )
    _write(inputs / "authority.json", canonical_json_bytes(authority))

    initial = _adapter_tensors(0.0, depth_bank_size=depth_bank_size)
    terminal = _adapter_tensors(1.0, depth_bank_size=depth_bank_size)
    initial_sha = adapter_tensor_fingerprint(initial)
    terminal_sha = adapter_tensor_fingerprint(terminal)
    topology = adapter_topology_sha256(initial)
    save_checkpoint(
        training,
        adapter_tensors=initial,
        optimizer_tensors={"optimizer.state": mx.array([0.0])},
        state=_state(
            authority,
            sequence=1,
            step=0,
            invocation_count=1,
            initial_sha256=initial_sha,
            current_sha256=initial_sha,
            topology_sha256=topology,
            terminal=False,
        ),
    )
    save_checkpoint(
        training,
        adapter_tensors=terminal,
        optimizer_tensors={"optimizer.state": mx.array([1.0])},
        state=_state(
            authority,
            sequence=2,
            step=1,
            invocation_count=1,
            initial_sha256=initial_sha,
            current_sha256=terminal_sha,
            topology_sha256=topology,
            terminal=True,
        ),
    )
    pointer = json.loads((training / "latest.json").read_bytes())
    invocation_body = {
        "schema": "aura.resident_recurrent_sft_bootstrap_invocation.v1",
        "campaign_id": authority["campaign_id"],
        "campaign_scope": "full_bootstrap",
        "authority_sha256": authority["authority_sha256"],
        "invocation_count": 1,
        "required_end_step": 1,
        "step": 1,
        "max_steps": 1,
        "checkpoint_sequence": 2,
        "checkpoint_complete_sha256": pointer["complete_sha256"],
        "terminal": True,
        "halt_reason": "max_steps",
        "canary_lifecycle_complete": False,
        "bootstrap_complete": True,
        "base_checkpoint_before": base_identity,
        "base_checkpoint_after": base_identity,
        "base_checkpoint_immutable": True,
        "claim_state": {
            "resident_sft_complete": True,
            "causal_gain_proven": False,
            "reasoning_gain_proven": False,
            "frontier_level_proven": False,
            "grpo_admission": False,
            "promotion_allowed": False,
        },
    }
    invocation = {**invocation_body, "receipt_sha256": sha256_json(invocation_body)}
    _write(training / "invocation-0001.json", canonical_json_bytes(invocation))
    status_body = {
        "schema": "aura.resident_recurrent_sft_bootstrap_status.v1",
        "authority_sha256": authority["authority_sha256"],
        "step": 1,
        "max_steps": 1,
        "latest_invocation": 1,
        "latest_receipt_sha256": invocation["receipt_sha256"],
        "terminal": True,
        "halt_reason": "max_steps",
    }
    _write(
        training / "status.json",
        canonical_json_bytes({**status_body, "status_sha256": sha256_json(status_body)}),
    )

    plan = CampaignPlan.build(
        authority["campaign_id"],
        [{"invocation_ordinal": 1, "expected_start_step": 0, "required_end_step": 1}],
        metadata={"strict_execution_order": True},
    )
    plan_payload = canonical_json_bytes(plan.to_dict())
    _write(inputs / "campaign-plan.json", plan_payload)
    with CampaignJournal(controller / "campaign.journal.jsonl", plan) as journal:
        cell_id = plan.cell_ids[0]
        attempt_id = journal.start_cell(cell_id)
        journal.record_arm_result(cell_id, attempt_id, {"step": 1})
        journal.record_verified(cell_id, attempt_id, {"verified": True})
        journal.commit_cell(cell_id, attempt_id, {"step": 1})
        campaign_manifest = journal.finalize(controller / "campaign-manifest.json")
    config_body = {
        "schema": "aura.resident_recurrent_sft_controller_config.v1",
        "campaign_id": authority["campaign_id"],
        "profile": "full",
        "source": {"branch": "main", "commit": "1" * 40, "origin_main": "1" * 40},
        "authority": {
            **_binding("artifacts/cp796/inputs/authority.json", canonical_json_bytes(authority)),
            "semantic_sha256": authority["authority_sha256"],
        },
        "plan": {
            **_binding("artifacts/cp796/inputs/campaign-plan.json", plan_payload),
            "semantic_sha256": plan.plan_sha256,
        },
        "paths": {},
        "path_custody": {},
        "path_custody_threat_model": {},
        "watchdog": {},
        "launch": {},
        "claim_state": {},
    }
    _write(
        campaign / "controller-config.json",
        canonical_json_bytes({**config_body, "config_sha256": sha256_json(config_body)}),
    )
    completion_body = {
        "schema": "aura.resident_recurrent_sft_controller_completion.v1",
        "campaign_id": authority["campaign_id"],
        "config_sha256": sha256_json(config_body),
        "authority_sha256": authority["authority_sha256"],
        "plan_sha256": plan.plan_sha256,
        "journal_manifest_sha256": campaign_manifest["manifest_sha256"],
        "checkpoint": {
            "present": True,
            "step": 1,
            "checkpoint_sequence": 2,
            "invocation_count": 1,
            "terminal": True,
            "halt_reason": "max_steps",
            "complete_sha256": pointer["complete_sha256"],
            "model_identity_sha256": base_identity["fingerprint"],
        },
        "campaign_scope": "full_bootstrap",
        "canary_lifecycle_complete": False,
        "bootstrap_complete": True,
        "base_checkpoint_immutable": True,
        "post_training_gates_required": True,
        "execution_supervision": {},
        "claim_state": {},
        "claims_supported": ["resident_recurrent_sft_bootstrap_completed"],
    }
    _write(
        controller / "completion-receipt.json",
        canonical_json_bytes(
            {**completion_body, "completion_sha256": sha256_json(completion_body)}
        ),
    )
    return CampaignFixture(
        capsule=capsule,
        campaign=campaign,
        model=model,
        destination=tmp_path / "frozen-adapter",
        evaluation_runtime=evaluation_runtime,
    )


def _materialize(
    fixture: CampaignFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    monkeypatch.setattr(
        materializer,
        "runtime_environment_identity",
        lambda: fixture.evaluation_runtime,
    )
    return materializer.materialize_resident_recurrent_sft_adapter(
        campaign_root=fixture.campaign,
        source_capsule_root=fixture.capsule,
        destination=fixture.destination,
        adapter_id="resident-32b-recurrent-sft-cp796-test",
        model_path=fixture.model,
    )


def test_derived_completion_and_admission_are_canonical_and_cycle_free() -> None:
    authority = {"authority_sha256": "a" * 64}
    state = {"step": 96, "checkpoint_sequence": 97, "invocation_count": 24}
    manifest_payload = canonical_json_bytes({"schema": "manifest"})
    completion = {
        "schema": materializer.PACKAGE_COMPLETION_SCHEMA,
        "complete": True,
        "halt_reason": "max_steps",
        "step": 96,
        "adapter_sha256": "b" * 64,
        "checkpoint_complete_sha256": "c" * 64,
        "authority_sha256": "a" * 64,
        "manifest_sha256": sha256_bytes(manifest_payload),
    }
    completion_payload = canonical_json_bytes(completion)

    assert (
        materializer._validate_completion(
            completion_payload,
            manifest_payload=manifest_payload,
            authority=authority,
            state=state,
            adapter_sha256="b" * 64,
            checkpoint_complete_sha256="c" * 64,
        )
        == completion
    )

    admission = materializer._training_admission(
        identity_receipt={"schema": "identity", "promotion_allowed": False},
        authority=authority,
        state=state,
        adapter_sha256="b" * 64,
        checkpoint_complete_sha256="c" * 64,
    )
    body = dict(admission)
    claimed = body.pop("admission_sha256")
    assert canonical_json_bytes(admission) == canonical_json_bytes(
        json.loads(canonical_json_bytes(admission))
    )
    assert admission["schema"] == materializer.TRAINING_ADMISSION_SCHEMA
    assert admission["identity_receipt"]["complete"] is True
    assert admission["identity_receipt"]["load_eligible"] is True
    assert admission["identity_receipt"]["training_scope"] == "resident_recurrent_sft"
    assert not any(admission["claim_flags"].values())
    assert claimed == sha256_json(body)


def test_materializes_validated_atomic_package_and_adjacent_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_campaign(tmp_path)
    result = _materialize(fixture, monkeypatch)

    manifest_payload = (fixture.destination / "recurrence_adapter_manifest.json").read_bytes()
    manifest = json.loads(manifest_payload)
    completion_payload = (fixture.destination / "training_completion.json").read_bytes()
    completion = json.loads(completion_payload)
    admission_path = Path(result["training_admission_path"])
    admission_payload = admission_path.read_bytes()
    admission = json.loads(admission_payload)

    assert manifest_payload == canonical_json_bytes(manifest)
    assert completion_payload == canonical_json_bytes(completion)
    assert completion["schema"] == materializer.PACKAGE_COMPLETION_SCHEMA
    assert completion["complete"] is True
    assert completion["manifest_sha256"] == sha256_bytes(manifest_payload)
    assert "training_completion.json" not in {
        binding["path"] for _role, binding in declared_bindings(manifest)
    }
    assert "training_admission.json" not in {
        binding["path"] for _role, binding in declared_bindings(manifest)
    }
    assert admission_path.parent == fixture.destination.parent
    assert not (fixture.destination / admission_path.name).exists()
    assert admission_payload == canonical_json_bytes(admission)
    assert admission["decision"] == "admit_to_freeze_and_mechanics"
    assert admission["identity_receipt"]["complete"] is True
    assert admission["identity_receipt"]["load_eligible"] is True
    assert not any(admission["claim_flags"].values())
    body = dict(admission)
    claimed = body.pop("admission_sha256")
    assert claimed == sha256_json(body)
    assert result["checkpoint_generation_count"] == 2
    assert len(manifest["tensors"]) == 48


def test_materializes_depth_conditioned_tensor_bank_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_campaign(tmp_path, depth_bank_size=2)
    _materialize(fixture, monkeypatch)

    manifest = json.loads(
        (fixture.destination / "recurrence_adapter_manifest.json").read_bytes()
    )

    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["lora"]["depth_bank_size"] == 2
    assert manifest["lora"]["conditioning_schema"] == "aura.depth_conditioned_lora.v1"
    assert len(manifest["tensors"]) == 144
    assert any(record["key"].endswith(".depth_a.1") for record in manifest["tensors"])
    assert any(record["key"].endswith(".depth_b.1") for record in manifest["tensors"])


def test_refuses_existing_destination_or_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_campaign(tmp_path)
    fixture.destination.mkdir()
    with pytest.raises(
        materializer.ResidentRecurrentSFTMaterializationError, match="destination_exists"
    ):
        _materialize(fixture, monkeypatch)

    fixture.destination.rmdir()
    fixture.destination.with_name(f"{fixture.destination.name}.training_admission.json").write_text(
        "occupied"
    )
    with pytest.raises(
        materializer.ResidentRecurrentSFTMaterializationError,
        match="admission_destination_exists",
    ):
        _materialize(fixture, monkeypatch)


def test_refuses_symlinked_bound_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_campaign(tmp_path)
    train = fixture.campaign / "inputs" / "train.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(train.read_bytes())
    train.unlink()
    train.symlink_to(outside)

    with pytest.raises(
        materializer.ResidentRecurrentSFTMaterializationError,
        match="train_dataset_symlink_forbidden",
    ):
        _materialize(fixture, monkeypatch)
    assert not fixture.destination.exists()


def test_refuses_partial_checkpoint_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_campaign(tmp_path)
    checkpoints = fixture.campaign / "training" / "checkpoints"
    first = sorted(checkpoints.iterdir())[0]
    for path in first.iterdir():
        path.unlink()
    first.rmdir()

    with pytest.raises(
        materializer.ResidentRecurrentSFTMaterializationError,
        match="checkpoint_chain_incomplete",
    ):
        _materialize(fixture, monkeypatch)
    assert not fixture.destination.exists()


def test_materializer_replays_embedded_generated_rollin_evidence() -> None:
    record = _objective_record()
    materializer._verify_objective_record(record)

    record["branch_weights"][0] += 0.01
    with pytest.raises(
        materializer.ResidentRecurrentSFTMaterializationError,
        match="objective_receipt_drift",
    ):
        materializer._verify_objective_record(record)


def test_refuses_rehashed_historical_prefix_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_campaign(tmp_path)
    checkpoints = fixture.campaign / "training" / "checkpoints"
    first_complete = sorted(checkpoints.iterdir())[0] / "complete.json"
    record = json.loads(first_complete.read_bytes())
    record["state"]["baseline_validation"]["mean_loss"] = 99.0
    first_complete.write_bytes(canonical_json_bytes(record))

    with pytest.raises(
        materializer.ResidentRecurrentSFTMaterializationError,
        match="checkpoint_prefix_drift",
    ):
        _materialize(fixture, monkeypatch)
    assert not fixture.destination.exists()
