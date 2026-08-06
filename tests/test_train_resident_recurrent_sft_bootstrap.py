from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest
from mlx_lm.models.qwen2 import Model, ModelArgs

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.recurrence_curriculum import RECURRENCE_TRAINING_FAMILIES
from core.learning.recurrence_native_objective_v5 import (
    GeneratedRollinSelectionConfig,
)
from core.learning.recurrence_native_objective_v6 import BranchSpecializationConfig
from core.learning.recurrent_sft_execution import adapter_tensor_fingerprint
from core.learning.resident_recurrent_sft_bootstrap_authority import (
    OBJECTIVE_NAME_V2,
    OBJECTIVE_NAME_V3,
    REQUIRED_SOURCE_ROLES,
    TRAINER_CONFIG_SCHEMA_V2,
    TRAINER_CONFIG_SCHEMA_V3,
    TRAINER_CONFIG_SCHEMA_V4,
    ResidentSFTBootstrapConfig,
    build_authority,
    build_dataset_commitment,
    canonical_dataset_payloads,
    sha256_bytes,
    sha256_json,
)
from core.learning.resident_recurrent_sft_bootstrap_execution import (
    family_depth_balanced_order,
    initial_sample_history,
)
from core.learning.resident_recurrent_sft_bootstrap_state import (
    BINDING_ROLES,
    authority_state_bindings,
    load_checkpoint,
    validate_checkpoint_state,
)
from tools import train_resident_recurrent_sft_bootstrap as trainer


def _bindings() -> dict[str, str]:
    return {role: sha256_json(role) for role in BINDING_ROLES}


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "example_id": sha256_json(f"row:{index}"),
            "family": "logic" if index != 1 else "code",
            "depth": index + 1,
            "prompt_tokens": [10 + index, 20 + index],
            "answer_tokens": [30 + index, 3],
            "bridge_tokens": [],
        }
        for index in range(3)
    ]


def _config(**changes: Any) -> ResidentSFTBootstrapConfig:
    values = {
        "seed": 17,
        "max_steps": 4,
        "max_invocation_steps": 2,
        "evaluate_every": 2,
        "validation_examples": 2,
        "branch_indices": (0,),
    }
    values.update(changes)
    return ResidentSFTBootstrapConfig(**values)


def test_canonical_reader_rejects_equivalent_noncanonical_json() -> None:
    canonical = b'{"a":1,"b":2}'
    assert trainer._read_json_bytes(canonical, role="test") == {"a": 1, "b": 2}

    with pytest.raises(trainer.ResidentSFTBootstrapTrainingError, match="noncanonical"):
        trainer._read_json_bytes(b'{"b": 2, "a": 1}', role="test")


def test_v3_phase_machine_resets_only_after_measured_structural_target() -> None:
    config = ResidentSFTBootstrapConfig(
        seed=17,
        max_steps=5,
        max_invocation_steps=1,
        evaluate_every=1,
        schema=TRAINER_CONFIG_SCHEMA_V3,
        objective=OBJECTIVE_NAME_V3,
        generated_rollin=GeneratedRollinSelectionConfig(),
        branch_specialization=BranchSpecializationConfig(
            weight=8.0,
            target_separation=0.3,
        ),
        structural_warmup_steps=4,
        structural_warmup_learning_rate=1e-4,
        role_conditioned_branches=2,
        branch_indices=(0, 1),
    )
    incomplete = {
        "phase": trainer.STRUCTURAL_WARMUP_PHASE,
        "warmup_target_reached": False,
    }
    complete = {
        "phase": trainer.STRUCTURAL_WARMUP_PHASE,
        "warmup_target_reached": True,
    }

    assert trainer._next_training_phase(config, []) == trainer.STRUCTURAL_WARMUP_PHASE
    assert (
        trainer._next_training_phase(config, [incomplete])
        == trainer.STRUCTURAL_WARMUP_PHASE
    )
    assert trainer._next_training_phase(config, [complete]) == trainer.JOINT_PHASE
    assert (
        trainer._optimizer_phase_from_checkpoint(config, [complete])
        == trainer.STRUCTURAL_WARMUP_PHASE
    )
    with pytest.raises(
        trainer.ResidentSFTBootstrapTrainingError,
        match="warmup_target_not_reached",
    ):
        trainer._next_training_phase(config, [incomplete] * 4)


def test_trainer_releases_materialized_graphs_before_the_next_objective() -> None:
    source = Path(trainer.__file__).read_text(encoding="utf-8")
    resume_load = source.index("loaded = load_checkpoint(")
    resume_release = source.index("del loaded", resume_load)
    training_loop = source.index("while step < config.max_steps", resume_release)
    objective = source.index(
        "result = generated_rollin_specialization_value_and_grad(",
        training_loop,
    )
    optimizer_update = source.index("optimizer.update(model, result.gradients)", objective)
    result_release = source.index("del result", optimizer_update)
    adapter_snapshot = source.index("adapter = adapter_tensor_dict(model)", result_release)
    checkpoint = source.index("save_checkpoint(", adapter_snapshot)
    adapter_release = source.index("del adapter", checkpoint)

    assert resume_load < resume_release < training_loop
    assert optimizer_update < result_release < adapter_snapshot
    assert checkpoint < adapter_release
    assert source.index("mx.clear_cache()", result_release) < adapter_snapshot
    assert source.index("mx.clear_cache()", adapter_release) < source.index(
        "if terminal:", adapter_release
    )


def test_state_document_is_accepted_by_durable_state_contract() -> None:
    rows = _rows()
    config = _config()
    order = family_depth_balanced_order(rows, seed=config.seed, epoch=0)
    state = trainer._state_document(
        _bindings(),
        sequence=1,
        step=0,
        epoch=0,
        cursor=0,
        order=order,
        config=config,
        train_count=len(rows),
        validation_count=2,
        elapsed_s=1.5,
        invocation_count=1,
        sample_history_sha256=initial_sample_history(),
        initial_adapter_sha256="a" * 64,
        adapter_topology_identity="b" * 64,
        loss_trail=[],
        validation_trail=[],
        baseline_validation={"mean_loss": 2.0, "examples": 2},
        terminal=False,
        halt_reason=None,
    )

    validated = validate_checkpoint_state(state)
    assert validated["order"] == order
    assert validated["elapsed_training_s"] == 1.5


def test_validation_summary_binds_exact_objective_results(monkeypatch) -> None:
    rows = _rows()[:2]
    config = _config()
    spec = RLCExecutionSpec(
        branch_roles=("constructive_solution",),
        recurrent_steps=2,
    )

    def evaluate(
        _model,
        prompt_tokens,
        answer_tokens,
        *,
        spec,
        bridge_tokens,
        branch_indices,
    ):
        assert bridge_tokens == []
        assert branch_indices == (0,)
        return SimpleNamespace(
            value=float(prompt_tokens[0]) / 10.0,
            branch_values=(float(prompt_tokens[0]) / 10.0,),
            branch_indices=(0,),
            answer_token_count=len(answer_tokens),
            execution_spec_sha256=spec.sha256,
            prompt_tokens_sha256=sha256_json(prompt_tokens),
            answer_tokens_sha256=sha256_json(answer_tokens),
        )

    monkeypatch.setattr(trainer, "cached_supervised_live_path_loss", evaluate)
    result = trainer._validation_summary(
        object(),
        rows,
        spec=spec,
        config=config,
    )

    assert result["examples"] == 2
    assert result["mean_loss"] == pytest.approx(1.05)
    assert result["receipt_sha256"] == sha256_json(
        {key: value for key, value in result.items() if key != "receipt_sha256"}
    )


def test_v4_intermediate_validation_rotates_over_bound_panel() -> None:
    rows = [
        {
            "example_id": sha256_json(f"validation:{index}"),
            "family": "logic",
            "depth": 2,
            "prompt_tokens": [10 + index],
            "answer_tokens": [30 + index],
            "bridge_tokens": [],
        }
        for index in range(8)
    ]
    config = ResidentSFTBootstrapConfig(
        seed=17,
        max_steps=8,
        max_invocation_steps=2,
        evaluate_every=2,
        validation_examples=8,
        intermediate_validation_examples=2,
        schema=TRAINER_CONFIG_SCHEMA_V4,
        objective=OBJECTIVE_NAME_V3,
        generated_rollin=GeneratedRollinSelectionConfig(),
        branch_specialization=BranchSpecializationConfig(),
        structural_warmup_steps=2,
        structural_warmup_learning_rate=1e-4,
        role_conditioned_branches=2,
        branch_indices=(0, 1),
    )

    full = trainer._validation_selection(
        rows,
        config=config,
        intermediate_cycle=None,
    )
    windows = [
        trainer._validation_selection(
            rows,
            config=config,
            intermediate_cycle=cycle,
        )
        for cycle in range(4)
    ]

    assert [position for position, _index in full] == list(range(8))
    assert [position for window in windows for position, _index in window] == list(
        range(8)
    )
    assert {index for window in windows for _position, index in window} == {
        index for _position, index in full
    }


def test_validation_summary_rejects_objective_identity_drift(monkeypatch) -> None:
    rows = _rows()[:2]
    config = _config()
    spec = RLCExecutionSpec(
        branch_roles=("constructive_solution",),
        recurrent_steps=2,
    )
    monkeypatch.setattr(
        trainer,
        "cached_supervised_live_path_loss",
        lambda *_args, **_kwargs: SimpleNamespace(
            value=1.0,
            branch_values=(1.0,),
            answer_token_count=2,
            execution_spec_sha256="d" * 64,
            prompt_tokens_sha256="e" * 64,
            answer_tokens_sha256="f" * 64,
        ),
    )

    with pytest.raises(trainer.ResidentSFTBootstrapTrainingError, match="objective_drift"):
        trainer._validation_summary(object(), rows, spec=spec, config=config)


def test_sampling_receipt_publication_is_idempotent_and_detects_drift(
    tmp_path: Path,
) -> None:
    rows = _rows()
    order = family_depth_balanced_order(rows, seed=17, epoch=0)
    first = trainer._publish_sampling_receipt(
        tmp_path,
        rows,
        order,
        seed=17,
        epoch=0,
    )
    replay = trainer._publish_sampling_receipt(
        tmp_path,
        rows,
        order,
        seed=17,
        epoch=0,
    )
    assert first == replay

    path = tmp_path / "sampling" / "epoch-00000000.json"
    path.write_bytes(b"tampered")
    with pytest.raises(trainer.ResidentSFTBootstrapTrainingError, match="receipt_drift"):
        trainer._publish_sampling_receipt(
            tmp_path,
            rows,
            order,
            seed=17,
            epoch=0,
        )


def test_invocation_receipt_is_nonpromotable_and_refuses_base_drift(
    tmp_path: Path,
) -> None:
    authority = {
        "authority_sha256": "a" * 64,
        "campaign_id": "resident-32b-recurrent-sft-bootstrap-cp-test",
        "campaign_scope": "full_bootstrap",
        "trainer": {"max_steps": 4},
    }
    state = {
        "invocation_count": 1,
        "checkpoint_sequence": 5,
        "step": 4,
        "terminal": True,
    }
    base = {"fingerprint": "b" * 64, "method": "sha256", "files": 1}

    receipt = trainer._publish_receipt(
        tmp_path,
        authority=authority,
        state=state,
        checkpoint_sha256="c" * 64,
        base_before=base,
        base_after=base,
        halt_reason="max_steps",
        required_end_step=4,
    )

    assert receipt["bootstrap_complete"] is True
    assert receipt["claim_state"]["resident_sft_complete"] is True
    assert receipt["claim_state"]["reasoning_gain_proven"] is False
    assert receipt["claim_state"]["grpo_admission"] is False
    assert json.loads((tmp_path / "status.json").read_text())["terminal"] is True

    canary_authority = {**authority, "campaign_scope": "canary_lifecycle"}
    canary_receipt = trainer._publish_receipt(
        tmp_path / "canary",
        authority=canary_authority,
        state=state,
        checkpoint_sha256="c" * 64,
        base_before=base,
        base_after=base,
        halt_reason="max_steps",
        required_end_step=4,
    )
    assert canary_receipt["canary_lifecycle_complete"] is True
    assert canary_receipt["bootstrap_complete"] is False
    assert canary_receipt["claim_state"]["resident_sft_complete"] is False

    changed = dict(base)
    changed["fingerprint"] = "d" * 64
    with pytest.raises(trainer.ResidentSFTBootstrapTrainingError, match="base_checkpoint_changed"):
        trainer._publish_receipt(
            tmp_path,
            authority=authority,
            state=state,
            checkpoint_sha256="c" * 64,
            base_before=base,
            base_after=changed,
            halt_reason="max_steps",
            required_end_step=4,
        )


def test_wall_clock_boundary_publishes_the_actual_committed_step(tmp_path: Path) -> None:
    authority = {
        "authority_sha256": "a" * 64,
        "campaign_id": "resident-32b-recurrent-sft-bootstrap-cp-test",
        "campaign_scope": "full_bootstrap",
        "trainer": {"max_steps": 4},
    }
    state = {
        "invocation_count": 7,
        "checkpoint_sequence": 64,
        "step": 3,
        "terminal": True,
        "halt_reason": "wall_clock",
    }
    base = {"fingerprint": "b" * 64, "method": "sha256", "files": 1}

    receipt = trainer._publish_wall_clock_boundary_receipt(
        tmp_path,
        authority=authority,
        state=state,
        checkpoint_sha256="c" * 64,
        base_before=base,
        base_after=base,
        required_end_step=4,
    )

    assert receipt is not None
    assert receipt["step"] == 3
    assert receipt["required_end_step"] == 3
    assert receipt["halt_reason"] == "wall_clock"
    assert receipt["claim_state"]["resident_sft_complete"] is False
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["step"] == 3
    assert status["terminal"] is True
    assert status["halt_reason"] == "wall_clock"


@pytest.mark.parametrize("objective_version", ("v1", "v2", "v3"))
def test_tiny_real_mlx_training_exactly_resumes_cached_update(
    monkeypatch,
    tmp_path: Path,
    objective_version: str,
) -> None:
    monkeypatch.setattr(trainer, "REPO_ROOT", tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "placeholder.bin").write_bytes(b"immutable-base")

    spec = RLCExecutionSpec(
        n_slots=2,
        branch_roles=(
            ("constructive_solution", "critical_audit")
            if objective_version == "v3"
            else ("constructive_solution",)
        ),
        recurrent_steps=1,
        exchange_interval=1,
        prelude_frac=0.25,
        coda_frac=0.25,
    )
    spec_payload = trainer._canonical_json_bytes(spec.to_dict())
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(spec_payload)
    trust = {"policy": "test-only-source-bound"}
    trust_payload = trainer._canonical_json_bytes(trust)
    trust_path = tmp_path / "trust.json"
    trust_path.write_bytes(trust_payload)

    family = sorted(RECURRENCE_TRAINING_FAMILIES)[0]
    train_rows = [
        {
            "task_id": "train.1",
            "family": family,
            "depth": 1,
            "prompt": "Choose the valid transition.",
            "answer": 'FINAL_ANSWER: {"value":1}',
        }
    ]
    validation_rows = [
        {
            "task_id": "validation.1",
            "family": family,
            "depth": 1,
            "prompt": "Choose a different valid transition.",
            "answer": 'FINAL_ANSWER: {"value":2}',
        }
    ]
    train_payload, validation_payload = canonical_dataset_payloads(
        train_rows,
        validation_rows,
    )
    train_path = tmp_path / "train.json"
    validation_path = tmp_path / "validation.json"
    train_path.write_bytes(train_payload)
    validation_path.write_bytes(validation_payload)

    source_payloads = {role: f"source:{role}\n".encode("ascii") for role in REQUIRED_SOURCE_ROLES}
    source_bindings = {}
    for role, payload in source_payloads.items():
        path = tmp_path / f"source-{role}.txt"
        path.write_bytes(payload)
        source_bindings[role] = {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        }

    model_identity = {"fingerprint": "a" * 64, "method": "sha256", "files": 1}
    behavior_identity = {"bundle_sha256": "b" * 64, "file_count": 0, "files": []}
    personality_identity = {
        "present": False,
        "bundle_sha256": "",
        "file_count": 0,
        "files": [],
        "identity_sha256": "c" * 64,
    }
    tokenizer_identity = {
        "identity_sha256": "d" * 64,
        "artifact_sha256": "e" * 64,
        "runtime_sha256": "f" * 64,
    }
    runtime_identity = {"identity_sha256": "1" * 64, "runtime": "test"}
    config = ResidentSFTBootstrapConfig(
        seed=19,
        **(
            {
                "schema": TRAINER_CONFIG_SCHEMA_V2,
                "objective": OBJECTIVE_NAME_V2,
                "generated_rollin": GeneratedRollinSelectionConfig(
                    student_forcing_probability=1.0,
                    sampling_temperature=0.0,
                    branch_softmin_temperature=0.5,
                ),
            }
            if objective_version == "v2"
            else {
                "schema": TRAINER_CONFIG_SCHEMA_V3,
                "objective": OBJECTIVE_NAME_V3,
                "generated_rollin": GeneratedRollinSelectionConfig(
                    student_forcing_probability=1.0,
                    sampling_temperature=0.0,
                    branch_softmin_temperature=0.5,
                ),
                "branch_specialization": BranchSpecializationConfig(
                    weight=8.0,
                    target_separation=0.15,
                ),
                "structural_warmup_steps": 1,
                "structural_warmup_learning_rate": 1e-2,
                "role_conditioned_branches": 2,
            }
            if objective_version == "v3"
            else {}
        ),
        lora_initialization_seed=23,
        max_steps=2,
        max_invocation_steps=1,
        max_minutes=10.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        lora_rank=2,
        lora_scale=1.0,
        lora_targets=("o_proj",),
        lora_layers=1,
        evaluate_every=1,
        validation_examples=1,
        max_seq_length=128,
        memory_fraction=0.2,
        branch_indices=((0, 1) if objective_version == "v3" else (0,)),
    )
    now = datetime.now(UTC)
    run_root = tmp_path / "artifacts" / "run"
    run_root.mkdir(parents=True)
    run_stat = run_root.stat()
    authority = build_authority(
        campaign_id="resident-32b-recurrent-sft-bootstrap-cp-test",
        campaign_scope="full_bootstrap",
        committed_at=(now - timedelta(minutes=1)).isoformat(),
        expires_at=(now + timedelta(hours=1)).isoformat(),
        model_path="model",
        model_identity=model_identity,
        behavior_identity=behavior_identity,
        personality_identity=personality_identity,
        tokenizer_identity=tokenizer_identity,
        execution_spec={
            "path": "spec.json",
            "sha256": sha256_bytes(spec_payload),
            "size_bytes": len(spec_payload),
            "semantic_sha256": spec.sha256,
        },
        dataset=build_dataset_commitment(train_rows, validation_rows),
        dataset_artifacts={
            "train": {
                "path": "train.json",
                "sha256": sha256_bytes(train_payload),
                "size_bytes": len(train_payload),
            },
            "validation": {
                "path": "validation.json",
                "sha256": sha256_bytes(validation_payload),
                "size_bytes": len(validation_payload),
            },
        },
        sources=source_bindings,
        runtime_identity=runtime_identity,
        trust_policy={
            "path": "trust.json",
            "sha256": sha256_bytes(trust_payload),
            "size_bytes": len(trust_payload),
            "semantic_sha256": sha256_json(trust),
        },
        artifact_root="artifacts/run",
        artifact_root_identity={"st_dev": run_stat.st_dev, "st_ino": run_stat.st_ino},
        config=config,
    )
    authority_path = tmp_path / "authority.json"
    authority_path.write_bytes(trainer._canonical_json_bytes(authority))

    class Tokenizer:
        eos_token_id = 3

        def apply_chat_template(self, _messages, **_kwargs):
            return "<user>test</user><assistant>"

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            if text == 'FINAL_ANSWER: {"value":1}':
                return [7, 11]
            if text == 'FINAL_ANSWER: {"value":2}':
                return [7, 12]
            return [5, 9]

        def decode(self, tokens, *, skip_special_tokens):
            assert skip_special_tokens is False
            if tokens == [7, 11]:
                return 'FINAL_ANSWER: {"value":1}'
            if tokens == [7, 12]:
                return 'FINAL_ANSWER: {"value":2}'
            return ""

    def tiny_model() -> Model:
        mx.random.seed(29)
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
        return model

    monkeypatch.setattr("mlx_lm.load", lambda _path: (tiny_model(), Tokenizer()))
    monkeypatch.setattr(trainer, "full_weight_checkpoint_identity", lambda _path: model_identity)
    monkeypatch.setattr(trainer, "model_behavior_bundle_identity", lambda _path: behavior_identity)
    monkeypatch.setattr(trainer, "absent_personality_identity", lambda: personality_identity)
    monkeypatch.setattr(
        trainer,
        "resident_bootstrap_tokenizer_identity",
        lambda _path, _tokenizer: tokenizer_identity,
    )
    monkeypatch.setattr(trainer, "resident_bootstrap_runtime_identity", lambda: runtime_identity)
    monkeypatch.setattr(trainer, "standalone_model_lane", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(trainer, "mlx_memory_envelope", lambda **_kwargs: nullcontext())
    trainer.INTERRUPTED = False

    first_result = trainer._run(
        SimpleNamespace(
            authority=authority_path,
            expected_authority_sha256=authority["authority_sha256"],
            resume_policy="auto",
            invocation_step_budget=1,
            required_end_step=1,
        )
    )

    assert first_result == 0
    run = tmp_path / "artifacts" / "run"
    first_status = json.loads((run / "status.json").read_text(encoding="ascii"))
    assert first_status["step"] == 1
    assert first_status["terminal"] is False
    assert first_status["halt_reason"] == "invocation_step_limit"
    first_receipt = json.loads((run / "invocation-0001.json").read_text(encoding="ascii"))
    assert first_receipt["bootstrap_complete"] is False

    # Replaying the exact detached command after a supervisor crash must certify
    # the already-durable target, never spend the relative budget a second time.
    assert (
        trainer._run(
            SimpleNamespace(
                authority=authority_path,
                expected_authority_sha256=authority["authority_sha256"],
                resume_policy="required",
                invocation_step_budget=1,
                required_end_step=1,
            )
        )
        == 0
    )
    replay_status = json.loads((run / "status.json").read_text(encoding="ascii"))
    assert replay_status["step"] == 1
    assert len(list((run / "checkpoints").iterdir())) == 2

    second_result = trainer._run(
        SimpleNamespace(
            authority=authority_path,
            expected_authority_sha256=authority["authority_sha256"],
            resume_policy="required",
            invocation_step_budget=1,
            required_end_step=2,
        )
    )

    assert second_result == 0
    status = json.loads((run / "status.json").read_text(encoding="ascii"))
    assert status["step"] == 2
    assert status["terminal"] is True
    assert status["halt_reason"] == "max_steps"
    checkpoints = list((run / "checkpoints").iterdir())
    assert len(checkpoints) == 3
    receipt = json.loads((run / "invocation-0002.json").read_text(encoding="ascii"))
    assert receipt["bootstrap_complete"] is True
    assert receipt["claim_state"]["reasoning_gain_proven"] is False

    uninterrupted_config = replace(config, max_invocation_steps=2)
    uninterrupted_root = tmp_path / "artifacts" / "uninterrupted"
    uninterrupted_root.mkdir(parents=True)
    uninterrupted_stat = uninterrupted_root.stat()
    uninterrupted_authority = build_authority(
        campaign_id="resident-32b-recurrent-sft-bootstrap-cp-test-uninterrupted",
        campaign_scope="full_bootstrap",
        committed_at=(now - timedelta(minutes=1)).isoformat(),
        expires_at=(now + timedelta(hours=1)).isoformat(),
        model_path="model",
        model_identity=model_identity,
        behavior_identity=behavior_identity,
        personality_identity=personality_identity,
        tokenizer_identity=tokenizer_identity,
        execution_spec={
            "path": "spec.json",
            "sha256": sha256_bytes(spec_payload),
            "size_bytes": len(spec_payload),
            "semantic_sha256": spec.sha256,
        },
        dataset=build_dataset_commitment(train_rows, validation_rows),
        dataset_artifacts={
            "train": {
                "path": "train.json",
                "sha256": sha256_bytes(train_payload),
                "size_bytes": len(train_payload),
            },
            "validation": {
                "path": "validation.json",
                "sha256": sha256_bytes(validation_payload),
                "size_bytes": len(validation_payload),
            },
        },
        sources=source_bindings,
        runtime_identity=runtime_identity,
        trust_policy={
            "path": "trust.json",
            "sha256": sha256_bytes(trust_payload),
            "size_bytes": len(trust_payload),
            "semantic_sha256": sha256_json(trust),
        },
        artifact_root="artifacts/uninterrupted",
        artifact_root_identity={
            "st_dev": uninterrupted_stat.st_dev,
            "st_ino": uninterrupted_stat.st_ino,
        },
        config=uninterrupted_config,
    )
    uninterrupted_authority_path = tmp_path / "authority-uninterrupted.json"
    uninterrupted_authority_path.write_bytes(trainer._canonical_json_bytes(uninterrupted_authority))
    assert (
        trainer._run(
            SimpleNamespace(
                authority=uninterrupted_authority_path,
                expected_authority_sha256=uninterrupted_authority["authority_sha256"],
                resume_policy="auto",
                invocation_step_budget=2,
                required_end_step=2,
            )
        )
        == 0
    )

    resumed = load_checkpoint(run, expected_bindings=authority_state_bindings(authority))
    uninterrupted = load_checkpoint(
        tmp_path / "artifacts" / "uninterrupted",
        expected_bindings=authority_state_bindings(uninterrupted_authority),
    )
    assert adapter_tensor_fingerprint(resumed.adapter_tensors) == adapter_tensor_fingerprint(
        uninterrupted.adapter_tensors
    )
    assert resumed.optimizer_tensors.keys() == uninterrupted.optimizer_tensors.keys()
    assert all(
        bool(mx.array_equal(resumed.optimizer_tensors[key], uninterrupted.optimizer_tensors[key]))
        for key in resumed.optimizer_tensors
    )
    for field in (
        "step",
        "epoch",
        "cursor",
        "order",
        "order_sha256",
        "sample_history_sha256",
        "loss_trail",
        "validation_trail",
        "baseline_validation",
    ):
        assert resumed.state[field] == uninterrupted.state[field]
    if objective_version == "v2":
        for entry in resumed.state["loss_trail"]:
            validated = trainer.validate_generated_rollin_receipt(
                entry["objective_receipt"]
            )
            assert (
                validated["receipt_sha256"]
                == entry["objective_receipt_sha256"]
            )
            assert entry["branch_weights"] == pytest.approx(
                [branch["selection_weight"] for branch in validated["branches"]],
                abs=1e-12,
            )
    elif objective_version == "v3":
        assert [entry["phase"] for entry in resumed.state["loss_trail"]] == [
            trainer.STRUCTURAL_WARMUP_PHASE,
            trainer.JOINT_PHASE,
        ]
        warmup, joint = resumed.state["loss_trail"]
        assert warmup["warmup_target_reached"] is True
        trainer.validate_branch_specialization_receipt(
            warmup["objective_receipt"]
        )
        trainer.validate_generated_rollin_specialization_receipt(
            joint["objective_receipt"]
        )


def test_execution_spec_reader_accepts_pretty_strict_json() -> None:
    spec = RLCExecutionSpec(recurrent_steps=2)
    payload = json.dumps(spec.to_dict(), indent=2, sort_keys=True).encode("utf-8") + b"\n"

    observed = trainer._read_json_bytes(
        payload,
        role="execution_spec",
        canonical_required=False,
    )

    assert observed == spec.to_dict()


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (b'{"schema":"one","schema":"two"}', "execution_spec_duplicate_key"),
        (b'{"value":NaN}', "execution_spec_non_finite"),
    ],
)
def test_execution_spec_reader_rejects_ambiguous_json(payload: bytes, code: str) -> None:
    with pytest.raises(trainer.ResidentSFTBootstrapTrainingError, match=code):
        trainer._read_json_bytes(
            payload,
            role="execution_spec",
            canonical_required=False,
        )
