from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from core.learning import structured_sft_research_authority as authority
from core.learning import structured_sft_research_state as state
from core.learning.recurrent_sft_sampling import (
    FAMILY_BALANCED_SAMPLER,
    family_balanced_epoch_order,
)
from tools import train_structured_sft_research as trainer
from tools import verify_structured_sft_research_resume as resume_verifier

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _runner_sha256(value: Any) -> str:
    """Hash exactly as ``tools/run_detached_step.py`` does, independently.

    Reproduced here rather than imported so a drift in either canonicaliser
    fails this test instead of cancelling out against itself.
    """
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _identified(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": authority.sha256_json(body)}


def _source_closure() -> dict[str, Any]:
    body = {"schema": authority.SOURCE_CLOSURE_SCHEMA, "files": []}
    return {**body, "closure_sha256": authority.sha256_json(body)}


def _authority(
    *,
    max_seq_length: int = 512,
    sampler: str = authority.SAMPLER,
) -> dict[str, Any]:
    upstream = _identified(
        {
            "status": "externally_witnessed_audit_head_verified_offline",
            "sequence": 1,
        }
    )
    candidate = _identified(
        {
            "classification": "repository_generated_structured_synthetic",
            "candidate_package_sha256": SHA_A,
            "custody_commit_sha256": SHA_B,
            "contains_user_content": False,
            "contains_verified_replay": False,
            "holdout_present": False,
        }
    )
    tokenization = _identified(
        {
            "candidate_package_sha256": SHA_A,
            "custody_commit_sha256": SHA_B,
            "max_seq_length": max_seq_length,
        }
    )
    model = _identified(
        {
            "resident_checkpoint_allowed": False,
            "estimated_dense_parameters": 1_500_000_000,
            "total_weight_bytes": 900_000_000,
        }
    )
    execution = _identified(
        {
            "adapter_scope": "latent_slot_positions_only",
            "semantic_sha256": SHA_C,
        }
    )
    config = authority.RecurrentSFTTrainerConfig(
        max_steps=10,
        sampler=sampler,
        checkpoint_every=5,
        evaluate_every=5,
        validation_examples=2,
        max_seq_length=max_seq_length,
        max_minutes=10.0,
    )
    return authority.build_authority(
        issued_at_unix=1_000,
        expires_at_unix=2_000,
        upstream_witness=upstream,
        candidate=candidate,
        tokenization=tokenization,
        model=model,
        execution_spec=execution,
        trainer_config=config,
        sources=_source_closure(),
    )


def _dataset() -> dict[str, Any]:
    body = {
        "schema": trainer.DATASET_SCHEMA,
        "candidate_identity_sha256": SHA_A,
        "train": [{"example_id": "train-1"}],
        "validation": [{"example_id": "valid-1"}],
        "holdout": None,
        "verified_replay": None,
    }
    return {**body, "dataset_sha256": authority.sha256_json(body)}


def _bindings(document: dict[str, Any], dataset: dict[str, Any]) -> dict[str, str]:
    return {
        "authority_sha256": document["authority_sha256"],
        "dataset_sha256": dataset["dataset_sha256"],
        "tokenization_identity_sha256": document["tokenization"]["identity_sha256"],
        "model_identity_sha256": document["model"]["identity_sha256"],
        "source_closure_sha256": document["sources"]["closure_sha256"],
        "execution_spec_sha256": document["execution_spec"]["semantic_sha256"],
        "trainer_config_sha256": authority.sha256_json(document["trainer"]),
    }


def _checkpoint_state(
    bindings: dict[str, str],
    *,
    terminal: bool = False,
    elapsed_s: float = 12.0,
) -> dict[str, Any]:
    order = authority.deterministic_order(1, seed=7, epoch=0)
    return {
        **bindings,
        "step": 5,
        "optimizer_updates": 5,
        "epoch": 0,
        "cursor": 1,
        "order": order,
        "sampler": authority.SAMPLER,
        "seed": 7,
        "train_example_count": 1,
        "validation_example_count": 1,
        "elapsed_training_s": elapsed_s,
        "invocation_count": 1,
        "loss_trail": [],
        "validation_trail": [],
        "pending_losses": [],
        "baseline_validation": {"mean_loss": 1.0},
        "last_step_committed": True,
        "terminal": terminal,
    }


def _write_checkpoint(
    run_dir: Path,
    *,
    document: dict[str, Any],
    dataset: dict[str, Any],
    terminal: bool = False,
    elapsed_s: float = 12.0,
) -> Path:
    run_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    generation = run_dir / "checkpoints" / "step-00000005-test"
    generation.mkdir()
    adapter = b"adapter-tensors"
    optimizer = b"optimizer-tensors"
    (generation / "quarantine_adapter.safetensors").write_bytes(adapter)
    (generation / "optimizer.safetensors").write_bytes(optimizer)
    complete = {
        **_checkpoint_state(
            _bindings(document, dataset),
            terminal=terminal,
            elapsed_s=elapsed_s,
        ),
        "schema": state.CHECKPOINT_SCHEMA,
        "checkpoint_id": generation.name,
        "created_unix": 1_100.0,
        "adapter": {
            "path": "quarantine_adapter.safetensors",
            "sha256": state.sha256_bytes(adapter),
            "size_bytes": len(adapter),
        },
        "optimizer": {
            "path": "optimizer.safetensors",
            "sha256": state.sha256_bytes(optimizer),
            "size_bytes": len(optimizer),
        },
    }
    complete_payload = state.canonical_json_bytes(complete)
    (generation / "complete.json").write_bytes(complete_payload)
    pointer = {
        "schema": state.POINTER_SCHEMA,
        "checkpoint": f"checkpoints/{generation.name}",
        "complete_sha256": state.sha256_bytes(complete_payload),
    }
    (run_dir / "latest.json").write_bytes(state.canonical_json_bytes(pointer))
    (run_dir / "projected_dataset_manifest.json").write_bytes(
        state.canonical_json_bytes(dataset)
    )
    state.append_journal_event(
        run_dir,
        # A terminal checkpoint is published before the completion receipt and
        # TERMINAL journal event. This fixture models that crash boundary.
        event_type="CHECKPOINT",
        payload={"step": 5, "checkpoint": generation.name},
    )
    return generation


def test_authority_is_exact_and_time_bounded() -> None:
    document = _authority()
    assert (
        authority.validate_authority(
            document,
            expected_authority_sha256=document["authority_sha256"],
            now_unix=1_500,
        )
        == document
    )
    with pytest.raises(
        authority.StructuredSFTResearchAuthorityError,
        match="authority_time_invalid",
    ):
        authority.validate_authority(document, now_unix=2_001)


def test_authority_rejects_tokenization_trainer_length_drift() -> None:
    document = copy.deepcopy(_authority())
    tokenization = dict(document["tokenization"])
    tokenization.pop("identity_sha256")
    tokenization["max_seq_length"] = 1_024
    document["tokenization"] = _identified(tokenization)
    body = dict(document)
    body.pop("authority_sha256")
    document["authority_sha256"] = authority.sha256_json(body)
    with pytest.raises(
        authority.StructuredSFTResearchAuthorityError,
        match="authority_scope_invalid",
    ):
        authority.validate_authority(document)


def test_deterministic_order_has_no_hidden_rng_state() -> None:
    first = authority.deterministic_order(31, seed=42, epoch=3)
    assert first == authority.deterministic_order(31, seed=42, epoch=3)
    assert first != authority.deterministic_order(31, seed=42, epoch=4)
    assert sorted(first) == list(range(31))


def test_balanced_sampler_authority_is_explicit_and_legacy_evidence_is_readable() -> None:
    legacy = _authority()
    balanced = _authority(
        sampler="sha256_stateless_family_balanced_epoch.v1"
    )
    assert authority.validate_authority(legacy)["trainer"]["sampler"] == (
        authority.SAMPLER
    )
    validated = authority.validate_authority(balanced)
    assert validated["trainer"]["sampler"] == (
        "sha256_stateless_family_balanced_epoch.v1"
    )
    assert validated["trainer"]["validation_scope"] == (
        "candidate_and_source_bound_retention_validation_no_evaluator_holdout"
    )
    assert validated["resumability"]["sample_order"] == (
        "sha256_stateless_family_balanced_epoch.v1"
    )


def test_checkpoint_state_accepts_bounded_repeated_balanced_order() -> None:
    document = _authority(sampler=FAMILY_BALANCED_SAMPLER)
    dataset = _dataset()
    rows = [
        {
            "example_id": f"{index + 1:064x}",
            "family": "small" if index < 2 else "large",
        }
        for index in range(6)
    ]
    order = family_balanced_epoch_order(rows, seed=7, epoch=0)
    checkpoint = _checkpoint_state(_bindings(document, dataset))
    checkpoint.update(
        {
            "step": len(order),
            "optimizer_updates": len(order),
            "cursor": len(order),
            "order": order,
            "sampler": FAMILY_BALANCED_SAMPLER,
            "train_example_count": len(rows),
            "initial_adapter_sha256": SHA_C,
        }
    )
    assert state.validate_checkpoint_state(checkpoint) == checkpoint

    checkpoint["order"][0] = len(rows)
    with pytest.raises(
        state.StructuredSFTResearchStateError,
        match="state_order_invalid",
    ):
        state.validate_checkpoint_state(checkpoint)

    valid_order = family_balanced_epoch_order(rows, seed=7, epoch=0)
    history_tamper = _checkpoint_state(_bindings(document, dataset))
    history_tamper.update(
        {
            "step": len(valid_order) + 1,
            "optimizer_updates": len(valid_order) + 1,
            "cursor": len(valid_order),
            "order": valid_order,
            "sampler": FAMILY_BALANCED_SAMPLER,
            "train_example_count": len(rows),
            "initial_adapter_sha256": SHA_C,
        }
    )
    with pytest.raises(
        state.StructuredSFTResearchStateError,
        match="sample_history_invalid",
    ):
        state.validate_checkpoint_state(history_tamper)


def test_balanced_dataset_identity_commits_retention_and_epoch_order() -> None:
    rows = [
        {
            "example_id": f"{index + 1:064x}",
            "family": "logic" if index < 2 else "tool",
        }
        for index in range(6)
    ]
    dataset = trainer._dataset_identity(
        candidate_sha256=SHA_A,
        train_rows=rows,
        validation_rows=[{"family": "validation"}],
        sampler=FAMILY_BALANCED_SAMPLER,
        seed=7,
    )
    assert dataset["schema"] == trainer.DATASET_SCHEMA_V2
    assert dataset["retention"]["split_case_overlap_count"] == 0
    assert dataset["sampler"]["name"] == FAMILY_BALANCED_SAMPLER
    assert dataset["sampler"]["epoch_zero_order"] == (
        family_balanced_epoch_order(rows, seed=7, epoch=0)
    )
    assert dataset["sampler"]["epoch_zero_balance"][
        "exact_family_balance"
    ] is True
    assert dataset["dataset_sha256"] == authority.sha256_json(
        {
            key: value
            for key, value in dataset.items()
            if key != "dataset_sha256"
        }
    )


def _prevalidated_candidate_material() -> tuple[
    dict[str, bytes],
    dict[str, Any],
    dict[str, Any],
]:
    train = b'{"messages":[{"role":"assistant","content":"train"}]}\n'
    validation = b'{"messages":[{"role":"assistant","content":"valid"}]}\n'
    source_sha256 = "d" * 64
    curriculum_sha256 = "e" * 64
    custody_root_sha256 = "f" * 64
    artifact_bindings = {
        "candidate_train.jsonl": {
            "sha256": hashlib.sha256(train).hexdigest(),
            "size_bytes": len(train),
        },
        "candidate_valid.jsonl": {
            "sha256": hashlib.sha256(validation).hexdigest(),
            "size_bytes": len(validation),
        },
    }
    manifest_body = {
        "schema": "test",
        "curriculum_manifest": {
            "curriculum_sha256": curriculum_sha256,
            "source_binding": {"sha256": source_sha256},
        },
        "artifacts": artifact_bindings,
        "candidate_filenames": {
            "train": "candidate_train.jsonl",
            "validation": "candidate_valid.jsonl",
        },
        "custody_root_sha256": custody_root_sha256,
        "validation_scope": "train_validation_replay_only",
        "trainer_ready": False,
    }
    manifest = {
        **manifest_body,
        "package_sha256": authority.sha256_json(manifest_body),
    }
    artifacts = {
        "candidate_train.jsonl": train,
        "candidate_valid.jsonl": validation,
        "manifest.json": authority.canonical_json_bytes(manifest),
    }
    files = [
        {
            "name": name,
            "sha256": hashlib.sha256(artifacts[name]).hexdigest(),
            "size_bytes": len(artifacts[name]),
        }
        for name in (
            "candidate_train.jsonl",
            "candidate_valid.jsonl",
            "manifest.json",
        )
    ]
    custody_body = {
        "schema": "aura.rlc.structured_sft_custody_commit.v1",
        "state": "committed",
        "generation_id": "1" * 32,
        "candidate_directory": "candidate",
        "evaluator_directory": "evaluator",
        "candidate_package_sha256": manifest["package_sha256"],
        "evaluator_package_sha256": SHA_B,
        "custody_root_sha256": custody_root_sha256,
        "custody_report_sha256": SHA_C,
    }
    custody = {
        **custody_body,
        "commit_sha256": authority.sha256_json(custody_body),
    }
    candidate = {
        "files": files,
        "candidate_package_sha256": manifest["package_sha256"],
        "custody_commit_sha256": custody["commit_sha256"],
        "evaluator_package_sha256": SHA_B,
        "custody_root_sha256": custody_root_sha256,
        "curriculum_sha256": curriculum_sha256,
        "source_closure_sha256": source_sha256,
    }
    return artifacts, custody, candidate


def test_prevalidated_candidate_authorization_accepts_only_frozen_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts, custody, candidate = _prevalidated_candidate_material()
    document = {"candidate": candidate}
    monkeypatch.setattr(
        authority,
        "validate_authority",
        lambda *_args, **_kwargs: document,
    )
    accepted = authority.authorize_prevalidated_candidate_bytes(
        document,
        candidate_artifacts=artifacts,
        custody_attestation=custody,
        candidate_directory_name="candidate",
        now_unix=1,
        expected_authority_sha256=SHA_A,
    )
    assert accepted == artifacts

    mutated = dict(artifacts)
    mutated["candidate_train.jsonl"] += b" "
    with pytest.raises(
        authority.StructuredSFTResearchAuthorityError,
        match="prevalidated_candidate_file_drift",
    ):
        authority.authorize_prevalidated_candidate_bytes(
            document,
            candidate_artifacts=mutated,
            custody_attestation=custody,
            candidate_directory_name="candidate",
            now_unix=1,
            expected_authority_sha256=SHA_A,
        )


def test_prevalidated_candidate_authorization_rejects_custody_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts, custody, candidate = _prevalidated_candidate_material()
    document = {"candidate": candidate}
    monkeypatch.setattr(
        authority,
        "validate_authority",
        lambda *_args, **_kwargs: document,
    )
    mutated = dict(custody)
    mutated["custody_report_sha256"] = "9" * 64
    with pytest.raises(
        authority.StructuredSFTResearchAuthorityError,
        match="prevalidated_custody_drift",
    ):
        authority.authorize_prevalidated_candidate_bytes(
            document,
            candidate_artifacts=artifacts,
            custody_attestation=mutated,
            candidate_directory_name="candidate",
            now_unix=1,
            expected_authority_sha256=SHA_A,
        )


def test_upstream_witness_binds_global_and_active_shard_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = {
        "campaign": "SPARK-059",
        "trainer_ready": False,
        "training_authority": "none",
        "research_scope": {
            "small_checkpoint_falsification_may_use": ["structured_synthetic"],
            "forbidden_from_research_trainer": [
                "verified_replay_user_content",
                "evaluation_holdout",
            ],
            "production_promotion_allowed": False,
        },
    }
    monkeypatch.setattr(
        authority,
        "validate_spark_059_production_audit_packet",
        lambda raw: raw,
    )
    monkeypatch.setattr(
        authority,
        "validate_rekor_witness_bundle",
        lambda *_args, **_kwargs: {
            "audit_packet_sha256": SHA_A,
            "bundle_sha256": SHA_B,
            "statement_sha256": SHA_C,
            "sequence": 1,
            "rekor_uuid": "uuid",
            "rekor_log_index": 2_257_039_380,
            "rekor_integrated_time": 1_785_137_442,
            "trusted_log_key_sha256": SHA_A,
            "status": "externally_witnessed_audit_head_verified_offline",
        },
    )
    witness = {
        "rekor_entry": {
            "verification": {
                "inclusionProof": {"logIndex": 2_135_135_118},
            }
        }
    }
    observed = authority.upstream_witness_identity(
        audit_packet=packet,
        witness_bundle=witness,
        trusted_log_public_key_pem=b"key",
        expected_sequence=1,
        expected_previous_statement_sha256="0" * 64,
        expected_previous_rekor_uuid=None,
        minimum_active_shard_log_index=2_135_135_118,
    )
    assert observed["global_log_index"] == 2_257_039_380
    assert observed["active_shard_log_index"] == 2_135_135_118
    with pytest.raises(
        authority.StructuredSFTResearchAuthorityError,
        match="active_shard_index_rollback",
    ):
        authority.upstream_witness_identity(
            audit_packet=packet,
            witness_bundle=witness,
            trusted_log_public_key_pem=b"key",
            expected_sequence=1,
            expected_previous_statement_sha256="0" * 64,
            expected_previous_rekor_uuid=None,
            minimum_active_shard_log_index=2_135_135_119,
        )


def test_small_model_identity_rejects_resident_scale_config(tmp_path: Path) -> None:
    config = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 5_120,
        "num_hidden_layers": 64,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "intermediate_size": 27_648,
        "vocab_size": 152_064,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="ascii")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="ascii")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="ascii")
    (tmp_path / "model.safetensors").write_bytes(b"not-loaded")
    with pytest.raises(
        authority.StructuredSFTResearchAuthorityError,
        match="model_not_small_checkpoint",
    ):
        authority.small_model_identity(tmp_path)


def test_small_model_identity_accepts_public_read_only_model_directory(
    tmp_path: Path,
) -> None:
    config = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "vocab_size": 256,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="ascii")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="ascii")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="ascii")
    (tmp_path / "model.safetensors").write_bytes(b"identity-only")
    tmp_path.chmod(0o755)
    identity = authority.small_model_identity(tmp_path)
    assert identity["resident_checkpoint_allowed"] is False
    assert identity["estimated_dense_parameters"] < 2_500_000_000


def test_inspect_checkpoint_verifies_bindings_without_loading_mlx(
    tmp_path: Path,
) -> None:
    document = _authority()
    dataset = _dataset()
    run_dir = tmp_path / "run"
    generation = _write_checkpoint(
        run_dir,
        document=document,
        dataset=dataset,
    )
    inspected = state.inspect_checkpoint(
        run_dir,
        expected_bindings=_bindings(document, dataset),
    )
    assert inspected.checkpoint_dir == generation
    assert inspected.state["step"] == 5


def test_inspect_checkpoint_rejects_tensor_tamper(tmp_path: Path) -> None:
    document = _authority()
    dataset = _dataset()
    run_dir = tmp_path / "run"
    generation = _write_checkpoint(
        run_dir,
        document=document,
        dataset=dataset,
    )
    (generation / "quarantine_adapter.safetensors").write_bytes(b"tampered")
    with pytest.raises(
        state.StructuredSFTResearchStateError,
        match="adapter_commitment_mismatch",
    ):
        state.inspect_checkpoint(
            run_dir,
            expected_bindings=_bindings(document, dataset),
        )


def test_checkpoint_round_trip_restores_adapter_and_optimizer_tree(
    tmp_path: Path,
) -> None:
    mx = pytest.importorskip("mlx.core")
    document = _authority()
    dataset = _dataset()
    run_dir = tmp_path / "run"
    checkpoint = state.save_checkpoint(
        run_dir,
        adapter_tensors={"layers.0.lora_a": mx.array([[1.0, 2.0]])},
        optimizer_tensors={
            "state.layers.0.lora_a.m": mx.array([[0.1, 0.2]]),
        },
        state=_checkpoint_state(_bindings(document, dataset)),
    )
    loaded = state.load_checkpoint(
        run_dir,
        expected_bindings=_bindings(document, dataset),
    )
    assert loaded.checkpoint_dir == checkpoint
    assert set(loaded.adapter_tensors) == {"layers.0.lora_a"}
    assert loaded.optimizer_state["state"]["layers"][0]["lora_a"]["m"].shape == (
        1,
        2,
    )


def test_journal_rejects_mutated_event(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    event = state.append_journal_event(
        run_dir,
        event_type="ADMITTED",
        payload={"authority_sha256": SHA_A},
    )
    path = next((run_dir / "journal").glob("*.json"))
    raw = json.loads(path.read_text(encoding="ascii"))
    raw["payload"]["authority_sha256"] = SHA_B
    path.write_text(json.dumps(raw), encoding="ascii")
    with pytest.raises(
        state.StructuredSFTResearchStateError,
        match="journal_event_invalid",
    ):
        state.validate_journal(run_dir)
    assert event["sequence"] == 1


def test_create_or_verify_refuses_manifest_replacement(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    trainer._write_create_or_verify(path, b"first")
    trainer._write_create_or_verify(path, b"first")
    with pytest.raises(
        trainer.StructuredSFTResearchTrainingError,
        match="existing_artifact_commitment_mismatch",
    ):
        trainer._write_create_or_verify(path, b"second")


def test_trainer_cli_has_auto_resume_but_no_evaluator_surface() -> None:
    parsed = trainer._parser().parse_args(
        [
            "--authority",
            "authority.json",
            "--expected-authority-sha256",
            SHA_A,
            "--audit-packet",
            "audit.json",
            "--witness-bundle",
            "bundle.json",
            "--trusted-log-key",
            "rekor.pem",
            "--witness-sequence",
            "1",
            "--candidate-dir",
            "candidate",
            "--tokenizer-dir",
            "tokenizer",
            "--snapshot-root",
            "snapshot",
            "--model-dir",
            "model",
            "--execution-spec",
            "spec.json",
            "--out-dir",
            "out",
            "--resume-policy",
            "auto",
        ]
    )
    assert parsed.resume_policy == "auto"
    with pytest.raises(SystemExit):
        trainer._parser().parse_args(
            [
                "--evaluator-dir",
                "forbidden",
            ]
        )


def test_resume_verdict_is_attempt_bound_and_safe(tmp_path: Path) -> None:
    document = _authority()
    dataset = _dataset()
    authority_path = tmp_path / "authority.json"
    authority_path.write_bytes(authority.canonical_json_bytes(document))
    run_dir = tmp_path / "run"
    _write_checkpoint(run_dir, document=document, dataset=dataset)
    context = {
        "plan_sha256": SHA_A,
        "command_sha256": SHA_B,
        "prior_attempt": 2,
        "prior_journal_head_sha256": SHA_C,
    }
    verdict = resume_verifier.build_verdict(
        authority_path=authority_path,
        expected_authority_sha256=document["authority_sha256"],
        run_dir=run_dir,
        detached_context=context,
    )
    assert verdict["verdict"] == "safe_to_resume"
    assert verdict["checkpoint_sequence"] == 5
    # The runner carries evidence inline over stdout-v3; no evidence file exists.
    assert verdict["schema"] == resume_verifier.VERDICT_SCHEMA
    assert verdict["evidence"]["schema"] == resume_verifier.EVIDENCE_SCHEMA
    assert verdict["evidence_sha256"] == _runner_sha256(verdict["evidence"])
    assert verdict["checkpoint_identity"] == _runner_sha256(
        {
            "prior_attempt": 2,
            "prior_journal_head_sha256": SHA_C,
            "checkpoint_sequence": verdict["checkpoint_sequence"],
            "evidence_sha256": verdict["evidence_sha256"],
        }
    )
    assert not list(tmp_path.glob("*resume-evidence*"))


def test_resume_verdict_rejects_dataset_digest_tamper(tmp_path: Path) -> None:
    document = _authority()
    dataset = _dataset()
    authority_path = tmp_path / "authority.json"
    authority_path.write_bytes(authority.canonical_json_bytes(document))
    run_dir = tmp_path / "run"
    _write_checkpoint(run_dir, document=document, dataset=dataset)
    manifest_path = run_dir / "projected_dataset_manifest.json"
    tampered = json.loads(manifest_path.read_text(encoding="ascii"))
    tampered["train"][0]["example_id"] = "changed"
    manifest_path.write_text(json.dumps(tampered), encoding="ascii")
    with pytest.raises(
        resume_verifier.StructuredSFTResumeVerifierError,
        match="dataset_manifest_invalid",
    ):
        resume_verifier.build_verdict(
            authority_path=authority_path,
            expected_authority_sha256=document["authority_sha256"],
            run_dir=run_dir,
            detached_context={
                "plan_sha256": SHA_A,
                "command_sha256": SHA_B,
                "prior_attempt": 1,
                "prior_journal_head_sha256": SHA_C,
            },
        )


def test_resume_verdict_repairs_terminal_checkpoint_before_completion(
    tmp_path: Path,
) -> None:
    document = _authority()
    dataset = _dataset()
    authority_path = tmp_path / "authority.json"
    authority_path.write_bytes(authority.canonical_json_bytes(document))
    run_dir = tmp_path / "run"
    _write_checkpoint(
        run_dir,
        document=document,
        dataset=dataset,
        terminal=True,
    )
    verdict = resume_verifier.build_verdict(
        authority_path=authority_path,
        expected_authority_sha256=document["authority_sha256"],
        run_dir=run_dir,
        detached_context={
            "plan_sha256": SHA_A,
            "command_sha256": SHA_B,
            "prior_attempt": 1,
            "prior_journal_head_sha256": SHA_C,
        },
    )
    assert verdict["verdict"] == "safe_to_resume"


def test_resume_verdict_marks_bound_terminal_completion_complete(
    tmp_path: Path,
) -> None:
    document = _authority()
    dataset = _dataset()
    authority_path = tmp_path / "authority.json"
    authority_path.write_bytes(authority.canonical_json_bytes(document))
    run_dir = tmp_path / "run"
    _write_checkpoint(
        run_dir,
        document=document,
        dataset=dataset,
        terminal=True,
    )
    inspected = state.inspect_checkpoint(
        run_dir,
        expected_bindings=_bindings(document, dataset),
    )
    completion_body = {
        "schema": "aura.rlc.synthetic_recurrent_sft_completion.v1",
        "authority_sha256": document["authority_sha256"],
        "dataset_sha256": dataset["dataset_sha256"],
        "model_identity_sha256": document["model"]["identity_sha256"],
        "execution_spec_sha256": document["execution_spec"]["semantic_sha256"],
        "step": inspected.state["step"],
        "halt_reason": "max_steps",
        "terminal": True,
        "baseline_validation": {"mean_loss": 1.0},
        "final_validation": {"mean_loss": 0.9},
        "checkpoint": inspected.checkpoint_dir.name,
        "base_weights_unchanged": True,
        "output_disposition": "quarantined_research_only",
        "ordinary_lexical_adapter_activation": False,
        "production_effect": False,
        "promotion_allowed": False,
        "claims_not_supported": document["claims_not_supported"],
    }
    completion = {
        **completion_body,
        "completion_sha256": authority.sha256_json(completion_body),
    }
    (run_dir / "research_completion.json").write_bytes(
        state.canonical_json_bytes(completion)
    )
    verdict = resume_verifier.build_verdict(
        authority_path=authority_path,
        expected_authority_sha256=document["authority_sha256"],
        run_dir=run_dir,
        detached_context={
            "plan_sha256": SHA_A,
            "command_sha256": SHA_B,
            "prior_attempt": 2,
            "prior_journal_head_sha256": SHA_C,
        },
    )
    assert verdict["verdict"] == "already_completed"


def test_resume_verdict_accepts_checkpoint_after_unfinished_model_bind(
    tmp_path: Path,
) -> None:
    document = _authority()
    dataset = _dataset()
    authority_path = tmp_path / "authority.json"
    authority_path.write_bytes(authority.canonical_json_bytes(document))
    run_dir = tmp_path / "run"
    _write_checkpoint(run_dir, document=document, dataset=dataset)
    state.append_journal_event(
        run_dir,
        event_type="MODEL_BOUND",
        payload={"model_identity_sha256": document["model"]["identity_sha256"]},
    )
    verdict = resume_verifier.build_verdict(
        authority_path=authority_path,
        expected_authority_sha256=document["authority_sha256"],
        run_dir=run_dir,
        detached_context={
            "plan_sha256": SHA_A,
            "command_sha256": SHA_B,
            "prior_attempt": 1,
            "prior_journal_head_sha256": SHA_C,
        },
    )
    assert verdict["verdict"] == "safe_to_resume"


def test_resume_verdict_refuses_to_reset_total_wall_clock_budget(
    tmp_path: Path,
) -> None:
    document = _authority()
    dataset = _dataset()
    authority_path = tmp_path / "authority.json"
    authority_path.write_bytes(authority.canonical_json_bytes(document))
    run_dir = tmp_path / "run"
    _write_checkpoint(
        run_dir,
        document=document,
        dataset=dataset,
        elapsed_s=601.0,
    )
    verdict = resume_verifier.build_verdict(
        authority_path=authority_path,
        expected_authority_sha256=document["authority_sha256"],
        run_dir=run_dir,
        detached_context={
            "plan_sha256": SHA_A,
            "command_sha256": SHA_B,
            "prior_attempt": 1,
            "prior_journal_head_sha256": SHA_C,
        },
    )
    assert verdict["verdict"] == "indeterminate"
