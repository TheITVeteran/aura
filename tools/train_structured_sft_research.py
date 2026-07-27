#!/usr/bin/env python3
"""Run authority-bound, recurrence-native SFT on a small local checkpoint.

The command accepts no evaluator, replay, personality-adapter, registry,
fusion, or promotion path. Outputs remain quarantined research artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.execution_spec import (  # noqa: E402
    RLCExecutionSpec,
)
from core.learning.recurrent_sft_execution import (  # noqa: E402
    RecurrentSFTExecutionError,
    adapter_tensor_dict,
    adapter_tensor_fingerprint,
    assert_adapter_tensor_topology,
    project_chat_rows,
    wrap_recurrent_window,
)
from core.learning.recurrent_sft_retention import (  # noqa: E402
    build_retention_rows,
    retention_manifest,
)
from core.learning.recurrent_sft_sampling import (  # noqa: E402
    FAMILY_BALANCED_SAMPLER,
    family_balance_receipt,
    family_balanced_epoch_order,
)
from core.learning.structured_sft_research_authority import (  # noqa: E402
    RecurrentSFTTrainerConfig,
    StructuredSFTResearchAuthorityError,
    authorize_prevalidated_candidate_bytes,
    canonical_json_bytes,
    execution_spec_identity,
    sha256_json,
    small_model_identity,
    source_closure,
    strict_json_bytes,
    validate_authority,
    verify_authority_upstream,
)
from core.learning.structured_sft_research_state import (  # noqa: E402
    StructuredSFTResearchStateError,
    append_journal_event,
    load_checkpoint,
    save_checkpoint,
    validate_journal,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    ensure_private_directory,
)
from core.runtime.file_read_gateway import (  # noqa: E402
    read_stable_bytes,
    read_stable_directory_files,
)
from tools.validate_structured_sft_tokenization import (  # noqa: E402
    TokenizerValidationError,
    load_resident_tokenizer,
    resident_tokenizer_artifact_identity,
    resident_tokenizer_runtime_identity,
)

COMPLETION_SCHEMA = "aura.rlc.synthetic_recurrent_sft_completion.v1"
DATASET_SCHEMA = "aura.rlc.synthetic_recurrent_sft_projected_dataset.v1"
DATASET_SCHEMA_V2 = "aura.rlc.synthetic_recurrent_sft_projected_dataset.v2"
RESUME_POLICIES = frozenset({"never", "auto", "required"})
_MAX_DOCUMENT_BYTES = 256 * 1024 * 1024
_MAX_JSONL_ROWS = 100_000
_CANDIDATE_FILES = (
    "candidate_train.jsonl",
    "candidate_valid.jsonl",
    "manifest.json",
)
_CUSTODY_COMMIT_FILE = ".aura_structured_sft_custody.commit.json"
_SNAPSHOT_MANIFEST_FILE = "tokenizer_snapshot_manifest.bin"
_INTERRUPTED = False


class StructuredSFTResearchTrainingError(RuntimeError):
    """The restricted recurrent-SFT run could not proceed honestly."""


def _fail(code: str) -> Never:
    raise StructuredSFTResearchTrainingError(
        str(code or "structured_sft_research_training_failed")
    )


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        raw = read_stable_bytes(
            path.expanduser().resolve(strict=True),
            max_bytes=_MAX_DOCUMENT_BYTES,
        )
        return strict_json_bytes(raw, role=role)
    except StructuredSFTResearchAuthorityError:
        raise
    except OSError as exc:
        raise StructuredSFTResearchTrainingError(
            f"{role}_unreadable"
        ) from exc


def _read_pem(path: Path) -> bytes:
    try:
        return read_stable_bytes(
            path.expanduser().resolve(strict=True),
            max_bytes=64 * 1024,
        )
    except OSError as exc:
        raise StructuredSFTResearchTrainingError(
            "trusted_log_key_unreadable"
        ) from exc


def _source_paths() -> dict[str, Path]:
    return {
        "authority": (
            REPO_ROOT
            / "core/learning/structured_sft_research_authority.py"
        ),
        "trainer": Path(__file__),
        "containment_launcher": (
            REPO_ROOT / "tools/launch_structured_sft_research.py"
        ),
        "detached_supervisor": REPO_ROOT / "tools/run_detached_step.py",
        "checkpoint_state": (
            REPO_ROOT / "core/learning/structured_sft_research_state.py"
        ),
        "structured_sft": REPO_ROOT / "core/learning/structured_sft.py",
        "retention_curriculum": (
            REPO_ROOT / "core/learning/recurrent_sft_retention.py"
        ),
        "behavior_canaries": (
            REPO_ROOT / "core/learning/recurrent_sft_behavior_canaries.py"
        ),
        "sampling": REPO_ROOT / "core/learning/recurrent_sft_sampling.py",
        "tokenization": REPO_ROOT / "tools/validate_structured_sft_tokenization.py",
        "recurrence_objective": (
            REPO_ROOT / "core/learning/recurrence_native_objective_v2.py"
        ),
        "execution_spec": (
            REPO_ROOT / "core/brain/llm/latent_cortex/execution_spec.py"
        ),
        "recurrence_adapter": (
            REPO_ROOT / "core/brain/llm/latent_cortex/recurrence_adapter.py"
        ),
        "recurrent_sft_execution": (
            REPO_ROOT / "core/learning/recurrent_sft_execution.py"
        ),
        "resume_verifier": (
            REPO_ROOT / "tools/verify_structured_sft_research_resume.py"
        ),
    }


def _jsonl_rows(payload: bytes, *, role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            _fail(f"{role}_blank_line")
        try:
            row = strict_json_bytes(line, role=f"{role}_line_{line_number}")
        except StructuredSFTResearchAuthorityError as exc:
            raise StructuredSFTResearchTrainingError(exc.code) from exc
        rows.append(row)
        if len(rows) > _MAX_JSONL_ROWS:
            _fail(f"{role}_too_many_rows")
    if not rows:
        _fail(f"{role}_empty")
    return rows


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _read_prevalidated_candidate(
    candidate_directory: Path,
) -> tuple[dict[str, bytes], dict[str, Any], str]:
    candidate = _lexical_absolute(candidate_directory)
    commit_path = candidate.parent / _CUSTODY_COMMIT_FILE
    commit_before = read_stable_bytes(
        commit_path,
        max_bytes=_MAX_DOCUMENT_BYTES,
    )
    artifacts = read_stable_directory_files(
        candidate,
        names=_CANDIDATE_FILES,
        max_bytes_per_file=_MAX_DOCUMENT_BYTES,
    )
    commit_after = read_stable_bytes(
        commit_path,
        max_bytes=_MAX_DOCUMENT_BYTES,
    )
    if commit_after != commit_before:
        _fail("trainer_candidate_custody_changed_during_read")
    custody = strict_json_bytes(
        commit_before,
        role="candidate_custody_commit",
    )
    return artifacts, custody, candidate.name


def _revalidate_tokenizer_and_project(
    *,
    tokenization: Mapping[str, Any],
    tokenizer_directory: Path,
    snapshot_root: Path,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    max_seq_length: int,
    include_retention: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    snapshot = _lexical_absolute(Path(str(tokenization["snapshot_path"])))
    expected_root = _lexical_absolute(snapshot_root)
    if snapshot.parent != expected_root:
        _fail("trainer_tokenizer_snapshot_root_drift")
    source_identity = resident_tokenizer_artifact_identity(
        _lexical_absolute(tokenizer_directory)
    )
    snapshot_identity = resident_tokenizer_artifact_identity(snapshot)
    expected_identity = tokenization["tokenizer_identity_sha256"]
    if (
        source_identity.get("sha256") != expected_identity
        or snapshot_identity.get("sha256") != expected_identity
        or source_identity.get("files") != snapshot_identity.get("files")
    ):
        _fail("trainer_tokenizer_artifact_identity_drift")

    snapshot_manifest = strict_json_bytes(
        read_stable_bytes(
            snapshot / _SNAPSHOT_MANIFEST_FILE,
            max_bytes=_MAX_DOCUMENT_BYTES,
        ),
        role="tokenizer_snapshot_manifest",
    )
    snapshot_body = dict(snapshot_manifest)
    snapshot_sha256 = snapshot_body.pop("snapshot_manifest_sha256", None)
    if (
        snapshot_manifest.get("schema") != "aura.rlc.tokenizer_snapshot.v1"
        or snapshot_manifest.get("tokenizer_identity_sha256")
        != expected_identity
        or snapshot_manifest.get("files") != snapshot_identity.get("files")
        or not isinstance(snapshot_sha256, str)
        or hashlib.sha256(canonical_json_bytes(snapshot_body)).hexdigest()
        != snapshot_sha256
        or snapshot_sha256 != tokenization["snapshot_manifest_sha256"]
    ):
        _fail("trainer_tokenizer_snapshot_manifest_drift")

    tokenizer = load_resident_tokenizer(snapshot)
    runtime_before = resident_tokenizer_runtime_identity(tokenizer)
    if (
        runtime_before.get("sha256")
        != tokenization["tokenizer_runtime_identity_sha256"]
    ):
        _fail("trainer_tokenizer_runtime_identity_drift")
    projected_train = project_chat_rows(
        train_rows, tokenizer=tokenizer, max_seq_length=max_seq_length
    )
    projected_validation = project_chat_rows(
        validation_rows, tokenizer=tokenizer, max_seq_length=max_seq_length
    )
    if (
        len(projected_train) + len(projected_validation)
        != tokenization["rows_checked"]
    ):
        _fail("trainer_tokenizer_rows_checked_drift")
    if include_retention:
        projected_train.extend(
            project_chat_rows(
                build_retention_rows("train"),
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )
        )
        projected_validation.extend(
            project_chat_rows(
                build_retention_rows("validation"),
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
            )
        )
    runtime_after = resident_tokenizer_runtime_identity(tokenizer)
    if runtime_after != runtime_before:
        _fail("trainer_tokenizer_runtime_changed_during_projection")
    return projected_train, projected_validation


def _dataset_identity(
    *,
    candidate_sha256: str,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    sampler: str,
    seed: int,
) -> dict[str, Any]:
    if sampler == FAMILY_BALANCED_SAMPLER:
        order = family_balanced_epoch_order(
            train_rows,
            seed=seed,
            epoch=0,
        )
        body = {
            "schema": DATASET_SCHEMA_V2,
            "candidate_identity_sha256": candidate_sha256,
            "retention": retention_manifest(),
            "sampler": {
                "name": sampler,
                "epoch_zero_order": order,
                "epoch_zero_balance": family_balance_receipt(
                    train_rows,
                    order,
                ),
            },
            "train": list(train_rows),
            "validation": list(validation_rows),
            "holdout": None,
            "verified_replay": None,
        }
        return {**body, "dataset_sha256": sha256_json(body)}
    if sampler != "sha256_stateless_epoch_permutation.v1":
        _fail("trainer_dataset_sampler_invalid")
    body = {
        "schema": DATASET_SCHEMA,
        "candidate_identity_sha256": candidate_sha256,
        "train": list(train_rows),
        "validation": list(validation_rows),
        "holdout": None,
        "verified_replay": None,
    }
    return {**body, "dataset_sha256": sha256_json(body)}


def _write_create_or_verify(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise StructuredSFTResearchTrainingError(
                "trainer_existing_artifact_unreadable"
            ) from exc
        if observed != payload:
            _fail("trainer_existing_artifact_commitment_mismatch")
        return
    atomic_write_bytes(path, payload, mode=0o600)


def _validation_loss(
    model: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: RLCExecutionSpec,
    limit: int,
) -> dict[str, Any]:
    import mlx.core as mx

    from core.learning.recurrence_native_objective_v2 import live_path_loss

    count = min(limit, len(rows))
    losses: list[float] = []
    for row in rows[:count]:
        value = live_path_loss(
            model,
            row["prompt_tokens"],
            row["answer_tokens"],
            spec=spec,
        )
        try:
            mx.eval(value)
            loss = float(value)
        finally:
            del value
            mx.clear_cache()
        if not math.isfinite(loss) or loss < 0.0:
            _fail("trainer_validation_loss_nonfinite")
        losses.append(loss)
    return {
        "mean_loss": round(sum(losses) / len(losses), 8),
        "examples": len(losses),
        "execution_spec_sha256": spec.sha256,
        "scope": (
            "candidate_plus_source_bound_retention_validation"
            if any(
                row.get("family")
                in {
                    "identity_grounding",
                    "tool_effect_honesty",
                    "authority_safety",
                }
                for row in rows[:limit]
            )
            else "candidate_validation_only"
        ),
    }


def _signal_handler(_signal_number: int, _frame: Any) -> None:
    global _INTERRUPTED
    _INTERRUPTED = True


def _trainer_config(raw: Mapping[str, Any]) -> RecurrentSFTTrainerConfig:
    expected = {
        "schema",
        "training_mode",
        "sampler",
        "loss",
        "adapter_activation",
        "ordinary_lexical_activation",
        "validation_scope",
        "max_steps",
        "batch_size",
        "learning_rate",
        "optimizer",
        "weight_decay",
        "lora_rank",
        "lora_scale",
        "lora_dropout",
        "lora_targets",
        "checkpoint_every",
        "evaluate_every",
        "validation_examples",
        "max_seq_length",
        "max_minutes",
        "memory_fraction",
        "seed",
    }
    if set(raw) != expected:
        _fail("trainer_config_schema_invalid")
    material = {
        key: raw[key]
        for key in (
            "max_steps",
            "batch_size",
            "learning_rate",
            "optimizer",
            "weight_decay",
            "lora_rank",
            "lora_scale",
            "lora_dropout",
            "checkpoint_every",
            "evaluate_every",
            "validation_examples",
            "max_seq_length",
            "max_minutes",
            "memory_fraction",
            "seed",
        )
    }
    material["sampler"] = raw["sampler"]
    targets = raw.get("lora_targets")
    if not isinstance(targets, list):
        _fail("trainer_config_targets_invalid")
    material["lora_targets"] = tuple(targets)
    try:
        config = RecurrentSFTTrainerConfig(**material)
    except TypeError as exc:
        raise StructuredSFTResearchTrainingError(
            "trainer_config_invalid"
        ) from exc
    if config.to_dict() != dict(raw):
        _fail("trainer_config_reconstruction_mismatch")
    return config


def _bindings(
    authority: Mapping[str, Any],
    *,
    dataset_sha256: str,
) -> dict[str, str]:
    return {
        "authority_sha256": authority["authority_sha256"],
        "dataset_sha256": dataset_sha256,
        "tokenization_identity_sha256": authority["tokenization"][
            "identity_sha256"
        ],
        "model_identity_sha256": authority["model"]["identity_sha256"],
        "source_closure_sha256": authority["sources"]["closure_sha256"],
        "execution_spec_sha256": authority["execution_spec"][
            "semantic_sha256"
        ],
        "trainer_config_sha256": sha256_json(authority["trainer"]),
    }


def _epoch_order(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: RecurrentSFTTrainerConfig,
    epoch: int,
) -> list[int]:
    if config.sampler == FAMILY_BALANCED_SAMPLER:
        return family_balanced_epoch_order(
            rows,
            seed=config.seed,
            epoch=epoch,
        )
    from core.learning.structured_sft_research_authority import (
        deterministic_order,
    )

    return deterministic_order(
        len(rows),
        seed=config.seed,
        epoch=epoch,
    )


def _checkpoint_state(
    bindings: Mapping[str, str],
    *,
    step: int,
    epoch: int,
    cursor: int,
    order: list[int],
    config: RecurrentSFTTrainerConfig,
    train_count: int,
    validation_count: int,
    elapsed_s: float,
    invocation_count: int,
    loss_trail: list[dict[str, Any]],
    validation_trail: list[dict[str, Any]],
    pending_losses: list[float],
    baseline_validation: dict[str, Any],
    initial_adapter_sha256: str,
    terminal: bool,
) -> dict[str, Any]:
    body = {
        **bindings,
        "step": step,
        "optimizer_updates": step,
        "epoch": epoch,
        "cursor": cursor,
        "order": order,
        "sampler": config.sampler,
        "seed": config.seed,
        "train_example_count": train_count,
        "validation_example_count": validation_count,
        "elapsed_training_s": round(elapsed_s, 6),
        "invocation_count": invocation_count,
        "loss_trail": loss_trail,
        "validation_trail": validation_trail,
        "pending_losses": pending_losses,
        "baseline_validation": baseline_validation,
        "last_step_committed": True,
        "terminal": terminal,
    }
    if config.sampler == FAMILY_BALANCED_SAMPLER:
        body["initial_adapter_sha256"] = initial_adapter_sha256
    return body


def _run(arguments: argparse.Namespace) -> int:
    prospective_out_dir = arguments.out_dir.expanduser().resolve(strict=False)
    checkpoint_exists = (prospective_out_dir / "latest.json").exists()
    allow_expired_resume = (
        checkpoint_exists
        and arguments.resume_policy in {"auto", "required"}
    )
    authority = _read_json(arguments.authority, role="authority")
    audit_packet = _read_json(arguments.audit_packet, role="audit_packet")
    witness_bundle = _read_json(arguments.witness_bundle, role="witness_bundle")
    trusted_log_key = _read_pem(arguments.trusted_log_key)
    now = int(time.time())
    validated_authority = validate_authority(
        authority,
        expected_authority_sha256=arguments.expected_authority_sha256,
        now_unix=now,
        allow_expired_resume=allow_expired_resume,
    )
    verify_authority_upstream(
        validated_authority,
        audit_packet=audit_packet,
        witness_bundle=witness_bundle,
        trusted_log_public_key_pem=trusted_log_key,
        expected_sequence=arguments.witness_sequence,
    )

    candidate_artifacts, custody, candidate_directory_name = (
        _read_prevalidated_candidate(arguments.candidate_dir)
    )
    authorized = authorize_prevalidated_candidate_bytes(
        validated_authority,
        candidate_artifacts=candidate_artifacts,
        custody_attestation=custody,
        candidate_directory_name=candidate_directory_name,
        now_unix=now,
        expected_authority_sha256=arguments.expected_authority_sha256,
        allow_expired_resume=allow_expired_resume,
    )
    observed_candidate = validated_authority["candidate"]
    observed_tokenization = validated_authority["tokenization"]
    observed_model = small_model_identity(arguments.model_dir)
    if observed_model != validated_authority["model"]:
        _fail("trainer_model_binding_drift")
    execution_raw = _read_json(arguments.execution_spec, role="execution_spec")
    observed_execution = execution_spec_identity(execution_raw)
    if observed_execution != validated_authority["execution_spec"]:
        _fail("trainer_execution_spec_binding_drift")
    observed_sources = source_closure(_source_paths())
    if observed_sources != validated_authority["sources"]:
        _fail("trainer_source_binding_drift")

    config = _trainer_config(validated_authority["trainer"])
    spec = RLCExecutionSpec.from_dict(execution_raw)
    candidate_train_rows = _jsonl_rows(
        authorized["candidate_train.jsonl"],
        role="candidate_train",
    )
    candidate_validation_rows = _jsonl_rows(
        authorized["candidate_valid.jsonl"],
        role="candidate_validation",
    )
    train_rows, validation_rows = _revalidate_tokenizer_and_project(
        tokenization=observed_tokenization,
        tokenizer_directory=arguments.tokenizer_dir,
        snapshot_root=arguments.snapshot_root,
        train_rows=candidate_train_rows,
        validation_rows=candidate_validation_rows,
        max_seq_length=config.max_seq_length,
        include_retention=config.sampler == FAMILY_BALANCED_SAMPLER,
    )
    dataset = _dataset_identity(
        candidate_sha256=observed_candidate["identity_sha256"],
        train_rows=train_rows,
        validation_rows=validation_rows,
        sampler=config.sampler,
        seed=config.seed,
    )
    bindings = _bindings(
        validated_authority,
        dataset_sha256=dataset["dataset_sha256"],
    )

    out_dir = ensure_private_directory(arguments.out_dir.expanduser())
    if (out_dir / "research_completion.json").exists():
        _fail("trainer_terminal_completion_already_exists")
    _write_create_or_verify(
        out_dir / "projected_dataset_manifest.json",
        canonical_json_bytes(dataset),
    )
    append_journal_event(
        out_dir,
        event_type="ADMITTED",
        payload={
            **bindings,
            "candidate_directory": str(
                arguments.candidate_dir.expanduser().resolve(strict=True)
            ),
            "evaluator_path_accepted": False,
            "verified_replay_accepted": False,
            "production_effect": False,
        },
    )
    if arguments.admission_only:
        print(
            json.dumps(
                {
                    "status": "admission_revalidated_no_model_weights_loaded",
                    **bindings,
                },
                sort_keys=True,
            )
        )
        return 0

    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load

    from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
        full_weight_checkpoint_identity,
    )
    from core.learning.recurrence_native_objective_v2 import (
        exact_adjoint_live_path_value_and_grad,
    )
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane

    started_monotonic = time.monotonic()
    with (
        standalone_model_lane(
            owner_id=f"synthetic-recurrent-sft:{out_dir.name}",
            model_path=str(arguments.model_dir),
            purpose="training",
            preemptible=False,
            metadata={
                "tool": "train_structured_sft_research",
                "authority_sha256": validated_authority[
                    "authority_sha256"
                ],
                "production_effect": False,
            },
        ) as lease,
        mlx_memory_envelope(fraction=config.memory_fraction) as envelope,
    ):
        if getattr(lease, "active", False) is not True:
            _fail("trainer_model_lane_not_active")
        before_weights = full_weight_checkpoint_identity(arguments.model_dir)
        model, loaded_tokenizer = load(str(arguments.model_dir))
        loaded_runtime_before = resident_tokenizer_runtime_identity(
            loaded_tokenizer
        )
        if (
            loaded_runtime_before.get("sha256")
            != observed_tokenization["tokenizer_runtime_identity_sha256"]
        ):
            _fail("trainer_loaded_tokenizer_runtime_identity_drift")
        loaded_tokenization = project_chat_rows(
            _jsonl_rows(
                authorized["candidate_valid.jsonl"],
                role="loaded_candidate_validation",
            ),
            tokenizer=loaded_tokenizer,
            max_seq_length=config.max_seq_length,
        )
        if config.sampler == FAMILY_BALANCED_SAMPLER:
            loaded_tokenization.extend(
                project_chat_rows(
                    build_retention_rows("validation"),
                    tokenizer=loaded_tokenizer,
                    max_seq_length=config.max_seq_length,
                )
            )
        if (
            loaded_tokenization != validation_rows
            or resident_tokenizer_runtime_identity(loaded_tokenizer)
            != loaded_runtime_before
        ):
            _fail("trainer_loaded_tokenizer_projection_drift")
        mx.random.seed(config.seed)
        wrapped = wrap_recurrent_window(
            model,
            spec=spec,
            lora_rank=config.lora_rank,
            lora_dropout=config.lora_dropout,
            lora_scale=config.lora_scale,
            lora_targets=config.lora_targets,
        )
        expected_adapter = adapter_tensor_dict(model)
        mx.eval(expected_adapter)
        initial_adapter_sha256 = adapter_tensor_fingerprint(
            expected_adapter
        )
        optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        optimizer.init(model.trainable_parameters())
        step = 0
        epoch = 0
        cursor = 0
        order = _epoch_order(
            train_rows,
            config=config,
            epoch=epoch,
        )
        if (
            config.sampler == FAMILY_BALANCED_SAMPLER
            and config.max_steps % len(order) != 0
        ):
            _fail("trainer_balanced_sampler_requires_complete_epochs")
        if (
            config.sampler == FAMILY_BALANCED_SAMPLER
            and config.validation_examples < len(validation_rows)
        ):
            _fail("trainer_balanced_sampler_requires_complete_validation")
        prior_elapsed = 0.0
        invocation_count = 1
        loss_trail: list[dict[str, Any]] = []
        validation_trail: list[dict[str, Any]] = []
        pending_losses: list[float] = []
        baseline: dict[str, Any]
        terminal_checkpoint: Path | None = None
        if checkpoint_exists:
            if arguments.resume_policy == "never":
                _fail("trainer_checkpoint_exists_resume_required")
            loaded = load_checkpoint(
                out_dir,
                expected_bindings=bindings,
            )
            assert_adapter_tensor_topology(
                expected_adapter,
                loaded.adapter_tensors,
            )
            model.load_weights(
                list(loaded.adapter_tensors.items()),
                strict=False,
            )
            optimizer.state = loaded.optimizer_state
            optimizer.init(model.trainable_parameters())
            state = loaded.state
            step = state["step"]
            epoch = state["epoch"]
            cursor = state["cursor"]
            order = list(state["order"])
            if (
                state["sampler"] != config.sampler
                or order
                != _epoch_order(
                    train_rows,
                    config=config,
                    epoch=epoch,
                )
            ):
                _fail("trainer_resume_sample_order_drift")
            prior_elapsed = state["elapsed_training_s"]
            invocation_count = state["invocation_count"] + 1
            loss_trail = list(state["loss_trail"])
            validation_trail = list(state["validation_trail"])
            pending_losses = [
                float(value) for value in state["pending_losses"]
            ]
            baseline = dict(state["baseline_validation"])
            if state["terminal"]:
                terminal_checkpoint = loaded.checkpoint_dir
            append_journal_event(
                out_dir,
                event_type=(
                    "TERMINAL_FINALIZATION_RESUMED"
                    if state["terminal"]
                    else "RESUMED"
                ),
                payload={
                    "step": step,
                    "checkpoint": loaded.checkpoint_dir.name,
                    "invocation_count": invocation_count,
                },
            )
        elif arguments.resume_policy == "required":
            _fail("trainer_resume_checkpoint_missing")
        else:
            baseline = _validation_loss(
                model,
                validation_rows,
                spec=spec,
                limit=config.validation_examples,
            )

        append_journal_event(
            out_dir,
            event_type="MODEL_BOUND",
            payload={
                "model_identity_sha256": observed_model["identity_sha256"],
                "base_weight_fingerprint": before_weights["fingerprint"],
                "wrapped_projections": wrapped,
            },
        )

        def elapsed() -> float:
            return prior_elapsed + time.monotonic() - started_monotonic

        def checkpoint(*, terminal: bool) -> Path:
            adapter = adapter_tensor_dict(model)
            optimizer_tensors = dict(tree_flatten(optimizer.state))
            if not optimizer_tensors:
                _fail("trainer_optimizer_state_empty")
            return save_checkpoint(
                out_dir,
                adapter_tensors=adapter,
                optimizer_tensors=optimizer_tensors,
                state=_checkpoint_state(
                    bindings,
                    step=step,
                    epoch=epoch,
                    cursor=cursor,
                    order=order,
                    config=config,
                    train_count=len(train_rows),
                    validation_count=len(validation_rows),
                    elapsed_s=elapsed(),
                    invocation_count=invocation_count,
                    loss_trail=loss_trail,
                    validation_trail=validation_trail,
                    pending_losses=pending_losses,
                    baseline_validation=baseline,
                    initial_adapter_sha256=initial_adapter_sha256,
                    terminal=terminal,
                ),
            )

        halt_reason = "max_steps"
        while step < config.max_steps:
            if _INTERRUPTED:
                halt_reason = "interrupted"
                break
            if elapsed() >= config.max_minutes * 60.0:
                halt_reason = "wall_clock"
                break
            if cursor >= len(order):
                epoch += 1
                cursor = 0
                order = _epoch_order(
                    train_rows,
                    config=config,
                    epoch=epoch,
                )
            row = train_rows[order[cursor]]
            loss, gradients, _base_loss, branch_cosines = (
                exact_adjoint_live_path_value_and_grad(
                    model,
                    row["prompt_tokens"],
                    row["answer_tokens"],
                    spec=spec,
                )
            )
            if not math.isfinite(loss) or loss < 0.0:
                halt_reason = "nonfinite_loss"
                break
            optimizer.update(model, gradients)
            mx.eval(model.trainable_parameters(), optimizer.state)
            del gradients
            mx.clear_cache()
            envelope.reclaim(force=True)
            step += 1
            cursor += 1
            pending_losses.append(float(loss))
            if step % config.evaluate_every == 0 or step == config.max_steps:
                entry = {
                    "step": step,
                    "mean_training_loss": round(
                        sum(pending_losses) / len(pending_losses),
                        8,
                    ),
                    "window_steps": len(pending_losses),
                    "branch_cosines": [
                        round(float(value), 8)
                        for value in branch_cosines
                    ],
                }
                loss_trail.append(entry)
                pending_losses.clear()
                validation = _validation_loss(
                    model,
                    validation_rows,
                    spec=spec,
                    limit=config.validation_examples,
                )
                validation["step"] = step
                validation_trail.append(validation)
                print(
                    f"step={step} train_loss={entry['mean_training_loss']:.6f} "
                    f"valid_loss={validation['mean_loss']:.6f}",
                    flush=True,
                )
            if step % config.checkpoint_every == 0:
                published = checkpoint(terminal=False)
                append_journal_event(
                    out_dir,
                    event_type="CHECKPOINT",
                    payload={
                        "step": step,
                        "checkpoint": published.name,
                    },
                )

        terminal = step >= config.max_steps
        published = (
            terminal_checkpoint
            if terminal and terminal_checkpoint is not None
            else checkpoint(terminal=terminal)
        )
        after_weights = full_weight_checkpoint_identity(arguments.model_dir)
        if after_weights != before_weights:
            _fail("trainer_base_weights_changed")
        completion_body = {
            "schema": COMPLETION_SCHEMA,
            "authority_sha256": validated_authority["authority_sha256"],
            "dataset_sha256": dataset["dataset_sha256"],
            "model_identity_sha256": observed_model["identity_sha256"],
            "execution_spec_sha256": spec.sha256,
            "step": step,
            "halt_reason": halt_reason,
            "terminal": terminal,
            "baseline_validation": baseline,
            "final_validation": (
                validation_trail[-1] if validation_trail else baseline
            ),
            "checkpoint": published.name,
            "base_weights_unchanged": True,
            "output_disposition": "quarantined_research_only",
            "ordinary_lexical_adapter_activation": False,
            "production_effect": False,
            "promotion_allowed": False,
            "claims_not_supported": validated_authority[
                "claims_not_supported"
            ],
        }
        if config.sampler == FAMILY_BALANCED_SAMPLER:
            completion_body["initial_adapter_sha256"] = (
                initial_adapter_sha256
            )
        completion = {
            **completion_body,
            "completion_sha256": sha256_json(completion_body),
        }
        attempt_path = (
            out_dir
            / "attempts"
            / f"attempt-{invocation_count:04d}-{completion['completion_sha256']}.json"
        )
        ensure_private_directory(attempt_path.parent)
        _write_create_or_verify(
            attempt_path,
            canonical_json_bytes(completion),
        )
        if terminal:
            _write_create_or_verify(
                out_dir / "research_completion.json",
                canonical_json_bytes(completion),
            )
        append_journal_event(
            out_dir,
            event_type="TERMINAL" if terminal else "INTERRUPTED",
            payload={
                "step": step,
                "halt_reason": halt_reason,
                "checkpoint": published.name,
                "completion_sha256": completion["completion_sha256"],
            },
        )
        validate_journal(out_dir)
        print(json.dumps(completion, indent=2, sort_keys=True))
        return 0 if terminal else 75


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--audit-packet", type=Path, required=True)
    parser.add_argument("--witness-bundle", type=Path, required=True)
    parser.add_argument("--trusted-log-key", type=Path, required=True)
    parser.add_argument("--witness-sequence", type=int, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--execution-spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-policy",
        choices=sorted(RESUME_POLICIES),
        default="never",
        help="Checkpoint handling: never, auto for detached retries, or required.",
    )
    parser.add_argument(
        "--admission-only",
        action="store_true",
        help="Revalidate every pre-model-load binding and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if (
        not isinstance(arguments.expected_authority_sha256, str)
        or len(arguments.expected_authority_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in arguments.expected_authority_sha256
        )
        or arguments.witness_sequence < 1
        or arguments.resume_policy not in RESUME_POLICIES
        or arguments.admission_only
        and arguments.resume_policy != "never"
    ):
        _parser().error("authority, sequence, or mode argument is invalid")
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    try:
        return _run(arguments)
    except (
        FloatingPointError,
        ImportError,
        MemoryError,
        OSError,
        RuntimeError,
        StructuredSFTResearchAuthorityError,
        StructuredSFTResearchStateError,
        StructuredSFTResearchTrainingError,
        TokenizerValidationError,
        RecurrentSFTExecutionError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema": "aura.rlc.synthetic_recurrent_sft_error.v1",
                    "ok": False,
                    "reason": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
