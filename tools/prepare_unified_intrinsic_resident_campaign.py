#!/usr/bin/env python3
"""Freeze, but never launch, a resident unified-recurrence campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import secrets
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.frontier_tasks import (  # noqa: E402
    CONTAMINATION_SAFE_REGISTRY_VERSION,
)
from core.learning import recurrence_curriculum as curriculum  # noqa: E402
from core.learning.frontier_process_supervision import (  # noqa: E402
    frontier_process_task_battery,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes_if_absent,
    ensure_private_directory,
)
from tools.resident_recurrent_sft_bootstrap_identity import (  # noqa: E402
    load_resident_bootstrap_tokenizer,
    resident_bootstrap_tokenizer_identity,
)
from tools.unified_intrinsic_checkpoint import (  # noqa: E402
    UnifiedCheckpointError,
    resolve_checkpoint_generation,
)
from tools.unified_intrinsic_resident_identity import (  # noqa: E402
    build_model_manifest,
    build_source_git_identity,
    build_source_manifest,
    canonical_bytes,
    canonical_sha256,
    runtime_identity,
    verify_source_git_identity,
    verify_source_manifest,
)
from tools.unified_intrinsic_tokenization_contract import (  # noqa: E402
    SOURCE_DATASET_FILENAME,
    TOKENIZED_DATASET_FILENAME,
    freeze_source_dataset,
    freeze_tokenized_dataset,
)

PREPARATION_SCHEMA: Final = "aura.unified_intrinsic.resident_preparation.v1"
CONFIG_SCHEMA: Final = "aura.unified_intrinsic.resident_campaign.v1"
PROFILES: Final = frozenset(
    {
        "canary",
        "full",
        "process_action_canary",
        "process_canary",
        "process_family_acquisition",
        "process_neural_acquisition",
        "recovery",
    }
)
DEFAULT_MODEL: Final = Path(
    "/Users/bryan/.aura/live-source/training/fused-model/"
    "Aura-32B-crsm-closeout-jul1-20260701-215118"
)
DEFAULT_CAPSULE_ROOT: Final = Path.home() / ".aura/training-capsules"
DEFAULT_CAMPAIGN_ROOT: Final = Path.home() / ".aura/training-campaigns"
_CAMPAIGN_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")


class UnifiedResidentPreparationError(RuntimeError):
    """The campaign could not be frozen without weakening its identity."""


def _fail(code: str) -> Never:
    raise UnifiedResidentPreparationError(code)


def _canonical_document(value: Any) -> bytes:
    return canonical_bytes(value) + b"\n"


def _write_once(path: Path, payload: bytes, *, mode: int = 0o400) -> None:
    if path.is_symlink():
        _fail(f"immutable_path_is_symlink:{path.name}")
    ensure_private_directory(path.parent)
    atomic_write_bytes_if_absent(path, payload, mode=mode)
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            before = os.fstat(descriptor)
            chunks: list[bytes] = []
            remaining = len(payload) + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise UnifiedResidentPreparationError(f"immutable_artifact_unreadable:{path.name}") from exc
    observed = b"".join(chunks)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) != stat.S_IMODE(mode)
        or observed != payload
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        _fail(f"immutable_artifact_drift:{path.name}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str, timeout: float = 120.0) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().replace("\n", " ")[:500]
        _fail(f"git_command_failed:{arguments[0]}:{detail}")
    return result.stdout.strip()


def _full_commit(root: Path, value: str) -> str:
    commit = _git(root, "rev-parse", "--verify", f"{value}^{{commit}}")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        _fail("source_commit_invalid")
    return commit


def _require_published(root: Path, commit: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, "origin/main"],
        capture_output=True,
        check=False,
        timeout=30.0,
    )
    if result.returncode != 0:
        _fail("source_commit_not_published_on_origin_main")


def _validate_campaign_id(value: str) -> str:
    if _CAMPAIGN_ID.fullmatch(value) is None:
        _fail("campaign_id_invalid")
    return value


def _private_directory(path: Path, *, must_be_new: bool = False) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        _fail("campaign_directory_is_symlink")
    path = expanded.resolve(strict=False)
    if must_be_new and path.exists():
        _fail("campaign_directory_already_exists")
    ensure_private_directory(path)
    observed = path.stat()
    if not stat.S_ISDIR(observed.st_mode) or observed.st_uid != os.geteuid():
        _fail("campaign_directory_identity_invalid")
    os.chmod(path, 0o700)
    return path


def _profile_training(profile: str) -> dict[str, Any]:
    if profile not in PROFILES:
        _fail("campaign_profile_invalid")
    common: dict[str, Any] = {
        "window_tissue_mode": "controller_only",
        "task_source": "curriculum",
        "frontier_difficulties": "1",
        "frontier_registry_version": CONTAMINATION_SAFE_REGISTRY_VERSION,
        "prelude_fraction": 0.25,
        "coda_fraction": 0.25,
        "train_depths": "1,2,4",
        "heldout_depths": "8,16",
        "families": "khop,modular,register_trace",
        "task_depths": "1,2,4",
        "controller_rank": 64,
        "lora_rank": 8,
        "lora_targets": "o_proj,v_proj",
        "state_weight": 6.0,
        "stutter_weight": 0.1,
        "depth_basis_size": 4,
        "learning_rate": 0.00005,
        "recurrent_learning_rate": 0.00005,
        "state_learning_rate": 0.00005,
        "answer_bridge_learning_rate": 0.0005,
        "answer_bridge_inner_steps": 1,
        "answer_bridge_rollin_probability": 0.2,
        "answer_bridge_rollin_final_probability": 0.6,
        "student_rollin_probability": 0.0,
        "student_rollin_final_probability": 0.0,
        "state_teacher_forcing_probability": 0.0,
        "state_teacher_forcing_final_probability": 0.0,
        "process_curriculum": "joint",
        "process_family_batch_size": 1,
        "process_family_batch_mode": "same_family",
        "process_transformer_gradient_scale": 0.0,
        "process_query_gradient_scale": 0.0,
        "max_gradient_norm": 0.5,
        "checkpoint_every": 1,
        "checkpoint_group": 4,
        "grounding_batch_size": 32,
        "seed": 20260811433,
        "init_seed": 20260811433,
        "bridge": "assistant_answer",
        "memory_fraction": 0.48,
        "memory_limit_gb": 40.0,
        "cache_limit_gb": 2.0,
        "wired_limit_gb": 48.0,
    }
    if profile in {
        "process_action_canary",
        "process_canary",
        "process_family_acquisition",
        "process_neural_acquisition",
    }:
        families = (
            "novel_algorithms,mathematics,coding,scientific_inference,"
            "long_horizon_planning,calibration,misleading_premise"
        )
        acquisition = profile in {
            "process_family_acquisition",
            "process_neural_acquisition",
        }
        neural_acquisition = profile == "process_neural_acquisition"
        per_cell = 8 if acquisition else 2
        family_batch_size = 7 if neural_acquisition else 2 if acquisition else 1
        family_batch_mode = "balanced_families" if neural_acquisition else "same_family"
        training_examples = 7 * per_cell
        action_only = profile != "process_canary"
        process_steps = (
            per_cell * 8
            if neural_acquisition
            else 7 * (per_cell // family_batch_size) * 8
            if acquisition
            else training_examples * (8 if action_only else 20)
        )
        return {
            **common,
            "window_tissue_mode": "scoped_lora",
            "lora_targets": (
                "q_proj,o_proj,v_proj"
                if neural_acquisition
                else common["lora_targets"]
            ),
            "task_source": "frontier_process",
            "families": families,
            "task_depths": "3,4,5,6,8,10",
            "train_depths": "1,3,4,5,6,8,10",
            "heldout_depths": "12,16",
            "per_cell": per_cell,
            "holdout_per_cell": 3 if acquisition else 1,
            "max_steps": process_steps,
            "semantic_warmup_steps": 0,
            "state_warmup_steps": process_steps,
            "process_curriculum": ("action_workspace" if action_only else "factorized"),
            "process_family_batch_size": family_batch_size,
            "process_family_batch_mode": family_batch_mode,
            "process_transformer_gradient_scale": 0.1 if neural_acquisition else 0.0,
            "process_query_gradient_scale": 0.01 if neural_acquisition else 0.0,
            "answer_bridge_steps": 0,
            "answer_bridge_inner_steps": 1,
            "student_rollin_probability": 0.0,
            "student_rollin_final_probability": 0.5,
            "state_teacher_forcing_probability": 1.0,
            "state_teacher_forcing_final_probability": 0.0,
            "eval_every": (
                8
                if neural_acquisition
                else 28
                if acquisition
                else training_examples * (2 if action_only else 5)
            ),
            "checkpoint_every": (
                8 if neural_acquisition else 28 if acquisition else training_examples
            ),
            "state_learning_rate": 0.0005 if action_only else 0.00005,
            "seed": 2026081401,
            "init_seed": 2026081402,
            "memory_fraction": 0.35,
            "memory_limit_gb": 24.0,
            "wired_limit_gb": 28.0,
            "max_minutes": 90.0 if acquisition else 60.0,
        }
    if profile == "canary":
        # Two examples per cell twice reached 8/9 exact resident admission but
        # did not generalize the register-value readout reliably. Four remains
        # bounded while covering enough values to test transfer rather than
        # one-shot memorization.
        canary_per_cell = 4
        family_count = len(str(common["families"]).split(","))
        task_depth_count = len(str(common["task_depths"]).split(","))
        canary_bridge_steps = family_count * task_depth_count * canary_per_cell
        return {
            **common,
            "per_cell": canary_per_cell,
            "holdout_per_cell": 1,
            "max_steps": canary_bridge_steps + 1,
            "semantic_warmup_steps": 0,
            "state_warmup_steps": 0,
            "answer_bridge_steps": canary_bridge_steps,
            "answer_bridge_inner_steps": 32,
            "eval_every": family_count * task_depth_count,
            "max_minutes": 240.0,
        }
    if profile == "recovery":
        return {
            **common,
            "per_cell": 8,
            "holdout_per_cell": 3,
            "max_steps": 36,
            "semantic_warmup_steps": 0,
            "state_warmup_steps": 0,
            "answer_bridge_steps": 18,
            "answer_bridge_inner_steps": 16,
            "answer_bridge_learning_rate": 0.0001,
            "recurrent_learning_rate": 0.000025,
            "answer_bridge_rollin_probability": 0.6,
            "answer_bridge_rollin_final_probability": 0.8,
            "eval_every": 9,
            "seed": 20260812263,
            "max_minutes": 720.0,
        }
    full_per_cell = 8
    family_count = len(str(common["families"]).split(","))
    task_depth_count = len(str(common["task_depths"]).split(","))
    full_bridge_steps = family_count * task_depth_count * full_per_cell
    return {
        **common,
        "per_cell": full_per_cell,
        "holdout_per_cell": 3,
        "max_steps": full_bridge_steps + 1,
        "semantic_warmup_steps": 0,
        "state_warmup_steps": 0,
        "answer_bridge_steps": full_bridge_steps,
        "answer_bridge_inner_steps": 32,
        "eval_every": family_count * task_depth_count,
        "max_minutes": 2880.0,
    }


def _training_cli(training: Mapping[str, Any]) -> list[str]:
    flag_names = {
        "window_tissue_mode": "--window-tissue-mode",
        "task_source": "--task-source",
        "frontier_difficulties": "--frontier-difficulties",
        "frontier_registry_version": "--frontier-registry-version",
        "prelude_fraction": "--prelude-fraction",
        "coda_fraction": "--coda-fraction",
        "train_depths": "--train-depths",
        "heldout_depths": "--heldout-depths",
        "families": "--families",
        "task_depths": "--task-depths",
        "per_cell": "--per-cell",
        "holdout_per_cell": "--holdout-per-cell",
        "controller_rank": "--controller-rank",
        "lora_rank": "--lora-rank",
        "lora_targets": "--lora-targets",
        "state_weight": "--state-weight",
        "stutter_weight": "--stutter-weight",
        "depth_basis_size": "--depth-basis-size",
        "learning_rate": "--learning-rate",
        "recurrent_learning_rate": "--recurrent-learning-rate",
        "state_learning_rate": "--state-learning-rate",
        "answer_bridge_learning_rate": "--answer-bridge-learning-rate",
        "answer_bridge_inner_steps": "--answer-bridge-inner-steps",
        "answer_bridge_rollin_probability": "--answer-bridge-rollin-probability",
        "answer_bridge_rollin_final_probability": ("--answer-bridge-rollin-final-probability"),
        "student_rollin_probability": "--student-rollin-probability",
        "student_rollin_final_probability": "--student-rollin-final-probability",
        "state_teacher_forcing_probability": ("--state-teacher-forcing-probability"),
        "state_teacher_forcing_final_probability": ("--state-teacher-forcing-final-probability"),
        "process_curriculum": "--process-curriculum",
        "process_family_batch_size": "--process-family-batch-size",
        "process_family_batch_mode": "--process-family-batch-mode",
        "process_transformer_gradient_scale": ("--process-transformer-gradient-scale"),
        "process_query_gradient_scale": "--process-query-gradient-scale",
        "max_gradient_norm": "--max-gradient-norm",
        "max_steps": "--max-steps",
        "semantic_warmup_steps": "--semantic-warmup-steps",
        "state_warmup_steps": "--state-warmup-steps",
        "answer_bridge_steps": "--answer-bridge-steps",
        "max_minutes": "--max-minutes",
        "eval_every": "--eval-every",
        "checkpoint_every": "--checkpoint-every",
        "checkpoint_group": "--checkpoint-group",
        "grounding_batch_size": "--grounding-batch-size",
        "seed": "--seed",
        "init_seed": "--init-seed",
        "bridge": "--bridge",
        "memory_fraction": "--memory-fraction",
        "memory_limit_gb": "--memory-limit-gb",
        "cache_limit_gb": "--cache-limit-gb",
        "wired_limit_gb": "--wired-limit-gb",
    }
    if set(training) != set(flag_names):
        _fail("training_profile_contract_drift")
    arguments: list[str] = []
    for name, flag in flag_names.items():
        arguments.extend((flag, str(training[name])))
    return arguments


def _freeze_bootstrap_checkpoint(
    source_output: Path,
    destination_inputs: Path,
    *,
    stem: str,
    expected_checkpoint_sha256: str,
    expected_step: int,
) -> dict[str, Any]:
    try:
        selected = resolve_checkpoint_generation(
            source_output.expanduser(),
            stem=stem,
            required=True,
        )
    except (OSError, UnifiedCheckpointError, ValueError) as exc:
        raise UnifiedResidentPreparationError("bootstrap_checkpoint_invalid") from exc
    if selected is None:  # pragma: no cover - required=True is authoritative
        _fail("bootstrap_checkpoint_unavailable")
    if (
        selected.receipt.get("checkpoint_sha256") != expected_checkpoint_sha256
        or selected.receipt.get("step") != expected_step
    ):
        _fail("bootstrap_checkpoint_pin_mismatch")
    output = _private_directory(destination_inputs / "bootstrap-output", must_be_new=True)
    generations = _private_directory(output / "checkpoint_generations")
    generation = _private_directory(generations / selected.generation_dir.name)
    weights = generation / selected.weights_path.name
    shutil.copyfile(selected.weights_path, weights)
    os.chmod(weights, 0o400)
    if (
        weights.stat().st_size != selected.receipt["checkpoint_size_bytes"]
        or _file_sha256(weights) != selected.receipt["checkpoint_sha256"]
    ):
        _fail("bootstrap_checkpoint_copy_drift")
    _write_once(
        generation / "complete.json",
        canonical_bytes(selected.receipt) + b"\n",
        mode=0o400,
    )
    _write_once(
        output / f"{stem}_pointer.json",
        canonical_bytes(selected.pointer) + b"\n",
        mode=0o600,
    )
    os.chmod(generation, 0o500)
    identity = selected.receipt.get("identity")
    if not isinstance(identity, dict):
        _fail("bootstrap_checkpoint_identity_invalid")
    body = {
        "schema": "aura.unified_intrinsic.bootstrap_input.v1",
        "stem": stem,
        "output": str(output),
        "parent_step": selected.receipt["step"],
        "parent_checkpoint_sha256": selected.receipt["checkpoint_sha256"],
        "parent_receipt_sha256": selected.receipt["receipt_sha256"],
        "parent_identity_sha256": identity["identity_sha256"],
    }
    return {**body, "bootstrap_sha256": canonical_sha256(body)}


def _freeze_campaign(
    *,
    source_root: Path,
    source_commit: str,
    campaign_root: Path,
    campaign_id: str,
    profile: str,
    model_path: Path,
    bootstrap_output_dir: Path | None = None,
    bootstrap_stem: str = "checkpoint_latest",
    bootstrap_checkpoint_sha256: str | None = None,
    bootstrap_step: int | None = None,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve(strict=True)
    campaign_id = _validate_campaign_id(campaign_id)
    source_git = build_source_git_identity(source_root, source_commit=source_commit)
    source_manifest = build_source_manifest(source_root, source_commit=source_commit)
    verify_source_git_identity(source_root, source_git)
    verify_source_manifest(source_root, source_manifest)
    model_manifest = build_model_manifest(model_path)
    environment = runtime_identity()
    training = _profile_training(profile)
    training_args = _training_cli(training)

    bootstrap_profiles = {
        "process_action_canary",
        "process_family_acquisition",
        "process_neural_acquisition",
        "recovery",
    }
    if (profile in bootstrap_profiles) != (bootstrap_output_dir is not None):
        _fail("selected_profile_requires_exactly_one_bootstrap_checkpoint")
    bootstrap_pin_present = (
        bootstrap_checkpoint_sha256 is not None and bootstrap_step is not None
    )
    if (bootstrap_output_dir is not None) != bootstrap_pin_present:
        _fail("bootstrap_checkpoint_requires_exact_identity_pin")
    if bootstrap_pin_present and (
        _SHA256.fullmatch(str(bootstrap_checkpoint_sha256)) is None
        or not isinstance(bootstrap_step, int)
        or isinstance(bootstrap_step, bool)
        or bootstrap_step < 0
    ):
        _fail("bootstrap_checkpoint_identity_pin_invalid")

    root = _private_directory(campaign_root / campaign_id, must_be_new=True)
    inputs = _private_directory(root / "inputs")
    _private_directory(root / "training-output")
    detached_attempts = _private_directory(root / "detached-attempts")
    del detached_attempts
    bootstrap = (
        _freeze_bootstrap_checkpoint(
            bootstrap_output_dir,
            inputs,
            stem=bootstrap_stem,
            expected_checkpoint_sha256=str(bootstrap_checkpoint_sha256),
            expected_step=int(bootstrap_step),
        )
        if bootstrap_output_dir is not None
        else None
    )
    families = tuple(str(training["families"]).split(","))
    task_depths = tuple(int(value) for value in str(training["task_depths"]).split(","))
    if training["task_source"] == "frontier_process":
        difficulties = tuple(
            int(value) for value in str(training["frontier_difficulties"]).split(",")
        )
        train_tasks = frontier_process_task_battery(
            families,
            difficulties,
            int(training["per_cell"]),
            seed=int(training["seed"]),
            registry_version=str(training["frontier_registry_version"]),
        )
        holdout_tasks = frontier_process_task_battery(
            families,
            difficulties,
            int(training["holdout_per_cell"]),
            seed=int(training["seed"]) + 9_973,
            registry_version=str(training["frontier_registry_version"]),
            excluded_prompts=tuple(task.prompt for task in train_tasks),
        )
    else:
        train_tasks = curriculum.task_battery(
            families,
            task_depths,
            int(training["per_cell"]),
            seed=int(training["seed"]),
        )
        holdout_tasks = curriculum.task_battery(
            families,
            task_depths,
            int(training["holdout_per_cell"]),
            seed=int(training["seed"]) + 9_973,
            excluded_prompts=tuple(task.prompt for task in train_tasks),
            excluded_task_ids=tuple(task.task_id for task in train_tasks),
        )
    random.Random(int(training["seed"])).shuffle(train_tasks)
    dataset_identity = freeze_source_dataset(
        inputs / SOURCE_DATASET_FILENAME,
        train_tasks,
        holdout_tasks,
    )
    tokenizer = load_resident_bootstrap_tokenizer(model_path)
    tokenizer_identity = resident_bootstrap_tokenizer_identity(model_path, tokenizer)
    bridge = {"assistant_answer": "\n\nFINAL_ANSWER: "}.get(
        str(training["bridge"]),
        str(training["bridge"]),
    )
    tokenized_dataset_identity = freeze_tokenized_dataset(
        inputs / TOKENIZED_DATASET_FILENAME,
        tokenizer,
        train_tasks,
        holdout_tasks,
        bridge=bridge,
        dataset_identity=dataset_identity,
        tokenizer_identity_sha256=tokenizer_identity["identity_sha256"],
    )
    heartbeat_key = secrets.token_bytes(32)
    key_path = root / "heartbeat.key"
    _write_once(key_path, heartbeat_key, mode=0o400)
    now = datetime.now(UTC).isoformat()
    body: dict[str, Any] = {
        "schema": CONFIG_SCHEMA,
        "campaign_id": campaign_id,
        "profile": profile,
        "prepared_at": now,
        "source": {
            "git": source_git,
            "manifest": source_manifest,
        },
        "model": model_manifest,
        "runtime": environment,
        "dataset": dataset_identity,
        "tokenizer": tokenizer_identity,
        "tokenized_dataset": tokenized_dataset_identity,
        "bootstrap": bootstrap,
        "paths": {
            "workspace_root": str(Path.home() / ".aura"),
            "campaign_root": str(root),
            "training_output": str(root / "training-output"),
            "inputs": str(inputs),
            "dataset": str(inputs / SOURCE_DATASET_FILENAME),
            "tokenized_dataset": str(inputs / TOKENIZED_DATASET_FILENAME),
            "detached_attempts": str(root / "detached-attempts"),
            "heartbeat_key": str(key_path),
            **(
                {"bootstrap_output": str(inputs / "bootstrap-output")}
                if bootstrap is not None
                else {}
            ),
        },
        "heartbeat_key_sha256": hashlib.sha256(heartbeat_key).hexdigest(),
        "training": training,
        "training_args": training_args,
        "watchdog": {
            "poll_interval_s": 15.0,
            "heartbeat_stale_s": 180.0,
            "attempt_timeout_s": (
                5.0 * 3600.0
                if profile
                in {
                    "canary",
                    "process_action_canary",
                    "process_canary",
                    "process_family_acquisition",
                    "process_neural_acquisition",
                }
                else 14.0 * 3600.0
                if profile == "recovery"
                else 54.0 * 3600.0
            ),
            "max_attempts": 8,
            "max_consecutive_no_progress": 2,
            "retry_backoff_s": 30.0,
        },
        "launch": {
            "label": f"com.aura.unified-intrinsic.{campaign_id}",
            "launchd_required": True,
            "trainer_caffeinate_required": True,
            "immutable_target_command": True,
        },
        "claims": {
            "resident_mechanics_proven": False,
            "reasoning_gain_proven": False,
            "frontier_level_proven": False,
            "fusion_allowed": False,
        },
    }
    config = {**body, "config_sha256": canonical_sha256(body)}
    config_path = root / "campaign.json"
    _write_once(config_path, _canonical_document(config), mode=0o400)
    preparation_body = {
        "schema": PREPARATION_SCHEMA,
        "campaign_id": campaign_id,
        "profile": profile,
        "config_path": str(config_path),
        "config_sha256": config["config_sha256"],
        "source_commit": source_commit,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "model_manifest_sha256": model_manifest["manifest_sha256"],
        "runtime_identity_sha256": environment["identity_sha256"],
        "dataset_identity_sha256": dataset_identity["identity_sha256"],
        "tokenizer_identity_sha256": tokenizer_identity["identity_sha256"],
        "tokenized_dataset_identity_sha256": tokenized_dataset_identity["identity_sha256"],
        "training_arguments_sha256": canonical_sha256(training_args),
        "claims_supported": [
            "source_model_runtime_and_exact_tokenized_dataset_frozen_before_launch"
        ],
        "claims_not_supported": [
            "resident_mechanics_proven",
            "reasoning_gain_proven",
            "frontier_level_proven",
            "fusion_allowed",
        ],
        "prepared_at": now,
    }
    preparation = {
        **preparation_body,
        "preparation_sha256": canonical_sha256(preparation_body),
    }
    _write_once(
        root / "preparation.json",
        _canonical_document(preparation),
        mode=0o400,
    )
    return preparation


def _create_capsule_and_freeze(args: argparse.Namespace) -> dict[str, Any]:
    if (args.bootstrap_output_dir is not None) != (
        args.bootstrap_checkpoint_sha256 is not None and args.bootstrap_step is not None
    ):
        _fail("bootstrap_checkpoint_requires_exact_identity_pin")
    commit = _full_commit(REPO_ROOT, args.source_commit)
    _require_published(REPO_ROOT, commit)
    capsule_root = _private_directory(args.capsule_root)
    capsule = capsule_root / f"unified-intrinsic-{commit[:12]}"
    if capsule.exists():
        identity = build_source_git_identity(capsule, source_commit=commit)
        verify_source_git_identity(capsule, identity)
    else:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", str(capsule), commit],
            capture_output=True,
            check=False,
            text=True,
            timeout=300.0,
        )
        if result.returncode != 0:
            _fail(f"source_capsule_creation_failed:{result.stderr.strip()[:500]}")
    command = [
        sys.executable,
        str(capsule / "tools/prepare_unified_intrinsic_resident_campaign.py"),
        "_freeze",
        "--source-root",
        str(capsule),
        "--source-commit",
        commit,
        "--campaign-root",
        str(args.campaign_root),
        "--campaign-id",
        args.campaign_id,
        "--profile",
        args.profile,
        "--model",
        str(args.model),
    ]
    if args.bootstrap_output_dir is not None:
        command.extend(
            (
                "--bootstrap-output-dir",
                str(args.bootstrap_output_dir.expanduser().resolve(strict=True)),
                "--bootstrap-stem",
                args.bootstrap_stem,
                "--bootstrap-checkpoint-sha256",
                args.bootstrap_checkpoint_sha256,
                "--bootstrap-step",
                str(args.bootstrap_step),
            )
        )
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=1800.0,
    )
    if result.returncode != 0:
        _fail(f"capsule_preparation_failed:{result.stderr.strip()[:1000]}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise UnifiedResidentPreparationError("capsule_preparation_receipt_invalid") from exc
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--profile", choices=sorted(PROFILES), required=True)
    prepare.add_argument("--campaign-id", required=True)
    prepare.add_argument("--source-commit", default="HEAD")
    prepare.add_argument("--capsule-root", type=Path, default=DEFAULT_CAPSULE_ROOT)
    prepare.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    prepare.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    prepare.add_argument("--bootstrap-output-dir", type=Path)
    prepare.add_argument("--bootstrap-stem", default="checkpoint_latest")
    prepare.add_argument("--bootstrap-checkpoint-sha256")
    prepare.add_argument("--bootstrap-step", type=int)

    freeze = commands.add_parser("_freeze", help=argparse.SUPPRESS)
    freeze.add_argument("--source-root", type=Path, required=True)
    freeze.add_argument("--source-commit", required=True)
    freeze.add_argument("--campaign-root", type=Path, required=True)
    freeze.add_argument("--campaign-id", required=True)
    freeze.add_argument("--profile", choices=sorted(PROFILES), required=True)
    freeze.add_argument("--model", type=Path, required=True)
    freeze.add_argument("--bootstrap-output-dir", type=Path)
    freeze.add_argument("--bootstrap-stem", default="checkpoint_latest")
    freeze.add_argument("--bootstrap-checkpoint-sha256")
    freeze.add_argument("--bootstrap-step", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "prepare":
            payload = _create_capsule_and_freeze(args)
        else:
            payload = _freeze_campaign(
                source_root=args.source_root,
                source_commit=args.source_commit,
                campaign_root=args.campaign_root,
                campaign_id=args.campaign_id,
                profile=args.profile,
                model_path=args.model.expanduser().resolve(strict=True),
                bootstrap_output_dir=args.bootstrap_output_dir,
                bootstrap_stem=args.bootstrap_stem,
                bootstrap_checkpoint_sha256=args.bootstrap_checkpoint_sha256,
                bootstrap_step=args.bootstrap_step,
            )
    except Exception as exc:  # noqa: BLE001 - stable CLI failure boundary
        print(
            f"prepare_unified_intrinsic_resident_campaign: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
