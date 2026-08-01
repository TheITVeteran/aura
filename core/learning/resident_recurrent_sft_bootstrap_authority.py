"""Resident-only authority for cached recurrent supervised bootstrapping.

This authority is intentionally separate from structured synthetic SFT research.
It permits a source-bound resident checkpoint to learn the exact cached recurrent
answer channel, but it cannot establish reasoning gain, authorize GRPO, promote
an adapter, or modify the base checkpoint. Those transitions require fresh
heldout causal lesion/restoration and broad-regression evidence.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Final, Never

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.frontier_tasks import (
    CURRENT_EXCLUDED_TRAINING_FAMILIES,
    CURRENT_REGISTRY_VERSION,
)
from core.learning.recurrence_curriculum import RECURRENCE_TRAINING_FAMILIES

AUTHORITY_SCHEMA: Final = "aura.resident_recurrent_sft_bootstrap_authority.v1"
DATASET_SCHEMA: Final = "aura.resident_recurrent_sft_bootstrap_dataset.v1"
TRAINER_CONFIG_SCHEMA: Final = "aura.resident_recurrent_sft_bootstrap_config.v1"
TRAINING_AUTHORITY: Final = "resident_32b_cached_recurrent_sft_bootstrap_only"
OBJECTIVE_NAME: Final = "cached_supervised_live_path_ce.v1"

REQUIRED_SOURCE_ROLES: Final = frozenset(
    {
        "authority",
        "state",
        "trainer",
        "preparer",
        "controller",
        "objective",
        "execution_spec",
        "recurrence_adapter",
        "adapter_identity",
        "curriculum",
        "tokenizer_validator",
        "campaign_journal",
        "campaign_trust",
        "campaign_launch_bundle",
        "detached_campaign_evidence",
        "detached_runner",
        "atomic_writer",
        "model_lane_control",
        "mlx_memory_guard",
    }
)

CLAIMS_NOT_SUPPORTED: Final = (
    "reasoning_gain",
    "frontier_level_reasoning",
    "positive_recurrent_interaction",
    "broad_capability_gain",
    "production_promotion",
    "release_eligibility",
    "grpo_admission",
    "wow_signal",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_MAX_ROWS = 100_000
_MAX_TEXT_BYTES = 64 * 1024
_MAX_ARTIFACT_BYTES = 1 << 50


class ResidentSFTBootstrapAuthorityError(ValueError):
    """Stable fail-closed authority error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ResidentSFTBootstrapAuthorityError(code)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _exact(value: Any, keys: set[str], *, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        _fail(f"{role}_schema_invalid")
    return value


def _sha(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{role}_sha256_invalid")
    return value


def _identifier(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(f"{role}_identifier_invalid")
    return value


def _relative_path(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail(f"{role}_path_invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail(f"{role}_path_invalid")
    return value


def _integer(value: Any, *, role: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{role}_invalid")
    return value


def _finite(value: Any, *, role: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        _fail(f"{role}_invalid")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        _fail(f"{role}_invalid")
    if not math.isfinite(number) or not minimum <= number <= maximum:
        _fail(f"{role}_invalid")
    return number


def artifact_binding(value: Any, *, role: str) -> dict[str, Any]:
    record = _exact(value, {"path", "sha256", "size_bytes"}, role=role)
    return {
        "path": _relative_path(record["path"], role=role),
        "sha256": _sha(record["sha256"], role=role),
        "size_bytes": _integer(
            record["size_bytes"],
            role=f"{role}_size",
            minimum=1,
            maximum=_MAX_ARTIFACT_BYTES,
        ),
    }


@dataclass(frozen=True, slots=True)
class ResidentSFTBootstrapConfig:
    seed: int
    max_steps: int = 96
    max_invocation_steps: int = 4
    max_minutes: float = 480.0
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    lora_rank: int = 8
    lora_scale: float = 20.0
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj", "o_proj")
    lora_layers: int = 8
    checkpoint_every: int = 1
    evaluate_every: int = 8
    validation_examples: int = 24
    max_seq_length: int = 512
    memory_fraction: float = 0.72
    branch_indices: tuple[int, ...] = (0, 1)
    objective: str = OBJECTIVE_NAME
    optimizer: str = "adamw"
    sampler: str = "seeded_family_depth_balanced_without_replacement"
    token_weighting: str = "uniform_nonnegative_normalized"
    schema: str = TRAINER_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRAINER_CONFIG_SCHEMA:
            _fail("resident_sft_config_schema_invalid")
        if self.objective != OBJECTIVE_NAME:
            _fail("resident_sft_config_objective_invalid")
        if self.optimizer != "adamw":
            _fail("resident_sft_config_optimizer_invalid")
        if self.sampler != "seeded_family_depth_balanced_without_replacement":
            _fail("resident_sft_config_sampler_invalid")
        if self.token_weighting != "uniform_nonnegative_normalized":
            _fail("resident_sft_config_token_weighting_invalid")
        _integer(self.seed, role="resident_sft_seed", minimum=0, maximum=2**63 - 1)
        _integer(self.max_steps, role="resident_sft_max_steps", minimum=1, maximum=10_000)
        _integer(
            self.max_invocation_steps,
            role="resident_sft_max_invocation_steps",
            minimum=1,
            maximum=128,
        )
        if self.max_invocation_steps > self.max_steps:
            _fail("resident_sft_invocation_exceeds_campaign")
        _finite(
            self.max_minutes,
            role="resident_sft_max_minutes",
            minimum=1.0,
            maximum=7 * 24 * 60.0,
        )
        _finite(
            self.learning_rate,
            role="resident_sft_learning_rate",
            minimum=1e-9,
            maximum=1e-2,
        )
        _finite(
            self.weight_decay,
            role="resident_sft_weight_decay",
            minimum=0.0,
            maximum=1.0,
        )
        _integer(self.lora_rank, role="resident_sft_lora_rank", minimum=1, maximum=256)
        _finite(
            self.lora_scale,
            role="resident_sft_lora_scale",
            minimum=1e-6,
            maximum=1024.0,
        )
        _finite(
            self.lora_dropout,
            role="resident_sft_lora_dropout",
            minimum=0.0,
            maximum=0.5,
        )
        allowed_targets = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        if (
            not self.lora_targets
            or len(self.lora_targets) != len(set(self.lora_targets))
            or any(target not in allowed_targets for target in self.lora_targets)
        ):
            _fail("resident_sft_lora_targets_invalid")
        _integer(self.lora_layers, role="resident_sft_lora_layers", minimum=1, maximum=256)
        _integer(
            self.checkpoint_every,
            role="resident_sft_checkpoint_every",
            minimum=1,
            maximum=128,
        )
        if self.checkpoint_every != 1:
            _fail("resident_sft_checkpoint_must_cover_every_committed_step")
        _integer(
            self.evaluate_every,
            role="resident_sft_evaluate_every",
            minimum=1,
            maximum=self.max_steps,
        )
        _integer(
            self.validation_examples,
            role="resident_sft_validation_examples",
            minimum=1,
            maximum=_MAX_ROWS,
        )
        _integer(
            self.max_seq_length,
            role="resident_sft_max_seq_length",
            minimum=32,
            maximum=32_768,
        )
        _finite(
            self.memory_fraction,
            role="resident_sft_memory_fraction",
            minimum=0.1,
            maximum=0.9,
        )
        if (
            not self.branch_indices
            or len(self.branch_indices) != len(set(self.branch_indices))
            or any(type(index) is not int or index < 0 or index > 31 for index in self.branch_indices)
        ):
            _fail("resident_sft_branch_indices_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "objective": self.objective,
            "optimizer": self.optimizer,
            "sampler": self.sampler,
            "token_weighting": self.token_weighting,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "max_invocation_steps": self.max_invocation_steps,
            "max_minutes": self.max_minutes,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "lora_rank": self.lora_rank,
            "lora_scale": self.lora_scale,
            "lora_dropout": self.lora_dropout,
            "lora_targets": list(self.lora_targets),
            "lora_layers": self.lora_layers,
            "checkpoint_every": self.checkpoint_every,
            "evaluate_every": self.evaluate_every,
            "validation_examples": self.validation_examples,
            "max_seq_length": self.max_seq_length,
            "memory_fraction": self.memory_fraction,
            "branch_indices": list(self.branch_indices),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ResidentSFTBootstrapConfig:
        expected = set(cls(seed=0).to_dict())
        record = _exact(raw, expected, role="resident_sft_config")
        try:
            return cls(
                schema=record["schema"],
                objective=record["objective"],
                optimizer=record["optimizer"],
                sampler=record["sampler"],
                token_weighting=record["token_weighting"],
                seed=record["seed"],
                max_steps=record["max_steps"],
                max_invocation_steps=record["max_invocation_steps"],
                max_minutes=record["max_minutes"],
                learning_rate=record["learning_rate"],
                weight_decay=record["weight_decay"],
                lora_rank=record["lora_rank"],
                lora_scale=record["lora_scale"],
                lora_dropout=record["lora_dropout"],
                lora_targets=tuple(record["lora_targets"]),
                lora_layers=record["lora_layers"],
                checkpoint_every=record["checkpoint_every"],
                evaluate_every=record["evaluate_every"],
                validation_examples=record["validation_examples"],
                max_seq_length=record["max_seq_length"],
                memory_fraction=record["memory_fraction"],
                branch_indices=tuple(record["branch_indices"]),
            )
        except (TypeError, ValueError, KeyError) as exc:
            if isinstance(exc, ResidentSFTBootstrapAuthorityError):
                raise
            raise ResidentSFTBootstrapAuthorityError(
                "resident_sft_config_schema_invalid"
            ) from exc


def _normalized_row(value: Any, *, split: str, index: int) -> dict[str, Any]:
    row = _exact(
        value,
        {"task_id", "family", "depth", "prompt", "answer"},
        role=f"resident_sft_{split}_row",
    )
    task_id = _identifier(row["task_id"], role=f"resident_sft_{split}_task_id")
    family = _identifier(row["family"], role=f"resident_sft_{split}_family")
    if family not in RECURRENCE_TRAINING_FAMILIES:
        _fail(f"resident_sft_{split}_family_not_training_only")
    depth = _integer(
        row["depth"],
        role=f"resident_sft_{split}_depth",
        minimum=1,
        maximum=64,
    )
    prompt = row["prompt"]
    answer = row["answer"]
    for role, text in (("prompt", prompt), ("answer", answer)):
        if (
            not isinstance(text, str)
            or not text
            or text != text.strip()
            or "\x00" in text
            or len(text.encode("utf-8")) > _MAX_TEXT_BYTES
        ):
            _fail(f"resident_sft_{split}_{role}_invalid")
    if not answer.startswith("FINAL_ANSWER:"):
        _fail(f"resident_sft_{split}_answer_contract_invalid")
    if prompt == answer:
        _fail(f"resident_sft_{split}_answer_leak")
    return {
        "task_id": task_id,
        "family": family,
        "depth": depth,
        "prompt": prompt,
        "answer": answer,
        "ordinal": index,
    }


def _normalized_splits(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 1 <= len(train_rows) <= _MAX_ROWS:
        _fail("resident_sft_train_count_invalid")
    if not 1 <= len(validation_rows) <= _MAX_ROWS:
        _fail("resident_sft_validation_count_invalid")
    train = [
        _normalized_row(row, split="train", index=index)
        for index, row in enumerate(train_rows)
    ]
    validation = [
        _normalized_row(row, split="validation", index=index)
        for index, row in enumerate(validation_rows)
    ]
    train_ids = {row["task_id"] for row in train}
    validation_ids = {row["task_id"] for row in validation}
    if len(train_ids) != len(train) or len(validation_ids) != len(validation):
        _fail("resident_sft_task_id_duplicate")
    if train_ids & validation_ids:
        _fail("resident_sft_train_validation_id_overlap")
    train_prompts = {sha256_bytes(row["prompt"].encode()) for row in train}
    validation_prompts = {sha256_bytes(row["prompt"].encode()) for row in validation}
    if train_prompts & validation_prompts:
        _fail("resident_sft_train_validation_prompt_overlap")
    return train, validation


def canonical_dataset_payloads(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> tuple[bytes, bytes]:
    train, validation = _normalized_splits(train_rows, validation_rows)
    return canonical_json_bytes(train), canonical_json_bytes(validation)


def build_dataset_commitment(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train, validation = _normalized_splits(train_rows, validation_rows)
    train_payload = canonical_json_bytes(train)
    validation_payload = canonical_json_bytes(validation)
    family_counts = Counter(row["family"] for row in (*train, *validation))
    material = {
        "schema": DATASET_SCHEMA,
        "train_sha256": sha256_bytes(train_payload),
        "train_size_bytes": len(train_payload),
        "train_count": len(train),
        "validation_sha256": sha256_bytes(validation_payload),
        "validation_size_bytes": len(validation_payload),
        "validation_count": len(validation),
        "families": sorted(family_counts),
        "family_counts": dict(sorted(family_counts.items())),
        "depths": sorted({row["depth"] for row in (*train, *validation)}),
        "train_validation_id_overlap": 0,
        "train_validation_prompt_overlap": 0,
        "evaluation_registry": CURRENT_REGISTRY_VERSION,
        "training_only_families": sorted(CURRENT_EXCLUDED_TRAINING_FAMILIES),
        "answer_contract": "final_answer_v1",
    }
    if not set(material["families"]).issubset(CURRENT_EXCLUDED_TRAINING_FAMILIES):
        _fail("resident_sft_evaluation_family_contamination")
    return {**material, "dataset_sha256": sha256_json(material)}


def _validate_dataset(value: Any) -> dict[str, Any]:
    expected = {
        "schema",
        "train_sha256",
        "train_size_bytes",
        "train_count",
        "validation_sha256",
        "validation_size_bytes",
        "validation_count",
        "families",
        "family_counts",
        "depths",
        "train_validation_id_overlap",
        "train_validation_prompt_overlap",
        "evaluation_registry",
        "training_only_families",
        "answer_contract",
        "dataset_sha256",
    }
    record = dict(_exact(value, expected, role="resident_sft_dataset"))
    claimed = _sha(record.pop("dataset_sha256"), role="resident_sft_dataset")
    if record.get("schema") != DATASET_SCHEMA or claimed != sha256_json(record):
        _fail("resident_sft_dataset_digest_mismatch")
    for role in ("train", "validation"):
        _sha(record[f"{role}_sha256"], role=f"resident_sft_{role}")
        _integer(
            record[f"{role}_size_bytes"],
            role=f"resident_sft_{role}_size",
            minimum=1,
            maximum=_MAX_ARTIFACT_BYTES,
        )
        _integer(
            record[f"{role}_count"],
            role=f"resident_sft_{role}_count",
            minimum=1,
            maximum=_MAX_ROWS,
        )
    families = record.get("families")
    family_counts = record.get("family_counts")
    depths = record.get("depths")
    if (
        not isinstance(families, list)
        or families != sorted(set(families))
        or not set(families).issubset(CURRENT_EXCLUDED_TRAINING_FAMILIES)
        or not isinstance(family_counts, Mapping)
        or sorted(family_counts) != families
        or any(type(count) is not int or count < 1 for count in family_counts.values())
        or not isinstance(depths, list)
        or depths != sorted(set(depths))
        or any(type(depth) is not int or not 1 <= depth <= 64 for depth in depths)
        or record.get("train_validation_id_overlap") != 0
        or record.get("train_validation_prompt_overlap") != 0
        or record.get("evaluation_registry") != CURRENT_REGISTRY_VERSION
        or record.get("training_only_families")
        != sorted(CURRENT_EXCLUDED_TRAINING_FAMILIES)
        or record.get("answer_contract") != "final_answer_v1"
    ):
        _fail("resident_sft_dataset_policy_invalid")
    return {**record, "dataset_sha256": claimed}


def _identity_mapping(value: Any, *, role: str, digest_field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        _fail(f"{role}_identity_invalid")
    normalized = dict(value)
    _sha(normalized.get(digest_field), role=role)
    return normalized


def _validate_time_window(committed_at: Any, expires_at: Any) -> tuple[str, str]:
    if not isinstance(committed_at, str) or not isinstance(expires_at, str):
        _fail("resident_sft_authority_time_invalid")
    try:
        committed = datetime.fromisoformat(committed_at)
        expires = datetime.fromisoformat(expires_at)
    except ValueError as exc:
        raise ResidentSFTBootstrapAuthorityError(
            "resident_sft_authority_time_invalid"
        ) from exc
    if committed.tzinfo is None or expires.tzinfo is None or expires <= committed:
        _fail("resident_sft_authority_time_invalid")
    return committed_at, expires_at


def build_authority(
    *,
    campaign_id: str,
    committed_at: str,
    expires_at: str,
    model_path: str,
    model_identity: Mapping[str, Any],
    behavior_identity: Mapping[str, Any],
    personality_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    execution_spec: Mapping[str, Any],
    dataset: Mapping[str, Any],
    dataset_artifacts: Mapping[str, Any],
    sources: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    trust_policy: Mapping[str, Any],
    artifact_root: str,
    config: ResidentSFTBootstrapConfig,
) -> dict[str, Any]:
    _identifier(campaign_id, role="resident_sft_campaign")
    if not campaign_id.startswith("resident-32b-recurrent-sft-bootstrap-cp"):
        _fail("resident_sft_campaign_identity_invalid")
    committed_at, expires_at = _validate_time_window(committed_at, expires_at)
    if not isinstance(config, ResidentSFTBootstrapConfig):
        _fail("resident_sft_config_invalid")
    normalized_dataset = _validate_dataset(dataset)
    normalized_artifacts = {
        split: artifact_binding(dataset_artifacts.get(split), role=f"dataset_{split}")
        for split in ("train", "validation")
    }
    for split in ("train", "validation"):
        binding = normalized_artifacts[split]
        if (
            binding["sha256"] != normalized_dataset[f"{split}_sha256"]
            or binding["size_bytes"] != normalized_dataset[f"{split}_size_bytes"]
        ):
            _fail(f"resident_sft_{split}_artifact_binding_mismatch")
    if set(sources) != REQUIRED_SOURCE_ROLES:
        _fail("resident_sft_source_roles_invalid")
    normalized_sources = {
        role: artifact_binding(sources[role], role=f"source_{role}")
        for role in sorted(REQUIRED_SOURCE_ROLES)
    }
    normalized_model = _exact(
        model_identity,
        {"fingerprint", "method", "files"},
        role="resident_sft_model_identity",
    )
    if (
        _sha(normalized_model["fingerprint"], role="resident_sft_model")
        != normalized_model["fingerprint"]
        or normalized_model["method"] != "sha256"
        or type(normalized_model["files"]) is not int
        or normalized_model["files"] < 1
    ):
        _fail("resident_sft_model_identity_invalid")
    normalized_behavior = _identity_mapping(
        behavior_identity,
        role="resident_sft_behavior",
        digest_field="bundle_sha256",
    )
    normalized_personality = _identity_mapping(
        personality_identity,
        role="resident_sft_personality",
        digest_field="identity_sha256",
    )
    normalized_tokenizer = _identity_mapping(
        tokenizer_identity,
        role="resident_sft_tokenizer",
        digest_field="identity_sha256",
    )
    normalized_spec = _exact(
        execution_spec,
        {"path", "sha256", "size_bytes", "semantic_sha256"},
        role="resident_sft_execution_spec",
    )
    spec_binding = artifact_binding(
        {key: normalized_spec[key] for key in ("path", "sha256", "size_bytes")},
        role="resident_sft_execution_spec",
    )
    spec_binding["semantic_sha256"] = _sha(
        normalized_spec["semantic_sha256"],
        role="resident_sft_execution_spec_semantic",
    )
    runtime = _identity_mapping(
        runtime_identity,
        role="resident_sft_runtime",
        digest_field="identity_sha256",
    )
    trust = _exact(
        trust_policy,
        {"path", "sha256", "size_bytes", "semantic_sha256"},
        role="resident_sft_trust_policy",
    )
    trust_binding = artifact_binding(
        {key: trust[key] for key in ("path", "sha256", "size_bytes")},
        role="resident_sft_trust_policy",
    )
    trust_binding["semantic_sha256"] = _sha(
        trust["semantic_sha256"],
        role="resident_sft_trust_policy_semantic",
    )
    material = {
        "schema": AUTHORITY_SCHEMA,
        "training_authority": TRAINING_AUTHORITY,
        "campaign_id": campaign_id,
        "committed_at": committed_at,
        "expires_at": expires_at,
        "model": {
            "path": _relative_path(model_path, role="resident_sft_model"),
            "base_checkpoint": dict(normalized_model),
            "behavior_bundle": normalized_behavior,
            "personality_bundle": normalized_personality,
            "base_checkpoint_immutable": True,
            "resident_checkpoint_required": True,
        },
        "tokenizer": normalized_tokenizer,
        "execution_spec": spec_binding,
        "dataset": normalized_dataset,
        "dataset_artifacts": normalized_artifacts,
        "sources": normalized_sources,
        "runtime": runtime,
        "trust_policy": trust_binding,
        "artifact_root": _relative_path(artifact_root, role="resident_sft_artifact_root"),
        "trainer": config.to_dict(),
        "checkpoint_contract": {
            "every_committed_step_durable": True,
            "adapter_optimizer_cursor_order_epoch_bound": True,
            "complete_marker_before_latest_pointer": True,
            "resume_exact_identity_only": True,
            "process_rotation_after_durable_progress_only": True,
            "max_consecutive_no_progress_failures": 2,
        },
        "post_training_gate": {
            "fresh_heldout_tasks_required": True,
            "final_edge_lesion_required": True,
            "restoration_required": True,
            "broad_regression_required": True,
            "independent_verification_required": True,
            "grpo_admission_before_gate": False,
        },
        "claims_not_supported": list(CLAIMS_NOT_SUPPORTED),
        "claim_state": {
            "resident_sft_complete": False,
            "causal_gain_proven": False,
            "reasoning_gain_proven": False,
            "frontier_level_proven": False,
            "grpo_admission": False,
            "promotion_allowed": False,
        },
        "required_stage_order": [
            "verify_authority_and_artifacts",
            "run_two_step_resident_canary",
            "verify_canary_checkpoint_resume_and_base_immutability",
            "train_or_exactly_resume_to_max_steps",
            "freeze_and_verify_bootstrap_adapter",
            "fresh_heldout_final_edge_lesion",
            "restoration_control",
            "broad_regression_battery",
            "independent_bootstrap_verdict",
        ],
    }
    return {**material, "authority_sha256": sha256_json(material)}


def validate_authority(
    authority: Mapping[str, Any],
    *,
    expected_authority_sha256: str | None = None,
    observed_model_identity: Mapping[str, Any] | None = None,
    observed_behavior_identity: Mapping[str, Any] | None = None,
    observed_personality_identity: Mapping[str, Any] | None = None,
    observed_tokenizer_identity: Mapping[str, Any] | None = None,
    observed_execution_spec: Mapping[str, Any] | None = None,
    observed_sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "training_authority",
        "campaign_id",
        "committed_at",
        "expires_at",
        "model",
        "tokenizer",
        "execution_spec",
        "dataset",
        "dataset_artifacts",
        "sources",
        "runtime",
        "trust_policy",
        "artifact_root",
        "trainer",
        "checkpoint_contract",
        "post_training_gate",
        "claims_not_supported",
        "claim_state",
        "required_stage_order",
        "authority_sha256",
    }
    record = dict(_exact(authority, expected_keys, role="resident_sft_authority"))
    claimed = _sha(record.pop("authority_sha256"), role="resident_sft_authority")
    if claimed != sha256_json(record):
        _fail("resident_sft_authority_digest_mismatch")
    if expected_authority_sha256 is not None and claimed != _sha(
        expected_authority_sha256,
        role="resident_sft_expected_authority",
    ):
        _fail("resident_sft_authority_identity_mismatch")
    if (
        record.get("schema") != AUTHORITY_SCHEMA
        or record.get("training_authority") != TRAINING_AUTHORITY
        or not str(record.get("campaign_id", "")).startswith(
            "resident-32b-recurrent-sft-bootstrap-cp"
        )
    ):
        _fail("resident_sft_authority_policy_invalid")
    _validate_time_window(record.get("committed_at"), record.get("expires_at"))
    config = ResidentSFTBootstrapConfig.from_dict(record.get("trainer", {}))
    dataset = _validate_dataset(record.get("dataset"))
    artifacts = record.get("dataset_artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"train", "validation"}:
        _fail("resident_sft_dataset_artifacts_invalid")
    for split in ("train", "validation"):
        binding = artifact_binding(artifacts[split], role=f"dataset_{split}")
        if (
            binding["sha256"] != dataset[f"{split}_sha256"]
            or binding["size_bytes"] != dataset[f"{split}_size_bytes"]
        ):
            _fail(f"resident_sft_{split}_artifact_binding_mismatch")
    sources = record.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != REQUIRED_SOURCE_ROLES:
        _fail("resident_sft_source_roles_invalid")
    normalized_sources = {
        role: artifact_binding(sources[role], role=f"source_{role}")
        for role in sorted(REQUIRED_SOURCE_ROLES)
    }
    model = _exact(
        record.get("model"),
        {
            "path",
            "base_checkpoint",
            "behavior_bundle",
            "personality_bundle",
            "base_checkpoint_immutable",
            "resident_checkpoint_required",
        },
        role="resident_sft_model",
    )
    _relative_path(model["path"], role="resident_sft_model")
    if (
        model.get("base_checkpoint_immutable") is not True
        or model.get("resident_checkpoint_required") is not True
    ):
        _fail("resident_sft_model_policy_invalid")
    base_identity = _exact(
        model.get("base_checkpoint"),
        {"fingerprint", "method", "files"},
        role="resident_sft_model_identity",
    )
    _sha(base_identity.get("fingerprint"), role="resident_sft_model")
    if base_identity.get("method") != "sha256" or type(base_identity.get("files")) is not int:
        _fail("resident_sft_model_identity_invalid")
    _identity_mapping(
        model.get("behavior_bundle"),
        role="resident_sft_behavior",
        digest_field="bundle_sha256",
    )
    _identity_mapping(
        model.get("personality_bundle"),
        role="resident_sft_personality",
        digest_field="identity_sha256",
    )
    _identity_mapping(
        record.get("tokenizer"),
        role="resident_sft_tokenizer",
        digest_field="identity_sha256",
    )
    spec = _exact(
        record.get("execution_spec"),
        {"path", "sha256", "size_bytes", "semantic_sha256"},
        role="resident_sft_execution_spec",
    )
    artifact_binding(
        {key: spec[key] for key in ("path", "sha256", "size_bytes")},
        role="resident_sft_execution_spec",
    )
    _sha(spec["semantic_sha256"], role="resident_sft_execution_spec_semantic")
    _identity_mapping(
        record.get("runtime"),
        role="resident_sft_runtime",
        digest_field="identity_sha256",
    )
    trust = _exact(
        record.get("trust_policy"),
        {"path", "sha256", "size_bytes", "semantic_sha256"},
        role="resident_sft_trust_policy",
    )
    artifact_binding(
        {key: trust[key] for key in ("path", "sha256", "size_bytes")},
        role="resident_sft_trust_policy",
    )
    _sha(trust["semantic_sha256"], role="resident_sft_trust_policy_semantic")
    _relative_path(record.get("artifact_root"), role="resident_sft_artifact_root")
    checkpoint_contract = record.get("checkpoint_contract")
    post_training = record.get("post_training_gate")
    claim_state = record.get("claim_state")
    if (
        checkpoint_contract
        != {
            "every_committed_step_durable": True,
            "adapter_optimizer_cursor_order_epoch_bound": True,
            "complete_marker_before_latest_pointer": True,
            "resume_exact_identity_only": True,
            "process_rotation_after_durable_progress_only": True,
            "max_consecutive_no_progress_failures": 2,
        }
        or post_training
        != {
            "fresh_heldout_tasks_required": True,
            "final_edge_lesion_required": True,
            "restoration_required": True,
            "broad_regression_required": True,
            "independent_verification_required": True,
            "grpo_admission_before_gate": False,
        }
        or record.get("claims_not_supported") != list(CLAIMS_NOT_SUPPORTED)
        or claim_state
        != {
            "resident_sft_complete": False,
            "causal_gain_proven": False,
            "reasoning_gain_proven": False,
            "frontier_level_proven": False,
            "grpo_admission": False,
            "promotion_allowed": False,
        }
    ):
        _fail("resident_sft_claim_boundary_invalid")
    if observed_model_identity is not None and dict(observed_model_identity) != dict(
        base_identity
    ):
        _fail("resident_sft_model_binding_drift")
    observed_pairs = (
        (observed_behavior_identity, model["behavior_bundle"], "behavior"),
        (observed_personality_identity, model["personality_bundle"], "personality"),
        (observed_tokenizer_identity, record["tokenizer"], "tokenizer"),
        (observed_execution_spec, record["execution_spec"], "execution_spec"),
        (observed_sources, normalized_sources, "sources"),
    )
    for observed, expected, role in observed_pairs:
        if observed is not None and dict(observed) != dict(expected):
            _fail(f"resident_sft_{role}_binding_drift")
    return {
        **record,
        "authority_sha256": claimed,
        "trainer": config.to_dict(),
        "dataset": dataset,
        "sources": normalized_sources,
    }


def authorize_bound_artifacts(
    authority: Mapping[str, Any],
    *,
    train_payload: bytes,
    validation_payload: bytes,
    source_payloads: Mapping[str, bytes],
    expected_authority_sha256: str,
) -> dict[str, Any]:
    validated = validate_authority(
        authority,
        expected_authority_sha256=expected_authority_sha256,
    )
    for split, payload in (("train", train_payload), ("validation", validation_payload)):
        if not isinstance(payload, bytes) or not payload:
            _fail(f"resident_sft_{split}_artifact_invalid")
        binding = validated["dataset_artifacts"][split]
        if len(payload) != binding["size_bytes"] or sha256_bytes(payload) != binding["sha256"]:
            _fail(f"resident_sft_{split}_artifact_binding_drift")
    if set(source_payloads) != REQUIRED_SOURCE_ROLES:
        _fail("resident_sft_source_payload_roles_invalid")
    for role in sorted(REQUIRED_SOURCE_ROLES):
        payload = source_payloads[role]
        binding = validated["sources"][role]
        if (
            not isinstance(payload, bytes)
            or len(payload) != binding["size_bytes"]
            or sha256_bytes(payload) != binding["sha256"]
        ):
            _fail(f"resident_sft_source_{role}_binding_drift")
    return validated


__all__ = [
    "AUTHORITY_SCHEMA",
    "CLAIMS_NOT_SUPPORTED",
    "DATASET_SCHEMA",
    "OBJECTIVE_NAME",
    "REQUIRED_SOURCE_ROLES",
    "ResidentSFTBootstrapAuthorityError",
    "ResidentSFTBootstrapConfig",
    "TRAINER_CONFIG_SCHEMA",
    "TRAINING_AUTHORITY",
    "artifact_binding",
    "authorize_bound_artifacts",
    "build_authority",
    "build_dataset_commitment",
    "canonical_dataset_payloads",
    "sha256_bytes",
    "sha256_json",
    "validate_authority",
]
