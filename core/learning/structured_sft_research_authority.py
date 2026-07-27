"""Fail-closed authority for SPARK synthetic recurrent-SFT research.

This contract is intentionally narrower than production training authority.
It admits one exact structured-synthetic candidate, one small local checkpoint,
one tokenizer/runtime projection, and one recurrence-native trainer protocol.
Verified replay, evaluator custody, resident checkpoints, ordinary lexical
adapter activation, and promotion are outside the authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Never

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
    full_weight_checkpoint_identity,
    model_behavior_bundle_identity,
    runtime_environment_identity,
)
from core.learning.external_monotonic_witness import (
    ZERO_SHA256,
    validate_rekor_witness_bundle,
    validate_spark_059_production_audit_packet,
)
from core.learning.structured_sft import (
    STRUCTURED_SFT_CANDIDATE_FILES,
    validate_candidate_dataset_artifacts,
)
from core.runtime.file_read_gateway import read_stable_bytes

AUTHORITY_SCHEMA: Final = "aura.rlc.synthetic_recurrent_sft_authority.v1"
AUTHORITY_VERSION: Final = "2026.07.27.1"
TRAINER_CONFIG_SCHEMA: Final = "aura.rlc.synthetic_recurrent_sft_trainer_config.v1"
MODEL_IDENTITY_SCHEMA: Final = "aura.rlc.small_checkpoint_identity.v1"
TOKENIZATION_BINDING_SCHEMA: Final = (
    "aura.rlc.synthetic_recurrent_sft_tokenization_binding.v1"
)
SOURCE_CLOSURE_SCHEMA: Final = "aura.rlc.synthetic_recurrent_sft_source_closure.v1"
TRAINING_AUTHORITY: Final = "synthetic_small_checkpoint_recurrent_sft_research_only"
TRAINING_MODE: Final = "recurrent_latent_slot_sft"
SAMPLER: Final = "sha256_stateless_epoch_permutation.v1"
MAX_AUTHORITY_TTL_S: Final = 7 * 24 * 60 * 60
MAX_SMALL_MODEL_PARAMETERS: Final = 2_500_000_000
MAX_SMALL_MODEL_WEIGHT_BYTES: Final = 2 * 1024 * 1024 * 1024
MAX_FILE_BYTES: Final = 2 * 1024 * 1024 * 1024
MAX_JSON_BYTES: Final = 256 * 1024 * 1024

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_ROLES: Final = (
    "authority",
    "trainer",
    "containment_launcher",
    "detached_supervisor",
    "checkpoint_state",
    "structured_sft",
    "tokenization",
    "recurrence_objective",
    "execution_spec",
    "recurrence_adapter",
    "resume_verifier",
)
_MODEL_BEHAVIOR_FILES: Final = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_FORBIDDEN_INPUTS: Final = (
    "verified_replay_user_content",
    "evaluation_holdout",
    "production_replay",
    "resident_32b_checkpoint",
    "personality_adapter",
    "unbound_remote_dataset",
)
_REQUIRED_NONCLAIMS: Final = (
    "production_replay_admission",
    "resident_32b_training",
    "reasoning_gain",
    "frontier_performance",
    "production_promotion",
    "wow_signal",
)
_CUSTODY_COMMIT_SCHEMA: Final = "aura.rlc.structured_sft_custody_commit.v1"
_CUSTODY_COMMIT_FIELDS: Final = frozenset(
    {
        "schema",
        "state",
        "generation_id",
        "candidate_directory",
        "evaluator_directory",
        "candidate_package_sha256",
        "evaluator_package_sha256",
        "custody_root_sha256",
        "custody_report_sha256",
        "commit_sha256",
    }
)


class StructuredSFTResearchAuthorityError(ValueError):
    """Stable contract error for synthetic recurrent-SFT research."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    normalized = str(code or "").strip()
    raise StructuredSFTResearchAuthorityError(
        normalized or "structured_sft_research_authority_invalid"
    )


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        _fail("structured_sft_research_noncanonical_value")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _sha(value: Any, *, code: str) -> str:
    if not _is_sha256(value):
        _fail(code)
    return str(value)


def _positive_int(value: Any, *, code: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        _fail(code)
    return value


def _finite_positive(value: Any, *, code: str, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 < float(value) <= maximum
    ):
        _fail(code)
    return float(value)


def strict_json_bytes(raw: bytes, *, role: str) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_JSON_BYTES:
        _fail(f"{role}_json_size_invalid")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{role}_duplicate_json_key")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda _value: _fail(f"{role}_nonfinite_number"),
        )
    except StructuredSFTResearchAuthorityError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError, OverflowError):
        _fail(f"{role}_json_invalid")
    if not isinstance(parsed, dict):
        _fail(f"{role}_json_object_required")
    return parsed


@dataclass(frozen=True, slots=True)
class RecurrentSFTTrainerConfig:
    """Bounded optimizer and recurrent-window adaptation contract."""

    max_steps: int = 60
    batch_size: int = 1
    learning_rate: float = 1e-5
    optimizer: str = "AdamW"
    weight_decay: float = 0.01
    lora_rank: int = 8
    lora_scale: float = 20.0
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj", "o_proj")
    checkpoint_every: int = 5
    evaluate_every: int = 5
    validation_examples: int = 8
    max_seq_length: int = 4096
    max_minutes: float = 180.0
    memory_fraction: float = 0.55
    seed: int = 2026072701

    def __post_init__(self) -> None:
        _positive_int(self.max_steps, code="trainer_max_steps_invalid", maximum=500)
        if self.batch_size != 1:
            _fail("trainer_batch_size_must_be_one_for_exact_resume")
        _finite_positive(
            self.learning_rate,
            code="trainer_learning_rate_invalid",
            maximum=1e-2,
        )
        if self.optimizer != "AdamW":
            _fail("trainer_optimizer_invalid")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(float(self.weight_decay))
            or not 0.0 <= float(self.weight_decay) <= 0.2
        ):
            _fail("trainer_weight_decay_invalid")
        _positive_int(self.lora_rank, code="trainer_lora_rank_invalid", maximum=64)
        _finite_positive(
            self.lora_scale,
            code="trainer_lora_scale_invalid",
            maximum=128.0,
        )
        if self.lora_dropout != 0.0:
            _fail("trainer_lora_dropout_must_be_zero_for_exact_resume")
        allowed_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
        if (
            not self.lora_targets
            or len(set(self.lora_targets)) != len(self.lora_targets)
            or any(target not in allowed_targets for target in self.lora_targets)
        ):
            _fail("trainer_lora_targets_invalid")
        _positive_int(
            self.checkpoint_every,
            code="trainer_checkpoint_interval_invalid",
            maximum=self.max_steps,
        )
        _positive_int(
            self.evaluate_every,
            code="trainer_evaluation_interval_invalid",
            maximum=self.max_steps,
        )
        _positive_int(
            self.validation_examples,
            code="trainer_validation_count_invalid",
            maximum=10_000,
        )
        if not 256 <= self.max_seq_length <= 8192:
            _fail("trainer_max_seq_length_invalid")
        _finite_positive(
            self.max_minutes,
            code="trainer_max_minutes_invalid",
            maximum=24 * 60,
        )
        if not 0.1 <= self.memory_fraction <= 0.8:
            _fail("trainer_memory_fraction_invalid")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            _fail("trainer_seed_invalid")

    def to_dict(self) -> dict[str, Any]:
        material = asdict(self)
        material["lora_targets"] = list(self.lora_targets)
        return {
            "schema": TRAINER_CONFIG_SCHEMA,
            "training_mode": TRAINING_MODE,
            "sampler": SAMPLER,
            "loss": "recurrent_live_path_final_assistant_cross_entropy",
            "adapter_activation": "latent_slot_positions_only",
            "ordinary_lexical_activation": False,
            "validation_scope": "candidate_validation_only_no_evaluator_holdout",
            **material,
        }


def _artifact_binding(name: str, payload: bytes) -> dict[str, Any]:
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not isinstance(payload, bytes)
        or not payload
        or len(payload) > MAX_FILE_BYTES
    ):
        _fail("structured_sft_research_artifact_invalid")
    return {
        "name": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def candidate_identity(
    candidate_artifacts: Mapping[str, bytes],
    custody_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind trainer-visible synthetic bytes without opening evaluator custody."""

    manifest = validate_candidate_dataset_artifacts(candidate_artifacts)
    if (
        not isinstance(custody_attestation, Mapping)
        or custody_attestation.get("state") != "committed"
        or custody_attestation.get("candidate_package_sha256")
        != manifest["package_sha256"]
        or custody_attestation.get("custody_root_sha256")
        != manifest["custody_root_sha256"]
        or not _is_sha256(custody_attestation.get("commit_sha256"))
        or not _is_sha256(custody_attestation.get("evaluator_package_sha256"))
        or not _is_sha256(custody_attestation.get("custody_report_sha256"))
    ):
        _fail("structured_sft_research_candidate_custody_invalid")
    files = [
        _artifact_binding(name, candidate_artifacts[name])
        for name in STRUCTURED_SFT_CANDIDATE_FILES
    ]
    body = {
        "classification": "repository_generated_structured_synthetic",
        "candidate_package_sha256": manifest["package_sha256"],
        "custody_root_sha256": manifest["custody_root_sha256"],
        "custody_commit_sha256": custody_attestation["commit_sha256"],
        "evaluator_package_sha256": custody_attestation[
            "evaluator_package_sha256"
        ],
        "curriculum_sha256": manifest["curriculum_manifest"][
            "curriculum_sha256"
        ],
        "source_closure_sha256": manifest["curriculum_manifest"][
            "source_binding"
        ]["sha256"],
        "files": files,
        "trainer_projection": {
            "candidate_train.jsonl": "in_memory_train",
            "candidate_valid.jsonl": "in_memory_validation",
            "manifest.json": "authority_revalidation_only",
        },
        "candidate_filenames_directly_loadable_by_mlx": False,
        "evaluator_filesystem_accessed": False,
        "holdout_present": False,
        "contains_user_content": False,
        "contains_verified_replay": False,
    }
    return {**body, "identity_sha256": sha256_json(body)}


def tokenization_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a full tokenizer receipt to the fields required by training."""

    tokenizer = report.get("tokenizer") if isinstance(report, Mapping) else None
    runtime = tokenizer.get("runtime") if isinstance(tokenizer, Mapping) else None
    snapshot = (
        tokenizer.get("snapshot_manifest")
        if isinstance(tokenizer, Mapping)
        else None
    )
    custody = (
        report.get("candidate_custody_attestation")
        if isinstance(report, Mapping)
        else None
    )
    contract = (
        report.get("trainer_binding_contract")
        if isinstance(report, Mapping)
        else None
    )
    if (
        not isinstance(tokenizer, Mapping)
        or not isinstance(runtime, Mapping)
        or not isinstance(snapshot, Mapping)
        or not isinstance(custody, Mapping)
        or not isinstance(contract, Mapping)
        or report.get("schema")
        != "aura.rlc.structured_sft_tokenizer_validation_bundle.v3"
        or report.get("status") != "passed_exact_masked_prefix"
        or report.get("rows_with_truncation") != 0
        or report.get("holdout_tokenized") is not False
        or report.get("tokenization_scope") != "candidate_train_validation_only"
        or custody.get("evaluator_filesystem_accessed") is not False
        or contract.get("revalidate_in_trainer_process") is not True
        or contract.get("candidate_only_revalidation") is not True
        or contract.get("evaluator_filesystem_access_required") is not False
        or contract.get("path_substitution_allowed") is not False
    ):
        _fail("structured_sft_research_tokenization_report_invalid")
    body = {
        "schema": TOKENIZATION_BINDING_SCHEMA,
        "validation_bundle_sha256": _sha(
            report.get("validation_bundle_sha256"),
            code="structured_sft_research_tokenization_digest_invalid",
        ),
        "candidate_package_sha256": _sha(
            report.get("candidate_package_sha256"),
            code="structured_sft_research_tokenization_candidate_invalid",
        ),
        "custody_commit_sha256": _sha(
            custody.get("commit_sha256"),
            code="structured_sft_research_tokenization_custody_invalid",
        ),
        "tokenizer_identity_sha256": _sha(
            tokenizer.get("sha256"),
            code="structured_sft_research_tokenizer_identity_invalid",
        ),
        "tokenizer_runtime_identity_sha256": _sha(
            runtime.get("sha256"),
            code="structured_sft_research_tokenizer_runtime_invalid",
        ),
        "snapshot_manifest_sha256": _sha(
            snapshot.get("snapshot_manifest_sha256"),
            code="structured_sft_research_tokenizer_snapshot_invalid",
        ),
        "snapshot_path": str(tokenizer.get("snapshot_path") or ""),
        "rows_checked": _positive_int(
            report.get("rows_checked"),
            code="structured_sft_research_tokenization_rows_invalid",
            maximum=100_000,
        ),
        "max_seq_length": _positive_int(
            report.get("max_seq_length"),
            code="structured_sft_research_tokenization_length_invalid",
            maximum=65_536,
        ),
        "mask_policy": "mlx_chat_dataset_final_assistant_only",
        "revalidate_in_trainer_process": True,
        "evaluator_filesystem_access_required": False,
    }
    if not body["snapshot_path"].startswith("/"):
        _fail("structured_sft_research_tokenizer_snapshot_path_invalid")
    return {**body, "identity_sha256": sha256_json(body)}


def _estimated_dense_parameters(config: Mapping[str, Any]) -> int:
    required = {
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "intermediate_size": config.get("intermediate_size"),
        "vocab_size": config.get("vocab_size"),
    }
    if any(type(value) is not int or value <= 0 for value in required.values()):
        _fail("structured_sft_research_model_config_invalid")
    hidden = required["hidden_size"]
    heads = required["num_attention_heads"]
    kv_heads = required["num_key_value_heads"]
    if hidden % heads != 0 or kv_heads > heads:
        _fail("structured_sft_research_model_attention_shape_invalid")
    head_dim = hidden // heads
    attention = (
        hidden * hidden
        + 2 * hidden * kv_heads * head_dim
        + hidden * hidden
    )
    mlp = 3 * hidden * required["intermediate_size"]
    embeddings = required["vocab_size"] * hidden
    norms_and_bias = 16 * hidden * required["num_hidden_layers"]
    return int(
        embeddings
        + required["num_hidden_layers"] * (attention + mlp + norms_and_bias)
    )


def small_model_identity(model_directory: Path) -> dict[str, Any]:
    """Full-hash a bounded checkpoint and reject resident-scale models."""

    root = model_directory.expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        _fail("structured_sft_research_model_directory_invalid")
    config_raw = read_stable_bytes(root / "config.json", max_bytes=16 * 1024 * 1024)
    config = strict_json_bytes(config_raw, role="small_model_config")
    if (
        config.get("model_type") != "qwen2"
        or config.get("architectures") != ["Qwen2ForCausalLM"]
    ):
        _fail("structured_sft_research_model_architecture_invalid")
    estimated_parameters = _estimated_dense_parameters(config)
    if estimated_parameters > MAX_SMALL_MODEL_PARAMETERS:
        _fail("structured_sft_research_model_not_small_checkpoint")

    weight_paths = sorted(root.glob("*.safetensors"))
    if not weight_paths:
        _fail("structured_sft_research_model_weights_missing")
    weight_files: list[dict[str, Any]] = []
    total_weight_bytes = 0
    for path in weight_paths:
        if path.is_symlink() or not path.is_file():
            _fail("structured_sft_research_model_weight_invalid")
        payload = read_stable_bytes(path, max_bytes=MAX_FILE_BYTES)
        total_weight_bytes += len(payload)
        if total_weight_bytes > MAX_SMALL_MODEL_WEIGHT_BYTES:
            _fail("structured_sft_research_model_weight_budget_exceeded")
        weight_files.append(_artifact_binding(path.name, payload))

    behavior_payloads = {
        name: read_stable_bytes(root / name, max_bytes=MAX_FILE_BYTES)
        for name in _MODEL_BEHAVIOR_FILES
    }
    body = {
        "schema": MODEL_IDENTITY_SCHEMA,
        "directory": str(root),
        "architecture": "Qwen2ForCausalLM",
        "model_type": "qwen2",
        "estimated_dense_parameters": estimated_parameters,
        "parameter_limit": MAX_SMALL_MODEL_PARAMETERS,
        "total_weight_bytes": total_weight_bytes,
        "weight_byte_limit": MAX_SMALL_MODEL_WEIGHT_BYTES,
        "weight_files": weight_files,
        "behavior_files": [
            _artifact_binding(name, behavior_payloads[name])
            for name in _MODEL_BEHAVIOR_FILES
        ],
        "full_weight_identity": full_weight_checkpoint_identity(root),
        "behavior_identity": model_behavior_bundle_identity(root),
        "runtime_identity": runtime_environment_identity(),
        "resident_checkpoint_allowed": False,
    }
    return {**body, "identity_sha256": sha256_json(body)}


def source_closure(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Bind every implementation file that can affect the authorized run."""

    if set(paths) != set(_SOURCE_ROLES):
        _fail("structured_sft_research_source_roles_invalid")
    records: list[dict[str, Any]] = []
    for role in _SOURCE_ROLES:
        path = paths[role].expanduser().resolve(strict=True)
        if path.is_symlink() or not path.is_file():
            _fail("structured_sft_research_source_file_invalid")
        payload = read_stable_bytes(path, max_bytes=MAX_FILE_BYTES)
        records.append(
            {
                "role": role,
                "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    body = {"schema": SOURCE_CLOSURE_SCHEMA, "files": records}
    return {**body, "closure_sha256": sha256_json(body)}


def execution_spec_identity(raw: Mapping[str, Any]) -> dict[str, Any]:
    try:
        spec = RLCExecutionSpec.from_dict(dict(raw))
    except (TypeError, ValueError) as exc:
        raise StructuredSFTResearchAuthorityError(
            "structured_sft_research_execution_spec_invalid"
        ) from exc
    problems = spec.validate()
    if problems:
        _fail("structured_sft_research_execution_spec_invalid")
    if (
        spec.adaptive_halting
        or spec.fast_weights_mode != "disabled"
        or spec.latent_opt_mode != "disabled"
        or spec.recurrent_steps < 2
        or spec.recurrent_steps > 8
    ):
        _fail("structured_sft_research_execution_spec_unbounded")
    body = {
        "schema": "aura.rlc.synthetic_recurrent_sft_execution_binding.v1",
        "spec": spec.to_dict(),
        "semantic_sha256": spec.sha256,
        "training_depth": spec.recurrent_steps,
        "adapter_scope": "latent_slot_positions_only",
    }
    return {**body, "identity_sha256": sha256_json(body)}


def _validate_upstream_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_spark_059_production_audit_packet(packet)
    research = validated.get("research_scope")
    if (
        validated.get("campaign") != "SPARK-059"
        or validated.get("trainer_ready") is not False
        or validated.get("training_authority") != "none"
        or not isinstance(research, Mapping)
        or research.get("small_checkpoint_falsification_may_use")
        != ["structured_synthetic"]
        or research.get("forbidden_from_research_trainer")
        != ["verified_replay_user_content", "evaluation_holdout"]
        or research.get("production_promotion_allowed") is not False
    ):
        _fail("structured_sft_research_upstream_scope_invalid")
    return validated


def upstream_witness_identity(
    *,
    audit_packet: Mapping[str, Any],
    witness_bundle: Mapping[str, Any],
    trusted_log_public_key_pem: bytes,
    expected_sequence: int,
    expected_previous_statement_sha256: str,
    expected_previous_rekor_uuid: str | None,
    minimum_active_shard_log_index: int | None = None,
    minimum_integrated_time: int | None = None,
) -> dict[str, Any]:
    packet = _validate_upstream_packet(audit_packet)
    verification = validate_rekor_witness_bundle(
        witness_bundle,
        audit_packet=packet,
        trusted_log_public_key_pem=trusted_log_public_key_pem,
        expected_sequence=expected_sequence,
        expected_previous_statement_sha256=expected_previous_statement_sha256,
        expected_previous_rekor_uuid=expected_previous_rekor_uuid,
        minimum_integrated_time=minimum_integrated_time,
    )
    try:
        active_shard_log_index = witness_bundle["rekor_entry"]["verification"][
            "inclusionProof"
        ]["logIndex"]
    except (KeyError, TypeError) as exc:
        raise StructuredSFTResearchAuthorityError(
            "structured_sft_research_upstream_active_shard_index_invalid"
        ) from exc
    if (
        type(active_shard_log_index) is not int
        or active_shard_log_index < 0
        or (
            minimum_active_shard_log_index is not None
            and (
                type(minimum_active_shard_log_index) is not int
                or active_shard_log_index < minimum_active_shard_log_index
            )
        )
    ):
        _fail("structured_sft_research_upstream_active_shard_index_rollback")
    body = {
        "audit_packet_sha256": verification["audit_packet_sha256"],
        "bundle_sha256": verification["bundle_sha256"],
        "statement_sha256": verification["statement_sha256"],
        "sequence": verification["sequence"],
        "rekor_uuid": verification["rekor_uuid"],
        "global_log_index": verification["rekor_log_index"],
        "active_shard_log_index": active_shard_log_index,
        "integrated_time": verification["rekor_integrated_time"],
        "trusted_log_key_sha256": verification["trusted_log_key_sha256"],
        "status": verification["status"],
    }
    return {**body, "identity_sha256": sha256_json(body)}


def build_authority(
    *,
    issued_at_unix: int,
    expires_at_unix: int,
    upstream_witness: Mapping[str, Any],
    candidate: Mapping[str, Any],
    tokenization: Mapping[str, Any],
    model: Mapping[str, Any],
    execution_spec: Mapping[str, Any],
    trainer_config: RecurrentSFTTrainerConfig,
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and self-validate a restricted research authority receipt."""

    if (
        type(issued_at_unix) is not int
        or type(expires_at_unix) is not int
        or issued_at_unix <= 0
        or not issued_at_unix < expires_at_unix
        or expires_at_unix - issued_at_unix > MAX_AUTHORITY_TTL_S
    ):
        _fail("structured_sft_research_authority_time_invalid")
    body = {
        "schema": AUTHORITY_SCHEMA,
        "version": AUTHORITY_VERSION,
        "campaign": "SPARK-059",
        "issued_at_unix": issued_at_unix,
        "expires_at_unix": expires_at_unix,
        "training_authority": TRAINING_AUTHORITY,
        "trainer_ready": True,
        "upstream_witness": dict(upstream_witness),
        "candidate": dict(candidate),
        "tokenization": dict(tokenization),
        "model": dict(model),
        "execution_spec": dict(execution_spec),
        "trainer": trainer_config.to_dict(),
        "sources": dict(sources),
        "resumability": {
            "checkpoint_state": "adapter_optimizer_cursor_and_evidence",
            "publication": "immutable_generation_then_atomic_latest_pointer",
            "sample_order": SAMPLER,
            "hidden_rng_state": False,
            "adapter_only_resume_accepted": False,
            "checkpoint_interval_steps": trainer_config.checkpoint_every,
        },
        "isolation": {
            "candidate_only_loader": True,
            "candidate_files_directly_loadable_by_mlx": False,
            "evaluator_path_argument_accepted": False,
            "holdout_access": False,
            "verified_replay_access": False,
            "remote_dataset_access": False,
            "model_lane_required": True,
            "ordinary_lexical_adapter_activation": False,
        },
        "forbidden_inputs": list(_FORBIDDEN_INPUTS),
        "claims_not_supported": list(_REQUIRED_NONCLAIMS),
        "production_promotion_allowed": False,
        "status": "authorized_synthetic_small_checkpoint_recurrent_sft_research_only",
    }
    authority = {**body, "authority_sha256": sha256_json(body)}
    validate_authority(authority, now_unix=issued_at_unix)
    return authority


def validate_authority(
    raw: Any,
    *,
    expected_authority_sha256: str | None = None,
    now_unix: int | None = None,
    allow_expired_resume: bool = False,
) -> dict[str, Any]:
    """Validate an authority without opening candidate, model, or evaluator."""

    fields = {
        "schema",
        "version",
        "campaign",
        "issued_at_unix",
        "expires_at_unix",
        "training_authority",
        "trainer_ready",
        "upstream_witness",
        "candidate",
        "tokenization",
        "model",
        "execution_spec",
        "trainer",
        "sources",
        "resumability",
        "isolation",
        "forbidden_inputs",
        "claims_not_supported",
        "production_promotion_allowed",
        "status",
        "authority_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != fields:
        _fail("structured_sft_research_authority_schema_invalid")
    body = dict(raw)
    authority_sha = body.pop("authority_sha256", None)
    if (
        not _is_sha256(authority_sha)
        or authority_sha != sha256_json(body)
        or (
            expected_authority_sha256 is not None
            and authority_sha != expected_authority_sha256
        )
    ):
        _fail("structured_sft_research_authority_commitment_invalid")
    issued = raw.get("issued_at_unix")
    expires = raw.get("expires_at_unix")
    if (
        type(issued) is not int
        or type(expires) is not int
        or issued <= 0
        or not issued < expires
        or expires - issued > MAX_AUTHORITY_TTL_S
        or (
            now_unix is not None
            and (
                type(now_unix) is not int
                or now_unix < issued
                or (now_unix > expires and not allow_expired_resume)
            )
        )
    ):
        _fail("structured_sft_research_authority_time_invalid")
    upstream = raw.get("upstream_witness")
    candidate = raw.get("candidate")
    tokenization = raw.get("tokenization")
    model = raw.get("model")
    execution = raw.get("execution_spec")
    trainer = raw.get("trainer")
    sources = raw.get("sources")
    if any(
        not isinstance(value, Mapping)
        for value in (
            upstream,
            candidate,
            tokenization,
            model,
            execution,
            trainer,
            sources,
        )
    ):
        _fail("structured_sft_research_authority_component_invalid")
    for value, digest_field, code in (
        (upstream, "identity_sha256", "structured_sft_research_upstream_invalid"),
        (candidate, "identity_sha256", "structured_sft_research_candidate_invalid"),
        (
            tokenization,
            "identity_sha256",
            "structured_sft_research_tokenization_invalid",
        ),
        (model, "identity_sha256", "structured_sft_research_model_invalid"),
        (
            execution,
            "identity_sha256",
            "structured_sft_research_execution_invalid",
        ),
    ):
        committed = dict(value)
        observed = committed.pop(digest_field, None)
        if not _is_sha256(observed) or observed != sha256_json(committed):
            _fail(code)
    source_body = dict(sources)
    source_sha = source_body.pop("closure_sha256", None)
    if (
        sources.get("schema") != SOURCE_CLOSURE_SCHEMA
        or source_sha != sha256_json(source_body)
    ):
        _fail("structured_sft_research_sources_invalid")
    if (
        raw.get("schema") != AUTHORITY_SCHEMA
        or raw.get("version") != AUTHORITY_VERSION
        or raw.get("campaign") != "SPARK-059"
        or raw.get("training_authority") != TRAINING_AUTHORITY
        or raw.get("trainer_ready") is not True
        or upstream.get("status")
        != "externally_witnessed_audit_head_verified_offline"
        or candidate.get("classification")
        != "repository_generated_structured_synthetic"
        or candidate.get("contains_user_content") is not False
        or candidate.get("contains_verified_replay") is not False
        or candidate.get("holdout_present") is not False
        or tokenization.get("candidate_package_sha256")
        != candidate.get("candidate_package_sha256")
        or tokenization.get("custody_commit_sha256")
        != candidate.get("custody_commit_sha256")
        or trainer.get("schema") != TRAINER_CONFIG_SCHEMA
        or trainer.get("training_mode") != TRAINING_MODE
        or trainer.get("sampler") != SAMPLER
        or trainer.get("ordinary_lexical_activation") is not False
        or trainer.get("validation_scope")
        != "candidate_validation_only_no_evaluator_holdout"
        or tokenization.get("max_seq_length") != trainer.get("max_seq_length")
        or model.get("resident_checkpoint_allowed") is not False
        or model.get("estimated_dense_parameters", MAX_SMALL_MODEL_PARAMETERS + 1)
        > MAX_SMALL_MODEL_PARAMETERS
        or model.get("total_weight_bytes", MAX_SMALL_MODEL_WEIGHT_BYTES + 1)
        > MAX_SMALL_MODEL_WEIGHT_BYTES
        or execution.get("adapter_scope") != "latent_slot_positions_only"
        or raw.get("forbidden_inputs") != list(_FORBIDDEN_INPUTS)
        or raw.get("claims_not_supported") != list(_REQUIRED_NONCLAIMS)
        or raw.get("production_promotion_allowed") is not False
        or raw.get("status")
        != "authorized_synthetic_small_checkpoint_recurrent_sft_research_only"
    ):
        _fail("structured_sft_research_authority_scope_invalid")
    isolation = raw.get("isolation")
    resumability = raw.get("resumability")
    if (
        isolation
        != {
            "candidate_only_loader": True,
            "candidate_files_directly_loadable_by_mlx": False,
            "evaluator_path_argument_accepted": False,
            "holdout_access": False,
            "verified_replay_access": False,
            "remote_dataset_access": False,
            "model_lane_required": True,
            "ordinary_lexical_adapter_activation": False,
        }
        or resumability
        != {
            "checkpoint_state": "adapter_optimizer_cursor_and_evidence",
            "publication": "immutable_generation_then_atomic_latest_pointer",
            "sample_order": SAMPLER,
            "hidden_rng_state": False,
            "adapter_only_resume_accepted": False,
            "checkpoint_interval_steps": trainer.get("checkpoint_every"),
        }
    ):
        _fail("structured_sft_research_authority_execution_policy_invalid")
    return json.loads(canonical_json_bytes(raw))


def verify_authority_upstream(
    authority: Mapping[str, Any],
    *,
    audit_packet: Mapping[str, Any],
    witness_bundle: Mapping[str, Any],
    trusted_log_public_key_pem: bytes,
    expected_sequence: int,
    expected_previous_statement_sha256: str = ZERO_SHA256,
    expected_previous_rekor_uuid: str | None = None,
) -> dict[str, Any]:
    """Reverify the public witness and compare it to the authority binding."""

    validated = validate_authority(authority)
    observed = upstream_witness_identity(
        audit_packet=audit_packet,
        witness_bundle=witness_bundle,
        trusted_log_public_key_pem=trusted_log_public_key_pem,
        expected_sequence=expected_sequence,
        expected_previous_statement_sha256=expected_previous_statement_sha256,
        expected_previous_rekor_uuid=expected_previous_rekor_uuid,
    )
    if observed != validated["upstream_witness"]:
        _fail("structured_sft_research_upstream_witness_drift")
    return observed


def authorize_candidate_bytes(
    authority: Mapping[str, Any],
    *,
    candidate_artifacts: Mapping[str, bytes],
    custody_attestation: Mapping[str, Any],
    now_unix: int,
    expected_authority_sha256: str,
    allow_expired_resume: bool = False,
) -> dict[str, bytes]:
    """Return exact train/validation bytes only after authority verification."""

    validated = validate_authority(
        authority,
        expected_authority_sha256=expected_authority_sha256,
        now_unix=now_unix,
        allow_expired_resume=allow_expired_resume,
    )
    identity = candidate_identity(candidate_artifacts, custody_attestation)
    if identity != validated["candidate"]:
        _fail("structured_sft_research_candidate_identity_drift")
    return {
        name: bytes(candidate_artifacts[name])
        for name in STRUCTURED_SFT_CANDIDATE_FILES
    }


def authorize_prevalidated_candidate_bytes(
    authority: Mapping[str, Any],
    *,
    candidate_artifacts: Mapping[str, bytes],
    custody_attestation: Mapping[str, Any],
    candidate_directory_name: str,
    now_unix: int,
    expected_authority_sha256: str,
    allow_expired_resume: bool = False,
) -> dict[str, bytes]:
    """Revalidate authority-bound bytes without replaying executable oracles.

    Authority creation performs the complete deterministic candidate replay.
    A contained trainer cannot repeat that replay because its kernel policy
    denies process creation. This path therefore proves that the trainer sees
    the exact bytes and custody commitment accepted before the authority and
    source closure were frozen.
    """

    validated = validate_authority(
        authority,
        expected_authority_sha256=expected_authority_sha256,
        now_unix=now_unix,
        allow_expired_resume=allow_expired_resume,
    )
    candidate = validated["candidate"]
    if (
        not isinstance(candidate_artifacts, Mapping)
        or set(candidate_artifacts) != set(STRUCTURED_SFT_CANDIDATE_FILES)
        or not isinstance(candidate_directory_name, str)
        or not candidate_directory_name
        or Path(candidate_directory_name).name != candidate_directory_name
    ):
        _fail("structured_sft_research_prevalidated_candidate_invalid")
    observed_files = [
        _artifact_binding(name, candidate_artifacts[name])
        for name in STRUCTURED_SFT_CANDIDATE_FILES
    ]
    if observed_files != candidate.get("files"):
        _fail("structured_sft_research_prevalidated_candidate_file_drift")

    if (
        not isinstance(custody_attestation, Mapping)
        or set(custody_attestation) != _CUSTODY_COMMIT_FIELDS
        or custody_attestation.get("schema") != _CUSTODY_COMMIT_SCHEMA
        or custody_attestation.get("state") != "committed"
        or custody_attestation.get("candidate_directory")
        != candidate_directory_name
        or custody_attestation.get("evaluator_directory")
        == candidate_directory_name
        or not isinstance(custody_attestation.get("evaluator_directory"), str)
        or Path(str(custody_attestation["evaluator_directory"])).name
        != custody_attestation["evaluator_directory"]
        or not isinstance(custody_attestation.get("generation_id"), str)
        or re.fullmatch(
            r"[0-9a-f]{32}",
            str(custody_attestation["generation_id"]),
        )
        is None
    ):
        _fail("structured_sft_research_prevalidated_custody_invalid")
    custody_body = dict(custody_attestation)
    custody_sha256 = custody_body.pop("commit_sha256", None)
    if (
        not _is_sha256(custody_sha256)
        or sha256_json(custody_body) != custody_sha256
        or custody_sha256 != candidate.get("custody_commit_sha256")
        or custody_attestation.get("candidate_package_sha256")
        != candidate.get("candidate_package_sha256")
        or custody_attestation.get("evaluator_package_sha256")
        != candidate.get("evaluator_package_sha256")
        or custody_attestation.get("custody_root_sha256")
        != candidate.get("custody_root_sha256")
        or not _is_sha256(custody_attestation.get("custody_report_sha256"))
    ):
        _fail("structured_sft_research_prevalidated_custody_drift")

    manifest = strict_json_bytes(
        candidate_artifacts["manifest.json"],
        role="prevalidated_candidate_manifest",
    )
    manifest_body = dict(manifest)
    package_sha256 = manifest_body.pop("package_sha256", None)
    curriculum = manifest.get("curriculum_manifest")
    source_binding = (
        curriculum.get("source_binding")
        if isinstance(curriculum, Mapping)
        else None
    )
    train_valid_bindings = {
        row["name"]: {
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
        }
        for row in observed_files
        if row["name"] != "manifest.json"
    }
    if (
        not _is_sha256(package_sha256)
        or sha256_json(manifest_body) != package_sha256
        or package_sha256 != candidate.get("candidate_package_sha256")
        or manifest.get("custody_root_sha256")
        != candidate.get("custody_root_sha256")
        or not isinstance(curriculum, Mapping)
        or curriculum.get("curriculum_sha256")
        != candidate.get("curriculum_sha256")
        or not isinstance(source_binding, Mapping)
        or source_binding.get("sha256")
        != candidate.get("source_closure_sha256")
        or manifest.get("artifacts") != train_valid_bindings
        or manifest.get("candidate_filenames")
        != {
            "train": "candidate_train.jsonl",
            "validation": "candidate_valid.jsonl",
        }
        or manifest.get("validation_scope")
        != "train_validation_replay_only"
        or manifest.get("trainer_ready") is not False
    ):
        _fail("structured_sft_research_prevalidated_manifest_drift")
    return {
        name: bytes(candidate_artifacts[name])
        for name in STRUCTURED_SFT_CANDIDATE_FILES
    }


def deterministic_order(size: int, *, seed: int, epoch: int) -> list[int]:
    """Produce an exact epoch permutation with no hidden PRNG state."""

    if type(size) is not int or size <= 0:
        _fail("structured_sft_research_order_size_invalid")
    if type(seed) is not int or seed < 0 or type(epoch) is not int or epoch < 0:
        _fail("structured_sft_research_order_seed_invalid")
    return sorted(
        range(size),
        key=lambda index: hashlib.sha256(
            f"{seed}:{epoch}:{index}".encode("ascii")
        ).digest(),
    )


def validate_order(
    values: Sequence[int],
    *,
    size: int,
    seed: int,
    epoch: int,
) -> list[int]:
    observed = list(values)
    if observed != deterministic_order(size, seed=seed, epoch=epoch):
        _fail("structured_sft_research_order_drift")
    return observed


__all__ = [
    "AUTHORITY_SCHEMA",
    "AUTHORITY_VERSION",
    "MAX_AUTHORITY_TTL_S",
    "MAX_SMALL_MODEL_PARAMETERS",
    "MAX_SMALL_MODEL_WEIGHT_BYTES",
    "MODEL_IDENTITY_SCHEMA",
    "RecurrentSFTTrainerConfig",
    "SAMPLER",
    "SOURCE_CLOSURE_SCHEMA",
    "StructuredSFTResearchAuthorityError",
    "TOKENIZATION_BINDING_SCHEMA",
    "TRAINER_CONFIG_SCHEMA",
    "TRAINING_AUTHORITY",
    "TRAINING_MODE",
    "authorize_candidate_bytes",
    "authorize_prevalidated_candidate_bytes",
    "build_authority",
    "candidate_identity",
    "canonical_json_bytes",
    "deterministic_order",
    "execution_spec_identity",
    "sha256_json",
    "small_model_identity",
    "source_closure",
    "strict_json_bytes",
    "tokenization_identity",
    "upstream_witness_identity",
    "validate_authority",
    "validate_order",
    "verify_authority_upstream",
]
