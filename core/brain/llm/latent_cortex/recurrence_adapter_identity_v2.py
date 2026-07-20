"""Strict, self-contained identity contract for recurrence-native v2 adapters.

V1 manifests were reconstructed after training and therefore could only make
post-hoc claims about the objective that produced an adapter.  V2 binds the
complete training object at emission time: effective base stack, exact data,
execution graph, source snapshots, receipt, completion state, and tensor
topology.  Validation is model-free and fails closed before any adapter tensor
is installed into a resident model.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Never

from core.brain.llm.latent_cortex.adapter_identity import (
    TensorIdentity,
    normalize_tensor_metadata,
)
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec

MANIFEST_SCHEMA_V2 = "aura.recurrence_adapter_manifest.v2"
TRAINING_SCHEMA_V2 = "aura.recurrence_native_train.v2"
TRAINING_CONFIG_SCHEMA_V2 = "aura.recurrence_native_training_config.v2"
DATASET_SCHEMA_V2 = "aura.recurrence_native_dataset.v2"
COMPLETION_SCHEMA_V1 = "aura.recurrence_native_training_completion.v1"
OBJECTIVE_SCHEMA_V2 = "aura.recurrence_native_objective.v2"
# CP181: the v3 objective (margin hinge + branch diversity + bridge parity
# + held-out validation) ships inside the SAME v2 bundle format. A v3
# bundle declares objective_schema v3 in receipt AND config, and carries
# exactly the additional v3 evidence keys — nothing optional, nothing
# extra. v2 bundles validate byte-for-byte as before.
OBJECTIVE_SCHEMA_V3 = "aura.recurrence_native_objective.v3"
# CP211: v4 (compute-priced depth selection + separation-based width +
# trajectory per-step improvement) ships in the same bundle format.
OBJECTIVE_SCHEMA_V4 = "aura.recurrence_native_objective.v4"
ACCEPTED_OBJECTIVE_SCHEMAS = frozenset(
    {OBJECTIVE_SCHEMA_V2, OBJECTIVE_SCHEMA_V3, OBJECTIVE_SCHEMA_V4}
)
RECEIPT_V3_EXTRA_KEYS = frozenset({"objective_options", "holdout_trail"})
CONFIG_V3_EXTRA_KEYS = frozenset({"objective_options", "bridge", "holdout"})
DATASET_V3_EXTRA_KEYS = frozenset({"holdout_per_cell", "holdout_indices"})
RESUME_MIGRATION_SCHEMA = "aura.recurrence_checkpoint_migration.v2"
RESUME_MIGRATION_KEYS = frozenset(
    {
        "schema",
        "migration_sha256",
        "source_checkpoint",
        "source_step",
        "source_config_sha256",
        "dataset_sha256",
        "execution_spec_sha256",
        "checkpoint_complete_sha256",
        "adapter_sha256",
        "optimizer_sha256",
        "failure_tombstone_sha256",
        "recovery_attempt_count",
        "recovery_attempts_sha256",
        "activation_rematerialization",
        "adjoint_schema",
        "new_trainer_sha256",
    }
)
IDENTITY_RECEIPT_SCHEMA_V2 = "aura.recurrence_adapter_identity_receipt.v2"

SOURCE_ROLES = frozenset(
    {
        "trainer",
        "objective",
        "execution_spec",
        "recurrence_adapter",
        "workspace",
        "recurrence",
        "branches",
        "task_generator",
    }
)
MAX_JSON_BYTES = 256 * 1024 * 1024
MAX_ARTIFACT_BYTES = 1 << 50
MAX_TENSORS = 1_000_000
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PROJECTION_RE = re.compile(
    r"model\.layers\.(?:0|[1-9][0-9]*)"
    r"(?:\.[A-Za-z][A-Za-z0-9_]*)+\Z"
)


class RecurrenceAdapterIdentityV2Error(ValueError):
    """Stable fail-closed identity error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    error = RecurrenceAdapterIdentityV2Error(code)
    raise error


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError, RecursionError):
        _fail("identity_not_canonical_json")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def strict_json_loads(raw: bytes, *, role: str) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_JSON_BYTES:
        _fail(f"{role}_size_invalid")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        _fail(f"{role}_not_ascii")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{role}_duplicate_json_key")
            result[key] = value
        return result

    def parse_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            _fail(f"{role}_number_invalid")
        return parsed

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_float=parse_float,
            parse_constant=lambda _value: _fail(f"{role}_number_invalid"),
        )
    except RecurrenceAdapterIdentityV2Error:
        raise
    except (TypeError, ValueError, OverflowError, RecursionError):
        _fail(f"{role}_json_invalid")
    if not isinstance(value, dict):
        _fail(f"{role}_schema_invalid")
    return value


def _exact(value: Any, keys: set[str], *, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        _fail(f"{role}_schema_invalid")
    return value


def _sha(value: Any, *, role: str, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{role}_sha256_invalid")
    return value


def _integer(value: Any, *, role: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{role}_invalid")
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


def _artifact_binding(
    value: Any,
    *,
    role: str,
    allow_empty: bool = False,
) -> dict[str, Any]:
    value = _exact(value, {"path", "sha256", "size_bytes"}, role=role)
    return {
        "path": _relative_path(value["path"], role=role),
        "sha256": _sha(value["sha256"], role=role),
        "size_bytes": _integer(
            value["size_bytes"],
            role=f"{role}_size",
            minimum=0 if allow_empty else 1,
            maximum=MAX_ARTIFACT_BYTES,
        ),
    }


def _resume_migration_identity(value: Any) -> dict[str, Any]:
    value = _exact(value, set(RESUME_MIGRATION_KEYS), role="resume_migration")
    if value.get("schema") != RESUME_MIGRATION_SCHEMA:
        _fail("resume_migration_schema_invalid")
    source_checkpoint = _relative_path(
        value.get("source_checkpoint"),
        role="resume_migration_source_checkpoint",
    )
    if not source_checkpoint.startswith("checkpoints/remote-") and not source_checkpoint.startswith(
        "checkpoints/step-"
    ):
        _fail("resume_migration_source_checkpoint_invalid")
    normalized = {
        "schema": RESUME_MIGRATION_SCHEMA,
        "migration_sha256": _sha(value.get("migration_sha256"), role="resume_migration"),
        "source_checkpoint": source_checkpoint,
        "source_step": _integer(
            value.get("source_step"),
            role="resume_migration_source_step",
            minimum=1,
            maximum=100_000_000,
        ),
    }
    for key in (
        "source_config_sha256",
        "dataset_sha256",
        "execution_spec_sha256",
        "checkpoint_complete_sha256",
        "adapter_sha256",
        "optimizer_sha256",
        "failure_tombstone_sha256",
        "recovery_attempts_sha256",
        "new_trainer_sha256",
    ):
        normalized[key] = _sha(value.get(key), role=f"resume_migration_{key}")
    normalized["recovery_attempt_count"] = _integer(
        value.get("recovery_attempt_count"),
        role="resume_migration_recovery_attempt_count",
        minimum=0,
        maximum=100_000,
    )
    if value.get("activation_rematerialization") != "exact_discrete_adjoint":
        _fail("resume_migration_rematerialization_invalid")
    normalized["activation_rematerialization"] = "exact_discrete_adjoint"
    if value.get("adjoint_schema") != "aura.recurrence_exact_discrete_adjoint.v1":
        _fail("resume_migration_adjoint_schema_invalid")
    normalized["adjoint_schema"] = "aura.recurrence_exact_discrete_adjoint.v1"
    return normalized


def _migration_certificate_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("checkpoint_migration_schema_invalid")
    material = dict(value)
    claimed = material.pop("migration_sha256", None)
    if (
        value.get("schema") != RESUME_MIGRATION_SCHEMA
        or claimed != sha256_bytes(canonical_json_bytes(material))
    ):
        _fail("checkpoint_migration_digest_invalid")
    source = value.get("source")
    destination = value.get("destination")
    failure = value.get("failure")
    change = value.get("required_execution_change")
    trainer = value.get("new_trainer")
    recovery_attempts = value.get("recovery_attempts")
    if not all(
        isinstance(item, Mapping)
        for item in (source, destination, failure, change, trainer)
    ) or not isinstance(recovery_attempts, list):
        _fail("checkpoint_migration_schema_invalid")
    complete = destination.get("complete")
    adapter = destination.get("adapter")
    optimizer = destination.get("optimizer")
    tombstone = failure.get("tombstone")
    if not all(
        isinstance(item, Mapping)
        for item in (complete, adapter, optimizer, tombstone)
    ):
        _fail("checkpoint_migration_schema_invalid")
    return _resume_migration_identity(
        {
            "schema": value.get("schema"),
            "migration_sha256": claimed,
            "source_checkpoint": source.get("checkpoint"),
            "source_step": source.get("step"),
            "source_config_sha256": source.get("config_sha256"),
            "dataset_sha256": source.get("dataset_sha256"),
            "execution_spec_sha256": source.get("execution_spec_sha256"),
            "checkpoint_complete_sha256": complete.get("sha256"),
            "adapter_sha256": adapter.get("sha256"),
            "optimizer_sha256": optimizer.get("sha256"),
            "failure_tombstone_sha256": tombstone.get("sha256"),
            "recovery_attempt_count": len(recovery_attempts),
            "recovery_attempts_sha256": sha256_bytes(
                canonical_json_bytes(recovery_attempts)
            ),
            "activation_rematerialization": change.get("activation_rematerialization"),
            "adjoint_schema": change.get("adjoint_schema"),
            "new_trainer_sha256": trainer.get("sha256"),
        }
    )


def _checkpoint_identity(value: Any, *, role: str) -> dict[str, Any]:
    value = _exact(value, {"fingerprint", "method", "files"}, role=role)
    if value["method"] != "sha256":
        _fail(f"{role}_method_invalid")
    return {
        "fingerprint": _sha(value["fingerprint"], role=role),
        "method": "sha256",
        "files": _integer(value["files"], role=f"{role}_files", minimum=1, maximum=100_000),
    }


def _personality_identity(value: Any) -> dict[str, Any]:
    value = _exact(
        value,
        {"present", "bundle_sha256", "file_count", "files"},
        role="personality_adapter",
    )
    present = value["present"]
    if type(present) is not bool:
        _fail("personality_adapter_present_invalid")
    files = value["files"]
    if not isinstance(files, list):
        _fail("personality_adapter_files_invalid")
    normalized_files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in files:
        binding = _artifact_binding(
            record,
            role="personality_adapter_file",
            allow_empty=True,
        )
        if binding["path"] in seen:
            _fail("personality_adapter_file_duplicate")
        seen.add(binding["path"])
        normalized_files.append(binding)
    normalized_files.sort(key=lambda record: record["path"])
    expected_count = _integer(
        value["file_count"], role="personality_adapter_file_count", minimum=0, maximum=100_000
    )
    if expected_count != len(normalized_files):
        _fail("personality_adapter_file_count_mismatch")
    bundle_sha256 = _sha(
        value["bundle_sha256"], role="personality_adapter_bundle", allow_empty=not present
    )
    if present:
        if not normalized_files:
            _fail("personality_adapter_files_missing")
        expected_bundle = sha256_bytes(canonical_json_bytes(normalized_files))
        if bundle_sha256 != expected_bundle:
            _fail("personality_adapter_bundle_mismatch")
    elif normalized_files or bundle_sha256:
        _fail("personality_adapter_absent_identity_invalid")
    return {
        "present": present,
        "bundle_sha256": bundle_sha256,
        "file_count": expected_count,
        "files": normalized_files,
    }


def full_weight_checkpoint_identity(model_path: str | Path) -> dict[str, Any]:
    """Full-hash a stable checkpoint; structural hashes are not proof roots."""

    root = Path(model_path).expanduser().resolve(strict=True)
    files = (
        sorted(root.glob("*.safetensors"))
        or sorted(root.glob("*.npz"))
        or sorted(root.glob("*.gguf"))
    )
    if not files:
        _fail("base_checkpoint_files_missing")
    combined = hashlib.sha256()
    for path in files:
        if path.is_symlink():
            _fail("base_checkpoint_symlink_rejected")
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            _fail("base_checkpoint_changed_while_hashing")
        combined.update(f"{path.name}:{digest.hexdigest()};".encode("ascii"))
    return {"fingerprint": combined.hexdigest(), "method": "sha256", "files": len(files)}


def personality_bundle_identity(adapter_path: str | Path | None) -> dict[str, Any]:
    if not adapter_path:
        return {"present": False, "bundle_sha256": "", "file_count": 0, "files": []}
    root = Path(adapter_path).expanduser().resolve(strict=True)
    if not root.is_dir():
        _fail("personality_adapter_not_directory")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            _fail("personality_adapter_symlink_rejected")
        if not path.is_file():
            continue
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            _fail("personality_adapter_changed_while_hashing")
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    if not files:
        _fail("personality_adapter_files_missing")
    return {
        "present": True,
        "bundle_sha256": sha256_bytes(canonical_json_bytes(files)),
        "file_count": len(files),
        "files": files,
    }


def model_behavior_bundle_identity(model_path: str | Path) -> dict[str, Any]:
    """Bind tokenizer/config bytes that determine behavior around the weights."""

    root = Path(model_path).expanduser().resolve(strict=True)
    if not root.is_dir():
        _fail("model_behavior_bundle_not_directory")
    weight_suffixes = {".safetensors", ".npz", ".gguf"}
    files: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_symlink():
            _fail("model_behavior_bundle_symlink_rejected")
        if (
            not path.is_file()
            or path.name == "README.md"
            or path.suffix in weight_suffixes
        ):
            continue
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            _fail("model_behavior_bundle_changed_while_hashing")
        files.append(
            {
                "path": path.name,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if not required.issubset(record["path"] for record in files):
        _fail("model_behavior_bundle_incomplete")
    return {
        "bundle_sha256": sha256_bytes(canonical_json_bytes(files)),
        "file_count": len(files),
        "files": files,
    }


def runtime_environment_identity() -> dict[str, Any]:
    dependencies: dict[str, str] = {}
    for distribution in ("mlx", "mlx-lm", "numpy"):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            _fail(f"runtime_dependency_missing_{distribution}")
    body = {
        "python": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "dependencies": dependencies,
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json_bytes(body))}


def _model_behavior_identity(value: Any) -> dict[str, Any]:
    value = _exact(
        value,
        {"bundle_sha256", "file_count", "files"},
        role="model_behavior_bundle",
    )
    files = value["files"]
    if not isinstance(files, list) or not files:
        _fail("model_behavior_bundle_files_invalid")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in files:
        binding = _artifact_binding(
            record,
            role="model_behavior_bundle_file",
            allow_empty=True,
        )
        if "/" in binding["path"] or binding["path"] in seen:
            _fail("model_behavior_bundle_file_invalid")
        seen.add(binding["path"])
        normalized.append(binding)
    normalized.sort(key=lambda record: record["path"])
    file_count = _integer(
        value["file_count"],
        role="model_behavior_bundle_file_count",
        minimum=1,
        maximum=100_000,
    )
    if file_count != len(normalized):
        _fail("model_behavior_bundle_file_count_mismatch")
    bundle_sha256 = _sha(value["bundle_sha256"], role="model_behavior_bundle")
    if bundle_sha256 != sha256_bytes(canonical_json_bytes(normalized)):
        _fail("model_behavior_bundle_digest_mismatch")
    return {
        "bundle_sha256": bundle_sha256,
        "file_count": len(normalized),
        "files": normalized,
    }


def _runtime_identity(value: Any) -> dict[str, Any]:
    value = _exact(
        value,
        {
            "python",
            "platform_system",
            "platform_release",
            "platform_machine",
            "dependencies",
            "identity_sha256",
        },
        role="training_runtime",
    )
    dependencies = value["dependencies"]
    if (
        not isinstance(dependencies, Mapping)
        or set(dependencies) != {"mlx", "mlx-lm", "numpy"}
        or any(not isinstance(item, str) or not item for item in dependencies.values())
    ):
        _fail("training_runtime_dependencies_invalid")
    body = {
        key: value[key]
        for key in (
            "python",
            "platform_system",
            "platform_release",
            "platform_machine",
            "dependencies",
        )
    }
    if any(
        not isinstance(body[key], str) or not body[key]
        for key in (
            "python",
            "platform_system",
            "platform_release",
            "platform_machine",
        )
    ):
        _fail("training_runtime_value_invalid")
    identity_sha256 = _sha(value["identity_sha256"], role="training_runtime")
    if identity_sha256 != sha256_bytes(canonical_json_bytes(body)):
        _fail("training_runtime_digest_mismatch")
    return {**body, "identity_sha256": identity_sha256}


def _verify_artifact(
    binding: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    *,
    role: str,
) -> bytes:
    path = str(binding["path"])
    payload = artifacts.get(path)
    if not isinstance(payload, bytes):
        _fail(f"{role}_bytes_missing")
    if len(payload) != binding["size_bytes"]:
        _fail(f"{role}_size_mismatch")
    if sha256_bytes(payload) != binding["sha256"]:
        _fail(f"{role}_sha256_mismatch")
    return payload


def _normalize_lora(value: Any) -> dict[str, Any]:
    value = _exact(
        value,
        {"rank", "targets", "wrapped_projections", "projection_paths", "trainable_params"},
        role="lora",
    )
    rank = _integer(value["rank"], role="lora_rank", minimum=1, maximum=1 << 20)
    targets = value["targets"]
    projections = value["projection_paths"]
    if (
        not isinstance(targets, list)
        or not targets
        or len(set(targets)) != len(targets)
        or any(not isinstance(target, str) or not target for target in targets)
    ):
        _fail("lora_targets_invalid")
    if (
        not isinstance(projections, list)
        or not projections
        or len(set(projections)) != len(projections)
        or any(not isinstance(path, str) or _PROJECTION_RE.fullmatch(path) is None for path in projections)
    ):
        _fail("lora_projection_paths_invalid")
    if any(path.rsplit(".", 1)[-1] not in targets for path in projections):
        _fail("lora_projection_target_mismatch")
    wrapped = _integer(
        value["wrapped_projections"],
        role="lora_wrapped_projections",
        minimum=1,
        maximum=MAX_TENSORS // 2,
    )
    if wrapped != len(projections):
        _fail("lora_projection_count_mismatch")
    trainable_params = _integer(
        value["trainable_params"], role="lora_trainable_params", minimum=1, maximum=1 << 60
    )
    return {
        "rank": rank,
        "targets": list(targets),
        "wrapped_projections": wrapped,
        "projection_paths": list(projections),
        "trainable_params": trainable_params,
    }


def _validate_tensor_inventory(
    expected: Any,
    actual: Iterable[TensorIdentity | Mapping[str, Any]],
    *,
    lora: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(expected, list) or len(expected) > MAX_TENSORS:
        _fail("tensor_inventory_invalid")
    try:
        expected_tensors = normalize_tensor_metadata(expected)
        actual_tensors = normalize_tensor_metadata(actual)
    except ValueError as exc:
        raise RecurrenceAdapterIdentityV2Error("tensor_inventory_invalid") from exc
    if expected_tensors != actual_tensors:
        _fail("tensor_metadata_mismatch")
    expected_keys = {
        f"{projection}.{suffix}"
        for projection in lora["projection_paths"]
        for suffix in ("lora_a", "lora_b")
    }
    if {tensor.key for tensor in actual_tensors} != expected_keys:
        _fail("tensor_topology_mismatch")
    by_key = {tensor.key: tensor for tensor in actual_tensors}
    rank = int(lora["rank"])
    trainable_params = 0
    for projection in lora["projection_paths"]:
        left = by_key[f"{projection}.lora_a"]
        right = by_key[f"{projection}.lora_b"]
        if len(left.shape) != 2 or len(right.shape) != 2:
            _fail("tensor_rank_invalid")
        if left.shape[1] != rank or right.shape[0] != rank:
            _fail("tensor_lora_rank_mismatch")
        if left.dtype != right.dtype:
            _fail("tensor_pair_dtype_mismatch")
        trainable_params += math.prod(left.shape) + math.prod(right.shape)
    if trainable_params != lora["trainable_params"]:
        _fail("tensor_trainable_params_mismatch")
    return [tensor.to_dict() for tensor in actual_tensors]


def validate_v2_adapter_identity(
    manifest: Mapping[str, Any] | bytes,
    *,
    adapter_id: str,
    actual_base_checkpoint: Mapping[str, Any],
    actual_model_behavior_bundle: Mapping[str, Any],
    actual_personality_adapter: Mapping[str, Any],
    actual_runtime_environment: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    tensor_metadata: Iterable[TensorIdentity | Mapping[str, Any]],
    allow_bounded_partial: bool = False,
) -> dict[str, Any]:
    """Validate a v2 training bundle.

    Complete ``max_steps`` bundles retain the historical identity material and
    receipt shape exactly.  A protocol verifier may explicitly admit a
    durable ``wall_clock`` bundle for *mechanical evaluation only* by setting
    ``allow_bounded_partial``.  That scope is identity-bound and is never a
    load-eligibility or training-completion claim.
    """

    if type(allow_bounded_partial) is not bool:
        _fail("bounded_partial_policy_invalid")

    manifest_bytes = (
        manifest
        if isinstance(manifest, bytes)
        else canonical_json_bytes(dict(manifest)) + b"\n"
    )
    parsed = strict_json_loads(manifest_bytes, role="manifest")
    expected_manifest_keys = {
        "schema",
        "adapter_id",
        "base_checkpoint",
        "model_behavior_bundle",
        "personality_adapter",
        "training_runtime",
        "adapter",
        "adapter_alias",
        "loader_config",
        "training_receipt",
        "training_config",
        "dataset_manifest",
        "execution_spec",
        "config_sha256",
        "dataset_sha256",
        "execution_spec_sha256",
        "sources",
        "lora",
        "tensors",
    }
    migration_manifest_present = "checkpoint_migration" in parsed
    if migration_manifest_present:
        expected_manifest_keys.add("checkpoint_migration")
    parsed = dict(_exact(parsed, expected_manifest_keys, role="manifest"))
    if parsed["schema"] != MANIFEST_SCHEMA_V2:
        _fail("manifest_schema_unsupported")
    if (
        not isinstance(parsed["adapter_id"], str)
        or _IDENTIFIER_RE.fullmatch(parsed["adapter_id"]) is None
        or parsed["adapter_id"] != adapter_id
    ):
        _fail("adapter_id_mismatch")
    base = _checkpoint_identity(parsed["base_checkpoint"], role="base_checkpoint")
    if base != _checkpoint_identity(actual_base_checkpoint, role="actual_base_checkpoint"):
        _fail("base_checkpoint_mismatch")
    model_behavior = _model_behavior_identity(parsed["model_behavior_bundle"])
    if model_behavior != _model_behavior_identity(actual_model_behavior_bundle):
        _fail("model_behavior_bundle_mismatch")
    personality = _personality_identity(parsed["personality_adapter"])
    if personality != _personality_identity(actual_personality_adapter):
        _fail("personality_adapter_mismatch")
    training_runtime = _runtime_identity(parsed["training_runtime"])
    if training_runtime != _runtime_identity(actual_runtime_environment):
        _fail("training_runtime_mismatch")

    artifact_roles = [
        "adapter",
        "adapter_alias",
        "loader_config",
        "training_receipt",
        "training_config",
        "dataset_manifest",
        "execution_spec",
    ]
    if migration_manifest_present:
        artifact_roles.append("checkpoint_migration")
    bindings = {
        role: _artifact_binding(parsed[role], role=role)
        for role in artifact_roles
    }
    payloads = {
        role: _verify_artifact(binding, artifacts, role=role)
        for role, binding in bindings.items()
    }
    receipt = strict_json_loads(payloads["training_receipt"], role="training_receipt")
    config = strict_json_loads(payloads["training_config"], role="training_config")
    dataset = strict_json_loads(payloads["dataset_manifest"], role="dataset_manifest")
    spec_payload = strict_json_loads(payloads["execution_spec"], role="execution_spec")
    loader_config = strict_json_loads(payloads["loader_config"], role="loader_config")
    declared_objective = receipt.get("objective_schema") if isinstance(receipt, Mapping) else None
    objective_is_v3 = declared_objective in (
        OBJECTIVE_SCHEMA_V3,
        OBJECTIVE_SCHEMA_V4,
    )
    migration_receipt_present = "resume_migration" in receipt
    migration_config_present = "resume_migration" in config
    if not (
        migration_manifest_present
        == migration_receipt_present
        == migration_config_present
    ):
        _fail("resume_migration_cross_binding_mismatch")
    if migration_manifest_present and not objective_is_v3:
        _fail("resume_migration_requires_v3")
    receipt_keys = {
        "schema",
        "objective_schema",
        "objective_source_sha256",
        "trainer_source_sha256",
        "config_sha256",
        "dataset_sha256",
        "execution_spec_sha256",
        "base_checkpoint",
        "model_behavior_bundle",
        "personality_adapter",
        "training_runtime",
        "lora",
        "optimizer",
        "gradient_execution",
        "steps",
        "epoch",
        "cursor",
        "elapsed_training_s",
        "invocation_count",
        "halt_reason",
        "complete",
        "final_checkpoint",
        "loss_trail",
    }
    config_keys = {
        "schema",
        "model_path",
        "base_checkpoint",
        "model_behavior_bundle",
        "personality_adapter_path",
        "personality_adapter",
        "training_runtime",
        "execution_spec",
        "execution_spec_sha256",
        "dataset_sha256",
        "objective_schema",
        "curriculum_depths",
        "monotonicity_weight",
        "lora",
        "optimizer",
        "gradient_execution",
        "train_seed",
        "max_steps",
        "sources",
    }
    dataset_keys = {
        "schema",
        "generator",
        "train_seed",
        "families",
        "task_depths",
        "per_cell",
        "examples",
    }
    if objective_is_v3:
        # v3 evidence is REQUIRED with v3, forbidden with v2 — never optional.
        receipt_keys |= RECEIPT_V3_EXTRA_KEYS
        config_keys |= CONFIG_V3_EXTRA_KEYS
        dataset_keys |= DATASET_V3_EXTRA_KEYS
    if migration_manifest_present:
        receipt_keys.add("resume_migration")
        config_keys.add("resume_migration")
    _exact(receipt, receipt_keys, role="training_receipt")
    _exact(config, config_keys, role="training_config")
    _exact(dataset, dataset_keys, role="dataset_manifest")
    if (
        receipt.get("schema") != TRAINING_SCHEMA_V2
        or declared_objective not in ACCEPTED_OBJECTIVE_SCHEMAS
    ):
        _fail("training_receipt_schema_invalid")
    if config.get("schema") != TRAINING_CONFIG_SCHEMA_V2:
        _fail("training_config_schema_invalid")
    if dataset.get("schema") != DATASET_SCHEMA_V2:
        _fail("dataset_schema_invalid")
    if config.get("objective_schema") != declared_objective:
        _fail("training_config_objective_invalid")
    try:
        spec = RLCExecutionSpec.from_dict(spec_payload)
    except (TypeError, ValueError) as exc:
        raise RecurrenceAdapterIdentityV2Error("execution_spec_invalid") from exc

    lora = _normalize_lora(parsed["lora"])
    _exact(
        loader_config,
        {
            "schema",
            "fine_tune_type",
            "loader",
            "model",
            "num_layers",
            "wrapped_projection_count",
            "lora_parameters",
            "execution_spec_sha256",
        },
        role="loader_config",
    )
    loader_lora = _exact(
        loader_config.get("lora_parameters"),
        {"rank", "scale", "dropout", "keys"},
        role="loader_lora",
    )
    unique_layers = {
        int(path.split(".")[2]) for path in lora["projection_paths"]
    }
    if (
        loader_config.get("schema") != "aura.recurrence_scoped_lora_config.v1"
        or loader_config.get("fine_tune_type") != "recurrence_scoped_lora"
        or loader_config.get("loader") != "aura_custom_loader_required"
        or loader_config.get("wrapped_projection_count") != lora["wrapped_projections"]
        or loader_config.get("num_layers") != len(unique_layers)
        or loader_config.get("model") != config.get("model_path")
        or loader_config.get("execution_spec_sha256") != spec.sha256
        or loader_lora.get("rank") != lora["rank"]
        or loader_lora.get("keys") != lora["targets"]
    ):
        _fail("loader_config_cross_binding_mismatch")
    if payloads["adapter_alias"] != payloads["adapter"]:
        _fail("adapter_alias_mismatch")
    if receipt.get("lora") != lora or config.get("lora") != {
        "rank": lora["rank"],
        "targets": lora["targets"],
        "wrapped_projections": lora["projection_paths"],
    }:
        _fail("lora_cross_binding_mismatch")
    curriculum = config.get("curriculum_depths")
    if (
        not isinstance(curriculum, list)
        or len(curriculum) < 2
        or any(type(depth) is not int or depth < 1 for depth in curriculum)
        or sorted(set(curriculum)) != curriculum
        or curriculum[-1] != spec.recurrent_steps
    ):
        _fail("curriculum_depths_invalid")
    monotonicity_weight = config.get("monotonicity_weight")
    if (
        isinstance(monotonicity_weight, bool)
        or not isinstance(monotonicity_weight, (int, float))
        or not math.isfinite(float(monotonicity_weight))
        or not 0.0 <= float(monotonicity_weight) <= 10.0
    ):
        _fail("monotonicity_weight_invalid")
    optimizer = _exact(
        config.get("optimizer"),
        {"name", "learning_rate", "weight_decay"},
        role="optimizer",
    )
    if receipt.get("optimizer") != optimizer or optimizer.get("name") != "AdamW":
        _fail("optimizer_cross_binding_mismatch")
    raw_gradient_execution = config.get("gradient_execution")
    if not isinstance(raw_gradient_execution, Mapping):
        _fail("gradient_execution_invalid")
    schema = raw_gradient_execution.get("schema")
    expected_gradient_execution: dict[str, Any] = {
        "schema": schema,
        "mode": "depth_serial_exact_sum",
        "concurrent_depth_graphs": 1,
        "optimizer_updates_per_sample": 1,
        "finite_loss_and_gradient_required_before_update": True,
    }
    if schema == "aura.recurrence_streamed_depth_gradient.v6":
        expected_gradient_execution.update(
            {
                "activation_rematerialization": "exact_discrete_adjoint",
                "adjoint_schema": "aura.recurrence_exact_discrete_adjoint.v1",
                "boundary_state_storage": "materialized_stop_gradient",
                "terminal_branch_graphs_concurrent": 1,
                "recurrent_transition_graphs_concurrent": 1,
            }
        )
    elif schema == "aura.recurrence_streamed_depth_gradient.v5":
        expected_gradient_execution.update(
            {
                "activation_rematerialization": "transformer_layer_group_checkpoint",
                "layer_group_size": 4,
                "recurrent_transition_checkpointing": True,
            }
        )
    elif schema == "aura.recurrence_streamed_depth_gradient.v4":
        expected_gradient_execution.update(
            {
                "activation_rematerialization": "transformer_layer_group_checkpoint",
                "layer_group_size": 4,
            }
        )
    elif schema == "aura.recurrence_streamed_depth_gradient.v3":
        expected_gradient_execution["activation_rematerialization"] = (
            "per_transformer_layer_checkpoint"
        )
    elif schema == "aura.recurrence_streamed_depth_gradient.v2":
        expected_gradient_execution["activation_rematerialization"] = (
            "full_depth_graph_checkpoint"
        )
    elif schema != "aura.recurrence_streamed_depth_gradient.v1":
        _fail("gradient_execution_cross_binding_mismatch")
    gradient_execution = dict(
        _exact(
            raw_gradient_execution,
            set(expected_gradient_execution),
            role="gradient_execution",
        )
    )
    if (
        gradient_execution != expected_gradient_execution
        or receipt.get("gradient_execution") != gradient_execution
    ):
        _fail("gradient_execution_cross_binding_mismatch")
    resume_migration: dict[str, Any] | None = None
    if migration_manifest_present:
        if schema != "aura.recurrence_streamed_depth_gradient.v6":
            _fail("resume_migration_rematerialization_invalid")
        resume_migration = _resume_migration_identity(config.get("resume_migration"))
        if receipt.get("resume_migration") != resume_migration:
            _fail("resume_migration_cross_binding_mismatch")
        migration_certificate = strict_json_loads(
            payloads["checkpoint_migration"],
            role="checkpoint_migration",
        )
        if _migration_certificate_summary(migration_certificate) != resume_migration:
            _fail("resume_migration_certificate_mismatch")
        if (
            resume_migration["dataset_sha256"] != config.get("dataset_sha256")
            or resume_migration["execution_spec_sha256"]
            != config.get("execution_spec_sha256")
            or resume_migration["new_trainer_sha256"]
            != receipt.get("trainer_source_sha256")
        ):
            _fail("resume_migration_training_identity_mismatch")
    if (
        dataset.get("train_seed") != config.get("train_seed")
        or not isinstance(dataset.get("examples"), list)
        or not dataset["examples"]
        or not isinstance(dataset.get("families"), list)
        or not dataset["families"]
        or not isinstance(dataset.get("task_depths"), list)
        or not dataset["task_depths"]
        or type(dataset.get("per_cell")) is not int
        or dataset["per_cell"] < 1
    ):
        _fail("dataset_training_config_mismatch")
    if receipt.get("base_checkpoint") != base or config.get("base_checkpoint") != base:
        _fail("base_checkpoint_cross_binding_mismatch")
    if (
        receipt.get("model_behavior_bundle") != model_behavior
        or config.get("model_behavior_bundle") != model_behavior
    ):
        _fail("model_behavior_bundle_cross_binding_mismatch")
    if receipt.get("personality_adapter") != personality or config.get("personality_adapter") != personality:
        _fail("personality_adapter_cross_binding_mismatch")
    if (
        receipt.get("training_runtime") != training_runtime
        or config.get("training_runtime") != training_runtime
    ):
        _fail("training_runtime_cross_binding_mismatch")
    if config.get("execution_spec") != spec.to_dict():
        _fail("execution_spec_config_mismatch")

    config_sha256 = sha256_bytes(payloads["training_config"])
    dataset_sha256 = sha256_bytes(payloads["dataset_manifest"])
    spec_sha256 = spec.sha256
    for record, role, expected in (
        (parsed, "config_sha256", config_sha256),
        (receipt, "config_sha256", config_sha256),
        (parsed, "dataset_sha256", dataset_sha256),
        (receipt, "dataset_sha256", dataset_sha256),
        (config, "dataset_sha256", dataset_sha256),
        (parsed, "execution_spec_sha256", spec_sha256),
        (receipt, "execution_spec_sha256", spec_sha256),
        (config, "execution_spec_sha256", spec_sha256),
    ):
        if record.get(role) != expected:
            _fail(f"{role}_cross_binding_mismatch")
    max_steps = config.get("max_steps")
    steps = receipt.get("steps")
    if type(max_steps) is not int or max_steps < 1:
        _fail("training_max_steps_invalid")
    if receipt.get("complete") is True:
        if receipt.get("halt_reason") != "max_steps":
            _fail("training_completion_state_invalid")
        if type(steps) is not int or steps != max_steps:
            _fail("training_step_completion_mismatch")
        training_scope = "complete_training"
    elif allow_bounded_partial and receipt.get("complete") is False:
        if receipt.get("halt_reason") != "wall_clock":
            _fail("bounded_partial_halt_reason_invalid")
        if type(steps) is not int or not 1 <= steps < max_steps:
            _fail("bounded_partial_step_invalid")
        training_scope = "bounded_partial_training"
    else:
        _fail("training_incomplete")

    raw_sources = _exact(parsed["sources"], set(SOURCE_ROLES), role="sources")
    config_sources = _exact(config.get("sources"), set(SOURCE_ROLES), role="config_sources")
    normalized_sources: dict[str, dict[str, Any]] = {}
    for role in sorted(SOURCE_ROLES):
        source = _exact(
            raw_sources[role],
            {"origin_path", "snapshot_path", "sha256", "size_bytes"},
            role=f"source_{role}",
        )
        snapshot_binding = _artifact_binding(
            {
                "path": source["snapshot_path"],
                "sha256": source["sha256"],
                "size_bytes": source["size_bytes"],
            },
            role=f"source_{role}",
        )
        origin_path = _relative_path(source["origin_path"], role=f"source_{role}_origin")
        config_source = _exact(
            config_sources[role], {"path", "sha256", "size_bytes"}, role=f"config_source_{role}"
        )
        if (
            config_source["path"] != origin_path
            or config_source["sha256"] != snapshot_binding["sha256"]
            or config_source["size_bytes"] != snapshot_binding["size_bytes"]
        ):
            _fail(f"source_{role}_config_mismatch")
        _verify_artifact(snapshot_binding, artifacts, role=f"source_{role}")
        normalized_sources[role] = {
            "origin_path": origin_path,
            "snapshot_path": snapshot_binding["path"],
            "sha256": snapshot_binding["sha256"],
            "size_bytes": snapshot_binding["size_bytes"],
        }
    if receipt.get("objective_source_sha256") != normalized_sources["objective"]["sha256"]:
        _fail("objective_source_receipt_mismatch")
    if receipt.get("trainer_source_sha256") != normalized_sources["trainer"]["sha256"]:
        _fail("trainer_source_receipt_mismatch")
    if dataset.get("generator") != config_sources["task_generator"]:
        _fail("task_generator_dataset_mismatch")

    tensors = _validate_tensor_inventory(parsed["tensors"], tensor_metadata, lora=lora)
    manifest_sha256 = sha256_bytes(manifest_bytes)
    completion_raw = artifacts.get("training_completion.json")
    if not isinstance(completion_raw, bytes):
        _fail("training_completion_bytes_missing")
    completion = strict_json_loads(completion_raw, role="training_completion")
    completion = dict(
        _exact(
            completion,
            {"schema", "complete", "halt_reason", "step", "adapter_sha256", "receipt_sha256", "manifest_sha256"},
            role="training_completion",
        )
    )
    if (
        completion["schema"] != COMPLETION_SCHEMA_V1
        or completion["complete"] is not receipt["complete"]
        or completion["halt_reason"] != receipt["halt_reason"]
        or completion["step"] != steps
        or completion["adapter_sha256"] != bindings["adapter"]["sha256"]
        or completion["receipt_sha256"] != bindings["training_receipt"]["sha256"]
        or completion["manifest_sha256"] != manifest_sha256
    ):
        _fail("training_completion_mismatch")

    identity_material = {
        "schema": "aura.recurrence_adapter_identity.v2",
        "adapter_id": adapter_id,
        "base_checkpoint": base,
        "model_behavior_bundle": model_behavior,
        "personality_adapter": personality,
        "training_runtime": training_runtime,
        "adapter": bindings["adapter"],
        "adapter_alias": bindings["adapter_alias"],
        "loader_config": bindings["loader_config"],
        "training_receipt": bindings["training_receipt"],
        "training_config": bindings["training_config"],
        "dataset_manifest": bindings["dataset_manifest"],
        "execution_spec": bindings["execution_spec"],
        "training_completion_sha256": sha256_bytes(completion_raw),
        "sources": normalized_sources,
        "lora": lora,
        "gradient_execution": gradient_execution,
        "tensors": tensors,
    }
    if resume_migration is not None:
        identity_material["checkpoint_migration"] = bindings["checkpoint_migration"]
        identity_material["resume_migration"] = resume_migration
    if training_scope == "bounded_partial_training":
        identity_material["training_state"] = {
            "scope": training_scope,
            "complete": False,
            "halt_reason": "wall_clock",
            "steps": steps,
            "max_steps": max_steps,
        }
    result = {
        "schema": IDENTITY_RECEIPT_SCHEMA_V2,
        "manifest_sha256": manifest_sha256,
        "composite_identity_sha256": sha256_bytes(canonical_json_bytes(identity_material)),
        "adapter_id": adapter_id,
        "base_checkpoint_fingerprint": base["fingerprint"],
        "model_behavior_bundle_sha256": model_behavior["bundle_sha256"],
        "personality_adapter_bundle_sha256": personality["bundle_sha256"],
        "training_runtime_identity_sha256": training_runtime["identity_sha256"],
        "adapter_sha256": bindings["adapter"]["sha256"],
        "training_receipt_sha256": bindings["training_receipt"]["sha256"],
        "training_config_sha256": bindings["training_config"]["sha256"],
        "dataset_sha256": bindings["dataset_manifest"]["sha256"],
        "execution_spec_sha256": bindings["execution_spec"]["sha256"],
        "training_completion_sha256": sha256_bytes(completion_raw),
        "objective_name": declared_objective,
        "objective_source_sha256": normalized_sources["objective"]["sha256"],
        "objective_source_provenance": "training_time_archived_source",
        "rank": lora["rank"],
        "targets": lora["targets"],
        "gradient_execution": gradient_execution,
        "wrapped_projection_count": lora["wrapped_projections"],
        "tensor_count": len(tensors),
        "tensor_metadata_sha256": sha256_bytes(canonical_json_bytes(tensors)),
        "complete": True,
    }
    if resume_migration is not None:
        result["resume_migration"] = resume_migration
        result["checkpoint_migration_sha256"] = bindings["checkpoint_migration"]["sha256"]
    if training_scope == "bounded_partial_training":
        result.update(
            {
                "complete": False,
                "training_scope": training_scope,
                "training_halt_reason": "wall_clock",
                "training_steps": steps,
                "training_max_steps": max_steps,
                "load_eligible": False,
            }
        )
    return result


__all__ = [
    "COMPLETION_SCHEMA_V1",
    "DATASET_SCHEMA_V2",
    "IDENTITY_RECEIPT_SCHEMA_V2",
    "MANIFEST_SCHEMA_V2",
    "ACCEPTED_OBJECTIVE_SCHEMAS",
    "OBJECTIVE_SCHEMA_V2",
    "OBJECTIVE_SCHEMA_V3",
    "OBJECTIVE_SCHEMA_V4",
    "SOURCE_ROLES",
    "TRAINING_CONFIG_SCHEMA_V2",
    "TRAINING_SCHEMA_V2",
    "RecurrenceAdapterIdentityV2Error",
    "canonical_json_bytes",
    "full_weight_checkpoint_identity",
    "model_behavior_bundle_identity",
    "personality_bundle_identity",
    "runtime_environment_identity",
    "sha256_bytes",
    "strict_json_loads",
    "validate_v2_adapter_identity",
]
