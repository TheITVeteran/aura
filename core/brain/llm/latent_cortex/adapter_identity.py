"""Strict identity contracts for recurrence-native RLC adapters.

This module deliberately does not load a model.  The pure validation path
binds an adapter's bytes and tensor inventory to its base checkpoint, training
receipt, objective, and LoRA topology.  Runtime code may optionally obtain
tensor metadata through :func:`inspect_mlx_tensor_metadata`; MLX is imported
only inside that function.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Never, cast

MANIFEST_SCHEMA = "aura.latent_cortex.recurrence_adapter_manifest.v1"
MANIFEST_SCHEMA_VERSION = 1
TRAINING_RECEIPT_SCHEMA = "aura.recurrence_native_train.v1"
TRAINING_RECEIPT_SCHEMA_VERSION = 1
IDENTITY_RECEIPT_SCHEMA = "aura.latent_cortex.recurrence_adapter_identity_receipt.v1"
IDENTITY_RECEIPT_SCHEMA_VERSION = 1
COMPOSITE_IDENTITY_SCHEMA = "aura.latent_cortex.recurrence_adapter_identity.v1"

MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_TENSORS = 1_000_000
MAX_TENSOR_DIMENSIONS = 16
MAX_DIMENSION_SIZE = 1 << 40
MAX_ARTIFACT_BYTES = 1 << 50
MAX_RANK = 1 << 20
MAX_WRAPPED_PROJECTIONS = 1_000_000

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_TENSOR_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,1023}\Z")
_DTYPE_RE = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
_PROJECTION_RE = re.compile(
    r"model\.layers\.(?:0|[1-9][0-9]*)"
    r"(?:\.[A-Za-z][A-Za-z0-9_]*)+\Z"
)


class AdapterIdentityError(ValueError):
    """Fail-closed adapter identity error with a stable machine code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    error = AdapterIdentityError(code)
    raise error


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        _fail("identity_not_canonical_json")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_exact_keys(value: Any, keys: set[str], *, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        _fail(f"{role}_schema_invalid")
    return cast(Mapping[str, Any], value)


def _require_sha256(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{role}_sha256_invalid")
    return value


def _require_identifier(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        _fail(f"{role}_invalid")
    return value


def _require_int(
    value: Any,
    *,
    role: str,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{role}_invalid")
    return value


def _require_relative_path(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or "\x00" in value
    ):
        _fail(f"{role}_invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail(f"{role}_invalid")
    return value


def _require_dtype(value: Any, *, role: str) -> str:
    if not isinstance(value, str):
        _fail(f"{role}_invalid")
    normalized = value.removeprefix("mlx.core.")
    if value != value.strip() or _DTYPE_RE.fullmatch(normalized) is None:
        _fail(f"{role}_invalid")
    return normalized


def _strict_json_loads(raw: bytes, *, role: str) -> Any:
    if not isinstance(raw, bytes):
        _fail(f"{role}_bytes_required")
    if not raw or len(raw) > MAX_MANIFEST_BYTES:
        _fail(f"{role}_size_invalid")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _fail(f"{role}_not_utf8")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{role}_duplicate_json_key")
            result[key] = value
        return result

    def parse_int(raw_value: str) -> int:
        digits = raw_value.removeprefix("-")
        if not digits or len(digits) > 64:
            _fail(f"{role}_number_invalid")
        return int(raw_value)

    def parse_float(raw_value: str) -> float:
        value = float(raw_value)
        if not math.isfinite(value):
            _fail(f"{role}_number_invalid")
        return value

    def reject_constant(_raw_value: str) -> None:
        _fail(f"{role}_number_invalid")

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_int=parse_int,
            parse_float=parse_float,
            parse_constant=reject_constant,
        )
    except AdapterIdentityError:
        raise
    except (TypeError, ValueError, OverflowError, RecursionError):
        _fail(f"{role}_json_invalid")


@dataclass(frozen=True, slots=True, order=True)
class TensorIdentity:
    key: str
    shape: tuple[int, ...]
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True, slots=True)
class ObjectiveIdentity:
    name: str
    schema_version: int
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class TrainingReceiptBinding:
    path: str
    sha256: str
    size_bytes: int
    schema: str
    schema_version: int
    objective: ObjectiveIdentity

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "schema": self.schema,
            "schema_version": self.schema_version,
            "objective": self.objective.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LoraIdentity:
    rank: int
    targets: tuple[str, ...]
    wrapped_projection_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "targets": list(self.targets),
            "wrapped_projection_count": self.wrapped_projection_count,
        }


@dataclass(frozen=True, slots=True)
class RecurrenceAdapterManifest:
    adapter_id: str
    base_checkpoint_fingerprint: str
    adapter: ArtifactBinding
    training_receipt: TrainingReceiptBinding
    lora: LoraIdentity
    tensors: tuple[TensorIdentity, ...]
    schema: str = MANIFEST_SCHEMA
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "adapter_id": self.adapter_id,
            "base_checkpoint": {
                "method": "sha256",
                "fingerprint": self.base_checkpoint_fingerprint,
            },
            "adapter": self.adapter.to_dict(),
            "training_receipt": self.training_receipt.to_dict(),
            "lora": self.lora.to_dict(),
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }


@dataclass(frozen=True, slots=True)
class AdapterIdentityReceipt:
    manifest_sha256: str
    composite_identity_sha256: str
    adapter_id: str
    base_checkpoint_fingerprint: str
    adapter_sha256: str
    adapter_size_bytes: int
    training_receipt_sha256: str
    training_receipt_schema: str
    training_receipt_schema_version: int
    objective_name: str
    objective_schema_version: int
    objective_source_sha256: str
    objective_source_provenance: str
    rank: int
    targets: tuple[str, ...]
    wrapped_projection_count: int
    tensor_count: int
    tensor_metadata_sha256: str
    schema: str = IDENTITY_RECEIPT_SCHEMA
    schema_version: int = IDENTITY_RECEIPT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "manifest_sha256": self.manifest_sha256,
            "composite_identity_sha256": self.composite_identity_sha256,
            "adapter_id": self.adapter_id,
            "base_checkpoint_fingerprint": self.base_checkpoint_fingerprint,
            "adapter_sha256": self.adapter_sha256,
            "adapter_size_bytes": self.adapter_size_bytes,
            "training_receipt_sha256": self.training_receipt_sha256,
            "training_receipt_schema": self.training_receipt_schema,
            "training_receipt_schema_version": self.training_receipt_schema_version,
            "objective_name": self.objective_name,
            "objective_schema_version": self.objective_schema_version,
            "objective_source_sha256": self.objective_source_sha256,
            "objective_source_provenance": self.objective_source_provenance,
            "rank": self.rank,
            "targets": list(self.targets),
            "wrapped_projection_count": self.wrapped_projection_count,
            "tensor_count": self.tensor_count,
            "tensor_metadata_sha256": self.tensor_metadata_sha256,
        }


def _parse_objective(value: Any, *, role: str) -> ObjectiveIdentity:
    value = _require_exact_keys(
        value,
        {"name", "schema_version", "source_sha256"},
        role=role,
    )
    return ObjectiveIdentity(
        name=_require_identifier(value["name"], role=f"{role}_name"),
        schema_version=_require_int(
            value["schema_version"],
            role=f"{role}_schema_version",
            minimum=1,
            maximum=1_000_000,
        ),
        source_sha256=_require_sha256(value["source_sha256"], role=f"{role}_source"),
    )


def _parse_artifact(value: Any, *, role: str) -> ArtifactBinding:
    value = _require_exact_keys(value, {"path", "sha256", "size_bytes"}, role=role)
    return ArtifactBinding(
        path=_require_relative_path(value["path"], role=f"{role}_path"),
        sha256=_require_sha256(value["sha256"], role=role),
        size_bytes=_require_int(
            value["size_bytes"],
            role=f"{role}_size",
            minimum=1,
            maximum=MAX_ARTIFACT_BYTES,
        ),
    )


def _parse_training_binding(value: Any) -> TrainingReceiptBinding:
    value = _require_exact_keys(
        value,
        {"path", "sha256", "size_bytes", "schema", "schema_version", "objective"},
        role="training_receipt_binding",
    )
    schema = value["schema"]
    version = value["schema_version"]
    if schema != TRAINING_RECEIPT_SCHEMA:
        _fail("training_receipt_schema_unsupported")
    if version != TRAINING_RECEIPT_SCHEMA_VERSION or type(version) is not int:
        _fail("training_receipt_schema_version_unsupported")
    return TrainingReceiptBinding(
        path=_require_relative_path(value["path"], role="training_receipt_path"),
        sha256=_require_sha256(value["sha256"], role="training_receipt"),
        size_bytes=_require_int(
            value["size_bytes"],
            role="training_receipt_size",
            minimum=1,
            maximum=MAX_ARTIFACT_BYTES,
        ),
        schema=schema,
        schema_version=version,
        objective=_parse_objective(value["objective"], role="training_receipt_objective"),
    )


def _parse_targets(value: Any, *, role: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        _fail(f"{role}_invalid")
    targets = tuple(_require_identifier(item, role=f"{role}_item") for item in value)
    if len(set(targets)) != len(targets):
        _fail(f"{role}_duplicate")
    return tuple(sorted(targets))


def _parse_lora(value: Any) -> LoraIdentity:
    value = _require_exact_keys(
        value,
        {"rank", "targets", "wrapped_projection_count"},
        role="lora",
    )
    return LoraIdentity(
        rank=_require_int(value["rank"], role="lora_rank", minimum=1, maximum=MAX_RANK),
        targets=_parse_targets(value["targets"], role="lora_targets"),
        wrapped_projection_count=_require_int(
            value["wrapped_projection_count"],
            role="lora_wrapped_projection_count",
            minimum=1,
            maximum=MAX_WRAPPED_PROJECTIONS,
        ),
    )


def _parse_tensor(value: Any, *, role: str) -> TensorIdentity:
    value = _require_exact_keys(value, {"key", "shape", "dtype"}, role=role)
    key = value["key"]
    if (
        not isinstance(key, str)
        or key != key.strip()
        or _TENSOR_KEY_RE.fullmatch(key) is None
        or ".." in key.split("/")
    ):
        _fail(f"{role}_key_invalid")
    raw_shape = value["shape"]
    if (
        not isinstance(raw_shape, (list, tuple))
        or not raw_shape
        or len(raw_shape) > MAX_TENSOR_DIMENSIONS
    ):
        _fail(f"{role}_shape_invalid")
    shape = tuple(
        _require_int(
            dimension,
            role=f"{role}_shape",
            minimum=1,
            maximum=MAX_DIMENSION_SIZE,
        )
        for dimension in raw_shape
    )
    return TensorIdentity(
        key=key,
        shape=shape,
        dtype=_require_dtype(value["dtype"], role=f"{role}_dtype"),
    )


def normalize_tensor_metadata(value: Any) -> tuple[TensorIdentity, ...]:
    """Validate and canonicalize a complete tensor metadata collection."""

    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        _fail("tensor_metadata_invalid")
    tensors: list[TensorIdentity] = []
    for index, raw_tensor in enumerate(value):
        if index >= MAX_TENSORS:
            _fail("tensor_metadata_too_large")
        tensor = (
            raw_tensor
            if isinstance(raw_tensor, TensorIdentity)
            else _parse_tensor(raw_tensor, role=f"tensor_{index}")
        )
        tensors.append(tensor)
    if not tensors:
        _fail("tensor_metadata_empty")
    if len({tensor.key for tensor in tensors}) != len(tensors):
        _fail("tensor_key_duplicate")
    return tuple(sorted(tensors))


def _validate_tensor_topology(tensors: tuple[TensorIdentity, ...], lora: LoraIdentity) -> None:
    projections: dict[str, dict[str, TensorIdentity]] = {}
    for tensor in tensors:
        if tensor.key.endswith(".lora_a"):
            projection, side = tensor.key.removesuffix(".lora_a"), "a"
        elif tensor.key.endswith(".lora_b"):
            projection, side = tensor.key.removesuffix(".lora_b"), "b"
        else:
            _fail("tensor_key_not_lora_projection")
        if (
            _PROJECTION_RE.fullmatch(projection) is None
            or projection.rsplit(".", 1)[-1] not in lora.targets
        ):
            _fail("tensor_target_mismatch")
        if len(tensor.shape) != 2:
            _fail("tensor_shape_not_matrix")
        sides = projections.setdefault(projection, {})
        if side in sides:
            _fail("tensor_projection_side_duplicate")
        sides[side] = tensor
    for sides in projections.values():
        if set(sides) != {"a", "b"}:
            _fail("tensor_projection_pair_incomplete")
        lora_a = sides["a"]
        lora_b = sides["b"]
        if lora_a.shape[1] != lora.rank or lora_b.shape[0] != lora.rank:
            _fail("tensor_rank_mismatch")
    if len(projections) != lora.wrapped_projection_count:
        _fail("wrapped_projection_count_mismatch")


def parse_manifest(value: bytes | Mapping[str, Any]) -> RecurrenceAdapterManifest:
    """Parse the immutable v1 manifest and reject all schema ambiguity."""

    if isinstance(value, bytes):
        value = _strict_json_loads(value, role="adapter_manifest")
    value = _require_exact_keys(
        value,
        {
            "schema",
            "schema_version",
            "adapter_id",
            "base_checkpoint",
            "adapter",
            "training_receipt",
            "lora",
            "tensors",
        },
        role="adapter_manifest",
    )
    if value["schema"] != MANIFEST_SCHEMA:
        _fail("adapter_manifest_schema_unsupported")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != MANIFEST_SCHEMA_VERSION
    ):
        _fail("adapter_manifest_schema_version_unsupported")
    base = _require_exact_keys(
        value["base_checkpoint"], {"method", "fingerprint"}, role="base_checkpoint"
    )
    if base["method"] != "sha256":
        _fail("base_checkpoint_method_unsupported")
    adapter = _parse_artifact(value["adapter"], role="adapter")
    training = _parse_training_binding(value["training_receipt"])
    if adapter.path == training.path:
        _fail("artifact_paths_not_distinct")
    lora = _parse_lora(value["lora"])
    tensors = normalize_tensor_metadata(value["tensors"])
    _validate_tensor_topology(tensors, lora)
    return RecurrenceAdapterManifest(
        adapter_id=_require_identifier(value["adapter_id"], role="adapter_id"),
        base_checkpoint_fingerprint=_require_sha256(
            base["fingerprint"], role="base_checkpoint_fingerprint"
        ),
        adapter=adapter,
        training_receipt=training,
        lora=lora,
        tensors=tensors,
    )


def _validate_training_receipt(
    raw: bytes,
    *,
    manifest: RecurrenceAdapterManifest,
) -> None:
    binding = manifest.training_receipt
    if len(raw) != binding.size_bytes:
        _fail("training_receipt_size_mismatch")
    if _sha256_bytes(raw) != binding.sha256:
        _fail("training_receipt_sha256_mismatch")
    receipt = _strict_json_loads(raw, role="training_receipt")
    if not isinstance(receipt, Mapping):
        _fail("training_receipt_schema_invalid")
    required = {"schema", "objective_schema", "checkpoint", "lora"}
    if not required.issubset(receipt):
        _fail("training_receipt_schema_invalid")
    if receipt["schema"] != binding.schema:
        _fail("training_receipt_schema_mismatch")
    # Historical v1 encoded its version in ``schema`` and did not record an
    # objective-source digest. Preserve that provenance gap explicitly.
    if binding.schema_version != TRAINING_RECEIPT_SCHEMA_VERSION:
        _fail("training_receipt_schema_version_mismatch")
    if receipt["objective_schema"] != binding.objective.name:
        _fail("training_receipt_objective_mismatch")
    checkpoint = receipt["checkpoint"]
    if not isinstance(checkpoint, Mapping) or not {
        "method",
        "fingerprint",
    }.issubset(checkpoint):
        _fail("training_checkpoint_schema_invalid")
    if checkpoint["method"] != "sha256":
        _fail("training_checkpoint_method_unsupported")
    fingerprint = _require_sha256(checkpoint["fingerprint"], role="training_checkpoint_fingerprint")
    if fingerprint != manifest.base_checkpoint_fingerprint:
        _fail("training_checkpoint_fingerprint_mismatch")
    raw_lora = receipt["lora"]
    if not isinstance(raw_lora, Mapping) or not {
        "rank",
        "targets",
        "wrapped_projections",
    }.issubset(raw_lora):
        _fail("training_receipt_lora_schema_invalid")
    receipt_lora = LoraIdentity(
        rank=_require_int(raw_lora["rank"], role="training_lora_rank", minimum=1, maximum=MAX_RANK),
        targets=_parse_targets(raw_lora["targets"], role="training_lora_targets"),
        wrapped_projection_count=_require_int(
            raw_lora["wrapped_projections"],
            role="training_lora_wrapped_projection_count",
            minimum=1,
            maximum=MAX_WRAPPED_PROJECTIONS,
        ),
    )
    if receipt_lora != manifest.lora:
        _fail("training_receipt_lora_mismatch")


def _identity_material(manifest: RecurrenceAdapterManifest) -> dict[str, Any]:
    return {
        "schema": COMPOSITE_IDENTITY_SCHEMA,
        "manifest_schema": manifest.schema,
        "manifest_schema_version": manifest.schema_version,
        "adapter_id": manifest.adapter_id,
        "base_checkpoint_fingerprint": manifest.base_checkpoint_fingerprint,
        "adapter_sha256": manifest.adapter.sha256,
        "adapter_size_bytes": manifest.adapter.size_bytes,
        "training_receipt_sha256": manifest.training_receipt.sha256,
        "training_receipt_size_bytes": manifest.training_receipt.size_bytes,
        "training_receipt_schema": manifest.training_receipt.schema,
        "training_receipt_schema_version": manifest.training_receipt.schema_version,
        "objective": manifest.training_receipt.objective.to_dict(),
        "lora": manifest.lora.to_dict(),
        "tensors": [tensor.to_dict() for tensor in manifest.tensors],
    }


def manifest_sha256(manifest: RecurrenceAdapterManifest | Mapping[str, Any] | bytes) -> str:
    parsed = (
        manifest if isinstance(manifest, RecurrenceAdapterManifest) else parse_manifest(manifest)
    )
    return _sha256_bytes(_canonical_json_bytes(parsed.to_dict()))


def composite_identity_sha256(
    manifest: RecurrenceAdapterManifest | Mapping[str, Any] | bytes,
) -> str:
    parsed = (
        manifest if isinstance(manifest, RecurrenceAdapterManifest) else parse_manifest(manifest)
    )
    return _sha256_bytes(_canonical_json_bytes(_identity_material(parsed)))


def _receipt_for_manifest(manifest: RecurrenceAdapterManifest) -> AdapterIdentityReceipt:
    tensor_bytes = _canonical_json_bytes([tensor.to_dict() for tensor in manifest.tensors])
    objective = manifest.training_receipt.objective
    return AdapterIdentityReceipt(
        manifest_sha256=manifest_sha256(manifest),
        composite_identity_sha256=composite_identity_sha256(manifest),
        adapter_id=manifest.adapter_id,
        base_checkpoint_fingerprint=manifest.base_checkpoint_fingerprint,
        adapter_sha256=manifest.adapter.sha256,
        adapter_size_bytes=manifest.adapter.size_bytes,
        training_receipt_sha256=manifest.training_receipt.sha256,
        training_receipt_schema=manifest.training_receipt.schema,
        training_receipt_schema_version=manifest.training_receipt.schema_version,
        objective_name=objective.name,
        objective_schema_version=objective.schema_version,
        objective_source_sha256=objective.source_sha256,
        objective_source_provenance="posthoc_manifest_binding",
        rank=manifest.lora.rank,
        targets=manifest.lora.targets,
        wrapped_projection_count=manifest.lora.wrapped_projection_count,
        tensor_count=len(manifest.tensors),
        tensor_metadata_sha256=_sha256_bytes(tensor_bytes),
    )


def validate_adapter_identity(
    manifest: RecurrenceAdapterManifest | Mapping[str, Any] | bytes,
    *,
    actual_base_checkpoint_fingerprint: str,
    adapter_bytes: bytes,
    training_receipt_bytes: bytes,
    tensor_metadata: Iterable[TensorIdentity | Mapping[str, Any]],
) -> AdapterIdentityReceipt:
    """Validate all recurrence-adapter identity material or fail closed."""

    parsed = (
        manifest if isinstance(manifest, RecurrenceAdapterManifest) else parse_manifest(manifest)
    )
    actual_base = _require_sha256(
        actual_base_checkpoint_fingerprint, role="actual_base_checkpoint_fingerprint"
    )
    if actual_base != parsed.base_checkpoint_fingerprint:
        _fail("base_checkpoint_fingerprint_mismatch")
    if not isinstance(adapter_bytes, bytes):
        _fail("adapter_bytes_required")
    if len(adapter_bytes) != parsed.adapter.size_bytes:
        _fail("adapter_size_mismatch")
    if _sha256_bytes(adapter_bytes) != parsed.adapter.sha256:
        _fail("adapter_sha256_mismatch")
    if not isinstance(training_receipt_bytes, bytes):
        _fail("training_receipt_bytes_required")
    _validate_training_receipt(training_receipt_bytes, manifest=parsed)
    actual_tensors = normalize_tensor_metadata(tensor_metadata)
    _validate_tensor_topology(actual_tensors, parsed.lora)
    if actual_tensors != parsed.tensors:
        expected_keys = {tensor.key for tensor in parsed.tensors}
        actual_keys = {tensor.key for tensor in actual_tensors}
        if expected_keys != actual_keys:
            _fail("tensor_key_set_mismatch")
        _fail("tensor_metadata_mismatch")
    return _receipt_for_manifest(parsed)


def parse_identity_receipt(
    value: bytes | Mapping[str, Any],
    *,
    manifest: RecurrenceAdapterManifest | Mapping[str, Any] | bytes | None = None,
) -> AdapterIdentityReceipt:
    """Parse a deterministic v1 validation receipt and optionally bind it."""

    if isinstance(value, bytes):
        value = _strict_json_loads(value, role="adapter_identity_receipt")
    expected_keys = {
        "schema",
        "schema_version",
        "manifest_sha256",
        "composite_identity_sha256",
        "adapter_id",
        "base_checkpoint_fingerprint",
        "adapter_sha256",
        "adapter_size_bytes",
        "training_receipt_sha256",
        "training_receipt_schema",
        "training_receipt_schema_version",
        "objective_name",
        "objective_schema_version",
        "objective_source_sha256",
        "objective_source_provenance",
        "rank",
        "targets",
        "wrapped_projection_count",
        "tensor_count",
        "tensor_metadata_sha256",
    }
    value = _require_exact_keys(value, expected_keys, role="adapter_identity_receipt")
    if value["schema"] != IDENTITY_RECEIPT_SCHEMA:
        _fail("adapter_identity_receipt_schema_unsupported")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != IDENTITY_RECEIPT_SCHEMA_VERSION
    ):
        _fail("adapter_identity_receipt_schema_version_unsupported")
    receipt = AdapterIdentityReceipt(
        manifest_sha256=_require_sha256(value["manifest_sha256"], role="manifest"),
        composite_identity_sha256=_require_sha256(
            value["composite_identity_sha256"], role="composite_identity"
        ),
        adapter_id=_require_identifier(value["adapter_id"], role="adapter_id"),
        base_checkpoint_fingerprint=_require_sha256(
            value["base_checkpoint_fingerprint"], role="base_checkpoint_fingerprint"
        ),
        adapter_sha256=_require_sha256(value["adapter_sha256"], role="adapter"),
        adapter_size_bytes=_require_int(
            value["adapter_size_bytes"],
            role="adapter_size",
            minimum=1,
            maximum=MAX_ARTIFACT_BYTES,
        ),
        training_receipt_sha256=_require_sha256(
            value["training_receipt_sha256"], role="training_receipt"
        ),
        training_receipt_schema=_require_identifier(
            value["training_receipt_schema"], role="training_receipt_schema"
        ),
        training_receipt_schema_version=_require_int(
            value["training_receipt_schema_version"],
            role="training_receipt_schema_version",
            minimum=1,
            maximum=1_000_000,
        ),
        objective_name=_require_identifier(value["objective_name"], role="objective_name"),
        objective_schema_version=_require_int(
            value["objective_schema_version"],
            role="objective_schema_version",
            minimum=1,
            maximum=1_000_000,
        ),
        objective_source_sha256=_require_sha256(
            value["objective_source_sha256"], role="objective_source"
        ),
        objective_source_provenance=_require_identifier(
            value["objective_source_provenance"],
            role="objective_source_provenance",
        ),
        rank=_require_int(value["rank"], role="lora_rank", minimum=1, maximum=MAX_RANK),
        targets=_parse_targets(value["targets"], role="lora_targets"),
        wrapped_projection_count=_require_int(
            value["wrapped_projection_count"],
            role="lora_wrapped_projection_count",
            minimum=1,
            maximum=MAX_WRAPPED_PROJECTIONS,
        ),
        tensor_count=_require_int(
            value["tensor_count"],
            role="tensor_count",
            minimum=1,
            maximum=MAX_TENSORS,
        ),
        tensor_metadata_sha256=_require_sha256(
            value["tensor_metadata_sha256"], role="tensor_metadata"
        ),
    )
    if manifest is not None:
        parsed_manifest = (
            manifest
            if isinstance(manifest, RecurrenceAdapterManifest)
            else parse_manifest(manifest)
        )
        if receipt != _receipt_for_manifest(parsed_manifest):
            _fail("adapter_identity_receipt_manifest_mismatch")
    return receipt


def inspect_mlx_tensor_metadata(adapter_path: str | Path) -> tuple[TensorIdentity, ...]:
    """Inspect an adapter with MLX without making MLX an import-time dependency."""

    path = Path(adapter_path)
    try:
        from mlx import core as mx
    except ImportError:
        _fail("mlx_unavailable")
    try:
        loaded = mx.load(str(path))
    except Exception as exc:  # noqa: BLE001 - typed wrap; original exception chained
        raise AdapterIdentityError("adapter_tensor_inspection_failed") from exc
    if not isinstance(loaded, Mapping):
        _fail("adapter_tensor_container_invalid")
    tensors = cast(Mapping[str, Any], loaded)
    metadata = []
    for key, tensor in tensors.items():
        shape = getattr(tensor, "shape", None)
        dtype = getattr(tensor, "dtype", None)
        if shape is None or dtype is None:
            _fail("adapter_tensor_metadata_missing")
        metadata.append(
            {
                "key": key,
                "shape": list(cast(Iterable[int], shape)),
                "dtype": str(dtype),
            }
        )
    return normalize_tensor_metadata(metadata)


def build_legacy_v1_manifest(
    adapter_dir: str | Path,
    *,
    adapter_id: str,
    actual_base_checkpoint_fingerprint: str,
    objective_source_path: str | Path,
) -> RecurrenceAdapterManifest:
    """Build a strict identity manifest for a historical v1 adapter.

    Artifact bytes, receipt bytes, checkpoint, LoRA topology, and tensor
    inventory are exact. The objective source digest is a post-hoc manifest
    binding because the v1 trainer did not record it; identity receipts expose
    that limitation instead of upgrading the historical evidence by assertion.
    """

    root = Path(adapter_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        _fail("adapter_dir_not_directory")
    adapter_path = root / "adapter_final.safetensors"
    if not adapter_path.is_file():
        adapter_path = root / "adapter_latest.safetensors"
    receipt_path = root / "receipt.json"
    objective_path = Path(objective_source_path).expanduser().resolve(strict=True)
    if not adapter_path.is_file():
        _fail("adapter_artifact_missing")
    if not receipt_path.is_file():
        _fail("training_receipt_missing")
    if not objective_path.is_file():
        _fail("objective_source_missing")

    adapter_bytes = adapter_path.read_bytes()
    receipt_bytes = receipt_path.read_bytes()
    receipt = _strict_json_loads(receipt_bytes, role="training_receipt")
    if not isinstance(receipt, Mapping) or receipt.get("schema") != TRAINING_RECEIPT_SCHEMA:
        _fail("training_receipt_schema_unsupported")
    checkpoint = receipt.get("checkpoint")
    lora = receipt.get("lora")
    if not isinstance(checkpoint, Mapping) or not isinstance(lora, Mapping):
        _fail("training_receipt_schema_invalid")
    actual_base = _require_sha256(
        actual_base_checkpoint_fingerprint,
        role="actual_base_checkpoint_fingerprint",
    )
    if checkpoint.get("method") != "sha256" or checkpoint.get("fingerprint") != actual_base:
        _fail("training_checkpoint_fingerprint_mismatch")
    objective_schema = _require_identifier(
        receipt.get("objective_schema"), role="training_receipt_objective"
    )
    rank = _require_int(lora.get("rank"), role="training_lora_rank", minimum=1, maximum=MAX_RANK)
    targets = _parse_targets(lora.get("targets"), role="training_lora_targets")
    wrapped = _require_int(
        lora.get("wrapped_projections"),
        role="training_lora_wrapped_projection_count",
        minimum=1,
        maximum=MAX_WRAPPED_PROJECTIONS,
    )
    tensors = inspect_mlx_tensor_metadata(adapter_path)
    manifest = parse_manifest(
        {
            "schema": MANIFEST_SCHEMA,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "adapter_id": adapter_id,
            "base_checkpoint": {"method": "sha256", "fingerprint": actual_base},
            "adapter": {
                "path": adapter_path.name,
                "sha256": _sha256_bytes(adapter_bytes),
                "size_bytes": len(adapter_bytes),
            },
            "training_receipt": {
                "path": receipt_path.name,
                "sha256": _sha256_bytes(receipt_bytes),
                "size_bytes": len(receipt_bytes),
                "schema": TRAINING_RECEIPT_SCHEMA,
                "schema_version": TRAINING_RECEIPT_SCHEMA_VERSION,
                "objective": {
                    "name": objective_schema,
                    "schema_version": 1,
                    "source_sha256": _sha256_bytes(objective_path.read_bytes()),
                },
            },
            "lora": {
                "rank": rank,
                "targets": list(targets),
                "wrapped_projection_count": wrapped,
            },
            "tensors": [tensor.to_dict() for tensor in tensors],
        }
    )
    validate_adapter_identity(
        manifest,
        actual_base_checkpoint_fingerprint=actual_base,
        adapter_bytes=adapter_bytes,
        training_receipt_bytes=receipt_bytes,
        tensor_metadata=tensors,
    )
    return manifest
