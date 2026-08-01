"""Certified transfer of recurrence-native LoRA state into a new policy topology.

The transfer is an initialization aid, not capability evidence. A source
checkpoint may contribute only explicitly named, shape-identical LoRA factors
from the same immutable base model. Unmatched current factors retain their
deterministic initialization, unmatched source factors are reported as dropped,
and a fresh causal preflight remains mandatory after any topology change.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never, cast

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.runtime.file_read_gateway import read_stable_bytes

WARM_START_CONTRACT_SCHEMA = "aura.recurrent_policy_warm_start.v1"
WARM_START_RECEIPT_SCHEMA = "aura.recurrent_policy_warm_start_receipt.v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TENSOR_KEY_RE = re.compile(
    r"model\.layers\.(?P<layer>[0-9]+)\.[^.]+\."
    r"(?P<target>[A-Za-z0-9_]+)\.lora_(?P<factor>[ab])"
)
_MAX_JSON_BYTES = 256 * 1024 * 1024
_MAX_TENSOR_BYTES = 16 * 1024 * 1024 * 1024
_BINDING_KEYS = frozenset({"path", "sha256", "size_bytes"})
_SOURCE_KEYS = frozenset(
    {
        "checkpoint_status",
        "step",
        "max_steps",
        "base_checkpoint",
        "model_behavior_bundle",
        "complete",
        "adapter",
        "training_config",
        "execution_spec",
        "tensor_count",
        "tensor_keys_sha256",
    }
)
_TRANSFER_KEYS = frozenset(
    {
        "copy_targets",
        "initialize_targets",
        "layer_policy",
        "unmatched_source_policy",
        "unmatched_current_policy",
        "exact_shape_dtype_required",
        "causal_preflight_required",
        "claim_eligible",
    }
)
_CONTRACT_KEYS = frozenset(
    {
        "schema",
        "implementation",
        "source_checkpoint",
        "transfer",
        "contract_sha256",
    }
)


class RecurrentPolicyWarmStartError(RuntimeError):
    """Stable fail-closed warm-start construction or validation error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise RecurrentPolicyWarmStartError(code)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _document_sha256(value: Mapping[str, Any]) -> str:
    return _sha256(canonical_json_bytes(dict(value)))


def _clone(value: Any, *, role: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
        _fail(f"warm_start_{role}_not_canonical_json")


def _require_sha256(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"warm_start_{role}_sha256_invalid")
    return value


def _require_repo_file(
    repo_root: Path,
    relative: Any,
    *,
    role: str,
    max_bytes: int,
) -> tuple[Path, bytes]:
    if (
        not isinstance(relative, str)
        or not relative
        or relative.startswith("/")
        or "\\" in relative
    ):
        _fail(f"warm_start_{role}_path_invalid")
    root = repo_root.expanduser().resolve(strict=True)
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RecurrentPolicyWarmStartError(
            f"warm_start_{role}_path_invalid"
        ) from exc
    if candidate.is_symlink() or not resolved.is_file():
        _fail(f"warm_start_{role}_storage_invalid")
    try:
        raw = read_stable_bytes(resolved, max_bytes=max_bytes)
    except (OSError, ValueError) as exc:
        raise RecurrentPolicyWarmStartError(
            f"warm_start_{role}_read_failed"
        ) from exc
    return resolved, raw


def _validate_binding(
    value: Any,
    *,
    repo_root: Path,
    role: str,
    max_bytes: int,
) -> tuple[dict[str, Any], Path, bytes]:
    if not isinstance(value, Mapping) or set(value) != _BINDING_KEYS:
        _fail(f"warm_start_{role}_binding_invalid")
    binding = cast(dict[str, Any], _clone(value, role=f"{role}_binding"))
    size_bytes = binding.get("size_bytes")
    if type(size_bytes) is not int or not 0 < size_bytes <= max_bytes:
        _fail(f"warm_start_{role}_binding_invalid")
    expected_sha = _require_sha256(binding.get("sha256"), role=role)
    path, raw = _require_repo_file(
        repo_root,
        binding.get("path"),
        role=role,
        max_bytes=max_bytes,
    )
    if len(raw) != size_bytes or _sha256(raw) != expected_sha:
        _fail(f"warm_start_{role}_binding_mismatch")
    return binding, path, raw


def _build_binding(
    path: str | Path,
    *,
    repo_root: Path,
    role: str,
    max_bytes: int,
) -> tuple[dict[str, Any], Path, bytes]:
    root = repo_root.expanduser().resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RecurrentPolicyWarmStartError(
            f"warm_start_{role}_path_invalid"
        ) from exc
    if candidate.is_symlink() or not resolved.is_file():
        _fail(f"warm_start_{role}_storage_invalid")
    raw = read_stable_bytes(resolved, max_bytes=max_bytes)
    return (
        {
            "path": relative.as_posix(),
            "sha256": _sha256(raw),
            "size_bytes": len(raw),
        },
        resolved,
        raw,
    )


def _parse_json(raw: bytes, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RecurrentPolicyWarmStartError(
            f"warm_start_{role}_invalid"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"warm_start_{role}_invalid")
    if raw != canonical_json_bytes(value):
        _fail(f"warm_start_{role}_noncanonical")
    return value


def _normalize_targets(
    value: Sequence[str],
    *,
    role: str,
    allow_empty: bool = False,
) -> list[str]:
    if (
        isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, Sequence)
        or (not value and not allow_empty)
        or any(not isinstance(target, str) or not target for target in value)
    ):
        _fail(f"warm_start_{role}_targets_invalid")
    normalized = sorted(set(value))
    if len(normalized) != len(value):
        _fail(f"warm_start_{role}_targets_invalid")
    return normalized


def _tensor_metadata(tensors: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for key in sorted(tensors):
        value = tensors[key]
        match = _TENSOR_KEY_RE.fullmatch(key)
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if (
            match is None
            or not isinstance(shape, Sequence)
            or isinstance(shape, (str, bytes, bytearray))
            or dtype is None
        ):
            _fail("warm_start_tensor_topology_invalid")
        dimensions = [int(dimension) for dimension in shape]
        if not dimensions or any(dimension <= 0 for dimension in dimensions):
            _fail("warm_start_tensor_topology_invalid")
        metadata[key] = {
            "layer": int(match.group("layer")),
            "target": match.group("target"),
            "factor": match.group("factor"),
            "shape": dimensions,
            "dtype": str(dtype),
        }
    if not metadata:
        _fail("warm_start_tensor_topology_invalid")
    return metadata


def _load_source_tensors(
    path: Path,
    raw_before: bytes,
) -> Mapping[str, Any]:
    try:
        import mlx.core as mx

        tensors = mx.load(str(path))
        raw_after = read_stable_bytes(path, max_bytes=_MAX_TENSOR_BYTES)
    except Exception as exc:
        raise RecurrentPolicyWarmStartError(
            "warm_start_adapter_load_failed"
        ) from exc
    if raw_before != raw_after or not isinstance(tensors, Mapping):
        _fail("warm_start_adapter_changed_during_load")
    _tensor_metadata(tensors)
    return tensors


def build_recurrent_warm_start_contract(
    *,
    repo_root: str | Path,
    complete_path: str | Path,
    training_config_path: str | Path,
    execution_spec_path: str | Path,
    copy_targets: Sequence[str],
    initialize_targets: Sequence[str],
) -> dict[str, Any]:
    """Build and fully replay a source-checkpoint transfer contract."""

    root = Path(repo_root)
    complete_binding, complete_file, complete_raw = _build_binding(
        complete_path,
        repo_root=root,
        role="complete",
        max_bytes=_MAX_JSON_BYTES,
    )
    training_binding, _training_file, training_raw = _build_binding(
        training_config_path,
        repo_root=root,
        role="training_config",
        max_bytes=_MAX_JSON_BYTES,
    )
    spec_binding, _spec_file, spec_raw = _build_binding(
        execution_spec_path,
        repo_root=root,
        role="execution_spec",
        max_bytes=_MAX_JSON_BYTES,
    )
    complete = _parse_json(complete_raw, role="complete")
    training = _parse_json(training_raw, role="training_config")
    source_spec = _parse_json(spec_raw, role="execution_spec")
    adapter_record = complete.get("adapter")
    if (
        complete.get("schema") != "aura.recurrence_native_checkpoint.v3"
        or not isinstance(adapter_record, Mapping)
        or not isinstance(adapter_record.get("path"), str)
        or Path(adapter_record["path"]).name != adapter_record["path"]
    ):
        _fail("warm_start_source_checkpoint_invalid")
    adapter_binding, adapter_file, adapter_raw = _build_binding(
        complete_file.parent / adapter_record["path"],
        repo_root=root,
        role="adapter",
        max_bytes=_MAX_TENSOR_BYTES,
    )
    tensors = _load_source_tensors(adapter_file, adapter_raw)
    keys = sorted(tensors)
    step = complete.get("step")
    max_steps = training.get("max_steps")
    if type(step) is not int or type(max_steps) is not int:
        _fail("warm_start_source_checkpoint_invalid")
    checkpoint_status = (
        "complete_checkpoint"
        if step == max_steps
        else "bounded_partial_checkpoint"
    )
    source = {
        "checkpoint_status": checkpoint_status,
        "step": step,
        "max_steps": max_steps,
        "base_checkpoint": training.get("base_checkpoint"),
        "model_behavior_bundle": training.get("model_behavior_bundle"),
        "complete": complete_binding,
        "adapter": adapter_binding,
        "training_config": training_binding,
        "execution_spec": spec_binding,
        "tensor_count": len(keys),
        "tensor_keys_sha256": _sha256(canonical_json_bytes(keys)),
    }
    normalized_copy_targets = _normalize_targets(copy_targets, role="copy")
    normalized_initialize_targets = _normalize_targets(
        initialize_targets,
        role="initialize",
        allow_empty=True,
    )
    transfer = {
        "copy_targets": normalized_copy_targets,
        "initialize_targets": normalized_initialize_targets,
        "layer_policy": "current_layer_intersection",
        "unmatched_source_policy": "drop_and_receipt",
        "unmatched_current_policy": "retain_deterministic_initialization",
        "exact_shape_dtype_required": True,
        "causal_preflight_required": True,
        "claim_eligible": False,
    }
    if complete.get("execution_spec_sha256") != _sha256(
        canonical_json_bytes(source_spec)
    ):
        _fail("warm_start_source_checkpoint_invalid")
    body = {
        "schema": WARM_START_CONTRACT_SCHEMA,
        "implementation": _build_binding(
            Path(__file__),
            repo_root=root,
            role="implementation",
            max_bytes=_MAX_JSON_BYTES,
        )[0],
        "source_checkpoint": source,
        "transfer": transfer,
    }
    return validate_recurrent_warm_start_contract(
        {**body, "contract_sha256": _document_sha256(body)},
        repo_root=root,
    )


def validate_recurrent_warm_start_contract(
    value: Any,
    *,
    repo_root: str | Path,
    expected_base_checkpoint: Mapping[str, Any] | None = None,
    expected_model_behavior_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one self-hashed transfer contract and all bound source bytes."""

    if not isinstance(value, Mapping) or set(value) != _CONTRACT_KEYS:
        _fail("warm_start_contract_schema_invalid")
    contract = cast(dict[str, Any], _clone(value, role="contract"))
    unsigned = dict(contract)
    claimed_sha = unsigned.pop("contract_sha256", None)
    implementation = contract.get("implementation")
    source = contract.get("source_checkpoint")
    transfer = contract.get("transfer")
    if (
        contract.get("schema") != WARM_START_CONTRACT_SCHEMA
        or claimed_sha != _document_sha256(unsigned)
        or not isinstance(source, Mapping)
        or set(source) != _SOURCE_KEYS
        or not isinstance(transfer, Mapping)
        or set(transfer) != _TRANSFER_KEYS
    ):
        _fail("warm_start_contract_invalid")

    _validate_binding(
        implementation,
        repo_root=Path(repo_root),
        role="implementation",
        max_bytes=_MAX_JSON_BYTES,
    )

    copy_targets = transfer.get("copy_targets")
    initialize_targets = transfer.get("initialize_targets")
    if (
        not isinstance(copy_targets, list)
        or not copy_targets
        or copy_targets != sorted(set(copy_targets))
        or any(not isinstance(target, str) or not target for target in copy_targets)
        or not isinstance(initialize_targets, list)
        or initialize_targets != sorted(set(initialize_targets))
        or any(not isinstance(target, str) or not target for target in initialize_targets)
        or set(copy_targets) & set(initialize_targets)
        or transfer.get("layer_policy") != "current_layer_intersection"
        or transfer.get("unmatched_source_policy") != "drop_and_receipt"
        or transfer.get("unmatched_current_policy")
        != "retain_deterministic_initialization"
        or transfer.get("exact_shape_dtype_required") is not True
        or transfer.get("causal_preflight_required") is not True
        or transfer.get("claim_eligible") is not False
    ):
        _fail("warm_start_transfer_policy_invalid")

    root = Path(repo_root)
    complete_binding, complete_path, complete_raw = _validate_binding(
        source.get("complete"),
        repo_root=root,
        role="complete",
        max_bytes=_MAX_JSON_BYTES,
    )
    adapter_binding, adapter_path, adapter_raw = _validate_binding(
        source.get("adapter"),
        repo_root=root,
        role="adapter",
        max_bytes=_MAX_TENSOR_BYTES,
    )
    training_binding, _training_path, training_raw = _validate_binding(
        source.get("training_config"),
        repo_root=root,
        role="training_config",
        max_bytes=_MAX_JSON_BYTES,
    )
    spec_binding, _spec_path, spec_raw = _validate_binding(
        source.get("execution_spec"),
        repo_root=root,
        role="execution_spec",
        max_bytes=_MAX_JSON_BYTES,
    )
    complete = _parse_json(complete_raw, role="complete")
    training = _parse_json(training_raw, role="training_config")
    source_spec = _parse_json(spec_raw, role="execution_spec")
    complete_adapter = complete.get("adapter")
    step = source.get("step")
    max_steps = source.get("max_steps")
    base_checkpoint = source.get("base_checkpoint")
    behavior_bundle = source.get("model_behavior_bundle")
    if (
        source.get("checkpoint_status")
        not in {"bounded_partial_checkpoint", "complete_checkpoint"}
        or type(step) is not int
        or step <= 0
        or type(max_steps) is not int
        or max_steps < step
        or (
            source.get("checkpoint_status") == "bounded_partial_checkpoint"
            and max_steps <= step
        )
        or (
            source.get("checkpoint_status") == "complete_checkpoint"
            and max_steps != step
        )
        or complete.get("schema") != "aura.recurrence_native_checkpoint.v3"
        or complete.get("step") != step
        or complete.get("cursor") != step
        or complete.get("config_sha256") != training_binding["sha256"]
        or complete.get("execution_spec_sha256") != _sha256(
            canonical_json_bytes(source_spec)
        )
        or training.get("max_steps") != max_steps
        or training.get("execution_spec_sha256")
        != complete.get("execution_spec_sha256")
        or training.get("base_checkpoint") != base_checkpoint
        or training.get("model_behavior_bundle") != behavior_bundle
        or not isinstance(complete_adapter, Mapping)
        or complete_adapter.get("path") != adapter_path.name
        or complete_adapter.get("sha256") != adapter_binding["sha256"]
        or complete_adapter.get("size_bytes") != adapter_binding["size_bytes"]
        or complete_path.parent != adapter_path.parent
        or spec_binding["sha256"] != _sha256(spec_raw)
    ):
        _fail("warm_start_source_checkpoint_invalid")
    if not isinstance(base_checkpoint, Mapping) or not isinstance(
        behavior_bundle, Mapping
    ):
        _fail("warm_start_source_identity_invalid")
    if expected_base_checkpoint is not None and dict(base_checkpoint) != dict(
        expected_base_checkpoint
    ):
        _fail("warm_start_base_checkpoint_mismatch")
    if expected_model_behavior_bundle is not None and dict(behavior_bundle) != dict(
        expected_model_behavior_bundle
    ):
        _fail("warm_start_model_behavior_mismatch")

    tensors = _load_source_tensors(adapter_path, adapter_raw)
    keys = sorted(tensors)
    if (
        source.get("tensor_count") != len(keys)
        or source.get("tensor_keys_sha256") != _sha256(canonical_json_bytes(keys))
    ):
        _fail("warm_start_source_tensor_identity_mismatch")
    return contract


def load_recurrent_warm_start_contract(
    path: str | Path,
    *,
    repo_root: str | Path,
    expected_base_checkpoint: Mapping[str, Any] | None = None,
    expected_model_behavior_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate a canonical warm-start contract from stable storage."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    try:
        resolved = source.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RecurrentPolicyWarmStartError(
            "warm_start_contract_path_invalid"
        ) from exc
    if source.is_symlink() or not resolved.is_file():
        _fail("warm_start_contract_storage_invalid")
    raw = read_stable_bytes(resolved, max_bytes=_MAX_JSON_BYTES)
    contract = _parse_json(raw, role="contract")
    if raw != canonical_json_bytes(contract):
        _fail("warm_start_contract_noncanonical")
    return validate_recurrent_warm_start_contract(
        contract,
        repo_root=root,
        expected_base_checkpoint=expected_base_checkpoint,
        expected_model_behavior_bundle=expected_model_behavior_bundle,
    )


def plan_recurrent_warm_start(
    *,
    current_tensors: Mapping[str, Any],
    source_tensors: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Plan the exact current-topology tensor transfer without mutating a model."""

    current_metadata = _tensor_metadata(current_tensors)
    source_metadata = _tensor_metadata(source_tensors)
    transfer = contract.get("transfer")
    if not isinstance(transfer, Mapping):
        _fail("warm_start_transfer_policy_invalid")
    raw_copy_targets = transfer.get("copy_targets")
    raw_initialize_targets = transfer.get("initialize_targets")
    if not isinstance(raw_copy_targets, list) or not isinstance(
        raw_initialize_targets,
        list,
    ):
        _fail("warm_start_transfer_policy_invalid")
    copy_targets = set(_normalize_targets(raw_copy_targets, role="copy"))
    initialize_targets = set(
        _normalize_targets(
            raw_initialize_targets,
            role="initialize",
            allow_empty=True,
        )
    )
    if copy_targets & initialize_targets:
        _fail("warm_start_transfer_policy_invalid")
    copied: dict[str, Any] = {}
    initialized: list[str] = []
    for key, metadata in current_metadata.items():
        target = metadata["target"]
        if target in copy_targets:
            source_value = source_tensors.get(key)
            if source_value is None or source_metadata.get(key) != metadata:
                _fail("warm_start_required_tensor_missing_or_incompatible")
            copied[key] = source_value
        elif target in initialize_targets:
            initialized.append(key)
        else:
            _fail("warm_start_current_target_unclassified")
    dropped = sorted(set(source_metadata) - set(copied))
    if not copied or len(copied) + len(initialized) != len(current_metadata):
        _fail("warm_start_transfer_empty_or_incomplete")
    report = {
        "copied_tensor_keys": sorted(copied),
        "copied_tensor_count": len(copied),
        "initialized_tensor_keys": sorted(initialized),
        "initialized_tensor_count": len(initialized),
        "dropped_source_tensor_keys": dropped,
        "dropped_source_tensor_count": len(dropped),
        "current_tensor_count": len(current_metadata),
        "source_tensor_count": len(source_metadata),
    }
    return copied, report


def apply_recurrent_warm_start(
    model: Any,
    *,
    contract: Mapping[str, Any],
    repo_root: str | Path,
    policy_before_sha256: str,
    policy_after: Any,
) -> dict[str, Any]:
    """Apply a validated transfer and seal its exact topology and policy change."""

    validated = validate_recurrent_warm_start_contract(
        contract,
        repo_root=repo_root,
    )
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        current = dict(tree_flatten(model.trainable_parameters()))
        source_binding = cast(Mapping[str, Any], validated["source_checkpoint"])[
            "adapter"
        ]
        adapter_path, adapter_raw = _require_repo_file(
            Path(repo_root),
            cast(Mapping[str, Any], source_binding).get("path"),
            role="adapter",
            max_bytes=_MAX_TENSOR_BYTES,
        )
        source = _load_source_tensors(adapter_path, adapter_raw)
        copied, transfer_report = plan_recurrent_warm_start(
            current_tensors=current,
            source_tensors=source,
            contract=validated,
        )
        initialized_before = {
            key: current[key]
            for key in transfer_report["initialized_tensor_keys"]
        }
        model.load_weights(list(copied.items()), strict=False)
        after = dict(tree_flatten(model.trainable_parameters()))
        mx.eval(*after.values())
    except RecurrentPolicyWarmStartError:
        raise
    except Exception as exc:
        raise RecurrentPolicyWarmStartError("warm_start_apply_failed") from exc

    for key, expected in copied.items():
        if not bool(mx.array_equal(after[key], expected).item()):
            _fail("warm_start_copied_tensor_mismatch")
    for key, expected in initialized_before.items():
        if not bool(mx.array_equal(after[key], expected).item()):
            _fail("warm_start_initialized_tensor_changed")
    after_sha = policy_after(model)
    _require_sha256(policy_before_sha256, role="policy_before")
    _require_sha256(after_sha, role="policy_after")
    if after_sha == policy_before_sha256:
        _fail("warm_start_policy_unchanged")
    body = {
        "schema": WARM_START_RECEIPT_SCHEMA,
        "contract_sha256": validated["contract_sha256"],
        "source_step": cast(Mapping[str, Any], validated["source_checkpoint"])["step"],
        "policy_before_sha256": policy_before_sha256,
        "policy_after_sha256": after_sha,
        **transfer_report,
        "claim_eligible": False,
        "causal_preflight_required": True,
    }
    return {**body, "receipt_sha256": _document_sha256(body)}


def validate_recurrent_warm_start_receipt(value: Any) -> dict[str, Any]:
    """Validate a sealed transfer receipt independent of model execution."""

    required = {
        "schema",
        "contract_sha256",
        "source_step",
        "policy_before_sha256",
        "policy_after_sha256",
        "copied_tensor_keys",
        "copied_tensor_count",
        "initialized_tensor_keys",
        "initialized_tensor_count",
        "dropped_source_tensor_keys",
        "dropped_source_tensor_count",
        "current_tensor_count",
        "source_tensor_count",
        "claim_eligible",
        "causal_preflight_required",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        _fail("warm_start_receipt_schema_invalid")
    receipt = cast(dict[str, Any], _clone(value, role="receipt"))
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256")
    copied = receipt["copied_tensor_keys"]
    initialized = receipt["initialized_tensor_keys"]
    dropped = receipt["dropped_source_tensor_keys"]
    if (
        receipt["schema"] != WARM_START_RECEIPT_SCHEMA
        or claimed != _document_sha256(unsigned)
        or type(receipt["source_step"]) is not int
        or receipt["source_step"] <= 0
        or any(
            not isinstance(keys, list)
            or keys != sorted(set(keys))
            or any(not isinstance(key, str) or not key for key in keys)
            for keys in (copied, initialized, dropped)
        )
        or receipt["copied_tensor_count"] != len(copied)
        or receipt["initialized_tensor_count"] != len(initialized)
        or receipt["dropped_source_tensor_count"] != len(dropped)
        or receipt["current_tensor_count"] != len(copied) + len(initialized)
        or receipt["source_tensor_count"] != len(copied) + len(dropped)
        or set(copied) & set(initialized)
        or set(copied) & set(dropped)
        or receipt["claim_eligible"] is not False
        or receipt["causal_preflight_required"] is not True
        or receipt["policy_before_sha256"] == receipt["policy_after_sha256"]
    ):
        _fail("warm_start_receipt_invalid")
    for role in (
        "contract_sha256",
        "policy_before_sha256",
        "policy_after_sha256",
        "receipt_sha256",
    ):
        _require_sha256(receipt[role], role=role)
    return receipt


__all__ = [
    "RecurrentPolicyWarmStartError",
    "WARM_START_CONTRACT_SCHEMA",
    "WARM_START_RECEIPT_SCHEMA",
    "apply_recurrent_warm_start",
    "build_recurrent_warm_start_contract",
    "load_recurrent_warm_start_contract",
    "plan_recurrent_warm_start",
    "validate_recurrent_warm_start_contract",
    "validate_recurrent_warm_start_receipt",
]
