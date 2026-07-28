"""Independent exact replay of one verified recurrent policy transition.

This module is the arithmetic kernel used by the external campaign verifier. It
does not trust a producer's claim that an objective or optimizer update ran. A
caller supplies an independently reconstructed model and objective callback;
the kernel restores the sealed pre-state, recomputes the objective and every
gradient tensor, reconstructs the frozen Adam optimizer, and requires exact
post-state identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Never, cast

from core.learning.recurrent_grpo import (
    build_recurrent_policy_optimizer,
    recurrent_policy_optimizer_config,
    recurrent_policy_tensor_map_sha256,
)

POLICY_STATE_REPLAY_RECEIPT_SCHEMA = "aura.verified_transition.policy_state_replay_receipt.v1"
POLICY_STATE_REPLAY_CONTRACT_SCHEMA = "aura.verified_transition.policy_state_replay_contract.v1"
TENSOR_MAP_IDENTITY_SCHEMA = "aura.tensor_map_identity.v1"

_REPLAY_CONTRACT_KEYS = frozenset(
    {
        "schema",
        "preregistration_contract_sha256",
        "initial_policy_sha256",
        "model",
        "execution_spec",
        "source_bindings",
        "source_bindings_sha256",
        "initial_policy_state_custody",
        "initial_policy_state_custody_sha256",
        "optimizer_config",
        "optimizer_config_sha256",
        "recurrent_grpo_config",
        "recurrent_grpo_config_sha256",
        "verified_trajectory_config_json",
        "verified_trajectory_config_sha256",
        "external_verifier_max_seconds",
        "contract_sha256",
    }
)
_MODEL_CONTRACT_KEYS = frozenset(
    {
        "path",
        "base_checkpoint",
        "base_checkpoint_sha256",
        "behavior_bundle",
        "behavior_bundle_sha256",
    }
)
_EXECUTION_SPEC_CONTRACT_KEYS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "semantic_sha256",
        "document_json",
    }
)
_SOURCE_BINDING_KEYS = frozenset({"path", "sha256", "size_bytes"})

_TENSOR_IDENTITY_KEYS = frozenset(
    {
        "schema",
        "role",
        "tensor_count",
        "tensor_keys_sha256",
        "tensors",
        "tensor_root_sha256",
    }
)
_TENSOR_ENTRY_KEYS = frozenset({"name", "dtype", "shape", "value_sha256"})
_REPLAY_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "execution_spec_sha256",
        "policy_before_sha256",
        "policy_after_sha256",
        "objective_receipt_sha256",
        "optimizer_config",
        "optimizer_config_sha256",
        "pre_adapter_identity",
        "pre_optimizer_identity",
        "gradient_identity",
        "post_adapter_identity",
        "post_optimizer_identity",
        "optimizer_update_count",
        "objective_recomputed",
        "all_gradient_tensors_recomputed",
        "adapter_post_state_exact",
        "optimizer_post_state_exact",
        "external_policy_state_replayed",
        "receipt_sha256",
    }
)


class VerifiedTransitionPolicyStateReplayError(RuntimeError):
    """The independently replayed transition differs from producer evidence."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise VerifiedTransitionPolicyStateReplayError(code)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (
        TypeError,
        ValueError,
        UnicodeError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            "policy_state_replay_document_not_canonical"
        ) from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"policy_state_replay_{role}_invalid")
    return value


def _identifier(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or not value[0].isalnum()
        or any(not (character.isalnum() or character in "._:/;=+-") for character in value)
    ):
        _fail(f"policy_state_replay_{role}_invalid")
    return value


def _clone(value: Any, *, role: str) -> Any:
    try:
        return json.loads(_canonical_json_bytes(value))
    except (
        TypeError,
        ValueError,
        UnicodeError,
        OverflowError,
        RecursionError,
    ):
        _fail(f"policy_state_replay_{role}_not_canonical")


def _canonical_path(
    value: Any,
    *,
    role: str,
    directory: bool,
    verify_files: bool,
) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"policy_state_replay_{role}_path_invalid")
    path = Path(value)
    if not path.is_absolute() or path.resolve(strict=False) != path:
        _fail(f"policy_state_replay_{role}_path_invalid")
    if path.is_symlink():
        _fail(f"policy_state_replay_{role}_symlink_rejected")
    if verify_files:
        try:
            metadata = path.stat()
        except OSError as exc:
            raise VerifiedTransitionPolicyStateReplayError(
                f"policy_state_replay_{role}_unavailable"
            ) from exc
        expected = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected(metadata.st_mode):
            _fail(f"policy_state_replay_{role}_type_invalid")
    return path


def _stable_file_bytes(path: Path, *, maximum: int, role: str) -> bytes:
    from core.runtime.file_read_gateway import read_stable_bytes

    try:
        return read_stable_bytes(path, max_bytes=maximum)
    except OSError as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            f"policy_state_replay_{role}_unreadable"
        ) from exc


def _validate_file_binding(
    value: Any,
    *,
    role: str,
    verify_files: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SOURCE_BINDING_KEYS:
        _fail(f"policy_state_replay_{role}_binding_schema_invalid")
    binding = cast(dict[str, Any], _clone(value, role=f"{role}_binding"))
    path = _canonical_path(
        binding.get("path"),
        role=role,
        directory=False,
        verify_files=verify_files,
    )
    digest = _sha256(binding.get("sha256"), role=f"{role}_binding")
    size = binding.get("size_bytes")
    if type(size) is not int or size <= 0 or size > (1 << 40):
        _fail(f"policy_state_replay_{role}_binding_size_invalid")
    if verify_files:
        try:
            if path.stat().st_size != size:
                _fail(f"policy_state_replay_{role}_binding_mismatch")
        except OSError as exc:
            raise VerifiedTransitionPolicyStateReplayError(
                f"policy_state_replay_{role}_unreadable"
            ) from exc
        payload = _stable_file_bytes(path, maximum=size, role=role)
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            _fail(f"policy_state_replay_{role}_binding_mismatch")
    return binding


def _validate_checkpoint_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "fingerprint",
        "method",
        "files",
    }:
        _fail("policy_state_replay_model_base_checkpoint_invalid")
    identity = cast(dict[str, Any], _clone(value, role="base_checkpoint"))
    _sha256(identity.get("fingerprint"), role="base_checkpoint")
    if (
        identity.get("method") != "sha256"
        or type(identity.get("files")) is not int
        or not 1 <= identity["files"] <= 100_000
    ):
        _fail("policy_state_replay_model_base_checkpoint_invalid")
    return identity


def _validate_behavior_bundle_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "bundle_sha256",
        "file_count",
        "files",
    }:
        _fail("policy_state_replay_model_behavior_bundle_invalid")
    identity = cast(dict[str, Any], _clone(value, role="behavior_bundle"))
    files = identity.get("files")
    if (
        not isinstance(files, list)
        or type(identity.get("file_count")) is not int
        or identity["file_count"] != len(files)
        or identity["file_count"] < 3
    ):
        _fail("policy_state_replay_model_behavior_bundle_invalid")
    normalized = []
    for record in files:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256", "size_bytes"}
            or not isinstance(record.get("path"), str)
            or not record["path"]
            or Path(record["path"]).name != record["path"]
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] < 0
        ):
            _fail("policy_state_replay_model_behavior_bundle_invalid")
        _sha256(record.get("sha256"), role="behavior_bundle_file")
        normalized.append(dict(record))
    if (
        normalized != sorted(normalized, key=lambda item: item["path"])
        or len({item["path"] for item in normalized}) != len(normalized)
        or not {"config.json", "tokenizer.json", "tokenizer_config.json"}.issubset(
            item["path"] for item in normalized
        )
        or _sha256(identity.get("bundle_sha256"), role="behavior_bundle") != _digest(normalized)
    ):
        _fail("policy_state_replay_model_behavior_bundle_invalid")
    return identity


def _verify_private_custody_artifact(
    *,
    path: Any,
    artifact: Any,
    role: str,
) -> None:
    if not isinstance(artifact, Mapping):
        _fail(f"policy_state_replay_{role}_artifact_invalid")
    resolved = _canonical_path(
        path,
        role=role,
        directory=False,
        verify_files=True,
    )
    try:
        metadata = resolved.stat()
    except OSError as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            f"policy_state_replay_{role}_unavailable"
        ) from exc
    if (
        metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or metadata.st_nlink != 1
        or resolved.name != artifact.get("path")
        or type(artifact.get("size_bytes")) is not int
        or artifact["size_bytes"] <= 0
    ):
        _fail(f"policy_state_replay_{role}_custody_invalid")
    if metadata.st_size != artifact["size_bytes"]:
        _fail(f"policy_state_replay_{role}_artifact_mismatch")
    payload = _stable_file_bytes(
        resolved,
        maximum=artifact["size_bytes"],
        role=role,
    )
    if len(payload) != artifact["size_bytes"] or hashlib.sha256(payload).hexdigest() != _sha256(
        artifact.get("sha256"), role=f"{role}_artifact"
    ):
        _fail(f"policy_state_replay_{role}_artifact_mismatch")


def validate_policy_state_replay_contract(
    value: Any,
    *,
    verify_files: bool = False,
    verify_model: bool = False,
) -> dict[str, Any]:
    """Validate the complete source and state contract used by external replay."""

    if not isinstance(value, Mapping) or set(value) != _REPLAY_CONTRACT_KEYS:
        _fail("policy_state_replay_contract_schema_invalid")
    contract = cast(dict[str, Any], _clone(value, role="contract"))
    if contract.get("schema") != POLICY_STATE_REPLAY_CONTRACT_SCHEMA:
        _fail("policy_state_replay_contract_schema_invalid")
    if verify_model and not verify_files:
        _fail("policy_state_replay_model_verification_requires_files")
    unsigned = dict(contract)
    observed = _sha256(
        unsigned.pop("contract_sha256"),
        role="contract",
    )
    if observed != _digest(unsigned):
        _fail("policy_state_replay_contract_digest_mismatch")
    _sha256(
        contract.get("preregistration_contract_sha256"),
        role="preregistration_contract",
    )
    initial_policy = _sha256(
        contract.get("initial_policy_sha256"),
        role="initial_policy",
    )

    model = contract.get("model")
    if not isinstance(model, Mapping) or set(model) != _MODEL_CONTRACT_KEYS:
        _fail("policy_state_replay_model_schema_invalid")
    model_path = _canonical_path(
        model.get("path"),
        role="model",
        directory=True,
        verify_files=verify_files,
    )
    identities = {
        "base_checkpoint": _validate_checkpoint_identity(model.get("base_checkpoint")),
        "behavior_bundle": _validate_behavior_bundle_identity(model.get("behavior_bundle")),
    }
    for field, identity in identities.items():
        expected_digest = _sha256(
            model.get(f"{field}_sha256"),
            role=f"model_{field}",
        )
        if expected_digest != _digest(identity):
            _fail(f"policy_state_replay_model_{field}_digest_mismatch")
    if verify_model:
        from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
            full_weight_checkpoint_identity,
            model_behavior_bundle_identity,
        )

        if (
            full_weight_checkpoint_identity(model_path) != identities["base_checkpoint"]
            or model_behavior_bundle_identity(model_path) != identities["behavior_bundle"]
        ):
            _fail("policy_state_replay_model_identity_mismatch")

    execution = contract.get("execution_spec")
    if not isinstance(execution, Mapping) or set(execution) != _EXECUTION_SPEC_CONTRACT_KEYS:
        _fail("policy_state_replay_execution_spec_schema_invalid")
    execution_binding = _validate_file_binding(
        {
            "path": execution.get("path"),
            "sha256": execution.get("sha256"),
            "size_bytes": execution.get("size_bytes"),
        },
        role="execution_spec",
        verify_files=verify_files,
    )
    execution_semantic = _sha256(
        execution.get("semantic_sha256"),
        role="execution_spec_semantic",
    )
    execution_document_json = execution.get("document_json")
    if not isinstance(execution_document_json, str) or not execution_document_json:
        _fail("policy_state_replay_execution_spec_document_invalid")
    try:
        from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec

        execution_document = json.loads(execution_document_json)
        if (
            not isinstance(execution_document, Mapping)
            or _canonical_json_bytes(execution_document).decode("ascii") != execution_document_json
        ):
            _fail("policy_state_replay_execution_spec_document_invalid")
        reconstructed_spec = RLCExecutionSpec.from_dict(execution_document)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            "policy_state_replay_execution_spec_document_invalid"
        ) from exc
    if reconstructed_spec.sha256 != execution_semantic:
        _fail("policy_state_replay_execution_spec_semantic_mismatch")
    if verify_files:
        payload = _stable_file_bytes(
            Path(execution_binding["path"]),
            maximum=int(execution_binding["size_bytes"]),
            role="execution_spec",
        )
        try:
            parsed = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise VerifiedTransitionPolicyStateReplayError(
                "policy_state_replay_execution_spec_json_invalid"
            ) from exc
        if parsed != execution_document:
            _fail("policy_state_replay_execution_spec_document_mismatch")

    sources = contract.get("source_bindings")
    if not isinstance(sources, Mapping) or not sources:
        _fail("policy_state_replay_source_bindings_invalid")
    normalized_sources = {}
    for role in sorted(sources):
        normalized_role = _identifier(role, role="source_role")
        normalized_sources[normalized_role] = _validate_file_binding(
            sources[role],
            role=f"source_{normalized_role}",
            verify_files=verify_files,
        )
    if list(sources) != sorted(sources) or contract.get("source_bindings_sha256") != _digest(
        normalized_sources
    ):
        _fail("policy_state_replay_source_bindings_digest_mismatch")

    try:
        from core.learning.verified_transition_measurement_chain import (
            validate_recurrent_grpo_config_contract,
        )
        from core.learning.verified_transition_policy_probe import (
            validate_initial_policy_state_custody,
        )

        custody = validate_initial_policy_state_custody(
            contract.get("initial_policy_state_custody")
        )
        recurrent_config = validate_recurrent_grpo_config_contract(
            contract.get("recurrent_grpo_config")
        )
    except Exception as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            "policy_state_replay_state_or_objective_contract_invalid"
        ) from exc
    if (
        contract.get("initial_policy_state_custody_sha256") != custody["custody_sha256"]
        or custody["initial_policy_sha256"] != initial_policy
        or custody["execution_spec_sha256"] != execution_semantic
        or contract.get("recurrent_grpo_config_sha256") != _digest(recurrent_config)
    ):
        _fail("policy_state_replay_state_or_objective_contract_mismatch")
    if verify_files:
        _verify_private_custody_artifact(
            path=custody["initial_adapter_path"],
            artifact=custody["initial_adapter_artifact"],
            role="initial_adapter",
        )
        _verify_private_custody_artifact(
            path=custody["initial_optimizer_path"],
            artifact=custody["initial_optimizer_artifact"],
            role="initial_optimizer",
        )
    optimizer_config = contract.get("optimizer_config")
    if not isinstance(optimizer_config, Mapping):
        _fail("policy_state_replay_optimizer_config_invalid")
    _optimizer_learning_rate(optimizer_config)
    if dict(optimizer_config) != custody["optimizer_initialization"] or contract.get(
        "optimizer_config_sha256"
    ) != _digest(dict(optimizer_config)):
        _fail("policy_state_replay_optimizer_config_digest_mismatch")

    trajectory_json = contract.get("verified_trajectory_config_json")
    if not isinstance(trajectory_json, str) or not trajectory_json:
        _fail("policy_state_replay_trajectory_config_invalid")
    try:
        from core.learning.recurrent_grpo import VerifiedTrajectoryGroupConfig

        trajectory_config = json.loads(trajectory_json)
        if (
            not isinstance(trajectory_config, Mapping)
            or _canonical_json_bytes(trajectory_config).decode("ascii") != trajectory_json
        ):
            _fail("policy_state_replay_trajectory_config_invalid")
        reconstructed_trajectory = VerifiedTrajectoryGroupConfig.from_dict(trajectory_config)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            "policy_state_replay_trajectory_config_invalid"
        ) from exc
    canonical_trajectory = reconstructed_trajectory.to_dict()
    if (
        reconstructed_trajectory.intervention_config is None
        or canonical_trajectory != dict(trajectory_config)
        or contract.get("verified_trajectory_config_sha256") != _digest(canonical_trajectory)
    ):
        _fail("policy_state_replay_trajectory_config_mismatch")
    timeout = contract.get("external_verifier_max_seconds")
    if type(timeout) is not int or not 300 <= timeout <= 93_600:
        _fail("policy_state_replay_external_verifier_budget_invalid")
    return contract


def build_policy_state_replay_contract(
    *,
    preregistration_contract_sha256: str,
    initial_policy_sha256: str,
    model_path: str | Path,
    base_checkpoint: Mapping[str, Any],
    behavior_bundle: Mapping[str, Any],
    execution_spec_path: str | Path,
    execution_spec_document: Mapping[str, Any],
    source_bindings: Mapping[str, Mapping[str, Any]],
    initial_policy_state_custody: Mapping[str, Any],
    recurrent_grpo_config: Mapping[str, Any],
    verified_trajectory_config: Mapping[str, Any],
    external_verifier_max_seconds: int,
) -> dict[str, Any]:
    """Seal every input an external process needs to reconstruct one update."""

    resolved_model = Path(model_path).expanduser().resolve(strict=True)
    resolved_spec = Path(execution_spec_path).expanduser().resolve(strict=True)
    spec_payload = _stable_file_bytes(
        resolved_spec,
        maximum=16 * 1024 * 1024,
        role="execution_spec",
    )
    try:
        parsed_spec = json.loads(spec_payload)
        from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec

        spec = RLCExecutionSpec.from_dict(parsed_spec)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            "policy_state_replay_execution_spec_document_invalid"
        ) from exc
    if dict(execution_spec_document) != parsed_spec:
        _fail("policy_state_replay_execution_spec_document_mismatch")
    normalized_sources = {
        role: _validate_file_binding(binding, role=f"source_{role}", verify_files=True)
        for role, binding in sorted(source_bindings.items())
    }
    custody = cast(
        dict[str, Any],
        _clone(initial_policy_state_custody, role="initial_state_custody"),
    )
    optimizer_config = custody.get("optimizer_initialization")
    body = {
        "schema": POLICY_STATE_REPLAY_CONTRACT_SCHEMA,
        "preregistration_contract_sha256": preregistration_contract_sha256,
        "initial_policy_sha256": initial_policy_sha256,
        "model": {
            "path": str(resolved_model),
            "base_checkpoint": dict(base_checkpoint),
            "base_checkpoint_sha256": _digest(base_checkpoint),
            "behavior_bundle": dict(behavior_bundle),
            "behavior_bundle_sha256": _digest(behavior_bundle),
        },
        "execution_spec": {
            "path": str(resolved_spec),
            "sha256": hashlib.sha256(spec_payload).hexdigest(),
            "size_bytes": len(spec_payload),
            "semantic_sha256": spec.sha256,
            "document_json": _canonical_json_bytes(parsed_spec).decode("ascii"),
        },
        "source_bindings": normalized_sources,
        "source_bindings_sha256": _digest(normalized_sources),
        "initial_policy_state_custody": custody,
        "initial_policy_state_custody_sha256": custody.get("custody_sha256"),
        "optimizer_config": optimizer_config,
        "optimizer_config_sha256": _digest(optimizer_config),
        "recurrent_grpo_config": dict(recurrent_grpo_config),
        "recurrent_grpo_config_sha256": _digest(recurrent_grpo_config),
        "verified_trajectory_config_json": _canonical_json_bytes(verified_trajectory_config).decode(
            "ascii"
        ),
        "verified_trajectory_config_sha256": _digest(verified_trajectory_config),
        "external_verifier_max_seconds": external_verifier_max_seconds,
    }
    return validate_policy_state_replay_contract(
        {**body, "contract_sha256": _digest(body)},
        verify_files=True,
    )


def _flat_tensor_map(value: Any, *, role: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or not value
        or any(not isinstance(key, str) or not key for key in value)
        or any(isinstance(tensor, Mapping) for tensor in value.values())
    ):
        _fail(f"policy_state_replay_{role}_tensor_map_invalid")
    result = dict(value)
    try:
        import mlx.core as mx

        mx.eval(*result.values())
        for tensor in result.values():
            tuple(tensor.shape)
            str(tensor.dtype)
    except Exception as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            f"policy_state_replay_{role}_tensor_map_invalid"
        ) from exc
    return result


def _tensor_payload(value: Any) -> bytes:
    import mlx.core as mx
    import numpy as np

    try:
        array = np.asarray(value)
    except RuntimeError:
        array = np.asarray(value.astype(mx.float32))
    return array.tobytes(order="C")


def tensor_map_identity(
    tensors: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    """Bind every tensor's name, shape, dtype, and exact evaluated value."""

    normalized = _flat_tensor_map(tensors, role=role)
    entries = []
    for name in sorted(normalized):
        tensor = normalized[name]
        entries.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "value_sha256": hashlib.sha256(_tensor_payload(tensor)).hexdigest(),
            }
        )
    body = {
        "schema": TENSOR_MAP_IDENTITY_SCHEMA,
        "role": role,
        "tensor_count": len(entries),
        "tensor_keys_sha256": _digest([entry["name"] for entry in entries]),
        "tensors": entries,
    }
    return {**body, "tensor_root_sha256": _digest(body)}


def _validate_tensor_map_identity(value: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TENSOR_IDENTITY_KEYS:
        _fail(f"policy_state_replay_{role}_identity_schema_invalid")
    identity = cast(
        dict[str, Any],
        _clone(value, role=f"{role}_identity"),
    )
    entries = identity.get("tensors")
    if (
        identity.get("schema") != TENSOR_MAP_IDENTITY_SCHEMA
        or identity.get("role") != role
        or type(identity.get("tensor_count")) is not int
        or identity["tensor_count"] <= 0
        or not isinstance(entries, list)
        or len(entries) != identity["tensor_count"]
    ):
        _fail(f"policy_state_replay_{role}_identity_invalid")
    names = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != _TENSOR_ENTRY_KEYS:
            _fail(f"policy_state_replay_{role}_tensor_entry_invalid")
        name = entry.get("name")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(dtype, str)
            or not dtype
            or not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
        ):
            _fail(f"policy_state_replay_{role}_tensor_entry_invalid")
        _sha256(
            entry.get("value_sha256"),
            role=f"{role}_tensor_value",
        )
        names.append(name)
    if names != sorted(set(names)) or identity.get("tensor_keys_sha256") != _digest(names):
        _fail(f"policy_state_replay_{role}_tensor_inventory_invalid")
    unsigned = dict(identity)
    root = unsigned.pop("tensor_root_sha256")
    if root != _digest(unsigned):
        _fail(f"policy_state_replay_{role}_identity_digest_mismatch")
    return identity


def validate_policy_state_replay_receipt(value: Any) -> dict[str, Any]:
    """Strictly reconstruct a positive external policy-state replay receipt."""

    if not isinstance(value, Mapping) or set(value) != _REPLAY_RECEIPT_KEYS:
        _fail("policy_state_replay_receipt_schema_invalid")
    receipt = cast(dict[str, Any], _clone(value, role="receipt"))
    unsigned = dict(receipt)
    observed = _sha256(
        unsigned.pop("receipt_sha256"),
        role="receipt",
    )
    if observed != _digest(unsigned):
        _fail("policy_state_replay_receipt_digest_mismatch")
    for field in (
        "execution_spec_sha256",
        "policy_before_sha256",
        "policy_after_sha256",
        "objective_receipt_sha256",
        "optimizer_config_sha256",
    ):
        _sha256(receipt.get(field), role=field)
    optimizer_config = receipt.get("optimizer_config")
    if not isinstance(optimizer_config, Mapping):
        _fail("policy_state_replay_optimizer_config_invalid")
    _optimizer_learning_rate(optimizer_config)
    if receipt["optimizer_config_sha256"] != _digest(dict(optimizer_config)):
        _fail("policy_state_replay_optimizer_config_digest_mismatch")
    identities = {
        role: _validate_tensor_map_identity(receipt.get(f"{role}_identity"), role=role)
        for role in (
            "pre_adapter",
            "pre_optimizer",
            "gradient",
            "post_adapter",
            "post_optimizer",
        )
    }
    by_name = {
        role: {entry["name"]: entry for entry in identity["tensors"]}
        for role, identity in identities.items()
    }
    adapter_names = set(by_name["pre_adapter"])
    optimizer_names = set(by_name["pre_optimizer"])
    if (
        receipt.get("schema") != POLICY_STATE_REPLAY_RECEIPT_SCHEMA
        or receipt["policy_before_sha256"] == receipt["policy_after_sha256"]
        or set(by_name["post_adapter"]) != adapter_names
        or set(by_name["gradient"]) != adapter_names
        or set(by_name["post_optimizer"]) != optimizer_names
        or any(
            by_name["gradient"][name][field] != by_name["pre_adapter"][name][field]
            for name in adapter_names
            for field in ("dtype", "shape")
        )
        or receipt.get("optimizer_update_count") != 1
        or receipt.get("objective_recomputed") is not True
        or receipt.get("all_gradient_tensors_recomputed") is not True
        or receipt.get("adapter_post_state_exact") is not True
        or receipt.get("optimizer_post_state_exact") is not True
        or receipt.get("external_policy_state_replayed") is not True
    ):
        _fail("policy_state_replay_receipt_invalid")
    return receipt


def _tensor_maps_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_map = _flat_tensor_map(left, role="comparison_left")
    right_map = _flat_tensor_map(right, role="comparison_right")
    if set(left_map) != set(right_map):
        return False
    try:
        import mlx.core as mx

        comparisons = []
        for key in sorted(left_map):
            if tuple(left_map[key].shape) != tuple(right_map[key].shape) or str(
                left_map[key].dtype
            ) != str(right_map[key].dtype):
                return False
            comparisons.append(mx.array_equal(left_map[key], right_map[key]))
        mx.eval(*comparisons)
        return all(bool(value) for value in comparisons)
    except Exception:
        return False


def _require_same_layout(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    role: str,
) -> None:
    observed_map = _flat_tensor_map(observed, role=f"{role}_observed")
    expected_map = _flat_tensor_map(expected, role=f"{role}_expected")
    if set(observed_map) != set(expected_map):
        _fail(f"policy_state_replay_{role}_keyset_mismatch")
    for key in sorted(observed_map):
        if tuple(observed_map[key].shape) != tuple(expected_map[key].shape) or str(
            observed_map[key].dtype
        ) != str(expected_map[key].dtype):
            _fail(f"policy_state_replay_{role}_layout_mismatch")


def _optimizer_learning_rate(config: Mapping[str, Any]) -> float:
    if not isinstance(config, Mapping):
        _fail("policy_state_replay_optimizer_config_invalid")
    try:
        learning_rate = float.fromhex(cast(str, config.get("learning_rate_hex")))
    except (TypeError, ValueError):
        _fail("policy_state_replay_optimizer_config_invalid")
    expected = recurrent_policy_optimizer_config(learning_rate)
    if dict(config) != expected:
        _fail("policy_state_replay_optimizer_config_mismatch")
    return learning_rate


def _objective_material(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_method = getattr(value, "receipt", None)
    gradients = getattr(value, "gradients", None)
    if not callable(receipt_method) or gradients is None:
        _fail("policy_state_replay_objective_result_invalid")
    try:
        receipt = _clone(receipt_method(), role="objective_receipt")
        from mlx.utils import tree_flatten

        flat_gradients = dict(tree_flatten(gradients))
    except Exception as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            "policy_state_replay_objective_result_invalid"
        ) from exc
    return cast(dict[str, Any], receipt), _flat_tensor_map(
        flat_gradients,
        role="gradient",
    )


def replay_verified_policy_transition(
    *,
    model: Any,
    pre_adapter_tensors: Mapping[str, Any],
    pre_optimizer_tensors: Mapping[str, Any],
    expected_post_adapter_tensors: Mapping[str, Any],
    expected_post_optimizer_tensors: Mapping[str, Any],
    expected_objective_receipt: Mapping[str, Any],
    objective_factory: Callable[[Any], Any],
    optimizer_config: Mapping[str, Any],
    execution_spec_sha256: str,
    expected_policy_before_sha256: str,
    expected_policy_after_sha256: str,
) -> dict[str, Any]:
    """Recompute and exactly replay one producer transition in isolation."""

    from mlx.utils import tree_flatten, tree_map_with_path

    execution_spec = _sha256(
        execution_spec_sha256,
        role="execution_spec",
    )
    policy_before = _sha256(
        expected_policy_before_sha256,
        role="policy_before",
    )
    policy_after = _sha256(
        expected_policy_after_sha256,
        role="policy_after",
    )
    if policy_before == policy_after:
        _fail("policy_state_replay_policy_unchanged")
    if not callable(objective_factory):
        _fail("policy_state_replay_objective_factory_invalid")

    pre_adapter = _flat_tensor_map(pre_adapter_tensors, role="pre_adapter")
    pre_optimizer = _flat_tensor_map(
        pre_optimizer_tensors,
        role="pre_optimizer",
    )
    expected_post_adapter = _flat_tensor_map(
        expected_post_adapter_tensors,
        role="expected_post_adapter",
    )
    expected_post_optimizer = _flat_tensor_map(
        expected_post_optimizer_tensors,
        role="expected_post_optimizer",
    )
    if recurrent_policy_tensor_map_sha256(pre_adapter, execution_spec) != policy_before:
        _fail("policy_state_replay_pre_policy_digest_mismatch")
    if recurrent_policy_tensor_map_sha256(expected_post_adapter, execution_spec) != policy_after:
        _fail("policy_state_replay_post_policy_digest_mismatch")

    try:
        import mlx.core as mx

        current_adapter = dict(tree_flatten(model.trainable_parameters()))
        _require_same_layout(current_adapter, pre_adapter, role="adapter")
        model.load_weights(list(pre_adapter.items()), strict=False)
        mx.eval(model.trainable_parameters())
        restored_adapter = dict(tree_flatten(model.trainable_parameters()))
    except VerifiedTransitionPolicyStateReplayError:
        raise
    except Exception as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            "policy_state_replay_adapter_restore_failed"
        ) from exc
    if not _tensor_maps_equal(restored_adapter, pre_adapter):
        _fail("policy_state_replay_pre_adapter_restore_mismatch")

    learning_rate = _optimizer_learning_rate(optimizer_config)
    try:
        optimizer = build_recurrent_policy_optimizer(learning_rate)
        optimizer.init(model.trainable_parameters())
        initialized_optimizer = dict(tree_flatten(optimizer.state))
        _require_same_layout(
            initialized_optimizer,
            pre_optimizer,
            role="optimizer",
        )
        restored_optimizer = tree_map_with_path(
            lambda path, _value: pre_optimizer[path],
            optimizer.state,
        )
        if not isinstance(restored_optimizer, dict):
            _fail("policy_state_replay_optimizer_tree_invalid")
        optimizer.state = restored_optimizer
        optimizer.init(model.trainable_parameters())
        mx.eval(optimizer.state)
        observed_pre_optimizer = dict(tree_flatten(optimizer.state))
    except VerifiedTransitionPolicyStateReplayError:
        raise
    except Exception as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            "policy_state_replay_optimizer_restore_failed"
        ) from exc
    if not _tensor_maps_equal(observed_pre_optimizer, pre_optimizer):
        _fail("policy_state_replay_pre_optimizer_restore_mismatch")

    expected_receipt = cast(
        dict[str, Any],
        _clone(expected_objective_receipt, role="expected_objective_receipt"),
    )
    objective_receipt, gradients = _objective_material(objective_factory(model))
    if objective_receipt != expected_receipt:
        _fail("policy_state_replay_objective_receipt_mismatch")
    observed_after_objective = dict(tree_flatten(model.trainable_parameters()))
    if not _tensor_maps_equal(observed_after_objective, pre_adapter):
        _fail("policy_state_replay_objective_mutated_policy")
    _require_same_layout(gradients, pre_adapter, role="gradient")
    finite = [mx.all(mx.isfinite(value)) for value in gradients.values()]
    mx.eval(*finite)
    if not all(bool(value) for value in finite):
        _fail("policy_state_replay_gradient_nonfinite")

    try:
        gradient_tree = tree_map_with_path(
            lambda path, _value: gradients[path],
            model.trainable_parameters(),
        )
        optimizer.update(model, gradient_tree)
        mx.eval(model.trainable_parameters(), optimizer.state)
        observed_post_adapter = dict(tree_flatten(model.trainable_parameters()))
        observed_post_optimizer = dict(tree_flatten(optimizer.state))
    except Exception as exc:
        raise VerifiedTransitionPolicyStateReplayError(
            "policy_state_replay_optimizer_update_failed"
        ) from exc
    if not _tensor_maps_equal(observed_post_adapter, expected_post_adapter):
        _fail("policy_state_replay_post_adapter_mismatch")
    if not _tensor_maps_equal(observed_post_optimizer, expected_post_optimizer):
        _fail("policy_state_replay_post_optimizer_mismatch")

    pre_adapter_identity = tensor_map_identity(pre_adapter, role="pre_adapter")
    pre_optimizer_identity = tensor_map_identity(
        pre_optimizer,
        role="pre_optimizer",
    )
    gradient_identity = tensor_map_identity(gradients, role="gradient")
    post_adapter_identity = tensor_map_identity(
        observed_post_adapter,
        role="post_adapter",
    )
    post_optimizer_identity = tensor_map_identity(
        observed_post_optimizer,
        role="post_optimizer",
    )
    body = {
        "schema": POLICY_STATE_REPLAY_RECEIPT_SCHEMA,
        "execution_spec_sha256": execution_spec,
        "policy_before_sha256": policy_before,
        "policy_after_sha256": policy_after,
        "objective_receipt_sha256": _digest(expected_receipt),
        "optimizer_config": dict(optimizer_config),
        "optimizer_config_sha256": _digest(dict(optimizer_config)),
        "pre_adapter_identity": pre_adapter_identity,
        "pre_optimizer_identity": pre_optimizer_identity,
        "gradient_identity": gradient_identity,
        "post_adapter_identity": post_adapter_identity,
        "post_optimizer_identity": post_optimizer_identity,
        "optimizer_update_count": 1,
        "objective_recomputed": True,
        "all_gradient_tensors_recomputed": True,
        "adapter_post_state_exact": True,
        "optimizer_post_state_exact": True,
        "external_policy_state_replayed": True,
    }
    return validate_policy_state_replay_receipt({**body, "receipt_sha256": _digest(body)})


__all__ = [
    "POLICY_STATE_REPLAY_CONTRACT_SCHEMA",
    "POLICY_STATE_REPLAY_RECEIPT_SCHEMA",
    "TENSOR_MAP_IDENTITY_SCHEMA",
    "VerifiedTransitionPolicyStateReplayError",
    "build_policy_state_replay_contract",
    "replay_verified_policy_transition",
    "tensor_map_identity",
    "validate_policy_state_replay_contract",
    "validate_policy_state_replay_receipt",
]
