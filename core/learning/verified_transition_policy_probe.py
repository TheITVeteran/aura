"""Sealed identity of a deterministic recurrent policy before training."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Never, cast

from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
    REQUIRED_SOURCE_ROLES,
)
from core.learning.verified_token_trace import validate_tokenizer_bundle_identity
from core.learning.verified_transition_episode import canonical_json_bytes
from core.runtime.file_read_gateway import read_stable_bytes

INITIAL_RECURRENT_POLICY_PROBE_SCHEMA = (
    "aura.verified_transition.initial_recurrent_policy_probe.v1"
)
INITIAL_RECURRENT_POLICY_PROBE_SCHEMA_V2 = (
    "aura.verified_transition.initial_recurrent_policy_probe.v2"
)
INITIAL_POLICY_STATE_CUSTODY_SCHEMA = (
    "aura.verified_transition.initial_policy_state_custody.v1"
)
_PROBE_KEYS_V1 = frozenset(
    {
        "schema",
        "campaign_id",
        "initial_policy_sha256",
        "dataset_sha256",
        "execution_spec_sha256",
        "base_checkpoint",
        "model_behavior_bundle",
        "tokenizer_bundle",
        "adapter_initialization",
        "source_bindings",
        "created_at_unix_ns",
        "receipt_sha256",
    }
)
_PROBE_KEYS_V2 = _PROBE_KEYS_V1 | {
    "optimizer_initialization",
    "initial_adapter_artifact",
    "initial_optimizer_artifact",
}
_ADAPTER_KEYS = frozenset(
    {
        "seed",
        "rank",
        "layers",
        "targets",
    }
)
_INITIAL_ADAPTER_ARTIFACT_KEYS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "tensor_count",
        "tensor_keys",
        "tensor_keys_sha256",
        "policy_sha256",
    }
)
_INITIAL_OPTIMIZER_ARTIFACT_KEYS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "tensor_count",
        "tensor_keys",
        "tensor_keys_sha256",
    }
)
_OPTIMIZER_KEYS = frozenset(
    {
        "class_name",
        "learning_rate_hex",
        "betas_hex",
        "eps_hex",
        "bias_correction",
    }
)
_INITIAL_POLICY_STATE_CUSTODY_KEYS = frozenset(
    {
        "schema",
        "initial_policy_probe_sha256",
        "initial_policy_sha256",
        "execution_spec_sha256",
        "adapter_initialization",
        "optimizer_initialization",
        "initial_adapter_artifact",
        "initial_optimizer_artifact",
        "initial_adapter_path",
        "initial_optimizer_path",
        "custody_sha256",
    }
)
_MAX_INITIAL_ADAPTER_BYTES = 16 * 1024 * 1024 * 1024


class InitialRecurrentPolicyProbeError(RuntimeError):
    """Stable probe-construction or validation failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise InitialRecurrentPolicyProbeError(code)


def _clone(value: Any) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
        _fail("initial_policy_probe_not_canonical_json")


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"initial_policy_probe_{role}_invalid")
    return value


def _identifier(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
    ):
        _fail(f"initial_policy_probe_{role}_invalid")
    return value


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def validate_initial_adapter_artifact_binding(value: Any) -> dict[str, Any]:
    """Validate the immutable tensor artifact bound by a v2 policy probe."""

    if (
        not isinstance(value, Mapping)
        or set(value) != _INITIAL_ADAPTER_ARTIFACT_KEYS
    ):
        _fail("initial_policy_probe_adapter_artifact_schema_invalid")
    binding = cast(dict[str, Any], _clone(value))
    path = binding.get("path")
    tensor_keys = binding.get("tensor_keys")
    if (
        not isinstance(tensor_keys, list)
        or any(not isinstance(key, str) or not key for key in tensor_keys)
    ):
        _fail("initial_policy_probe_adapter_artifact_invalid")
    if (
        not isinstance(path, str)
        or not path
        or Path(path).name != path
        or "/" in path
        or "\\" in path
        or path in {".", ".."}
        or type(binding.get("size_bytes")) is not int
        or not 0 < binding["size_bytes"] <= _MAX_INITIAL_ADAPTER_BYTES
        or type(binding.get("tensor_count")) is not int
        or binding["tensor_count"] < 2
        or tensor_keys != sorted(set(tensor_keys))
        or len(tensor_keys) != binding["tensor_count"]
        or any(
            not isinstance(key, str)
            or not key
            or not (key.endswith(".lora_a") or key.endswith(".lora_b"))
            for key in tensor_keys
        )
    ):
        _fail("initial_policy_probe_adapter_artifact_invalid")
    for role in ("sha256", "tensor_keys_sha256", "policy_sha256"):
        _sha256(binding.get(role), role=f"adapter_artifact_{role}")
    if binding["tensor_keys_sha256"] != hashlib.sha256(
        canonical_json_bytes(tensor_keys)
    ).hexdigest():
        _fail("initial_policy_probe_adapter_tensor_keys_mismatch")
    factors: dict[str, set[str]] = {}
    for key in tensor_keys:
        base, factor = key.rsplit(".", 1)
        factors.setdefault(base, set()).add(factor)
    if any(value != {"lora_a", "lora_b"} for value in factors.values()):
        _fail("initial_policy_probe_adapter_factor_pair_missing")
    return binding


def validate_initial_optimizer_artifact_binding(
    value: Any,
) -> dict[str, Any]:
    """Validate the immutable initial optimizer-state artifact."""

    if (
        not isinstance(value, Mapping)
        or set(value) != _INITIAL_OPTIMIZER_ARTIFACT_KEYS
    ):
        _fail("initial_policy_probe_optimizer_artifact_schema_invalid")
    binding = cast(dict[str, Any], _clone(value))
    path = binding.get("path")
    tensor_keys = binding.get("tensor_keys")
    if (
        not isinstance(tensor_keys, list)
        or any(not isinstance(key, str) or not key for key in tensor_keys)
    ):
        _fail("initial_policy_probe_optimizer_artifact_invalid")
    if (
        not isinstance(path, str)
        or not path
        or Path(path).name != path
        or "/" in path
        or "\\" in path
        or path in {".", ".."}
        or type(binding.get("size_bytes")) is not int
        or not 0 < binding["size_bytes"] <= _MAX_INITIAL_ADAPTER_BYTES
        or type(binding.get("tensor_count")) is not int
        or binding["tensor_count"] < 1
        or tensor_keys != sorted(set(tensor_keys))
        or len(tensor_keys) != binding["tensor_count"]
        or any(not isinstance(key, str) or not key for key in tensor_keys)
    ):
        _fail("initial_policy_probe_optimizer_artifact_invalid")
    for role in ("sha256", "tensor_keys_sha256"):
        _sha256(binding.get(role), role=f"optimizer_artifact_{role}")
    if binding["tensor_keys_sha256"] != hashlib.sha256(
        canonical_json_bytes(tensor_keys)
    ).hexdigest():
        _fail("initial_policy_probe_optimizer_tensor_keys_mismatch")
    return binding


def validate_optimizer_initialization(value: Any) -> dict[str, Any]:
    """Require the explicit Adam constructor used by trainer and replayer."""

    if not isinstance(value, Mapping) or set(value) != _OPTIMIZER_KEYS:
        _fail("initial_policy_probe_optimizer_initialization_schema_invalid")
    config = cast(dict[str, Any], _clone(value))
    learning_rate_hex = config.get("learning_rate_hex")
    betas_hex = config.get("betas_hex")
    eps_hex = config.get("eps_hex")
    try:
        learning_rate = float.fromhex(learning_rate_hex)
        betas = [
            float.fromhex(item) for item in betas_hex
        ]
        eps = float.fromhex(eps_hex)
    except (TypeError, ValueError):
        _fail("initial_policy_probe_optimizer_initialization_invalid")
    if (
        config.get("class_name") != "mlx.optimizers.Adam"
        or not 0.0 < float(learning_rate) <= 1.0
        or betas != [0.9, 0.999]
        or float(eps) != 1e-8
        or learning_rate_hex != learning_rate.hex()
        or betas_hex != [value.hex() for value in betas]
        or eps_hex != eps.hex()
        or config.get("bias_correction") is not False
    ):
        _fail("initial_policy_probe_optimizer_initialization_invalid")
    return config


def _validate_optimizer_adapter_topology(
    adapter_artifact: Mapping[str, Any],
    optimizer_artifact: Mapping[str, Any],
) -> None:
    expected = {"step", "learning_rate"}
    for key in adapter_artifact["tensor_keys"]:
        expected.add(f"{key}.m")
        expected.add(f"{key}.v")
    if optimizer_artifact["tensor_keys"] != sorted(expected):
        _fail("initial_policy_probe_optimizer_adapter_topology_mismatch")


def inspect_initial_adapter_snapshot(
    path: str | Path,
    *,
    execution_spec_sha256: str,
) -> dict[str, Any]:
    """Independently inspect one private, stable safetensors policy snapshot."""

    _sha256(execution_spec_sha256, role="snapshot_execution_spec_sha256")
    lexical = Path(path).expanduser().absolute()
    if lexical.is_symlink():
        _fail("initial_policy_probe_adapter_snapshot_symlink_rejected")
    try:
        resolved = lexical.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise InitialRecurrentPolicyProbeError(
            "initial_policy_probe_adapter_snapshot_unreadable"
        ) from exc
    if (
        resolved != lexical
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or metadata.st_nlink != 1
    ):
        _fail("initial_policy_probe_adapter_snapshot_not_private_owned_file")
    try:
        payload_before = read_stable_bytes(
            resolved,
            max_bytes=_MAX_INITIAL_ADAPTER_BYTES,
        )
        import mlx.core as mx

        tensors = mx.load(str(resolved))
        payload_after = read_stable_bytes(
            resolved,
            max_bytes=_MAX_INITIAL_ADAPTER_BYTES,
        )
    except Exception as exc:
        raise InitialRecurrentPolicyProbeError(
            "initial_policy_probe_adapter_snapshot_load_failed"
        ) from exc
    if payload_before != payload_after or not isinstance(tensors, Mapping):
        _fail("initial_policy_probe_adapter_snapshot_changed")
    keys = tuple(sorted(tensors))
    if (
        len(keys) < 2
        or any(
            not isinstance(key, str)
            or not key
            or not (key.endswith(".lora_a") or key.endswith(".lora_b"))
            for key in keys
        )
    ):
        _fail("initial_policy_probe_adapter_snapshot_topology_invalid")
    from core.learning.recurrent_grpo import recurrent_policy_tensor_map_sha256

    return validate_initial_adapter_artifact_binding(
        {
            "path": resolved.name,
            "sha256": hashlib.sha256(payload_before).hexdigest(),
            "size_bytes": len(payload_before),
            "tensor_count": len(keys),
            "tensor_keys": list(keys),
            "tensor_keys_sha256": hashlib.sha256(
                canonical_json_bytes(list(keys))
            ).hexdigest(),
            "policy_sha256": recurrent_policy_tensor_map_sha256(
                tensors,
                execution_spec_sha256,
            ),
        }
    )


def inspect_initial_optimizer_snapshot(path: str | Path) -> dict[str, Any]:
    """Independently inspect one private, stable Adam-state snapshot."""

    lexical = Path(path).expanduser().absolute()
    if lexical.is_symlink():
        _fail("initial_policy_probe_optimizer_snapshot_symlink_rejected")
    try:
        resolved = lexical.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise InitialRecurrentPolicyProbeError(
            "initial_policy_probe_optimizer_snapshot_unreadable"
        ) from exc
    if (
        resolved != lexical
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or metadata.st_nlink != 1
    ):
        _fail(
            "initial_policy_probe_optimizer_snapshot_not_private_owned_file"
        )
    try:
        payload_before = read_stable_bytes(
            resolved,
            max_bytes=_MAX_INITIAL_ADAPTER_BYTES,
        )
        import mlx.core as mx

        tensors = mx.load(str(resolved))
        payload_after = read_stable_bytes(
            resolved,
            max_bytes=_MAX_INITIAL_ADAPTER_BYTES,
        )
    except Exception as exc:
        raise InitialRecurrentPolicyProbeError(
            "initial_policy_probe_optimizer_snapshot_load_failed"
        ) from exc
    if payload_before != payload_after or not isinstance(tensors, Mapping):
        _fail("initial_policy_probe_optimizer_snapshot_changed")
    keys = tuple(sorted(tensors))
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        _fail("initial_policy_probe_optimizer_snapshot_topology_invalid")
    return validate_initial_optimizer_artifact_binding(
        {
            "path": resolved.name,
            "sha256": hashlib.sha256(payload_before).hexdigest(),
            "size_bytes": len(payload_before),
            "tensor_count": len(keys),
            "tensor_keys": list(keys),
            "tensor_keys_sha256": hashlib.sha256(
                canonical_json_bytes(list(keys))
            ).hexdigest(),
        }
    )


def build_initial_policy_state_custody(
    *,
    initial_policy_probe_sha256: str,
    initial_policy_sha256: str,
    execution_spec_sha256: str,
    adapter_initialization: Mapping[str, Any],
    optimizer_initialization: Mapping[str, Any],
    initial_adapter_artifact: Mapping[str, Any],
    initial_optimizer_artifact: Mapping[str, Any],
    initial_adapter_path: str | Path,
    initial_optimizer_path: str | Path,
) -> dict[str, Any]:
    """Bind a copied initial tensor state to one canonical absolute path."""

    body = {
        "schema": INITIAL_POLICY_STATE_CUSTODY_SCHEMA,
        "initial_policy_probe_sha256": initial_policy_probe_sha256,
        "initial_policy_sha256": initial_policy_sha256,
        "execution_spec_sha256": execution_spec_sha256,
        "adapter_initialization": dict(adapter_initialization),
        "optimizer_initialization": dict(optimizer_initialization),
        "initial_adapter_artifact": dict(initial_adapter_artifact),
        "initial_optimizer_artifact": dict(initial_optimizer_artifact),
        "initial_adapter_path": str(initial_adapter_path),
        "initial_optimizer_path": str(initial_optimizer_path),
    }
    return validate_initial_policy_state_custody(
        {**body, "custody_sha256": _digest(body)}
    )


def validate_initial_policy_state_custody(value: Any) -> dict[str, Any]:
    """Validate a materialized initial-state custody contract."""

    if (
        not isinstance(value, Mapping)
        or set(value) != _INITIAL_POLICY_STATE_CUSTODY_KEYS
    ):
        _fail("initial_policy_state_custody_schema_invalid")
    document = cast(dict[str, Any], _clone(value))
    unsigned = dict(document)
    custody_sha256 = unsigned.pop("custody_sha256")
    adapter = document.get("adapter_initialization")
    optimizer = validate_optimizer_initialization(
        document.get("optimizer_initialization")
    )
    artifact = validate_initial_adapter_artifact_binding(
        document.get("initial_adapter_artifact")
    )
    optimizer_artifact = validate_initial_optimizer_artifact_binding(
        document.get("initial_optimizer_artifact")
    )
    _validate_optimizer_adapter_topology(artifact, optimizer_artifact)
    raw_path = document.get("initial_adapter_path")
    raw_optimizer_path = document.get("initial_optimizer_path")
    if (
        document.get("schema") != INITIAL_POLICY_STATE_CUSTODY_SCHEMA
        or custody_sha256 != _digest(unsigned)
        or not isinstance(adapter, Mapping)
        or set(adapter) != _ADAPTER_KEYS
        or type(adapter.get("seed")) is not int
        or not 0 <= adapter["seed"] <= 0xFFFFFFFF
        or type(adapter.get("rank")) is not int
        or adapter["rank"] <= 0
        or type(adapter.get("layers")) is not int
        or adapter["layers"] <= 0
        or not isinstance(adapter.get("targets"), list)
        or not adapter["targets"]
        or any(
            not isinstance(item, str) or not item
            for item in adapter["targets"]
        )
        or not isinstance(raw_path, str)
        or not Path(raw_path).is_absolute()
        or Path(raw_path).resolve(strict=False) != Path(raw_path)
        or Path(raw_path).name != artifact["path"]
        or not isinstance(raw_optimizer_path, str)
        or not Path(raw_optimizer_path).is_absolute()
        or Path(raw_optimizer_path).resolve(strict=False)
        != Path(raw_optimizer_path)
        or Path(raw_optimizer_path).name != optimizer_artifact["path"]
    ):
        _fail("initial_policy_state_custody_invalid")
    for role in (
        "initial_policy_probe_sha256",
        "initial_policy_sha256",
        "execution_spec_sha256",
        "custody_sha256",
    ):
        _sha256(document.get(role), role=role)
    if artifact["policy_sha256"] != document["initial_policy_sha256"]:
        _fail("initial_policy_state_custody_policy_mismatch")
    document["optimizer_initialization"] = optimizer
    return document


def build_initial_recurrent_policy_probe(
    *,
    campaign_id: str,
    initial_policy_sha256: str,
    dataset_sha256: str,
    execution_spec_sha256: str,
    base_checkpoint: Mapping[str, Any],
    model_behavior_bundle: Mapping[str, Any],
    tokenizer_bundle: Mapping[str, Any],
    adapter_initialization: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    created_at_unix_ns: int,
    initial_adapter_artifact: Mapping[str, Any] | None = None,
    optimizer_initialization: Mapping[str, Any] | None = None,
    initial_optimizer_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    v2_values = (
        initial_adapter_artifact,
        optimizer_initialization,
        initial_optimizer_artifact,
    )
    if any(value is not None for value in v2_values) and not all(
        value is not None for value in v2_values
    ):
        _fail("initial_policy_probe_v2_state_incomplete")
    v2 = all(value is not None for value in v2_values)
    body = {
        "schema": (
            INITIAL_RECURRENT_POLICY_PROBE_SCHEMA_V2
            if v2
            else INITIAL_RECURRENT_POLICY_PROBE_SCHEMA
        ),
        "campaign_id": campaign_id,
        "initial_policy_sha256": initial_policy_sha256,
        "dataset_sha256": dataset_sha256,
        "execution_spec_sha256": execution_spec_sha256,
        "base_checkpoint": dict(base_checkpoint),
        "model_behavior_bundle": dict(model_behavior_bundle),
        "tokenizer_bundle": dict(tokenizer_bundle),
        "adapter_initialization": dict(adapter_initialization),
        "source_bindings": dict(source_bindings),
        "created_at_unix_ns": created_at_unix_ns,
    }
    if v2:
        assert initial_adapter_artifact is not None
        assert optimizer_initialization is not None
        assert initial_optimizer_artifact is not None
        body["optimizer_initialization"] = validate_optimizer_initialization(
            optimizer_initialization
        )
        body["initial_adapter_artifact"] = (
            validate_initial_adapter_artifact_binding(
                initial_adapter_artifact
            )
        )
        body["initial_optimizer_artifact"] = (
            validate_initial_optimizer_artifact_binding(
                initial_optimizer_artifact
            )
        )
    return validate_initial_recurrent_policy_probe(
        {**body, "receipt_sha256": _digest(body)}
    )


def validate_initial_recurrent_policy_probe(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("initial_policy_probe_schema_invalid")
    document = cast(dict[str, Any], _clone(value))
    schema = document.get("schema")
    expected_keys = (
        _PROBE_KEYS_V1
        if schema == INITIAL_RECURRENT_POLICY_PROBE_SCHEMA
        else _PROBE_KEYS_V2
        if schema == INITIAL_RECURRENT_POLICY_PROBE_SCHEMA_V2
        else None
    )
    if expected_keys is None or set(document) != expected_keys:
        _fail("initial_policy_probe_schema_invalid")
    unsigned = dict(document)
    receipt = unsigned.pop("receipt_sha256")
    adapter = document.get("adapter_initialization")
    sources = document.get("source_bindings")
    if (
        receipt != _digest(unsigned)
        or not isinstance(document.get("base_checkpoint"), Mapping)
        or not isinstance(document.get("model_behavior_bundle"), Mapping)
        or not isinstance(adapter, Mapping)
        or set(adapter) != _ADAPTER_KEYS
        or type(adapter.get("seed")) is not int
        or not 0 <= adapter["seed"] <= 0xFFFFFFFF
        or type(adapter.get("rank")) is not int
        or adapter["rank"] <= 0
        or type(adapter.get("layers")) is not int
        or adapter["layers"] <= 0
        or not isinstance(adapter.get("targets"), list)
        or not adapter["targets"]
        or any(not isinstance(item, str) or not item for item in adapter["targets"])
        or not isinstance(sources, Mapping)
        or set(sources) != REQUIRED_SOURCE_ROLES
        or type(document.get("created_at_unix_ns")) is not int
        or document["created_at_unix_ns"] <= 0
    ):
        _fail("initial_policy_probe_invalid")
    _identifier(document.get("campaign_id"), role="campaign_id")
    for role in (
        "initial_policy_sha256",
        "dataset_sha256",
        "execution_spec_sha256",
        "receipt_sha256",
    ):
        _sha256(document.get(role), role=role)
    validate_tokenizer_bundle_identity(document.get("tokenizer_bundle"))
    if schema == INITIAL_RECURRENT_POLICY_PROBE_SCHEMA_V2:
        artifact = validate_initial_adapter_artifact_binding(
            document.get("initial_adapter_artifact")
        )
        document["optimizer_initialization"] = (
            validate_optimizer_initialization(
                document.get("optimizer_initialization")
            )
        )
        optimizer_artifact = validate_initial_optimizer_artifact_binding(
            document.get("initial_optimizer_artifact")
        )
        document["initial_optimizer_artifact"] = optimizer_artifact
        _validate_optimizer_adapter_topology(artifact, optimizer_artifact)
        if artifact["policy_sha256"] != document["initial_policy_sha256"]:
            _fail("initial_policy_probe_adapter_policy_mismatch")
    for role, binding in sources.items():
        _identifier(role, role="source_role")
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256", "size_bytes"}
            or not isinstance(binding.get("path"), str)
            or type(binding.get("size_bytes")) is not int
            or binding["size_bytes"] <= 0
        ):
            _fail("initial_policy_probe_source_binding_invalid")
        _sha256(binding.get("sha256"), role="source_binding_sha256")
    return document


def validate_initial_recurrent_policy_probe_identity(
    value: Any,
    *,
    campaign_id: str,
    initial_policy_sha256: str,
    dataset_sha256: str,
    execution_spec_sha256: str,
    base_checkpoint: Mapping[str, Any],
    model_behavior_bundle: Mapping[str, Any],
    tokenizer_bundle: Mapping[str, Any],
    adapter_initialization: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    initial_adapter_artifact: Mapping[str, Any] | None = None,
    optimizer_initialization: Mapping[str, Any] | None = None,
    initial_optimizer_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild a sealed probe at its original time and require exact identity."""

    document = validate_initial_recurrent_policy_probe(value)
    expected = build_initial_recurrent_policy_probe(
        campaign_id=campaign_id,
        initial_policy_sha256=initial_policy_sha256,
        dataset_sha256=dataset_sha256,
        execution_spec_sha256=execution_spec_sha256,
        base_checkpoint=base_checkpoint,
        model_behavior_bundle=model_behavior_bundle,
        tokenizer_bundle=tokenizer_bundle,
        adapter_initialization=adapter_initialization,
        source_bindings=source_bindings,
        created_at_unix_ns=document["created_at_unix_ns"],
        initial_adapter_artifact=initial_adapter_artifact,
        optimizer_initialization=optimizer_initialization,
        initial_optimizer_artifact=initial_optimizer_artifact,
    )
    if document != expected:
        _fail("initial_policy_probe_identity_mismatch")
    return document


__all__ = [
    "INITIAL_POLICY_STATE_CUSTODY_SCHEMA",
    "INITIAL_RECURRENT_POLICY_PROBE_SCHEMA",
    "INITIAL_RECURRENT_POLICY_PROBE_SCHEMA_V2",
    "InitialRecurrentPolicyProbeError",
    "build_initial_policy_state_custody",
    "build_initial_recurrent_policy_probe",
    "inspect_initial_adapter_snapshot",
    "inspect_initial_optimizer_snapshot",
    "validate_initial_adapter_artifact_binding",
    "validate_initial_optimizer_artifact_binding",
    "validate_initial_policy_state_custody",
    "validate_initial_recurrent_policy_probe",
    "validate_initial_recurrent_policy_probe_identity",
    "validate_optimizer_initialization",
]
