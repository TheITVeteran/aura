"""Fail-closed identity for resident recurrent-SFT adapter packages.

The resident bootstrap trainer emits crash-consistent checkpoint generations,
not the historical recurrence-native v2 directory layout.  This contract binds
that native evidence directly so post-training campaigns never need to relabel
an SFT checkpoint as a GRPO artifact.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any, Final, Never

from core.brain.llm.latent_cortex.adapter_identity import (
    TensorIdentity,
    normalize_tensor_metadata,
)
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
    canonical_json_bytes,
    strict_json_loads,
)
from core.learning.resident_recurrent_sft_bootstrap_authority import (
    TRAINING_AUTHORITY,
    sha256_json,
    validate_authority,
)
from core.learning.resident_recurrent_sft_bootstrap_state import (
    CHECKPOINT_SCHEMA,
    POINTER_SCHEMA,
    authority_state_bindings,
    validate_checkpoint_state,
)

LEGACY_MANIFEST_SCHEMA: Final = "aura.resident_recurrent_sft_adapter_manifest.v1"
MANIFEST_SCHEMA: Final = "aura.resident_recurrent_sft_adapter_manifest.v2"
ROLE_CONDITIONED_MANIFEST_SCHEMA: Final = (
    "aura.resident_recurrent_sft_adapter_manifest.v3"
)
MANIFEST_SCHEMAS: Final = frozenset(
    {
        LEGACY_MANIFEST_SCHEMA,
        MANIFEST_SCHEMA,
        ROLE_CONDITIONED_MANIFEST_SCHEMA,
    }
)
IDENTITY_RECEIPT_SCHEMA: Final = "aura.resident_recurrent_sft_adapter_identity_receipt.v1"
CONTROLLER_COMPLETION_SCHEMA: Final = "aura.resident_recurrent_sft_controller_completion.v1"
INVOCATION_SCHEMA: Final = "aura.resident_recurrent_sft_bootstrap_invocation.v1"
MAX_ARTIFACT_BYTES: Final = 1 << 50
MAX_TENSORS: Final = 1_000_000
PACKAGE_COMPLETION_SCHEMA: Final = "aura.resident_recurrent_sft_adapter_package_completion.v1"


class ResidentRecurrentSFTAdapterIdentityError(ValueError):
    """A resident recurrent-SFT package is incomplete or inconsistent."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ResidentRecurrentSFTAdapterIdentityError(code)


def _exact(value: Any, keys: set[str], *, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        _fail(f"resident_sft_adapter_{role}_schema_invalid")
    return value


def _sha(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"resident_sft_adapter_{role}_sha256_invalid")
    return value


def _positive_int(value: Any, *, role: str, maximum: int = 1 << 60) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        _fail(f"resident_sft_adapter_{role}_invalid")
    return value


def _relative_path(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail(f"resident_sft_adapter_{role}_path_invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail(f"resident_sft_adapter_{role}_path_invalid")
    return value


def _binding(value: Any, *, role: str) -> dict[str, Any]:
    record = _exact(value, {"path", "sha256", "size_bytes"}, role=role)
    return {
        "path": _relative_path(record["path"], role=role),
        "sha256": _sha(record["sha256"], role=role),
        "size_bytes": _positive_int(record["size_bytes"], role=f"{role}_size"),
    }


def _verified_bytes(
    binding: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    *,
    role: str,
) -> bytes:
    payload = artifacts.get(str(binding["path"]))
    if not isinstance(payload, bytes):
        _fail(f"resident_sft_adapter_{role}_bytes_missing")
    if len(payload) != binding["size_bytes"]:
        _fail(f"resident_sft_adapter_{role}_size_mismatch")
    if hashlib.sha256(payload).hexdigest() != binding["sha256"]:
        _fail(f"resident_sft_adapter_{role}_sha256_mismatch")
    return payload


def _identity(value: Any, *, role: str, digest_key: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"resident_sft_adapter_{role}_invalid")
    record = dict(value)
    _sha(record.get(digest_key), role=f"{role}_{digest_key}")
    return record


def _lora(value: Any, *, manifest_schema: str) -> dict[str, Any]:
    depth_conditioned = manifest_schema in {
        MANIFEST_SCHEMA,
        ROLE_CONDITIONED_MANIFEST_SCHEMA,
    }
    role_conditioned = manifest_schema == ROLE_CONDITIONED_MANIFEST_SCHEMA
    keys = {
        "rank",
        "scale",
        "dropout",
        "layers",
        "targets",
        "wrapped_projections",
        "projection_paths",
        "trainable_params",
    }
    if depth_conditioned:
        keys.update({"conditioning_schema", "depth_bank_size"})
    if role_conditioned:
        keys.update({"role_conditioning_schema", "role_bank_size"})
    record = _exact(
        value,
        keys,
        role="lora",
    )
    rank = _positive_int(record["rank"], role="lora_rank", maximum=1 << 20)
    layers = _positive_int(record["layers"], role="lora_layers", maximum=1 << 20)
    wrapped = _positive_int(
        record["wrapped_projections"],
        role="lora_wrapped_projections",
        maximum=MAX_TENSORS // 2,
    )
    trainable = _positive_int(record["trainable_params"], role="lora_trainable")
    targets = record["targets"]
    paths = record["projection_paths"]
    if (
        not isinstance(targets, list)
        or not targets
        or len(targets) != len(set(targets))
        or any(not isinstance(target, str) or not target for target in targets)
        or not isinstance(paths, list)
        or not paths
        or len(paths) != len(set(paths))
        or any(
            not isinstance(path, str)
            or not path.startswith("model.layers.")
            or path.rsplit(".", 1)[-1] not in targets
            for path in paths
        )
        or wrapped != len(paths)
    ):
        _fail("resident_sft_adapter_lora_topology_invalid")
    scale = record["scale"]
    dropout = record["dropout"]
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not 0.0 < float(scale) <= 1024.0
        or isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not 0.0 <= float(dropout) <= 0.5
    ):
        _fail("resident_sft_adapter_lora_numeric_invalid")
    normalized = {
        "rank": rank,
        "scale": float(scale),
        "dropout": float(dropout),
        "layers": layers,
        "targets": list(targets),
        "wrapped_projections": wrapped,
        "projection_paths": list(paths),
        "trainable_params": trainable,
    }
    if depth_conditioned:
        if record.get("conditioning_schema") != "aura.depth_conditioned_lora.v1":
            _fail("resident_sft_adapter_depth_conditioning_schema_invalid")
        normalized.update(
            {
                "conditioning_schema": record["conditioning_schema"],
                "depth_bank_size": _positive_int(
                    record.get("depth_bank_size"),
                    role="depth_bank_size",
                    maximum=64,
                ),
            }
        )
    if role_conditioned:
        if record.get("role_conditioning_schema") != "aura.role_conditioned_lora.v1":
            _fail("resident_sft_adapter_role_conditioning_schema_invalid")
        normalized.update(
            {
                "role_conditioning_schema": record["role_conditioning_schema"],
                "role_bank_size": _positive_int(
                    record.get("role_bank_size"),
                    role="role_bank_size",
                    maximum=32,
                ),
            }
        )
    return normalized


def _tensor_inventory(
    expected: Any,
    actual: Iterable[TensorIdentity | Mapping[str, Any]],
    *,
    lora: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(expected, list) or not expected or len(expected) > MAX_TENSORS:
        _fail("resident_sft_adapter_tensor_inventory_invalid")
    try:
        expected_rows = normalize_tensor_metadata(expected)
        actual_rows = normalize_tensor_metadata(actual)
    except ValueError as exc:
        raise ResidentRecurrentSFTAdapterIdentityError(
            "resident_sft_adapter_tensor_inventory_invalid"
        ) from exc
    if expected_rows != actual_rows:
        _fail("resident_sft_adapter_tensor_inventory_mismatch")
    keys = [row.key for row in expected_rows]
    depth_bank_size = int(lora.get("depth_bank_size", 0))
    role_bank_size = int(lora.get("role_bank_size", 0))
    projections = sorted(lora["projection_paths"])
    expected_keys = {
        f"{projection}.{suffix}" for projection in projections for suffix in ("lora_a", "lora_b")
    }
    expected_keys.update(
        f"{projection}.{suffix}.{depth}"
        for projection in projections
        for suffix in ("depth_a", "depth_b")
        for depth in range(depth_bank_size)
    )
    expected_keys.update(
        f"{projection}.{suffix}.{role}"
        for projection in projections
        for suffix in ("role_a", "role_b")
        for role in range(role_bank_size)
    )
    if set(keys) != expected_keys:
        _fail("resident_sft_adapter_tensor_pair_mismatch")
    trainable = sum(dimension_product(row.shape) for row in expected_rows)
    if trainable != lora["trainable_params"]:
        _fail("resident_sft_adapter_trainable_parameter_mismatch")
    return [row.to_dict() for row in expected_rows]


def dimension_product(shape: Iterable[int]) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def topology_sha256(tensors: Iterable[Mapping[str, Any]]) -> str:
    """Reproduce the topology digest committed by the SFT bootstrap trainer.

    ``normalize_tensor_metadata`` deliberately stores portable dtype names such
    as ``float32``.  The bootstrap contract predates that normalization and
    hashed MLX's runtime spelling (``mlx.core.float32``).  Reconstruct that
    spelling only at this historical hash boundary; manifests retain the
    normalized representation.
    """

    rows = [
        {
            "name": row["key"],
            "shape": list(row["shape"]),
            "dtype": f"mlx.core.{str(row['dtype']).removeprefix('mlx.core.')}",
        }
        for row in tensors
    ]
    return sha256_json(sorted(rows, key=lambda row: row["name"]))


def declared_bindings(manifest: Mapping[str, Any]) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Return every byte binding declared by a parsed package manifest."""

    bindings = manifest.get("bindings")
    if not isinstance(bindings, Mapping):
        _fail("resident_sft_adapter_bindings_schema_invalid")
    source_bindings = bindings.get("source_snapshots")
    if not isinstance(source_bindings, Mapping):
        _fail("resident_sft_adapter_source_snapshots_schema_invalid")
    result: list[tuple[str, dict[str, Any]]] = []
    for role in sorted(set(bindings) - {"source_snapshots"}):
        result.append((role, _binding(bindings[role], role=role)))
    for role in sorted(source_bindings):
        result.append((f"source_{role}", _binding(source_bindings[role], role=f"source_{role}")))
    paths = [binding["path"] for _role, binding in result]
    if len(paths) != len(set(paths)):
        _fail("resident_sft_adapter_binding_path_duplicate")
    return tuple(result)


def validate_resident_recurrent_sft_adapter_identity(
    manifest_bytes: bytes,
    *,
    adapter_id: str,
    actual_base_checkpoint: Mapping[str, Any],
    actual_model_behavior_bundle: Mapping[str, Any],
    actual_personality_adapter: Mapping[str, Any],
    actual_runtime_environment: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
    tensor_metadata: Iterable[TensorIdentity | Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify a packaged terminal bootstrap checkpoint without model loading."""

    manifest = strict_json_loads(manifest_bytes, role="resident_sft_adapter_manifest")
    record = dict(
        _exact(
            manifest,
            {
                "schema",
                "adapter_id",
                "training_protocol",
                "base_checkpoint",
                "model_behavior_bundle",
                "personality_adapter",
                "training_runtime",
                "bindings",
                "lora",
                "tensors",
                "claim_boundary",
            },
            role="manifest",
        )
    )
    manifest_schema = record.get("schema")
    if manifest_schema not in MANIFEST_SCHEMAS or record.get("adapter_id") != adapter_id:
        _fail("resident_sft_adapter_manifest_identity_invalid")
    if record.get("training_protocol") != TRAINING_AUTHORITY:
        _fail("resident_sft_adapter_training_protocol_invalid")
    if record.get("claim_boundary") != {
        "training_objective_learned": True,
        "reasoning_gain_proven": False,
        "causal_gain_proven": False,
        "frontier_level_proven": False,
        "promotion_allowed": False,
    }:
        _fail("resident_sft_adapter_claim_boundary_invalid")

    base = _identity(record["base_checkpoint"], role="base_checkpoint", digest_key="fingerprint")
    behavior = _identity(
        record["model_behavior_bundle"], role="model_behavior", digest_key="bundle_sha256"
    )
    personality = _identity(
        record["personality_adapter"], role="personality", digest_key="identity_sha256"
    )
    training_runtime = _identity(
        record["training_runtime"], role="training_runtime", digest_key="identity_sha256"
    )
    evaluation_runtime = dict(actual_runtime_environment)
    normalized_training_runtime = dict(training_runtime)
    normalized_training_runtime.pop("interpreter", None)
    normalized_training_runtime.pop("identity_sha256", None)
    normalized_training_runtime["identity_sha256"] = sha256_json(normalized_training_runtime)
    if (
        base != dict(actual_base_checkpoint)
        or behavior != dict(actual_model_behavior_bundle)
        or personality != dict(actual_personality_adapter)
        or normalized_training_runtime != evaluation_runtime
    ):
        _fail("resident_sft_adapter_effective_stack_mismatch")

    bindings = {role: binding for role, binding in declared_bindings(record)}
    payloads = {
        role: _verified_bytes(binding, artifacts, role=role) for role, binding in bindings.items()
    }
    authority = validate_authority(
        strict_json_loads(payloads["authority"], role="resident_sft_authority"),
        allow_expired_resume=True,
    )
    if (
        authority["model"]["base_checkpoint"] != base
        or authority["model"]["behavior_bundle"] != behavior
        or authority["model"]["personality_bundle"] != personality
        or authority["runtime"] != training_runtime
        or authority["campaign_scope"] != "full_bootstrap"
    ):
        _fail("resident_sft_adapter_authority_identity_mismatch")

    for split in ("train", "validation"):
        expected = authority["dataset_artifacts"][split]
        binding = bindings[f"{split}_dataset"]
        if (
            binding["sha256"] != expected["sha256"]
            or binding["size_bytes"] != expected["size_bytes"]
        ):
            _fail(f"resident_sft_adapter_{split}_dataset_mismatch")
    for role in authority["sources"]:
        expected = authority["sources"][role]
        binding = bindings[f"source_{role}"]
        if (
            binding["sha256"] != expected["sha256"]
            or binding["size_bytes"] != expected["size_bytes"]
        ):
            _fail(f"resident_sft_adapter_source_{role}_mismatch")
    for role, authority_role in (
        ("execution_spec", "execution_spec"),
        ("trust_policy", "trust_policy"),
    ):
        expected = authority[authority_role]
        binding = bindings[role]
        if (
            binding["sha256"] != expected["sha256"]
            or binding["size_bytes"] != expected["size_bytes"]
        ):
            _fail(f"resident_sft_adapter_{role}_mismatch")
    try:
        execution_spec = RLCExecutionSpec.from_dict(
            strict_json_loads(payloads["execution_spec"], role="resident_sft_execution_spec")
        )
    except (TypeError, ValueError) as exc:
        raise ResidentRecurrentSFTAdapterIdentityError(
            "resident_sft_adapter_execution_spec_invalid"
        ) from exc
    if execution_spec.sha256 != authority["execution_spec"]["semantic_sha256"]:
        _fail("resident_sft_adapter_execution_spec_semantic_mismatch")

    pointer = strict_json_loads(payloads["checkpoint_pointer"], role="resident_sft_pointer")
    complete = strict_json_loads(payloads["checkpoint_complete"], role="resident_sft_complete")
    if (
        set(pointer) != {"schema", "checkpoint", "checkpoint_sequence", "complete_sha256"}
        or pointer.get("schema") != POINTER_SCHEMA
        or pointer.get("complete_sha256") != bindings["checkpoint_complete"]["sha256"]
        or complete.get("schema") != CHECKPOINT_SCHEMA
        or PurePosixPath(str(pointer.get("checkpoint"))).name != complete.get("checkpoint_id")
    ):
        _fail("resident_sft_adapter_checkpoint_chain_invalid")
    state_raw = complete.get("state")
    if not isinstance(state_raw, Mapping):
        _fail("resident_sft_adapter_checkpoint_state_invalid")
    state = validate_checkpoint_state(state_raw)
    expected_state_bindings = authority_state_bindings(authority)
    if (
        any(state[role] != digest for role, digest in expected_state_bindings.items())
        or state["terminal"] is not True
        or state["halt_reason"] != "max_steps"
        or state["step"] != authority["trainer"]["max_steps"]
        or state["checkpoint_sequence"] != pointer["checkpoint_sequence"]
        or complete.get("adapter")
        != {
            "path": PurePosixPath(bindings["adapter"]["path"]).name,
            "sha256": bindings["adapter"]["sha256"],
            "size_bytes": bindings["adapter"]["size_bytes"],
        }
        or complete.get("optimizer")
        != {
            "path": PurePosixPath(bindings["optimizer"]["path"]).name,
            "sha256": bindings["optimizer"]["sha256"],
            "size_bytes": bindings["optimizer"]["size_bytes"],
        }
    ):
        _fail("resident_sft_adapter_terminal_checkpoint_invalid")

    controller = strict_json_loads(
        payloads["controller_completion"], role="resident_sft_controller_completion"
    )
    invocation = strict_json_loads(
        payloads["terminal_invocation"], role="resident_sft_terminal_invocation"
    )
    controller_body = dict(controller)
    controller_sha = controller_body.pop("completion_sha256", None)
    invocation_body = dict(invocation)
    invocation_sha = invocation_body.pop("receipt_sha256", None)
    if (
        controller.get("schema") != CONTROLLER_COMPLETION_SCHEMA
        or controller_sha != sha256_json(controller_body)
        or controller.get("authority_sha256") != authority["authority_sha256"]
        or controller.get("bootstrap_complete") is not True
        or controller.get("base_checkpoint_immutable") is not True
        or controller.get("checkpoint", {}).get("complete_sha256")
        != bindings["checkpoint_complete"]["sha256"]
        or invocation.get("schema") != INVOCATION_SCHEMA
        or invocation_sha != sha256_json(invocation_body)
        or invocation.get("authority_sha256") != authority["authority_sha256"]
        or invocation.get("checkpoint_complete_sha256") != bindings["checkpoint_complete"]["sha256"]
        or invocation.get("bootstrap_complete") is not True
        or invocation.get("base_checkpoint_immutable") is not True
        or invocation.get("base_checkpoint_before") != base
        or invocation.get("base_checkpoint_after") != base
    ):
        _fail("resident_sft_adapter_completion_evidence_invalid")

    lora = _lora(record["lora"], manifest_schema=str(manifest_schema))
    if manifest_schema in {
        MANIFEST_SCHEMA,
        ROLE_CONDITIONED_MANIFEST_SCHEMA,
    } and lora["depth_bank_size"] != max(
        authority["dataset"]["depths"]
    ):
        _fail("resident_sft_adapter_depth_bank_authority_mismatch")
    trainer = authority["trainer"]
    if manifest_schema == ROLE_CONDITIONED_MANIFEST_SCHEMA:
        if lora["role_bank_size"] != trainer.get("role_conditioned_branches"):
            _fail("resident_sft_adapter_role_bank_authority_mismatch")
    elif trainer.get("role_conditioned_branches", 0):
        _fail("resident_sft_adapter_role_bank_missing")
    if (
        lora["rank"] != trainer["lora_rank"]
        or lora["scale"] != float(trainer["lora_scale"])
        or lora["dropout"] != float(trainer["lora_dropout"])
        or lora["layers"] != trainer["lora_layers"]
        or lora["targets"] != trainer["lora_targets"]
    ):
        _fail("resident_sft_adapter_lora_authority_mismatch")
    tensors = _tensor_inventory(record["tensors"], tensor_metadata, lora=lora)
    if topology_sha256(tensors) != state["adapter_topology_sha256"]:
        _fail("resident_sft_adapter_topology_digest_mismatch")

    package_completion_raw = artifacts.get("training_completion.json")
    if not isinstance(package_completion_raw, bytes):
        _fail("resident_sft_adapter_package_completion_bytes_missing")
    package_completion = strict_json_loads(
        package_completion_raw, role="resident_sft_package_completion"
    )
    if (
        set(package_completion)
        != {
            "schema",
            "complete",
            "halt_reason",
            "step",
            "adapter_sha256",
            "checkpoint_complete_sha256",
            "authority_sha256",
            "manifest_sha256",
        }
        or package_completion.get("schema") != PACKAGE_COMPLETION_SCHEMA
        or package_completion.get("complete") is not True
        or package_completion.get("halt_reason") != "max_steps"
        or package_completion.get("step") != state["step"]
        or package_completion.get("adapter_sha256") != bindings["adapter"]["sha256"]
        or package_completion.get("checkpoint_complete_sha256")
        != bindings["checkpoint_complete"]["sha256"]
        or package_completion.get("authority_sha256") != authority["authority_sha256"]
        or package_completion.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()
    ):
        _fail("resident_sft_adapter_package_completion_invalid")

    identity_material = {
        "schema": "aura.resident_recurrent_sft_adapter_identity.v1",
        "adapter_id": adapter_id,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "authority_sha256": authority["authority_sha256"],
        "base_checkpoint": base,
        "model_behavior_bundle": behavior,
        "personality_adapter": personality,
        "training_runtime": training_runtime,
        "evaluation_runtime": evaluation_runtime,
        "execution_spec_sha256": execution_spec.sha256,
        "dataset_sha256": authority["dataset"]["dataset_sha256"],
        "checkpoint_complete_sha256": bindings["checkpoint_complete"]["sha256"],
        "adapter": bindings["adapter"],
        "optimizer": bindings["optimizer"],
        "lora": lora,
        "tensors": tensors,
        "terminal_step": state["step"],
        "validation": {
            "baseline_mean_loss": state["baseline_validation"]["mean_loss"],
            "terminal_mean_loss": state["validation_trail"][-1]["mean_loss"],
        },
        "claim_boundary": record["claim_boundary"],
    }
    return {
        "schema": IDENTITY_RECEIPT_SCHEMA,
        "manifest_sha256": identity_material["manifest_sha256"],
        "composite_identity_sha256": hashlib.sha256(
            canonical_json_bytes(identity_material)
        ).hexdigest(),
        "adapter_id": adapter_id,
        "base_checkpoint_fingerprint": base["fingerprint"],
        "model_behavior_bundle_sha256": behavior["bundle_sha256"],
        "personality_adapter_bundle_sha256": str(personality.get("bundle_sha256") or ""),
        "training_runtime_identity_sha256": training_runtime["identity_sha256"],
        "evaluation_runtime_identity_sha256": evaluation_runtime["identity_sha256"],
        "authority_sha256": authority["authority_sha256"],
        "dataset_sha256": authority["dataset"]["dataset_sha256"],
        "execution_spec_sha256": execution_spec.sha256,
        "checkpoint_complete_sha256": bindings["checkpoint_complete"]["sha256"],
        "adapter_sha256": bindings["adapter"]["sha256"],
        "optimizer_sha256": bindings["optimizer"]["sha256"],
        "terminal_step": state["step"],
        "complete": True,
        "training_objective_learned": True,
        "reasoning_gain_proven": False,
        "promotion_allowed": False,
    }


__all__ = [
    "IDENTITY_RECEIPT_SCHEMA",
    "LEGACY_MANIFEST_SCHEMA",
    "MANIFEST_SCHEMA",
    "MANIFEST_SCHEMAS",
    "ROLE_CONDITIONED_MANIFEST_SCHEMA",
    "PACKAGE_COMPLETION_SCHEMA",
    "ResidentRecurrentSFTAdapterIdentityError",
    "declared_bindings",
    "dimension_product",
    "topology_sha256",
    "validate_resident_recurrent_sft_adapter_identity",
]
