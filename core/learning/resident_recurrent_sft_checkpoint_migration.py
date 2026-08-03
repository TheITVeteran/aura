"""Exact, source-attested migration of resident recurrent-SFT checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Never

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.learning.resident_recurrent_sft_bootstrap_authority import validate_authority
from core.learning.resident_recurrent_sft_bootstrap_state import (
    BINDING_ROLES,
    CHECKPOINT_SCHEMA,
    POINTER_SCHEMA,
    ResidentSFTBootstrapStateError,
    authority_state_bindings,
    inspect_checkpoint,
    validate_checkpoint_state,
)
from core.runtime.atomic_writer import (
    atomic_write_bytes,
    durable_replace,
    ensure_private_directory,
)

MIGRATION_SCHEMA: Final = "aura.resident_recurrent_sft_checkpoint_migration.v1"
MAX_JSON_BYTES: Final = 16 * 1024 * 1024
COPY_CHUNK_BYTES: Final = 4 * 1024 * 1024
ALLOWED_CHANGED_SOURCE_ROLES: Final = frozenset(
    {"trainer", "controller", "objective", "specialization_objective"}
)
APPROVED_SEMANTICS_PRESERVING_TRANSITIONS: Final = {
    (
        "specialization_objective",
        "9d9e12f64bf6edb6ac6c9695b2c0e63cf57a377c2e598dfca9be4cdf9fae8f6e",
        "8299def67d36726a4c82601210ef20ca530aef0ab7f5cb0691d5fbcacdd8b165",
    ): "exact_composite_gradient_host_spill_v1",
    (
        "specialization_objective",
        "8299def67d36726a4c82601210ef20ca530aef0ab7f5cb0691d5fbcacdd8b165",
        "46ffabd2f81547b08a4353276014d4fd3159c0ca3249d9f6aa596782d03dc185",
    ): "exact_recomputed_adjoint_v2",
    (
        "objective",
        "30ac036e37ad1e77b80a0d64db5ad3f0329a09c7e00e916249511a5d88a7f147",
        "8244952d64d76301e8ff08f6323948f7ac0db4cf5063a9c3c132ed6183ac92f4",
    ): "context_bound_layer_rematerialization_v1",
    (
        "specialization_objective",
        "8299def67d36726a4c82601210ef20ca530aef0ab7f5cb0691d5fbcacdd8b165",
        "ac6dbff3fa3d05b946bc94a73081f182ca6146caf88b379853fb92be6c9481a5",
    ): "exact_layer_rematerialized_adjoint_v3",
}


class ResidentSFTCheckpointMigrationError(RuntimeError):
    """The proposed recovery does not preserve the scientific run."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ResidentSFTCheckpointMigrationError(code)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _binding(path: Path, *, max_bytes: int | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail("resident_sft_migration_artifact_invalid")
    size = path.stat().st_size
    if size <= 0 or (max_bytes is not None and size > max_bytes):
        _fail("resident_sft_migration_artifact_size_invalid")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": size}


def _binding_matches(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(observed.get(key) == expected.get(key) for key in ("sha256", "size_bytes"))


def _read_authority(path: Path) -> dict[str, Any]:
    _binding(path, max_bytes=MAX_JSON_BYTES)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResidentSFTCheckpointMigrationError(
            "resident_sft_migration_authority_json_invalid"
        ) from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        _fail("resident_sft_migration_authority_noncanonical")
    return validate_authority(value, allow_expired_resume=True)


def _resolve_artifact_root(repo_root: Path, authority: Mapping[str, Any]) -> Path:
    root = repo_root.expanduser().resolve(strict=True)
    relative = Path(str(authority["artifact_root"]))
    if relative.is_absolute() or ".." in relative.parts:
        _fail("resident_sft_migration_artifact_root_invalid")
    resolved = (root / relative).resolve(strict=True)
    stat = resolved.stat()
    if authority["artifact_root_identity"] != {
        "st_dev": stat.st_dev,
        "st_ino": stat.st_ino,
    }:
        _fail("resident_sft_migration_artifact_root_identity_drift")
    return resolved


def _identity(authority: Mapping[str, Any]) -> dict[str, Any]:
    model = authority["model"]
    tokenizer = authority["tokenizer"]
    return {
        "campaign_scope": authority["campaign_scope"],
        "dataset_sha256": authority["dataset"]["dataset_sha256"],
        "base_checkpoint_sha256": model["base_checkpoint"]["fingerprint"],
        "behavior_sha256": model["behavior_bundle"]["bundle_sha256"],
        "personality_sha256": model["personality_bundle"]["identity_sha256"],
        # The aggregate tokenizer identity intentionally includes its absolute
        # capsule directory.  Migration compares the immutable artifact and
        # executable runtime identities instead of pretending two paths match.
        "tokenizer_artifact_sha256": tokenizer["artifact_sha256"],
        "tokenizer_runtime_sha256": tokenizer["runtime_sha256"],
        "execution_spec_sha256": authority["execution_spec"]["semantic_sha256"],
        "trainer_config_sha256": _sha(canonical_json_bytes(authority["trainer"])),
        "runtime_sha256": authority["runtime"]["identity_sha256"],
    }


def _trust_policy_identity(repo_root: Path, authority: Mapping[str, Any]) -> str:
    binding = authority["trust_policy"]
    path = (repo_root.expanduser().resolve(strict=True) / binding["path"]).resolve(strict=True)
    if _binding(path, max_bytes=MAX_JSON_BYTES)["sha256"] != binding["sha256"]:
        _fail("resident_sft_migration_trust_policy_binding_drift")
    try:
        policy = json.loads(path.read_bytes())
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResidentSFTCheckpointMigrationError(
            "resident_sft_migration_trust_policy_invalid"
        ) from exc
    if not isinstance(policy, dict):
        _fail("resident_sft_migration_trust_policy_invalid")
    for path_bound in ("campaign_id", "policy_sha256", "source"):
        policy.pop(path_bound, None)
    return _sha(canonical_json_bytes(policy))


def _changed_source_roles(
    source: Mapping[str, Any], destination: Mapping[str, Any]
) -> tuple[str, ...]:
    if set(source) != set(destination):
        _fail("resident_sft_migration_source_roles_changed")
    return tuple(
        role
        for role in sorted(source)
        if source[role]["sha256"] != destination[role]["sha256"]
        or source[role]["size_bytes"] != destination[role]["size_bytes"]
    )


def _source_transition_attestations(
    source: Mapping[str, Any],
    destination: Mapping[str, Any],
    changed_roles: Sequence[str],
) -> dict[str, dict[str, str]]:
    attestations: dict[str, dict[str, str]] = {}
    for role in changed_roles:
        if role not in ALLOWED_CHANGED_SOURCE_ROLES:
            _fail("resident_sft_migration_source_change_not_authorized")
        if role in {"trainer", "controller"}:
            continue
        source_sha = str(source[role]["sha256"])
        destination_sha = str(destination[role]["sha256"])
        attestation = APPROVED_SEMANTICS_PRESERVING_TRANSITIONS.get(
            (role, source_sha, destination_sha)
        )
        if attestation is None:
            _fail("resident_sft_migration_source_change_not_authorized")
        attestations[role] = {
            "source_sha256": source_sha,
            "destination_sha256": destination_sha,
            "attestation": attestation,
        }
    return attestations


def _copy_exact(source: Path, destination: Path) -> dict[str, Any]:
    source_binding = _binding(source)
    ensure_private_directory(destination.parent)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        os.fchmod(fd, 0o600)
        with source.open("rb") as reader, os.fdopen(fd, "wb") as writer:
            while chunk := reader.read(COPY_CHUNK_BYTES):
                writer.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if size != source_binding["size_bytes"] or digest.hexdigest() != source_binding["sha256"]:
            _fail("resident_sft_migration_copy_drift")
        durable_replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    copied = _binding(destination)
    if copied["sha256"] != source_binding["sha256"] or copied["size_bytes"] != size:
        _fail("resident_sft_migration_copy_drift")
    return copied


def migrate_checkpoint(
    *,
    source_repo_root: Path,
    source_authority_path: Path,
    destination_repo_root: Path,
    destination_authority_path: Path,
) -> dict[str, Any]:
    """Rebind one exact durable checkpoint to a repaired source closure."""

    source_authority_path = source_authority_path.expanduser().resolve(strict=True)
    destination_authority_path = destination_authority_path.expanduser().resolve(strict=True)
    source_authority = _read_authority(source_authority_path)
    destination_authority = _read_authority(destination_authority_path)
    source_root = _resolve_artifact_root(source_repo_root, source_authority)
    destination_root = _resolve_artifact_root(destination_repo_root, destination_authority)
    if source_root == destination_root:
        _fail("resident_sft_migration_roots_must_differ")
    if (destination_root / "latest.json").exists() or (destination_root / "checkpoints").exists():
        _fail("resident_sft_migration_destination_not_fresh")

    scientific_identity = _identity(source_authority)
    if scientific_identity != _identity(destination_authority):
        _fail("resident_sft_migration_scientific_identity_changed")
    source_trust_identity = _trust_policy_identity(source_repo_root, source_authority)
    destination_trust_identity = _trust_policy_identity(
        destination_repo_root, destination_authority
    )
    if source_trust_identity != destination_trust_identity:
        _fail("resident_sft_migration_trust_policy_changed")
    changed_roles = _changed_source_roles(
        source_authority["sources"], destination_authority["sources"]
    )
    if not changed_roles:
        _fail("resident_sft_migration_source_change_not_authorized")
    transition_attestations = _source_transition_attestations(
        source_authority["sources"],
        destination_authority["sources"],
        changed_roles,
    )

    source_bindings = authority_state_bindings(source_authority)
    destination_bindings = authority_state_bindings(destination_authority)
    inspected = inspect_checkpoint(source_root, expected_bindings=source_bindings)
    if inspected.state["terminal"]:
        _fail("resident_sft_migration_terminal_checkpoint_forbidden")
    preserved_state = {
        key: value for key, value in inspected.state.items() if key not in BINDING_ROLES
    }
    rebound_state = validate_checkpoint_state({**inspected.state, **destination_bindings})
    if {
        key: value for key, value in rebound_state.items() if key not in BINDING_ROLES
    } != preserved_state:
        _fail("resident_sft_migration_state_changed")

    generation_name = (
        f"migration-sequence-{rebound_state['checkpoint_sequence']:08d}-"
        f"step-{rebound_state['step']:08d}-{uuid.uuid4().hex}"
    )
    generation = ensure_private_directory(destination_root / "checkpoints" / generation_name)
    adapter = _copy_exact(
        inspected.checkpoint_dir / "adapter.safetensors", generation / "adapter.safetensors"
    )
    optimizer = _copy_exact(
        inspected.checkpoint_dir / "optimizer.safetensors", generation / "optimizer.safetensors"
    )
    complete = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_id": generation_name,
        "created_at_unix": time.time(),
        "state": rebound_state,
        "adapter": {key: adapter[key] for key in ("path", "sha256", "size_bytes")}
        | {"path": "adapter.safetensors"},
        "optimizer": {key: optimizer[key] for key in ("path", "sha256", "size_bytes")}
        | {"path": "optimizer.safetensors"},
    }
    complete_payload = canonical_json_bytes(complete)
    atomic_write_bytes(generation / "complete.json", complete_payload, mode=0o600)
    source_complete = _binding(inspected.checkpoint_dir / "complete.json", max_bytes=MAX_JSON_BYTES)
    receipt_material = {
        "schema": MIGRATION_SCHEMA,
        "prepared_at_unix": time.time(),
        "source": {
            "repo_root": str(source_repo_root.expanduser().resolve(strict=True)),
            "authority": _binding(source_authority_path, max_bytes=MAX_JSON_BYTES),
            "artifact_root": str(source_root),
            "checkpoint": str(inspected.checkpoint_dir),
            "complete": source_complete,
            "adapter": _binding(inspected.checkpoint_dir / "adapter.safetensors"),
            "optimizer": _binding(inspected.checkpoint_dir / "optimizer.safetensors"),
            "bindings": source_bindings,
        },
        "destination": {
            "repo_root": str(destination_repo_root.expanduser().resolve(strict=True)),
            "authority": _binding(destination_authority_path, max_bytes=MAX_JSON_BYTES),
            "artifact_root": str(destination_root),
            "checkpoint": str(generation),
            "complete": _binding(generation / "complete.json", max_bytes=MAX_JSON_BYTES),
            "adapter": adapter,
            "optimizer": optimizer,
            "bindings": destination_bindings,
        },
        "scientific_identity": scientific_identity,
        "trust_policy_identity_sha256": source_trust_identity,
        "changed_source_roles": list(changed_roles),
        "source_transition_attestations": transition_attestations,
        "migration_implementation": _binding(Path(__file__).resolve()),
        "preserved_state_sha256": _sha(canonical_json_bytes(preserved_state)),
        "preservation": {
            "adapter_state_reset": False,
            "optimizer_state_reset": False,
            "sample_cursor_reset": False,
            "loss_or_validation_history_reset": False,
        },
    }
    receipt = {
        **receipt_material,
        "migration_sha256": _sha(canonical_json_bytes(receipt_material)),
    }
    atomic_write_bytes(
        destination_root / "checkpoint-migration.json",
        canonical_json_bytes(receipt),
        mode=0o600,
    )
    pointer = {
        "schema": POINTER_SCHEMA,
        "checkpoint": f"checkpoints/{generation_name}",
        "checkpoint_sequence": rebound_state["checkpoint_sequence"],
        "complete_sha256": _sha(complete_payload),
    }
    atomic_write_bytes(destination_root / "latest.json", canonical_json_bytes(pointer), mode=0o600)
    migrated = inspect_checkpoint(destination_root, expected_bindings=destination_bindings)
    if migrated.state != rebound_state or migrated.adapter_binding["sha256"] != adapter["sha256"]:
        _fail("resident_sft_migration_postcondition_failed")
    return receipt


def verify_migration(
    path: Path,
    *,
    destination_repo_root: Path,
    destination_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently verify a migration against the current destination."""

    path = path.expanduser().resolve(strict=True)
    raw = path.read_bytes()
    try:
        receipt = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResidentSFTCheckpointMigrationError("resident_sft_migration_receipt_invalid") from exc
    if not isinstance(receipt, dict) or canonical_json_bytes(receipt) != raw:
        _fail("resident_sft_migration_receipt_invalid")
    claimed = receipt.get("migration_sha256")
    material = dict(receipt)
    material.pop("migration_sha256", None)
    source = receipt.get("source")
    destination = receipt.get("destination")
    preservation = receipt.get("preservation")
    changed_roles = receipt.get("changed_source_roles")
    implementation = receipt.get("migration_implementation")
    if (
        receipt.get("schema") != MIGRATION_SCHEMA
        or claimed != _sha(canonical_json_bytes(material))
        or not isinstance(source, Mapping)
        or not isinstance(destination, Mapping)
        or preservation
        != {
            "adapter_state_reset": False,
            "optimizer_state_reset": False,
            "sample_cursor_reset": False,
            "loss_or_validation_history_reset": False,
        }
        or not isinstance(changed_roles, list)
        or not changed_roles
        or not isinstance(implementation, Mapping)
    ):
        _fail("resident_sft_migration_receipt_invalid")
    observed_implementation = _binding(Path(__file__).resolve())
    if any(
        observed_implementation[key] != implementation.get(key) for key in ("sha256", "size_bytes")
    ):
        _fail("resident_sft_migration_implementation_drift")
    validated_authority = validate_authority(destination_authority, allow_expired_resume=True)
    destination_authority_binding = destination.get("authority")
    if not isinstance(destination_authority_binding, Mapping):
        _fail("resident_sft_migration_destination_binding_drift")
    destination_authority_path = Path(str(destination_authority_binding.get("path", "")))
    if (
        not _binding_matches(
            _binding(destination_authority_path, max_bytes=MAX_JSON_BYTES),
            destination_authority_binding,
        )
        or _read_authority(destination_authority_path) != validated_authority
    ):
        _fail("resident_sft_migration_destination_binding_drift")
    destination_root = _resolve_artifact_root(destination_repo_root, validated_authority)
    if path != destination_root / "checkpoint-migration.json":
        _fail("resident_sft_migration_receipt_path_invalid")
    expected_bindings = authority_state_bindings(validated_authority)
    if destination.get("bindings") != expected_bindings:
        _fail("resident_sft_migration_destination_binding_drift")
    if receipt.get("scientific_identity") != _identity(validated_authority):
        _fail("resident_sft_migration_scientific_identity_changed")
    if receipt.get("trust_policy_identity_sha256") != _trust_policy_identity(
        destination_repo_root, validated_authority
    ):
        _fail("resident_sft_migration_trust_policy_changed")
    source_authority_binding = source.get("authority")
    source_repo_root_value = source.get("repo_root")
    if not isinstance(source_authority_binding, Mapping) or not isinstance(
        source_repo_root_value, str
    ):
        _fail("resident_sft_migration_source_binding_drift")
    source_authority_path = Path(str(source_authority_binding.get("path", "")))
    if not _binding_matches(
        _binding(source_authority_path, max_bytes=MAX_JSON_BYTES),
        source_authority_binding,
    ):
        _fail("resident_sft_migration_source_binding_drift")
    source_authority = _read_authority(source_authority_path)
    source_repo_root = Path(source_repo_root_value)
    source_root = _resolve_artifact_root(source_repo_root, source_authority)
    observed_changed_roles = _changed_source_roles(
        source_authority["sources"], validated_authority["sources"]
    )
    observed_transition_attestations = _source_transition_attestations(
        source_authority["sources"],
        validated_authority["sources"],
        observed_changed_roles,
    )
    if (
        receipt.get("scientific_identity") != _identity(source_authority)
        or receipt.get("trust_policy_identity_sha256")
        != _trust_policy_identity(source_repo_root, source_authority)
        or tuple(changed_roles) != observed_changed_roles
        or receipt.get("source_transition_attestations") != observed_transition_attestations
    ):
        _fail("resident_sft_migration_source_identity_drift")
    try:
        source_inspected = inspect_checkpoint(
            source_root, expected_bindings=authority_state_bindings(source_authority)
        )
    except ResidentSFTBootstrapStateError as exc:
        raise ResidentSFTCheckpointMigrationError(
            "resident_sft_migration_source_binding_drift"
        ) from exc
    if source_inspected.checkpoint_dir != Path(str(source.get("checkpoint", ""))).resolve(
        strict=True
    ):
        _fail("resident_sft_migration_source_checkpoint_drift")
    for role in ("complete", "adapter", "optimizer"):
        source_binding = source.get(role)
        if not isinstance(source_binding, Mapping):
            _fail("resident_sft_migration_source_binding_drift")
        observed = _binding(
            source_inspected.checkpoint_dir
            / ("complete.json" if role == "complete" else f"{role}.safetensors"),
            max_bytes=(MAX_JSON_BYTES if role == "complete" else None),
        )
        if not _binding_matches(observed, source_binding):
            _fail("resident_sft_migration_source_binding_drift")
    try:
        inspected = inspect_checkpoint(destination_root, expected_bindings=expected_bindings)
    except ResidentSFTBootstrapStateError as exc:
        raise ResidentSFTCheckpointMigrationError(
            "resident_sft_migration_destination_binding_drift"
        ) from exc
    expected_checkpoint = Path(str(destination.get("checkpoint", ""))).resolve(strict=True)
    if inspected.checkpoint_dir != expected_checkpoint:
        _fail("resident_sft_migration_destination_checkpoint_drift")
    for role in ("complete", "adapter", "optimizer"):
        binding = destination.get(role)
        if not isinstance(binding, Mapping):
            _fail("resident_sft_migration_destination_binding_drift")
        observed = _binding(
            inspected.checkpoint_dir
            / ("complete.json" if role == "complete" else f"{role}.safetensors"),
            max_bytes=(MAX_JSON_BYTES if role == "complete" else None),
        )
        if not _binding_matches(observed, binding):
            _fail("resident_sft_migration_destination_binding_drift")
    for role in ("adapter", "optimizer"):
        source_binding = source.get(role)
        destination_binding = destination.get(role)
        if not isinstance(source_binding, Mapping) or not isinstance(destination_binding, Mapping):
            _fail("resident_sft_migration_tensor_identity_drift")
        if not _binding_matches(source_binding, destination_binding):
            _fail("resident_sft_migration_tensor_identity_drift")
    preserved_state = {
        key: value for key, value in inspected.state.items() if key not in BINDING_ROLES
    }
    source_preserved_state = {
        key: value for key, value in source_inspected.state.items() if key not in BINDING_ROLES
    }
    if source_preserved_state != preserved_state or receipt.get("preserved_state_sha256") != _sha(
        canonical_json_bytes(preserved_state)
    ):
        _fail("resident_sft_migration_state_changed")
    return receipt


__all__ = [
    "ALLOWED_CHANGED_SOURCE_ROLES",
    "APPROVED_SEMANTICS_PRESERVING_TRANSITIONS",
    "MIGRATION_SCHEMA",
    "ResidentSFTCheckpointMigrationError",
    "migrate_checkpoint",
    "verify_migration",
]
