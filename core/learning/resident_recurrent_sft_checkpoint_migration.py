"""Exact, source-attested migration of resident recurrent-SFT checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
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
    inspect_checkpoint_generation,
    validate_checkpoint_state,
)
from core.runtime.atomic_writer import (
    atomic_write_bytes,
    durable_replace,
    ensure_private_directory,
)

MIGRATION_SCHEMA: Final = "aura.resident_recurrent_sft_checkpoint_migration.v1"
BUDGET_EXTENSION_SCHEMA: Final = "aura.resident_recurrent_sft_budget_extension.v1"
MAX_JSON_BYTES: Final = 16 * 1024 * 1024
COPY_CHUNK_BYTES: Final = 4 * 1024 * 1024
ALLOWED_CHANGED_SOURCE_ROLES: Final = frozenset(
    {
        "trainer",
        "controller",
        "preparer",
        "model_lane_control",
        "state",
        "objective",
        "objective_policy",
        "specialization_objective",
    }
)
APPROVED_SEMANTICS_PRESERVING_TRANSITIONS: Final = {
    (
        "state",
        "c704219c3613fa2e7e2eba1a5df2d94c838f7129461d075e0d99f65e6ea41c14",
        "5ce23700cfbf4e24dfd2839946e0440a75748f0288d7b09c8697a40a4129ece7",
    ): "historical_generation_verifier_and_monotonic_descendant_proof_v1",
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
    (
        "objective",
        "30ac036e37ad1e77b80a0d64db5ad3f0329a09c7e00e916249511a5d88a7f147",
        "dec1d4f54761c99416e61440a08191157a5f29b3ea8f6274be0380da1e89bef4",
    ): "nested_transition_layer_rematerialization_v2",
    (
        "objective",
        "30ac036e37ad1e77b80a0d64db5ad3f0329a09c7e00e916249511a5d88a7f147",
        "a43a9d9dc10c8a71fa6317c27029bc7bdde686fe013f297f3d746d2b3223a9f3",
    ): "functional_cached_kv_layer_rematerialization_v3",
    (
        "objective",
        "dec1d4f54761c99416e61440a08191157a5f29b3ea8f6274be0380da1e89bef4",
        "a43a9d9dc10c8a71fa6317c27029bc7bdde686fe013f297f3d746d2b3223a9f3",
    ): "functional_cached_kv_layer_rematerialization_v3",
    (
        "objective_policy",
        "5ad187adb9e5c6d0e8e5cfc2abb33c7ee53f7b4ba867a752a188661459f1d321",
        "0cf65e020b6d953309847662dbbed5c196b080efc0816540db2ad0d050a97652",
    ): "cached_generated_rollin_rematerialization_v1",
    (
        "objective_policy",
        "5ad187adb9e5c6d0e8e5cfc2abb33c7ee53f7b4ba867a752a188661459f1d321",
        "2642c39ec7b351c5662d858505430ee7dd5bd8e1e3ee198f6d3794a159737e42",
    ): "exact_cached_lexical_adjoint_v2",
    (
        "objective_policy",
        "0cf65e020b6d953309847662dbbed5c196b080efc0816540db2ad0d050a97652",
        "2642c39ec7b351c5662d858505430ee7dd5bd8e1e3ee198f6d3794a159737e42",
    ): "exact_cached_lexical_adjoint_v2",
    (
        "objective_policy",
        "2642c39ec7b351c5662d858505430ee7dd5bd8e1e3ee198f6d3794a159737e42",
        "0bcb27c3820b0c7f8518ed81925b51586aa301fddb2c672bf6108037e9ba2389",
    ): "exact_decoder_kv_adjoint_v3",
    (
        "model_lane_control",
        "a9e039cecddef5033a82c16910d6435d53a57ed379687d54cd849b1388cd14a5",
        "140737e7d09e455ae312b0e0c4c352541eedac77f5b6484b34c9d68e1ce70ee9",
    ): "transactional_owner_fencing_and_capacity_validation_v1",
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
    validated = validate_authority(value, allow_expired_resume=True)
    if not isinstance(validated, dict):
        _fail("resident_sft_migration_authority_invalid")
    return dict(validated)


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


def _checkpoint_ref(root: Path, value: Any, *, role: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"resident_sft_migration_{role}_checkpoint_drift")
    try:
        relative = Path(value).expanduser().resolve(strict=True).relative_to(root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ResidentSFTCheckpointMigrationError(
            f"resident_sft_migration_{role}_checkpoint_drift"
        ) from exc
    if relative.parts != ("checkpoints", relative.name):
        _fail(f"resident_sft_migration_{role}_checkpoint_drift")
    return relative.as_posix()


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


def _budget_extension(
    source: Mapping[str, Any],
    destination: Mapping[str, Any],
) -> dict[str, Any] | None:
    source_config = dict(source["trainer"])
    destination_config = dict(destination["trainer"])
    source_minutes = source_config.pop("max_minutes", None)
    destination_minutes = destination_config.pop("max_minutes", None)
    if source_config != destination_config:
        return None
    if (
        isinstance(source_minutes, bool)
        or isinstance(destination_minutes, bool)
        or not isinstance(source_minutes, (int, float))
        or not isinstance(destination_minutes, (int, float))
        or not math.isfinite(float(source_minutes))
        or not math.isfinite(float(destination_minutes))
        or float(destination_minutes) <= float(source_minutes)
    ):
        return None
    return {
        "schema": BUDGET_EXTENSION_SCHEMA,
        "source_max_minutes": float(source_minutes),
        "destination_max_minutes": float(destination_minutes),
        "additional_minutes": float(destination_minutes) - float(source_minutes),
        "optimization_config_sha256": _sha(canonical_json_bytes(source_config)),
        "elapsed_training_reset": False,
        "pre_evaluation_protocol_amendment": True,
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
        if role in {"trainer", "controller", "preparer"}:
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
    allow_budget_extension: bool = False,
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

    source_scientific_identity = _identity(source_authority)
    scientific_identity = _identity(destination_authority)
    budget_extension = (
        _budget_extension(source_authority, destination_authority)
        if allow_budget_extension
        else None
    )
    if budget_extension is None:
        if source_scientific_identity != scientific_identity:
            _fail("resident_sft_migration_scientific_identity_changed")
    else:
        comparable_source = dict(source_scientific_identity)
        comparable_destination = dict(scientific_identity)
        comparable_source.pop("trainer_config_sha256", None)
        comparable_destination.pop("trainer_config_sha256", None)
        if comparable_source != comparable_destination:
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
    if inspected.state["terminal"] and budget_extension is None:
        _fail("resident_sft_migration_terminal_checkpoint_forbidden")
    if budget_extension is not None and (
        inspected.state["terminal"] is not True
        or inspected.state.get("halt_reason") != "wall_clock"
        or float(inspected.state["elapsed_training_s"])
        < float(budget_extension["source_max_minutes"]) * 60.0
    ):
        _fail("resident_sft_migration_budget_extension_source_invalid")
    preserved_state = {
        key: value for key, value in inspected.state.items() if key not in BINDING_ROLES
    }
    rebound_state = validate_checkpoint_state(
        {
            **inspected.state,
            **destination_bindings,
            **(
                {"terminal": False, "halt_reason": None}
                if budget_extension is not None
                else {}
            ),
        }
    )
    comparable_rebound = {
        key: value for key, value in rebound_state.items() if key not in BINDING_ROLES
    }
    comparable_preserved = dict(preserved_state)
    if budget_extension is not None:
        for key in ("terminal", "halt_reason"):
            comparable_rebound.pop(key, None)
            comparable_preserved.pop(key, None)
    if comparable_rebound != comparable_preserved:
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
        "source_scientific_identity": source_scientific_identity,
        "budget_extension": budget_extension,
        "trust_policy_identity_sha256": source_trust_identity,
        "changed_source_roles": list(changed_roles),
        "source_transition_attestations": transition_attestations,
        "migration_implementation": _binding(Path(__file__).resolve()),
        "preserved_state_sha256": _sha(canonical_json_bytes(comparable_preserved)),
        "preservation": {
            "adapter_state_reset": False,
            "optimizer_state_reset": False,
            "sample_cursor_reset": False,
            "loss_or_validation_history_reset": False,
            "elapsed_training_reset": False,
            "terminal_latch_reopened": budget_extension is not None,
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
    budget_extension = receipt.get("budget_extension")
    expected_preservation = {
        "adapter_state_reset": False,
        "optimizer_state_reset": False,
        "sample_cursor_reset": False,
        "loss_or_validation_history_reset": False,
        "elapsed_training_reset": False,
        "terminal_latch_reopened": budget_extension is not None,
    }
    if (
        receipt.get("schema") != MIGRATION_SCHEMA
        or claimed != _sha(canonical_json_bytes(material))
        or not isinstance(source, Mapping)
        or not isinstance(destination, Mapping)
        or preservation != expected_preservation
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
    observed_budget_extension = _budget_extension(
        source_authority,
        validated_authority,
    )
    if budget_extension != observed_budget_extension:
        _fail("resident_sft_migration_budget_extension_drift")
    expected_source_identity = _identity(source_authority)
    if (
        receipt.get("source_scientific_identity") != expected_source_identity
        or (budget_extension is None and receipt.get("scientific_identity") != expected_source_identity)
        or receipt.get("trust_policy_identity_sha256")
        != _trust_policy_identity(source_repo_root, source_authority)
        or tuple(changed_roles) != observed_changed_roles
        or receipt.get("source_transition_attestations") != observed_transition_attestations
    ):
        _fail("resident_sft_migration_source_identity_drift")
    try:
        source_inspected = inspect_checkpoint_generation(
            source_root,
            checkpoint=_checkpoint_ref(
                source_root,
                source.get("checkpoint"),
                role="source",
            ),
            expected_bindings=authority_state_bindings(source_authority),
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
        inspected = inspect_checkpoint_generation(
            destination_root,
            checkpoint=_checkpoint_ref(
                destination_root,
                destination.get("checkpoint"),
                role="destination",
            ),
            expected_bindings=expected_bindings,
        )
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
    if budget_extension is not None:
        if (
            source_preserved_state.get("terminal") is not True
            or source_preserved_state.get("halt_reason") != "wall_clock"
            or preserved_state.get("terminal") is not False
            or preserved_state.get("halt_reason") is not None
        ):
            _fail("resident_sft_migration_budget_extension_state_invalid")
        for key in ("terminal", "halt_reason"):
            source_preserved_state.pop(key, None)
            preserved_state.pop(key, None)
    if source_preserved_state != preserved_state or receipt.get("preserved_state_sha256") != _sha(
        canonical_json_bytes(preserved_state)
    ):
        _fail("resident_sft_migration_state_changed")
    return receipt


__all__ = [
    "ALLOWED_CHANGED_SOURCE_ROLES",
    "APPROVED_SEMANTICS_PRESERVING_TRANSITIONS",
    "BUDGET_EXTENSION_SCHEMA",
    "MIGRATION_SCHEMA",
    "ResidentSFTCheckpointMigrationError",
    "migrate_checkpoint",
    "verify_migration",
]
