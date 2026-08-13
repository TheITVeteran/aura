"""Validation contract for byte-identical recurrent shadow source migrations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from tools.unified_intrinsic_resident_identity import canonical_sha256

SOURCE_MIGRATION_SCHEMA: Final = (
    "aura.unified_intrinsic.checkpoint_source_migration.v1"
)


def _sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def shadow_source_migration_errors(
    receipt: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return every identity error without trusting a package-local assertion."""

    errors: list[str] = []
    body = {key: value for key, value in receipt.items() if key != "migration_sha256"}
    source = receipt.get("source")
    destination = receipt.get("destination")
    differences = receipt.get("source_differences")
    identity = checkpoint.get("identity")
    if receipt.get("schema") != SOURCE_MIGRATION_SCHEMA:
        errors.append("schema")
    if receipt.get("state") != "complete":
        errors.append("state")
    if receipt.get("migration_sha256") != canonical_sha256(body):
        errors.append("receipt_commitment")
    if not isinstance(source, Mapping) or not isinstance(destination, Mapping):
        return tuple((*errors, "endpoint_shape"))
    if not isinstance(identity, Mapping):
        return tuple((*errors, "checkpoint_identity_shape"))
    if not isinstance(differences, Mapping) or not differences:
        errors.append("source_change_set")
    elif not any(
        isinstance(path, str) and path.startswith("core/learning/")
        for path in differences
    ):
        errors.append("learning_mechanics_change_missing")
    else:
        for path, change in differences.items():
            if (
                not isinstance(path, str)
                or not isinstance(change, Mapping)
                or set(change) != {"source", "target"}
                or not _sha(change.get("source"))
                or not _sha(change.get("target"))
                or change.get("source") == change.get("target")
            ):
                errors.append("source_change_shape")
                break
    checkpoint_sha = checkpoint.get("checkpoint_sha256")
    identity_sha = identity.get("identity_sha256")
    if (
        not _sha(checkpoint_sha)
        or source.get("checkpoint_sha256") != checkpoint_sha
        or destination.get("checkpoint_sha256") != checkpoint_sha
        or manifest.get("checkpoint_sha256") != checkpoint_sha
    ):
        errors.append("checkpoint_continuity")
    if (
        not _sha(identity_sha)
        or destination.get("identity_sha256") != identity_sha
        or manifest.get("checkpoint_identity_sha256") != identity_sha
    ):
        errors.append("destination_identity")
    if (
        not _sha(source.get("identity_sha256"))
        or source.get("identity_sha256") == identity_sha
    ):
        errors.append("source_identity")
    if (
        receipt.get("payload_byte_identical") is not True
        or receipt.get("optimizer_and_bundle_bytes_preserved") is not True
        or receipt.get("history_preserved") is not True
        or receipt.get("training_state_preserved") is not True
    ):
        errors.append("state_preservation")
    if (
        manifest.get("source_migration_sha256") != receipt.get("migration_sha256")
        or manifest.get("source_migration_config_sha256")
        != destination.get("config_sha256")
        or manifest.get("source_migration_campaign_id")
        != destination.get("campaign_id")
        or manifest.get("source_commit") != receipt.get("target_source_commit")
    ):
        errors.append("manifest_binding")
    if not _sha(identity.get("source_migration_controller_sha256")):
        errors.append("controller_identity")
    return tuple(dict.fromkeys(errors))


__all__ = ["SOURCE_MIGRATION_SCHEMA", "shadow_source_migration_errors"]
