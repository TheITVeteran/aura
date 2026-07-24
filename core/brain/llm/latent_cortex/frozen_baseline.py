"""SPARK-004 frozen baseline bundle: seal the pre-treatment surface.

The Spark training admission (SPARK-069) and the falsification matrix
(SPARK-070/071) compare a changed treatment against "the baseline".  That
comparison is only meaningful if the baseline was bound before the
treatment changed: the resident checkpoint, tokenizer/config behavior
bundle, attached adapters, decoding contract, task-generator identity,
control manifests, resource envelope, randomization policy, and the
current vanilla/RLC measurements.  This module builds and independently
verifies one immutable, hash-bound, Ed25519-signable bundle over exactly
that surface.

Small artifacts (control manifests, measurement receipts) are copied into
the bundle so it stays self-contained even when the originals were never
tracked.  The multi-gigabyte checkpoint is bound by full-content
fingerprint, not copied; ``verify_frozen_baseline_model`` re-hashes it on
demand.  Tracked source files are bound by commit plus content hash;
``verify_frozen_baseline_sources`` re-hashes them from a checkout.  The
bundle never chooses its own trust anchor: signature verification always
takes the public key from the caller.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Never

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_launch_bundle import (
    read_canonical_json,
    sha256_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import load_ed25519_public_key
from core.brain.llm.latent_cortex.execution_spec import RLC_EXECUTION_SPEC_SCHEMA
from core.runtime.atomic_writer import atomic_write_bytes_if_absent
from core.runtime.file_read_gateway import read_stable_bytes

FROZEN_BASELINE_SCHEMA = "aura.latent_cortex.spark_frozen_baseline.v1"
FROZEN_BASELINE_FILE = "frozen_baseline.json"
FROZEN_BASELINE_SIGNATURE_SCHEMA = (
    "aura.latent_cortex.spark_frozen_baseline_signature.v1"
)
FROZEN_BASELINE_SIGNATURE_FILE = "frozen_baseline_signature.json"

CONTROL_MANIFEST_DIR = "control_manifests"
MEASUREMENT_DIR = "measurements"
MEASUREMENT_ROLES = ("vanilla", "rlc", "paired")

_MATERIAL_KEYS = {
    "schema",
    "baseline_id",
    "purpose",
    "frozen_at_unix",
    "git_commit",
    "worktree_clean",
    "environment",
    "model",
    "adapters",
    "decoding",
    "task_generators",
    "control_manifests",
    "resource_envelope",
    "randomization",
    "measurements",
}
_CERTIFICATE_KEYS = _MATERIAL_KEYS | {"certificate_sha256"}
_ENVIRONMENT_KEYS = {
    "runtime",
    "observed_physical_memory_bytes",
    "observed_cpu_count",
}
_RUNTIME_KEYS = {
    "python",
    "platform_system",
    "platform_release",
    "platform_machine",
    "dependencies",
    "identity_sha256",
}
_MODEL_KEYS = {"path", "checkpoint", "behavior_bundle"}
_CHECKPOINT_KEYS = {"fingerprint", "method", "files"}
_BEHAVIOR_KEYS = {"bundle_sha256", "file_count", "files"}
_REQUIRED_BEHAVIOR_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json"}
_ADAPTER_KEYS = {"personality", "attached_at_baseline"}
_PERSONALITY_KEYS = {"present", "bundle_sha256", "file_count", "files"}
_DECODING_KEYS = {"execution_spec", "execution_spec_sha256", "sampling"}
_SAMPLING_KEYS = {"decode_max_tokens", "decode_temperature", "decode_top_p"}
_TASK_GENERATOR_KEYS = {
    "registry_version",
    "frontier_domains",
    "excluded_training_families",
    "recurrence_training_families",
    "sources",
}
_SOURCE_KEYS = {"repo_path", "sha256", "size_bytes"}
_CONTROL_MANIFEST_KEYS = {"name", "bundle_path", "source_path", "sha256", "size_bytes"}
_RESOURCE_KEYS = {"declared"}
_RANDOMIZATION_KEYS = {
    "training_seed",
    "slot_seed",
    "eval_seed",
    "seed_policy",
    "sources",
}
_MEASUREMENT_KEYS = {
    "name",
    "role",
    "bundle_path",
    "source_path",
    "sha256",
    "size_bytes",
    "summary",
}
_SIGNATURE_KEYS = {
    "schema",
    "algorithm",
    "key_id",
    "certificate_sha256",
    "signed_payload_sha256",
    "signature_b64",
}
_FILE_KEYS = {"path", "sha256", "size_bytes"}
_MAX_COPY_BYTES = 256 * 1024 * 1024
_MAX_BUNDLE_FILES = 256


class FrozenBaselineError(ValueError):
    """Stable fail-closed frozen-baseline error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise FrozenBaselineError(code)


def _sha256_value(value: Any) -> str:
    try:
        return sha256_bytes(canonical_json_bytes(value))
    except (TypeError, ValueError, OverflowError, RecursionError):
        _fail("frozen_baseline_value_not_canonical")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _identifier(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 200
    ):
        _fail(f"{role}_invalid")
    return value


def _positive_int(value: Any, *, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{role}_invalid")
    return value


def _non_negative_int(value: Any, *, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{role}_invalid")
    return value


def _finite_number(value: Any, *, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{role}_invalid")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{role}_invalid")
    return number


def _relative_bundle_path(value: Any, *, role: str, prefix: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail(f"{role}_path_invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or len(path.parts) != 2
        or path.parts[0] != prefix
    ):
        _fail(f"{role}_path_invalid")
    return value


def _repo_relative_path(value: Any, *, role: str) -> str:
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


def _string_list(value: Any, *, role: str, allow_empty: bool) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        _fail(f"{role}_invalid")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            _fail(f"{role}_invalid")
        items.append(item)
    if len(set(items)) != len(items):
        _fail(f"{role}_invalid")
    return items


def _validated_file_records(value: Any, *, role: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        _fail(f"{role}_files_invalid")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in value:
        if not isinstance(record, Mapping) or set(record) != _FILE_KEYS:
            _fail(f"{role}_files_invalid")
        path = record.get("path")
        if not isinstance(path, str) or not path or path in seen:
            _fail(f"{role}_files_invalid")
        if not _is_sha256(record.get("sha256")):
            _fail(f"{role}_files_invalid")
        _non_negative_int(record.get("size_bytes"), role=f"{role}_file_size")
        seen.add(path)
        records.append(dict(record))
    return records


def _validated_sources(value: Any, *, role: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        _fail(f"{role}_sources_invalid")
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in value:
        if not isinstance(record, Mapping) or set(record) != _SOURCE_KEYS:
            _fail(f"{role}_sources_invalid")
        repo_path = _repo_relative_path(record.get("repo_path"), role=f"{role}_source")
        if repo_path in seen:
            _fail(f"{role}_sources_invalid")
        if not _is_sha256(record.get("sha256")):
            _fail(f"{role}_sources_invalid")
        _positive_int(record.get("size_bytes"), role=f"{role}_source_size")
        seen.add(repo_path)
        sources.append(dict(record))
    return sources


def _validated_environment(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ENVIRONMENT_KEYS:
        _fail("frozen_baseline_environment_invalid")
    runtime = value.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != _RUNTIME_KEYS:
        _fail("frozen_baseline_environment_runtime_invalid")
    dependencies = runtime.get("dependencies")
    if (
        not isinstance(dependencies, Mapping)
        or not dependencies
        or any(
            not isinstance(name, str) or not isinstance(version, str) or not version
            for name, version in dependencies.items()
        )
    ):
        _fail("frozen_baseline_environment_runtime_invalid")
    for field in ("python", "platform_system", "platform_release", "platform_machine"):
        _identifier(runtime.get(field), role="frozen_baseline_environment_runtime")
    body = {key: runtime[key] for key in _RUNTIME_KEYS - {"identity_sha256"}}
    if runtime.get("identity_sha256") != _sha256_value(body):
        _fail("frozen_baseline_environment_identity_mismatch")
    _positive_int(
        value.get("observed_physical_memory_bytes"),
        role="frozen_baseline_environment_memory",
    )
    _positive_int(
        value.get("observed_cpu_count"), role="frozen_baseline_environment_cpu"
    )
    return {
        "runtime": dict(runtime),
        "observed_physical_memory_bytes": value["observed_physical_memory_bytes"],
        "observed_cpu_count": value["observed_cpu_count"],
    }


def _validated_model(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _MODEL_KEYS:
        _fail("frozen_baseline_model_invalid")
    path = value.get("path")
    if not isinstance(path, str) or not path or not path.startswith("/"):
        _fail("frozen_baseline_model_path_invalid")
    checkpoint = value.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != _CHECKPOINT_KEYS
        or not _is_sha256(checkpoint.get("fingerprint"))
        or checkpoint.get("method") != "sha256"
    ):
        _fail("frozen_baseline_model_checkpoint_invalid")
    _positive_int(checkpoint.get("files"), role="frozen_baseline_model_checkpoint_files")
    behavior = value.get("behavior_bundle")
    if not isinstance(behavior, Mapping) or set(behavior) != _BEHAVIOR_KEYS:
        _fail("frozen_baseline_model_behavior_invalid")
    files = _validated_file_records(
        behavior.get("files"), role="frozen_baseline_model_behavior"
    )
    if (
        behavior.get("file_count") != len(files)
        or not files
        or behavior.get("bundle_sha256") != _sha256_value(files)
        or not _REQUIRED_BEHAVIOR_FILES.issubset(
            record["path"] for record in files
        )
    ):
        _fail("frozen_baseline_model_behavior_invalid")
    return {
        "path": path,
        "checkpoint": dict(checkpoint),
        "behavior_bundle": {
            "bundle_sha256": behavior["bundle_sha256"],
            "file_count": len(files),
            "files": files,
        },
    }


def _validated_adapters(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ADAPTER_KEYS:
        _fail("frozen_baseline_adapters_invalid")
    personality = value.get("personality")
    if not isinstance(personality, Mapping) or set(personality) != _PERSONALITY_KEYS:
        _fail("frozen_baseline_adapters_personality_invalid")
    present = personality.get("present")
    files = _validated_file_records(
        personality.get("files"), role="frozen_baseline_adapters_personality"
    )
    if present is True:
        if (
            not files
            or personality.get("file_count") != len(files)
            or personality.get("bundle_sha256") != _sha256_value(files)
        ):
            _fail("frozen_baseline_adapters_personality_invalid")
    elif present is False:
        if (
            files
            or personality.get("file_count") != 0
            or personality.get("bundle_sha256") != ""
        ):
            _fail("frozen_baseline_adapters_personality_invalid")
    else:
        _fail("frozen_baseline_adapters_personality_invalid")
    attached = _string_list(
        value.get("attached_at_baseline"),
        role="frozen_baseline_adapters_attached",
        allow_empty=True,
    )
    return {
        "personality": dict(personality),
        "attached_at_baseline": attached,
    }


def _validated_decoding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _DECODING_KEYS:
        _fail("frozen_baseline_decoding_invalid")
    spec = value.get("execution_spec")
    if not isinstance(spec, Mapping) or spec.get("schema") != RLC_EXECUTION_SPEC_SCHEMA:
        _fail("frozen_baseline_decoding_spec_invalid")
    if value.get("execution_spec_sha256") != _sha256_value(dict(spec)):
        _fail("frozen_baseline_decoding_spec_digest_mismatch")
    sampling = value.get("sampling")
    if not isinstance(sampling, Mapping) or set(sampling) != _SAMPLING_KEYS:
        _fail("frozen_baseline_decoding_sampling_invalid")
    _positive_int(
        sampling.get("decode_max_tokens"),
        role="frozen_baseline_decoding_max_tokens",
    )
    temperature = _finite_number(
        sampling.get("decode_temperature"),
        role="frozen_baseline_decoding_temperature",
    )
    top_p = _finite_number(
        sampling.get("decode_top_p"), role="frozen_baseline_decoding_top_p"
    )
    if temperature < 0.0 or not 0.0 < top_p <= 1.0:
        _fail("frozen_baseline_decoding_sampling_invalid")
    return {
        "execution_spec": dict(spec),
        "execution_spec_sha256": value["execution_spec_sha256"],
        "sampling": dict(sampling),
    }


def _validated_task_generators(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TASK_GENERATOR_KEYS:
        _fail("frozen_baseline_task_generators_invalid")
    return {
        "registry_version": _identifier(
            value.get("registry_version"),
            role="frozen_baseline_task_generators_registry",
        ),
        "frontier_domains": _string_list(
            value.get("frontier_domains"),
            role="frozen_baseline_task_generators_domains",
            allow_empty=False,
        ),
        "excluded_training_families": _string_list(
            value.get("excluded_training_families"),
            role="frozen_baseline_task_generators_excluded",
            allow_empty=True,
        ),
        "recurrence_training_families": _string_list(
            value.get("recurrence_training_families"),
            role="frozen_baseline_task_generators_families",
            allow_empty=False,
        ),
        "sources": _validated_sources(
            value.get("sources"), role="frozen_baseline_task_generators"
        ),
    }


def _validated_control_manifests(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        _fail("frozen_baseline_control_manifests_invalid")
    manifests: list[dict[str, Any]] = []
    names: set[str] = set()
    paths: set[str] = set()
    for record in value:
        if not isinstance(record, Mapping) or set(record) != _CONTROL_MANIFEST_KEYS:
            _fail("frozen_baseline_control_manifests_invalid")
        name = _identifier(
            record.get("name"), role="frozen_baseline_control_manifest_name"
        )
        bundle_path = _relative_bundle_path(
            record.get("bundle_path"),
            role="frozen_baseline_control_manifest",
            prefix=CONTROL_MANIFEST_DIR,
        )
        source_path = record.get("source_path")
        if not isinstance(source_path, str):
            _fail("frozen_baseline_control_manifests_invalid")
        if not _is_sha256(record.get("sha256")):
            _fail("frozen_baseline_control_manifests_invalid")
        _positive_int(
            record.get("size_bytes"), role="frozen_baseline_control_manifest_size"
        )
        if name in names or bundle_path in paths:
            _fail("frozen_baseline_control_manifests_invalid")
        names.add(name)
        paths.add(bundle_path)
        manifests.append(dict(record))
    return manifests


def _validated_resource_envelope(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RESOURCE_KEYS:
        _fail("frozen_baseline_resource_envelope_invalid")
    declared = value.get("declared")
    if not isinstance(declared, Mapping) or not declared:
        _fail("frozen_baseline_resource_envelope_invalid")
    for key, item in declared.items():
        if not isinstance(key, str) or not key:
            _fail("frozen_baseline_resource_envelope_invalid")
        if isinstance(item, bool) or isinstance(item, str):
            continue
        if isinstance(item, (int, float)):
            _finite_number(item, role="frozen_baseline_resource_envelope_value")
            continue
        _fail("frozen_baseline_resource_envelope_invalid")
    return {"declared": dict(declared)}


def _validated_randomization(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RANDOMIZATION_KEYS:
        _fail("frozen_baseline_randomization_invalid")
    return {
        "training_seed": _positive_int(
            value.get("training_seed"), role="frozen_baseline_randomization_training"
        ),
        "slot_seed": _non_negative_int(
            value.get("slot_seed"), role="frozen_baseline_randomization_slot"
        ),
        "eval_seed": _positive_int(
            value.get("eval_seed"), role="frozen_baseline_randomization_eval"
        ),
        "seed_policy": _identifier(
            value.get("seed_policy"), role="frozen_baseline_randomization_policy"
        ),
        "sources": _validated_sources(
            value.get("sources"), role="frozen_baseline_randomization"
        ),
    }


def _validated_measurements(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        _fail("frozen_baseline_measurements_invalid")
    measurements: list[dict[str, Any]] = []
    names: set[str] = set()
    paths: set[str] = set()
    roles: set[str] = set()
    for record in value:
        if not isinstance(record, Mapping) or set(record) != _MEASUREMENT_KEYS:
            _fail("frozen_baseline_measurements_invalid")
        name = _identifier(record.get("name"), role="frozen_baseline_measurement_name")
        role = record.get("role")
        if role not in MEASUREMENT_ROLES:
            _fail("frozen_baseline_measurement_role_invalid")
        bundle_path = _relative_bundle_path(
            record.get("bundle_path"),
            role="frozen_baseline_measurement",
            prefix=MEASUREMENT_DIR,
        )
        source_path = record.get("source_path")
        if not isinstance(source_path, str):
            _fail("frozen_baseline_measurements_invalid")
        if not _is_sha256(record.get("sha256")):
            _fail("frozen_baseline_measurements_invalid")
        _positive_int(record.get("size_bytes"), role="frozen_baseline_measurement_size")
        summary = record.get("summary")
        if not isinstance(summary, Mapping):
            _fail("frozen_baseline_measurements_invalid")
        if name in names or bundle_path in paths:
            _fail("frozen_baseline_measurements_invalid")
        names.add(name)
        paths.add(bundle_path)
        roles.add(role)
        measurements.append(dict(record))
    if "paired" not in roles and not {"vanilla", "rlc"}.issubset(roles):
        _fail("frozen_baseline_measurement_coverage_invalid")
    return measurements


def _validated_material(material: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(material, Mapping) or set(material) != _MATERIAL_KEYS:
        _fail("frozen_baseline_material_schema_invalid")
    if material.get("schema") != FROZEN_BASELINE_SCHEMA:
        _fail("frozen_baseline_material_schema_invalid")
    if material.get("worktree_clean") is not True:
        _fail("frozen_baseline_worktree_dirty")
    if not _is_git_commit(material.get("git_commit")):
        _fail("frozen_baseline_git_commit_invalid")
    validated = {
        "schema": FROZEN_BASELINE_SCHEMA,
        "baseline_id": _identifier(
            material.get("baseline_id"), role="frozen_baseline_id"
        ),
        "purpose": _identifier(material.get("purpose"), role="frozen_baseline_purpose"),
        "frozen_at_unix": _positive_int(
            material.get("frozen_at_unix"), role="frozen_baseline_frozen_at"
        ),
        "git_commit": material["git_commit"],
        "worktree_clean": True,
        "environment": _validated_environment(material.get("environment")),
        "model": _validated_model(material.get("model")),
        "adapters": _validated_adapters(material.get("adapters")),
        "decoding": _validated_decoding(material.get("decoding")),
        "task_generators": _validated_task_generators(
            material.get("task_generators")
        ),
        "control_manifests": _validated_control_manifests(
            material.get("control_manifests")
        ),
        "resource_envelope": _validated_resource_envelope(
            material.get("resource_envelope")
        ),
        "randomization": _validated_randomization(material.get("randomization")),
        "measurements": _validated_measurements(material.get("measurements")),
    }
    return validated


def build_frozen_baseline_certificate(
    material: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate assembled baseline material and seal it with a self-digest."""

    validated = _validated_material(material)
    return {**validated, "certificate_sha256": _sha256_value(validated)}


def validate_frozen_baseline_certificate(
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly re-validate a complete certificate, including its self-digest."""

    if not isinstance(certificate, Mapping) or set(certificate) != _CERTIFICATE_KEYS:
        _fail("frozen_baseline_certificate_schema_invalid")
    material = {
        key: value
        for key, value in certificate.items()
        if key != "certificate_sha256"
    }
    validated = _validated_material(material)
    if certificate.get("certificate_sha256") != _sha256_value(validated):
        _fail("frozen_baseline_certificate_digest_mismatch")
    return {**validated, "certificate_sha256": certificate["certificate_sha256"]}


def planned_bundle_files(
    certificate: Mapping[str, Any], *, include_signature: bool
) -> set[str]:
    """Return the exact relative file set a published bundle must contain."""

    planned = {FROZEN_BASELINE_FILE}
    for record in certificate["control_manifests"]:
        planned.add(record["bundle_path"])
    for record in certificate["measurements"]:
        planned.add(record["bundle_path"])
    if include_signature:
        planned.add(FROZEN_BASELINE_SIGNATURE_FILE)
    if len(planned) > _MAX_BUNDLE_FILES:
        _fail("frozen_baseline_bundle_too_large")
    return planned


def sign_frozen_baseline_certificate(
    certificate: Mapping[str, Any], *, private_key: Any
) -> dict[str, Any]:
    """Produce a detached Ed25519 signature over the exact certificate bytes."""

    validated = validate_frozen_baseline_certificate(certificate)
    payload = canonical_json_bytes(validated)
    from cryptography.hazmat.primitives import serialization

    public_raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return {
        "schema": FROZEN_BASELINE_SIGNATURE_SCHEMA,
        "algorithm": "Ed25519",
        "key_id": hashlib.sha256(public_raw).hexdigest(),
        "certificate_sha256": validated["certificate_sha256"],
        "signed_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "signature_b64": base64.b64encode(private_key.sign(payload)).decode("ascii"),
    }


def verify_frozen_baseline_signature(
    certificate: Mapping[str, Any],
    signature: Mapping[str, Any],
    *,
    trusted_public_key_pem: bytes,
) -> dict[str, Any]:
    """Verify a detached signature against a caller-supplied trust anchor."""

    validated = validate_frozen_baseline_certificate(certificate)
    if not isinstance(signature, Mapping) or set(signature) != _SIGNATURE_KEYS:
        _fail("frozen_baseline_signature_schema_invalid")
    if (
        signature.get("schema") != FROZEN_BASELINE_SIGNATURE_SCHEMA
        or signature.get("algorithm") != "Ed25519"
    ):
        _fail("frozen_baseline_signature_schema_invalid")
    payload = canonical_json_bytes(validated)
    if (
        signature.get("certificate_sha256") != validated["certificate_sha256"]
        or signature.get("signed_payload_sha256")
        != hashlib.sha256(payload).hexdigest()
    ):
        _fail("frozen_baseline_signature_payload_mismatch")
    public_key, public_raw, key_id = load_ed25519_public_key(
        trusted_public_key_pem, role="frozen_baseline"
    )
    if signature.get("key_id") != key_id:
        _fail("frozen_baseline_signature_key_mismatch")
    try:
        raw_signature = base64.b64decode(
            str(signature.get("signature_b64")), validate=True
        )
    except (TypeError, ValueError, binascii.Error):
        _fail("frozen_baseline_signature_invalid")
    from cryptography.exceptions import InvalidSignature

    try:
        public_key.verify(raw_signature, payload)
    except InvalidSignature:
        _fail("frozen_baseline_signature_invalid")
    return dict(signature)


def _copy_bindings(certificate: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for record in certificate["control_manifests"]:
        bindings[record["bundle_path"]] = record
    for record in certificate["measurements"]:
        bindings[record["bundle_path"]] = record
    return bindings


def publish_frozen_baseline_bundle(
    root: Path,
    *,
    certificate: Mapping[str, Any],
    file_payloads: Mapping[str, bytes],
    signature: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one immutable bundle directory and verify it before returning."""

    validated = validate_frozen_baseline_certificate(certificate)
    bindings = _copy_bindings(validated)
    if set(file_payloads) != set(bindings):
        _fail("frozen_baseline_payload_set_mismatch")
    for relative, payload in file_payloads.items():
        binding = bindings[relative]
        if (
            not isinstance(payload, bytes)
            or len(payload) != binding["size_bytes"]
            or sha256_bytes(payload) != binding["sha256"]
            or len(payload) > _MAX_COPY_BYTES
        ):
            _fail("frozen_baseline_payload_binding_mismatch")
    if signature is not None:
        # Publishing never invents trust: the signature was produced by the
        # caller's key and is re-verified end-to-end by verify_frozen_baseline
        # with an independently supplied public key.
        if not isinstance(signature, Mapping) or set(signature) != _SIGNATURE_KEYS:
            _fail("frozen_baseline_signature_schema_invalid")
        if signature.get("certificate_sha256") != validated["certificate_sha256"]:
            _fail("frozen_baseline_signature_payload_mismatch")

    target = root.expanduser()
    if target.exists() or target.is_symlink():
        _fail("frozen_baseline_root_exists")
    directories = {target}
    planned = planned_bundle_files(validated, include_signature=signature is not None)
    for relative in planned:
        directories.add(target / PurePosixPath(relative).parent)
    for directory in sorted(directories, key=lambda item: len(item.parts)):
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)

    documents: dict[str, bytes] = {
        FROZEN_BASELINE_FILE: canonical_json_bytes(validated) + b"\n",
        **{relative: payload for relative, payload in file_payloads.items()},
    }
    if signature is not None:
        documents[FROZEN_BASELINE_SIGNATURE_FILE] = (
            canonical_json_bytes(dict(signature)) + b"\n"
        )
    for relative in sorted(documents):
        path = target / PurePosixPath(relative)
        if not atomic_write_bytes_if_absent(path, documents[relative], mode=0o600):
            _fail("frozen_baseline_publish_conflict")
        os.chmod(path, 0o444)
    for directory in sorted(
        directories, key=lambda item: len(item.parts), reverse=True
    ):
        os.chmod(directory, 0o555)
    return verify_frozen_baseline(target)


def verify_frozen_baseline(
    root: Path,
    *,
    trusted_public_key_pem: bytes | None = None,
) -> dict[str, Any]:
    """Independently verify one published bundle from its bytes alone."""

    supplied_root = root.expanduser()
    if supplied_root.is_symlink():
        _fail("frozen_baseline_root_symlink_rejected")
    try:
        resolved_root = supplied_root.resolve(strict=True)
        root_metadata = resolved_root.lstat()
    except OSError:
        _fail("frozen_baseline_root_unavailable")
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != os.geteuid()
        or root_metadata.st_mode & 0o222
    ):
        _fail("frozen_baseline_root_invalid")

    certificate = validate_frozen_baseline_certificate(
        read_canonical_json(
            resolved_root / FROZEN_BASELINE_FILE, role="frozen_baseline"
        )
    )

    observed: set[str] = set()
    for path in resolved_root.rglob("*"):
        metadata = path.lstat()
        relative = path.relative_to(resolved_root).as_posix()
        if stat.S_ISLNK(metadata.st_mode):
            _fail("frozen_baseline_symlink_rejected")
        if stat.S_ISDIR(metadata.st_mode):
            if metadata.st_mode & 0o222 or metadata.st_uid != os.geteuid():
                _fail("frozen_baseline_directory_writable")
            continue
        if not stat.S_ISREG(metadata.st_mode):
            _fail("frozen_baseline_special_file_rejected")
        if metadata.st_mode & 0o222 or metadata.st_uid != os.geteuid():
            _fail("frozen_baseline_file_writable")
        observed.add(relative)

    signature_present = FROZEN_BASELINE_SIGNATURE_FILE in observed
    planned = planned_bundle_files(certificate, include_signature=signature_present)
    if observed != planned:
        _fail("frozen_baseline_artifact_set_mismatch")

    for relative, binding in _copy_bindings(certificate).items():
        try:
            payload = read_stable_bytes(
                resolved_root / PurePosixPath(relative), max_bytes=_MAX_COPY_BYTES
            )
        except (OSError, ValueError):
            _fail("frozen_baseline_copy_unavailable")
        if (
            len(payload) != binding["size_bytes"]
            or sha256_bytes(payload) != binding["sha256"]
        ):
            _fail("frozen_baseline_copy_binding_mismatch")

    if trusted_public_key_pem is not None and not signature_present:
        _fail("frozen_baseline_signature_missing")
    if signature_present:
        signature = read_canonical_json(
            resolved_root / FROZEN_BASELINE_SIGNATURE_FILE,
            role="frozen_baseline_signature",
        )
        if not isinstance(signature, Mapping) or set(signature) != _SIGNATURE_KEYS:
            _fail("frozen_baseline_signature_schema_invalid")
        if trusted_public_key_pem is not None:
            verify_frozen_baseline_signature(
                certificate,
                signature,
                trusted_public_key_pem=trusted_public_key_pem,
            )
    return certificate


def verify_frozen_baseline_model(
    certificate: Mapping[str, Any], *, model_root: Path | None = None
) -> dict[str, Any]:
    """Re-hash the full checkpoint and behavior bundle against the freeze."""

    from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
        full_weight_checkpoint_identity,
        model_behavior_bundle_identity,
    )

    validated = validate_frozen_baseline_certificate(certificate)
    root = model_root if model_root is not None else Path(validated["model"]["path"])
    checkpoint = full_weight_checkpoint_identity(root)
    if checkpoint != validated["model"]["checkpoint"]:
        _fail("frozen_baseline_model_checkpoint_drift")
    behavior = model_behavior_bundle_identity(root)
    if behavior != validated["model"]["behavior_bundle"]:
        _fail("frozen_baseline_model_behavior_drift")
    return {"checkpoint": checkpoint, "behavior_bundle": behavior}


def verify_frozen_baseline_sources(
    certificate: Mapping[str, Any], *, repo_root: Path
) -> list[dict[str, Any]]:
    """Re-hash every bound source file from a repository checkout."""

    validated = validate_frozen_baseline_certificate(certificate)
    try:
        resolved_repo = repo_root.expanduser().resolve(strict=True)
    except OSError:
        _fail("frozen_baseline_repo_unavailable")
    checked: list[dict[str, Any]] = []
    sources = list(validated["task_generators"]["sources"]) + list(
        validated["randomization"]["sources"]
    )
    for record in sources:
        path = resolved_repo / PurePosixPath(record["repo_path"])
        try:
            payload = read_stable_bytes(path, max_bytes=_MAX_COPY_BYTES)
        except (OSError, ValueError):
            _fail("frozen_baseline_source_unavailable")
        if (
            len(payload) != record["size_bytes"]
            or sha256_bytes(payload) != record["sha256"]
        ):
            _fail("frozen_baseline_source_drift")
        checked.append(dict(record))
    return checked


__all__ = [
    "CONTROL_MANIFEST_DIR",
    "FROZEN_BASELINE_FILE",
    "FROZEN_BASELINE_SCHEMA",
    "FROZEN_BASELINE_SIGNATURE_FILE",
    "FROZEN_BASELINE_SIGNATURE_SCHEMA",
    "MEASUREMENT_DIR",
    "MEASUREMENT_ROLES",
    "FrozenBaselineError",
    "build_frozen_baseline_certificate",
    "planned_bundle_files",
    "publish_frozen_baseline_bundle",
    "sign_frozen_baseline_certificate",
    "validate_frozen_baseline_certificate",
    "verify_frozen_baseline",
    "verify_frozen_baseline_model",
    "verify_frozen_baseline_signature",
    "verify_frozen_baseline_sources",
]
