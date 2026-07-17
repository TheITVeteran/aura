"""Strict raw-artifact verification for Recursive Latent Cortex evidence.

The frontier certificate grades values embedded in an evidence bundle.  This
module proves that those values came from a complete, immutable set of raw
task, output, scorer, verifier, compute, and runtime-receipt records.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from core.brain.frontier_evidence_v5 import canonical_json_bytes
from core.brain.llm.latent_cortex.frontier_certification import canonical_sha256
from core.runtime.file_read_gateway import (
    StableFileReadError,
    open_stable_readonly_binary,
    read_stable_bytes,
)

RAW_ARTIFACT_MANIFEST_SCHEMA = "aura.latent_cortex.raw_artifact_manifest.v1"
ARTIFACT_VERIFICATION_SCHEMA = "aura.latent_cortex.raw_artifact_verification.v1"
TRIAL_VERIFIER_RECEIPT_SCHEMA = "aura.latent_cortex.trial_verifier_receipt.v1"

MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_STORE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_STORE_BYTES = 1024 * 1024 * 1024
MAX_RECORD_BYTES = 16 * 1024 * 1024
MAX_RECORDS = 1_000_000

_BLOB_STORES = {
    "task_payloads": "task_payload_sha256",
    "treatment_outputs": "treatment_output_sha256",
    "control_outputs": "control_output_sha256",
    "scorer_configs": "scorer_config_sha256",
    "verifier_receipts": "verifier_receipt_sha256",
}
_STRUCTURED_STORE = "structured_receipts"
# External-frontier comparisons must ship the raw provider responses so the
# per-trial provider_receipt_sha256 binding is independently recomputable —
# without them, "the frontier model said X" would be an unverifiable claim.
_PROVIDER_STORE = "provider_receipts"
_EXTERNAL_COMPARISON_KIND = "resident_32b_vs_external_frontier"
_REQUIRED_STORES = frozenset((*_BLOB_STORES, _STRUCTURED_STORE))
_STRUCTURED_FIELDS = (
    "treatment_receipt",
    "control_receipt",
    "treatment_compute",
    "control_compute",
)
_SHA256_LENGTH = 64


class FrontierArtifactError(ValueError):
    """Stable fail-closed reason raised for an invalid evidence package."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    error = FrontierArtifactError(code)
    raise error


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_int(raw: str) -> int:
    digits = raw.removeprefix("-")
    if not digits or len(digits) > 64:
        _fail("json_integer_out_of_bounds")
    return int(raw)


def _strict_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        _fail("json_non_finite_number")
    return value


def strict_json_loads(raw: bytes, *, role: str) -> Any:
    """Decode UTF-8 JSON while rejecting duplicate keys and numeric traps."""

    if len(raw) > MAX_JSON_BYTES:
        _fail(f"{role}_too_large")
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

    def reject_constant(_value: str) -> None:
        _fail(f"{role}_non_finite_number")

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_int=_strict_int,
            parse_float=_strict_float,
            parse_constant=reject_constant,
        )
    except FrontierArtifactError:
        raise
    except (TypeError, ValueError, RecursionError, OverflowError):
        _fail(f"{role}_json_invalid")


def read_strict_json_file(
    path: Path,
    *,
    role: str,
    max_bytes: int = MAX_JSON_BYTES,
) -> tuple[Any, bytes]:
    """Read one non-symlink regular file and decode strict JSON."""

    requested = Path(path).expanduser()
    try:
        raw = read_stable_bytes(requested, max_bytes=max_bytes)
    except StableFileReadError as exc:
        if exc.code == "symlink_rejected":
            _fail(f"{role}_symlink_rejected")
        if exc.code in {
            "size_exceeds_bound",
            "bounded_read_length_mismatch",
            "grew_beyond_bound",
        }:
            _fail(f"{role}_too_large")
        if exc.code == "changed_during_read":
            _fail(f"{role}_changed_during_read")
        if exc.code == "not_regular_file":
            _fail(f"{role}_not_regular_file")
        _fail(f"{role}_unreadable")
    except OSError:
        _fail(f"{role}_unreadable")
    return strict_json_loads(raw, role=role), raw


def _validate_descriptor(raw: Any, *, store: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {
        "path",
        "sha256",
        "size_bytes",
        "record_count",
    }:
        _fail(f"{store}_descriptor_invalid")
    path = raw.get("path")
    if not isinstance(path, str) or not path or path != path.strip() or "\\" in path:
        _fail(f"{store}_path_invalid")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail(f"{store}_path_invalid")
    if not pure.parts or pure.parts[0] != "raw" or str(pure) != path:
        _fail(f"{store}_path_outside_raw")
    if not _is_sha256(raw.get("sha256")):
        _fail(f"{store}_sha256_invalid")
    size = raw.get("size_bytes")
    count = raw.get("record_count")
    if type(size) is not int or not 0 < size <= MAX_STORE_BYTES:
        _fail(f"{store}_size_invalid")
    if type(count) is not int or not 0 < count <= MAX_RECORDS:
        _fail(f"{store}_record_count_invalid")
    return dict(raw)


def _resolve_store_path(root: Path, relative: str, *, store: str) -> Path:
    try:
        if root.is_symlink():
            _fail("artifact_root_symlink_rejected")
        resolved_root = root.resolve(strict=True)
    except FrontierArtifactError:
        raise
    except OSError:
        _fail("artifact_root_unreadable")
    candidate = resolved_root
    try:
        for part in PurePosixPath(relative).parts:
            candidate = candidate / part
            if candidate.is_symlink():
                _fail(f"{store}_symlink_rejected")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except FrontierArtifactError:
        raise
    except (OSError, ValueError):
        _fail(f"{store}_path_unreadable")
    return resolved


def _decode_blob_row(row: Any, *, store: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(row, dict) or set(row) != {"trial_id", "payload_b64"}:
        _fail(f"{store}_record_invalid")
    trial_id = row.get("trial_id")
    encoded = row.get("payload_b64")
    if not isinstance(trial_id, str) or not trial_id or trial_id != trial_id.strip():
        _fail(f"{store}_trial_id_invalid")
    if not isinstance(encoded, str) or not encoded or encoded != encoded.strip():
        _fail(f"{store}_payload_invalid")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError):
        _fail(f"{store}_payload_invalid")
    if len(payload) > MAX_RECORD_BYTES or base64.b64encode(payload).decode("ascii") != encoded:
        _fail(f"{store}_payload_noncanonical")
    value: dict[str, Any] = {"payload_sha256": hashlib.sha256(payload).hexdigest()}
    if store == "verifier_receipts":
        value["receipt"] = strict_json_loads(payload, role="trial_verifier_receipt")
    return trial_id, value


def _decode_structured_row(row: Any, *, store: str) -> tuple[str, dict[str, Any]]:
    expected = {"trial_id", *_STRUCTURED_FIELDS}
    if not isinstance(row, dict) or set(row) != expected:
        _fail(f"{store}_record_invalid")
    trial_id = row.get("trial_id")
    if not isinstance(trial_id, str) or not trial_id or trial_id != trial_id.strip():
        _fail(f"{store}_trial_id_invalid")
    return trial_id, {field: row[field] for field in _STRUCTURED_FIELDS}


def _read_jsonl_store(
    root: Path,
    descriptor: Mapping[str, Any],
    *,
    store: str,
    decoder: Callable[..., tuple[str, dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = _resolve_store_path(root, str(descriptor["path"]), store=store)
    records: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    bytes_seen = 0
    try:
        with open_stable_readonly_binary(path, max_bytes=MAX_STORE_BYTES) as (
            handle,
            identity,
        ):
            if identity.size != descriptor["size_bytes"]:
                _fail(f"{store}_size_mismatch")
            # Read at most one record beyond the declared count so malformed
            # manifests cannot turn verification into an unbounded scan.
            for _record_index in range(int(descriptor["record_count"]) + 1):
                line = handle.readline(MAX_RECORD_BYTES + 2)
                if not line:
                    break
                bytes_seen += len(line)
                digest.update(line)
                if len(line) > MAX_RECORD_BYTES + 1 or not line.endswith(b"\n"):
                    _fail(f"{store}_record_too_large_or_unterminated")
                content = line[:-1]
                if not content:
                    _fail(f"{store}_blank_record")
                row = strict_json_loads(content, role=f"{store}_record")
                try:
                    if canonical_json_bytes(row) != content:
                        _fail(f"{store}_record_noncanonical")
                except (TypeError, ValueError, OverflowError, RecursionError):
                    _fail(f"{store}_record_noncanonical")
                trial_id, decoded = decoder(row, store=store)
                if trial_id in records:
                    _fail(f"{store}_duplicate_trial_id")
                records[trial_id] = decoded
                if len(records) > MAX_RECORDS:
                    _fail(f"{store}_too_many_records")
    except StableFileReadError as exc:
        if exc.code == "changed_during_read":
            _fail(f"{store}_changed_during_read")
        if exc.code == "not_regular_file":
            _fail(f"{store}_not_regular_file")
        if exc.code in {
            "size_exceeds_bound",
            "bounded_read_length_mismatch",
            "grew_beyond_bound",
        }:
            _fail(f"{store}_size_mismatch")
        _fail(f"{store}_unreadable")
    except OSError:
        _fail(f"{store}_unreadable")
    if bytes_seen != descriptor["size_bytes"]:
        _fail(f"{store}_size_mismatch")
    if digest.hexdigest() != descriptor["sha256"]:
        _fail(f"{store}_hash_mismatch")
    if len(records) != descriptor["record_count"]:
        _fail(f"{store}_record_count_mismatch")
    return records, {
        "path": descriptor["path"],
        "sha256": digest.hexdigest(),
        "size_bytes": bytes_seen,
        "record_count": len(records),
    }


def _validate_raw_directory(root: Path, expected_paths: set[str]) -> None:
    try:
        resolved_root = root.resolve(strict=True)
        raw_root = (resolved_root / "raw").resolve(strict=True)
        raw_root.relative_to(resolved_root)
    except (OSError, ValueError):
        _fail("raw_artifact_directory_missing")
    observed: set[str] = set()
    entries = 0
    try:
        for path in raw_root.rglob("*"):
            entries += 1
            if entries > len(expected_paths) + 64:
                _fail("raw_artifact_directory_unbounded")
            if path.is_symlink():
                _fail("raw_artifact_symlink_rejected")
            if path.is_file():
                observed.add(path.relative_to(resolved_root).as_posix())
            elif not path.is_dir():
                _fail("raw_artifact_special_file_rejected")
    except FrontierArtifactError:
        raise
    except OSError:
        _fail("raw_artifact_directory_unreadable")
    if observed != expected_paths:
        _fail("raw_artifact_file_set_mismatch")


def _validate_trial_verifier_receipt(receipt: Any, trial: Mapping[str, Any]) -> None:
    expected_fields = {
        "schema",
        "trial_id",
        "task_payload_sha256",
        "scorer_config_sha256",
        "treatment_output_sha256",
        "control_output_sha256",
        "treatment_success",
        "control_success",
        "verifier_blinded",
        "scorer_implementation_sha256",
        "verified_at",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_fields:
        _fail("trial_verifier_receipt_schema_invalid")
    if receipt.get("schema") != TRIAL_VERIFIER_RECEIPT_SCHEMA:
        _fail("trial_verifier_receipt_schema_invalid")
    for field in (
        "trial_id",
        "task_payload_sha256",
        "scorer_config_sha256",
        "treatment_output_sha256",
        "control_output_sha256",
        "treatment_success",
        "control_success",
        "verifier_blinded",
    ):
        if receipt.get(field) != trial.get(field):
            _fail("trial_verifier_receipt_binding_mismatch")
    if not _is_sha256(receipt.get("scorer_implementation_sha256")):
        _fail("trial_verifier_implementation_invalid")
    verified_at = receipt.get("verified_at")
    evaluation_started_at = trial.get("evaluation_started_at")
    if (
        isinstance(verified_at, bool)
        or not isinstance(verified_at, (int, float))
        or not math.isfinite(float(verified_at))
        or isinstance(evaluation_started_at, bool)
        or not isinstance(evaluation_started_at, (int, float))
        or float(verified_at) < float(evaluation_started_at)
    ):
        _fail("trial_verifier_chronology_invalid")


def verify_raw_artifact_package(
    bundle: Mapping[str, Any],
    manifest: Any,
    manifest_bytes: bytes,
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    """Recompute every raw binding and return a deterministic receipt."""

    if not isinstance(manifest, dict) or set(manifest) != {
        "schema",
        "producer_id",
        "preregistration_sha256",
        "task_commitment_sha256",
        "stores",
    }:
        _fail("raw_artifact_manifest_schema_invalid")
    if manifest.get("schema") != RAW_ARTIFACT_MANIFEST_SCHEMA:
        _fail("raw_artifact_manifest_schema_invalid")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if bundle.get("raw_artifact_manifest_sha256") != manifest_sha256:
        _fail("raw_artifact_manifest_hash_mismatch")
    for field in ("producer_id", "preregistration_sha256", "task_commitment_sha256"):
        if manifest.get(field) != bundle.get(field):
            _fail("raw_artifact_manifest_bundle_mismatch")
    preregistration = bundle.get("preregistration")
    comparison_kind = (
        str(preregistration.get("comparison_kind") or "")
        if isinstance(preregistration, Mapping)
        else ""
    )
    external_comparison = comparison_kind == _EXTERNAL_COMPARISON_KIND
    required_stores = (
        _REQUIRED_STORES | {_PROVIDER_STORE}
        if external_comparison
        else _REQUIRED_STORES
    )
    stores = manifest.get("stores")
    if not isinstance(stores, dict) or set(stores) != required_stores:
        _fail("raw_artifact_store_set_invalid")
    descriptors = {name: _validate_descriptor(stores[name], store=name) for name in sorted(stores)}
    paths = [str(descriptor["path"]) for descriptor in descriptors.values()]
    if len(paths) != len(set(paths)):
        _fail("raw_artifact_store_path_reused")
    total_declared = sum(int(descriptor["size_bytes"]) for descriptor in descriptors.values())
    if total_declared > MAX_TOTAL_STORE_BYTES:
        _fail("raw_artifact_total_size_exceeded")
    _validate_raw_directory(Path(artifact_root), set(paths))

    trials_raw = bundle.get("trials")
    if not isinstance(trials_raw, list) or not trials_raw or len(trials_raw) > MAX_RECORDS:
        _fail("raw_artifact_trials_invalid")
    trials: dict[str, Mapping[str, Any]] = {}
    for trial in trials_raw:
        trial_id = trial.get("trial_id") if isinstance(trial, Mapping) else None
        if not isinstance(trial_id, str) or not trial_id or trial_id in trials:
            _fail("raw_artifact_trial_id_invalid")
        trials[trial_id] = trial
    expected_trial_ids = set(trials)

    decoded_stores: dict[str, dict[str, dict[str, Any]]] = {}
    store_receipts: dict[str, dict[str, Any]] = {}
    for name in sorted(_BLOB_STORES):
        decoded, receipt = _read_jsonl_store(
            Path(artifact_root),
            descriptors[name],
            store=name,
            decoder=_decode_blob_row,
        )
        if set(decoded) != expected_trial_ids:
            _fail(f"{name}_trial_set_mismatch")
        decoded_stores[name] = decoded
        store_receipts[name] = receipt
    if external_comparison:
        decoded, receipt = _read_jsonl_store(
            Path(artifact_root),
            descriptors[_PROVIDER_STORE],
            store=_PROVIDER_STORE,
            decoder=_decode_blob_row,
        )
        if set(decoded) != expected_trial_ids:
            _fail("provider_receipts_trial_set_mismatch")
        decoded_stores[_PROVIDER_STORE] = decoded
        store_receipts[_PROVIDER_STORE] = receipt
    structured, structured_receipt = _read_jsonl_store(
        Path(artifact_root),
        descriptors[_STRUCTURED_STORE],
        store=_STRUCTURED_STORE,
        decoder=_decode_structured_row,
    )
    if set(structured) != expected_trial_ids:
        _fail("structured_receipts_trial_set_mismatch")
    store_receipts[_STRUCTURED_STORE] = structured_receipt

    lineage: list[dict[str, Any]] = []
    for trial_id in sorted(trials):
        trial = trials[trial_id]
        row: dict[str, Any] = {"trial_id": trial_id}
        for store, field in _BLOB_STORES.items():
            record = decoded_stores[store][trial_id]
            digest = record["payload_sha256"]
            if digest != trial.get(field):
                _fail(f"{store}_bundle_hash_mismatch")
            row[field] = digest
        if external_comparison:
            control_receipt = trial.get("control_receipt")
            expected_provider_digest = (
                control_receipt.get("provider_receipt_sha256")
                if isinstance(control_receipt, Mapping)
                else None
            )
            provider_record = decoded_stores[_PROVIDER_STORE][trial_id]
            if provider_record["payload_sha256"] != expected_provider_digest:
                _fail("provider_receipts_bundle_hash_mismatch")
            row["provider_receipt_sha256"] = expected_provider_digest
        verifier_receipt = decoded_stores["verifier_receipts"][trial_id].get("receipt")
        _validate_trial_verifier_receipt(verifier_receipt, trial)
        for field in _STRUCTURED_FIELDS:
            if structured[trial_id].get(field) != trial.get(field):
                _fail("structured_receipts_bundle_mismatch")
        row["structured_receipt_sha256"] = canonical_sha256(structured[trial_id])
        lineage.append(row)

    receipt: dict[str, Any] = {
        "schema": ARTIFACT_VERIFICATION_SCHEMA,
        "accepted": True,
        "manifest_sha256": manifest_sha256,
        "artifact_store_count": len(store_receipts),
        "artifact_bytes": sum(item["size_bytes"] for item in store_receipts.values()),
        "trial_count": len(trials),
        "lineage_sha256": canonical_sha256(lineage),
        "store_receipts": store_receipts,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


__all__ = [
    "ARTIFACT_VERIFICATION_SCHEMA",
    "FrontierArtifactError",
    "RAW_ARTIFACT_MANIFEST_SCHEMA",
    "TRIAL_VERIFIER_RECEIPT_SCHEMA",
    "read_strict_json_file",
    "strict_json_loads",
    "verify_raw_artifact_package",
]
