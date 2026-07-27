"""Crash-recoverable publication for combined SFT lineage custody."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.governance_context import local_internal_governed_scope
from core.learning.combined_sft_lineage import (
    COMBINED_SFT_LINEAGE_CANDIDATE_FILES,
    COMBINED_SFT_LINEAGE_EVALUATOR_FILES,
    CombinedSFTLineageBundle,
    validate_combined_sft_lineage_custody,
)
from core.runtime.atomic_writer import interprocess_file_lock
from core.runtime.file_read_gateway import read_stable_bytes, read_stable_directory_files
from core.runtime.file_write_gateway import (
    DirectoryFileWriteBatchEntry,
    get_file_write_gateway,
)

COMBINED_SFT_LINEAGE_PUBLICATION_SCHEMA: Final = "aura.rlc.combined_sft_lineage_publication.v1"
COMBINED_SFT_LINEAGE_PUBLICATION_REPORT_SCHEMA: Final = (
    "aura.rlc.combined_sft_lineage_publication_report.v1"
)
_COMMIT = ".aura_combined_sft_lineage.commit.json"
_LOCK = ".aura_combined_sft_lineage.lock"
_BATCH_LOCK = ".aura_file_write_batch.lock"
_MAX_COMMIT_BYTES = 128 * 1024


class CombinedSFTLineagePublicationError(RuntimeError):
    """Combined lineage custody could not be published safely."""


def _error(reason: str) -> CombinedSFTLineagePublicationError:
    return CombinedSFTLineagePublicationError(
        str(reason or "combined_sft_lineage_publication_failed")
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise _error(f"combined_lineage_publication_duplicate_key:{key}")
        value[key] = child
    return value


def _private_directory(path: Path, *, create: bool) -> Path:
    target = Path(os.path.abspath(os.fspath(path.expanduser())))
    if target.is_symlink():
        raise _error("combined_lineage_publication_symlink_rejected")
    for parent in (target, *target.parents):
        try:
            metadata = parent.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise _error("combined_lineage_publication_symlink_rejected")
    if create:
        with local_internal_governed_scope(
            "combined_sft_lineage.publication_directory",
            domain="file_write",
        ):
            get_file_write_gateway().ensure_directory(
                target,
                source="combined_sft_lineage.publication_directory",
            )
    if not target.is_dir():
        raise _error("combined_lineage_publication_directory_missing")
    metadata = target.stat(follow_symlinks=False)
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise _error("combined_lineage_publication_directory_not_private")
    return target


def _write_commit(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    record = {**dict(body), "commit_sha256": _sha(body)}
    with local_internal_governed_scope(
        "combined_sft_lineage.publication_commit",
        domain="file_write",
    ):
        get_file_write_gateway().write_bytes(
            path,
            canonical_json_bytes(record),
            source="combined_sft_lineage.publication_commit",
        )
    return record


def _ensure_lock_file(root: Path) -> Path:
    lock_path = root / _LOCK
    with local_internal_governed_scope(
        "combined_sft_lineage.publication_lock",
        domain="file_write",
    ):
        get_file_write_gateway().write_bytes_if_absent(
            lock_path,
            b"",
            source="combined_sft_lineage.publication_lock",
        )
    metadata = lock_path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise _error("combined_lineage_publication_lock_invalid")
    return lock_path


def _read_commit(path: Path) -> dict[str, Any]:
    try:
        raw = read_stable_bytes(path, max_bytes=_MAX_COMMIT_BYTES)
        record = json.loads(raw, object_pairs_hook=_strict_object)
    except (OSError, RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise CombinedSFTLineagePublicationError(
            "combined_lineage_publication_commit_invalid"
        ) from exc
    common = {
        "schema",
        "state",
        "generation_id",
        "candidate_directory",
        "evaluator_directory",
        "manifest_sha256",
        "commitment_sha256",
        "combined_semantic_index_sha256",
        "record_count",
        "commit_sha256",
    }
    fields = {
        "preparing": common,
        "committed": common | {"candidate_transaction_id", "evaluator_transaction_id"},
    }
    state = record.get("state") if isinstance(record, dict) else None
    body = dict(record) if isinstance(record, dict) else {}
    observed = body.pop("commit_sha256", None)
    if (
        state not in fields
        or set(record) != fields[state]
        or record.get("schema") != COMBINED_SFT_LINEAGE_PUBLICATION_SCHEMA
        or not isinstance(record.get("generation_id"), str)
        or len(record["generation_id"]) != 32
        or record.get("candidate_directory") != "candidate"
        or record.get("evaluator_directory") != "evaluator"
        or type(record.get("record_count")) is not int
        or record["record_count"] < 1
        or observed != _sha(body)
        or canonical_json_bytes(record) != raw
    ):
        raise _error("combined_lineage_publication_commit_invalid")
    return record


def _read_artifacts(directory: Path, names: tuple[str, ...]) -> dict[str, bytes]:
    target = _private_directory(directory, create=False)
    expected = {*names, _BATCH_LOCK}
    if set(os.listdir(target)) != expected:
        raise _error("combined_lineage_publication_inventory_invalid")
    for name in expected:
        metadata = (target / name).stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise _error("combined_lineage_publication_artifact_invalid")
    return read_stable_directory_files(target, names=names, max_bytes_per_file=512 * 1024 * 1024)


def read_combined_sft_lineage_publication(
    candidate_directory: Path,
    *,
    evaluator_directory: Path | None = None,
) -> dict[str, Any]:
    """Read a committed candidate, optionally reconstructing evaluator custody."""

    candidate = _private_directory(candidate_directory, create=False)
    before = _read_commit(candidate.parent / _COMMIT)
    if before["state"] != "committed" or candidate.name != "candidate":
        raise _error("combined_lineage_publication_not_committed")
    candidate_artifacts = _read_artifacts(
        candidate,
        COMBINED_SFT_LINEAGE_CANDIDATE_FILES,
    )
    result: dict[str, Any] = {
        "candidate_artifacts": candidate_artifacts,
        "commit": before,
    }
    if evaluator_directory is not None:
        evaluator = _private_directory(evaluator_directory, create=False)
        if evaluator != candidate.parent / "evaluator":
            raise _error("combined_lineage_publication_evaluator_binding_invalid")
        evaluator_artifacts = _read_artifacts(
            evaluator,
            COMBINED_SFT_LINEAGE_EVALUATOR_FILES,
        )
        custody = validate_combined_sft_lineage_custody(
            candidate_artifacts,
            evaluator_artifacts,
        )
        if any(
            custody[field] != before[field]
            for field in (
                "manifest_sha256",
                "commitment_sha256",
                "combined_semantic_index_sha256",
                "record_count",
            )
        ):
            raise _error("combined_lineage_publication_custody_binding_invalid")
        result["evaluator_artifacts"] = evaluator_artifacts
        result["custody_report"] = custody
    if _read_commit(candidate.parent / _COMMIT) != before:
        raise _error("combined_lineage_publication_commit_changed")
    return result


def publish_combined_sft_lineage_custody(
    *,
    bundle: CombinedSFTLineageBundle,
    publication_root: Path,
) -> dict[str, Any]:
    """Publish evaluator then candidate under one preparing/committed record."""

    custody = validate_combined_sft_lineage_custody(
        bundle.candidate_artifacts,
        bundle.evaluator_artifacts,
    )
    root = _private_directory(publication_root, create=True)
    candidate = root / "candidate"
    evaluator = root / "evaluator"
    generation_id = uuid.uuid4().hex
    recovered = ""
    lock_path = _ensure_lock_file(root)
    with interprocess_file_lock(lock_path):
        _ensure_lock_file(root)
        allowed = {_LOCK, _COMMIT, "candidate", "evaluator"}
        if set(os.listdir(root)) - allowed:
            raise _error("combined_lineage_publication_root_inventory_invalid")
        commit_path = root / _COMMIT
        if commit_path.exists():
            prior = _read_commit(commit_path)
            if prior["state"] == "committed":
                existing = read_combined_sft_lineage_publication(
                    candidate,
                    evaluator_directory=evaluator,
                )
                if all(
                    prior[field] == custody[field]
                    for field in (
                        "manifest_sha256",
                        "commitment_sha256",
                        "combined_semantic_index_sha256",
                        "record_count",
                    )
                ):
                    return {
                        **custody,
                        "schema": COMBINED_SFT_LINEAGE_PUBLICATION_REPORT_SCHEMA,
                        "status": "existing_committed_generation_revalidated",
                        "generation_id": prior["generation_id"],
                        "recovered_preparing_generation_id": "",
                    }
                del existing
            else:
                recovered = prior["generation_id"]
        common = {
            "schema": COMBINED_SFT_LINEAGE_PUBLICATION_SCHEMA,
            "generation_id": generation_id,
            "candidate_directory": "candidate",
            "evaluator_directory": "evaluator",
            "manifest_sha256": custody["manifest_sha256"],
            "commitment_sha256": custody["commitment_sha256"],
            "combined_semantic_index_sha256": custody["combined_semantic_index_sha256"],
            "record_count": custody["record_count"],
        }
        _write_commit(commit_path, {**common, "state": "preparing"})
        gateway = get_file_write_gateway()
        with local_internal_governed_scope(
            "combined_sft_lineage.publication_artifacts",
            domain="file_write",
        ):
            gateway.ensure_directory(evaluator, source="combined_lineage.evaluator")
            gateway.ensure_directory(candidate, source="combined_lineage.candidate")
            evaluator_receipt = gateway.write_bytes_batch_in_directory(
                evaluator,
                tuple(
                    DirectoryFileWriteBatchEntry(name, bundle.evaluator_artifacts[name])
                    for name in COMBINED_SFT_LINEAGE_EVALUATOR_FILES
                ),
                allowed_existing_names=COMBINED_SFT_LINEAGE_EVALUATOR_FILES,
                commit_marker=COMBINED_SFT_LINEAGE_EVALUATOR_FILES[0],
                source="combined_sft_lineage.evaluator_artifacts",
            )
            candidate_receipt = gateway.write_bytes_batch_in_directory(
                candidate,
                tuple(
                    DirectoryFileWriteBatchEntry(name, bundle.candidate_artifacts[name])
                    for name in COMBINED_SFT_LINEAGE_CANDIDATE_FILES
                ),
                allowed_existing_names=COMBINED_SFT_LINEAGE_CANDIDATE_FILES,
                commit_marker=COMBINED_SFT_LINEAGE_CANDIDATE_FILES[0],
                source="combined_sft_lineage.candidate_artifacts",
            )
        validate_combined_sft_lineage_custody(
            _read_artifacts(candidate, COMBINED_SFT_LINEAGE_CANDIDATE_FILES),
            _read_artifacts(evaluator, COMBINED_SFT_LINEAGE_EVALUATOR_FILES),
        )
        committed = _write_commit(
            commit_path,
            {
                **common,
                "state": "committed",
                "candidate_transaction_id": candidate_receipt.transaction_id,
                "evaluator_transaction_id": evaluator_receipt.transaction_id,
            },
        )
        read_combined_sft_lineage_publication(
            candidate,
            evaluator_directory=evaluator,
        )
    return {
        **custody,
        "schema": COMBINED_SFT_LINEAGE_PUBLICATION_REPORT_SCHEMA,
        "status": "combined_lineage_custody_published",
        "generation_id": generation_id,
        "recovered_preparing_generation_id": recovered,
        "publication_commit_sha256": committed["commit_sha256"],
    }


__all__ = [
    "COMBINED_SFT_LINEAGE_PUBLICATION_REPORT_SCHEMA",
    "COMBINED_SFT_LINEAGE_PUBLICATION_SCHEMA",
    "CombinedSFTLineagePublicationError",
    "publish_combined_sft_lineage_custody",
    "read_combined_sft_lineage_publication",
]
