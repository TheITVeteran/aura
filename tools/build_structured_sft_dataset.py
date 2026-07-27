#!/usr/bin/env python3
"""Build or independently validate Aura's custodied structured-SFT bundles."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.governance_context import local_internal_governed_scope  # noqa: E402
from core.learning.structured_sft import (  # noqa: E402
    STRUCTURED_SFT_CANDIDATE_FILES,
    STRUCTURED_SFT_EVALUATOR_FILES,
    StructuredSFTCurriculumSpec,
    StructuredSFTError,
    build_structured_sft_custody_bundles,
    validate_candidate_dataset_artifacts,
    validate_evaluator_dataset_artifacts,
    validate_structured_sft_custody_pair,
)
from core.runtime.file_read_gateway import (  # noqa: E402
    open_stable_readonly_binary,
    read_stable_bytes,
    read_stable_directory_files,
)
from core.runtime.file_write_gateway import (  # noqa: E402
    DirectoryFileWriteBatchEntry,
    FileWriteTransactionError,
    get_file_write_gateway,
)

REPORT_SCHEMA = "aura.rlc.structured_sft_custody_build_report.v2"
_CUSTODY_COMMIT_SCHEMA = "aura.rlc.structured_sft_custody_commit.v1"
_CUSTODY_COMMIT_FILE = ".aura_structured_sft_custody.commit.json"
_CUSTODY_LOCK_FILE = ".aura_structured_sft_custody.lock"
_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_CUSTODY_COMMIT_BYTES = 64 * 1024


class CandidateDatasetBuildError(RuntimeError):
    """The custody bundles could not be built or validated safely."""


def _fail(reason: str) -> Never:
    normalized = str(reason or "").strip()
    if not normalized:
        normalized = "candidate_dataset_build_failed"
    raise CandidateDatasetBuildError(normalized)


def _lexical_absolute(path: Path) -> Path:
    requested = Path(os.path.abspath(os.fspath(path.expanduser())))
    if requested.is_symlink():
        _fail(f"symlink_output_path_rejected:{requested}")
    return requested


def _validated_output_directory(path: Path, *, must_exist: bool) -> Path:
    target = _lexical_absolute(path)
    for component in reversed((target, *target.parents)):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            _fail(f"symlink_output_path_rejected:{component}")
        if component != target and not stat.S_ISDIR(mode):
            _fail(f"non_directory_output_ancestor:{component}")
    if target.exists() and not target.is_dir():
        _fail(f"output_path_not_directory:{target}")
    if must_exist and not target.is_dir():
        _fail(f"output_directory_missing:{target}")
    return target


def _validated_custody_directories(
    candidate_directory: Path,
    evaluator_directory: Path,
    *,
    must_exist: bool,
) -> tuple[Path, Path]:
    candidate = _validated_output_directory(
        candidate_directory,
        must_exist=must_exist,
    )
    evaluator = _validated_output_directory(
        evaluator_directory,
        must_exist=must_exist,
    )
    if (
        candidate == evaluator
        or candidate in evaluator.parents
        or evaluator in candidate.parents
        or candidate.parent != evaluator.parent
    ):
        _fail(
            "custody_directories_must_be_distinct_non_nested_siblings"
        )
    return candidate, evaluator


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _committed_custody_record(body: dict[str, Any]) -> dict[str, Any]:
    return {
        **body,
        "commit_sha256": hashlib.sha256(
            _canonical_json_bytes(body)
        ).hexdigest(),
    }


def _custody_commit_path(candidate: Path, evaluator: Path) -> Path:
    if candidate.parent != evaluator.parent:
        _fail("custody_directories_must_share_private_root")
    return candidate.parent / _CUSTODY_COMMIT_FILE


@contextmanager
def _custody_publication_lock(root: Path) -> Iterator[None]:
    root = _validated_output_directory(root, must_exist=True)
    root_metadata = root.stat()
    if (
        root_metadata.st_uid != os.getuid()
        or stat.S_IMODE(root_metadata.st_mode) & 0o077
    ):
        _fail(f"custody_root_not_owner_private:{root}")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int) or nofollow <= 0:
        _fail("custody_publication_requires_nofollow")
    lock_path = root / _CUSTODY_LOCK_FILE
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | nofollow,
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        opened = os.fstat(descriptor)
        entry = lock_path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or (opened.st_dev, opened.st_ino)
            != (entry.st_dev, entry.st_ino)
        ):
            _fail("custody_publication_lock_binding_invalid")
        yield
        entry_after = lock_path.stat(follow_symlinks=False)
        if (
            opened.st_dev,
            opened.st_ino,
        ) != (entry_after.st_dev, entry_after.st_ino):
            _fail("custody_publication_lock_binding_changed")
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_custody_record(
    path: Path,
    record: dict[str, Any],
    *,
    source: str,
) -> None:
    with local_internal_governed_scope(
        "structured_sft.custody_commit",
        domain="file_write",
    ):
        get_file_write_gateway().write_bytes(
            path,
            _canonical_json_bytes(record),
            source=source,
        )


def _read_custody_record(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            read_stable_bytes(
                path,
                max_bytes=_MAX_CUSTODY_COMMIT_BYTES,
            )
        )
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise CandidateDatasetBuildError(
            "custody_commit_json_invalid"
        ) from exc
    fields_by_state = {
        "preparing": {
            "schema",
            "state",
            "generation_id",
            "candidate_directory",
            "evaluator_directory",
            "commit_sha256",
        },
        "committed": {
            "schema",
            "state",
            "generation_id",
            "candidate_directory",
            "evaluator_directory",
            "candidate_package_sha256",
            "evaluator_package_sha256",
            "custody_root_sha256",
            "custody_report_sha256",
            "commit_sha256",
        },
    }
    state = value.get("state") if isinstance(value, dict) else None
    candidate_name = (
        value.get("candidate_directory") if isinstance(value, dict) else None
    )
    evaluator_name = (
        value.get("evaluator_directory") if isinstance(value, dict) else None
    )
    names_valid = all(
        isinstance(name, str)
        and bool(name)
        and name not in {".", ".."}
        and Path(name).name == name
        and "/" not in name
        and "\\" not in name
        and "\0" not in name
        for name in (candidate_name, evaluator_name)
    )
    generation_id = value.get("generation_id") if isinstance(value, dict) else None
    if (
        state not in fields_by_state
        or set(value) != fields_by_state[state]
        or value.get("schema") != _CUSTODY_COMMIT_SCHEMA
        or not names_valid
        or candidate_name == evaluator_name
        or not isinstance(generation_id, str)
        or len(generation_id) != 32
        or any(character not in "0123456789abcdef" for character in generation_id)
        or (
            state == "committed"
            and any(
                not isinstance(value.get(field), str)
                or len(value[field]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value[field]
                )
                for field in (
                    "candidate_package_sha256",
                    "evaluator_package_sha256",
                    "custody_root_sha256",
                    "custody_report_sha256",
                )
            )
        )
    ):
        _fail("custody_commit_schema_invalid")
    body = dict(value)
    observed_sha256 = body.pop("commit_sha256")
    expected_sha256 = hashlib.sha256(
        _canonical_json_bytes(body)
    ).hexdigest()
    if observed_sha256 != expected_sha256:
        _fail("custody_commit_commitment_invalid")
    return value


def _read_private_holdout_seed(path: Path) -> bytes:
    parent = _validated_output_directory(path.parent, must_exist=True)
    target = parent / path.name
    with open_stable_readonly_binary(target, max_bytes=32) as (
        handle,
        identity,
    ):
        metadata = os.fstat(handle.fileno())
        if (
            identity.size != 32
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            _fail("holdout_seed_file_must_be_owner_only_single_link")
        seed = handle.read(33)
    if len(seed) != 32:
        _fail("holdout_seed_file_must_contain_exactly_32_bytes")
    return seed


def _read_exact_directory(
    directory: Path,
    *,
    names: tuple[str, ...],
) -> dict[str, bytes]:
    target = _validated_output_directory(directory, must_exist=True)
    try:
        return read_stable_directory_files(
            target,
            names=names,
            max_bytes_per_file=_MAX_ARTIFACT_BYTES,
        )
    except OSError as exc:
        raise CandidateDatasetBuildError(
            f"custody_directory_read_failed:{target}:{exc}"
        ) from exc


def read_candidate_dataset_directory_with_attestation(
    directory: Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Read trainer-visible bytes and the pair commitment, never holdout bytes."""

    candidate = _validated_output_directory(directory, must_exist=True)
    commit = _read_custody_record(candidate.parent / _CUSTODY_COMMIT_FILE)
    if (
        commit["state"] != "committed"
        or commit["candidate_directory"] != candidate.name
    ):
        _fail("candidate_custody_generation_not_committed")
    candidate_artifacts = _read_exact_directory(
        candidate,
        names=STRUCTURED_SFT_CANDIDATE_FILES,
    )
    candidate_manifest = validate_candidate_dataset_artifacts(
        candidate_artifacts
    )
    if (
        candidate_manifest["package_sha256"]
        != commit["candidate_package_sha256"]
        or candidate_manifest["custody_root_sha256"]
        != commit["custody_root_sha256"]
        or _read_custody_record(
            candidate.parent / _CUSTODY_COMMIT_FILE
        )
        != commit
    ):
        _fail("candidate_custody_commit_binding_invalid")
    return candidate_artifacts, commit


def read_candidate_dataset_directory(directory: Path) -> dict[str, bytes]:
    """Read candidate bytes without opening the evaluator custody root."""

    artifacts, _attestation = read_candidate_dataset_directory_with_attestation(
        directory
    )
    return artifacts


def read_evaluator_dataset_directory(directory: Path) -> dict[str, bytes]:
    """Read an evaluator only when the committed candidate pair verifies."""

    evaluator = _validated_output_directory(directory, must_exist=True)
    commit = _read_custody_record(evaluator.parent / _CUSTODY_COMMIT_FILE)
    if (
        commit["state"] != "committed"
        or commit["evaluator_directory"] != evaluator.name
    ):
        _fail("evaluator_custody_generation_not_committed")
    candidate, committed_evaluator = _validated_custody_directories(
        evaluator.parent / commit["candidate_directory"],
        evaluator.parent / commit["evaluator_directory"],
        must_exist=True,
    )
    if committed_evaluator != evaluator:
        _fail("evaluator_custody_generation_not_committed")
    candidate_artifacts = _read_exact_directory(
        candidate,
        names=STRUCTURED_SFT_CANDIDATE_FILES,
    )
    evaluator_artifacts = _read_exact_directory(
        evaluator,
        names=STRUCTURED_SFT_EVALUATOR_FILES,
    )
    pair = validate_structured_sft_custody_pair(
        candidate_artifacts,
        evaluator_artifacts,
    )
    if (
        pair["candidate_package_sha256"]
        != commit["candidate_package_sha256"]
        or pair["evaluator_package_sha256"]
        != commit["evaluator_package_sha256"]
        or pair["custody_root_sha256"] != commit["custody_root_sha256"]
        or pair["custody_report_sha256"]
        != commit["custody_report_sha256"]
        or _read_custody_record(
            evaluator.parent / _CUSTODY_COMMIT_FILE
        )
        != commit
    ):
        _fail("evaluator_custody_commit_binding_invalid")
    return evaluator_artifacts


def validate_candidate_dataset_directory(directory: Path) -> dict[str, Any]:
    """Replay only the trainer-visible candidate package."""

    return validate_candidate_dataset_artifacts(
        read_candidate_dataset_directory(directory)
    )


def validate_custody_directories(
    *,
    candidate_directory: Path,
    evaluator_directory: Path,
) -> dict[str, Any]:
    """Validate both custody roots and their shared commitments."""

    candidate, evaluator = _validated_custody_directories(
        candidate_directory,
        evaluator_directory,
        must_exist=True,
    )
    candidate_artifacts = read_candidate_dataset_directory(candidate)
    evaluator_artifacts = read_evaluator_dataset_directory(evaluator)
    return validate_structured_sft_custody_pair(
        candidate_artifacts,
        evaluator_artifacts,
    )


def build_custodied_dataset_directories(
    *,
    candidate_directory: Path,
    evaluator_directory: Path,
    spec: StructuredSFTCurriculumSpec,
    holdout_seed: bytes,
) -> dict[str, Any]:
    """Build, publish, re-read, and replay both custody bundles."""

    candidate, evaluator = _validated_custody_directories(
        candidate_directory,
        evaluator_directory,
        must_exist=False,
    )
    bundles = build_structured_sft_custody_bundles(
        spec,
        holdout_seed=holdout_seed,
    )
    gateway = get_file_write_gateway()
    generation_id = uuid.uuid4().hex
    commit_path = _custody_commit_path(candidate, evaluator)
    with _custody_publication_lock(candidate.parent):
        preparing = _committed_custody_record(
            {
                "schema": _CUSTODY_COMMIT_SCHEMA,
                "state": "preparing",
                "generation_id": generation_id,
                "candidate_directory": candidate.name,
                "evaluator_directory": evaluator.name,
            }
        )
        _write_custody_record(
            commit_path,
            preparing,
            source="structured_sft.custody_preparing",
        )
        with local_internal_governed_scope(
            "structured_sft.custody_dataset",
            domain="file_write",
        ):
            gateway.ensure_directory(
                candidate,
                source="structured_sft.candidate_dataset",
            )
            gateway.ensure_directory(
                evaluator,
                source="structured_sft.evaluator_dataset",
            )
            evaluator_receipt = gateway.write_bytes_batch_in_directory(
                evaluator,
                tuple(
                    DirectoryFileWriteBatchEntry(
                        name,
                        bundles.evaluator_artifacts[name],
                        mode=0o600,
                    )
                    for name in STRUCTURED_SFT_EVALUATOR_FILES
                ),
                allowed_existing_names=STRUCTURED_SFT_EVALUATOR_FILES,
                commit_marker="evaluator_manifest.json",
                source="structured_sft.evaluator_dataset",
            )
            candidate_receipt = gateway.write_bytes_batch_in_directory(
                candidate,
                tuple(
                    DirectoryFileWriteBatchEntry(
                        name,
                        bundles.candidate_artifacts[name],
                        mode=0o600,
                    )
                    for name in STRUCTURED_SFT_CANDIDATE_FILES
                ),
                allowed_existing_names=STRUCTURED_SFT_CANDIDATE_FILES,
                commit_marker="manifest.json",
                source="structured_sft.candidate_dataset",
            )
        durable_candidate = _read_exact_directory(
            candidate,
            names=STRUCTURED_SFT_CANDIDATE_FILES,
        )
        durable_evaluator = _read_exact_directory(
            evaluator,
            names=STRUCTURED_SFT_EVALUATOR_FILES,
        )
        candidate_manifest = validate_candidate_dataset_artifacts(
            durable_candidate
        )
        evaluator_manifest = validate_evaluator_dataset_artifacts(
            durable_evaluator,
            candidate_artifacts=durable_candidate,
        )
        custody = validate_structured_sft_custody_pair(
            durable_candidate,
            durable_evaluator,
        )
        committed = _committed_custody_record(
            {
                "schema": _CUSTODY_COMMIT_SCHEMA,
                "state": "committed",
                "generation_id": generation_id,
                "candidate_directory": candidate.name,
                "evaluator_directory": evaluator.name,
                "candidate_package_sha256": candidate_manifest[
                    "package_sha256"
                ],
                "evaluator_package_sha256": evaluator_manifest[
                    "evaluator_package_sha256"
                ],
                "custody_root_sha256": custody["custody_root_sha256"],
                "custody_report_sha256": custody[
                    "custody_report_sha256"
                ],
            }
        )
        _write_custody_record(
            commit_path,
            committed,
            source="structured_sft.custody_committed",
        )
        validate_structured_sft_custody_pair(
            read_candidate_dataset_directory(candidate),
            read_evaluator_dataset_directory(evaluator),
        )
    return {
        "schema": REPORT_SCHEMA,
        "status": "custody_bundles_built_and_replay_validated",
        "candidate_directory": str(candidate),
        "evaluator_directory": str(evaluator),
        "candidate_transaction_id": candidate_receipt.transaction_id,
        "evaluator_transaction_id": evaluator_receipt.transaction_id,
        "candidate_package_sha256": candidate_manifest["package_sha256"],
        "evaluator_package_sha256": evaluator_manifest[
            "evaluator_package_sha256"
        ],
        "custody_root_sha256": custody["custody_root_sha256"],
        "custody_report_sha256": custody["custody_report_sha256"],
        "custody_commit_sha256": committed["commit_sha256"],
        "custody_generation_id": generation_id,
        "candidate_artifact_sha256": dict(candidate_receipt.sha256),
        "evaluator_artifact_sha256": dict(evaluator_receipt.sha256),
        "trainer_ready": candidate_manifest["trainer_ready"],
        "training_authority": candidate_manifest["training_authority"],
        "required_next_gates": candidate_manifest["required_next_gates"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--evaluator-dir", type=Path, required=True)
    parser.add_argument("--holdout-seed-file", type=Path)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--train-cases-per-family", type=int, default=16)
    parser.add_argument("--validation-cases-per-family", type=int, default=4)
    parser.add_argument("--holdout-cases-per-family", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Replay existing candidate and evaluator bundles without writing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.validate_only:
            custody = validate_custody_directories(
                candidate_directory=arguments.candidate_dir,
                evaluator_directory=arguments.evaluator_dir,
            )
            report = {
                "schema": REPORT_SCHEMA,
                "status": "custody_bundles_replay_validated",
                "candidate_directory": str(
                    _validated_output_directory(
                        arguments.candidate_dir,
                        must_exist=True,
                    )
                ),
                "evaluator_directory": str(
                    _validated_output_directory(
                        arguments.evaluator_dir,
                        must_exist=True,
                    )
                ),
                "candidate_package_sha256": custody[
                    "candidate_package_sha256"
                ],
                "evaluator_package_sha256": custody[
                    "evaluator_package_sha256"
                ],
                "custody_root_sha256": custody["custody_root_sha256"],
                "custody_report_sha256": custody["custody_report_sha256"],
            }
        else:
            if arguments.holdout_seed_file is None:
                _fail("holdout_seed_file_required")
            holdout_seed = _read_private_holdout_seed(
                arguments.holdout_seed_file
            )
            report = build_custodied_dataset_directories(
                candidate_directory=arguments.candidate_dir,
                evaluator_directory=arguments.evaluator_dir,
                spec=StructuredSFTCurriculumSpec(
                    seed=arguments.seed,
                    train_cases_per_family=arguments.train_cases_per_family,
                    validation_cases_per_family=(
                        arguments.validation_cases_per_family
                    ),
                    holdout_cases_per_family=arguments.holdout_cases_per_family,
                    max_seq_length=arguments.max_seq_length,
                ),
                holdout_seed=holdout_seed,
            )
    except (
        CandidateDatasetBuildError,
        FileWriteTransactionError,
        OSError,
        StructuredSFTError,
    ) as exc:
        print(
            f"build_structured_sft_dataset: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
