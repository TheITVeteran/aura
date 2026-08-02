"""Governed durable publication for verified-replay SFT custody bundles.

The projection builder is deliberately pure. This module is the stateful
boundary that publishes its candidate and evaluator packages through Aura's
file-write gateway, under separate sibling directories and one fail-closed
pair commit. A preparing commit makes either directory unreadable to consumers;
after a crash, the next publication may replace that incomplete generation.
Only a fully re-read and reconstructed pair receives a committed record.

Runtime publication requires the live Horcrux-backed BlackHole. The Horcrux
derives independent partition and dedup keys by domain, while the root key
never leaves its service. Publication remains quarantined and grants no model
training authority.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Never

from core.brain.llm.latent_cortex.verified_replay_buffer import (
    ReplayProtector,
    VerifiedReplayBuffer,
    validate_verified_replay_store,
)
from core.governance_context import local_internal_governed_scope
from core.learning.verified_replay_sft import (
    VERIFIED_REPLAY_SFT_CANDIDATE_FILES,
    VERIFIED_REPLAY_SFT_EVALUATOR_FILES,
    build_verified_replay_sft_custody_bundles,
    validate_verified_replay_sft_candidate_artifacts,
    validate_verified_replay_sft_custody_pair,
)
from core.runtime.file_read_gateway import (
    read_stable_bytes,
    read_stable_directory_files,
)
from core.runtime.file_write_gateway import (
    DirectoryFileWriteBatchEntry,
    get_file_write_gateway,
)
from core.runtime.service_registry import get_runtime_service
from core.runtime.state_ownership import state_root

VERIFIED_REPLAY_SFT_PUBLICATION_SCHEMA: Final = (
    "aura.rlc.verified_replay_sft_publication_commit.v1"
)
VERIFIED_REPLAY_SFT_PUBLICATION_REPORT_SCHEMA: Final = (
    "aura.rlc.verified_replay_sft_publication_report.v1"
)
_COMMIT_FILE = ".aura_verified_replay_sft.commit.json"
_LOCK_FILE = ".aura_verified_replay_sft.lock"
_CANDIDATE_MARKER = "verified_replay_candidate_manifest.json"
_EVALUATOR_MARKER = "verified_replay_evaluator_manifest.json"
_MAX_COMMIT_BYTES = 128 * 1024
_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GENERATION_RE = re.compile(r"[0-9a-f]{32}\Z")
_PARTITION_CONTEXT = "rlc.verified_replay_sft.partition.v1"
_DEDUP_CONTEXT = "rlc.verified_replay_sft.dedup.v1"


class VerifiedReplaySFTPublicationError(RuntimeError):
    """A replay projection could not be published or reconstructed safely."""


def _fail(reason: str) -> Never:
    normalized = str(reason or "").strip() or "verified_replay_sft_publication_failed"
    raise VerifiedReplaySFTPublicationError(normalized)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            _fail(f"publication_commit_duplicate_json_key:{key}")
        value[key] = child
    return value


def _reject_constant(value: str) -> Never:
    _fail(f"publication_commit_nonfinite_constant:{value}")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise VerifiedReplaySFTPublicationError(
            "publication_commit_json_invalid"
        ) from exc


def _committed_record(body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(body)
    return {
        **normalized,
        "commit_sha256": hashlib.sha256(
            _canonical_json_bytes(normalized)
        ).hexdigest(),
    }


def _json_snapshot(value: Any, *, code: str) -> Any:
    try:
        return json.loads(
            _canonical_json_bytes(value),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except VerifiedReplaySFTPublicationError:
        raise
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise VerifiedReplaySFTPublicationError(code) from exc


def _lexical_absolute(path: Path) -> Path:
    target = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if target.is_symlink():
        _fail(f"publication_symlink_path_rejected:{target}")
    return target


def _validate_path_chain(path: Path) -> None:
    for component in reversed((path, *path.parents)):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            _fail(f"publication_symlink_path_rejected:{component}")
        if component != path and not stat.S_ISDIR(metadata.st_mode):
            _fail(f"publication_non_directory_ancestor:{component}")


def _validate_directory(path: Path, *, must_exist: bool) -> Path:
    target = _lexical_absolute(path)
    _validate_path_chain(target)
    if target.exists() and not target.is_dir():
        _fail(f"publication_path_not_directory:{target}")
    if must_exist and not target.is_dir():
        _fail(f"publication_directory_missing:{target}")
    if target.is_dir():
        metadata = target.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            _fail(f"publication_directory_not_owner_private:{target}")
    return target


def _validate_pair_directories(
    candidate_directory: Path,
    evaluator_directory: Path,
    *,
    must_exist: bool,
) -> tuple[Path, Path]:
    candidate = _validate_directory(candidate_directory, must_exist=must_exist)
    evaluator = _validate_directory(evaluator_directory, must_exist=must_exist)
    if (
        candidate == evaluator
        or candidate in evaluator.parents
        or evaluator in candidate.parents
        or candidate.parent != evaluator.parent
    ):
        _fail("publication_directories_must_be_distinct_non_nested_siblings")
    root = _validate_directory(candidate.parent, must_exist=True)
    if evaluator.parent != root:
        _fail("publication_directories_must_share_private_root")
    return candidate, evaluator


def _ensure_private_root(root: Path) -> Path:
    target = _validate_directory(root, must_exist=False)
    with local_internal_governed_scope(
        "verified_replay_sft.publication_root",
        domain="file_write",
    ):
        get_file_write_gateway().ensure_directory(
            target,
            source="verified_replay_sft.publication_root",
        )
    return _validate_directory(target, must_exist=True)


@contextmanager
def _publication_lock(root: Path) -> Iterator[None]:
    root = _validate_directory(root, must_exist=True)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int) or nofollow <= 0:
        _fail("publication_lock_requires_nofollow")
    lock_path = root / _LOCK_FILE
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | nofollow,
        0o600,
    )
    try:
        opened = os.fstat(descriptor)
        entry = lock_path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or (opened.st_dev, opened.st_ino) != (entry.st_dev, entry.st_ino)
        ):
            _fail("publication_lock_binding_invalid")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        opened = os.fstat(descriptor)
        if stat.S_IMODE(opened.st_mode) != 0o600:
            _fail("publication_lock_permissions_invalid")
        yield
        after = lock_path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or after.st_uid != os.getuid()
            or stat.S_IMODE(after.st_mode) != 0o600
            or (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino)
        ):
            _fail("publication_lock_binding_changed")
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_commit(path: Path, record: Mapping[str, Any], *, source: str) -> None:
    with local_internal_governed_scope(
        "verified_replay_sft.publication_commit",
        domain="file_write",
    ):
        get_file_write_gateway().write_bytes(
            path,
            _canonical_json_bytes(record),
            source=source,
        )


def _flat_directory_name(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value not in {".", ".."}
        and Path(value).name == value
        and "/" not in value
        and "\\" not in value
        and "\0" not in value
    )


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _read_commit(path: Path) -> dict[str, Any]:
    try:
        raw = read_stable_bytes(path, max_bytes=_MAX_COMMIT_BYTES)
        decoded = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except VerifiedReplaySFTPublicationError:
        raise
    except (OSError, RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise VerifiedReplaySFTPublicationError(
            "publication_commit_json_invalid"
        ) from exc
    common = {
        "schema",
        "state",
        "generation_id",
        "candidate_directory",
        "evaluator_directory",
        "source_store_sha256",
        "source_store_revision",
        "protector_key_provenance",
        "protector_key_identity_sha256",
        "partition_key_commitment_sha256",
        "dedup_key_commitment_sha256",
        "commit_sha256",
    }
    fields = {
        "preparing": common,
        "committed": common
        | {
            "candidate_package_sha256",
            "evaluator_package_sha256",
            "custody_root_sha256",
            "candidate_transaction_id",
            "evaluator_transaction_id",
        },
    }
    state = decoded.get("state") if isinstance(decoded, dict) else None
    generation_id = decoded.get("generation_id") if isinstance(decoded, dict) else None
    revision = decoded.get("source_store_revision") if isinstance(decoded, dict) else None
    if (
        not isinstance(decoded, dict)
        or state not in fields
        or set(decoded) != fields[state]
        or decoded.get("schema") != VERIFIED_REPLAY_SFT_PUBLICATION_SCHEMA
        or _GENERATION_RE.fullmatch(str(generation_id or "")) is None
        or not _flat_directory_name(decoded.get("candidate_directory"))
        or not _flat_directory_name(decoded.get("evaluator_directory"))
        or decoded["candidate_directory"] == decoded["evaluator_directory"]
        or type(revision) is not int
        or revision < 0
        or decoded.get("protector_key_provenance") != "horcrux"
        or any(
            not _sha256(decoded.get(field))
            for field in (
                "source_store_sha256",
                "protector_key_identity_sha256",
                "partition_key_commitment_sha256",
                "dedup_key_commitment_sha256",
            )
        )
        or (
            state == "committed"
            and (
                any(
                    not _sha256(decoded.get(field))
                    for field in (
                        "candidate_package_sha256",
                        "evaluator_package_sha256",
                        "custody_root_sha256",
                    )
                )
                or any(
                    not isinstance(decoded.get(field), str)
                    or not decoded[field]
                    or len(decoded[field]) > 128
                    for field in (
                        "candidate_transaction_id",
                        "evaluator_transaction_id",
                    )
                )
            )
        )
    ):
        _fail("publication_commit_schema_invalid")
    body = dict(decoded)
    observed = body.pop("commit_sha256")
    expected = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
    if observed != expected:
        _fail("publication_commit_commitment_invalid")
    if _canonical_json_bytes(decoded) != raw:
        _fail("publication_commit_not_canonical")
    return decoded


def _assert_exact_artifact_inventory(
    directory: Path,
    *,
    names: tuple[str, ...],
) -> None:
    target = _validate_directory(directory, must_exist=True)
    expected = {*names, ".aura_file_write_batch.lock"}
    try:
        inventory = set(os.listdir(target))
    except OSError as exc:
        raise VerifiedReplaySFTPublicationError(
            f"publication_inventory_read_failed:{target}"
        ) from exc
    if inventory != expected:
        _fail("publication_artifact_inventory_invalid")
    for name in inventory:
        metadata = (target / name).stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            _fail("publication_artifact_inventory_not_private_regular")


def _assert_publication_root_inventory(
    root: Path,
    *,
    candidate_name: str,
    evaluator_name: str,
    allow_missing_evaluator: bool,
) -> None:
    target = _validate_directory(root, must_exist=True)
    required = {candidate_name, _COMMIT_FILE, _LOCK_FILE}
    allowed = {*required, evaluator_name}
    inventory = set(os.listdir(target))
    if not required <= inventory or not inventory <= allowed:
        _fail("publication_root_inventory_invalid")
    if not allow_missing_evaluator and evaluator_name not in inventory:
        _fail("publication_root_inventory_invalid")
    for name in (_COMMIT_FILE, _LOCK_FILE):
        metadata = (target / name).stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            _fail("publication_root_control_file_invalid")
    for name in inventory & {candidate_name, evaluator_name}:
        metadata = (target / name).stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            _fail("publication_root_custody_directory_invalid")


def _read_artifacts(directory: Path, *, names: tuple[str, ...]) -> dict[str, bytes]:
    target = _validate_directory(directory, must_exist=True)
    _assert_exact_artifact_inventory(target, names=names)
    try:
        artifacts = read_stable_directory_files(
            target,
            names=names,
            max_bytes_per_file=_MAX_ARTIFACT_BYTES,
        )
    except OSError as exc:
        raise VerifiedReplaySFTPublicationError(
            f"publication_artifact_read_failed:{target}"
        ) from exc
    _assert_exact_artifact_inventory(target, names=names)
    return artifacts


def read_candidate_publication_with_attestation(
    candidate_directory: Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Read candidate bytes without opening the evaluator directory."""

    candidate = _validate_directory(candidate_directory, must_exist=True)
    commit_path = candidate.parent / _COMMIT_FILE
    before = _read_commit(commit_path)
    if (
        before["state"] != "committed"
        or before["candidate_directory"] != candidate.name
    ):
        _fail("candidate_publication_not_committed")
    _assert_publication_root_inventory(
        candidate.parent,
        candidate_name=before["candidate_directory"],
        evaluator_name=before["evaluator_directory"],
        allow_missing_evaluator=True,
    )
    artifacts = _read_artifacts(
        candidate,
        names=VERIFIED_REPLAY_SFT_CANDIDATE_FILES,
    )
    candidate = validate_verified_replay_sft_candidate_artifacts(artifacts)
    manifest = candidate["manifest"]
    after = _read_commit(commit_path)
    if (
        after != before
        or manifest["candidate_package_sha256"]
        != before["candidate_package_sha256"]
        or manifest["custody_root_sha256"] != before["custody_root_sha256"]
        or manifest["source_store_sha256"] != before["source_store_sha256"]
        or manifest["trainer_ready"] is not False
        or manifest["training_authority"] != "none_quarantined_projection"
    ):
        _fail("candidate_publication_commit_binding_invalid")
    return artifacts, before


def read_candidate_publication(candidate_directory: Path) -> dict[str, bytes]:
    artifacts, _commit = read_candidate_publication_with_attestation(
        candidate_directory
    )
    return artifacts


def read_evaluator_publication(evaluator_directory: Path) -> dict[str, bytes]:
    """Read evaluator bytes only after reconstructing the committed pair."""

    evaluator = _validate_directory(evaluator_directory, must_exist=True)
    commit_path = evaluator.parent / _COMMIT_FILE
    before = _read_commit(commit_path)
    if (
        before["state"] != "committed"
        or before["evaluator_directory"] != evaluator.name
    ):
        _fail("evaluator_publication_not_committed")
    _assert_publication_root_inventory(
        evaluator.parent,
        candidate_name=before["candidate_directory"],
        evaluator_name=before["evaluator_directory"],
        allow_missing_evaluator=False,
    )
    candidate, committed_evaluator = _validate_pair_directories(
        evaluator.parent / before["candidate_directory"],
        evaluator.parent / before["evaluator_directory"],
        must_exist=True,
    )
    if committed_evaluator != evaluator:
        _fail("evaluator_publication_not_committed")
    candidate_artifacts = _read_artifacts(
        candidate,
        names=VERIFIED_REPLAY_SFT_CANDIDATE_FILES,
    )
    evaluator_artifacts = _read_artifacts(
        evaluator,
        names=VERIFIED_REPLAY_SFT_EVALUATOR_FILES,
    )
    pair = validate_verified_replay_sft_custody_pair(
        candidate_artifacts,
        evaluator_artifacts,
    )
    after = _read_commit(commit_path)
    if (
        after != before
        or pair["candidate_manifest"]["candidate_package_sha256"]
        != before["candidate_package_sha256"]
        or pair["evaluator_manifest"]["evaluator_package_sha256"]
        != before["evaluator_package_sha256"]
        or pair["candidate_manifest"]["custody_root_sha256"]
        != before["custody_root_sha256"]
    ):
        _fail("evaluator_publication_commit_binding_invalid")
    return evaluator_artifacts


def validate_publication_directories(
    *,
    candidate_directory: Path,
    evaluator_directory: Path,
) -> dict[str, Any]:
    candidate, evaluator = _validate_pair_directories(
        candidate_directory,
        evaluator_directory,
        must_exist=True,
    )
    return validate_verified_replay_sft_custody_pair(
        read_candidate_publication(candidate),
        read_evaluator_publication(evaluator),
    )


@dataclass(frozen=True, slots=True)
class RuntimeProjectionKeys:
    """Live resident protector and domain-separated projection keys."""

    protector: ReplayProtector
    key_identity_sha256: str
    partition_key: bytes
    dedup_key: bytes


def require_runtime_projection_keys() -> RuntimeProjectionKeys:
    """Resolve only a live Horcrux-backed BlackHole and matching Horcrux."""

    protector = get_runtime_service("black_hole", default=None)
    horcrux = get_runtime_service("horcrux", default=None)
    if (
        protector is None
        or getattr(protector, "encryption_active", False) is not True
        or getattr(protector, "key_provenance", "") != "horcrux"
        or not callable(getattr(protector, "encrypt", None))
        or not callable(getattr(protector, "decrypt", None))
    ):
        _fail("runtime_horcrux_black_hole_required")
    protector_identity = str(
        getattr(protector, "key_identity_sha256", "") or ""
    )
    if not _sha256(protector_identity):
        _fail("runtime_black_hole_key_identity_invalid")
    derive = getattr(horcrux, "derive_subkey", None)
    try:
        horcrux_identity = str(
            getattr(horcrux, "key_identity_sha256", "") or ""
        )
    except RuntimeError as exc:
        raise VerifiedReplaySFTPublicationError(
            "runtime_horcrux_not_initialized"
        ) from exc
    if (
        horcrux is None
        or not callable(derive)
        or horcrux_identity != protector_identity
    ):
        _fail("runtime_horcrux_black_hole_identity_mismatch")
    try:
        partition_key = bytes(derive(_PARTITION_CONTEXT))
        dedup_key = bytes(derive(_DEDUP_CONTEXT))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise VerifiedReplaySFTPublicationError(
            "runtime_horcrux_subkey_derivation_failed"
        ) from exc
    if (
        len(partition_key) != 32
        or len(dedup_key) != 32
        or partition_key == dedup_key
    ):
        _fail("runtime_horcrux_subkeys_invalid")
    return RuntimeProjectionKeys(
        protector=protector,
        key_identity_sha256=protector_identity,
        partition_key=partition_key,
        dedup_key=dedup_key,
    )


def default_verified_replay_sft_publication_root() -> Path:
    override = os.environ.get(
        "AURA_RLC_VERIFIED_REPLAY_SFT_PUBLICATION_ROOT",
        "",
    ).strip()
    if override:
        return Path(override).expanduser()
    try:
        from core.config import config

        home = Path(config.paths.home_dir)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        home = state_root()
    return home / "private" / "rlc" / "verified_replay_sft"


def _context(
    *,
    source_store: Mapping[str, Any],
    key_identity_sha256: str,
    partition_key: bytes,
    dedup_key: bytes,
) -> dict[str, Any]:
    return {
        "source_store_sha256": source_store["store_sha256"],
        "source_store_revision": source_store["revision"],
        "protector_key_provenance": "horcrux",
        "protector_key_identity_sha256": key_identity_sha256,
        "partition_key_commitment_sha256": hashlib.sha256(
            partition_key
        ).hexdigest(),
        "dedup_key_commitment_sha256": hashlib.sha256(dedup_key).hexdigest(),
    }


def _same_generation(
    commit: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    candidate_name: str,
    evaluator_name: str,
    candidate_package_sha256: str,
    evaluator_package_sha256: str,
    custody_root_sha256: str,
) -> bool:
    return all(
        commit.get(key) == value
        for key, value in {
            **dict(context),
            "candidate_directory": candidate_name,
            "evaluator_directory": evaluator_name,
            "candidate_package_sha256": candidate_package_sha256,
            "evaluator_package_sha256": evaluator_package_sha256,
            "custody_root_sha256": custody_root_sha256,
        }.items()
    )


def publish_verified_replay_sft_custody(
    *,
    replay_store: Mapping[str, Any],
    protector: ReplayProtector,
    privacy_clearances: Mapping[str, Mapping[str, Any]],
    reference_index: Mapping[str, Any],
    partition_key: bytes,
    dedup_key: bytes,
    key_identity_sha256: str,
    candidate_directory: Path,
    evaluator_directory: Path,
    partition_ratios: Mapping[str, int] | None = None,
    minimum_rows_per_split: int = 1,
) -> dict[str, Any]:
    """Build, transactionally publish, and reconstruct one replay snapshot."""

    if getattr(protector, "key_provenance", "") != "horcrux":
        _fail("publication_requires_horcrux_protector")
    if not _sha256(key_identity_sha256):
        _fail("publication_key_identity_invalid")
    if getattr(protector, "key_identity_sha256", "") != key_identity_sha256:
        _fail("publication_protector_identity_mismatch")
    store = validate_verified_replay_store(replay_store)
    partition_secret = bytes(partition_key)
    dedup_secret = bytes(dedup_key)
    privacy_inventory = _json_snapshot(
        privacy_clearances,
        code="publication_privacy_inventory_invalid",
    )
    references = _json_snapshot(
        reference_index,
        code="publication_reference_index_invalid",
    )
    ratios = (
        None
        if partition_ratios is None
        else _json_snapshot(
            partition_ratios,
            code="publication_partition_ratios_invalid",
        )
    )
    root = _ensure_private_root(Path(candidate_directory).expanduser().parent)
    candidate, evaluator = _validate_pair_directories(
        candidate_directory,
        evaluator_directory,
        must_exist=False,
    )
    if candidate.parent != root:
        _fail("publication_directories_must_share_private_root")
    bundles = build_verified_replay_sft_custody_bundles(
        replay_store=store,
        protector=protector,
        privacy_clearances=privacy_inventory,
        partition_key=partition_secret,
        dedup_key=dedup_secret,
        reference_index=references,
        partition_ratios=ratios,
        minimum_rows_per_split=minimum_rows_per_split,
    )
    candidate_validation = validate_verified_replay_sft_candidate_artifacts(
        bundles.candidate_artifacts
    )
    candidate_manifest = candidate_validation["manifest"]
    pair = validate_verified_replay_sft_custody_pair(
        bundles.candidate_artifacts,
        bundles.evaluator_artifacts,
    )
    evaluator_manifest = pair["evaluator_manifest"]
    custody_root_sha256 = candidate_manifest["custody_root_sha256"]
    if getattr(protector, "key_identity_sha256", "") != key_identity_sha256:
        _fail("publication_protector_identity_changed")
    context = _context(
        source_store=store,
        key_identity_sha256=key_identity_sha256,
        partition_key=partition_secret,
        dedup_key=dedup_secret,
    )
    if (
        candidate_manifest["source_store_sha256"]
        != context["source_store_sha256"]
    ):
        _fail("publication_source_store_binding_invalid")
    commit_path = root / _COMMIT_FILE
    generation_id = uuid.uuid4().hex
    recovered_generation_id = ""
    superseded_generation_id = ""
    with _publication_lock(root):
        if (
            getattr(protector, "key_provenance", "") != "horcrux"
            or getattr(protector, "key_identity_sha256", "")
            != key_identity_sha256
        ):
            _fail("publication_protector_identity_changed")
        allowed_root_entries = {
            _LOCK_FILE,
            _COMMIT_FILE,
            candidate.name,
            evaluator.name,
        }
        unexpected_root_entries = set(os.listdir(root)) - allowed_root_entries
        if unexpected_root_entries:
            _fail("publication_root_inventory_invalid")
        if os.path.lexists(commit_path):
            prior = _read_commit(commit_path)
            if prior["state"] == "committed" and _same_generation(
                prior,
                context=context,
                candidate_name=candidate.name,
                evaluator_name=evaluator.name,
                candidate_package_sha256=candidate_manifest[
                    "candidate_package_sha256"
                ],
                evaluator_package_sha256=evaluator_manifest[
                    "evaluator_package_sha256"
                ],
                custody_root_sha256=custody_root_sha256,
            ):
                validate_publication_directories(
                    candidate_directory=candidate,
                    evaluator_directory=evaluator,
                )
                return {
                    "schema": VERIFIED_REPLAY_SFT_PUBLICATION_REPORT_SCHEMA,
                    "status": "existing_committed_generation_replay_validated",
                    "generation_id": prior["generation_id"],
                    "recovered_preparing_generation": False,
                    "recovered_generation_id": "",
                    "superseded_committed_generation": False,
                    "superseded_generation_id": "",
                    "candidate_directory": str(candidate),
                    "evaluator_directory": str(evaluator),
                    "candidate_package_sha256": prior[
                        "candidate_package_sha256"
                    ],
                    "evaluator_package_sha256": prior[
                        "evaluator_package_sha256"
                    ],
                    "custody_root_sha256": prior["custody_root_sha256"],
                    "publication_commit_sha256": prior["commit_sha256"],
                    "trainer_ready": False,
                    "training_authority": "none_quarantined_projection",
                }
            if prior["state"] == "committed":
                validate_publication_directories(
                    candidate_directory=candidate,
                    evaluator_directory=evaluator,
                )
                superseded_generation_id = prior["generation_id"]
            else:
                recovered_generation_id = prior["generation_id"]
        preparing = _committed_record(
            {
                "schema": VERIFIED_REPLAY_SFT_PUBLICATION_SCHEMA,
                "state": "preparing",
                "generation_id": generation_id,
                "candidate_directory": candidate.name,
                "evaluator_directory": evaluator.name,
                **context,
            }
        )
        _write_commit(
            commit_path,
            preparing,
            source="verified_replay_sft.publication_preparing",
        )
        gateway = get_file_write_gateway()
        with local_internal_governed_scope(
            "verified_replay_sft.publication_artifacts",
            domain="file_write",
        ):
            gateway.ensure_directory(
                candidate,
                source="verified_replay_sft.candidate_directory",
            )
            gateway.ensure_directory(
                evaluator,
                source="verified_replay_sft.evaluator_directory",
            )
            evaluator_receipt = gateway.write_bytes_batch_in_directory(
                evaluator,
                tuple(
                    DirectoryFileWriteBatchEntry(
                        name,
                        bundles.evaluator_artifacts[name],
                        mode=0o600,
                    )
                    for name in VERIFIED_REPLAY_SFT_EVALUATOR_FILES
                ),
                allowed_existing_names=VERIFIED_REPLAY_SFT_EVALUATOR_FILES,
                commit_marker=_EVALUATOR_MARKER,
                source="verified_replay_sft.evaluator_artifacts",
            )
            candidate_receipt = gateway.write_bytes_batch_in_directory(
                candidate,
                tuple(
                    DirectoryFileWriteBatchEntry(
                        name,
                        bundles.candidate_artifacts[name],
                        mode=0o600,
                    )
                    for name in VERIFIED_REPLAY_SFT_CANDIDATE_FILES
                ),
                allowed_existing_names=VERIFIED_REPLAY_SFT_CANDIDATE_FILES,
                commit_marker=_CANDIDATE_MARKER,
                source="verified_replay_sft.candidate_artifacts",
            )
        durable_candidate = _read_artifacts(
            candidate,
            names=VERIFIED_REPLAY_SFT_CANDIDATE_FILES,
        )
        durable_evaluator = _read_artifacts(
            evaluator,
            names=VERIFIED_REPLAY_SFT_EVALUATOR_FILES,
        )
        durable_pair = validate_verified_replay_sft_custody_pair(
            durable_candidate,
            durable_evaluator,
        )
        durable_candidate_manifest = durable_pair["candidate_manifest"]
        durable_evaluator_manifest = durable_pair["evaluator_manifest"]
        if (
            durable_candidate_manifest != candidate_manifest
            or durable_evaluator_manifest != evaluator_manifest
        ):
            _fail("publication_durable_reconstruction_mismatch")
        committed = _committed_record(
            {
                "schema": VERIFIED_REPLAY_SFT_PUBLICATION_SCHEMA,
                "state": "committed",
                "generation_id": generation_id,
                "candidate_directory": candidate.name,
                "evaluator_directory": evaluator.name,
                **context,
                "candidate_package_sha256": candidate_manifest[
                    "candidate_package_sha256"
                ],
                "evaluator_package_sha256": evaluator_manifest[
                    "evaluator_package_sha256"
                ],
                "custody_root_sha256": custody_root_sha256,
                "candidate_transaction_id": candidate_receipt.transaction_id,
                "evaluator_transaction_id": evaluator_receipt.transaction_id,
            }
        )
        _write_commit(
            commit_path,
            committed,
            source="verified_replay_sft.publication_committed",
        )
        validate_publication_directories(
            candidate_directory=candidate,
            evaluator_directory=evaluator,
        )
    return {
        "schema": VERIFIED_REPLAY_SFT_PUBLICATION_REPORT_SCHEMA,
        "status": "custody_bundles_published_and_replay_validated",
        "generation_id": generation_id,
        "recovered_preparing_generation": bool(recovered_generation_id),
        "recovered_generation_id": recovered_generation_id,
        "superseded_committed_generation": bool(superseded_generation_id),
        "superseded_generation_id": superseded_generation_id,
        "candidate_directory": str(candidate),
        "evaluator_directory": str(evaluator),
        "candidate_transaction_id": candidate_receipt.transaction_id,
        "evaluator_transaction_id": evaluator_receipt.transaction_id,
        "candidate_package_sha256": candidate_manifest[
            "candidate_package_sha256"
        ],
        "evaluator_package_sha256": evaluator_manifest[
            "evaluator_package_sha256"
        ],
        "custody_root_sha256": custody_root_sha256,
        "publication_commit_sha256": committed["commit_sha256"],
        "protector_key_provenance": "horcrux",
        "protector_key_identity_sha256": key_identity_sha256,
        "trainer_ready": False,
        "training_authority": "none_quarantined_projection",
    }


def publish_runtime_verified_replay_sft(
    *,
    privacy_clearances: Mapping[str, Mapping[str, Any]],
    reference_index: Mapping[str, Any],
    publication_root: Path | None = None,
    replay_path: Path | None = None,
    partition_ratios: Mapping[str, int] | None = None,
    minimum_rows_per_split: int = 1,
) -> dict[str, Any]:
    """Publish the current live replay snapshot under resident key custody."""

    keys = require_runtime_projection_keys()
    store = VerifiedReplayBuffer(path=replay_path).load()
    root = _ensure_private_root(
        default_verified_replay_sft_publication_root()
        if publication_root is None
        else publication_root
    )
    return publish_verified_replay_sft_custody(
        replay_store=store,
        protector=keys.protector,
        privacy_clearances=privacy_clearances,
        reference_index=reference_index,
        partition_key=keys.partition_key,
        dedup_key=keys.dedup_key,
        key_identity_sha256=keys.key_identity_sha256,
        candidate_directory=root / "candidate",
        evaluator_directory=root / "evaluator",
        partition_ratios=partition_ratios,
        minimum_rows_per_split=minimum_rows_per_split,
    )


__all__ = [
    "RuntimeProjectionKeys",
    "VERIFIED_REPLAY_SFT_PUBLICATION_REPORT_SCHEMA",
    "VERIFIED_REPLAY_SFT_PUBLICATION_SCHEMA",
    "VerifiedReplaySFTPublicationError",
    "default_verified_replay_sft_publication_root",
    "publish_runtime_verified_replay_sft",
    "publish_verified_replay_sft_custody",
    "read_candidate_publication",
    "read_candidate_publication_with_attestation",
    "read_evaluator_publication",
    "require_runtime_projection_keys",
    "validate_publication_directories",
]
