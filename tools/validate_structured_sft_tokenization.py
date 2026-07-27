#!/usr/bin/env python3
"""Validate candidate SFT masking against an attested tokenizer snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import marshal
import os
import stat
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.governance_context import local_internal_governed_scope  # noqa: E402
from core.learning.structured_sft import (  # noqa: E402
    StructuredSFTCurriculumSpec,
    StructuredSFTError,
    build_structured_sft_curriculum,
    validate_candidate_dataset_artifacts,
    validate_trainer_tokenization,
)
from core.runtime.file_read_gateway import (  # noqa: E402
    read_stable_bytes,
    read_stable_directory_files,
)
from core.runtime.file_write_gateway import (  # noqa: E402
    DirectoryFileWriteBatchEntry,
    get_file_write_gateway,
)
from tools.build_structured_sft_dataset import (  # noqa: E402
    CandidateDatasetBuildError,
    read_candidate_dataset_directory_with_attestation,
)

_ELIGIBLE_SUFFIXES = frozenset(
    {
        ".jinja",
        ".json",
        ".jsonl",
        ".model",
        ".py",
        ".tiktoken",
        ".txt",
    }
)
_DEPENDENCY_DISTRIBUTIONS = ("mlx-lm", "tokenizers", "transformers")
_MAX_TOKENIZER_FILE_BYTES = 512 * 1024 * 1024
_MAX_TOKENIZER_IDENTITY_BYTES = 1024 * 1024 * 1024
_SNAPSHOT_MANIFEST_FILE = "tokenizer_snapshot_manifest.bin"
_SNAPSHOT_SCHEMA = "aura.rlc.tokenizer_snapshot.v1"
RESIDENT_TOKENIZER_SNAPSHOT_SCHEMA = _SNAPSHOT_SCHEMA
_VALIDATION_BUNDLE_SCHEMA = (
    "aura.rlc.structured_sft_tokenizer_validation_bundle.v3"
)


class TokenizerValidationError(RuntimeError):
    """Tokenizer identity or projection validation failed."""


def _is_tokenizer_identity_file(name: str) -> bool:
    return Path(name.lower()).suffix in _ELIGIBLE_SUFFIXES


def _open_directory_no_follow(path: Path) -> tuple[int, Path]:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open("/", flags)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int) or nofollow <= 0:
        raise TokenizerValidationError("tokenizer_nofollow_unsupported")
    try:
        for component in lexical.parts[1:]:
            next_descriptor = os.open(
                component,
                flags | nofollow,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise TokenizerValidationError(
                "tokenizer_directory_not_directory"
            )
        return descriptor, lexical
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _read_identity_files(directory: Path) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    descriptor, target = _open_directory_no_follow(directory)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int) or nofollow <= 0:
        os.close(descriptor)
        raise TokenizerValidationError("tokenizer_nofollow_unsupported")
    try:
        bindings: list[dict[str, Any]] = []
        payloads: dict[str, bytes] = {}
        aggregate = 0
        for entry in sorted(os.scandir(descriptor), key=lambda row: row.name):
            if not _is_tokenizer_identity_file(entry.name):
                continue
            if entry.is_symlink():
                raise TokenizerValidationError(
                    f"tokenizer_identity_symlink_rejected:{entry.name}"
                )
            if not entry.is_file(follow_symlinks=False):
                continue
            file_fd = os.open(
                entry.name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | nofollow,
                dir_fd=descriptor,
            )
            try:
                before = os.fstat(file_fd)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or before.st_uid != os.getuid()
                    or before.st_size < 0
                    or before.st_size > _MAX_TOKENIZER_FILE_BYTES
                ):
                    raise TokenizerValidationError(
                        f"tokenizer_identity_file_invalid:{entry.name}"
                    )
                chunks: list[bytes] = []
                remaining = before.st_size
                while remaining:
                    chunk = os.read(file_fd, min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                payload = b"".join(chunks)
                after = os.fstat(file_fd)
                if (
                    len(payload) != before.st_size
                    or (
                        before.st_dev,
                        before.st_ino,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    )
                    != (
                        after.st_dev,
                        after.st_ino,
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    )
                ):
                    raise TokenizerValidationError(
                        f"tokenizer_identity_file_changed:{entry.name}"
                    )
            finally:
                os.close(file_fd)
            aggregate += len(payload)
            if aggregate > _MAX_TOKENIZER_IDENTITY_BYTES:
                raise TokenizerValidationError(
                    "tokenizer_identity_aggregate_too_large"
                )
            payloads[entry.name] = payload
            bindings.append(
                {
                    "name": entry.name,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        reopened, _ = _open_directory_no_follow(target)
        try:
            original = os.fstat(descriptor)
            current = os.fstat(reopened)
            if (original.st_dev, original.st_ino) != (
                current.st_dev,
                current.st_ino,
            ):
                raise TokenizerValidationError(
                    "tokenizer_directory_changed_during_snapshot"
                )
        finally:
            os.close(reopened)
    finally:
        os.close(descriptor)
    if not bindings or not any(
        row["name"] == "tokenizer_config.json" for row in bindings
    ):
        raise TokenizerValidationError("tokenizer_identity_files_missing")
    return bindings, payloads


def _identity_from_bindings(
    *,
    directory: Path,
    bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    canonical = json.dumps(
        bindings,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return {
        "directory": str(directory),
        "files": bindings,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _tokenizer_identity(directory: Path) -> dict[str, Any]:
    bindings, _payloads = _read_identity_files(directory)
    return _identity_from_bindings(
        directory=Path(os.path.abspath(os.fspath(directory.expanduser()))),
        bindings=bindings,
    )


@contextmanager
def _tokenizer_snapshot(
    directory: Path,
    snapshot_root: Path,
) -> Iterator[tuple[Path, dict[str, Any], dict[str, Any]]]:
    source = Path(os.path.abspath(os.fspath(directory.expanduser())))
    bindings, payloads = _read_identity_files(source)
    source_identity = _identity_from_bindings(
        directory=source,
        bindings=bindings,
    )
    root = Path(
        os.path.abspath(os.fspath(snapshot_root.expanduser()))
    )
    gateway = get_file_write_gateway()
    with local_internal_governed_scope(
        "structured_sft.tokenizer_snapshot",
        domain="file_write",
    ):
        gateway.ensure_directory(
            root,
            source="structured_sft.tokenizer_snapshot_root",
        )
        snapshot = root / source_identity["sha256"]
        manifest_body = {
            "schema": _SNAPSHOT_SCHEMA,
            "tokenizer_identity_sha256": source_identity["sha256"],
            "files": bindings,
        }
        snapshot_manifest = {
            **manifest_body,
            "snapshot_manifest_sha256": hashlib.sha256(
                json.dumps(
                    manifest_body,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest(),
        }
        expected = {
            **payloads,
            _SNAPSHOT_MANIFEST_FILE: json.dumps(
                snapshot_manifest,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii"),
        }
        if snapshot.exists():
            observed = read_stable_directory_files(
                snapshot,
                names=tuple(expected),
                max_bytes_per_file=_MAX_TOKENIZER_FILE_BYTES,
            )
            if observed != expected:
                raise TokenizerValidationError(
                    "tokenizer_content_addressed_snapshot_mismatch"
                )
        else:
            gateway.ensure_directory(
                snapshot,
                source="structured_sft.tokenizer_snapshot",
            )
            gateway.write_bytes_batch_in_directory(
                snapshot,
                tuple(
                    DirectoryFileWriteBatchEntry(
                        name,
                        payload,
                        mode=0o400,
                    )
                    for name, payload in expected.items()
                ),
                allowed_existing_names=tuple(expected),
                commit_marker=_SNAPSHOT_MANIFEST_FILE,
                source="structured_sft.tokenizer_snapshot",
            )
    yield snapshot, source_identity, snapshot_manifest


def _load_bound_tokenizer(directory: Path) -> Any:
    config_path = directory / "config.json"
    try:
        config = json.loads(
            read_stable_bytes(config_path, max_bytes=4 * 1024 * 1024)
        )
        eos_token_ids = config.get("eos_token_id")
        eos_values = (
            eos_token_ids if isinstance(eos_token_ids, list) else [eos_token_ids]
        )
        if not eos_values or any(
            type(token_id) is not int or not 0 <= token_id < 2**31
            for token_id in eos_values
        ):
            raise TokenizerValidationError("tokenizer_eos_contract_invalid")
        from mlx_lm.utils import load_tokenizer

        return load_tokenizer(directory, eos_token_ids=eos_token_ids)
    except TokenizerValidationError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TokenizerValidationError("tokenizer_dependency_unavailable") from exc


def _module_artifact_identity(module_name: str) -> dict[str, str]:
    try:
        module = importlib.import_module(module_name)
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            raise TokenizerValidationError(
                "tokenizer_module_artifact_unavailable"
            )
        path = Path(raw_path).resolve(strict=True)
        payload = read_stable_bytes(
            path,
            max_bytes=_MAX_TOKENIZER_FILE_BYTES,
        )
    except TokenizerValidationError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TokenizerValidationError(
            "tokenizer_module_artifact_unavailable"
        ) from exc
    return {
        "module": module_name,
        "artifact_name": path.name,
        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _closure_value_identity(
    value: Any,
    *,
    depth: int = 0,
    seen: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    if depth > 12:
        raise TokenizerValidationError("tokenizer_callable_closure_too_deep")
    value_type = type(value)
    identity: dict[str, Any] = {
        "type_module": value_type.__module__,
        "type_qualname": value_type.__qualname__,
    }
    if value is None or type(value) in {bool, int, float, str}:
        payload = repr(value).encode("utf-8", errors="surrogatepass")
        return {**identity, "kind": "scalar", "value_sha256": hashlib.sha256(payload).hexdigest()}
    if isinstance(value, bytes):
        return {**identity, "kind": "bytes", "value_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Path):
        payload = os.fspath(value).encode("utf-8", errors="surrogatepass")
        return {**identity, "kind": "path", "value_sha256": hashlib.sha256(payload).hexdigest()}
    object_id = id(value)
    if object_id in seen:
        raise TokenizerValidationError("tokenizer_callable_closure_cycle")
    next_seen = seen | {object_id}
    if callable(value):
        effective = value.__func__ if inspect.ismethod(value) else value
        code = getattr(effective, "__code__", None)
        nested_closure: list[dict[str, Any]] = []
        for cell in getattr(effective, "__closure__", None) or ():
            try:
                nested_closure.append(
                    _closure_value_identity(
                        cell.cell_contents,
                        depth=depth + 1,
                        seen=next_seen,
                    )
                )
            except ValueError:
                nested_closure.append({"kind": "empty_cell"})
        return {
            **identity,
            "kind": "callable",
            "module": str(getattr(effective, "__module__", "") or ""),
            "qualname": str(getattr(effective, "__qualname__", "") or ""),
            "runtime_code_sha256": (
                hashlib.sha256(marshal.dumps(code)).hexdigest()
                if code is not None
                else None
            ),
            "defaults_sha256": hashlib.sha256(
                repr(
                    (
                        getattr(effective, "__defaults__", None),
                        getattr(effective, "__kwdefaults__", None),
                    )
                ).encode("utf-8", errors="surrogatepass")
            ).hexdigest(),
            "closure_cells": nested_closure,
        }
    if isinstance(value, (tuple, list)):
        if len(value) > 64:
            raise TokenizerValidationError("tokenizer_callable_closure_too_large")
        return {
            **identity,
            "kind": "sequence",
            "items": [
                _closure_value_identity(
                    item,
                    depth=depth + 1,
                    seen=next_seen,
                )
                for item in value
            ],
        }
    if isinstance(value, Mapping):
        if len(value) > 64 or any(not isinstance(key, str) for key in value):
            raise TokenizerValidationError("tokenizer_callable_closure_mapping_invalid")
        return {
            **identity,
            "kind": "mapping",
            "items": {
                key: _closure_value_identity(
                    value[key],
                    depth=depth + 1,
                    seen=next_seen,
                )
                for key in sorted(value)
            },
        }
    state = getattr(value, "__dict__", None)
    if isinstance(state, Mapping):
        return {
            **identity,
            "kind": "object_state",
            "state": _closure_value_identity(
                state,
                depth=depth + 1,
                seen=next_seen,
            ),
        }
    raise TokenizerValidationError(
        "tokenizer_callable_closure_identity_unavailable"
    )


def _callable_identity(
    value: Any,
    *,
    module_hint: str = "",
    qualname_hint: str = "",
) -> dict[str, Any]:
    effective = inspect.unwrap(
        value.__func__ if inspect.ismethod(value) else value
    )
    module_name = str(
        getattr(effective, "__module__", "") or module_hint
    )
    qualname = str(
        getattr(effective, "__qualname__", "") or qualname_hint
    )
    if not module_name or not qualname:
        raise TokenizerValidationError(
            "tokenizer_callable_identity_unavailable"
        )
    code = getattr(effective, "__code__", None)
    runtime_code_sha256 = (
        hashlib.sha256(marshal.dumps(code)).hexdigest()
        if code is not None
        else None
    )
    defaults_sha256 = hashlib.sha256(
        repr(
            (
                getattr(effective, "__defaults__", None),
                getattr(effective, "__kwdefaults__", None),
            )
        ).encode("utf-8")
    ).hexdigest()
    closure_rows: list[dict[str, Any]] = []
    for cell in getattr(effective, "__closure__", None) or ():
        try:
            cell_value = cell.cell_contents
            closure_rows.append(_closure_value_identity(cell_value))
        except ValueError:
            closure_rows.append({"kind": "empty_cell"})
    closure_sha256 = hashlib.sha256(
        json.dumps(
            closure_rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    source_identity: dict[str, str] | None = None
    try:
        source_path = inspect.getsourcefile(effective)
    except TypeError:
        source_path = None
    if source_path:
        try:
            source = Path(source_path).resolve(strict=True)
            payload = read_stable_bytes(
                source,
                max_bytes=_MAX_TOKENIZER_FILE_BYTES,
            )
        except OSError:
            # Loaded code can retain a co_filename from a deleted build or
            # worktree. The runtime code and current module artifact below
            # remain mandatory identities; only this optional source facet is
            # unavailable.
            source_identity = None
        else:
            try:
                callable_source = inspect.getsource(effective).encode("utf-8")
            except (OSError, TypeError):
                callable_source = b""
            source_identity = {
                "distribution_relative_source": "/".join(source.parts[-3:]),
                "source_file_sha256": hashlib.sha256(payload).hexdigest(),
                "callable_source_sha256": hashlib.sha256(
                    callable_source
                ).hexdigest(),
            }
    module_artifact = _module_artifact_identity(module_name)
    return {
        "module": module_name,
        "qualname": qualname,
        "runtime_code_sha256": runtime_code_sha256,
        "defaults_sha256": defaults_sha256,
        "closure_sha256": closure_sha256,
        "closure_cells": closure_rows,
        "module_artifact": module_artifact,
        "disk_source": source_identity,
    }


def _class_callable_identities(value: Any) -> dict[str, Any]:
    identities: dict[str, Any] = {}
    for name in ("apply_chat_template", "encode", "decode", "_tokenize"):
        candidate = getattr(value, name, None)
        if callable(candidate):
            identities[name] = _callable_identity(
                candidate,
                module_hint=type(value).__module__,
                qualname_hint=f"{type(value).__qualname__}.{name}",
            )
    if not identities:
        raise TokenizerValidationError(
            "tokenizer_effective_callables_missing"
        )
    return identities


def _effective_chat_template_identity(tokenizer: Any) -> dict[str, Any]:
    effective = getattr(tokenizer, "_chat_template", None)
    source = "_chat_template"
    if effective is None:
        effective = getattr(tokenizer, "chat_template", None)
        source = "chat_template"
    if callable(effective):
        return {
            "source": source,
            "kind": "callable",
            "callable": _callable_identity(effective),
        }
    rendered = str(effective or "")
    return {
        "source": source,
        "kind": "literal",
        "type_module": type(effective).__module__,
        "type_qualname": type(effective).__qualname__,
        "value_sha256": hashlib.sha256(
            rendered.encode("utf-8", errors="surrogatepass")
        ).hexdigest(),
    }


def _compiled_package_artifacts(module_name: str) -> list[dict[str, Any]]:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return []
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        return []
    root = Path(raw_path).resolve(strict=True).parent
    rows: list[dict[str, Any]] = []
    aggregate = 0
    for path in sorted(
        (
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file()
            and candidate.suffix.lower() in {".so", ".dylib", ".pyd"}
        ),
        key=lambda candidate: str(candidate.relative_to(root)),
    ):
        payload = read_stable_bytes(
            path,
            max_bytes=_MAX_TOKENIZER_FILE_BYTES,
        )
        aggregate += len(payload)
        if aggregate > _MAX_TOKENIZER_IDENTITY_BYTES or len(rows) >= 256:
            raise TokenizerValidationError(
                "tokenizer_compiled_artifact_bound_exceeded"
            )
        rows.append(
            {
                "module_root": module_name,
                "name": str(path.relative_to(root)),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return rows


def _dependency_identity(tokenizer: Any) -> dict[str, Any]:
    try:
        from mlx_lm.tokenizer_utils import TokenizerWrapper, load
        from mlx_lm.tuner.datasets import ChatDataset
        from mlx_lm.utils import load_tokenizer
    except ImportError as exc:
        raise TokenizerValidationError("tokenizer_dependency_unavailable") from exc
    versions: dict[str, str] = {}
    for distribution in _DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    wrapped = getattr(tokenizer, "_tokenizer", tokenizer)
    body = {
        "dependency_versions": versions,
        "effective_callables": {
            "mlx_load": _callable_identity(load),
            "public_loader": _callable_identity(load_tokenizer),
            "chat_template": _callable_identity(
                TokenizerWrapper.apply_chat_template
            ),
            "chat_dataset_process": _callable_identity(ChatDataset.process),
        },
        "tokenizer_class": {
            "module": type(tokenizer).__module__,
            "qualname": type(tokenizer).__qualname__,
        },
        "wrapped_tokenizer_class": {
            "module": type(wrapped).__module__,
            "qualname": type(wrapped).__qualname__,
        },
        "tokenizer_runtime_callables": _class_callable_identities(tokenizer),
        "wrapped_tokenizer_runtime_callables": (
            _class_callable_identities(wrapped)
        ),
        "effective_chat_template": _effective_chat_template_identity(
            tokenizer
        ),
        "tokenizer_class_module_artifact": _module_artifact_identity(
            type(tokenizer).__module__
        ),
        "wrapped_class_module_artifact": _module_artifact_identity(
            type(wrapped).__module__
        ),
        "compiled_dependency_artifacts": {
            module_name: _compiled_package_artifacts(module_name)
            for module_name in ("mlx", "mlx_lm", "tokenizers", "transformers")
        },
        "loaded_chat_template_sha256": hashlib.sha256(
            str(getattr(tokenizer, "chat_template", "") or "").encode("utf-8")
        ).hexdigest(),
    }
    return {
        **body,
        "sha256": hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest(),
    }


def resident_tokenizer_snapshot(
    directory: Path,
    snapshot_root: Path,
):
    """Return the shared no-follow, content-addressed tokenizer snapshot."""

    return _tokenizer_snapshot(directory, snapshot_root)


def load_resident_tokenizer(directory: Path) -> Any:
    """Load a tokenizer only after binding its configured EOS contract."""

    return _load_bound_tokenizer(directory)


def resident_tokenizer_artifact_identity(directory: Path) -> dict[str, Any]:
    """Recompute the complete loader-eligible tokenizer artifact identity."""

    return _tokenizer_identity(directory)


def resident_tokenizer_runtime_identity(tokenizer: Any) -> dict[str, Any]:
    """Bind Python, native, template, and ChatDataset runtime behavior."""

    return _dependency_identity(tokenizer)


def validate(
    *,
    candidate_directory: Path,
    tokenizer_directory: Path,
    snapshot_root: Path,
) -> dict[str, Any]:
    artifacts, custody_attestation = (
        read_candidate_dataset_directory_with_attestation(candidate_directory)
    )
    manifest = validate_candidate_dataset_artifacts(artifacts)
    spec = StructuredSFTCurriculumSpec(
        **dict(manifest["curriculum_manifest"]["spec"])
    )
    visible_curriculum = build_structured_sft_curriculum(
        spec,
        holdout_seed=b"\0" * 32,
    )
    with _tokenizer_snapshot(tokenizer_directory, snapshot_root) as (
        snapshot,
        tokenizer_identity,
        snapshot_manifest,
    ):
        tokenizer = _load_bound_tokenizer(snapshot)
        dependency_before = _dependency_identity(tokenizer)
        report = validate_trainer_tokenization(
            visible_curriculum,
            tokenizer=tokenizer,
        )
        dependency_after = _dependency_identity(tokenizer)
        if dependency_after != dependency_before:
            raise TokenizerValidationError(
                "tokenizer_runtime_identity_changed_during_validation"
            )
        snapshot_identity = _tokenizer_identity(snapshot)
        if (
            snapshot_identity["files"] != tokenizer_identity["files"]
            or snapshot_identity["sha256"] != tokenizer_identity["sha256"]
        ):
            raise TokenizerValidationError(
                "tokenizer_snapshot_changed_during_validation"
            )
    projection_schema = report.pop("schema")
    projection_curriculum_sha256 = report.pop("curriculum_sha256")
    projection_report_sha256 = report.pop("report_sha256")
    body = {
        **report,
        "schema": _VALIDATION_BUNDLE_SCHEMA,
        "projection_schema": projection_schema,
        "projection_curriculum_sha256": projection_curriculum_sha256,
        "projection_report_sha256": projection_report_sha256,
        "candidate_curriculum_commitment_sha256": manifest[
            "curriculum_manifest"
        ]["curriculum_sha256"],
        "tokenization_scope": "candidate_train_validation_only",
        "candidate_package_sha256": manifest["package_sha256"],
        "candidate_validation_scope": manifest["validation_scope"],
        "candidate_custody_attestation": {
            "schema": custody_attestation["schema"],
            "generation_id": custody_attestation["generation_id"],
            "candidate_package_sha256": custody_attestation[
                "candidate_package_sha256"
            ],
            "evaluator_package_sha256": custody_attestation[
                "evaluator_package_sha256"
            ],
            "custody_root_sha256": custody_attestation[
                "custody_root_sha256"
            ],
            "custody_report_sha256": custody_attestation[
                "custody_report_sha256"
            ],
            "commit_sha256": custody_attestation["commit_sha256"],
            "evaluator_filesystem_accessed": False,
        },
        "tokenizer": {
            **tokenizer_identity,
            "loaded_from_persistent_content_addressed_snapshot": True,
            "snapshot_path": str(snapshot),
            "snapshot_manifest": snapshot_manifest,
            "runtime": dependency_before,
        },
        "trainer_binding_contract": {
            "tokenizer_path": str(snapshot),
            "tokenizer_identity_sha256": tokenizer_identity["sha256"],
            "runtime_identity_sha256": dependency_before["sha256"],
            "snapshot_manifest_sha256": snapshot_manifest[
                "snapshot_manifest_sha256"
            ],
            "revalidate_in_trainer_process": True,
            "candidate_only_revalidation": True,
            "evaluator_filesystem_access_required": False,
            "path_substitution_allowed": False,
        },
    }
    return {
        **body,
        "validation_bundle_sha256": hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        report = validate(
            candidate_directory=arguments.candidate_dir,
            tokenizer_directory=arguments.tokenizer_dir,
            snapshot_root=arguments.snapshot_root,
        )
    except (
        CandidateDatasetBuildError,
        OSError,
        StructuredSFTError,
        TokenizerValidationError,
        ValueError,
    ) as exc:
        print(
            "validate_structured_sft_tokenization: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
