"""Durable custody for domain-qualified recurrent serving authority."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Never

from core.governance_context import local_internal_governed_scope
from core.runtime.atomic_writer import ensure_private_directory, interprocess_file_lock
from core.runtime.file_read_gateway import read_stable_bytes
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root

from .unified_recurrent_qualified_activation import qualified_activation_errors
from .unified_recurrent_shadow_pointer import (
    read_shadow_pointer,
    shadow_pointer_publication_lock_path,
)

MAX_ACTIVATION_BYTES: Final = 64 * 1024


class UnifiedRecurrentQualifiedActivationStoreError(RuntimeError):
    """Qualified serving authority is absent, stale, or outside owned custody."""


def _fail(message: str) -> Never:
    raise UnifiedRecurrentQualifiedActivationStoreError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def default_qualified_activation_path() -> Path:
    return state_root() / "data/adapters/unified-recurrent-shadow/qualified-active.json"


def _reject_symlink_chain(path: Path, *, must_exist: bool) -> Path:
    lexical = path.expanduser().absolute()
    current = Path(lexical.anchor)
    for index, part in enumerate(lexical.parts[1:]):
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if not must_exist and index == len(lexical.parts[1:]) - 1:
                return lexical
            _fail("qualified activation path is unavailable")
        except OSError as exc:
            raise UnifiedRecurrentQualifiedActivationStoreError(
                "qualified activation path is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            _fail("qualified activation path contains a symlink")
    return lexical


def _strict_json(payload: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = item
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, ValueError) as exc:
        raise UnifiedRecurrentQualifiedActivationStoreError(
            "qualified activation JSON is invalid"
        ) from exc
    if not isinstance(value, dict) or _canonical_bytes(value) != payload:
        _fail("qualified activation is not canonical")
    return value


def read_qualified_activation(path: Path | None = None) -> dict[str, Any]:
    """Read a stable, private, canonical activation document."""

    target = _reject_symlink_chain(
        default_qualified_activation_path() if path is None else path,
        must_exist=True,
    )
    try:
        before = target.stat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or not 0 < before.st_size <= MAX_ACTIVATION_BYTES
        ):
            _fail("qualified activation custody differs")
        payload = read_stable_bytes(target, max_bytes=MAX_ACTIVATION_BYTES)
        after = target.stat()
    except (OSError, ValueError) as exc:
        raise UnifiedRecurrentQualifiedActivationStoreError(
            "qualified activation is unreadable"
        ) from exc
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(payload) != before.st_size:
        _fail("qualified activation changed while reading")
    activation = _strict_json(payload)
    errors = qualified_activation_errors(activation)
    if errors:
        _fail("qualified activation is invalid:" + ",".join(errors))
    return activation


def publish_qualified_activation(
    activation: Mapping[str, Any],
    *,
    activation_path: Path | None = None,
    shadow_pointer_path: Path,
    expected_current_sha256: str | None = None,
) -> dict[str, Any]:
    """Linearize authority publication against active shadow replacement."""

    target = default_qualified_activation_path() if activation_path is None else activation_path
    target = target.expanduser().absolute()
    pointer = shadow_pointer_path.expanduser().absolute()
    if target.parent != pointer.parent:
        _fail("qualified activation and shadow pointer custody differ")
    ensure_private_directory(target.parent)
    with interprocess_file_lock(shadow_pointer_publication_lock_path(pointer)):
        return _publish_qualified_activation_locked(
            activation,
            activation_path=target,
            shadow_pointer_path=pointer,
            expected_current_sha256=expected_current_sha256,
        )


def _publish_qualified_activation_locked(
    activation: Mapping[str, Any],
    *,
    activation_path: Path | None = None,
    shadow_pointer_path: Path,
    expected_current_sha256: str | None = None,
) -> dict[str, Any]:
    """CAS-publish authority only while its exact shadow pointer is active."""

    value = dict(activation)
    errors = qualified_activation_errors(value)
    if errors:
        _fail("qualified activation publication is invalid:" + ",".join(errors))
    if value.get("mode") not in {"qualified_typed_pending", "qualified_typed_only"}:
        _fail("qualified activation publication requires persisted typed authority")
    pointer = read_shadow_pointer(shadow_pointer_path)
    if any(
        (
            pointer.get("package_id") != value.get("package_id"),
            pointer.get("manifest_sha256") != value.get("manifest_sha256"),
            pointer.get("pointer_sha256") != value.get("pointer_sha256"),
        )
    ):
        _fail("qualified activation shadow pointer identity differs")
    target = default_qualified_activation_path() if activation_path is None else activation_path
    target = _reject_symlink_chain(target, must_exist=False)
    ensure_private_directory(target.parent)
    payload = _canonical_bytes(value)
    lock_path = target.parent / ".qualified-activation.lock"
    with interprocess_file_lock(lock_path):
        if target.exists() or target.is_symlink():
            current = read_qualified_activation(target)
            if current == value:
                return current
            if (
                expected_current_sha256 is None
                or current["activation_sha256"] != expected_current_sha256
            ):
                _fail("qualified activation publication lost compare-and-swap")
        elif expected_current_sha256 is not None:
            _fail("qualified activation expected current value is absent")
        with local_internal_governed_scope(
            "unified_recurrent_qualified_activation_publish"
        ):
            get_file_write_gateway().write_bytes(
                target,
                payload,
                source="unified_recurrent_qualified_activation.publish",
            )
        reopened = read_qualified_activation(target)
        if reopened != value:
            _fail("qualified activation publication reopened different authority")
        return reopened


def deactivate_qualified_activation(
    *,
    activation_path: Path | None = None,
    expected_current_sha256: str,
) -> dict[str, Any]:
    """Revoke authority under CAS while retaining an immutable private receipt."""

    target = default_qualified_activation_path() if activation_path is None else activation_path
    lock_path = target.expanduser().absolute().parent / ".qualified-activation.lock"
    with interprocess_file_lock(lock_path):
        current = read_qualified_activation(target)
        if current["activation_sha256"] != expected_current_sha256:
            _fail("qualified activation deactivation lost compare-and-swap")
        retired = ensure_private_directory(target.parent / "qualified-retired")
        archive = retired / f"{expected_current_sha256}.json"
        payload = _canonical_bytes(current)
        with local_internal_governed_scope(
            "unified_recurrent_qualified_activation_deactivate"
        ):
            gateway = get_file_write_gateway()
            created = gateway.write_bytes_if_absent(
                archive,
                payload,
                mode=0o600,
                source="unified_recurrent_qualified_activation.retire",
            )
            if not created and read_qualified_activation(archive) != current:
                _fail("qualified activation retirement receipt differs")
            gateway.delete_file(
                target,
                source="unified_recurrent_qualified_activation.deactivate",
            )
        if target.exists() or target.is_symlink():
            _fail("qualified activation remained active after revocation")
        return current


__all__ = [
    "MAX_ACTIVATION_BYTES",
    "UnifiedRecurrentQualifiedActivationStoreError",
    "deactivate_qualified_activation",
    "default_qualified_activation_path",
    "publish_qualified_activation",
    "read_qualified_activation",
]
