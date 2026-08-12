#!/usr/bin/env python3
"""Publish, inspect, or revoke typed recurrent serving authority."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.brain.llm.unified_recurrent_qualified_activation import (  # noqa: E402
    activation_matches_shadow_receipt,
    seal_qualified_activation,
)
from core.brain.llm.unified_recurrent_qualified_activation_store import (  # noqa: E402
    deactivate_qualified_activation,
    default_qualified_activation_path,
    publish_qualified_activation,
    read_qualified_activation,
)
from core.brain.llm.unified_recurrent_shadow import (  # noqa: E402
    inspect_shadow_package,
)
from core.brain.llm.unified_recurrent_shadow_pointer import (  # noqa: E402
    deactivate_shadow_pointer,
    default_shadow_activation_paths,
    publish_shadow_pointer,
    read_shadow_pointer,
    resolve_shadow_pointer,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402

MAX_LIFECYCLE_BYTES: Final = 4 * 1024 * 1024


class UnifiedRecurrentQualifiedActivationCommandError(RuntimeError):
    """A qualified activation operation could not preserve exact custody."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _read_lifecycle(path: Path) -> dict[str, Any]:
    target = path.expanduser().absolute()
    if target.is_symlink():
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified lifecycle result is a symlink"
        )
    try:
        before = target.stat()
        payload = read_stable_bytes(target, max_bytes=MAX_LIFECYCLE_BYTES)
        after = target.stat()
        value = json.loads(payload.decode("ascii"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified lifecycle result is unreadable"
        ) from exc
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or not 0 < before.st_size <= MAX_LIFECYCLE_BYTES
        or identity(before) != identity(after)
        or not isinstance(value, dict)
        or payload not in {_canonical_bytes(value), _canonical_bytes(value) + b"\n"}
    ):
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified lifecycle result custody differs"
        )
    return value


def _paths(arguments: argparse.Namespace) -> tuple[Path, Path, Path]:
    default_pointer, default_releases = default_shadow_activation_paths()
    pointer = (
        arguments.pointer.expanduser().absolute()
        if arguments.pointer is not None
        else default_pointer
    )
    releases = (
        arguments.releases_root.expanduser().absolute()
        if arguments.releases_root is not None
        else default_releases
    )
    activation = (
        arguments.activation.expanduser().absolute()
        if arguments.activation is not None
        else default_qualified_activation_path()
    )
    if pointer.parent != activation.parent:
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified activation and shadow pointer custody differ"
        )
    return pointer, releases, activation


def _shadow_receipt(
    manifest: Mapping[str, Any],
    *,
    controller_sha256: str,
) -> dict[str, Any]:
    domain = manifest.get("domain_contract")
    if not isinstance(domain, Mapping):
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified package domain contract is unavailable"
        )
    return {
        "package_id": manifest.get("package_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "controller_sha256": controller_sha256,
        "families": domain.get("families"),
        "task_depths": domain.get("task_depths"),
        "recurrence_depth": domain.get("recurrence_depth"),
    }


def _activate(arguments: argparse.Namespace) -> dict[str, Any]:
    pointer_path, releases_root, activation_path = _paths(arguments)
    package = arguments.package.expanduser().absolute()
    verified = inspect_shadow_package(package)
    manifest = verified.get("manifest")
    if not isinstance(manifest, Mapping):
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified package manifest is unavailable"
        )
    lifecycle = _read_lifecycle(arguments.lifecycle_result)
    pointer_preexisted = pointer_path.exists() or pointer_path.is_symlink()
    activation_preexisted = activation_path.exists() or activation_path.is_symlink()
    if pointer_preexisted and resolve_shadow_pointer(
        pointer_path,
        releases_root=releases_root,
    ) != package:
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified activation refuses to replace another shadow package"
        )
    pointer: dict[str, Any] | None = None
    published: dict[str, Any] | None = None
    try:
        pointer = publish_shadow_pointer(
            package,
            pointer_path=pointer_path,
            releases_root=releases_root,
            expected_current_sha256=arguments.expected_current_pointer_sha256,
        )
        activation = seal_qualified_activation(manifest, lifecycle, pointer)
        published = publish_qualified_activation(
            activation,
            activation_path=activation_path,
            shadow_pointer_path=pointer_path,
            expected_current_sha256=arguments.expected_current_activation_sha256,
        )
        resolved = resolve_shadow_pointer(pointer_path, releases_root=releases_root)
        if resolved != package or not activation_matches_shadow_receipt(
            published,
            _shadow_receipt(
                manifest,
                controller_sha256=str(lifecycle.get("controller_sha256") or ""),
            ),
        ):
            raise UnifiedRecurrentQualifiedActivationCommandError(
                "qualified activation reopened a different package identity"
            )
    except BaseException as exc:  # noqa: BLE001 - preserve rollback context
        if published is not None and not activation_preexisted:
            try:
                deactivate_qualified_activation(
                    activation_path=activation_path,
                    expected_current_sha256=published["activation_sha256"],
                )
            except BaseException as rollback_exc:  # noqa: BLE001
                exc.add_note(f"qualified authority rollback also failed: {rollback_exc}")
        if pointer is not None and not pointer_preexisted:
            try:
                deactivate_shadow_pointer(
                    pointer_path=pointer_path,
                    releases_root=releases_root,
                    expected_current_sha256=pointer["pointer_sha256"],
                )
            except BaseException as rollback_exc:  # noqa: BLE001
                exc.add_note(f"shadow pointer rollback also failed: {rollback_exc}")
        raise
    if pointer is None or published is None:  # pragma: no cover - guarded above
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified activation publication returned no authority"
        )
    return {
        "action": "activate",
        "active": True,
        "package": str(package),
        "pointer_path": str(pointer_path),
        "activation_path": str(activation_path),
        "pointer_sha256": pointer["pointer_sha256"],
        "activation_sha256": published["activation_sha256"],
        "mode": published["mode"],
        "families": published["families"],
        "task_depths": published["task_depths"],
    }


def _deactivate(arguments: argparse.Namespace) -> dict[str, Any]:
    pointer_path, releases_root, activation_path = _paths(arguments)
    activation = deactivate_qualified_activation(
        activation_path=activation_path,
        expected_current_sha256=arguments.expected_current_activation_sha256,
    )
    pointer = deactivate_shadow_pointer(
        pointer_path=pointer_path,
        releases_root=releases_root,
        expected_current_sha256=arguments.expected_current_pointer_sha256,
    )
    if activation["pointer_sha256"] != pointer["pointer_sha256"]:
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "retired qualified activation and pointer identity differ"
        )
    return {
        "action": "deactivate",
        "active": False,
        "activation_sha256": activation["activation_sha256"],
        "pointer_sha256": pointer["pointer_sha256"],
        "pointer_path": str(pointer_path),
        "activation_path": str(activation_path),
    }


def _status(arguments: argparse.Namespace) -> dict[str, Any]:
    pointer_path, releases_root, activation_path = _paths(arguments)
    has_pointer = pointer_path.exists() or pointer_path.is_symlink()
    has_activation = activation_path.exists() or activation_path.is_symlink()
    if not has_activation:
        return {
            "action": "status",
            "active": False,
            "shadow_pointer_active": has_pointer,
            "pointer_path": str(pointer_path),
            "activation_path": str(activation_path),
        }
    if not has_pointer:
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified authority exists without a shadow pointer"
        )
    activation = read_qualified_activation(activation_path)
    pointer = read_shadow_pointer(pointer_path)
    package = resolve_shadow_pointer(pointer_path, releases_root=releases_root)
    manifest = inspect_shadow_package(package).get("manifest")
    if (
        not isinstance(manifest, Mapping)
        or activation["pointer_sha256"] != pointer["pointer_sha256"]
        or not activation_matches_shadow_receipt(
            activation,
            _shadow_receipt(
                manifest,
                controller_sha256=str(activation.get("controller_sha256") or ""),
            ),
        )
    ):
        raise UnifiedRecurrentQualifiedActivationCommandError(
            "qualified authority no longer matches its active package"
        )
    return {
        "action": "status",
        "active": True,
        "package": str(package),
        "pointer_path": str(pointer_path),
        "activation_path": str(activation_path),
        "pointer_sha256": pointer["pointer_sha256"],
        "activation_sha256": activation["activation_sha256"],
        "mode": activation["mode"],
        "families": activation["families"],
        "task_depths": activation["task_depths"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer", type=Path)
    parser.add_argument("--releases-root", type=Path)
    parser.add_argument("--activation", type=Path)
    subparsers = parser.add_subparsers(dest="action", required=True)

    activate = subparsers.add_parser("activate")
    activate.add_argument("package", type=Path)
    activate.add_argument("--lifecycle-result", type=Path, required=True)
    activate.add_argument("--expected-current-pointer-sha256")
    activate.add_argument("--expected-current-activation-sha256")

    deactivate = subparsers.add_parser("deactivate")
    deactivate.add_argument("--expected-current-pointer-sha256", required=True)
    deactivate.add_argument("--expected-current-activation-sha256", required=True)

    subparsers.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = {
            "activate": _activate,
            "deactivate": _deactivate,
            "status": _status,
        }[arguments.action](arguments)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"unified qualified activation operation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
