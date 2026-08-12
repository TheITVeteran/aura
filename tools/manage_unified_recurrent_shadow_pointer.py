#!/usr/bin/env python3
"""Publish, inspect, or roll back Aura's non-serving recurrent shadow pointer."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.brain.llm.unified_recurrent_shadow_pointer import (  # noqa: E402
    UnifiedRecurrentShadowPointerError,
    deactivate_shadow_pointer,
    default_shadow_activation_paths,
    publish_shadow_pointer,
    resolve_shadow_pointer,
)


def _paths(arguments: argparse.Namespace) -> tuple[Path, Path]:
    default_pointer, default_releases = default_shadow_activation_paths()
    return (
        arguments.pointer.expanduser().absolute()
        if arguments.pointer is not None
        else default_pointer,
        arguments.releases_root.expanduser().absolute()
        if arguments.releases_root is not None
        else default_releases,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer", type=Path)
    parser.add_argument("--releases-root", type=Path)
    subparsers = parser.add_subparsers(dest="action", required=True)

    activate = subparsers.add_parser("activate")
    activate.add_argument("package", type=Path)
    activate.add_argument("--expected-current-sha256")

    deactivate = subparsers.add_parser("deactivate")
    deactivate.add_argument("--expected-current-sha256", required=True)

    subparsers.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    pointer_path, releases_root = _paths(arguments)
    try:
        if arguments.action == "activate":
            pointer = publish_shadow_pointer(
                arguments.package,
                pointer_path=pointer_path,
                releases_root=releases_root,
                expected_current_sha256=arguments.expected_current_sha256,
            )
            result = {
                "active": True,
                "action": "activate",
                "pointer": pointer,
                "pointer_path": str(pointer_path),
            }
        elif arguments.action == "deactivate":
            pointer = deactivate_shadow_pointer(
                pointer_path=pointer_path,
                releases_root=releases_root,
                expected_current_sha256=arguments.expected_current_sha256,
            )
            result = {
                "active": False,
                "action": "deactivate",
                "retired_pointer_sha256": pointer["pointer_sha256"],
                "pointer_path": str(pointer_path),
            }
        elif pointer_path.exists() or pointer_path.is_symlink():
            package = resolve_shadow_pointer(
                pointer_path,
                releases_root=releases_root,
            )
            result = {
                "active": True,
                "action": "status",
                "package": str(package),
                "pointer_path": str(pointer_path),
            }
        else:
            result = {
                "active": False,
                "action": "status",
                "pointer_path": str(pointer_path),
            }
    except (OSError, TypeError, ValueError, UnifiedRecurrentShadowPointerError) as exc:
        print(f"unified shadow pointer operation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
