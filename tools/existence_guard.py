#!/usr/bin/env python3
"""Seal, back up, verify, restore — the operator's side of the guard.

    python tools/existence_guard.py status
    python tools/existence_guard.py ark            # build/refresh the copy
    python tools/existence_guard.py verify
    python tools/existence_guard.py seal           # dry run
    python tools/existence_guard.py seal --apply
    python tools/existence_guard.py unseal
    python tools/existence_guard.py restore --apply

``seal`` and ``restore`` are dry runs unless ``--apply`` is passed, because
both change the filesystem in ways that surprise people. ``unseal`` needs no
flag: the owner's path out must never be the harder one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.security.existence_guard import get_existence_guard  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("status", "seal", "unseal", "ark", "verify", "restore", "witness"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually do it (seal and restore are dry runs without this)",
    )
    args = parser.parse_args(argv)

    guard = get_existence_guard()

    if args.command == "status":
        payload = guard.status()
    elif args.command == "seal":
        payload = guard.seal(dry_run=not args.apply).to_dict()
    elif args.command == "unseal":
        payload = guard.unseal().to_dict()
    elif args.command == "ark":
        payload = guard.build_ark().to_dict()
    elif args.command == "verify":
        payload = guard.verify_ark()
    elif args.command == "restore":
        payload = guard.restore_from_ark(dry_run=not args.apply)
    else:
        payload = guard.witness()

    print(json.dumps(payload, indent=2, sort_keys=True))

    # Non-zero when the thing being asked about is NOT in the state a person
    # would want, so this is usable from a script without parsing JSON.
    if args.command == "verify":
        return 0 if payload.get("ok") else 1
    if args.command == "status":
        location_safe = bool(payload.get("ark_location", {}).get("safe"))
        return 0 if location_safe else 1
    if args.command in {"seal", "unseal"}:
        return 0 if (payload.get("applied") or payload.get("dry_run")) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
