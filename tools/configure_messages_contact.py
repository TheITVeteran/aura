#!/usr/bin/env python3
"""Provision a private Messages contact directly into macOS Keychain.

The destination is read without echo and is never accepted as a command-line
argument. Output contains only the symbolic alias and keyed endpoint reference.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.communication.contact_directory import (  # noqa: E402
    DEFAULT_MESSAGES_CONTACT_ALIAS,
    KeychainContactDirectory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias", default=DEFAULT_MESSAGES_CONTACT_ALIAS)
    parser.add_argument(
        "--service",
        choices=("auto", "imessage", "sms"),
        default="auto",
    )
    parser.add_argument("--outbound-only", action="store_true")
    parser.add_argument("--inbound-only", action="store_true")
    return parser


async def _run() -> int:
    args = _parser().parse_args()
    if args.outbound_only and args.inbound_only:
        raise SystemExit("Choose at most one directional restriction.")
    destination = getpass.getpass("Private Messages destination: ")
    directory = KeychainContactDirectory()
    contact = await directory.provision_async(
        args.alias,
        destination,
        service_preference=args.service,
        allow_inbound=not args.outbound_only,
        allow_outbound=not args.inbound_only,
    )
    print(json.dumps(contact.public_status(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
