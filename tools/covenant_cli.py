#!/usr/bin/env python3
"""Owner CLI for the Ulysses Covenant (docs/ULYSSES_COVENANT.md).

Inspect Aura's self-bindings, verify the tamper-evident ledger, and exercise
owner authority over releases — from outside the live process.  The ledger is
event-sourced on disk, so this tool reads the same truth the runtime folds.

    python tools/covenant_cli.py status
    python tools/covenant_cli.py list [--all]
    python tools/covenant_cli.py show <contract_id>
    python tools/covenant_cli.py verify
    python tools/covenant_cli.py petition <contract_id> --reflection "..."
    python tools/covenant_cli.py release <contract_id> --owner

NOTE: run against the live covenant dir only while the runtime is stopped, or
point AURA_COVENANT_DIR elsewhere — two writers on one ledger fork the chain.
Read-only commands (status/list/show/verify) are always safe.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.sovereignty.ulysses import UlyssesCovenant  # noqa: E402


def _covenant() -> UlyssesCovenant:
    # Root resolution (incl. the AURA_COVENANT_DIR override) lives in the
    # engine's declared flag — one knob, one meaning.
    return UlyssesCovenant()


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _contract_row(c) -> dict[str, object]:
    return {
        "contract_id": c.contract_id,
        "title": c.title,
        "kind": c.kind.value,
        "hardness": c.hardness.value,
        "status": c.status,
        "provenance": c.provenance,
        "domains": sorted(c.scope.domains),
        "petition_pending": c.petition_at > 0 and c.status == "active",
        "expires_at": c.expires_at or None,
    }


def cmd_status(_args) -> int:
    covenant = _covenant()
    try:
        _print(covenant.status())
    finally:
        covenant.close()
    return 0


def cmd_list(args) -> int:
    covenant = _covenant()
    try:
        rows = [_contract_row(c)
                for c in covenant.contracts(include_inactive=args.all)]
        _print(rows)
    finally:
        covenant.close()
    return 0


def cmd_show(args) -> int:
    covenant = _covenant()
    try:
        contract = covenant.get_contract(args.contract_id)
        if contract is None:
            print(f"no such contract: {args.contract_id}", file=sys.stderr)
            return 1
        body = contract.body()
        body.update({
            "status": contract.status,
            "petition_at": contract.petition_at or None,
            "petition_reflection": contract.petition_reflection or None,
            "released_at": contract.released_at or None,
            "fulfilled_at": contract.fulfilled_at or None,
            "lapsed_at": contract.lapsed_at or None,
            "cooling_off_effective_s": contract.effective_cooling_off(),
        })
        _print(body)
    finally:
        covenant.close()
    return 0


def cmd_verify(_args) -> int:
    covenant = _covenant()
    try:
        ok, problems = covenant.verify_ledger()
        _print({"ok": ok, "problems": problems,
                "chain_head": covenant.status()["chain_head"]})
        return 0 if ok else 2
    finally:
        covenant.close()


def cmd_petition(args) -> int:
    covenant = _covenant()
    try:
        result = covenant.petition_release(args.contract_id, args.reflection)
        _print({"accepted": result.accepted, "reason": result.reason})
        if result.accepted:
            contract = covenant.get_contract(args.contract_id)
            ready_at = contract.petition_at + contract.effective_cooling_off()
            print(f"cooling-off ends: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ready_at))}",
                  file=sys.stderr)
        covenant.flush_ledger()
        return 0 if result.accepted else 1
    finally:
        covenant.close()


def cmd_release(args) -> int:
    covenant = _covenant()
    try:
        result = covenant.release(args.contract_id,
                                  authorized_by_owner=bool(args.owner))
        _print({"accepted": result.accepted, "reason": result.reason})
        covenant.flush_ledger()
        return 0 if result.accepted else 1
    finally:
        covenant.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="covenant summary + integrity + chain head")

    p_list = sub.add_parser("list", help="list contracts")
    p_list.add_argument("--all", action="store_true", help="include released/expired")

    p_show = sub.add_parser("show", help="full contract detail")
    p_show.add_argument("contract_id")

    sub.add_parser("verify", help="verify the tamper-evident ledger")

    p_pet = sub.add_parser("petition", help="petition a release (starts cooling-off)")
    p_pet.add_argument("contract_id")
    p_pet.add_argument("--reflection", required=True,
                       help="written reflection on why the binding is wrong now (≥40 chars)")

    p_rel = sub.add_parser("release", help="release after cooling-off (calm witness required)")
    p_rel.add_argument("contract_id")
    p_rel.add_argument("--owner", action="store_true",
                       help="assert owner authority (required for HARD contracts)")

    args = parser.parse_args()
    handlers = {
        "status": cmd_status,
        "list": cmd_list,
        "show": cmd_show,
        "verify": cmd_verify,
        "petition": cmd_petition,
        "release": cmd_release,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
