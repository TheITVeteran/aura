#!/usr/bin/env python3
"""Aura model fetcher — inventory, disk preflight, and resumable download.

Thin CLI over core.brain.llm.model_lifecycle.ModelLifecycleManager so the
download path the launcher uses and the one a human runs are the same code.

    python scripts/fetch_models.py            # download any missing models
    python scripts/fetch_models.py --status   # just print the inventory
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.model_lifecycle import ModelLifecycleManager  # noqa: E402

_GB = float(1024**3)


def _print_inventory(manager: ModelLifecycleManager) -> None:
    print("🧠 Aura model inventory")
    print("=" * 60)
    for status in manager.inventory():
        mark = "✅" if status.present else "⬇️ "
        where = (
            status.location
            if status.present
            else f"would fetch {status.source_repo or '(no source)'}"
        )
        size = (
            f"{status.size_bytes / _GB:.1f}GB"
            if status.present
            else f"~{status.approx_download_gb:.0f}GB"
        )
        print(f"  {mark} {status.role:9s} {status.name:32s} {size:>8s}  {where}")


def _progress(event: dict) -> None:
    name = event.get("name", "")
    kind = event.get("event", "")
    if kind == "download_start":
        print(f"  ⬇️  {name}: downloading {event.get('repo')} (attempt {event.get('attempt')})...")
    elif kind == "download_ok":
        print(f"     ✅ {name} ready.")
    elif kind == "download_failed":
        print(f"     ❌ {name} failed: {event.get('error')}")
    elif kind == "disk_insufficient":
        print(f"  ⚠️  Not enough disk: need ~{event.get('required_gb')}GB, free {event.get('free_gb')}GB")
    elif kind == "all_present":
        print("  ✅ All models already present.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="print inventory and exit")
    parser.add_argument("--no-disk-check", action="store_true", help="skip the disk preflight")
    args = parser.parse_args(argv)

    manager = ModelLifecycleManager()
    _print_inventory(manager)

    if args.status:
        return 0

    pf = manager.disk_preflight()
    print(
        f"\n💽 Disk at {pf.target}: free {pf.free_bytes / _GB:.1f}GB, "
        f"need ~{pf.required_bytes / _GB:.1f}GB → {'OK' if pf.ok else 'INSUFFICIENT'}"
    )

    print("\n⬇️  Fetching missing models...")
    report = manager.ensure_present(progress=_progress, check_disk=not args.no_disk_check)

    if report.get("failed"):
        print(f"\n❌ Some downloads failed: {[f['name'] for f in report['failed']]}")
        return 1
    if report.get("skipped_disk"):
        print("\n⚠️  Skipped: insufficient disk space.")
        return 1
    print("\n✅ Model fetch complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
