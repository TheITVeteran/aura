#!/usr/bin/env python3
"""Fail closed unless source discovery, validation, and the live registry agree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.capability_engine import CapabilityEngine  # noqa: E402
from core.skills.discovery import build_skill_catalog, validate_skill_catalog  # noqa: E402


def audit_skill_catalog(*, require_rust: bool = False) -> dict[str, Any]:
    catalog = build_skill_catalog()
    validations = validate_skill_catalog(catalog, use_cache=False)
    engine = CapabilityEngine()
    accepted_names = {declaration.name for declaration in catalog.accepted}
    live_names = set(engine.skills)
    quarantined = sorted(
        validation.get("name") or catalog_id
        for catalog_id, validation in validations.items()
        if validation.get("status") != "valid"
    )
    dry_run = engine.dry_run_catalog()
    failures: list[str] = []
    if not catalog.ok:
        failures.append("source_catalog_has_blocking_issues")
    if require_rust and catalog.parity_status != "matched":
        failures.append("rust_python_parity_not_proven")
    if quarantined:
        failures.append("isolated_validation_quarantined_skills")
    if live_names != accepted_names:
        failures.append("source_catalog_live_registry_mismatch")
    if not engine.is_ready():
        failures.append("capability_engine_not_ready")
    if not dry_run.get("ok"):
        failures.append("capability_engine_dry_run_failed")
    return {
        "accepted_count": len(accepted_names),
        "backend": catalog.backend,
        "catalog_digest": catalog.digest,
        "excluded_count": len(catalog.excluded),
        "failures": failures,
        "live_count": len(live_names),
        "missing_live": sorted(accepted_names - live_names),
        "ok": not failures,
        "parity_status": catalog.parity_status,
        "quarantined": quarantined,
        "unexpected_live": sorted(live_names - accepted_names),
        "validated_count": sum(
            validation.get("status") == "valid" for validation in validations.values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-rust",
        action="store_true",
        help="Fail unless the installed Rust extension exactly matches Python discovery.",
    )
    args = parser.parse_args()
    result = audit_skill_catalog(require_rust=bool(args.require_rust))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
