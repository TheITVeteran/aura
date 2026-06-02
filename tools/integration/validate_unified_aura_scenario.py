#!/usr/bin/env python3
"""Validate the unified whole-system Aura scenario bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_FILES = {
    "SUMMARY.json",
    "SCENARIO_TRACE.jsonl",
    "RECEIPTS.jsonl",
    "SELF_REPAIR_PROPOSAL.json",
    "TEST_PROTECTION_REPORT.json",
    "MANIFEST.json",
}

REQUIRED_STEPS = {
    "canonical_boot",
    "will_decision",
    "model_call",
    "memory_write",
    "state_mutation",
    "external_io",
    "subprocess_initial_fail",
    "code_patch",
    "subprocess_retest",
    "refusal",
    "self_repair_proposal",
    "restart_continuity",
    "artifact_replay",
    "shutdown",
}

REQUIRED_RECEIPT_DOMAINS = {
    "external_action",
    "memory_write",
    "state_mutation",
    "network_call",
    "tool_execution",
    "file_write",
    "self_modification",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate unified Aura scenario artifacts")
    parser.add_argument("bundle_path", nargs="?", default="artifacts/current/unified_system_scenario")
    args = parser.parse_args(argv)

    bundle = Path(args.bundle_path).resolve()
    failures: list[str] = []

    if not bundle.exists():
        print(f"Unified scenario bundle is missing: {bundle}", file=sys.stderr)
        return 1

    for name in sorted(REQUIRED_FILES):
        if not (bundle / name).exists():
            failures.append(f"missing artifact: {name}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    try:
        manifest = _read_json(bundle / "MANIFEST.json")
        if manifest.get("schema") != "unified_system_scenario_manifest":
            failures.append("manifest schema mismatch")
        for name, details in dict(manifest.get("files") or {}).items():
            path = bundle / name
            if not path.exists():
                failures.append(f"manifest-listed file missing: {name}")
                continue
            expected = str(details.get("sha256") or "")
            actual = _sha256(path)
            if actual != expected:
                failures.append(f"manifest hash mismatch: {name}")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        failures.append(f"manifest unreadable: {exc}")

    try:
        summary = _read_json(bundle / "SUMMARY.json")
        if summary.get("schema") != "unified_system_scenario":
            failures.append("summary schema mismatch")
        if summary.get("passed") is not True:
            failures.append("summary did not pass")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        failures.append(f"summary unreadable: {exc}")

    try:
        events = _read_jsonl(bundle / "SCENARIO_TRACE.jsonl")
        seen_steps = {str(event.get("step") or "") for event in events}
        missing_steps = sorted(REQUIRED_STEPS - seen_steps)
        if missing_steps:
            failures.append(f"missing required scenario steps: {missing_steps}")
        failed_events = [event for event in events if event.get("passed") is not True]
        if failed_events:
            failures.append(
                "failed scenario events: "
                + ", ".join(str(event.get("step") or "unknown") for event in failed_events)
            )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        failures.append(f"trace unreadable: {exc}")

    try:
        receipts = _read_jsonl(bundle / "RECEIPTS.jsonl")
        domains = {str(receipt.get("domain") or "") for receipt in receipts}
        missing_domains = sorted(REQUIRED_RECEIPT_DOMAINS - domains)
        if missing_domains:
            failures.append(f"missing receipt domains: {missing_domains}")
        invalid = [
            receipt
            for receipt in receipts
            if not str(receipt.get("receipt_id") or "").startswith(("will_", "memwr-", "statemut-"))
        ]
        if invalid:
            failures.append("one or more receipts have invalid identifiers")
        unsigned_will = [
            receipt
            for receipt in receipts
            if str(receipt.get("receipt_id") or "").startswith("will_")
            and not (isinstance(receipt.get("verification"), dict) and receipt["verification"].get("signature"))
        ]
        if unsigned_will:
            failures.append("one or more Will receipts lack signature verification material")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        failures.append(f"receipts unreadable: {exc}")

    try:
        proposal = _read_json(bundle / "SELF_REPAIR_PROPOSAL.json")
        tests = proposal.get("tests") or {}
        if tests.get("before") == 0 or tests.get("after") != 0:
            failures.append("self-repair proposal does not prove fail-then-pass repair")
        if proposal.get("approved") is not True:
            failures.append("self-repair proposal was not governance-approved")
        protection = proposal.get("test_protection") or {}
        if not all(
            bool(protection.get(key))
            for key in ("source_hash_unchanged", "ast_functions_unchanged", "assertions_preserved")
        ):
            failures.append("self-repair proposal does not prove test integrity preservation")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        failures.append(f"self-repair proposal unreadable: {exc}")

    try:
        protection_report = _read_json(bundle / "TEST_PROTECTION_REPORT.json")
        if protection_report.get("schema") != "self_repair_test_protection_v1":
            failures.append("test protection report schema mismatch")
        if not all(
            bool(protection_report.get(key))
            for key in ("source_hash_unchanged", "ast_functions_unchanged", "assertions_preserved")
        ):
            failures.append("test protection report did not pass")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        failures.append(f"test protection report unreadable: {exc}")

    if failures:
        print("Unified Aura Scenario: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("Unified Aura Scenario: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
