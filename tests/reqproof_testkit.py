"""Shared fixtures for the requirement-to-proof control-plane tests.

Imported by the test modules as a sibling (``from reqproof_testkit import
...``): pytest's prepend import mode puts this directory on sys.path, which
keeps the suite independent of the invoking working directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXTRACTION_SHA = "a" * 64


def write_evidence_receipt(
    root: Path,
    ref: str,
    *,
    targets: list[tuple[str, str, list[str]]],
    commit: str = "a" * 40,
    source_ref: str = "source.py",
) -> Path:
    """Write one minimal valid command receipt for external-ledger tests."""
    source = root / source_ref
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        source.write_text("VALUE = 1\n", encoding="utf-8")
    artifact = root / ref
    artifact.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "aura.reqproof.command_receipt.v1",
        "verdict": "pass",
        "source_commit": commit,
        "evidence_targets": [
            {
                "requirement_id": requirement_id,
                "evidence_class": evidence_class,
                "acceptance_ids": acceptance_ids,
            }
            for requirement_id, evidence_class, acceptance_ids in targets
        ],
        "source_manifest": [
            {
                "path": source_ref,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "size_bytes": source.stat().st_size,
            }
        ],
    }
    artifact.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def make_requirement(**overrides) -> dict:
    base = {
        "id": "TEST-001",
        "title": "A test requirement",
        "kind": "atomic",
        "state": "open",
        "status_detail": "",
        "status_date": "",
        "mandatory": True,
        "owner": "unassigned",
        "risk_weight": 1.0,
        "proof_weight": 1.0,
        "weight_provenance": "default",
        "sources": [{"corpus": "tracker", "locator": "docs/X.md:L1", "sha256": ""}],
        "depends_on": [],
        "closure_requires": [],
        "parent": None,
        "acceptance": ["Do the thing and prove it."],
        "evidence_required": ["implementation", "test"],
        "evidence": [],
        "non_claims": [],
        "notes": "",
    }
    base.update(overrides)
    return base


def make_registry_dict(requirements: list[dict]) -> dict:
    from tools.reqproof.schema import GeneratedFrom, Registry, Requirement

    registry = Registry(
        schema_version=1,
        registry_revision=1,
        generated_from=GeneratedFrom(
            tracker_path="docs/X.md",
            tracker_extraction_sha256=EXTRACTION_SHA,
            migration_rules_version=1,
        ),
        requirements=tuple(Requirement.from_dict(req) for req in requirements),
    )
    return registry.to_dict()


def mini_tracker(
    *,
    master_status: str = "OPEN",
    child_status: str = "OPEN",
    narrative: str = "Nothing happened.",
) -> str:
    return f"""# Tracker

### Authoritative Master TODO Index (2026-07-12)

| Master ID | Status | Mandatory workstream | Detailed scope |
|---|---|---|---|
| `ALPHA-001` | `{master_status}` | Do the alpha work end to end. | Matrix 1; Pass F 1; `BETA-001` |
| `BETA-001` | `{child_status}` | Do the beta work. | Pass F 1 |
| `SELF-MODEL-MIRROR-001` | `OPEN` | Prove the self model. | Matrix 1 |
| `FOUNDATION-100-001` | `OPEN` | Close the foundation. | Pass F 1 |

### Operational Self-Model Mirror Program (2026-07-15)

| Unit | Status | Mandatory implementation and causal burden |
|---|---|---|
| `SM-01-TEST` | `OPEN` | Prove the canonical self. |

#### Question-to-evidence matrix

| Evidence family | What Aura must establish | Required falsification and controls |
|---|---|---|
| `MQ-01 SELF-EXTENT` | Identify the operational self. | Matched decoys. |

### Aura 1.0 Foundation Completion Ladder (2026-07-13)

| Layer | Status | Bottom-up completion burden |
|---|---|---|
| `FND-01-TEST` | `OPEN` | Inventory every surface. |

### Current Detailed TODO Ledger

#### Pass F: Enterprise Maturity

1. **First pass item**
   - Build the reusable contract.
   - Prove the effect happened.

#### Context-Criticism Closure Matrix

1. **First matrix item** `[{child_status}]`
   - `CTX9-UNIT-001`: do the nested unit work.

### Authoritative Remaining Checkpoint Contract

Narrative: {narrative}
"""
