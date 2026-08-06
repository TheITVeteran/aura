#!/usr/bin/env python3
"""Deterministic migration: tracker extraction -> requirement registry.

The registry baseline is a pure function of the tracker's normative
extraction plus these versioned rules. Re-running migration on the same
tracker content produces a byte-identical registry; any change to normative
tracker content changes the pinned extraction hash and is detected as a
stale migration by the gate.

Rules of honesty applied here:

* Statuses are carried over exactly as the tracker asserts them — but a
  migrated ``complete`` carries no machine evidence, so the validator reports
  every one as unproven-closure debt instead of treating prose as proof.
* Weights default to 1.0 with ``weight_provenance="default"``: no invented
  precision. Refining weights is explicit later work.
* IDs referenced in prose without a defining row are minted as open
  requirements (zero-drop), never silently ignored.

Usage:
  python tools/reqproof/migrate.py --write     # regenerate the registry
  python tools/reqproof/migrate.py --check     # verify registry is current
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.reqproof.schema import (  # noqa: E402
    EVIDENCE_CLASSES,
    SCHEMA_VERSION,
    GeneratedFrom,
    Registry,
    Requirement,
    load_registry,
    write_registry_atomic,
)
from tools.reqproof.tracker_parse import (  # noqa: E402
    TRACKER_RELPATH,
    TrackerExtraction,
    parse_tracker,
)

MIGRATION_RULES_VERSION = 3

DEFAULT_REGISTRY_PATH = ROOT / "config" / "requirement_registry.json"
DEFAULT_ALLOWLIST_PATH = ROOT / "config" / "reqproof_prose_token_allowlist.json"

SELF_MODEL_PARENT = "SELF-MODEL-MIRROR-001"
FOUNDATION_PARENT = "FOUNDATION-100-001"

REF_FAMILY_RE = re.compile(
    r"\b(Pass F|Matrix|Addendum)\s+(\d+(?:-\d+)?(?:(?:,\s*|\s+and\s+)\d+(?:-\d+)?)*)"
)
BACKTICKED_DEP_RE = re.compile(r"`([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)`")
STATE_PREFIXES = (
    ("IN PROGRESS", "in_progress"),
    ("OPEN", "open"),
    ("COMPLETE", "complete"),
    ("DEFERRED", "deferred"),
    ("WITHDRAWN", "withdrawn"),
    ("BLOCKED", "blocked"),
)
ISO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# The matrix numbered list is continuous 1..33; items 1-18 come from the
# first criticism corpus ("Matrix N"), 19-33 from the second ("Addendum N").
MATRIX_ADDENDUM_BOUNDARY = 19

STANDARD_OPERATIONAL_NON_CLAIM = (
    "Operational/functional evidence only: passing does not establish private "
    "qualia, metaphysical consciousness, legal or moral personhood, solved "
    "AGI, or ASI."
)
STANDARD_CAPABILITY_NON_CLAIM = (
    "A provider, architecture, or model label is never proof: capability "
    "parity claims require contamination-resistant matched sealed "
    "evaluations."
)
OPERATIONAL_NON_CLAIM_RE = re.compile(
    r"conscious|sentien|personhood|\bAGI\b|\bASI\b|qualia|inner.life|superintelligen",
    re.IGNORECASE,
)
CAPABILITY_NON_CLAIM_RE = re.compile(
    r"frontier|GPT|parity with|Fable|Mythos|Sol Ultra", re.IGNORECASE
)

EVIDENCE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("live", re.compile(r"live|exact-app|exact app|exact-main|exact main|launched|desktop lane", re.IGNORECASE)),
    ("gui", re.compile(r"\b(GUI|UI|shell|screenshot)\b")),
    ("security", re.compile(r"securit|adversar|hostile|penetration|threat model|DNS|privacy", re.IGNORECASE)),
    ("portability", re.compile(r"portab|clean.machine|clean machine|managed-device|locale", re.IGNORECASE)),
    ("release", re.compile(r"release", re.IGNORECASE)),
    ("soak", re.compile(r"soak|24-72|endurance|multi-hour", re.IGNORECASE)),
)


class MigrationError(ValueError):
    """Migration could not proceed deterministically."""


def _previous_registry_identity(path: Path) -> tuple[int, str]:
    """Read a hash-verified previous registry, including schema v1 upgrades."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError(f"cannot read previous registry {path}: {exc}") from exc
    if data.get("schema_version") == SCHEMA_VERSION:
        current = load_registry(path)
        return current.registry_revision, current.compute_content_sha256()
    if data.get("schema_version") != 1:
        raise MigrationError(
            f"unsupported previous registry schema {data.get('schema_version')!r}"
        )
    recorded = data.get("content_sha256")
    revision = data.get("registry_revision")
    if not isinstance(recorded, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded):
        raise MigrationError("legacy registry content hash is missing or malformed")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise MigrationError("legacy registry revision is invalid")
    body = {key: value for key, value in data.items() if key != "content_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual != recorded:
        raise MigrationError("legacy registry content hash does not match content")
    return revision, recorded


def parse_state(status_raw: str, *, context: str) -> tuple[str, str]:
    """Map a raw tracker status string to (state, iso_date)."""
    text = status_raw.strip()
    if not text:
        return "open", ""
    upper = text.upper()
    for prefix, state in STATE_PREFIXES:
        if upper.startswith(prefix):
            date_match = ISO_DATE_RE.search(text)
            return state, date_match.group(1) if date_match else ""
    raise MigrationError(f"unrecognized status {status_raw!r} in {context}")


def derive_evidence_required(text: str) -> tuple[str, ...]:
    classes = ["implementation", "test"]
    for class_name, pattern in EVIDENCE_RULES:
        if pattern.search(text):
            classes.append(class_name)
    return tuple(classes)


def derive_acceptance_evidence_required(
    acceptance: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    """Derive proof modalities from each criterion, never from its siblings."""
    return tuple(derive_evidence_required(criterion) for criterion in acceptance)


def evidence_union(
    acceptance_evidence_required: tuple[tuple[str, ...], ...],
) -> tuple[str, ...]:
    return tuple(
        class_name
        for class_name in EVIDENCE_CLASSES
        if any(class_name in classes for classes in acceptance_evidence_required)
    )


def derive_non_claims(text: str) -> tuple[str, ...]:
    claims: list[str] = []
    if OPERATIONAL_NON_CLAIM_RE.search(text):
        claims.append(STANDARD_OPERATIONAL_NON_CLAIM)
    if CAPABILITY_NON_CLAIM_RE.search(text):
        claims.append(STANDARD_CAPABILITY_NON_CLAIM)
    return tuple(claims)


def _first_sentence(text: str, *, limit: int = 160) -> str:
    stripped = " ".join(text.split())
    sentence_end = stripped.find(". ")
    sentence = stripped[: sentence_end + 1] if sentence_end != -1 else stripped
    if len(sentence) > limit:
        sentence = sentence[: limit - 1].rstrip() + "…"
    return sentence or "(no burden text)"


def _item_registry_id(list_key: str, number: int) -> str:
    if list_key == "passf":
        return f"PASSF-{number:02d}"
    if list_key == "carryover":
        return f"CARRYOVER-{number:02d}"
    if list_key == "matrix":
        if number >= MATRIX_ADDENDUM_BOUNDARY:
            return f"ADDENDUM-{number:02d}"
        return f"MATRIX-{number:02d}"
    raise MigrationError(f"unknown list key {list_key!r}")


def _expand_ref_numbers(spec: str) -> list[int]:
    numbers: list[int] = []
    for chunk in re.split(r",\s*|\s+and\s+", spec.strip()):
        if not chunk:
            continue
        if "-" in chunk:
            low_s, high_s = chunk.split("-", 1)
            low, high = int(low_s), int(high_s)
            if high < low:
                raise MigrationError(f"reference range {chunk!r} is inverted")
            numbers.extend(range(low, high + 1))
        else:
            numbers.append(int(chunk))
    return numbers


def parse_scope_refs(refs_raw: str) -> list[str]:
    """Expand 'Pass F 12-13; Matrix 2 and 17; Addendum 22' style references."""
    targets: list[str] = []
    for family, spec in REF_FAMILY_RE.findall(refs_raw):
        for number in _expand_ref_numbers(spec):
            if family == "Pass F":
                targets.append(f"PASSF-{number:02d}")
            else:
                # 'Matrix' and 'Addendum' share the continuous 1..33 list;
                # normalize by number, not by which word the row used.
                targets.append(_item_registry_id("matrix", number))
    seen: set[str] = set()
    unique: list[str] = []
    for target in targets:
        if target not in seen:
            seen.add(target)
            unique.append(target)
    return unique


def parse_dep_ids(refs_raw: str, *, own_id: str, declared: set[str]) -> list[str]:
    deps: list[str] = []
    for token in BACKTICKED_DEP_RE.findall(refs_raw):
        if token == own_id or token in deps:
            continue
        deps.append(token)
    # Dependencies may legitimately reference minted prose-only IDs; the
    # validator rejects anything that never becomes a requirement.
    del declared
    return deps


def _acceptance_from_body(body: str) -> tuple[str, ...]:
    """Top-level '- ' bullets of an item's body become acceptance units."""
    acceptance: list[str] = []
    current: list[str] | None = None
    for line in body.splitlines():
        if re.match(r"^\s{0,4}-\s+", line):
            if current:
                acceptance.append(" ".join(current))
            current = [line.strip().lstrip("- ").strip()]
        elif current is not None and line.strip():
            current.append(line.strip())
        elif current is not None:
            acceptance.append(" ".join(current))
            current = None
    if current:
        acceptance.append(" ".join(current))
    return tuple(entry for entry in acceptance if entry)


def load_prose_allowlist(path: Path) -> dict[str, str]:
    """Load {token: reason} for ID-like tokens that are not requirements."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value.strip()
        for key, value in data.items()
    ):
        raise MigrationError(
            f"prose allowlist {path} must map token -> non-empty reason string"
        )
    return dict(data)


def _mandatory_for(state: str) -> bool:
    return state != "withdrawn"


def build_registry(
    extraction: TrackerExtraction,
    *,
    allowlist: dict[str, str],
    registry_revision: int = 1,
) -> Registry:
    """Build the registry deterministically from a tracker extraction."""
    draft: dict[str, dict[str, Any]] = {}

    def add(record: dict[str, Any]) -> None:
        req_id = record["id"]
        if req_id in draft:
            raise MigrationError(f"duplicate requirement id during migration: {req_id}")
        draft[req_id] = record

    def base_record(
        req_id: str,
        title: str,
        state: str,
        status_detail: str,
        status_date: str,
        burden_text: str,
        locator: str,
        acceptance: tuple[str, ...],
        notes: str = "",
    ) -> dict[str, Any]:
        derivation_text = " ".join((burden_text, status_detail))
        acceptance_evidence_required = derive_acceptance_evidence_required(acceptance)
        return {
            "id": req_id,
            "title": title,
            "kind": "atomic",  # promoted to parent when closure edges appear
            "state": state,
            "status_detail": status_detail,
            "status_date": status_date,
            "mandatory": _mandatory_for(state),
            "owner": "unassigned",
            "risk_weight": 1.0,
            "proof_weight": 1.0,
            "weight_provenance": "default",
            "sources": [
                {"corpus": "tracker", "locator": locator, "sha256": ""}
            ],
            "depends_on": [],
            "closure_requires": [],
            "parent": None,
            "acceptance": list(acceptance),
            "acceptance_evidence_required": [
                list(classes) for classes in acceptance_evidence_required
            ],
            "evidence_required": list(evidence_union(acceptance_evidence_required)),
            "evidence": [],
            "non_claims": list(derive_non_claims(derivation_text)),
            "notes": notes,
        }

    declared = extraction.declared_ids()

    # --- Requirement tables -------------------------------------------------
    for row in extraction.table_rows:
        state, date = parse_state(row.status_raw, context=f"table row {row.row_id}")
        locator = f"{extraction.tracker_path}:L{row.line}"
        title = row.id_suffix or _first_sentence(row.burden)
        acceptance: tuple[str, ...] = (row.burden,)
        if row.table == "mq" and row.refs_raw:
            # The MQ table's third column is falsification controls — part of
            # the acceptance bar, not a reference list.
            acceptance = (row.burden, f"Falsification and controls: {row.refs_raw}")
        record = base_record(
            req_id=row.row_id,
            title=title,
            state=state,
            status_detail=row.status_raw,
            status_date=date,
            burden_text=" ".join((row.burden, row.refs_raw)),
            locator=locator,
            acceptance=acceptance,
            notes=f"refs: {row.refs_raw}" if row.refs_raw and row.table == "master" else "",
        )
        if row.table == "master":
            record["closure_requires"] = parse_scope_refs(row.refs_raw)
            record["depends_on"] = parse_dep_ids(
                row.refs_raw, own_id=row.row_id, declared=declared
            )
        elif row.table in ("self_model", "mq"):
            record["parent"] = SELF_MODEL_PARENT
        elif row.table == "foundation":
            record["parent"] = FOUNDATION_PARENT
        add(record)

    # --- Numbered obligations (Pass F, matrix, addendum, carryover) ---------
    for item in extraction.items:
        req_id = _item_registry_id(item.list_key, item.number)
        state, date = parse_state(item.status_raw, context=f"item {req_id}")
        acceptance = _acceptance_from_body(item.body) or (item.title,)
        record = base_record(
            req_id=req_id,
            title=item.title,
            state=state,
            status_detail=item.status_raw,
            status_date=date,
            burden_text=" ".join((item.title, item.body)),
            locator=f"{extraction.tracker_path}:L{item.line}",
            acceptance=acceptance,
        )
        if item.list_key == "carryover":
            record["notes"] = (
                "Historical carryover obligation; valid but only closable "
                "against a runnable validator or live proof artifact."
            )
        add(record)
        for unit in item.nested:
            unit_state, unit_date = (
                parse_state(unit.status_raw, context=f"nested unit {unit.unit_id}")
                if unit.status_raw
                else (state, date)
            )
            nested_record = base_record(
                req_id=unit.unit_id,
                title=_first_sentence(unit.text),
                state=unit_state,
                status_detail=unit.status_raw or item.status_raw,
                status_date=unit_date,
                burden_text=unit.text,
                locator=f"{extraction.tracker_path}:L{unit.line}",
                acceptance=(unit.text,),
            )
            nested_record["parent"] = req_id
            add(nested_record)

    # --- Prose-only referenced IDs (zero-drop minting) ----------------------
    known_after_items = set(draft)
    for token in extraction.all_id_tokens:
        if token in known_after_items or token in allowlist:
            continue
        add(
            {
                **base_record(
                    req_id=token,
                    title=(
                        "Obligation referenced in tracker prose without a "
                        "defining row (minted by migration)"
                    ),
                    state="open",
                    status_detail="",
                    status_date="",
                    burden_text="",
                    locator=f"{extraction.tracker_path}:prose-reference",
                    acceptance=(
                        "Define this requirement's scope, acceptance, and "
                        "evidence in the tracker so migration can replace "
                        "this prose-minted requirement.",
                    ),
                    notes="minted_from_prose",
                ),
            }
        )

    # --- Parent/closure edge reconciliation ---------------------------------
    for record in draft.values():
        parent_id = record["parent"]
        if parent_id is None:
            continue
        if parent_id not in draft:
            raise MigrationError(
                f"requirement {record['id']} names missing parent {parent_id}"
            )
        parent_record = draft[parent_id]
        if record["id"] not in parent_record["closure_requires"]:
            parent_record["closure_requires"].append(record["id"])

    for record in draft.values():
        record["closure_requires"] = sorted(set(record["closure_requires"]))
        if record["closure_requires"]:
            record["kind"] = "parent"

    requirements = tuple(
        Requirement.from_dict(draft[req_id]) for req_id in sorted(draft)
    )
    return Registry(
        schema_version=SCHEMA_VERSION,
        registry_revision=registry_revision,
        generated_from=GeneratedFrom(
            tracker_path=TRACKER_RELPATH,
            tracker_extraction_sha256=extraction.extraction_sha256(),
            migration_rules_version=MIGRATION_RULES_VERSION,
        ),
        requirements=requirements,
    )


def migrate(
    *,
    tracker_path: Path,
    registry_path: Path,
    allowlist_path: Path,
    write: bool,
) -> dict[str, Any]:
    extraction = parse_tracker(tracker_path)
    allowlist = load_prose_allowlist(allowlist_path)
    revision = 1
    previous_hash = ""
    if registry_path.exists():
        revision, previous_hash = _previous_registry_identity(registry_path)
    registry = build_registry(
        extraction, allowlist=allowlist, registry_revision=revision
    )
    if previous_hash and registry.compute_content_sha256() != previous_hash:
        registry = build_registry(
            extraction, allowlist=allowlist, registry_revision=revision + 1
        )
    current = registry.compute_content_sha256() == previous_hash
    result = {
        "tracker_extraction_sha256": extraction.extraction_sha256(),
        "requirements": len(registry.requirements),
        "registry_revision": registry.registry_revision,
        "registry_current": current,
        "written": False,
    }
    if write and not current:
        write_registry_atomic(registry, registry_path)
        result["written"] = True
    result["content_sha256"] = registry.compute_content_sha256()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", default=str(ROOT / TRACKER_RELPATH))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST_PATH))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate the registry")
    mode.add_argument(
        "--check",
        action="store_true",
        help="exit 1 unless the registry matches the current tracker",
    )
    args = parser.parse_args()
    result = migrate(
        tracker_path=Path(args.tracker),
        registry_path=Path(args.registry),
        allowlist_path=Path(args.allowlist),
        write=bool(args.write),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.check and not result["registry_current"]:
        print("registry is STALE: rerun tools/reqproof/migrate.py --write", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
