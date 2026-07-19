#!/usr/bin/env python3
"""Source-corpus coverage: every obligation passage maps to requirements.

The requirement registry answers "what must be done"; this module answers
"did anything in the source corpora fail to become a requirement". Each
authoritative corpus is snapshotted under ``config/requirement_sources/`` and
pinned by hash in ``MANIFEST.json``. The coverage map
(``config/requirement_coverage_map.json``) assigns every non-blank line of
every corpus to one of:

* ``normative``  — maps to one or more registry requirement IDs;
* ``rationale``  — explanatory/theoretical text with a recorded reason;
* ``duplicate``  — a repeat of an already-mapped obligation, with the
                   repeated region recorded.

Every map entry pins the exact text it covers by hash, so silently editing a
corpus (or drifting a snapshot) surfaces as ``stale-coverage`` rather than
being absorbed. Unmapped non-blank lines are ``unmapped-passage`` defects and
always fail the gate: zero-unmapped is enforced, not aspirational.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.reqproof.validate import Defect

MANIFEST_RELPATH = "config/requirement_sources/MANIFEST.json"
COVERAGE_MAP_RELPATH = "config/requirement_coverage_map.json"

ENTRY_CLASSES = ("normative", "rationale", "duplicate")
LINES_RE = re.compile(r"^(\d+)-(\d+)$")


class CoverageError(ValueError):
    """The coverage manifest or map is structurally invalid."""


@dataclass(frozen=True)
class Corpus:
    corpus_id: str
    snapshot: str
    original_path: str
    original_sha256: str
    description: str


@dataclass(frozen=True)
class MapEntry:
    corpus: str
    start_line: int
    end_line: int
    sha256: str
    entry_class: str
    requirements: tuple[str, ...]
    reason: str

    @property
    def locator(self) -> str:
        return f"{self.corpus}:L{self.start_line}-L{self.end_line}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def range_text(lines: list[str], start_line: int, end_line: int) -> str:
    """Canonical text of an inclusive 1-based line range."""
    selected = lines[start_line - 1 : end_line]
    return "\n".join(line.rstrip() for line in selected)


def load_manifest(root: Path) -> dict[str, Corpus]:
    path = root / MANIFEST_RELPATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CoverageError(f"corpus manifest missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CoverageError(f"corpus manifest is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise CoverageError("corpus manifest must be an object with schema_version 1")
    corpora_raw = data.get("corpora")
    if not isinstance(corpora_raw, dict) or not corpora_raw:
        raise CoverageError("corpus manifest must declare a non-empty 'corpora' object")
    corpora: dict[str, Corpus] = {}
    for corpus_id, entry in sorted(corpora_raw.items()):
        if not isinstance(entry, dict):
            raise CoverageError(f"corpus {corpus_id!r} entry must be an object")
        required = {"snapshot", "original_path", "original_sha256", "description"}
        missing = required - set(entry)
        if missing:
            raise CoverageError(f"corpus {corpus_id!r} missing fields: {sorted(missing)}")
        unknown = set(entry) - required
        if unknown:
            raise CoverageError(f"corpus {corpus_id!r} has unknown fields: {sorted(unknown)}")
        corpora[corpus_id] = Corpus(
            corpus_id=corpus_id,
            snapshot=str(entry["snapshot"]),
            original_path=str(entry["original_path"]),
            original_sha256=str(entry["original_sha256"]),
            description=str(entry["description"]),
        )
    return corpora


def load_coverage_map(root: Path) -> list[MapEntry]:
    path = root / COVERAGE_MAP_RELPATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CoverageError(f"coverage map missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CoverageError(f"coverage map is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise CoverageError("coverage map must be an object with schema_version 1")
    entries_raw = data.get("entries")
    if not isinstance(entries_raw, list):
        raise CoverageError("coverage map must declare an 'entries' list")
    entries: list[MapEntry] = []
    for index, raw in enumerate(entries_raw):
        name = f"entries[{index}]"
        if not isinstance(raw, dict):
            raise CoverageError(f"{name} must be an object")
        required = {"corpus", "lines", "sha256", "class"}
        allowed = required | {"requirements", "reason"}
        missing = required - set(raw)
        if missing:
            raise CoverageError(f"{name} missing fields: {sorted(missing)}")
        unknown = set(raw) - allowed
        if unknown:
            raise CoverageError(f"{name} has unknown fields: {sorted(unknown)}")
        lines_match = LINES_RE.match(str(raw["lines"]))
        if not lines_match:
            raise CoverageError(f"{name}.lines must look like '12-34'")
        start_line, end_line = int(lines_match.group(1)), int(lines_match.group(2))
        if start_line < 1 or end_line < start_line:
            raise CoverageError(f"{name}.lines range {raw['lines']!r} is invalid")
        entry_class = str(raw["class"])
        if entry_class not in ENTRY_CLASSES:
            raise CoverageError(f"{name}.class {entry_class!r} not in {ENTRY_CLASSES}")
        requirements_raw = raw.get("requirements", [])
        if not isinstance(requirements_raw, list) or not all(
            isinstance(item, str) and item for item in requirements_raw
        ):
            raise CoverageError(f"{name}.requirements must be a list of non-empty strings")
        reason = str(raw.get("reason", ""))
        if entry_class == "normative" and not requirements_raw:
            raise CoverageError(
                f"{name} is normative but maps to no requirements"
            )
        if entry_class in ("rationale", "duplicate") and not reason.strip():
            raise CoverageError(f"{name} is {entry_class} and must record a reason")
        entries.append(
            MapEntry(
                corpus=str(raw["corpus"]),
                start_line=start_line,
                end_line=end_line,
                sha256=str(raw["sha256"]),
                entry_class=entry_class,
                requirements=tuple(requirements_raw),
                reason=reason,
            )
        )
    return entries


def check_coverage(
    root: Path,
    *,
    registry_ids: set[str],
) -> tuple[list[Defect], dict[str, Any]]:
    """Verify total coverage. Returns (defects, deterministic report)."""
    defects: list[Defect] = []
    corpora = load_manifest(root)
    entries = load_coverage_map(root)

    corpus_lines: dict[str, list[str]] = {}
    for corpus in corpora.values():
        snapshot_path = root / corpus.snapshot
        if not snapshot_path.is_file():
            defects.append(
                Defect(
                    defect_class="missing-corpus",
                    subject=corpus.corpus_id,
                    detail=f"snapshot {corpus.snapshot} does not exist",
                )
            )
            continue
        corpus_lines[corpus.corpus_id] = snapshot_path.read_text(
            encoding="utf-8"
        ).splitlines()

    covered: dict[str, set[int]] = {corpus_id: set() for corpus_id in corpus_lines}
    per_class_counts: dict[str, int] = {cls: 0 for cls in ENTRY_CLASSES}

    for entry in entries:
        if entry.corpus not in corpora:
            defects.append(
                Defect(
                    defect_class="coverage-orphan-ref",
                    subject=entry.locator,
                    detail=f"map entry references unknown corpus {entry.corpus!r}",
                )
            )
            continue
        if entry.corpus not in corpus_lines:
            continue  # snapshot missing; already reported
        lines = corpus_lines[entry.corpus]
        if entry.end_line > len(lines):
            defects.append(
                Defect(
                    defect_class="stale-coverage",
                    subject=entry.locator,
                    detail=(
                        f"range ends at line {entry.end_line} but corpus has "
                        f"{len(lines)} lines"
                    ),
                )
            )
            continue
        actual = _sha256_text(range_text(lines, entry.start_line, entry.end_line))
        if actual != entry.sha256:
            defects.append(
                Defect(
                    defect_class="stale-coverage",
                    subject=entry.locator,
                    detail=(
                        "range content hash mismatch: recorded "
                        f"{entry.sha256[:12]}…, actual {actual[:12]}… "
                        "(corpus text changed under the map)"
                    ),
                )
            )
            continue
        for requirement_id in entry.requirements:
            if requirement_id not in registry_ids:
                defects.append(
                    Defect(
                        defect_class="coverage-orphan-ref",
                        subject=f"{entry.locator}::{requirement_id}",
                        detail="map entry references unknown requirement",
                    )
                )
        per_class_counts[entry.entry_class] += 1
        covered[entry.corpus].update(range(entry.start_line, entry.end_line + 1))

    unmapped_total = 0
    unmapped_by_corpus: dict[str, list[str]] = {}
    for corpus_id, lines in sorted(corpus_lines.items()):
        missing_lines = [
            index + 1
            for index, line in enumerate(lines)
            if line.strip() and (index + 1) not in covered[corpus_id]
        ]
        if missing_lines:
            unmapped_total += len(missing_lines)
            spans: list[str] = []
            span_start = prev = missing_lines[0]
            for line_no in missing_lines[1:]:
                if line_no != prev + 1:
                    spans.append(
                        f"L{span_start}" if span_start == prev else f"L{span_start}-L{prev}"
                    )
                    span_start = line_no
                prev = line_no
            spans.append(
                f"L{span_start}" if span_start == prev else f"L{span_start}-L{prev}"
            )
            unmapped_by_corpus[corpus_id] = spans
            for span in spans:
                defects.append(
                    Defect(
                        defect_class="unmapped-passage",
                        subject=f"{corpus_id}:{span}",
                        detail="corpus passage is not mapped to any requirement",
                    )
                )

    report = {
        "corpora": {
            corpus_id: {
                "snapshot": corpora[corpus_id].snapshot,
                "original_sha256": corpora[corpus_id].original_sha256,
                "total_lines": len(lines),
                "nonblank_lines": sum(1 for line in lines if line.strip()),
                "unmapped_spans": unmapped_by_corpus.get(corpus_id, []),
            }
            for corpus_id, lines in sorted(corpus_lines.items())
        },
        "entries": len(entries),
        "entries_by_class": per_class_counts,
        "unmapped_lines": unmapped_total,
    }
    return (
        sorted(defects, key=lambda d: (d.defect_class, d.subject, d.detail)),
        report,
    )
