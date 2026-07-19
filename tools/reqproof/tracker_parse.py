#!/usr/bin/env python3
"""Strict normative extraction from docs/AURA_EXECUTION_TRACKER.md.

The execution tracker mixes normative content (requirement tables, the Pass F
ledger, the criticism/capability matrices, the carryover list) with thousands
of lines of checkpoint narrative. This module extracts ONLY the normative
content, canonicalizes it, and hashes it, so that:

* the registry migration (tools/reqproof/migrate.py) is a deterministic
  function of this extraction;
* narrative/prose edits do NOT invalidate the registry, but any change to an
  ID, status, burden, or reference DOES (stale-migration detection);
* every ID-like token in the tracker is accounted for — a requirement that
  exists only in prose is surfaced as a defect instead of silently dropped.

Parsing is strict: an unrecognized requirement-table shape or a malformed
numbered list is a hard error, so new tracker structures must be consciously
integrated here rather than silently ignored.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TRACKER_RELPATH = "docs/AURA_EXECUTION_TRACKER.md"

ID_TOKEN_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+$")
# Anchored on the ID shape itself rather than generic backtick pairing:
# stray/fenced backticks elsewhere in the document must not desync the census.
BACKTICKED_ID_RE = re.compile(r"`([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)(?: [A-Z][A-Z0-9 -]*)?`")
STATUS_MARKER_RE = re.compile(r"\[([A-Z][A-Z0-9 /:+;.-]*?(?:\d{4}-\d{2}-\d{2})?)\]")
ISO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
NUMBERED_ITEM_RE = re.compile(r"^(\d+)\.\s+\*\*(.+?)\*\*\s*(.*)$")
TABLE_ROW_RE = re.compile(r"^\|(.+)\|\s*$")
NESTED_ID_RE = re.compile(
    r"^\s*-\s+`([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)`\s*(?:`\[([^\]]+)\]`)?\s*:?"
)

# Section headings that bound the normative regions. Matching is by prefix so
# date suffixes can change without breaking extraction.
MASTER_INDEX_HEADING = "### Authoritative Master TODO Index"
SELF_MODEL_HEADING = "### Operational Self-Model Mirror Program"
FOUNDATION_HEADING = "### Aura 1.0 Foundation Completion Ladder"
PASSF_HEADING = "#### Pass F:"
MATRIX_HEADING = "#### Context-Criticism Closure Matrix"
LEDGER_END_HEADING = "### Authoritative Remaining Checkpoint Contract"


class TrackerParseError(ValueError):
    """The tracker's normative structure could not be parsed strictly."""


@dataclass(frozen=True)
class TableRow:
    """One row of an ID-bearing requirement table."""

    table: str  # master | self_model | foundation | mq
    row_id: str
    id_suffix: str  # e.g. "SELF-EXTENT" for "`MQ-01 SELF-EXTENT`"
    status_raw: str  # empty for tables without a status column
    burden: str
    refs_raw: str  # detailed-scope / controls column (empty when absent)
    line: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "row_id": self.row_id,
            "id_suffix": self.id_suffix,
            "status_raw": self.status_raw,
            "burden": self.burden,
            "refs_raw": self.refs_raw,
            "line": self.line,
        }


@dataclass(frozen=True)
class NestedUnit:
    """A stable-ID bullet nested inside a numbered obligation."""

    unit_id: str
    status_raw: str
    text: str
    line: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "status_raw": self.status_raw,
            "text": self.text,
            "line": self.line,
        }


@dataclass(frozen=True)
class NumberedItem:
    """One numbered obligation from the Pass F / matrix / carryover lists."""

    list_key: str  # passf | matrix | carryover
    number: int
    title: str
    status_raw: str
    body: str
    line: int
    nested: tuple[NestedUnit, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "list_key": self.list_key,
            "number": self.number,
            "title": self.title,
            "status_raw": self.status_raw,
            "body": self.body,
            "line": self.line,
            "nested": [unit.to_dict() for unit in self.nested],
        }


@dataclass(frozen=True)
class TrackerExtraction:
    """The complete normative extraction plus the full ID-token census."""

    tracker_path: str
    table_rows: tuple[TableRow, ...]
    items: tuple[NumberedItem, ...]
    all_id_tokens: tuple[str, ...]  # every backticked ID-like token, sorted

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracker_path": self.tracker_path,
            "table_rows": [row.to_dict() for row in self.table_rows],
            "items": [item.to_dict() for item in self.items],
            "all_id_tokens": list(self.all_id_tokens),
        }

    def extraction_sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def declared_ids(self) -> set[str]:
        """IDs that own a table row or a nested unit definition."""
        ids = {row.row_id for row in self.table_rows}
        for item in self.items:
            ids.update(unit.unit_id for unit in item.nested)
        return ids


def _split_table_cells(line: str) -> list[str]:
    match = TABLE_ROW_RE.match(line)
    if not match:
        raise TrackerParseError(f"not a table row: {line[:80]!r}")
    return [cell.strip() for cell in match.group(1).split("|")]


def _find_heading(lines: list[str], prefix: str) -> int:
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            return index
    raise TrackerParseError(f"tracker heading not found: {prefix!r}")


def _next_heading(lines: list[str], start: int, levels: tuple[str, ...]) -> int:
    for index in range(start + 1, len(lines)):
        if any(lines[index].startswith(level) for level in levels):
            return index
    return len(lines)


def _parse_id_cell(cell: str) -> tuple[str, str]:
    """Parse a first-column cell like ``\\`SM-01-CANONICAL-SELF\\``.

    Returns (row_id, suffix). Cells such as ``\\`MQ-01 SELF-EXTENT\\`` carry a
    human suffix after the ID token.
    """
    stripped = cell.strip()
    if not (stripped.startswith("`") and stripped.endswith("`")):
        raise TrackerParseError(f"requirement-table id cell is not backticked: {cell!r}")
    inner = stripped[1:-1].strip()
    parts = inner.split(" ", 1)
    row_id = parts[0]
    suffix = parts[1].strip() if len(parts) > 1 else ""
    if not ID_TOKEN_RE.match(row_id):
        raise TrackerParseError(f"requirement-table id {row_id!r} is not a valid ID token")
    return row_id, suffix


def _strip_backticks(text: str) -> str:
    return text.replace("`", "").strip()


def _parse_requirement_table(
    lines: list[str], start: int, end: int, table: str
) -> list[TableRow]:
    """Parse one requirement table between ``start`` and ``end`` (exclusive)."""
    rows: list[TableRow] = []
    in_table = False
    columns = 0
    for index in range(start, end):
        line = lines[index]
        if not line.strip().startswith("|"):
            if in_table and rows:
                break  # table ended
            continue
        cells = _split_table_cells(line)
        if not in_table:
            in_table = True
            columns = len(cells)
            continue  # header row
        if set(line.replace("|", "").strip()) <= {"-", " ", ":"}:
            continue  # separator row
        if len(cells) != columns:
            raise TrackerParseError(
                f"{table} table row at line {index + 1} has {len(cells)} cells, "
                f"expected {columns}"
            )
        row_id, suffix = _parse_id_cell(cells[0])
        if table == "master":
            if columns != 4:
                raise TrackerParseError(
                    f"master table must have 4 columns, found {columns}"
                )
            status_raw = _strip_backticks(cells[1])
            burden = cells[2].strip()
            refs_raw = cells[3].strip()
        elif table in ("self_model", "foundation"):
            if columns != 3:
                raise TrackerParseError(
                    f"{table} table must have 3 columns, found {columns}"
                )
            status_raw = _strip_backticks(cells[1])
            burden = cells[2].strip()
            refs_raw = ""
        elif table == "mq":
            if columns != 3:
                raise TrackerParseError(f"mq table must have 3 columns, found {columns}")
            status_raw = ""
            burden = cells[1].strip()
            refs_raw = cells[2].strip()
        else:
            raise TrackerParseError(f"unknown table kind {table!r}")
        rows.append(
            TableRow(
                table=table,
                row_id=row_id,
                id_suffix=suffix,
                status_raw=status_raw,
                burden=burden,
                refs_raw=refs_raw,
                line=index + 1,
            )
        )
    if not rows:
        raise TrackerParseError(f"no rows parsed for {table} table")
    return rows


def _parse_item_status(trailing: str) -> str:
    """Extract a ``[STATUS]`` marker from a numbered item's title line."""
    cleaned = trailing.replace("`", " ")
    match = STATUS_MARKER_RE.search(cleaned)
    return match.group(1).strip() if match else ""


def _parse_numbered_lists(
    lines: list[str], start: int, end: int, region: str
) -> list[NumberedItem]:
    """Parse the numbered obligations in ``lines[start:end]``.

    Returns items in document order. Within the matrix region a numbering
    restart begins the ``carryover`` list; anywhere else a restart is an
    error. Numbering inside one list must be exactly sequential.
    """
    raw_items: list[tuple[int, str, str, int, list[str]]] = []
    current_body: list[str] | None = None
    for index in range(start, end):
        line = lines[index]
        match = NUMBERED_ITEM_RE.match(line)
        if match:
            number = int(match.group(1))
            title = match.group(2).strip()
            status_raw = _parse_item_status(match.group(3))
            current_body = []
            raw_items.append((number, title, status_raw, index + 1, current_body))
        elif current_body is not None:
            current_body.append(line)

    if not raw_items:
        raise TrackerParseError(f"no numbered obligations found in {region} region")

    items: list[NumberedItem] = []
    if region == "passf":
        list_keys = ["passf"]
    elif region == "matrix":
        list_keys = ["matrix", "carryover"]
    else:
        raise TrackerParseError(f"unknown numbered region {region!r}")

    list_index = 0
    expected = 1
    for number, title, status_raw, line_no, body_lines in raw_items:
        if number != expected:
            if number == 1 and list_index + 1 < len(list_keys):
                list_index += 1
                expected = 1
            else:
                raise TrackerParseError(
                    f"{region} numbering broke at line {line_no}: "
                    f"expected {expected}, found {number}"
                )
        body = "\n".join(body_lines).rstrip()
        nested: list[NestedUnit] = []
        for offset, body_line in enumerate(body_lines):
            nested_match = NESTED_ID_RE.match(body_line)
            if nested_match:
                nested.append(
                    NestedUnit(
                        unit_id=nested_match.group(1),
                        status_raw=(nested_match.group(2) or "").strip(),
                        text=body_line.strip(),
                        line=line_no + 1 + offset,
                    )
                )
        items.append(
            NumberedItem(
                list_key=list_keys[list_index],
                number=number,
                title=title,
                status_raw=status_raw,
                body=body,
                line=line_no,
                nested=tuple(nested),
            )
        )
        expected = number + 1

    return items


def _collect_id_tokens(text: str) -> tuple[str, ...]:
    """Every backticked ID-like token in the whole tracker, sorted unique."""
    tokens: set[str] = set()
    for line in text.splitlines():
        for match in BACKTICKED_ID_RE.finditer(line):
            tokens.add(match.group(1))
    return tuple(sorted(tokens))


def parse_tracker(path: Path) -> TrackerExtraction:
    """Parse the tracker file into its strict normative extraction."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise TrackerParseError(f"tracker not found: {path}") from exc
    lines = text.splitlines()

    master_start = _find_heading(lines, MASTER_INDEX_HEADING)
    self_model_start = _find_heading(lines, SELF_MODEL_HEADING)
    foundation_start = _find_heading(lines, FOUNDATION_HEADING)
    passf_start = _find_heading(lines, PASSF_HEADING)
    matrix_start = _find_heading(lines, MATRIX_HEADING)
    ledger_end = _find_heading(lines, LEDGER_END_HEADING)

    if not (
        master_start
        < self_model_start
        < foundation_start
        < passf_start
        < matrix_start
        < ledger_end
    ):
        raise TrackerParseError(
            "tracker normative sections are out of the expected order; "
            "update tools/reqproof/tracker_parse.py deliberately"
        )

    section_levels = ("### ", "#### ")
    master_end = _next_heading(lines, master_start, ("### ",))
    self_model_end = _next_heading(lines, self_model_start, ("### ",))
    foundation_end = _next_heading(lines, foundation_start, ("### ",))
    passf_end = _next_heading(lines, passf_start, section_levels)
    matrix_end = ledger_end

    table_rows: list[TableRow] = []
    table_rows.extend(_parse_requirement_table(lines, master_start, master_end, "master"))
    table_rows.extend(
        _parse_requirement_table(lines, self_model_start, self_model_end, "self_model")
    )
    # The self-model section holds two tables: the SM unit table and the MQ
    # question-to-evidence matrix. Parse the MQ table from where the SM one
    # ended.
    sm_last_line = max(row.line for row in table_rows if row.table == "self_model")
    table_rows.extend(
        _parse_requirement_table(lines, sm_last_line + 1, self_model_end, "mq")
    )
    table_rows.extend(
        _parse_requirement_table(lines, foundation_start, foundation_end, "foundation")
    )

    items: list[NumberedItem] = []
    items.extend(_parse_numbered_lists(lines, passf_start, passf_end, "passf"))
    items.extend(_parse_numbered_lists(lines, matrix_start, matrix_end, "matrix"))

    row_ids = [row.row_id for row in table_rows]
    duplicate_rows = sorted({rid for rid in row_ids if row_ids.count(rid) > 1})
    if duplicate_rows:
        raise TrackerParseError(f"duplicate requirement-table IDs: {duplicate_rows}")

    return TrackerExtraction(
        tracker_path=TRACKER_RELPATH,
        table_rows=tuple(table_rows),
        items=tuple(items),
        all_id_tokens=_collect_id_tokens(text),
    )
