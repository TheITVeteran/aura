#!/usr/bin/env python3
"""Record and summarize semantic closeout review coverage.

The mechanical closeout audit can prove that every tracked text line was
enumerated and hashed. This ledger covers the separate claim that a human or
agent actually reviewed specific file spans. Each entry stores the current file
hash and span hash so later audit runs can detect stale review evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_LEDGER = ROOT / "artifacts" / "closeout" / "semantic_review" / "SEMANTIC_REVIEW_LEDGER.jsonl"


@dataclass(frozen=True)
class CurrentFile:
    path: str
    sha256: str
    text: bool
    code: bool
    line_count: int
    lines: tuple[str, ...]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rel(path: Path, root: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _run_git_ls_files(root: Path = ROOT) -> list[Path]:
    from tools.closeout.run_codebase_closeout_audit import tracked_files

    return tracked_files() if root.resolve() == ROOT.resolve() else []


def _is_text(path: Path, data: bytes) -> bool:
    try:
        from tools.closeout.run_codebase_closeout_audit import _is_probably_text

        return bool(_is_probably_text(path, data))
    except (ImportError, AttributeError):
        if b"\0" in data[:8192]:
            return False
        try:
            data[:8192].decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False


def _is_code(path: Path) -> bool:
    try:
        from tools.closeout.run_codebase_closeout_audit import CODE_EXTENSIONS

        return path.suffix.lower() in CODE_EXTENSIONS or path.name == "Makefile"
    except (ImportError, AttributeError):
        return path.suffix.lower() in {
            ".css",
            ".html",
            ".js",
            ".jsx",
            ".mjs",
            ".py",
            ".rs",
            ".sh",
            ".sql",
            ".ts",
            ".tsx",
        } or path.name == "Makefile"


def current_file(path: Path, *, root: Path = ROOT) -> CurrentFile:
    data = path.read_bytes()
    text = _is_text(path, data)
    lines: tuple[str, ...] = ()
    if text:
        lines = tuple(data.decode("utf-8", errors="replace").splitlines())
    return CurrentFile(
        path=_rel(path, root),
        sha256=_sha256_bytes(data),
        text=text,
        code=_is_code(path),
        line_count=len(lines),
        lines=lines,
    )


def span_sha256(lines: Iterable[str], first_line: int, last_line: int) -> str:
    selected = list(lines)[first_line - 1 : last_line]
    return _sha256_bytes("\n".join(selected).encode("utf-8", errors="replace"))


def _normalize_span(line_count: int, first_line: int | None, last_line: int | None) -> tuple[int, int]:
    if line_count <= 0:
        return (0, 0)
    first = 1 if first_line is None else int(first_line)
    last = line_count if last_line is None else int(last_line)
    if first < 1 or last < first or last > line_count:
        raise ValueError(f"invalid span {first}:{last} for {line_count} lines")
    return (first, last)


def build_review_entry(
    path: Path,
    *,
    reviewer: str,
    checkpoint_id: str,
    note: str = "",
    tests: list[str] | None = None,
    findings: list[dict[str, Any]] | None = None,
    first_line: int | None = None,
    last_line: int | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    info = current_file(path, root=root)
    first, last = _normalize_span(info.line_count, first_line, last_line)
    entry = {
        "schema": "aura.closeout.semantic_review_entry.v1",
        "reviewed_at_unix": time.time(),
        "reviewer": reviewer,
        "checkpoint_id": checkpoint_id,
        "file": info.path,
        "file_sha256": info.sha256,
        "text": info.text,
        "line_count": info.line_count,
        "first_line": first,
        "last_line": last,
        "span_sha256": span_sha256(info.lines, first, last) if info.text and first else "",
        "tests": tests or [],
        "findings": findings or [],
        "note": note,
        "claim_supported": "semantic_review_of_recorded_file_span",
        "claim_not_supported": [
            "semantic_review_of_unrecorded_spans",
            "all_issues_fixed",
            "full_closeout_complete",
        ],
    }
    return entry


def append_entries(ledger_path: Path, entries: list[dict[str, Any]]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


def read_entries(ledger_path: Path) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    entries = []
    for line_no, raw in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            entries.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid semantic review ledger JSON at {ledger_path}:{line_no}") from exc
    return entries


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    merged: list[tuple[int, int]] = []
    for first, last in sorted(spans):
        if first <= 0 or last <= 0:
            continue
        if not merged or first > merged[-1][1] + 1:
            merged.append((first, last))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], last))
    return merged


def _span_count(spans: list[tuple[int, int]]) -> int:
    return sum(last - first + 1 for first, last in spans)


def _entry_is_current(entry: dict[str, Any], info: CurrentFile) -> tuple[bool, str]:
    if not info.text:
        return (False, "current_file_is_binary")
    if entry.get("file_sha256") != info.sha256:
        return (False, "file_hash_changed")
    first = int(entry.get("first_line", 0))
    last = int(entry.get("last_line", 0))
    if first < 1 or last < first or last > info.line_count:
        return (False, "recorded_span_invalid_for_current_file")
    if entry.get("span_sha256") != span_sha256(info.lines, first, last):
        return (False, "span_hash_changed")
    return (True, "")


def summarize_semantic_reviews(
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    tracked_paths: list[Path] | None = None,
    root: Path = ROOT,
    unreviewed_limit: int = 100,
) -> dict[str, Any]:
    tracked_paths = tracked_paths if tracked_paths is not None else _run_git_ls_files(root)
    current_by_path: dict[str, CurrentFile] = {}
    for path in tracked_paths:
        if path.is_file():
            info = current_file(path, root=root)
            if info.text:
                current_by_path[info.path] = info

    entries = read_entries(ledger_path)
    current_spans: dict[str, list[tuple[int, int]]] = {}
    stale_candidates: list[dict[str, Any]] = []
    orphan_entries: list[dict[str, Any]] = []
    for entry in entries:
        file_name = str(entry.get("file", ""))
        info = current_by_path.get(file_name)
        if info is None:
            orphan_entries.append(
                {
                    "file": file_name,
                    "checkpoint_id": entry.get("checkpoint_id", ""),
                    "file_sha256": entry.get("file_sha256", ""),
                    "reason": "file_not_tracked_or_not_text",
                }
            )
            continue
        ok, reason = _entry_is_current(entry, info)
        if not ok:
            stale_candidates.append(
                {
                    "file": file_name,
                    "checkpoint_id": entry.get("checkpoint_id", ""),
                    "reason": reason,
                }
            )
            continue
        current_spans.setdefault(file_name, []).append((int(entry["first_line"]), int(entry["last_line"])))

    reviewed_files: dict[str, dict[str, Any]] = {}
    reviewed_line_count = 0
    reviewed_code_line_count = 0
    fully_reviewed_count = 0
    fully_reviewed_code_count = 0
    fully_reviewed_files: set[str] = set()
    for file_name, spans in sorted(current_spans.items()):
        info = current_by_path[file_name]
        merged = _merge_spans(spans)
        line_count = _span_count(merged)
        reviewed_line_count += line_count
        if info.code:
            reviewed_code_line_count += line_count
        fully_reviewed = line_count == info.line_count
        fully_reviewed_count += int(fully_reviewed)
        fully_reviewed_code_count += int(fully_reviewed and info.code)
        if fully_reviewed:
            fully_reviewed_files.add(file_name)
        reviewed_files[file_name] = {
            "code": info.code,
            "line_count": info.line_count,
            "reviewed_line_count": line_count,
            "fully_reviewed": fully_reviewed,
            "spans": [{"first_line": first, "last_line": last} for first, last in merged],
        }

    stale_entries: list[dict[str, Any]] = []
    superseded_stale_entries: list[dict[str, Any]] = []
    superseded_orphan_entries: list[dict[str, Any]] = []
    current_hashes_by_review_state = {
        current_by_path[file_name].sha256
        for file_name in fully_reviewed_files
        if file_name in current_by_path
    }
    active_orphan_entries: list[dict[str, Any]] = []
    for orphan in orphan_entries:
        if orphan.get("file_sha256") in current_hashes_by_review_state:
            superseded_orphan_entries.append(orphan)
        else:
            active_orphan_entries.append(orphan)
    for candidate in stale_candidates:
        if str(candidate.get("file", "")) in fully_reviewed_files:
            superseded_stale_entries.append(candidate)
        else:
            stale_entries.append(candidate)

    total_text_lines = sum(info.line_count for info in current_by_path.values())
    total_code_lines = sum(info.line_count for info in current_by_path.values() if info.code)
    coverage = reviewed_line_count / total_text_lines if total_text_lines else 0.0
    code_coverage = reviewed_code_line_count / total_code_lines if total_code_lines else 0.0
    unreviewed_files: list[dict[str, Any]] = []
    for file_name, info in sorted(current_by_path.items()):
        reviewed = reviewed_files.get(file_name, {})
        reviewed_lines = int(reviewed.get("reviewed_line_count", 0))
        if reviewed_lines >= info.line_count:
            continue
        unreviewed_files.append(
            {
                "file": file_name,
                "code": info.code,
                "line_count": info.line_count,
                "reviewed_line_count": reviewed_lines,
                "missing_line_count": info.line_count - reviewed_lines,
            }
        )
    unreviewed_files.sort(
        key=lambda item: (
            not bool(item["code"]),
            -int(item["missing_line_count"]),
            str(item["file"]),
        )
    )
    return {
        "schema": "aura.closeout.semantic_review_status.v1",
        "ledger_path": str(ledger_path),
        "ledger_exists": ledger_path.exists(),
        "entry_count": len(entries),
        "tracked_text_file_count": len(current_by_path),
        "tracked_code_file_count": sum(1 for info in current_by_path.values() if info.code),
        "tracked_text_line_count": total_text_lines,
        "tracked_code_line_count": total_code_lines,
        "reviewed_file_count": len(reviewed_files),
        "fully_reviewed_text_file_count": fully_reviewed_count,
        "fully_reviewed_code_file_count": fully_reviewed_code_count,
        "semantic_reviewed_line_count": reviewed_line_count,
        "semantic_reviewed_code_line_count": reviewed_code_line_count,
        "semantic_review_coverage_ratio": round(coverage, 6),
        "semantic_code_review_coverage_ratio": round(code_coverage, 6),
        "unreviewed_file_count": len(unreviewed_files),
        "unreviewed_code_file_count": sum(1 for item in unreviewed_files if item["code"]),
        "unreviewed_files": unreviewed_files[: max(0, int(unreviewed_limit))],
        "stale_review_count": len(stale_entries),
        "superseded_stale_review_count": len(superseded_stale_entries),
        "orphan_review_count": len(active_orphan_entries),
        "superseded_orphan_review_count": len(superseded_orphan_entries),
        "stale_reviews": stale_entries[:100],
        "superseded_stale_reviews": superseded_stale_entries[:100],
        "orphan_reviews": active_orphan_entries[:100],
        "superseded_orphan_reviews": superseded_orphan_entries[:100],
        "reviewed_files": reviewed_files,
        "full_semantic_review_current": (
            bool(current_by_path)
            and fully_reviewed_count == len(current_by_path)
            and reviewed_line_count == total_text_lines
            and not stale_entries
            and not active_orphan_entries
        ),
        "claim_supported": "semantic_review_coverage_status",
        "claim_not_supported": [
            "semantic_review_without_matching_hash_receipts",
            "all_issues_fixed",
            "full_closeout_complete",
        ],
    }


def _select_paths(args: argparse.Namespace) -> list[Path]:
    if args.paths:
        return [Path(path).resolve() if Path(path).is_absolute() else (ROOT / path).resolve() for path in args.paths]
    if not args.path_prefix and not args.all_tracked:
        raise ValueError("record requires explicit paths, --path-prefix, or --all-tracked")

    tracked = _run_git_ls_files(ROOT)
    prefixes = [prefix.strip("/") for prefix in args.path_prefix]
    if prefixes:
        tracked = [path for path in tracked if any(_rel(path).startswith(prefix) for prefix in prefixes)]
    if args.max_files is not None:
        tracked = tracked[: max(0, int(args.max_files))]
    return tracked


def record_reviews_from_args(args: argparse.Namespace) -> dict[str, Any]:
    paths = _select_paths(args)
    tests = list(args.test)
    entries = [
        build_review_entry(
            path,
            reviewer=args.reviewer,
            checkpoint_id=args.checkpoint_id,
            note=args.note,
            tests=tests,
            first_line=args.first_line,
            last_line=args.last_line,
        )
        for path in paths
        if path.is_file()
    ]
    ledger_path = Path(args.ledger)
    if entries:
        append_entries(ledger_path, entries)
    summary = summarize_semantic_reviews(ledger_path=ledger_path)
    return {
        "schema": "aura.closeout.semantic_review_record_result.v1",
        "ledger_path": str(ledger_path),
        "recorded_entry_count": len(entries),
        "summary": summary,
    }


def build_semantic_review_queue(
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    limit: int = 50,
    code_only: bool = False,
    tracked_paths: list[Path] | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    limit = max(0, int(limit))
    summary = summarize_semantic_reviews(
        ledger_path=ledger_path,
        tracked_paths=tracked_paths,
        root=root,
        unreviewed_limit=max(limit, 100),
    )
    files = summary["unreviewed_files"]
    if code_only:
        files = [item for item in files if item["code"]]
    return {
        "schema": "aura.closeout.semantic_review_queue.v1",
        "ledger_path": summary["ledger_path"],
        "limit": limit,
        "code_only": bool(code_only),
        "unreviewed_file_count": summary["unreviewed_file_count"],
        "unreviewed_code_file_count": summary["unreviewed_code_file_count"],
        "files": files[:limit],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Append semantic review receipts for paths.")
    record.add_argument("--ledger", default=os.environ.get("AURA_SEMANTIC_REVIEW_LEDGER", str(DEFAULT_LEDGER)))
    record.add_argument("--reviewer", default=os.environ.get("AURA_REVIEWER", "codex"))
    record.add_argument("--checkpoint-id", default=os.environ.get("AURA_CLOSEOUT_CHECKPOINT", "manual"))
    record.add_argument("--note", default="")
    record.add_argument("--test", action="append", default=[])
    record.add_argument("--first-line", type=int, default=None)
    record.add_argument("--last-line", type=int, default=None)
    record.add_argument("--path-prefix", action="append", default=[])
    record.add_argument("--all-tracked", action="store_true")
    record.add_argument("--max-files", type=int, default=None)
    record.add_argument("paths", nargs="*")

    status = subparsers.add_parser("status", help="Summarize current semantic review coverage.")
    status.add_argument("--ledger", default=os.environ.get("AURA_SEMANTIC_REVIEW_LEDGER", str(DEFAULT_LEDGER)))

    queue = subparsers.add_parser("queue", help="List unreviewed files in closeout priority order.")
    queue.add_argument("--ledger", default=os.environ.get("AURA_SEMANTIC_REVIEW_LEDGER", str(DEFAULT_LEDGER)))
    queue.add_argument("--limit", type=int, default=50)
    queue.add_argument("--code-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "record":
        payload = record_reviews_from_args(args)
    elif args.command == "status":
        payload = summarize_semantic_reviews(ledger_path=Path(args.ledger))
    else:
        payload = build_semantic_review_queue(
            ledger_path=Path(args.ledger),
            limit=args.limit,
            code_only=args.code_only,
        )
    try:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True, default=str))
        sys.stdout.write("\n")
        sys.stdout.flush()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except OSError:
            pass
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
