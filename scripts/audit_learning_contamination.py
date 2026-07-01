#!/usr/bin/env python3
"""Audit or rewrite Aura learning JSONL files that contain contaminated rows."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.live_learner import training_contamination_reasons  # noqa: E402


def _row_texts(row: object) -> list[str]:
    if not isinstance(row, dict):
        return [str(row)]
    texts: list[str] = []
    if isinstance(row.get("text"), str):
        texts.append(str(row.get("text") or ""))
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                texts.append(str(message.get("content") or ""))
    return texts


def audit_file(path: Path, *, rewrite: bool, quarantine_dir: Path | None) -> dict[str, object]:
    total = kept = contaminated = malformed = 0
    clean_lines: list[str] = []
    bad_lines: list[str] = []
    reasons: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            bad_lines.append(line)
            continue
        row_reasons = training_contamination_reasons(*_row_texts(row))
        if row_reasons:
            contaminated += 1
            bad_lines.append(line)
            for reason in row_reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
            continue
        kept += 1
        clean_lines.append(line)

    if rewrite and (contaminated or malformed):
        if quarantine_dir is None:
            raise ValueError("quarantine_dir is required when rewrite=True")
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        rel_name = str(path).strip("/").replace("/", "__")
        quarantine_path = quarantine_dir / f"{rel_name}.contaminated.jsonl"
        quarantine_path.write_text("\n".join(bad_lines) + ("\n" if bad_lines else ""), encoding="utf-8")
        path.write_text("\n".join(clean_lines) + ("\n" if clean_lines else ""), encoding="utf-8")

    return {
        "path": str(path),
        "total": total,
        "kept": kept,
        "contaminated": contaminated,
        "malformed": malformed,
        "reasons": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / ".aura" / "data" / "learning")
    parser.add_argument("--rewrite", action="store_true", help="rewrite files in place and quarantine rejected rows")
    parser.add_argument("--quarantine-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    quarantine_dir = args.quarantine_dir
    if args.rewrite and quarantine_dir is None:
        quarantine_dir = args.root / "quarantine" / time.strftime("%Y%m%d-%H%M%S")

    files = [
        path for path in sorted(args.root.rglob("*.jsonl"))
        if "quarantine" not in path.relative_to(args.root).parts
    ] if args.root.exists() else []
    reports = [audit_file(path, rewrite=args.rewrite, quarantine_dir=quarantine_dir) for path in files]
    total_contaminated = sum(int(report["contaminated"]) for report in reports)
    total_malformed = sum(int(report["malformed"]) for report in reports)
    touched = [report for report in reports if report["contaminated"] or report["malformed"]]
    print(
        json.dumps(
            {
                "ok": total_contaminated == 0 and total_malformed == 0,
                "mode": "rewrite" if args.rewrite else "audit",
                "root": str(args.root),
                "files_scanned": len(files),
                "files_with_findings": len(touched),
                "contaminated_rows": total_contaminated,
                "malformed_rows": total_malformed,
                "quarantine_dir": str(quarantine_dir or ""),
                "findings": touched[:200],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if total_contaminated or total_malformed else 0


if __name__ == "__main__":
    raise SystemExit(main())
