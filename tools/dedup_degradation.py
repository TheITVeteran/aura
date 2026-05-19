#!/usr/bin/env python3
"""Remove duplicate consecutive record_degradation calls.

Pattern to fix:
    record_degradation('foo', e)
    record_degradation('foo', e)   # <-- duplicate, remove this

Also fixes double-spacing left behind.
"""

import re
import sys
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv"}


def find_python_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def process_file(filepath: Path, dry_run: bool = False) -> int:
    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return 0

    lines = content.split("\n")
    new_lines = []
    removals = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this line is a record_degradation call
        if stripped.startswith("record_degradation("):
            # Check if the NEXT line is identical
            if i + 1 < len(lines) and lines[i + 1].strip() == stripped:
                # Skip the duplicate
                new_lines.append(line)
                i += 2  # Skip both, keep only one
                removals += 1
                continue

        new_lines.append(line)
        i += 1

    if removals > 0 and not dry_run:
        filepath.write_text("\n".join(new_lines), encoding="utf-8")

    return removals


def main():
    dry_run = "--dry-run" in sys.argv
    root = Path(__file__).resolve().parents[1]

    search_dirs = [root / "core", root / "skills"]
    all_files = []
    for d in search_dirs:
        if d.exists():
            all_files.extend(find_python_files(d))

    total = 0
    changed = 0

    for filepath in all_files:
        rel = filepath.relative_to(root)
        count = process_file(filepath, dry_run=dry_run)
        if count > 0:
            print(f"  {'[DRY] ' if dry_run else ''}Removed {count:3d} duplicates in {rel}")
            total += count
            changed += 1

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Total: {total} duplicate record_degradation calls removed across {changed} files.")


if __name__ == "__main__":
    main()
