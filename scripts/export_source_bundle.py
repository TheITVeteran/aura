"""Export Aura architecture + infrastructure source for external review.

Writes:
  - ~/Downloads/aura_source_part_N.txt (up to ~4M chars each, with file headers)
  - ~/Downloads/aura_source_copy/ (folder copy, max 1000 files, priority-ordered)

Architecture-only: skips caches, build artifacts, node_modules, data DBs,
screen recordings, model weights, large binary assets, and any redacted
personal-content paths. Keeps: Python core, infrastructure (scripts, CI,
docker, configs), HTML/CSS/JS interface, markdown docs, shell, Makefile.
"""
from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.atomic_writer import atomic_write_text

DOWNLOADS = Path.home() / "Downloads"

MAX_CHARS_PER_PART = 3_900_000  # 100K-char safety margin below the stated 4M cap
MAX_FOLDER_FILES = 1000

# Directories never included
EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".github",
    "node_modules",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".aura",
    ".zenflow",
    ".claude",
    "proof_kernel",  # separate repo
    ".scratch",
    "data",
    "logs",
    "recordings",
    "screenshots",
    "models",
    "weights",
    "cache",
    ".mlx",
    "tests/long_run_autonomy_state",
    "tests/life_trial",
}

# Filename globs never included (binary/data/generated)
EXCLUDE_GLOBS = [
    "*.sqlite3",
    "*.sqlite",
    "*.db",
    "*.log",
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dylib",
    "*.mp4",
    "*.mov",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.pdf",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.bin",
    "*.safetensors",
    "*.npz",
    "*.npy",
    "*.ico",
    "*.woff*",
    "*.ttf",
    ".DS_Store",
    "*.lock",  # yarn.lock etc
    "package-lock.json",
    "*.min.js",
    "*.min.css",
    "*.map",
]

# File types included (priority order)
PRIORITY_ORDER: List[Tuple[str, List[str]]] = [
    ("python_core", ["core/**/*.py"]),
    ("infrastructure", ["scripts/**/*", "Dockerfile*", "docker-compose*", "Makefile*", ".github/**/*", "pyproject.toml", "setup.cfg", "requirements*.txt"]),
    ("python_other", ["*.py", "tests/**/*.py", "tools/**/*.py", "aura_main/**/*.py"]),
    ("shell", ["*.sh", "scripts/*.sh"]),
    ("js_frontend", ["interface/**/*.js", "interface/**/*.jsx", "interface/**/*.ts", "interface/**/*.tsx"]),
    ("html_css", ["interface/**/*.html", "interface/**/*.css"]),
    ("docs", ["*.md", "docs/**/*.md"]),
    ("config", ["*.toml", "*.yml", "*.yaml", "*.json", "*.cfg", "*.ini"]),
]


def _should_skip(rel_path: Path, full_path: Path) -> bool:
    parts = set(rel_path.parts)
    if any(p in EXCLUDE_DIRS for p in parts):
        return True
    rel_posix = rel_path.as_posix()
    if any("/" in excluded and (rel_posix == excluded or rel_posix.startswith(f"{excluded}/")) for excluded in EXCLUDE_DIRS):
        return True
    name = rel_path.name
    for pattern in EXCLUDE_GLOBS:
        if fnmatch.fnmatch(name, pattern):
            return True
    try:
        if full_path.stat().st_size > 2_000_000:
            return True
    except OSError:
        return True
    return False


def _collect(root: Path) -> List[Path]:
    """Return architecture files in priority order, no duplicates."""
    seen: set[Path] = set()
    ordered: List[Path] = []
    for _category, patterns in PRIORITY_ORDER:
        for pattern in patterns:
            for p in root.glob(pattern):
                if not p.is_file():
                    continue
                rel = p.resolve()
                if rel in seen:
                    continue
                if _should_skip(p.relative_to(root), p):
                    continue
                seen.add(rel)
                ordered.append(p)
    return ordered


def _write_parts(files: List[Path], root: Path, downloads: Path) -> List[Path]:
    downloads.mkdir(parents=True, exist_ok=True)
    # Remove old parts so stale content never lingers
    warnings: list[str] = []
    for old in downloads.glob("aura_source_part_*.txt"):
        try:
            old.unlink()
        except OSError as exc:
            warnings.append(f"old export part cleanup skipped for {old}: {exc}")
    parts: List[Path] = []
    current_parts: List[str] = []
    current_size = 0
    part_index = 1

    def flush() -> None:
        nonlocal current_parts, current_size, part_index
        if not current_parts:
            return
        path = downloads / f"aura_source_part_{part_index}.txt"
        atomic_write_text(path, "".join(current_parts), encoding="utf-8")
        parts.append(path)
        part_index += 1
        current_parts = []
        current_size = 0

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeError) as exc:
            warnings.append(f"text export skipped {path}: {exc}")
            continue
        rel = path.relative_to(root)
        header = (
            "\n" + "=" * 80 + "\n"
            f"# {rel}\n"
            + "=" * 80 + "\n\n"
        )
        chunk = header + text
        if current_size + len(chunk) > MAX_CHARS_PER_PART and current_parts:
            flush()
        current_parts.append(chunk)
        current_size += len(chunk)
    flush()
    _print_warnings(warnings)
    return parts


def _copy_folder(files: List[Path], root: Path, downloads: Path, limit: int = MAX_FOLDER_FILES) -> Path:
    dest = downloads / "aura_source_copy"
    warnings: list[str] = []
    if dest.exists():
        try:
            shutil.rmtree(dest)
        except (OSError, shutil.Error) as exc:
            raise RuntimeError(f"Unable to reset source copy folder {dest}: {exc}") from exc
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in files:
        if written >= limit:
            break
        rel = path.relative_to(root)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(path, target)
            written += 1
        except (OSError, shutil.Error) as exc:
            warnings.append(f"folder copy skipped {path}: {exc}")
            continue
    _print_warnings(warnings)
    return dest


def _print_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    print(f"Completed with {len(warnings)} skipped source-bundle operation(s):", file=sys.stderr)
    for warning in warnings[:20]:
        print(f"  - {warning}", file=sys.stderr)
    if len(warnings) > 20:
        print(f"  - ... {len(warnings) - 20} more", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--downloads", type=Path, default=DOWNLOADS)
    args = parser.parse_args()

    files = _collect(args.root)
    print(f"Collected {len(files)} architecture files")

    parts = _write_parts(files, args.root, args.downloads)
    print(f"Wrote {len(parts)} part files:")
    for p in parts:
        size = p.stat().st_size
        lines = len(p.read_text(encoding="utf-8", errors="replace").splitlines())
        print(f"  {p.name}: {size:,} bytes, {lines:,} lines")

    folder = _copy_folder(files, args.root, args.downloads)
    copied = sum(1 for _ in folder.rglob("*") if _.is_file())
    print(f"Copied {copied} files to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
