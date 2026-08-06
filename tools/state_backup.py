"""Verified, bounded state backup for Aura's persistent stores.

The previous `make backup` piped tar through `|| true` — it reported
success unconditionally, copied hot WAL SQLite files byte-raw (a recipe
for unreadable snapshots), kept no manifest, and the ring grew without
bound (17GB of stale archives by July 2026). A backup that has never been
restored is a hope, not a backup.

This tool:
  - snapshots every SQLite store through the sqlite3 backup API, so hot
    databases (live instance running) still yield consistent copies;
  - copies everything else as plain files, honoring the historical
    excludes (data/training, data/error_logs, caches);
  - writes a sha256+size manifest line per archive to manifest.jsonl;
  - prunes its own ring past --keep (legacy aura_backup_* archives are
    reported, never touched);
  - `verify` extracts an archive to a temp dir, re-hashes it against the
    manifest, and runs PRAGMA quick_check on every extracted SQLite store.

Archive layout matches the historical backups (relative data/, storage/,
.aura_runtime/, .aura_snapshots/), so `make restore` keeps working.

Exit codes: 0 success, 1 failure — loudly, never `|| true`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from core.runtime.sqlite_support import connecting

STATE_ROOTS = ("data", "storage", ".aura_runtime", ".aura_snapshots")
EXCLUDE_RELATIVE = ("data/training", "data/error_logs", "data/bench")
EXCLUDE_NAMES = {"__pycache__", ".pytest_cache"}
SQLITE_MAGIC = b"SQLite format 3\x00"
ARCHIVE_PREFIX = "aura_state_"


def _is_sqlite(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(16) == SQLITE_MAGIC
    except OSError:
        return False


def _consistent_db_copy(src: Path, dest: Path) -> str:
    """Copy a SQLite store through the backup API (WAL-safe). Returns how."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with connecting(
            sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=30.0)
        ) as conn, connecting(sqlite3.connect(dest)) as out:
            conn.backup(out)
        return "sqlite-backup-api"
    except sqlite3.Error:
        # Corrupt or locked-beyond-timeout stores still deserve a byte copy:
        # a raw snapshot of a broken DB beats no snapshot when doing forensics.
        shutil.copy2(src, dest)
        return "raw-copy-fallback"


def _should_exclude(rel: Path) -> bool:
    rel_str = rel.as_posix()
    if any(rel_str == e or rel_str.startswith(e + "/") for e in EXCLUDE_RELATIVE):
        return True
    return any(part in EXCLUDE_NAMES for part in rel.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class StageStats:
    files: int = 0
    db_api_copies: int = 0
    db_raw_fallbacks: int = 0
    bytes: int = 0


def _stage_tree(root: Path, stage: Path) -> StageStats:
    stats = StageStats()
    for state_root in STATE_ROOTS:
        src_root = root / state_root
        if not src_root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(src_root):
            dpath = Path(dirpath)
            rel_dir = dpath.relative_to(root)
            dirnames[:] = [
                d for d in dirnames if not _should_exclude(rel_dir / d)
            ]
            for fname in filenames:
                src = dpath / fname
                rel = src.relative_to(root)
                if _should_exclude(rel):
                    continue
                # WAL/SHM siblings are folded into the backup-API copy of
                # their main store; a standalone copy would be inconsistent.
                if fname.endswith(("-wal", "-shm")):
                    continue
                if src.is_symlink():
                    continue
                dest = stage / rel
                if _is_sqlite(src):
                    how = _consistent_db_copy(src, dest)
                    if how == "sqlite-backup-api":
                        stats.db_api_copies += 1
                    else:
                        stats.db_raw_fallbacks += 1
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                stats.files += 1
                stats.bytes += dest.stat().st_size
    return stats


def _port_serving(port: int = 8000) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def create_backup(root: Path, out_dir: Path, keep: int = 7) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if _port_serving():
        print("ℹ️  live instance detected on :8000 — SQLite stores are still "
              "snapshotted consistently via the backup API")
    name = f"{ARCHIVE_PREFIX}{time.strftime('%Y%m%d_%H%M%S')}-{os.getpid()}"
    archive = out_dir / f"{name}.tar.gz"
    serial = 0
    while archive.exists():
        serial += 1
        archive = out_dir / f"{name}-{serial}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="aura_backup_stage_") as tmp:
        stage = Path(tmp)
        stats = _stage_tree(root, stage)
        if stats.files == 0:
            raise SystemExit(f"❌ nothing to back up under {root} "
                             f"(state roots: {', '.join(STATE_ROOTS)})")
        with tarfile.open(archive, "w:gz") as tar:
            for entry in sorted(stage.iterdir()):
                tar.add(entry, arcname=entry.name)
    manifest_entry = {
        "archive": archive.name,
        "sha256": _sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "staged_files": stats.files,
        "staged_bytes": stats.bytes,
        "sqlite_api_copies": stats.db_api_copies,
        "sqlite_raw_fallbacks": stats.db_raw_fallbacks,
        "root": str(root),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("a") as fh:
        fh.write(json.dumps(manifest_entry) + "\n")
    pruned = prune_ring(out_dir, keep)
    print(f"✅ backup {archive.name}: {stats.files} files, "
          f"{stats.db_api_copies} consistent DB snapshots"
          + (f", {stats.db_raw_fallbacks} raw DB fallbacks" if stats.db_raw_fallbacks else "")
          + f", {archive.stat().st_size / 1e6:.0f}MB archive"
          + (f"; pruned {pruned} old ring archives" if pruned else ""))
    legacy = [p for p in out_dir.glob("aura_backup_*") if p.is_file()]
    if legacy:
        legacy_gb = sum(p.stat().st_size for p in legacy) / 1e9
        print(f"ℹ️  {len(legacy)} legacy aura_backup_* archives ({legacy_gb:.1f}GB) "
              "left untouched — prune manually when confident")
    return archive


def prune_ring(out_dir: Path, keep: int) -> int:
    ring = sorted(out_dir.glob(f"{ARCHIVE_PREFIX}*.tar.gz"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    pruned = 0
    for old in ring[keep:]:
        old.unlink()
        pruned += 1
    return pruned


def _manifest_entry_for(out_dir: Path, archive: Path) -> dict[str, Any] | None:
    manifest = out_dir / "manifest.jsonl"
    if not manifest.exists():
        return None
    entry = None
    for line in manifest.read_text().splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate.get("archive") == archive.name:
            entry = candidate  # last write wins
    return entry


def verify_backup(out_dir: Path, archive: Path | None = None) -> int:
    if archive is None:
        ring = sorted(out_dir.glob(f"{ARCHIVE_PREFIX}*.tar.gz"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not ring:
            print(f"❌ no {ARCHIVE_PREFIX}*.tar.gz archives in {out_dir}")
            return 1
        archive = ring[0]
    if not archive.exists():
        print(f"❌ archive missing: {archive}")
        return 1

    entry = _manifest_entry_for(out_dir, archive)
    if entry is None:
        print(f"⚠️  no manifest entry for {archive.name} — hash provenance "
              "unavailable; continuing with structural checks")
    elif _sha256(archive) != entry["sha256"]:
        print(f"❌ sha256 mismatch for {archive.name} — archive corrupted "
              "or tampered since creation")
        return 1

    checked = failed = 0
    with tempfile.TemporaryDirectory(prefix="aura_backup_verify_") as tmp:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp, filter="data")
        for path in Path(tmp).rglob("*"):
            if not (path.is_file() and _is_sqlite(path)):
                continue
            checked += 1
            try:
                with connecting(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
                    verdict = conn.execute("PRAGMA quick_check").fetchone()[0]
            except sqlite3.Error as exc:
                verdict = f"unreadable: {exc}"
            if verdict != "ok":
                failed += 1
                rel = path.relative_to(tmp)
                print(f"❌ quick_check failed for {rel}: {verdict}")
    if failed:
        print(f"❌ verify FAILED: {failed}/{checked} SQLite stores unhealthy "
              f"in {archive.name}")
        return 1
    print(f"✅ verify OK: {archive.name} — hash matches manifest, "
          f"{checked} SQLite stores pass quick_check")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    default_out = Path.home() / ".aura" / "backups"

    p_create = sub.add_parser("create", help="create a verified backup archive")
    p_create.add_argument("--root", type=Path,
                          default=Path(__file__).resolve().parents[1])
    p_create.add_argument("--out", type=Path, default=default_out)
    p_create.add_argument("--keep", type=int, default=7)

    p_verify = sub.add_parser("verify", help="restore-verify an archive")
    p_verify.add_argument("--out", type=Path, default=default_out)
    p_verify.add_argument("--archive", type=Path, default=None,
                          help="defaults to the newest ring archive")

    args = parser.parse_args(argv)
    if args.cmd == "create":
        create_backup(args.root, args.out, keep=args.keep)
        return 0
    return verify_backup(args.out, args.archive)


if __name__ == "__main__":
    raise SystemExit(main())
