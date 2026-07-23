"""Rollback packet creation, dry-run verification, and restoration.

Restore WRITES INTO LIVE SOURCE from a JSON packet on disk, so the packet is
treated as untrusted input: its identity is recomputed, its paths are
contained, and a restore is all-or-nothing against the generation it claims
to revert.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from core.architect.config import ASAConfig
from core.architect.errors import RollbackError
from core.architect.models import RefactorPlan, RollbackPacket
from core.architect.shadow_workspace import ShadowRun
from core.runtime.atomic_writer import atomic_write_bytes, atomic_write_text

#: A file that did not exist in the pre-promotion generation. Recorded so a
#: plan that CREATES files can still be truthfully rolled back (CP126
#: 78c68165) — restoring "absent" means deleting it again.
ABSENT_SENTINEL = "absent"


def _contained(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root`` or refuse.

    CP126 dd364ff8: plan paths were joined to the live and packet roots with
    no canonical containment and a loaded ``packet_path`` was trusted
    directly, so crafted plans or packet JSON could read and overwrite files
    outside the repository and artifact store.
    """
    raw = str(rel or "").strip()
    if not raw:
        raise RollbackError("rollback path is empty")
    candidate = Path(raw.replace("\\", "/"))
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise RollbackError(f"rollback path escapes its root: {rel}")
    base = root.resolve(strict=False)
    target = (base / candidate).resolve(strict=False)
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise RollbackError(f"rollback path escapes its root: {rel}") from exc
    return target


def compute_receipt_hash(
    run_id: str,
    repo_hash: str,
    original_hashes: dict[str, str],
    candidate_hashes: dict[str, str],
) -> str:
    """The packet's identity, recomputed the same way everywhere."""
    return hashlib.sha256(
        json.dumps(
            {
                "run_id": run_id,
                "repo_hash": repo_hash,
                "original_hashes": original_hashes,
                "candidate_hashes": candidate_hashes,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


class RollbackManager:
    """Create immutable rollback packets before promotion."""

    def __init__(self, config: ASAConfig | None = None):
        self.config = config or ASAConfig.from_env()
        self.root = self.config.artifacts / "rollback"
        self.root.mkdir(parents=True, exist_ok=True)

    def create_packet(self, plan: RefactorPlan, shadow: ShadowRun) -> RollbackPacket:
        changed = tuple(dict.fromkeys(plan.changed_files or shadow.changed_files))
        if not changed:
            raise RollbackError("cannot create rollback packet without changed files")
        packet_dir = self.root / shadow.run_id
        original_dir = packet_dir / "original"
        candidate_dir = packet_dir / "candidate"
        original_hashes: dict[str, str] = {}
        candidate_hashes: dict[str, str] = {}
        for rel in changed:
            live = _contained(self.config.repo_root, rel)
            raw_candidate = str(shadow.candidate_files.get(rel, "") or "").strip()
            candidate = (
                Path(raw_candidate)
                if raw_candidate
                else _contained(Path(shadow.shadow_root), rel)
            )
            # CP126 78c68165: requiring BOTH files to exist meant a plan that
            # CREATES a file could not get truthful rollback coverage at all.
            # An absent original is recorded as such; restoring it deletes the
            # file the promotion created.
            if live.exists():
                live_bytes = live.read_bytes()
                original_hashes[rel] = hashlib.sha256(live_bytes).hexdigest()
                original_target = _contained(original_dir, rel)
                original_target.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(original_target, live_bytes)
            else:
                original_hashes[rel] = ABSENT_SENTINEL
            if not candidate.is_file():
                raise RollbackError(f"candidate file missing before rollback packet: {rel}")
            candidate_bytes = candidate.read_bytes()
            candidate_hashes[rel] = hashlib.sha256(candidate_bytes).hexdigest()
            candidate_target = _contained(candidate_dir, rel)
            candidate_target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(candidate_target, candidate_bytes)
        repo_hash = self._repo_hash(changed)
        receipt_hash = compute_receipt_hash(
            shadow.run_id, repo_hash, original_hashes, candidate_hashes
        )
        packet = RollbackPacket(
            run_id=shadow.run_id,
            timestamp=time.time(),
            repo_root_hash=repo_hash,
            changed_files=changed,
            original_hashes=original_hashes,
            candidate_hashes=candidate_hashes,
            packet_path=str(packet_dir),
            receipt_hash=receipt_hash,
            dry_run_passed=False,
            post_restore_verified=False,
        )
        atomic_write_text(packet_dir / "packet.json", json.dumps(packet_to_dict(packet), indent=2, sort_keys=True, default=str))
        return packet

    def dry_run(self, packet: RollbackPacket) -> RollbackPacket:
        packet_dir = Path(packet.packet_path)
        failures: list[str] = []
        for rel, expected in packet.original_hashes.items():
            live = _contained(self.config.repo_root, rel)
            if expected == ABSENT_SENTINEL:
                # The original generation did not have this file; a dry run
                # succeeds when the live tree still agrees.
                if live.exists():
                    failures.append(f"{rel}:live_created")
                continue
            saved = _contained(packet_dir / "original", rel)
            if not live.exists() or not saved.exists():
                failures.append(rel)
                continue
            if hashlib.sha256(saved.read_bytes()).hexdigest() != expected:
                failures.append(f"{rel}:saved_hash")
            if hashlib.sha256(live.read_bytes()).hexdigest() != expected:
                failures.append(f"{rel}:live_changed")
        if failures:
            raise RollbackError(f"rollback dry-run failed: {failures[:5]}")
        verified = RollbackPacket(
            run_id=packet.run_id,
            timestamp=packet.timestamp,
            repo_root_hash=packet.repo_root_hash,
            changed_files=packet.changed_files,
            original_hashes=packet.original_hashes,
            candidate_hashes=packet.candidate_hashes,
            packet_path=packet.packet_path,
            receipt_hash=packet.receipt_hash,
            dry_run_passed=True,
            post_restore_verified=packet.post_restore_verified,
        )
        atomic_write_text(Path(packet.packet_path) / "packet.json", json.dumps(packet_to_dict(verified), indent=2, sort_keys=True, default=str))
        return verified

    def restore(
        self,
        run_id: str | RollbackPacket,
        *,
        require_candidate_generation: bool = True,
    ) -> RollbackPacket:
        """Revert the promoted generation, atomically, against what it promoted.

        CP126 2bcfa2c3: restore rewrote originals without requiring the live
        files to still BE the promoted candidates, so a user edit or a later
        promotion was silently replaced. Pass
        ``require_candidate_generation=False`` only for a deliberate
        force-revert.

        CP126 23fb4004: originals were written sequentially and verified only
        afterwards, so a missing artifact or a write failure after earlier
        replacements left a MIXED generation with no compensating recovery.
        Everything is staged and checked first; a failure mid-write restores
        what was already replaced.
        """
        packet = run_id if isinstance(run_id, RollbackPacket) else self.load_packet(run_id)
        packet_dir = self._verified_packet_dir(packet)

        # 1. STAGE: resolve, read and verify every original before touching
        #    the live tree.
        staged: list[tuple[str, Path, bytes | None]] = []
        drift: list[str] = []
        for rel, expected in packet.original_hashes.items():
            dest = _contained(self.config.repo_root, rel)
            if require_candidate_generation:
                promoted = str(packet.candidate_hashes.get(rel) or "")
                live_hash = (
                    hashlib.sha256(dest.read_bytes()).hexdigest()
                    if dest.is_file()
                    else ABSENT_SENTINEL
                )
                if promoted and live_hash != promoted:
                    # The live file is not what this packet promoted.
                    drift.append(f"{rel}:live_is_not_the_promoted_generation")
                    continue
            if expected == ABSENT_SENTINEL:
                staged.append((rel, dest, None))
                continue
            src = _contained(packet_dir / "original", rel)
            if not src.is_file():
                raise RollbackError(f"rollback original missing: {rel}")
            data = src.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RollbackError(f"rollback original hash mismatch: {rel}")
            staged.append((rel, dest, data))
        if drift:
            raise RollbackError(
                "refusing to overwrite intervening work: " + ", ".join(drift[:5])
            )

        # 2. APPLY: with compensating recovery for a partial failure.
        undo: list[tuple[Path, bytes | None]] = []
        try:
            for _rel, dest, data in staged:
                previous = dest.read_bytes() if dest.is_file() else None
                undo.append((dest, previous))
                if data is None:
                    if dest.exists():
                        os.unlink(dest)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(dest, data)
        except (OSError, RollbackError) as exc:
            for dest, previous in reversed(undo):
                try:
                    if previous is None:
                        if dest.exists():
                            os.unlink(dest)
                    else:
                        atomic_write_bytes(dest, previous)
                except OSError:
                    raise RollbackError(
                        f"restore failed ({exc}) AND compensating recovery of {dest} "
                        "failed — the tree is a MIXED generation"
                    ) from exc
            raise RollbackError(f"restore rolled back after failure: {exc}") from exc

        # 3. VERIFY.
        failures: list[str] = []
        for rel, expected in packet.original_hashes.items():
            live = _contained(self.config.repo_root, rel)
            if expected == ABSENT_SENTINEL:
                if live.exists():
                    failures.append(f"{rel}:still_present")
                continue
            if not live.is_file() or hashlib.sha256(live.read_bytes()).hexdigest() != expected:
                failures.append(rel)
        if failures:
            raise RollbackError(f"post-restore verification failed: {failures[:5]}")
        restored = RollbackPacket(
            run_id=packet.run_id,
            timestamp=packet.timestamp,
            repo_root_hash=packet.repo_root_hash,
            changed_files=packet.changed_files,
            original_hashes=packet.original_hashes,
            candidate_hashes=packet.candidate_hashes,
            packet_path=packet.packet_path,
            receipt_hash=packet.receipt_hash,
            dry_run_passed=packet.dry_run_passed,
            post_restore_verified=True,
        )
        atomic_write_text(packet_dir / "packet.json", json.dumps(packet_to_dict(restored), indent=2, sort_keys=True, default=str))
        return restored

    def _verified_packet_dir(self, packet: RollbackPacket) -> Path:
        """The packet's own directory, and proof the packet is what it claims.

        CP126 f4bc655d: load_packet trusted receipt_hash, dry_run_passed, the
        hashes, the changed-file list and packet_path straight out of mutable
        JSON, and restore never recomputed the receipt before acting.
        """
        expected = compute_receipt_hash(
            packet.run_id,
            packet.repo_root_hash,
            packet.original_hashes,
            packet.candidate_hashes,
        )
        if str(packet.receipt_hash or "") != expected:
            raise RollbackError(
                "rollback packet receipt does not match its contents "
                f"(expected {expected[:12]}, found {str(packet.receipt_hash or '')[:12]})"
            )
        # packet_path is attacker-controlled in a loaded packet: derive the
        # directory from the run id under OUR artifact root instead.
        packet_dir = _contained(self.root, packet.run_id)
        declared = Path(str(packet.packet_path or "")).resolve(strict=False)
        if declared != packet_dir.resolve(strict=False):
            raise RollbackError(
                f"rollback packet_path does not match its run id: {packet.packet_path}"
            )
        return packet_dir

    def load_packet(self, run_id: str) -> RollbackPacket:
        """Load a packet by run id, re-verifying its identity before returning."""
        packet_dir = _contained(self.root, str(run_id))
        payload = json.loads((packet_dir / "packet.json").read_text(encoding="utf-8"))
        packet = packet_from_dict(payload)
        # Recompute identity here too, so no caller can act on an unverified
        # packet just because it did not go through restore().
        self._verified_packet_dir(packet)
        return packet

    def _repo_hash(self, changed: tuple[str, ...]) -> str:
        hashes = []
        for rel in sorted(changed):
            path = self.config.repo_root / rel
            hashes.append(f"{rel}:{hashlib.sha256(path.read_bytes()).hexdigest()}")
        return hashlib.sha256("|".join(hashes).encode("utf-8")).hexdigest()


def packet_to_dict(packet: RollbackPacket) -> dict[str, object]:
    return {
        "run_id": packet.run_id,
        "timestamp": packet.timestamp,
        "repo_root_hash": packet.repo_root_hash,
        "changed_files": list(packet.changed_files),
        "original_hashes": packet.original_hashes,
        "candidate_hashes": packet.candidate_hashes,
        "packet_path": packet.packet_path,
        "receipt_hash": packet.receipt_hash,
        "dry_run_passed": packet.dry_run_passed,
        "post_restore_verified": packet.post_restore_verified,
    }


def packet_from_dict(payload: dict[str, object]) -> RollbackPacket:
    return RollbackPacket(
        run_id=str(payload["run_id"]),
        timestamp=float(payload["timestamp"]),
        repo_root_hash=str(payload["repo_root_hash"]),
        changed_files=tuple(str(item) for item in payload.get("changed_files", ())),
        original_hashes={str(key): str(value) for key, value in dict(payload.get("original_hashes", {})).items()},
        candidate_hashes={str(key): str(value) for key, value in dict(payload.get("candidate_hashes", {})).items()},
        packet_path=str(payload["packet_path"]),
        receipt_hash=str(payload.get("receipt_hash", "")),
        dry_run_passed=bool(payload.get("dry_run_passed", False)),
        post_restore_verified=bool(payload.get("post_restore_verified", False)),
    )
