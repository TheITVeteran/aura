"""Canonical persistence owner for latent-cortex runtime artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.governance_context import local_internal_governed_scope
from core.runtime.atomic_writer import ensure_private_directory, interprocess_file_lock
from core.runtime.file_write_gateway import (
    DirectoryFileWriteBatchEntry,
    FileWriteBatchEntry,
    FileWriteBatchReceipt,
    get_file_write_gateway,
)


class StaleScheduleLibraryError(RuntimeError):
    """A schedule writer attempted to commit from an obsolete revision."""


class LatentCortexPersistence:
    """Publish latent artifacts through one governed transactional boundary."""

    @staticmethod
    def _commit(entries: tuple[FileWriteBatchEntry, ...], *, source: str) -> FileWriteBatchReceipt:
        with local_internal_governed_scope(source):
            receipt = get_file_write_gateway().write_bytes_batch(
                entries,
                source=f"latent_cortex.persistence.{source}",
            )
        expected: dict[str, str] = {}
        for entry in entries:
            target = Path(entry.path).expanduser()
            normalized = target.parent.resolve() / target.name
            expected[str(normalized)] = hashlib.sha256(bytes(entry.payload)).hexdigest()
        if (
            not receipt.transaction_id
            or receipt.paths != tuple(expected)
            or dict(receipt.sha256) != expected
        ):
            raise RuntimeError(f"{source} batch receipt does not match committed payloads")
        return receipt

    def publish_fast_weight_candidate(
        self,
        target_dir: Path,
        *,
        delta_payload: bytes,
        evidence_payload: bytes,
    ) -> FileWriteBatchReceipt:
        return self._commit(
            (
                FileWriteBatchEntry(target_dir / "delta_weights.npz", delta_payload),
                FileWriteBatchEntry(target_dir / "evidence.json", evidence_payload),
            ),
            source="latent_cortex_consolidation",
        )

    def publish_verified_trajectory_artifact(
        self,
        target_dir: Path,
        *,
        factors_payload: bytes,
        manifest_payload: bytes,
    ) -> FileWriteBatchReceipt:
        """Atomically publish one exact trajectory package generation."""

        return self._publish_directory_artifact(
            target_dir,
            (
                DirectoryFileWriteBatchEntry("factors.npz", factors_payload),
                DirectoryFileWriteBatchEntry("manifest.json", manifest_payload),
            ),
            commit_marker="manifest.json",
            source="latent_cortex_trajectory_artifact",
        )

    @staticmethod
    def _publish_directory_artifact(
        target_dir: Path,
        entries: tuple[DirectoryFileWriteBatchEntry, ...],
        *,
        commit_marker: str,
        source: str,
    ) -> FileWriteBatchReceipt:
        directory = ensure_private_directory(target_dir).resolve()
        with local_internal_governed_scope(source):
            receipt = get_file_write_gateway().write_bytes_batch_in_directory(
                target_dir,
                entries,
                allowed_existing_names={entry.name for entry in entries},
                commit_marker=commit_marker,
                source=source,
            )
        expected = {
            str(directory / entry.name): hashlib.sha256(bytes(entry.payload)).hexdigest()
            for entry in entries
        }
        if (
            not receipt.transaction_id
            or receipt.paths != tuple(expected)
            or dict(receipt.sha256) != expected
        ):
            raise RuntimeError(
                f"{source} directory receipt does not match committed payloads"
            )
        return receipt

    def publish_neural_tissue_artifact(
        self,
        target_dir: Path,
        *,
        weights_payload: bytes,
        manifest_payload: bytes,
    ) -> FileWriteBatchReceipt:
        """Publish weights and their manifest as one rollback-safe generation."""

        return self._publish_directory_artifact(
            target_dir,
            (
                DirectoryFileWriteBatchEntry(
                    "weights.safetensors",
                    weights_payload,
                ),
                DirectoryFileWriteBatchEntry("manifest.json", manifest_payload),
            ),
            commit_marker="manifest.json",
            source="latent_cortex_neural_tissue_artifact",
        )

    def save_schedule_library(
        self,
        path: Path,
        payload: bytes,
        *,
        expected_revision: int,
    ) -> FileWriteBatchReceipt:
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected schedule revision must be a non-negative integer")
        try:
            proposed = json.loads(bytes(payload).decode("utf-8"))
        except (TypeError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError("schedule library payload must be UTF-8 JSON") from exc
        from core.brain.llm.latent_cortex.schedules import ScheduleLibrary

        proposed_revision, _proposed_records = ScheduleLibrary._parse_store(proposed)
        if proposed_revision != expected_revision + 1:
            raise ValueError("schedule library payload revision is not the next CAS revision")

        requested = Path(path).expanduser()
        target = requested.parent.resolve(strict=False) / requested.name
        lock_path = target.parent / ".aura_file_write_batch.lock"
        with interprocess_file_lock(lock_path):
            current_revision = 0
            if target.exists():
                try:
                    current = json.loads(target.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, ValueError) as exc:
                    raise ValueError(
                        "refusing to overwrite an unreadable schedule library"
                    ) from exc
                try:
                    current_revision, _current_records = ScheduleLibrary._parse_store(current)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "refusing to overwrite a schedule library with an invalid schema"
                    ) from exc
            if current_revision != expected_revision:
                raise StaleScheduleLibraryError(
                    f"schedule library revision changed from {expected_revision} "
                    f"to {current_revision}"
                )
            return self._commit(
                (FileWriteBatchEntry(target, payload),),
                source="latent_cortex_schedule_library",
            )

    def save_lab_report(
        self,
        path: Path,
        payload: bytes,
    ) -> FileWriteBatchReceipt:
        return self._commit(
            (FileWriteBatchEntry(path, payload),),
            source="latent_cortex_lab_report",
        )

    def save_action_intervention_replay_ledger(
        self,
        path: Path,
        payload: bytes,
    ) -> FileWriteBatchReceipt:
        """Atomically replace the bounded, externally locked replay ledger."""

        return self._commit(
            (FileWriteBatchEntry(path, payload),),
            source="latent_cortex_action_intervention_replay",
        )

    def save_verified_replay_buffer(
        self,
        path: Path,
        payload: bytes,
    ) -> FileWriteBatchReceipt:
        """Atomically replace the encrypted verified-repair ledger."""

        return self._commit(
            (FileWriteBatchEntry(path, payload),),
            source="latent_cortex_verified_replay",
        )

    def save_frontier_verification(
        self,
        path: Path,
        payload: bytes,
    ) -> FileWriteBatchReceipt:
        """Publish a standalone verification certificate or signing request."""

        return self._commit(
            (FileWriteBatchEntry(path, payload),),
            source="latent_cortex_frontier_verification",
        )


_PERSISTENCE = LatentCortexPersistence()


def get_latent_cortex_persistence() -> LatentCortexPersistence:
    return _PERSISTENCE


__all__ = [
    "LatentCortexPersistence",
    "StaleScheduleLibraryError",
    "get_latent_cortex_persistence",
]
