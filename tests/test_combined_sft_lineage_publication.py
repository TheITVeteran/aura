from __future__ import annotations

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from test_combined_sft_lineage import _build

from core.learning.combined_sft_lineage_publication import (
    CombinedSFTLineagePublicationError,
    publish_combined_sft_lineage_custody,
    read_combined_sft_lineage_publication,
)
from core.runtime.file_write_gateway import FileWriteTransactionError

pytest_plugins = ("test_combined_sft_lineage",)


def test_publication_is_private_committed_and_replayable(
    tmp_path: Path,
    lineage_inputs,
) -> None:
    bundle = _build(lineage_inputs)
    root = tmp_path / "combined-publication"
    report = publish_combined_sft_lineage_custody(
        bundle=bundle,
        publication_root=root,
    )
    restored = read_combined_sft_lineage_publication(
        root / "candidate",
        evaluator_directory=root / "evaluator",
    )

    assert report["status"] == "combined_lineage_custody_published"
    assert restored["custody_report"] == bundle.custody_report
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) & 0o077 == 0
        for directory in (root, root / "candidate", root / "evaluator")
        for path in directory.iterdir()
    )


def test_identical_publication_is_idempotent(
    tmp_path: Path,
    lineage_inputs,
) -> None:
    bundle = _build(lineage_inputs)
    root = tmp_path / "combined-publication"
    first = publish_combined_sft_lineage_custody(
        bundle=bundle,
        publication_root=root,
    )
    second = publish_combined_sft_lineage_custody(
        bundle=bundle,
        publication_root=root,
    )
    assert second["status"] == "existing_committed_generation_revalidated"
    assert second["generation_id"] == first["generation_id"]


def test_concurrent_identical_publishers_converge(
    tmp_path: Path,
    lineage_inputs,
) -> None:
    bundle = _build(lineage_inputs)
    root = tmp_path / "combined-publication"
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(
            lambda _ordinal: publish_combined_sft_lineage_custody(
                bundle=bundle,
                publication_root=root,
            ),
            range(2),
        )
    assert {first["status"], second["status"]} == {
        "combined_lineage_custody_published",
        "existing_committed_generation_revalidated",
    }
    assert first["generation_id"] == second["generation_id"]


def test_preparing_generation_recovers_after_interrupted_candidate_write(
    tmp_path: Path,
    lineage_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build(lineage_inputs)
    root = tmp_path / "combined-publication"
    from core.learning import combined_sft_lineage_publication as publication

    gateway = publication.get_file_write_gateway()
    real_write = gateway.write_bytes_batch_in_directory

    def fail_candidate(*args, source="unknown", **kwargs):
        if source == "combined_sft_lineage.candidate_artifacts":
            raise FileWriteTransactionError("injected candidate failure")
        return real_write(*args, source=source, **kwargs)

    monkeypatch.setattr(gateway, "write_bytes_batch_in_directory", fail_candidate)
    with pytest.raises(FileWriteTransactionError, match="injected candidate"):
        publish_combined_sft_lineage_custody(bundle=bundle, publication_root=root)
    with pytest.raises(
        CombinedSFTLineagePublicationError,
        match="not_committed",
    ):
        read_combined_sft_lineage_publication(root / "candidate")

    monkeypatch.setattr(gateway, "write_bytes_batch_in_directory", real_write)
    recovered = publish_combined_sft_lineage_custody(
        bundle=bundle,
        publication_root=root,
    )
    assert recovered["recovered_preparing_generation_id"]


def test_publication_rejects_hardlinked_lock_without_mutating_target_mode(
    tmp_path: Path,
    lineage_inputs,
) -> None:
    bundle = _build(lineage_inputs)
    root = tmp_path / "combined-publication"
    root.mkdir(mode=0o700)
    outside = tmp_path / "outside.lock"
    outside.write_bytes(b"external")
    outside.chmod(0o640)
    os.link(outside, root / ".aura_combined_sft_lineage.lock")

    with pytest.raises(
        CombinedSFTLineagePublicationError,
        match="lock_invalid",
    ):
        publish_combined_sft_lineage_custody(bundle=bundle, publication_root=root)
    assert outside.read_bytes() == b"external"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o640
