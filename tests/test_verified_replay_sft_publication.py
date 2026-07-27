from __future__ import annotations

import hashlib
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from test_rlc_verified_replay_buffer import _Protector
from test_verified_replay_sft import _clearance, _variant

from core.brain.latent_cortex_service import LatentCortexService
from core.brain.llm.latent_cortex.verified_replay_buffer import VerifiedReplayBuffer
from core.learning import verified_replay_sft_publication as publication
from core.learning.verified_replay_sft import (
    VERIFIED_REPLAY_SFT_CANDIDATE_FILES,
    VERIFIED_REPLAY_SFT_EVALUATOR_FILES,
    empty_reference_index,
)
from core.memory.black_hole import BlackHole
from core.memory.horcrux import HorcruxManager
from core.runtime.file_write_gateway import FileWriteTransactionError

PARTITION_KEY = b"partition-fixture-key" * 2
DEDUP_KEY = b"dedup-fixture-key" * 2
KEY_IDENTITY = hashlib.sha256(b"fixture-horcrux-identity").hexdigest()


class _HorcruxProtector(_Protector):
    key_provenance = "horcrux"
    key_identity_sha256 = KEY_IDENTITY


class _Horcrux:
    key_identity_sha256 = KEY_IDENTITY

    @staticmethod
    def derive_subkey(context: str) -> bytes:
        if context == publication._PARTITION_CONTEXT:
            return PARTITION_KEY[:32]
        if context == publication._DEDUP_CONTEXT:
            return DEDUP_KEY[:32]
        raise AssertionError(context)


def _source(tmp_path: Path, *, count: int = 20):
    protector = _HorcruxProtector()
    buffer = VerifiedReplayBuffer(
        tmp_path / "replay.json",
        max_entries=max(64, count + 8),
    )
    payloads = [_variant(index) for index in range(count)]
    for index, payload in enumerate(payloads):
        buffer.append(
            payload,
            protector=protector,
            created_at_unix_ns=100_000 + index,
        )
    store = buffer.load()
    clearances = {
        entry["entry_sha256"]: _clearance(entry, payload)
        for entry, payload in zip(store["entries"], payloads, strict=True)
    }
    return protector, buffer, store, clearances, payloads


def _publish(tmp_path: Path):
    protector, buffer, store, clearances, payloads = _source(tmp_path)
    root = tmp_path / "projection"
    report = publication.publish_verified_replay_sft_custody(
        replay_store=store,
        protector=protector,
        privacy_clearances=clearances,
        reference_index=empty_reference_index(dedup_key=DEDUP_KEY),
        partition_key=PARTITION_KEY,
        dedup_key=DEDUP_KEY,
        key_identity_sha256=KEY_IDENTITY,
        candidate_directory=root / "candidate",
        evaluator_directory=root / "evaluator",
    )
    return protector, buffer, store, clearances, payloads, root, report


def test_publication_is_private_separate_committed_and_replayable(
    tmp_path: Path,
) -> None:
    _protector, _buffer, _store, _clearances, _payloads, root, report = (
        _publish(tmp_path)
    )
    candidate = root / "candidate"
    evaluator = root / "evaluator"

    assert report["status"] == "custody_bundles_published_and_replay_validated"
    assert report["trainer_ready"] is False
    assert report["training_authority"] == "none_quarantined_projection"
    assert report["protector_key_provenance"] == "horcrux"
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert {
        item.name for item in candidate.iterdir()
    } == {*VERIFIED_REPLAY_SFT_CANDIDATE_FILES, ".aura_file_write_batch.lock"}
    assert {
        item.name for item in evaluator.iterdir()
    } == {*VERIFIED_REPLAY_SFT_EVALUATOR_FILES, ".aura_file_write_batch.lock"}
    assert all(
        stat.S_IMODE((directory / name).stat().st_mode) == 0o600
        for directory, names in (
            (candidate, VERIFIED_REPLAY_SFT_CANDIDATE_FILES),
            (evaluator, VERIFIED_REPLAY_SFT_EVALUATOR_FILES),
        )
        for name in names
    )
    candidate_bytes, attestation = (
        publication.read_candidate_publication_with_attestation(candidate)
    )
    pair = publication.validate_publication_directories(
        candidate_directory=candidate,
        evaluator_directory=evaluator,
    )
    assert set(candidate_bytes) == set(VERIFIED_REPLAY_SFT_CANDIDATE_FILES)
    assert attestation["state"] == "committed"
    assert pair["candidate_manifest"]["candidate_package_sha256"] == report[
        "candidate_package_sha256"
    ]
    assert b'"split":"holdout"' not in b"".join(candidate_bytes.values())


def test_identical_publication_is_idempotent_and_revalidates_disk(
    tmp_path: Path,
) -> None:
    protector, _buffer, store, clearances, _payloads, root, first = _publish(
        tmp_path
    )
    second = publication.publish_verified_replay_sft_custody(
        replay_store=store,
        protector=protector,
        privacy_clearances=clearances,
        reference_index=empty_reference_index(dedup_key=DEDUP_KEY),
        partition_key=PARTITION_KEY,
        dedup_key=DEDUP_KEY,
        key_identity_sha256=KEY_IDENTITY,
        candidate_directory=root / "candidate",
        evaluator_directory=root / "evaluator",
    )

    assert second["status"] == "existing_committed_generation_replay_validated"
    assert second["generation_id"] == first["generation_id"]
    assert second["publication_commit_sha256"] == first[
        "publication_commit_sha256"
    ]


def test_concurrent_identical_publishers_converge_on_one_generation(
    tmp_path: Path,
) -> None:
    protector, _buffer, store, clearances, _payloads = _source(tmp_path)
    root = tmp_path / "projection"

    def publish() -> dict:
        return publication.publish_verified_replay_sft_custody(
            replay_store=store,
            protector=protector,
            privacy_clearances=clearances,
            reference_index=empty_reference_index(dedup_key=DEDUP_KEY),
            partition_key=PARTITION_KEY,
            dedup_key=DEDUP_KEY,
            key_identity_sha256=KEY_IDENTITY,
            candidate_directory=root / "candidate",
            evaluator_directory=root / "evaluator",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(lambda _ordinal: publish(), range(2))

    assert {first["status"], second["status"]} == {
        "custody_bundles_published_and_replay_validated",
        "existing_committed_generation_replay_validated",
    }
    assert first["generation_id"] == second["generation_id"]
    assert first["publication_commit_sha256"] == second[
        "publication_commit_sha256"
    ]


def test_candidate_only_read_does_not_require_evaluator_custody_access(
    tmp_path: Path,
) -> None:
    _protector, _buffer, _store, _clearances, _payloads, root, report = (
        _publish(tmp_path)
    )
    shutil.rmtree(root / "evaluator")

    artifacts, commit = publication.read_candidate_publication_with_attestation(
        root / "candidate"
    )

    assert set(artifacts) == set(VERIFIED_REPLAY_SFT_CANDIDATE_FILES)
    assert commit["candidate_package_sha256"] == report[
        "candidate_package_sha256"
    ]


def test_candidate_reader_rejects_any_extra_file_even_when_manifest_is_valid(
    tmp_path: Path,
) -> None:
    _protector, _buffer, _store, _clearances, _payloads, root, _report = (
        _publish(tmp_path)
    )
    injected = root / "candidate" / "verified_replay_holdout.json"
    injected.write_text('{"leak":true}')
    injected.chmod(0o600)

    with pytest.raises(
        publication.VerifiedReplaySFTPublicationError,
        match="artifact_inventory_invalid",
    ):
        publication.read_candidate_publication(root / "candidate")


def test_preparing_generation_blocks_candidate_and_is_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protector, _buffer, store, clearances, _payloads = _source(tmp_path)
    root = tmp_path / "projection"
    gateway = publication.get_file_write_gateway()
    real_write = gateway.write_bytes_batch_in_directory

    def fail_candidate(*args, source="unknown", **kwargs):
        if source == "verified_replay_sft.candidate_artifacts":
            raise FileWriteTransactionError("injected candidate failure")
        return real_write(*args, source=source, **kwargs)

    monkeypatch.setattr(gateway, "write_bytes_batch_in_directory", fail_candidate)
    with pytest.raises(FileWriteTransactionError, match="injected candidate"):
        publication.publish_verified_replay_sft_custody(
            replay_store=store,
            protector=protector,
            privacy_clearances=clearances,
            reference_index=empty_reference_index(dedup_key=DEDUP_KEY),
            partition_key=PARTITION_KEY,
            dedup_key=DEDUP_KEY,
            key_identity_sha256=KEY_IDENTITY,
            candidate_directory=root / "candidate",
            evaluator_directory=root / "evaluator",
        )
    preparing = publication._read_commit(root / publication._COMMIT_FILE)
    assert preparing["state"] == "preparing"
    with pytest.raises(
        publication.VerifiedReplaySFTPublicationError,
        match="candidate_publication_not_committed",
    ):
        publication.read_candidate_publication(root / "candidate")

    monkeypatch.setattr(gateway, "write_bytes_batch_in_directory", real_write)
    recovered = publication.publish_verified_replay_sft_custody(
        replay_store=store,
        protector=protector,
        privacy_clearances=clearances,
        reference_index=empty_reference_index(dedup_key=DEDUP_KEY),
        partition_key=PARTITION_KEY,
        dedup_key=DEDUP_KEY,
        key_identity_sha256=KEY_IDENTITY,
        candidate_directory=root / "candidate",
        evaluator_directory=root / "evaluator",
    )
    assert recovered["recovered_preparing_generation"] is True
    assert recovered["recovered_generation_id"] == preparing["generation_id"]
    publication.validate_publication_directories(
        candidate_directory=root / "candidate",
        evaluator_directory=root / "evaluator",
    )


def test_valid_committed_generation_can_be_superseded_by_new_store_revision(
    tmp_path: Path,
) -> None:
    protector, buffer, _store, clearances, payloads, root, first = _publish(
        tmp_path
    )
    for index in range(20, 24):
        payload = _variant(index)
        payloads.append(payload)
        buffer.append(
            payload,
            protector=protector,
            created_at_unix_ns=100_000 + index,
        )
    updated = buffer.load()
    clearances = {
        entry["entry_sha256"]: _clearance(entry, payload)
        for entry, payload in zip(updated["entries"], payloads, strict=True)
    }
    second = publication.publish_verified_replay_sft_custody(
        replay_store=updated,
        protector=protector,
        privacy_clearances=clearances,
        reference_index=empty_reference_index(dedup_key=DEDUP_KEY),
        partition_key=PARTITION_KEY,
        dedup_key=DEDUP_KEY,
        key_identity_sha256=KEY_IDENTITY,
        candidate_directory=root / "candidate",
        evaluator_directory=root / "evaluator",
    )

    assert second["superseded_committed_generation"] is True
    assert second["superseded_generation_id"] == first["generation_id"]
    assert second["generation_id"] != first["generation_id"]


def test_tampered_committed_generation_is_not_silently_overwritten(
    tmp_path: Path,
) -> None:
    protector, buffer, _store, clearances, payloads, root, _report = _publish(
        tmp_path
    )
    target = root / "candidate" / "verified_replay_train.jsonl"
    target.write_bytes(target.read_bytes() + b"{}\n")
    payload = _variant(99)
    payloads.append(payload)
    buffer.append(payload, protector=protector, created_at_unix_ns=999_999)
    updated = buffer.load()
    clearances = {
        entry["entry_sha256"]: _clearance(entry, item)
        for entry, item in zip(updated["entries"], payloads, strict=True)
    }

    with pytest.raises(ValueError, match="verified_replay_sft_"):
        publication.publish_verified_replay_sft_custody(
            replay_store=updated,
            protector=protector,
            privacy_clearances=clearances,
            reference_index=empty_reference_index(dedup_key=DEDUP_KEY),
            partition_key=PARTITION_KEY,
            dedup_key=DEDUP_KEY,
            key_identity_sha256=KEY_IDENTITY,
            candidate_directory=root / "candidate",
            evaluator_directory=root / "evaluator",
        )
    assert target.read_bytes().endswith(b"{}\n")


def test_publication_rejects_non_horcrux_or_mismatched_key_identity(
    tmp_path: Path,
) -> None:
    _protector, _buffer, store, clearances, _payloads = _source(tmp_path)
    root = tmp_path / "projection"
    local = _Protector()
    with pytest.raises(
        publication.VerifiedReplaySFTPublicationError,
        match="requires_horcrux",
    ):
        publication.publish_verified_replay_sft_custody(
            replay_store=store,
            protector=local,
            privacy_clearances=clearances,
            reference_index=empty_reference_index(dedup_key=DEDUP_KEY),
            partition_key=PARTITION_KEY,
            dedup_key=DEDUP_KEY,
            key_identity_sha256=KEY_IDENTITY,
            candidate_directory=root / "candidate",
            evaluator_directory=root / "evaluator",
        )

    wrong = _HorcruxProtector()
    wrong.key_identity_sha256 = "f" * 64
    with pytest.raises(
        publication.VerifiedReplaySFTPublicationError,
        match="identity_mismatch",
    ):
        publication.publish_verified_replay_sft_custody(
            replay_store=store,
            protector=wrong,
            privacy_clearances=clearances,
            reference_index=empty_reference_index(dedup_key=DEDUP_KEY),
            partition_key=PARTITION_KEY,
            dedup_key=DEDUP_KEY,
            key_identity_sha256=KEY_IDENTITY,
            candidate_directory=root / "candidate",
            evaluator_directory=root / "evaluator",
        )


def test_runtime_keys_require_matching_horcrux_backed_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protector = _HorcruxProtector()
    services = {"black_hole": protector, "horcrux": _Horcrux()}
    monkeypatch.setattr(
        publication,
        "get_runtime_service",
        lambda name, default=None: services.get(name, default),
    )

    keys = publication.require_runtime_projection_keys()
    assert keys.protector is protector
    assert keys.partition_key == PARTITION_KEY[:32]
    assert keys.dedup_key == DEDUP_KEY[:32]

    services["horcrux"].key_identity_sha256 = "e" * 64
    with pytest.raises(
        publication.VerifiedReplaySFTPublicationError,
        match="identity_mismatch",
    ):
        publication.require_runtime_projection_keys()


def test_real_horcrux_black_hole_runtime_publication_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets: dict[str, str] = {}
    horcrux = HorcruxManager(
        base_dir=str(tmp_path / "horcrux"),
        secret_getter=secrets.get,
        secret_setter=secrets.__setitem__,
    )
    horcrux.derived_key = hashlib.sha256(b"resident-fixture-root").digest()
    monkeypatch.setattr(
        "core.memory.black_hole.get_runtime_service",
        lambda name, default=None: horcrux if name == "horcrux" else default,
    )
    black_hole = BlackHole()
    black_hole.on_start()
    services = {"black_hole": black_hole, "horcrux": horcrux}
    monkeypatch.setattr(
        publication,
        "get_runtime_service",
        lambda name, default=None: services.get(name, default),
    )
    replay_path = tmp_path / "runtime" / "verified-replay.json"
    buffer = VerifiedReplayBuffer(replay_path, max_entries=64)
    payloads = [_variant(index + 200) for index in range(20)]
    for index, payload in enumerate(payloads):
        buffer.append(
            payload,
            protector=black_hole,
            created_at_unix_ns=200_000 + index,
        )
    store = buffer.load()
    clearances = {
        entry["entry_sha256"]: _clearance(entry, payload)
        for entry, payload in zip(store["entries"], payloads, strict=True)
    }
    dedup_key = horcrux.derive_subkey(publication._DEDUP_CONTEXT)

    report = publication.publish_runtime_verified_replay_sft(
        privacy_clearances=clearances,
        reference_index=empty_reference_index(dedup_key=dedup_key),
        publication_root=tmp_path / "runtime-publication",
        replay_path=replay_path,
    )

    assert report["status"] == "custody_bundles_published_and_replay_validated"
    assert report["protector_key_provenance"] == "horcrux"
    assert report["protector_key_identity_sha256"] == (
        horcrux.key_identity_sha256
    )
    assert report["trainer_ready"] is False
    assert report["training_authority"] == "none_quarantined_projection"


def test_commit_parser_rejects_duplicate_keys(tmp_path: Path) -> None:
    _protector, _buffer, _store, _clearances, _payloads, root, _report = (
        _publish(tmp_path)
    )
    commit = root / publication._COMMIT_FILE
    raw = commit.read_text()
    commit.write_text(raw[:-1] + ',"state":"committed"}')

    with pytest.raises(
        publication.VerifiedReplaySFTPublicationError,
        match="duplicate_json_key:state",
    ):
        publication.read_candidate_publication(root / "candidate")


@pytest.mark.asyncio
async def test_latent_cortex_service_exposes_non_authoritative_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = {
        "status": "custody_bundles_published_and_replay_validated",
        "generation_id": "a" * 32,
        "trainer_ready": False,
        "training_authority": "none_quarantined_projection",
    }
    monkeypatch.setattr(
        publication,
        "publish_runtime_verified_replay_sft",
        lambda **_kwargs: dict(expected),
    )
    service = LatentCortexService()

    report = await service.publish_verified_replay_sft(
        privacy_clearances={},
        reference_index={},
        publication_root=tmp_path,
    )

    assert report == expected
    assert service.get_status()["last_replay_sft_publication"] == expected


def test_publication_refuses_symlinked_or_overlapping_directories(
    tmp_path: Path,
) -> None:
    protector, _buffer, store, clearances, _payloads = _source(tmp_path)
    root = tmp_path / "projection"
    root.mkdir(mode=0o700)
    real = root / "real"
    real.mkdir(mode=0o700)
    link = root / "candidate"
    os.symlink(real, link)

    with pytest.raises(
        publication.VerifiedReplaySFTPublicationError,
        match="symlink_path_rejected",
    ):
        publication.publish_verified_replay_sft_custody(
            replay_store=store,
            protector=protector,
            privacy_clearances=clearances,
            reference_index=empty_reference_index(dedup_key=DEDUP_KEY),
            partition_key=PARTITION_KEY,
            dedup_key=DEDUP_KEY,
            key_identity_sha256=KEY_IDENTITY,
            candidate_directory=link,
            evaluator_directory=root / "evaluator",
        )

    link.unlink()
    with pytest.raises(
        publication.VerifiedReplaySFTPublicationError,
        match="distinct_non_nested_siblings",
    ):
        publication.publish_verified_replay_sft_custody(
            replay_store=store,
            protector=protector,
            privacy_clearances=clearances,
            reference_index=empty_reference_index(dedup_key=DEDUP_KEY),
            partition_key=PARTITION_KEY,
            dedup_key=DEDUP_KEY,
            key_identity_sha256=KEY_IDENTITY,
            candidate_directory=root / "same",
            evaluator_directory=root / "same",
        )


def test_publication_lock_rejects_hardlink_before_changing_target_mode(
    tmp_path: Path,
) -> None:
    protector, _buffer, store, clearances, _payloads = _source(tmp_path)
    root = tmp_path / "projection"
    root.mkdir(mode=0o700)
    outside = tmp_path / "outside.lock"
    outside.write_bytes(b"do not mutate")
    outside.chmod(0o640)
    os.link(outside, root / publication._LOCK_FILE)

    with pytest.raises(
        publication.VerifiedReplaySFTPublicationError,
        match="lock_binding_invalid",
    ):
        publication.publish_verified_replay_sft_custody(
            replay_store=store,
            protector=protector,
            privacy_clearances=clearances,
            reference_index=empty_reference_index(dedup_key=DEDUP_KEY),
            partition_key=PARTITION_KEY,
            dedup_key=DEDUP_KEY,
            key_identity_sha256=KEY_IDENTITY,
            candidate_directory=root / "candidate",
            evaluator_directory=root / "evaluator",
        )

    assert outside.read_bytes() == b"do not mutate"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o640
