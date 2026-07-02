from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import tarfile
from collections.abc import Iterator
from itertools import repeat
from pathlib import Path

from core.runtime.update_manager import Channel, LocalFileTransport, Release, UpdateManager


def _write_source(root: Path, marker: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "aura_main.py").write_text(f"print({marker!r})\n", encoding="utf-8")
    (root / "MARKER").write_text(marker, encoding="utf-8")


def _make_archive(source: Path, archive: Path, *, arcname: str = "live-source") -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(source, arcname=arcname)


def _sign_release(manager: UpdateManager, archive: Path) -> Path:
    sig = archive.with_suffix(archive.suffix + ".sig")
    mac = hmac.new(manager._key(), digestmod=hashlib.sha256)
    with archive.open("rb") as fh:
        while chunk := fh.read(128):
            mac.update(chunk)
    sig.write_bytes(mac.digest())
    return sig


def _release(archive: Path, sig: Path | None, *, version: str = "2026.7.1") -> Release:
    return Release(
        version=version,
        channel="stable",
        archive_path=str(archive),
        signature_path=str(sig) if sig else None,
        changelog="contract test release",
        published_at=1.0,
    )


def _continuity_sequence(values: list[str]) -> Iterator[str]:
    while values:
        yield values.pop(0)
    yield from repeat("stable")


def test_local_file_transport_discovers_full_tar_gz_version_and_changelog(tmp_path: Path) -> None:
    release_dir = tmp_path / "releases"
    stable = release_dir / "stable"
    stable.mkdir(parents=True)
    archive = stable / "aura-2026.7.1.tar.gz"
    archive.write_bytes(b"not-used-by-discovery")
    sig = stable / "aura-2026.7.1.tar.gz.sig"
    sig.write_bytes(b"sig")
    changelog = stable / "aura-2026.7.1.changelog.md"
    changelog.write_text("changed safely", encoding="utf-8")

    releases = asyncio.run(LocalFileTransport(release_dir).list_available(Channel.STABLE))

    assert len(releases) == 1
    assert releases[0].version == "2026.7.1"
    assert releases[0].signature_path == str(sig)
    assert releases[0].changelog == "changed safely"


def test_apply_rejects_unsigned_release_before_cutover(tmp_path: Path) -> None:
    live_target = tmp_path / "current"
    _write_source(live_target, "current")
    live_link = tmp_path / "live-source"
    live_link.symlink_to(live_target, target_is_directory=True)
    manager = UpdateManager(
        backup_dir=tmp_path / "backups",
        release_dir=tmp_path / "releases",
        live_link=live_link,
    )
    candidate = tmp_path / "candidate"
    _write_source(candidate, "candidate")
    archive = tmp_path / "releases" / "stable" / "aura-2026.7.1.tar.gz"
    _make_archive(candidate, archive)

    attempt = asyncio.run(manager.apply(_release(archive, None)))

    assert attempt.failed_reason == "signature_missing"
    assert live_link.resolve() == live_target.resolve()
    assert not attempt.staged_at


def test_symlink_cutover_rolls_back_on_continuity_drift(monkeypatch, tmp_path: Path) -> None:
    live_target = tmp_path / "current"
    _write_source(live_target, "current")
    live_link = tmp_path / "live-source"
    live_link.symlink_to(live_target, target_is_directory=True)
    manager = UpdateManager(
        backup_dir=tmp_path / "backups",
        release_dir=tmp_path / "releases",
        live_link=live_link,
    )
    candidate = tmp_path / "candidate"
    _write_source(candidate, "candidate")
    archive = tmp_path / "releases" / "stable" / "aura-2026.7.1.tar.gz"
    _make_archive(candidate, archive)
    sig = _sign_release(manager, archive)
    hashes = _continuity_sequence(["before", "after"])
    monkeypatch.setattr(UpdateManager, "_continuity_hash", staticmethod(lambda: next(hashes)))

    attempt = asyncio.run(manager.apply(_release(archive, sig)))

    assert attempt.failed_reason == "continuity_drift"
    assert live_link.is_symlink()
    assert live_link.resolve() == live_target.resolve()
    assert (live_link / "MARKER").read_text(encoding="utf-8") == "current"
    assert "rolled_back" in (tmp_path / "backups" / "updates.jsonl").read_text(encoding="utf-8")


def test_directory_cutover_rolls_back_on_continuity_drift(monkeypatch, tmp_path: Path) -> None:
    live_link = tmp_path / "live-source"
    _write_source(live_link, "current")
    manager = UpdateManager(
        backup_dir=tmp_path / "backups",
        release_dir=tmp_path / "releases",
        live_link=live_link,
    )
    candidate = tmp_path / "candidate"
    _write_source(candidate, "candidate")
    archive = tmp_path / "releases" / "stable" / "aura-2026.7.1.tar.gz"
    _make_archive(candidate, archive)
    sig = _sign_release(manager, archive)
    hashes = _continuity_sequence(["before", "after"])
    monkeypatch.setattr(UpdateManager, "_continuity_hash", staticmethod(lambda: next(hashes)))

    attempt = asyncio.run(manager.apply(_release(archive, sig)))

    assert attempt.failed_reason == "continuity_drift"
    assert not live_link.is_symlink()
    assert (live_link / "MARKER").read_text(encoding="utf-8") == "current"
    assert not Path(attempt.moved_aside_to or "").exists()


def test_signed_symlink_release_completes_with_candidate_root(monkeypatch, tmp_path: Path) -> None:
    live_target = tmp_path / "current"
    _write_source(live_target, "current")
    live_link = tmp_path / "live-source"
    live_link.symlink_to(live_target, target_is_directory=True)
    manager = UpdateManager(
        backup_dir=tmp_path / "backups",
        release_dir=tmp_path / "releases",
        live_link=live_link,
    )
    candidate = tmp_path / "candidate-root"
    _write_source(candidate, "candidate")
    archive = tmp_path / "releases" / "stable" / "aura-2026.7.1.tar.gz"
    _make_archive(candidate, archive, arcname="aura-candidate")
    sig = _sign_release(manager, archive)
    monkeypatch.setattr(UpdateManager, "_continuity_hash", staticmethod(lambda: "same"))

    attempt = asyncio.run(manager.apply(_release(archive, sig)))

    assert attempt.failed_reason is None
    assert attempt.completed_at is not None
    assert live_link.is_symlink()
    assert (live_link / "MARKER").read_text(encoding="utf-8") == "candidate"
    assert Path(attempt.candidate_root or "").name == "aura-candidate"
    assert "completed" in (tmp_path / "backups" / "updates.jsonl").read_text(encoding="utf-8")



def test_signing_key_stays_consistent_when_persistence_fails(monkeypatch, tmp_path: Path) -> None:
    """A rejected key-file write must not make signer and verifier disagree.

    Regression: _key() minted a fresh key on every call when the file-write
    gateway refused the persist, so every release verified as
    signature_invalid (observed as chunk-order flakiness; in a strict-runtime
    deployment it would have broken updates entirely).
    """
    import core.runtime.update_manager as um

    class _RefusingGateway:
        refused = 0

        def write_bytes(self, *args, **kwargs):
            _RefusingGateway.refused += 1
            raise RuntimeError("file-write gateway is in a rejecting mode")

    monkeypatch.setattr(um, "get_file_write_gateway", lambda: _RefusingGateway())

    manager = UpdateManager(
        backup_dir=tmp_path / "backups",
        release_dir=tmp_path / "releases",
        live_link=tmp_path / "live-source",
    )

    first = manager._key()
    second = manager._key()

    assert first == second, "one manager must sign and verify with the same key"
    assert not (tmp_path / "backups" / "update_key").exists()

    candidate = tmp_path / "candidate"
    _write_source(candidate, "candidate")
    archive = tmp_path / "releases" / "stable" / "aura-2026.7.2.tar.gz"
    _make_archive(candidate, archive)
    sig = _sign_release(manager, archive)

    assert manager._verify_signature(archive, sig) is True
