from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.runtime.secure_path_custody import (
    DirectoryCustody,
    SecurePathCustodyError,
    path_custody_threat_model,
    validate_path_custody_threat_model,
)


def test_path_custody_threat_model_is_self_bound_and_exact() -> None:
    threat_model = path_custody_threat_model()

    assert threat_model["security_boundary"] == "exclusive_effective_os_user"
    assert threat_model["trusted_principal"] == {"effective_uid": os.geteuid()}
    assert validate_path_custody_threat_model(threat_model) == threat_model

    changed = dict(threat_model)
    changed["excluded_adversary"] = "none"
    with pytest.raises(SecurePathCustodyError, match="threat_model_mismatch"):
        validate_path_custody_threat_model(changed)


def test_descriptor_custody_creates_and_publishes_without_following_symlinks(
    tmp_path: Path,
) -> None:
    with DirectoryCustody.acquire(tmp_path / "root", create=True, private=True) as custody:
        custody.ensure_directory("inputs/nested")
        assert custody.write_bytes_once("inputs/nested/value.bin", b"first") is True
        assert custody.write_bytes_once("inputs/nested/value.bin", b"second") is False
        assert custody.read_bytes("inputs/nested/value.bin", max_bytes=32) == b"first"
        custody.atomic_write_bytes("inputs/nested/value.bin", b"replacement")
        assert custody.read_bytes("inputs/nested/value.bin", max_bytes=32) == b"replacement"


def test_descriptor_custody_detects_pathname_exchange(tmp_path: Path) -> None:
    root = tmp_path / "root"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    with DirectoryCustody.acquire(root, create=True, private=True) as custody:
        held_identity = custody.identity
        displaced = tmp_path / "displaced"
        root.rename(displaced)
        replacement.rename(root)
        with pytest.raises(SecurePathCustodyError, match="pathname_replaced"):
            custody.verify()
        assert held_identity["st_ino"] == os.stat(displaced).st_ino


def test_descriptor_custody_refuses_symlink_component(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SecurePathCustodyError, match="directory_open_failed"):
        DirectoryCustody.acquire(root)


def test_descriptor_custody_refuses_nested_symlink_without_external_write(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    outside.mkdir()
    with DirectoryCustody.acquire(root, create=True, private=True) as custody:
        (root / "nested").symlink_to(outside, target_is_directory=True)
        with pytest.raises(SecurePathCustodyError, match="parent_open_failed"):
            custody.atomic_write_bytes("nested/value.bin", b"forbidden")
    assert list(outside.iterdir()) == []


def test_descriptor_custody_file_lock_releases_before_close(tmp_path: Path) -> None:
    custody = DirectoryCustody.acquire(tmp_path / "root", create=True, private=True)
    with custody.file_lock("locks/controller.lock") as descriptor:
        assert os.fstat(descriptor).st_size == 0
    custody.close()


def test_descriptor_custody_rejects_external_hardlink_without_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside.log"
    outside.write_bytes(b"unchanged")
    with DirectoryCustody.acquire(root, create=True, private=True) as custody:
        os.link(outside, root / "worker.log")
        with pytest.raises(SecurePathCustodyError, match="file_identity_unsafe"):
            custody.open_file("worker.log", os.O_WRONLY | os.O_APPEND)
        with pytest.raises(SecurePathCustodyError, match="file_invalid"):
            custody.read_bytes("worker.log", max_bytes=32)
    assert outside.read_bytes() == b"unchanged"


def test_descriptor_custody_rolls_back_live_descendant_exchange(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    escaped = tmp_path / "escaped"
    with DirectoryCustody.acquire(root, create=True, private=True) as custody:
        custody.ensure_directory("nested")
        original_publish = DirectoryCustody._publish_temp

        def publish_then_move(
            active: DirectoryCustody,
            parent_fd: int,
            payload: bytes,
            mode: int,
        ) -> str:
            temporary = original_publish(active, parent_fd, payload, mode)
            (root / "nested").rename(escaped)
            return temporary

        monkeypatch.setattr(DirectoryCustody, "_publish_temp", publish_then_move)
        with pytest.raises(SecurePathCustodyError, match="descendant_parent_replaced"):
            custody.atomic_write_bytes("nested/value.bin", b"forbidden")

    assert list(escaped.iterdir()) == []
