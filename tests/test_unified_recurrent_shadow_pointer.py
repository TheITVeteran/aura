from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.brain.llm import unified_recurrent_shadow as shadow
from core.brain.llm.unified_recurrent_shadow_pointer import (
    UnifiedRecurrentShadowPointerError,
    deactivate_shadow_pointer,
    publish_shadow_pointer,
    resolve_shadow_pointer,
)


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    root = tmp_path / "activation"
    releases = root / "releases"
    package = releases / "cp267-package"
    package.mkdir(mode=0o700, parents=True)
    root.chmod(0o700)
    releases.chmod(0o700)
    package.chmod(0o700)

    def inspect(path: Path):
        name = Path(path).name
        digest = "a" * 64 if name == "cp267-package" else "b" * 64
        return {
            "manifest": {
                "package_id": name,
                "manifest_sha256": digest,
            }
        }

    monkeypatch.setattr(shadow, "inspect_shadow_package", inspect)
    return root / "active.json", releases, package


def test_pointer_publication_is_restart_stable_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer_path, releases, package = _fixture(tmp_path, monkeypatch)

    first = publish_shadow_pointer(
        package,
        pointer_path=pointer_path,
        releases_root=releases,
    )
    second = publish_shadow_pointer(
        package,
        pointer_path=pointer_path,
        releases_root=releases,
    )

    assert first == second
    assert pointer_path.stat().st_mode & 0o777 == 0o600
    assert resolve_shadow_pointer(pointer_path, releases_root=releases) == package


def test_pointer_replacement_requires_exact_compare_and_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer_path, releases, package = _fixture(tmp_path, monkeypatch)
    replacement = releases / "cp267-replacement"
    replacement.mkdir(mode=0o700)
    first = publish_shadow_pointer(
        package,
        pointer_path=pointer_path,
        releases_root=releases,
    )

    with pytest.raises(
        UnifiedRecurrentShadowPointerError,
        match="lost compare-and-swap",
    ):
        publish_shadow_pointer(
            replacement,
            pointer_path=pointer_path,
            releases_root=releases,
        )

    second = publish_shadow_pointer(
        replacement,
        pointer_path=pointer_path,
        releases_root=releases,
        expected_current_sha256=first["pointer_sha256"],
    )
    assert second["package_id"] == "cp267-replacement"


def test_pointer_cannot_select_outside_release_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer_path, releases, _package = _fixture(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)

    with pytest.raises(
        UnifiedRecurrentShadowPointerError,
        match="outside the release root",
    ):
        publish_shadow_pointer(
            outside,
            pointer_path=pointer_path,
            releases_root=releases,
        )


def test_pointer_tampering_fails_before_package_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer_path, releases, package = _fixture(tmp_path, monkeypatch)
    publish_shadow_pointer(
        package,
        pointer_path=pointer_path,
        releases_root=releases,
    )
    value = json.loads(pointer_path.read_text(encoding="ascii"))
    value["manifest_sha256"] = "f" * 64
    pointer_path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="ascii",
    )
    pointer_path.chmod(0o600)

    with pytest.raises(
        UnifiedRecurrentShadowPointerError,
        match="identity differs",
    ):
        resolve_shadow_pointer(pointer_path, releases_root=releases)


def test_deactivation_is_atomic_and_retains_exact_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer_path, releases, package = _fixture(tmp_path, monkeypatch)
    pointer = publish_shadow_pointer(
        package,
        pointer_path=pointer_path,
        releases_root=releases,
    )

    retired = deactivate_shadow_pointer(
        pointer_path=pointer_path,
        releases_root=releases,
        expected_current_sha256=pointer["pointer_sha256"],
    )

    assert retired == pointer
    assert not pointer_path.exists()
    archive = pointer_path.parent / "retired" / f"{pointer['pointer_sha256']}.json"
    assert archive.is_file()
    assert json.loads(archive.read_text(encoding="ascii")) == pointer


def test_active_qualified_authority_blocks_pointer_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointer_path, releases, package = _fixture(tmp_path, monkeypatch)
    replacement = releases / "cp267-replacement"
    replacement.mkdir(mode=0o700)
    pointer = publish_shadow_pointer(
        package,
        pointer_path=pointer_path,
        releases_root=releases,
    )
    qualified = pointer_path.parent / "qualified-active.json"
    qualified.write_text("active", encoding="ascii")
    qualified.chmod(0o600)

    with pytest.raises(
        UnifiedRecurrentShadowPointerError,
        match="qualified activation must be revoked",
    ):
        publish_shadow_pointer(
            replacement,
            pointer_path=pointer_path,
            releases_root=releases,
            expected_current_sha256=pointer["pointer_sha256"],
        )
    with pytest.raises(
        UnifiedRecurrentShadowPointerError,
        match="qualified activation must be revoked",
    ):
        deactivate_shadow_pointer(
            pointer_path=pointer_path,
            releases_root=releases,
            expected_current_sha256=pointer["pointer_sha256"],
        )

    assert resolve_shadow_pointer(pointer_path, releases_root=releases) == package
