from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tools import manage_unified_recurrent_qualified_activation as command


def _arguments(tmp_path: Path, package: Path, lifecycle: Path) -> argparse.Namespace:
    root = tmp_path / "authority"
    return argparse.Namespace(
        package=package,
        lifecycle_result=lifecycle,
        pointer=root / "active.json",
        releases_root=root / "releases",
        activation=root / "qualified-active.json",
        expected_current_pointer_sha256=None,
        expected_current_activation_sha256=None,
    )


def _lifecycle(tmp_path: Path) -> tuple[Path, dict]:
    value = {"controller_sha256": "c" * 64, "result_sha256": "d" * 64}
    path = tmp_path / "lifecycle.json"
    path.write_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    path.chmod(0o600)
    return path, value


def _manifest() -> dict:
    return {
        "package_id": "fixture",
        "manifest_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "domain_contract": {
            "families": ["khop"],
            "task_depths": [2],
            "recurrence_depth": 4,
        },
    }


def test_activation_transaction_publishes_exact_pointer_then_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    lifecycle_path, lifecycle = _lifecycle(tmp_path)
    arguments = _arguments(tmp_path, package, lifecycle_path)
    pointer = {"pointer_sha256": "e" * 64}
    activation = {
        "activation_sha256": "f" * 64,
        "mode": "qualified_typed_only",
        "families": ["khop"],
        "task_depths": [2],
    }
    observed: list[str] = []
    monkeypatch.setattr(
        command,
        "inspect_shadow_package",
        lambda _path: {"manifest": _manifest()},
    )

    def publish_pointer(*_args, **_kwargs):
        observed.append("pointer")
        return pointer

    def seal(manifest, received_lifecycle, received_pointer):
        assert manifest == _manifest()
        assert received_lifecycle == lifecycle
        assert received_pointer == pointer
        observed.append("seal")
        return activation

    def publish_activation(received, **_kwargs):
        assert received == activation
        observed.append("authority")
        return activation

    monkeypatch.setattr(command, "publish_shadow_pointer", publish_pointer)
    monkeypatch.setattr(command, "seal_qualified_activation", seal)
    monkeypatch.setattr(command, "publish_qualified_activation", publish_activation)
    monkeypatch.setattr(command, "resolve_shadow_pointer", lambda *_args, **_kwargs: package)
    monkeypatch.setattr(command, "activation_matches_shadow_receipt", lambda *_args: True)

    result = command._activate(arguments)

    assert observed == ["pointer", "seal", "authority"]
    assert result["active"] is True
    assert result["activation_sha256"] == "f" * 64
    assert result["pointer_sha256"] == "e" * 64


def test_activation_failure_rolls_back_a_new_shadow_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    lifecycle_path, _lifecycle_value = _lifecycle(tmp_path)
    arguments = _arguments(tmp_path, package, lifecycle_path)
    pointer = {"pointer_sha256": "e" * 64}
    retired: list[str] = []
    monkeypatch.setattr(
        command,
        "inspect_shadow_package",
        lambda _path: {"manifest": _manifest()},
    )
    monkeypatch.setattr(command, "publish_shadow_pointer", lambda *_args, **_kwargs: pointer)
    monkeypatch.setattr(command, "seal_qualified_activation", lambda *_args: {})
    monkeypatch.setattr(
        command,
        "publish_qualified_activation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("rejected")),
    )
    monkeypatch.setattr(
        command,
        "deactivate_shadow_pointer",
        lambda **kwargs: retired.append(kwargs["expected_current_sha256"]),
    )

    with pytest.raises(RuntimeError, match="rejected"):
        command._activate(arguments)

    assert retired == ["e" * 64]


def test_post_publication_verification_failure_revokes_authority_before_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    lifecycle_path, _lifecycle_value = _lifecycle(tmp_path)
    arguments = _arguments(tmp_path, package, lifecycle_path)
    pointer = {"pointer_sha256": "e" * 64}
    activation = {
        "activation_sha256": "f" * 64,
        "mode": "qualified_typed_only",
        "families": ["khop"],
        "task_depths": [2],
    }
    rollback: list[str] = []
    monkeypatch.setattr(
        command,
        "inspect_shadow_package",
        lambda _path: {"manifest": _manifest()},
    )
    monkeypatch.setattr(command, "publish_shadow_pointer", lambda *_args, **_kwargs: pointer)
    monkeypatch.setattr(command, "seal_qualified_activation", lambda *_args: activation)
    monkeypatch.setattr(
        command,
        "publish_qualified_activation",
        lambda *_args, **_kwargs: activation,
    )
    monkeypatch.setattr(command, "resolve_shadow_pointer", lambda *_args, **_kwargs: package)
    monkeypatch.setattr(command, "activation_matches_shadow_receipt", lambda *_args: False)
    monkeypatch.setattr(
        command,
        "deactivate_qualified_activation",
        lambda **_kwargs: rollback.append("authority"),
    )
    monkeypatch.setattr(
        command,
        "deactivate_shadow_pointer",
        lambda **_kwargs: rollback.append("pointer"),
    )

    with pytest.raises(
        command.UnifiedRecurrentQualifiedActivationCommandError,
        match="different package identity",
    ):
        command._activate(arguments)

    assert rollback == ["authority", "pointer"]


def test_activation_refuses_to_replace_another_active_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    lifecycle_path, _lifecycle_value = _lifecycle(tmp_path)
    arguments = _arguments(tmp_path, package, lifecycle_path)
    arguments.pointer.parent.mkdir(parents=True)
    arguments.pointer.write_text("present", encoding="ascii")
    other = tmp_path / "other-package"
    other.mkdir()
    monkeypatch.setattr(
        command,
        "inspect_shadow_package",
        lambda _path: {"manifest": _manifest()},
    )
    monkeypatch.setattr(command, "resolve_shadow_pointer", lambda *_args, **_kwargs: other)
    monkeypatch.setattr(
        command,
        "publish_shadow_pointer",
        lambda *_args, **_kwargs: pytest.fail("different pointer must not be replaced"),
    )

    with pytest.raises(
        command.UnifiedRecurrentQualifiedActivationCommandError,
        match="refuses to replace",
    ):
        command._activate(arguments)


def test_status_refuses_orphaned_qualified_authority(tmp_path: Path) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    (root / "qualified-active.json").write_text("present", encoding="ascii")
    arguments = argparse.Namespace(
        pointer=root / "active.json",
        releases_root=root / "releases",
        activation=root / "qualified-active.json",
    )

    with pytest.raises(
        command.UnifiedRecurrentQualifiedActivationCommandError,
        match="without a shadow pointer",
    ):
        command._status(arguments)
