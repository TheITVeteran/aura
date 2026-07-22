"""CP126: adapter promotion and quarantine are bounded operations.

promote_adapter is what puts trained weights in front of the user, and
quarantine_adapter performs a filesystem move. Both took a bare path.
"""
from __future__ import annotations

import pytest

from core.adaptation.post_training_validator import (
    PostTrainingValidator,
    ValidationResult,
)


def _validator(tmp_path):
    return PostTrainingValidator(
        model_path="unused",
        adapter_base_dir=str(tmp_path / "adapters"),
        quarantine_dir=tmp_path / "quarantine",
        validation_log_dir=tmp_path / "logs",
    )


def _result(adapter_path, *, passed=True):
    return ValidationResult(
        passed=passed,
        pass_rate=1.0 if passed else 0.2,
        total_probes=5,
        passed_probes=5 if passed else 1,
        failed_probes=0 if passed else 4,
        adapter_path=str(adapter_path),
    )


class TestPathContainment:
    def test_path_outside_the_adapter_root_is_rejected(self, tmp_path):
        validator = _validator(tmp_path)
        outside = tmp_path / "elsewhere" / "weights"
        outside.mkdir(parents=True)
        assert validator._contained_adapter_path(str(outside)) is None

    def test_traversal_is_rejected(self, tmp_path):
        validator = _validator(tmp_path)
        assert validator._contained_adapter_path("../../etc") is None

    def test_path_inside_the_root_is_accepted(self, tmp_path):
        validator = _validator(tmp_path)
        inside = tmp_path / "adapters" / "candidate"
        inside.mkdir(parents=True)
        assert validator._contained_adapter_path(str(inside)) == inside.resolve()

    @pytest.mark.asyncio
    async def test_quarantine_refuses_to_move_an_outside_path(self, tmp_path):
        validator = _validator(tmp_path)
        outside = tmp_path / "precious" / "data"
        outside.mkdir(parents=True)
        (outside / "keep.txt").write_text("keep me", encoding="utf-8")

        await validator.quarantine_adapter(str(outside), reason="test")

        # shutil.move must not have relocated anything.
        assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep me"


class TestPromotionRequiresItsEvidence:
    def test_missing_validation_is_refused(self, tmp_path):
        validator = _validator(tmp_path)
        adapter = tmp_path / "adapters" / "new"
        adapter.mkdir(parents=True)
        assert validator._promotion_is_authorized(adapter.resolve(), None) is False

    def test_failed_validation_is_refused(self, tmp_path):
        validator = _validator(tmp_path)
        adapter = tmp_path / "adapters" / "new"
        adapter.mkdir(parents=True)
        assert (
            validator._promotion_is_authorized(
                adapter.resolve(), _result(adapter, passed=False)
            )
            is False
        )

    def test_validation_for_a_different_adapter_is_refused(self, tmp_path):
        validator = _validator(tmp_path)
        adapter = tmp_path / "adapters" / "new"
        other = tmp_path / "adapters" / "other"
        adapter.mkdir(parents=True)
        other.mkdir()
        assert (
            validator._promotion_is_authorized(adapter.resolve(), _result(other))
            is False
        )

    def test_passing_validation_for_this_adapter_is_authorized(self, tmp_path):
        validator = _validator(tmp_path)
        adapter = tmp_path / "adapters" / "new"
        adapter.mkdir(parents=True)
        assert (
            validator._promotion_is_authorized(adapter.resolve(), _result(adapter))
            is True
        )

    @pytest.mark.asyncio
    async def test_promote_without_validation_does_not_activate(self, tmp_path, monkeypatch):
        from core.adaptation import post_training_validator as module

        adapter_root = tmp_path / "adapters"
        old_adapter = adapter_root / "old"
        new_adapter = adapter_root / "new"
        old_adapter.mkdir(parents=True)
        new_adapter.mkdir()
        active = adapter_root / "active"
        active.symlink_to(old_adapter)
        monkeypatch.setattr(module, "ACTIVE_ADAPTER_LINK", active)
        validator = _validator(tmp_path)

        assert await validator.promote_adapter(str(new_adapter)) is False
        # The previously active adapter is still serving.
        assert active.resolve() == old_adapter.resolve()


class TestRollbackPointIsRequired:
    def test_backup_failure_blocks_promotion(self, tmp_path, monkeypatch):
        """Losing the rollback point must not be a warning."""
        from core.adaptation import post_training_validator as module

        adapter_root = tmp_path / "adapters"
        old_adapter = adapter_root / "old"
        new_adapter = adapter_root / "new"
        old_adapter.mkdir(parents=True)
        new_adapter.mkdir()
        active = adapter_root / "active"
        active.symlink_to(old_adapter)

        class _Gateway:
            def replace_symlink(self, link, target, source=""):
                if "backup_adapter" in source:
                    raise OSError("read-only filesystem")
                link.symlink_to(target)

        monkeypatch.setattr(module, "get_file_write_gateway", lambda: _Gateway())
        validator = _validator(tmp_path)

        with pytest.raises(RuntimeError, match="no_rollback_point"):
            validator._promote_adapter_links_sync(new_adapter.resolve(), active)

        # Active must be untouched, so rollback remains possible.
        assert active.resolve() == old_adapter.resolve()
