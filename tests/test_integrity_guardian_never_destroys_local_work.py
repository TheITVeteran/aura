"""The integrity guardian must never silently overwrite explained changes.

Live incident (2026-07-18 soak): the guardian auto-restored
``interface/routes/chat.py`` and four latent-cortex modules from git HEAD
**eight times each** while a parallel session was actively editing them —
silently reverting live uncommitted work under the developer's feet, all
night. The safety check existed but was gated on the environment label
(``is_dev``), which the headless soak did not carry, and an unavailable
git status was indistinguishable from a clean tree, so failure meant
"destroy" rather than "don't".

The invariant this pins: **auto-restore is destructive, so it may only fire
when the change is genuinely unexplained.** Detection and alerting are
always correct and always still happen; destruction requires proof that no
local explanation exists. A working checkout is dangerous to overwrite
whatever environment label it carries; a real deployment has a clean tree
and is therefore unaffected.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

MONITORED = "interface/routes/chat.py"


@pytest.fixture()
def guardian(monkeypatch):
    from core.security import integrity_guardian as ig_mod

    instance = ig_mod.IntegrityGuardian()
    monkeypatch.setattr(instance, "_is_monitored_path", lambda _p: True)

    class _Security:
        auto_fix_enabled = True

    monkeypatch.setattr(
        "core.config.config",
        type("_Cfg", (), {"security": _Security(), "env": "PRODUCTION"})(),
        raising=False,
    )
    return instance


class TestNeverDestroysExplainedChanges:
    def test_locally_modified_file_is_not_overwritten(self, guardian):
        """The soak anatomy: git itself explains the change."""
        assert guardian._should_auto_restore(MONITORED, {MONITORED: "M"}) is False

    @pytest.mark.parametrize("code", ["M", "MM", "??", "A", "D", "R"])
    def test_every_working_tree_state_blocks_destruction(self, guardian, code):
        assert guardian._should_auto_restore(MONITORED, {MONITORED: code}) is False

    def test_environment_label_is_not_part_of_the_decision(self, guardian, monkeypatch):
        """The 2026-07-18 regression: the guard was gated on is_dev, so a
        headless/production-labelled run destroyed a live working tree."""

        class _Security:
            auto_fix_enabled = True

        for env in ("DEV", "PRODUCTION", "LIVE", "TEST"):
            monkeypatch.setattr(
                "core.config.config",
                type("_Cfg", (), {"security": _Security(), "env": env})(),
                raising=False,
            )
            assert guardian._should_auto_restore(MONITORED, {MONITORED: "M"}) is False

    def test_unknown_vcs_state_fails_safe(self, guardian):
        """git unavailable/failed → the change cannot be shown unexplained →
        never destroy. (Previously indistinguishable from a clean tree.)"""
        assert guardian._should_auto_restore(MONITORED, None) is False


class TestStillRepairsGenuineTamper:
    def test_unexplained_change_in_a_clean_tree_is_restored(self, guardian):
        """The control is preserved where it is actually correct: a clean
        deployment whose file changed with no local explanation."""
        assert guardian._should_auto_restore(MONITORED, {}) is True

    def test_other_files_being_dirty_does_not_shield_this_one(self, guardian):
        assert guardian._should_auto_restore(MONITORED, {"docs/README.md": "M"}) is True

    def test_disabled_auto_fix_is_honored(self, guardian, monkeypatch):
        class _Security:
            auto_fix_enabled = False

        monkeypatch.setattr(
            "core.config.config",
            type("_Cfg", (), {"security": _Security(), "env": "PRODUCTION"})(),
            raising=False,
        )
        assert guardian._should_auto_restore(MONITORED, {}) is False

    def test_unmonitored_path_is_never_restored(self, guardian, monkeypatch):
        monkeypatch.setattr(guardian, "_is_monitored_path", lambda _p: False)
        assert guardian._should_auto_restore(MONITORED, {}) is False


class TestEndToEndVerifyBehaviour:
    def _instance(self, monkeypatch, *, git_status):
        from core.security import integrity_guardian as ig_mod

        instance = ig_mod.IntegrityGuardian()
        instance._manifest = {MONITORED: "expected-hash"}
        monkeypatch.setattr(instance, "_is_monitored_path", lambda _p: True)
        monkeypatch.setattr(instance, "_hash_file", lambda _p: "different-hash")
        monkeypatch.setattr(instance, "_get_git_status_map", lambda: git_status)
        monkeypatch.setattr(instance, "_git_active_paths", lambda: set())
        monkeypatch.setattr(ig_mod.Path, "exists", lambda self: True, raising=False)
        return instance

    def test_explained_change_is_neither_destroyed_nor_alarming(self, monkeypatch):
        """A developer's own uncommitted edit is not a security incident:
        no overwrite, and no false tamper alarm either."""
        instance = self._instance(monkeypatch, git_status={MONITORED: "M"})
        restored: list[str] = []
        monkeypatch.setattr(
            instance, "_restore_file_via_git", lambda p: restored.append(p) or True
        )

        alerts = instance._verify_all()

        assert restored == [], "a locally-explained change must not be overwritten"
        assert alerts == [], "a locally-explained change must not raise tamper alarms"

    def test_unknown_vcs_state_alerts_without_destroying(self, monkeypatch):
        """The dangerous case the soak actually hit: git state unavailable.
        Stay loud (the operator must know) but never overwrite."""
        instance = self._instance(monkeypatch, git_status=None)
        restored: list[str] = []
        monkeypatch.setattr(
            instance, "_restore_file_via_git", lambda p: restored.append(p) or True
        )

        alerts = instance._verify_all()

        assert restored == [], "unknown provenance must never authorize destruction"
        assert any(MONITORED in str(alert) for alert in alerts), (
            "an unexplained mismatch must still reach the operator"
        )
