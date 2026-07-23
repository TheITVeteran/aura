"""CP126 file_actuator — the boundary between a request and a real write."""
from __future__ import annotations

import hashlib
import os

import pytest

from core.actuation.file_actuator import (
    HIGH_RISK_REPO_ACTIONS,
    MAX_WRITE_BYTES,
    FileActuationError,
    FileActuator,
    resolve_write_target,
)


@pytest.fixture(autouse=True)
def _workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_FILE_ACTUATOR_ROOTS", str(tmp_path))
    return tmp_path


class _RecordingActuator:
    def __init__(self):
        self.calls = []

    async def actuate(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True}


@pytest.fixture
def actuator(monkeypatch):
    recorder = _RecordingActuator()
    monkeypatch.setattr(
        "core.actuation.file_actuator.get_world_actuator", lambda: recorder
    )
    return recorder


class TestPathAuthority:
    """1fa38892: writes are contained, symlink-resolved and byte-bounded."""

    def test_outside_paths_are_refused(self, tmp_path):
        with pytest.raises(FileActuationError, match="outside every allowed"):
            resolve_write_target("/etc/passwd")

    def test_empty_path_refused(self):
        with pytest.raises(FileActuationError, match="empty"):
            resolve_write_target("  ")

    def test_inside_workspace_resolves(self, tmp_path):
        assert resolve_write_target(str(tmp_path / "a.txt")) == (tmp_path / "a.txt").resolve()

    def test_symlink_cannot_redirect_outside(self, tmp_path):
        outside = tmp_path.parent / "outside_dir"
        outside.mkdir(exist_ok=True)
        link = tmp_path / "escape"
        link.symlink_to(outside, target_is_directory=True)
        with pytest.raises(FileActuationError, match="outside every allowed"):
            resolve_write_target(str(link / "loot.txt"))

    @pytest.mark.asyncio
    async def test_oversized_write_refused(self, tmp_path, actuator):
        with pytest.raises(FileActuationError, match="boundary limit"):
            await FileActuator.write_file(
                path=str(tmp_path / "big.txt"), content="x" * (MAX_WRITE_BYTES + 1)
            )
        assert actuator.calls == []


class TestCompareAndSwap:
    """174481d8: concurrent changes are not silently overwritten."""

    @pytest.mark.asyncio
    async def test_stale_precondition_refuses(self, tmp_path, actuator):
        target = tmp_path / "f.txt"
        target.write_text("current", encoding="utf-8")
        stale = hashlib.sha256(b"what the caller last read").hexdigest()
        with pytest.raises(FileActuationError, match="compare-and-swap"):
            await FileActuator.write_file(
                path=str(target), content="new", expected_sha256=stale
            )
        assert actuator.calls == []

    @pytest.mark.asyncio
    async def test_matching_precondition_proceeds(self, tmp_path, actuator):
        target = tmp_path / "f.txt"
        target.write_text("current", encoding="utf-8")
        good = hashlib.sha256(b"current").hexdigest()
        result = await FileActuator.write_file(
            path=str(target), content="new", expected_sha256=good
        )
        assert result["ok"] is True
        assert len(actuator.calls) == 1

    @pytest.mark.asyncio
    async def test_create_only_mode(self, tmp_path, actuator):
        target = tmp_path / "f.txt"
        target.write_text("exists", encoding="utf-8")
        with pytest.raises(FileActuationError, match="replace an existing"):
            await FileActuator.write_file(
                path=str(target), content="x", allow_replace=False
            )

    @pytest.mark.asyncio
    async def test_rollback_data_rides_with_the_receipt(self, tmp_path, actuator):
        target = tmp_path / "f.txt"
        target.write_text("before", encoding="utf-8")
        result = await FileActuator.write_file(path=str(target), content="after")
        assert result["previous_sha256"] == hashlib.sha256(b"before").hexdigest()
        assert result["existed_before"] is True


class TestRepositoryRiskClassification:
    """c0ecefb0: destructive git operations are high risk; unknown is too."""

    def test_destructive_actions_are_high_risk(self):
        for action in ("reset", "force_push", "delete_branch", "remote_set_url", "clean"):
            assert FileActuator.classify_repo_action(action) is True, action

    def test_unknown_action_is_high_risk(self):
        assert FileActuator.classify_repo_action("do_something_novel") is True
        assert FileActuator.classify_repo_action("") is True

    def test_read_only_actions_are_not_high_risk(self):
        for action in ("status", "diff", "log"):
            assert FileActuator.classify_repo_action(action) is False, action

    def test_original_high_risk_set_is_preserved(self):
        assert {"push", "publish_code"} <= HIGH_RISK_REPO_ACTIONS


class TestParameterSmuggling:
    """b7894f2c: params cannot replace the action after classification."""

    @pytest.mark.asyncio
    async def test_smuggled_action_cannot_downgrade_risk(self, tmp_path, actuator):
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        await FileActuator.modify_repo(
            repo_path=str(repo),
            action="status",
            params={"action": "push"},  # the smuggle attempt
        )
        call = actuator.calls[0]
        # The action ACTUALLY sent is the classified one.
        assert call["params"]["action"] == "status"
        assert call["high_risk_flag"] is False

    @pytest.mark.asyncio
    async def test_declared_destructive_action_is_flagged(self, tmp_path, actuator):
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        await FileActuator.modify_repo(
            repo_path=str(repo), action="reset", params={}
        )
        assert actuator.calls[0]["high_risk_flag"] is True

    @pytest.mark.asyncio
    async def test_smuggled_repo_path_cannot_redirect(self, tmp_path, actuator):
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        await FileActuator.modify_repo(
            repo_path=str(repo),
            action="status",
            params={"repo_path": "/etc"},
        )
        assert actuator.calls[0]["params"]["repo_path"] == str(repo.resolve())


class TestRepositoryContainment:
    """e0f2716e (partial): the path is contained and is a real repository."""

    @pytest.mark.asyncio
    async def test_outside_repo_refused(self, tmp_path, actuator):
        with pytest.raises(FileActuationError, match="outside every allowed"):
            await FileActuator.modify_repo(
                repo_path="/etc", action="status", params={}
            )

    @pytest.mark.asyncio
    async def test_non_repository_refused(self, tmp_path, actuator):
        plain = tmp_path / "plain"
        plain.mkdir()
        with pytest.raises(FileActuationError, match="not a git repository"):
            await FileActuator.modify_repo(
                repo_path=str(plain), action="status", params={}
            )
