"""Tests for the local environmental-agency pipeline (#36)."""
from __future__ import annotations

from pathlib import Path

from core.agency.environment_pipeline import (
    run_workspace_digest,
    survey_workspace,
)


def _make_workspace(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "src" / "a.py").write_text("print('a')\n", encoding="utf-8")
    (root / "src" / "b.py").write_text("x = 1\n", encoding="utf-8")
    (root / "docs" / "readme.md").write_text("# hi\n" * 200, encoding="utf-8")
    # noise dir that must be skipped
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")


def test_survey_counts_files_and_skips_noise(tmp_path):
    _make_workspace(tmp_path)
    s = survey_workspace(tmp_path)
    assert s["total_files"] == 3  # a.py, b.py, readme.md — NOT .git/HEAD
    assert s["total_dirs"] == 2   # src, docs — NOT .git
    assert ".py" in s["by_extension"] and s["by_extension"][".py"] == 2
    assert s["largest_file"] == "docs/readme.md"  # biggest by size
    assert s["total_bytes"] > 0


def test_run_writes_governed_digest(tmp_path):
    _make_workspace(tmp_path)
    r = run_workspace_digest(tmp_path, safe_mode=False, ledger_path=tmp_path / "runs.jsonl")
    assert r.success is True
    assert r.wrote_digest is True
    digest = tmp_path / ".aura" / "workspace_digest.md"
    assert digest.exists()
    body = digest.read_text(encoding="utf-8")
    assert "Workspace digest" in body
    assert "3 files" in body
    # provable: a run-ledger line was appended
    assert (tmp_path / "runs.jsonl").read_text(encoding="utf-8").strip()


def test_safe_mode_surveys_but_does_not_write(tmp_path):
    _make_workspace(tmp_path)
    r = run_workspace_digest(tmp_path, safe_mode=True, ledger_path=tmp_path / "runs.jsonl")
    assert r.success is True            # the survey still succeeds
    assert r.wrote_digest is False      # but the brake held
    assert r.total_files == 3           # it did real read-only work
    assert not (tmp_path / ".aura" / "workspace_digest.md").exists()
    assert "safe mode" in r.note.lower()


def test_non_directory_is_handled(tmp_path):
    missing = tmp_path / "nope"
    r = run_workspace_digest(missing, safe_mode=False, ledger_path=tmp_path / "runs.jsonl")
    assert r.success is False
    assert r.wrote_digest is False


def test_survey_is_bounded(tmp_path):
    for i in range(20):
        (tmp_path / f"f{i}.txt").write_text("x", encoding="utf-8")
    s = survey_workspace(tmp_path, max_entries=5)
    assert s["truncated"] is True
