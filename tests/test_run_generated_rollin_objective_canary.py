from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools import run_generated_rollin_objective_canary as canary


def _repo(tmp_path: Path) -> tuple[Path, str]:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
    )
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="ascii")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=tmp_path, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", head],
        cwd=tmp_path,
        check=True,
    )
    return source, head


def test_source_state_requires_clean_published_exact_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, head = _repo(tmp_path)
    monkeypatch.setattr(canary, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(canary, "SOURCE_PATHS", ("source.py",))

    observed_head, bindings = canary._source_state()
    assert observed_head == head
    assert bindings["source.py"]["size_bytes"] == len(b"VALUE = 1\n")

    source.write_text("VALUE = 2\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="clean source"):
        canary._source_state()


def test_source_state_rejects_unpublished_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _source, _head = _repo(tmp_path)
    (tmp_path / "other.py").write_text("VALUE = 2\n", encoding="ascii")
    subprocess.run(["git", "add", "other.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "ahead"], cwd=tmp_path, check=True)
    monkeypatch.setattr(canary, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(canary, "SOURCE_PATHS", ("source.py",))

    with pytest.raises(RuntimeError, match="not published"):
        canary._source_state()


def test_branch_specialization_gate_rejects_rng_or_state_collapse() -> None:
    collapsed = [
        {
            "objective_receipt": {
                "generated_receipt": {
                    "branches": [
                        {"generated_tokens_sha256": "a" * 64},
                        {"generated_tokens_sha256": "a" * 64},
                    ]
                }
            }
        }
    ]
    gates = canary._branch_specialization_gates(collapsed, [0.08])
    assert gates == {
        "branch_generated_prefix_distinct": False,
        "branch_state_specialized": False,
    }

    specialized = [
        {
            "objective_receipt": {
                "generated_receipt": {
                    "branches": [
                        {"generated_tokens_sha256": "a" * 64},
                        {"generated_tokens_sha256": "b" * 64},
                    ]
                }
            }
        }
    ]
    assert all(canary._branch_specialization_gates(specialized, [0.31]).values())
