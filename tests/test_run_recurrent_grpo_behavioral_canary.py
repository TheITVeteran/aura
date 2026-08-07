from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from core.learning.recurrence_curriculum import task_battery
from tools import run_recurrent_grpo_behavioral_canary as canary


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


def test_stable_seed_binds_every_sampling_coordinate() -> None:
    first = canary._stable_seed(17, "grpo", 2, "task-a", 0)

    assert first == canary._stable_seed(17, "grpo", 2, "task-a", 0)
    assert first != canary._stable_seed(17, "grpo", 2, "task-a", 1)
    assert first != canary._stable_seed(17, "grpo", 3, "task-a", 0)


def test_training_cycle_covers_every_task_before_repeating() -> None:
    tasks = task_battery(["boolean", "modular"], [2], 2, seed=31)

    observed = [
        canary._cyclic_task(tasks, one_based_step=step).task_id
        for step in range(1, len(tasks) * 2 + 1)
    ]

    assert observed[: len(tasks)] == [task.task_id for task in tasks]
    assert observed[len(tasks) :] == observed[: len(tasks)]


def test_reward_is_correctness_dominant_and_format_credit_bounded() -> None:
    task = task_battery(["boolean"], [2], 1, seed=43)[0]

    correct_verdict, correct_reward = canary._grade_reward(
        task,
        task.answer,
        format_credit=0.1,
    )
    wrong_verdict, wrong_reward = canary._grade_reward(
        task,
        'FINAL_ANSWER: {"value":999}',
        format_credit=0.1,
    )
    invalid_verdict, invalid_reward = canary._grade_reward(
        task,
        "not an answer",
        format_credit=0.1,
    )

    assert correct_verdict["correct"] is True
    assert correct_reward == 1.0
    assert wrong_verdict["correct"] is False
    assert wrong_reward == 0.1
    assert invalid_verdict["correct"] is False
    assert invalid_reward == 0.0


def _rejected_sample_receipt() -> dict[str, object]:
    return {
        "episode_id": "episode-1",
        "seed": 17,
        "token_count": 4,
        "behavior_admitted": False,
        "max_abs_logprob_drift": 4.5,
        "mean_abs_logprob_drift": 0.2,
        "clipped_token_fraction": 0.5,
        "old_policy_approx_kl": 0.02,
        "sampling_config": {
            "max_abs_logprob_drift": 4.0,
            "max_mean_abs_logprob_drift": 0.5,
            "max_clipped_token_fraction": 0.25,
            "max_old_policy_approx_kl": 0.1,
        },
        "cached_params_unchanged": True,
        "cached_nonparametric_memory_status": "disabled_by_policy",
        "cached_runtime_integrity": {"ok": True},
        "cached_recurrence_adapter": {
            "active": True,
            "calls": 1,
            "adapted_positions": 4,
        },
        "episode_receipt": {"honest_flags": []},
    }


def test_sampling_rejection_diagnostics_name_every_failed_bound() -> None:
    diagnostic = canary._sampling_rejection_diagnostics(_rejected_sample_receipt())

    assert diagnostic["failed_gates"] == [
        "max_abs_logprob_drift",
        "clipped_token_fraction",
        "behavior_admitted",
    ]
    assert diagnostic["checks"]["recurrence_adapter_active"] is True


def test_sampling_failure_retains_receipts_and_compact_causes() -> None:
    receipt = _rejected_sample_receipt()
    error = canary.RecurrentCanarySamplingError(
        admitted=0,
        requested=2,
        rejected=[receipt, json.loads(json.dumps(receipt))],
    )

    assert error.admitted == 0
    assert error.requested == 2
    assert error.rejected == [receipt, receipt]
    assert "clipped_token_fraction" in str(error)
    assert len(error.diagnostics) == 2
