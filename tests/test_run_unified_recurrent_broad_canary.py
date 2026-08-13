from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.frontier_tasks import FRONTIER_DOMAINS
from tools import run_unified_recurrent_broad_canary as runner


def test_issuer_is_private_stable_and_reconstructs_blinded_tasks(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    issuer = runner._issuer(tmp_path, (101,), 2)
    reopened = runner._issuer(tmp_path, (101,), 2)
    tasks = runner._tasks(issuer)

    assert reopened == issuer
    assert len(tasks) == len(FRONTIER_DOMAINS)
    assert {task.domain for task in tasks} == set(FRONTIER_DOMAINS)
    assert (tmp_path / "issuer-private.json").stat().st_mode & 0o777 == 0o400
    assert all(
        task.public.answer_commitment_sha256
        == task.blinded_answer.commitment_sha256
        for task in tasks
    )


def test_issuer_refuses_seed_drift_on_resume(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    runner._issuer(tmp_path, (101,), 2)

    with pytest.raises(runner.BroadCanaryRunnerError, match="issuer identity differs"):
        runner._issuer(tmp_path, (102,), 2)


def test_candidate_journal_is_fsynced_canonical_and_plan_bound(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    candidate = {"task_id": "task", "arm": "base_greedy"}
    envelope = {
        "schema": runner.JOURNAL_SCHEMA,
        "plan_sha256": "a" * 64,
        "candidate": candidate,
        "raw_text": "candidate output",
        "score": {"correct": False},
    }
    runner._append_private(path, envelope)

    assert runner._read_lines(path) == [envelope]
    assert runner._candidate_rows(path, plan_sha256="a" * 64) == [candidate]
    with pytest.raises(runner.BroadCanaryRunnerError, match="journal identity differs"):
        runner._candidate_rows(path, plan_sha256="b" * 64)


def test_candidate_journal_refuses_duplicate_arm(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    envelope = {
        "schema": runner.JOURNAL_SCHEMA,
        "plan_sha256": "a" * 64,
        "candidate": {"task_id": "task", "arm": "base_greedy"},
        "raw_text": "candidate output",
        "score": {"correct": False},
    }
    runner._append_private(path, envelope)
    runner._append_private(path, envelope)

    with pytest.raises(runner.BroadCanaryRunnerError, match="duplicate arm"):
        runner._candidate_rows(path, plan_sha256="a" * 64)


def test_arm_order_is_deterministic_and_balanced() -> None:
    orders = [runner._arm_order(f"task-{index}") for index in range(40)]

    assert all(set(order) == set(runner.ARMS) for order in orders)
    assert len({order for order in orders}) > 1


def test_source_binding_commits_every_implementation_file() -> None:
    binding = runner._source_binding("a" * 40)

    assert binding["git_commit"] == "a" * 40
    assert set(binding["implementation_sha256s"]) == set(runner.SOURCE_PATHS)
    assert all(len(value) == 64 for value in binding["implementation_sha256s"].values())


def test_base_decode_uses_cached_canonical_greedy_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def generate_step(tokens, model, *, max_tokens, sampler):
        calls.update(tokens=tokens, model=model, max_tokens=max_tokens)
        assert sampler(runner.mx.array([[0.1, 0.9]])).item() == 1
        yield 11, None
        yield 22, None

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.generate",
        types.SimpleNamespace(generate_step=generate_step),
    )
    monkeypatch.setattr(runner, "_contract_complete", lambda _tokenizer, ids: ids == [11, 22])
    tokenizer = types.SimpleNamespace(eos_token_id=99)
    model = object()
    progress: list[int] = []

    generated, stopped, latency_ms = runner._base_decode(
        model,
        tokenizer,
        (1, 2, 3),
        max_tokens=8,
        progress=progress.append,
    )

    assert generated == (11, 22)
    assert stopped is True
    assert latency_ms >= 0
    assert calls["model"] is model
    assert calls["max_tokens"] == 8
    assert calls["tokens"].tolist() == [1, 2, 3]
    assert progress == [1, 2]


def test_progress_callback_reports_content_free_token_steps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    task = types.SimpleNamespace(task_id="task-1", domain="mathematics")
    callback = runner._progress_callback(
        task=task,
        arm="trained_t4",
        maximum_tokens=32,
        stage="token_step_started",
    )

    callback()
    callback()

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows == [
        {
            "arm": "trained_t4",
            "domain": "mathematics",
            "event": "token_step_started",
            "maximum_tokens": 32,
            "task_id": "task-1",
            "token_step": 1,
        },
        {
            "arm": "trained_t4",
            "domain": "mathematics",
            "event": "token_step_started",
            "maximum_tokens": 32,
            "task_id": "task-1",
            "token_step": 2,
        },
    ]


def test_private_append_rejects_noncanonical_reopen(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_text(json.dumps({"b": 1, "a": 2}) + "\n")

    with pytest.raises(runner.BroadCanaryRunnerError, match="non-canonical"):
        runner._read_lines(path)
