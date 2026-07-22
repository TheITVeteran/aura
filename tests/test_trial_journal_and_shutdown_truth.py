"""CP126: durable trial receipts, honest health, and shutdown that stops.

Covers the in-flight batch:
- TrialJournal (CP126 51654706): long experiment runs must survive a crash,
  resume only into the SAME experiment, and record failures as receipts.
- CognitiveEngine.stop (CP126 8d7a39ac): a stopped engine must refuse
  cognitive work, not merely empty its phase list.
- DeepDeliberationEngine.get_status (CP126 6b3e534c): health derived from
  what the engine actually did, never the literal True.
- execute_tool (CP126 128107a8): a missing skill_router is a readiness
  failure, not a license to construct an ad hoc authority.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from core.brain.llm.latent_cortex.trial_journal import TrialJournal, manifest_digest


# ── TrialJournal ───────────────────────────────────────────────────────────


def test_journal_persists_and_resumes_completed_trials(tmp_path):
    path = tmp_path / "run.jsonl"
    manifest = {"runner": "x", "arms": ["a", "b"]}

    first = TrialJournal(path, manifest=manifest).open()
    calls: list[str] = []

    def work_one():
        calls.append("one")
        return {"success": True, "cost": 3}

    record = first.run_trial("a:0", work_one)
    assert record.ok and record.payload == {"success": True, "cost": 3}

    # A resumed journal for the SAME manifest skips completed work exactly.
    second = TrialJournal(path, manifest=manifest).open()
    assert second.resumed is True
    assert second.is_complete("a:0")
    rerun = second.run_trial("a:0", work_one)
    assert rerun.payload == {"success": True, "cost": 3}
    assert calls == ["one"], "a completed trial must not re-execute on resume"


def test_journal_refuses_a_different_experiment(tmp_path):
    """Resume must attach only to the same manifest — mixing two experiments
    into one journal silently corrupts both."""
    path = tmp_path / "run.jsonl"
    TrialJournal(path, manifest={"arms": ["a"]}).open()

    with pytest.raises(ValueError, match="manifest_mismatch"):
        TrialJournal(path, manifest={"arms": ["a", "b"]}).open()


def test_failing_trial_becomes_a_receipt_not_a_lost_run(tmp_path):
    journal = TrialJournal(tmp_path / "run.jsonl", manifest={"m": 1}).open()

    def explode():
        raise RuntimeError("solver died")

    record = journal.run_trial("bad", explode)

    assert record.ok is False
    assert "RuntimeError" in record.error
    # The failure is durable and the journal keeps accepting work.
    ok = journal.run_trial("good", lambda: {"v": 1})
    assert ok.ok is True
    summary = journal.summary()
    assert summary["trials"] == 2
    assert summary["failed"] == 1


def test_truncated_final_line_is_skipped_not_fatal(tmp_path):
    """A crash mid-append truncates at most the final line; resume must
    shrug it off instead of losing the run."""
    path = tmp_path / "run.jsonl"
    manifest = {"m": 2}
    journal = TrialJournal(path, manifest=manifest).open()
    journal.record("done", ok=True, payload={"v": 1})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": "partial", "ok": tru')  # crash mid-write

    resumed = TrialJournal(path, manifest=manifest).open()

    assert resumed.is_complete("done")
    assert not resumed.is_complete("partial")
    assert resumed.skipped_corrupt_lines == 1


def test_manifest_digest_is_order_stable():
    assert manifest_digest({"a": 1, "b": 2}) == manifest_digest({"b": 2, "a": 1})
    assert manifest_digest({"a": 1}) != manifest_digest({"a": 2})


def test_factorial_runner_resumes_without_reexecuting(tmp_path):
    """End-to-end: the longest runner journals each trial and a resumed run
    re-executes nothing it already completed."""
    from core.brain.llm.latent_cortex.experiments import (
        run_factorial_ablations,
        task_battery,
    )

    battery = task_battery(["boolean"], [2], 2, seed=11)
    path = tmp_path / "factorial.jsonl"
    executed: list[str] = []

    def solve_arm(task, arm):
        executed.append(f"{arm}:{task.seed}")
        return True, 100

    first = run_factorial_ablations(
        solve_arm, {"boolean": battery}, arms=("latent_opt",), journal_path=path
    )
    first_count = len(executed)
    assert first_count > 0
    assert first["arms"]["latent_opt"]["boolean"]["n"] == 2

    executed.clear()
    second = run_factorial_ablations(
        solve_arm, {"boolean": battery}, arms=("latent_opt",), journal_path=path
    )

    assert executed == [], "a resumed run must skip every completed trial"
    assert second["arms"]["latent_opt"]["boolean"]["n"] == 2


# ── CognitiveEngine.stop ───────────────────────────────────────────────────


def test_stopped_cognitive_engine_refuses_new_work():
    from core.brain.cognitive_engine import CognitiveEngine

    engine = CognitiveEngine()
    engine.stop()

    assert engine.stopped is True
    with pytest.raises(RuntimeError, match="cognitive_engine_stopped"):
        asyncio.run(engine.think("anything"))


# ── DeepDeliberationEngine health ──────────────────────────────────────────


def test_deliberation_health_is_derived_not_constant():
    from core.brain.deep_deliberation import DeepDeliberationEngine

    engine = DeepDeliberationEngine()
    status = engine.get_status()
    # Untested is honestly unknown, reported healthy with zero completions.
    assert status["healthy"] is True
    assert status.get("model_backed", 0) == 0

    # A streak of unbacked deliberations must surface as ill health.
    engine._unbacked = 5
    engine._consecutive_failures = 5
    unhealthy = engine.get_status()
    assert unhealthy["healthy"] is False
    assert unhealthy["state"] == "degraded"
    assert unhealthy["unhealthy_reasons"], "ill health must say why"


# ── execute_tool readiness ─────────────────────────────────────────────────


def test_execute_tool_refuses_without_the_real_router():
    from core.capability_engine import execute_tool
    from core.container import ServiceContainer

    saved = ServiceContainer.get("skill_router", default=None)
    try:
        if saved is not None:
            ServiceContainer.unregister("skill_router")
        result = asyncio.run(execute_tool("clock", {}))
        assert result["ok"] is False
        # No ad hoc engine may have been published under the canonical name.
        assert ServiceContainer.get("skill_router", default=None) is None
    finally:
        if saved is not None:
            ServiceContainer.register_instance("skill_router", saved)
