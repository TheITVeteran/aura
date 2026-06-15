"""tests/test_ablation_tool_offline.py
==========================================
Exercises the ablation TOOL end-to-end with a fake model router (no live model),
proving the wiring is honest: it actually runs every condition over the tasks,
writes real graded scores, and the back-compat keys mirror the honest verdict
rather than fabricated constants.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import tools.agi.run_prompt_baseline_ablation as tool


class _FakeRouter:
    """Returns the prompt itself as the 'answer'. Under full_architecture the
    prompt carries the conversation history (which contains the answer), so it
    grades correct; the stateless conditions only see the bare question and fail.
    This mirrors the real memory/context effect without a live model."""

    async def generate_with_metadata(self, *, prompt, system_prompt=None, timeout=90.0, **kwargs):  # noqa: ASYNC109
        return {"ok": True, "text": prompt, "endpoint": "fake", "tokens": len(prompt.split())}


def _run_tool(monkeypatch, tmp_path: Path, tasks_path: str) -> dict:
    monkeypatch.setattr(
        "core.brain.llm_health_router.get_llm_router", lambda: _FakeRouter(), raising=False
    )
    out = tmp_path / "ablation.json"
    monkeypatch.setattr(
        sys, "argv",
        ["run_prompt_baseline_ablation.py", "--tasks", tasks_path, "--output", str(out)],
    )
    rc = asyncio.run(tool.main())
    report = json.loads(out.read_text())
    report["_rc"] = rc
    return report


def test_tool_runs_real_conditions_and_reports_honestly(monkeypatch, tmp_path):
    report = _run_tool(
        monkeypatch, tmp_path, "tests/agi/fixtures/hidden_tasks/recall_tasks.jsonl"
    )
    assert report["_rc"] == 0
    assert report["status"] == "ok"
    assert report["tasks_evaluated"] >= 5

    # Real per-condition scores were computed (not hardcoded).
    full = report["conditions"]["full_architecture"]
    raw = report["conditions"]["raw_model"]
    assert full["mean_score"] > raw["mean_score"]
    assert full["per_task"] and len(full["per_task"]) == report["tasks_evaluated"]

    # Back-compat keys carry the REAL values + honest verdict.
    assert report["aura_scores"]["mean_score"] == full["mean_score"]
    assert report["baseline_scores"]["raw_model"]["mean_score"] == raw["mean_score"]
    assert report["score_separation_verified"] == report["verdict"]["architecture_beats_stateless"]
    assert report["score_separation_verified"] is True


def test_tool_exits_nonzero_when_no_multiturn_tasks(monkeypatch, tmp_path):
    # The legacy single-prompt fixture has no multi-turn tasks → honest no-op,
    # not a fabricated pass.
    report = _run_tool(monkeypatch, tmp_path, "tests/agi/fixtures/hidden_tasks/tasks.jsonl")
    assert report["_rc"] == 2
    assert report["status"] == "no_tasks"
    assert report["tasks_evaluated"] == 0
