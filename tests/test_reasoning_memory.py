"""Tests for the Reflexion-style reasoning memory substrate."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.reasoning_memory import ReasoningMemory, get_reasoning_memory


@pytest.fixture()
def mem(tmp_path: Path) -> ReasoningMemory:
    return ReasoningMemory(path=tmp_path / "reflections.jsonl")


def test_record_and_recall_failure_mode(mem: ReasoningMemory):
    mem.record(
        task_type="repo_audit",
        objective="explain how the inference gate fallback works",
        passed=False,
        verifier_issues=["referenced path not found in repo: core/foo/bar.py"],
    )
    hits = mem.recall("describe the inference gate fallback cascade", task_type="repo_audit")
    assert hits
    assert "file-span evidence" in hits[0].lesson


def test_passing_records_excluded_from_failures_only(mem: ReasoningMemory):
    mem.record(task_type="math", objective="add two numbers", passed=True)
    assert mem.recall("add two numbers", task_type="math", failures_only=True) == []
    assert mem.recall("add two numbers", task_type="math", failures_only=False)


def test_guard_text_renders_lessons(mem: ReasoningMemory):
    mem.record(
        task_type="math",
        objective="compute the factorial growth",
        passed=False,
        verifier_issues=["arithmetic error: 5! = 100 (correct: 120)"],
    )
    guard = mem.as_guard_text("compute the factorial of n", task_type="math")
    assert "symbolic sandbox" in guard
    assert guard.startswith("Lessons from similar past reasoning")


def test_recall_prioritises_same_task_type(mem: ReasoningMemory):
    mem.record(task_type="code", objective="write a parser", passed=False,
               verifier_issues=["block#0: syntax error"])
    mem.record(task_type="planning", objective="write a parser plan", passed=False,
               verifier_issues=["no verification step"])
    hits = mem.recall("write a parser for tokens", task_type="code")
    assert hits[0].task_type == "code"


def test_persistence_across_instances(tmp_path: Path):
    path = tmp_path / "r.jsonl"
    m1 = ReasoningMemory(path=path)
    m1.record(task_type="logic", objective="prove the theorem", passed=False,
              verifier_issues=["non-sequitur: therefore X"])
    m2 = ReasoningMemory(path=path)
    hits = m2.recall("prove the theorem about X", task_type="logic")
    assert hits and "inference step" in hits[0].lesson


def test_singleton():
    assert get_reasoning_memory() is get_reasoning_memory()
