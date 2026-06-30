"""Tests for PlanFailureMemory — the cross-episode 'learn from death' loop.

Contract: a strategy that keeps failing for a class of goal becomes 'avoid'; one that
keeps working becomes 'prefer'; the lesson generalizes across similar-but-differently-
worded goals; and it persists across sessions (a fresh instance still knows).
"""
from __future__ import annotations

from core.planning.plan_failure_memory import AVOID_FAILURE_RATE, PlanFailureMemory


def test_repeated_failure_becomes_avoid(tmp_path):
    mem = PlanFailureMemory(db_path=tmp_path / "p.sqlite3")
    for _ in range(4):
        mem.record_outcome("open the browser and search the web", "use_fallback_app",
                           success=False, failure_mode="app not found")
    g = mem.guidance("open the browser and search the web")
    assert "use_fallback_app" in g.avoid
    assert g.failure_rates["use_fallback_app"] >= AVOID_FAILURE_RATE


def test_repeated_success_becomes_prefer(tmp_path):
    mem = PlanFailureMemory(db_path=tmp_path / "p.sqlite3")
    for _ in range(4):
        mem.record_outcome("write a file to disk", "retry_with_delay", success=True)
    g = mem.guidance("write a file to disk")
    assert "retry_with_delay" in g.prefer


def test_lesson_generalizes_across_similar_goals(tmp_path):
    """A failure on one wording cautions a differently-worded goal of the same class."""
    mem = PlanFailureMemory(db_path=tmp_path / "p.sqlite3")
    for _ in range(3):
        mem.record_outcome("open the browser and search", "launch_and_retry",
                           success=False, failure_mode="timeout")
    # A differently-worded goal that shares salient tokens (open/browser/search) inherits
    # the lesson via overlap matching — even though its full signature differs.
    g = mem.guidance("open a browser to search for something useful")
    assert "launch_and_retry" in g.avoid
    # An unrelated goal (no shared salient tokens) does NOT inherit it.
    assert "launch_and_retry" not in mem.guidance("cook dinner tonight").avoid


def test_single_failure_is_noise_not_a_lesson(tmp_path):
    mem = PlanFailureMemory(db_path=tmp_path / "p.sqlite3")
    mem.record_outcome("do a thing", "some_strategy", success=False)
    g = mem.guidance("do a thing")
    assert "some_strategy" not in g.avoid  # below MIN_ATTEMPTS → not yet a pattern


def test_persists_across_sessions(tmp_path):
    path = tmp_path / "p.sqlite3"
    mem = PlanFailureMemory(db_path=path)
    for _ in range(4):
        mem.record_outcome("compile the rust extension", "escalate_permission",
                           success=False, failure_mode="permission denied")
    # A fresh instance (new 'session') still knows the lesson — it learned across runs.
    reborn = PlanFailureMemory(db_path=path)
    assert "escalate_permission" in reborn.guidance("compile the rust extension").avoid


def test_caution_text_is_factual_readout(tmp_path):
    mem = PlanFailureMemory(db_path=tmp_path / "p.sqlite3")
    for _ in range(3):
        mem.record_outcome("send an email", "ask_user", success=False, failure_mode="no client")
    text = mem.caution_text("send an email")
    assert "Avoid" in text and "ask_user" in text
