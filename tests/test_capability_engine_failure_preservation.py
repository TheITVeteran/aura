"""Capability engine must preserve structured skill failures.

Part-2 browser proof rounds 1-2 both surfaced as "Failed. Completed 0/0
steps" / "returned error ... unknown" because the retry loop flattened
every failing skill dict to {"ok": False, "error": "Unknown"}, erasing
the step receipts and the real cause. These pin the fix:

- _extract_error pulls the cause out of failures[0] (real step error)
  instead of the bare "Failed";
- _execute_with_retry returns the skill's structured payload merged
  with the failure flag, so reply doors keep the receipts.
"""
from __future__ import annotations

import pytest

from core.capability_engine import CapabilityEngine


def _bare_engine() -> CapabilityEngine:
    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.max_retries = 1
    engine.retry_delay = 0.0
    engine.timeout = 5.0
    return engine


def test_extract_error_reads_structured_step_failure():
    engine = _bare_engine()
    payload = {
        "ok": False,
        "summary": "Completed 0/0 steps. Failed.",
        "failures": [
            {
                "action": "hotkey",
                "ok": False,
                "error": "keystroke dispatch failed: AppleScript timed out after 8s.",
            }
        ],
    }
    msg = engine._extract_error(payload)
    assert "hotkey" in msg
    assert "AppleScript timed out" in msg
    assert msg != "Failed"


def test_extract_error_falls_back_to_summary_then_failed():
    engine = _bare_engine()
    assert engine._extract_error({"ok": False, "summary": "everything broke"}) == "everything broke"
    assert engine._extract_error({"ok": False}) == "Failed"
    assert engine._extract_error("not a dict") == "Error"


@pytest.mark.asyncio
async def test_retry_loop_preserves_structured_failure_payload():
    engine = _bare_engine()

    class FailingSkill:
        async def execute(self, params, context):
            return {
                "ok": False,
                "steps_completed": 0,
                "steps_requested": 10,
                "summary": "Completed 0/10 steps.",
                "failures": [
                    {"action": "set_wallpaper", "ok": False, "error": "read-back mismatch"}
                ],
                "receipts": [{"action": "open_url", "ok": True}],
            }

    result = await engine._execute_with_retry(
        FailingSkill(), "desktop_task", {"objective": "x"}, {}
    )
    # Failure flag present, but the structured evidence survived.
    assert result["ok"] is False
    assert result["steps_requested"] == 10
    assert result["receipts"] == [{"action": "open_url", "ok": True}]
    assert result["failures"][0]["action"] == "set_wallpaper"


@pytest.mark.asyncio
async def test_desktop_task_outer_retry_is_disabled_to_prevent_plan_replay():
    engine = _bare_engine()
    engine.max_retries = 3
    calls = []

    class FailingDesktopTask:
        async def execute(self, params, context):
            calls.append((dict(params), dict(context or {})))
            return {
                "ok": False,
                "status": "failed",
                "steps_completed": 2,
                "steps_requested": 3,
                "summary": "Desktop task completed 2/3 governed computer-use steps.",
                "failures": [
                    {
                        "action": "open_url",
                        "ok": False,
                        "effect_evidence": "late transient timeout",
                    }
                ],
                "receipts": [
                    {"action": "type", "ok": True},
                    {"action": "write_text_file", "ok": True},
                ],
            }

    result = await engine._execute_with_retry(
        FailingDesktopTask(),
        "desktop_task",
        {"objective": "do not replay this plan"},
        {"desktop_execution_contract": True},
    )

    assert result["ok"] is False
    assert result["steps_completed"] == 2
    assert result["retries"] == 0
    assert len(calls) == 1
