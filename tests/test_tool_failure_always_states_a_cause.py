"""A tool failure must always state a cause (2026-07-18 soak).

The soak's live tool path produced `Task web_search failed: ` with an empty
message, and the intention loop banked
``{'status': 'failed', 'error': ''}`` at surprise 1.00 — a maximal
learning signal carrying zero information about what to do differently, and
nothing an operator (or Aura's own self-report) could act on.

Silence is not an acceptable failure surface anywhere on the live user
path. This pins the framework-level guarantee, which applies to every
skill at once: a failed skill result always names its cause, or says
plainly that the skill reported none.
"""
from __future__ import annotations

import asyncio

import pytest

from core.skills.base_skill import BaseSkill

pytestmark = pytest.mark.unit


class _Skill(BaseSkill):
    """Minimal skill whose return value each test controls."""

    name = "probe_skill"
    description = "test double"
    input_model = None

    def __init__(self, payload):
        super().__init__()
        self._payload = payload

    async def execute(self, params, context):  # type: ignore[override]
        return self._payload


def _run(payload) -> dict:
    return asyncio.run(_Skill(payload).safe_execute({}, {"source": "unit"}))


class TestFailuresNameTheirCause:
    def test_empty_error_string_is_replaced_with_a_diagnosable_cause(self):
        """The exact soak shape: status=failed with error=''."""
        result = _run({"status": "failed", "error": ""})
        assert result["ok"] is False
        assert result["error"].strip(), "a failure must never carry an empty cause"
        assert "probe_skill" in result["error"]

    def test_missing_error_key_is_filled(self):
        result = _run({"status": "failed"})
        assert result["ok"] is False
        assert result["error"].strip()
        assert "status=failed" in result["error"]

    def test_whitespace_only_error_is_treated_as_absent(self):
        result = _run({"ok": False, "error": "   \n "})
        assert result["error"].strip()
        assert "probe_skill" in result["error"]

    def test_reason_is_carried_into_the_stated_cause(self):
        result = _run({"status": "failed", "reason": "upstream_rate_limited"})
        assert "upstream_rate_limited" in result["error"]

    def test_a_real_cause_is_never_overwritten(self):
        result = _run({"ok": False, "error": "connection refused by search backend"})
        assert result["error"] == "connection refused by search backend"

    def test_success_results_are_untouched(self):
        result = _run({"ok": True, "content": "found 3 results"})
        assert result["ok"] is True
        assert "error" not in result or not result.get("error")

    def test_deferral_is_not_a_failure_and_needs_no_error(self):
        """A governed deferral is a decision, not a fault — it must not be
        given a manufactured error string."""
        result = _run({"status": "deferred", "reason": "foreground_chat_active"})
        assert not result.get("error")

    def test_error_result_helper_never_emits_an_empty_cause(self):
        skill = _Skill({"ok": True})
        assert skill._error_result("", 0.01)["error"].strip()
        assert skill._error_result("real cause", 0.01)["error"] == "real cause"


class TestSchedulerFailureIsDiagnosable:
    def test_bare_exception_still_names_its_origin(self):
        """A bare RuntimeError() renders as '' through %s — the scheduler
        must log an identity and a raise site instead of nothing."""
        from core.runtime.errors import describe_error

        described = describe_error(RuntimeError())
        assert "RuntimeError" in described
        assert described.strip() != "RuntimeError:"
        assert "no message" in described

    def test_scheduler_uses_the_describing_formatter(self):
        import inspect

        from core import scheduler

        source = inspect.getsource(scheduler)
        assert 'logger.error("Task %s failed: %s", spec.name, describe_error(e))' in source
