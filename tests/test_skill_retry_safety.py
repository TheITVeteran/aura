"""safe_execute must never re-run a side-effectful skill's effects.

The retry-on-transient loop is correct for read-only/idempotent skills but a
double-send/double-delete bug for side-effectful ones: a skill that performs
its effect then hits a transient error would re-execute. A skill is retried
only when retry_safe AND not requires_approval.
"""
from __future__ import annotations

import pytest

from core.skills.base_skill import BaseSkill


class _CountingSkill(BaseSkill):
    name = "counting_test_skill"
    description = "counts executions; always raises a transient error"

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def execute(self, params, context):
        self.calls += 1
        raise TimeoutError("transient")  # in _TRANSIENT set → would retry


class _IdempotentSkill(_CountingSkill):
    name = "idempotent_test_skill"
    retry_safe = True
    requires_approval = False


class _SideEffectSkill(_CountingSkill):
    name = "side_effect_test_skill"
    retry_safe = False
    requires_approval = False


class _DestructiveSkill(_CountingSkill):
    name = "destructive_test_skill"
    retry_safe = True  # even opted-in, approval-required must not double-fire
    requires_approval = True


@pytest.mark.asyncio
async def test_idempotent_skill_retries_transient():
    s = _IdempotentSkill()
    out = await s.safe_execute({})
    assert out["ok"] is False
    assert s.calls == 3, "read-only/idempotent skills should retry transient errors"


@pytest.mark.asyncio
async def test_side_effect_skill_runs_once():
    s = _SideEffectSkill()
    out = await s.safe_execute({})
    assert out["ok"] is False
    assert s.calls == 1, "retry_safe=False must execute exactly once — no double side effect"


@pytest.mark.asyncio
async def test_approval_required_skill_never_double_fires():
    s = _DestructiveSkill()
    out = await s.safe_execute({})
    assert out["ok"] is False
    assert s.calls == 1, "requires_approval must never retry, even with retry_safe=True"


@pytest.mark.asyncio
async def test_policy_deferred_skill_does_not_increment_failure_metrics():
    class DeferredSkill(BaseSkill):
        name = "deferred_test_skill"

        async def execute(self, params, context):
            return {
                "ok": False,
                "status": "deferred",
                "reason": "foreground_generation_active",
            }

    skill = DeferredSkill()
    result = await skill.safe_execute({})

    assert result["ok"] is False
    assert result["status"] == "deferred"
    assert skill.get_stats()["executions"] == 1
    assert skill.get_stats()["failures"] == 0


def test_send_skills_are_marked_retry_unsafe():
    """The external-communication skills must not silently double-send."""
    from core.skills.notify_user import NotifyUserSkill
    from core.skills.uplink_local import UplinkSkill

    assert NotifyUserSkill.retry_safe is False
    assert UplinkSkill.retry_safe is False


def test_base_default_requires_explicit_retry_safety_classification():
    assert BaseSkill.retry_safe is False


def test_infer_ok_flag_honors_success_key():
    """A skill reporting {'success': False} must not be marked ok=True."""
    from core.skills.base_skill import _infer_ok_flag

    assert _infer_ok_flag({"success": False, "message": "corpus empty"}) is False
    assert _infer_ok_flag({"success": True, "results": [1]}) is True
    # Contradictory negative evidence must prevent a dishonest success.
    assert _infer_ok_flag({"ok": True, "success": False}) is False
    assert _infer_ok_flag({"ok": True, "error": "boom"}) is False
    assert _infer_ok_flag({"ok": True, "status": "failed"}) is False
    assert _infer_ok_flag({"error": "boom"}) is False
    assert _infer_ok_flag({"status": "failed"}) is False
    # a plain successful payload stays ok
    assert _infer_ok_flag({"results": [1, 2, 3]}) is True
    # an EMPTY error/errors field is the absence of an error, not evidence of
    # one — a skill that populates error=stderr succeeds with "" on a clean run.
    assert _infer_ok_flag({"ok": True, "error": ""}) is True
    assert _infer_ok_flag({"ok": True, "error": None, "errors": []}) is True
    assert _infer_ok_flag({"ok": True, "output": "1 passed", "error": ""}) is True
