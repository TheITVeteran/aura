"""Bad skill input must not mint a fail-closed CRITICAL incident.

Live (July 2026): a turn was dispatched to image_gen with no `prompt`. The
schema-recovery path in capability_engine recorded the ValidationError as a
degradation, and because capability_engine is fail-closed, record_degradation
escalated the warning to CRITICAL — spiking existential threat and creating
INC-…-0001. But the engine did its job: it correctly rejected malformed
classifier input. That is not a subsystem failure and must not enforce the
fail-closed policy.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

import core.capability_engine as ce


class _RequiresPrompt(BaseModel):
    prompt: str
    style: str = "realistic"


@pytest.mark.asyncio
async def test_unfillable_skill_params_do_not_enforce_fail_closed(monkeypatch):
    captured: list[dict] = []

    def _capture(subsystem, error, *, severity="degraded", action="", enforce_failure_policy=True, **_):
        captured.append(
            {
                "subsystem": subsystem,
                "severity": severity,
                "enforce_failure_policy": enforce_failure_policy,
            }
        )

    monkeypatch.setattr(ce, "record_degradation", _capture)

    meta = ce.SkillMetadata(
        name="image_gen",
        description="test",
        input_model=_RequiresPrompt,
    )

    # Classifier produced params with the required `prompt` missing entirely.
    result = await meta.extract_and_validate_args('{"style": "anime"}', llm=None)

    # The engine returns a sanitized fallback carrying the validation error,
    # rather than raising — the turn degrades gracefully.
    assert "_error" in result

    # Exactly one degradation was recorded, and it did NOT enforce the
    # fail-closed policy (so it stays a warning, never a CRITICAL incident).
    assert captured, "the rejection should still be recorded as a signal"
    recovery = [c for c in captured if c["subsystem"] == "capability_engine"]
    assert recovery, "capability_engine should record the input rejection"
    assert all(c["enforce_failure_policy"] is False for c in recovery), (
        "rejecting bad classifier input must not trip the fail-closed policy"
    )
    assert all(c["severity"] == "warning" for c in recovery)
