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

from typing import Optional

import pytest
from pydantic import BaseModel

import core.capability_engine as ce


class _RequiresPrompt(BaseModel):
    prompt: str
    style: str = "realistic"


class _TypedParams(BaseModel):
    name: str
    limit: int
    temperature: float
    debug: bool
    default_val: int = 100


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


@pytest.mark.asyncio
async def test_non_object_skill_params_never_escape_the_object_contract():
    meta = ce.SkillMetadata(name="object_only", description="test")

    result = await meta.extract_and_validate_args('["not", "an", "object"]', llm=None)

    assert result["raw_params"] == '["not", "an", "object"]'
    assert "must decode to a JSON object" in result["_error"]


@pytest.mark.asyncio
async def test_uncoercible_param_value_does_not_enforce_fail_closed(monkeypatch):
    """The coercion-failure site (limit='not_an_int') must opt out too — this
    was the missed sibling of the July fix that surfaced as an in-chunk
    order-dependence failure of test_sota_hardeners once any earlier test left
    capability_engine registered fail-closed."""
    captured: list[dict] = []

    def _capture(subsystem, error, *, severity="degraded", action="",
                 enforce_failure_policy=True, **_):
        captured.append({"subsystem": subsystem, "severity": severity,
                         "enforce_failure_policy": enforce_failure_policy})

    monkeypatch.setattr(ce, "record_degradation", _capture)
    meta = ce.SkillMetadata(name="typed", description="test", input_model=_TypedParams)

    result = await meta.extract_and_validate_args(
        '{"name": "Aura", "limit": "not_an_int", "temperature": "0.1", "debug": "yes"}',
        llm=None,
    )

    assert "_error" in result
    recovery = [c for c in captured if c["subsystem"] == "capability_engine"]
    assert recovery
    assert all(c["enforce_failure_policy"] is False for c in recovery), (
        "coercion failure on bad input must not trip the fail-closed policy"
    )


@pytest.mark.asyncio
async def test_uncoercible_param_does_not_raise_under_live_fail_closed(monkeypatch):
    """End-to-end proof of the order-dependence root: even when an earlier test
    has left capability_engine registered fail-closed under production
    governance, a bad param value returns the sanitized fallback instead of
    raising CRITICAL SERVICE FAILURE. This is what made test_sota_hardeners
    fail in-chunk but pass alone."""
    from core.runtime import mode as mode_mod
    import core.runtime.service_registry as sr

    monkeypatch.setattr(mode_mod, "get_mode", lambda: mode_mod.AuraMode.PRODUCTION)
    monkeypatch.setattr(sr, "get_service_failure_policy",
                        lambda name: "fail-closed" if name == "capability_engine" else None)

    meta = ce.SkillMetadata(name="typed", description="test", input_model=_TypedParams)
    result = await meta.extract_and_validate_args(
        '{"name": "Aura", "limit": "not_an_int", "temperature": "0.1", "debug": "yes"}',
        llm=None,
    )
    # returned gracefully, did not raise
    assert "_error" in result
    assert result.get("name") == "Aura"
