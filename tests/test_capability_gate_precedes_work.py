"""tests/test_capability_gate_precedes_work.py — authorization runs before work.

``CapabilityEngine._prepare_skill_instance`` imports the skill's module and runs
its constructor. For a period that preparation ran BEFORE the permission model,
before the constitutional Will/AuthorityGateway closure, and before the derived
conscience gates. Two consequences, both real:

1.  Naming a skill was enough to execute its module-level code and ``__init__``
    without any authority having approved anything. Only the skill's *call* was
    gated, so "blocked" still meant "your code ran".
2.  The gates' verdicts were masked. A runtime whose executive core was down
    answered ``skill_preflight_failed`` — an authorization outage reported as a
    broken skill, which is the wrong incident, routed to the wrong owner.

Ordering is the property, so ordering is what these tests assert. They observe
the sequence directly rather than inspecting source, because the defect is not
that some line moved — it is that work became reachable before consent.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.capability_engine import CapabilityEngine, SkillMetadata


def _engine_with_recording_preflight(recorder: list[str]) -> CapabilityEngine:
    """A CapabilityEngine whose preflight announces itself when it runs."""
    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    engine.error_boundary = lambda fn: fn
    engine.skills = {
        "clock": SkillMetadata(
            name="clock",
            description="Return the time.",
            skill_class=lambda: object(),
        )
    }
    engine.instances = {}
    engine.sandbox = None
    engine.rosetta_stone = None
    engine.temporal = None
    engine.orchestrator = SimpleNamespace(mycelium=None)
    engine.skill_last_errors = {}
    engine._emit_skill_status = lambda *a, **k: None

    def _recording_prepare(skill_name, meta):
        recorder.append("prepared")
        return {"ok": True, "stage": "constructor"}, object()

    engine._prepare_skill_instance = _recording_prepare
    return engine


@pytest.mark.asyncio
async def test_executive_gate_failure_is_reported_as_itself(monkeypatch, service_container):
    """An authorization outage must not be reported as a skill defect.

    ``skill_preflight_failed`` sends an operator to look at the skill. The skill
    is fine; the executive core is down. Naming the real condition is what makes
    the incident actionable.
    """
    service_container.lock_registration()
    recorder: list[str] = []
    engine = _engine_with_recording_preflight(recorder)

    monkeypatch.setattr(
        "core.executive.executive_core.get_executive_core",
        lambda: (_ for _ in ()).throw(RuntimeError("down")),
    )

    result = await CapabilityEngine.execute(engine, "clock", {}, context={})

    assert result["ok"] is False
    assert result["status"] == "blocked_by_executive_gate_failure", (
        f"authorization outage reported as {result['status']!r}"
    )


@pytest.mark.asyncio
async def test_a_blocked_skill_is_never_constructed(monkeypatch, service_container):
    """The load-bearing assertion: refused work must not have already happened.

    Preparing the skill imports its module and runs its constructor. If that
    happens before the gate refuses, the refusal is a report about something
    that already ran.
    """
    service_container.lock_registration()
    recorder: list[str] = []
    engine = _engine_with_recording_preflight(recorder)

    monkeypatch.setattr(
        "core.executive.executive_core.get_executive_core",
        lambda: (_ for _ in ()).throw(RuntimeError("down")),
    )

    result = await CapabilityEngine.execute(engine, "clock", {}, context={})

    assert result["ok"] is False
    assert recorder == [], (
        "the skill was imported and constructed even though the authority gate "
        "refused the call — the gate came after the work it was gating"
    )
