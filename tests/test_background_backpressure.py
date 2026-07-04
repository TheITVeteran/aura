"""Background timeouts under foreground load are yields, not incidents.

Observed live (July 2026): a chat turn landing mid-consolidation minted
CRITICAL incidents from plain TimeoutErrors — INC-1783068731-0001
(sovereign_pruner) and INC-1783068780-0002 (dialectical_crucible) — and
spiked existential threat to 0.80. Both subsystems are fail-closed, so
any warning+ degradation escalates to critical; the fix is the shared
backpressure discipline in core/runtime/backpressure.py.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import core.runtime.backpressure as bp


@pytest.fixture(autouse=True)
def _fresh_state():
    bp.reset_backpressure_state()
    yield
    bp.reset_backpressure_state()


@pytest.fixture()
def recorded(monkeypatch):
    calls: list[dict] = []

    def _capture(subsystem, error, severity="degraded", action="", **kwargs):
        calls.append({"subsystem": subsystem, "severity": severity, "action": action})

    monkeypatch.setattr(bp, "record_degradation", _capture)
    return calls


class TestDiscipline:
    def test_yields_under_foreground_load_until_persistent(self, monkeypatch, recorded):
        monkeypatch.setattr(bp, "foreground_inference_active", lambda: True)
        exc = TimeoutError("model busy")
        outcomes = [
            bp.record_expected_backpressure("sovereign_pruner", exc, action="retry next pass")
            for _ in range(3)
        ]
        assert outcomes == ["yielded", "yielded", "escalated"]
        assert len(recorded) == 1
        assert recorded[0]["severity"] == "warning"
        assert "persistent" in recorded[0]["action"]

    def test_idle_foreground_escalates_immediately(self, monkeypatch, recorded):
        monkeypatch.setattr(bp, "foreground_inference_active", lambda: False)
        outcome = bp.record_expected_backpressure(
            "sovereign_pruner", TimeoutError(), action="retry next pass"
        )
        assert outcome == "escalated"
        assert len(recorded) == 1
        assert "unexplained" in recorded[0]["action"]

    def test_success_resets_the_streak(self, monkeypatch, recorded):
        monkeypatch.setattr(bp, "foreground_inference_active", lambda: True)
        exc = TimeoutError()
        bp.record_expected_backpressure("crucible", exc, action="a")
        bp.record_expected_backpressure("crucible", exc, action="a")
        bp.clear_backpressure("crucible")
        outcome = bp.record_expected_backpressure("crucible", exc, action="a")
        assert outcome == "yielded"
        assert recorded == []


class TestCrucibleYields:
    @pytest.mark.asyncio
    async def test_stage_yields_before_generating_under_foreground(self, monkeypatch, recorded):
        from core.adaptation.dialectics import DialecticalCrucible

        monkeypatch.setattr(bp, "foreground_inference_active", lambda: True)
        engine_calls = []

        async def _think(**kwargs):
            engine_calls.append(kwargs)
            return SimpleNamespace(content="should not run")

        import core.adaptation.dialectics as dialectics_mod

        monkeypatch.setattr(
            dialectics_mod.ServiceContainer,
            "get",
            staticmethod(
                lambda name, default=None: SimpleNamespace(think=_think)
                if name == "cognitive_engine"
                else default
            ),
        )
        crucible = DialecticalCrucible(stage_timeout_s=1.0)
        result = await crucible._think_stage(
            stage="antithesis", prompt="p", mode=None, priority=0.4, concept="c"
        )
        assert result is None
        assert engine_calls == [], "must yield before touching the engine"
        assert recorded == [], "yielding is not a degradation"

    @pytest.mark.asyncio
    async def test_mid_stage_timeout_is_backpressure_not_critical(self, monkeypatch, recorded):
        from core.adaptation.dialectics import DialecticalCrucible

        # Foreground idle at entry, busy by the time the timeout lands —
        # a user turn arrived mid-stage (the live incident shape).
        checks = iter([False, True, True, True])
        monkeypatch.setattr(bp, "foreground_inference_active", lambda: next(checks, True))

        async def _slow_think(**kwargs):
            await asyncio.sleep(5.0)

        import core.adaptation.dialectics as dialectics_mod

        monkeypatch.setattr(
            dialectics_mod.ServiceContainer,
            "get",
            staticmethod(
                lambda name, default=None: SimpleNamespace(think=_slow_think)
                if name == "cognitive_engine"
                else default
            ),
        )
        crucible = DialecticalCrucible(stage_timeout_s=0.05)
        result = await crucible._think_stage(
            stage="antithesis", prompt="p", mode=None, priority=0.4, concept="c"
        )
        assert result is None
        assert recorded == [], (
            "first mid-stage timeout under foreground load must yield, "
            f"not degrade: {recorded}"
        )


class TestPrunerYields:
    @pytest.mark.asyncio
    async def test_consolidate_yields_under_foreground(self, monkeypatch, recorded):
        from core.memory.sovereign_pruner import SovereignPruner

        monkeypatch.setattr(bp, "foreground_inference_active", lambda: True)
        pruner = SovereignPruner.__new__(SovereignPruner)
        pruner.orchestrator = SimpleNamespace(
            cognitive_engine=SimpleNamespace(think=None)
        )
        mem = SimpleNamespace(id="abcdef1234", content="c", source="s")
        result = await pruner._consolidate(mem)
        assert result == "c"
        assert recorded == []
