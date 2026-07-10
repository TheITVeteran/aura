"""Contracts for A3 (formal degradation ladder) and K5 (disruption budget).

A3: the cortex → brainstem → reflex → cloud → salvage ladder is declared
in core/brain/degradation_ladder.py; these tests pin the declaration to
the inference gate's actual behavior surface so drift fails the build.

K5: voluntary disruptions never remove the last warm lane.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.degradation_ladder import (
    CLOUD_ENDPOINT,
    DEGRADATION_LADDER,
    SALVAGE_STAGE,
    ladder_order,
    ladder_report,
    rung_for_endpoint,
    rungs_below,
)
from core.brain.llm.model_registry import (
    BRAINSTEM_ENDPOINT,
    FALLBACK_ENDPOINT,
    PRIMARY_ENDPOINT,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_SOURCE = (REPO_ROOT / "core" / "brain" / "inference_gate.py").read_text(
    encoding="utf-8"
)


class TestLadderDeclaration:
    def test_canonical_order(self):
        assert ladder_order() == (
            PRIMARY_ENDPOINT,
            BRAINSTEM_ENDPOINT,
            FALLBACK_ENDPOINT,
            CLOUD_ENDPOINT,
            SALVAGE_STAGE,
        )

    def test_every_generation_rung_has_a_first_token_sla(self):
        for rung in DEGRADATION_LADDER:
            if rung.endpoint != SALVAGE_STAGE:
                assert rung.first_token_sla_s and rung.first_token_sla_s > 0, rung.name

    def test_local_rungs_have_cold_start_budgets(self):
        for rung in DEGRADATION_LADDER:
            if rung.local and rung.endpoint != SALVAGE_STAGE:
                assert rung.cold_start_sla_s and rung.cold_start_sla_s > 0, rung.name

    def test_slas_tighten_down_the_local_ladder(self):
        """A lower rung exists because the one above was too slow or dead —
        it must come up faster and answer faster."""
        local = [
            r
            for r in DEGRADATION_LADDER
            if r.local and r.endpoint != SALVAGE_STAGE
        ]
        for higher, lower in zip(local, local[1:]):
            assert lower.first_token_sla_s < higher.first_token_sla_s
            assert lower.cold_start_sla_s < higher.cold_start_sla_s

    def test_lookup_and_below(self):
        assert rung_for_endpoint("cortex").name == "primary_cortex"
        below = [r.name for r in rungs_below(PRIMARY_ENDPOINT)]
        assert below == ["brainstem", "reflex", "cloud", "salvage"]
        assert rungs_below("unknown") == ()

    def test_report_is_serializable(self):
        report = ladder_report()
        assert len(report) == len(DEGRADATION_LADDER)
        assert report[0]["endpoint"] == PRIMARY_ENDPOINT


class TestGateExecutesTheLadder:
    """Source-level pins: the executor must keep every declared stage."""

    def test_gate_keeps_the_emergency_reflex_stage(self):
        assert "EMERGENCY REFLEX FALLBACK" in GATE_SOURCE, (
            "the declared reflex rung lost its executor stage in inference_gate"
        )

    def test_gate_keeps_the_cloud_stage_after_local(self):
        assert "continued to configured cloud or exhaustion path" in GATE_SOURCE, (
            "the declared cloud rung lost its executor stage in inference_gate"
        )

    def test_gate_falls_back_from_primary_to_brainstem(self):
        assert "fallback_label = BRAINSTEM_ENDPOINT" in GATE_SOURCE, (
            "primary tier no longer names the brainstem as its fallback lane"
        )


class TestDisruptionBudget:
    def test_last_warm_lane_is_protected_from_eviction(self, monkeypatch):
        import asyncio

        from core.brain.lane_admission import ActiveLane, QoSClass
        from core.runtime.lane_reconciler import (
            CrashLoopBreaker,
            LaneReconciler,
            disruption_budget_blocks,
        )

        monkeypatch.setenv("AURA_LANE_BUDGET_GB", "10")
        # One lane over budget — but it is the ONLY warm lane.
        lanes = [
            ActiveLane("trainer", QoSClass.BEST_EFFORT, 12.0, model_path="/m/trainer"),
        ]
        assert disruption_budget_blocks("/m/trainer", lanes) is not None

        evicted = []

        async def evict(path):
            evicted.append(path)
            return True

        async def spawn():
            return True

        rec = LaneReconciler(
            observe_lanes=lambda: lanes,
            primary_alive=lambda: True,
            primary_key=lambda: "/m/cortex",
            primary_age_s=lambda: 0.0,
            spawn_primary=spawn,
            evict_lane=evict,
            foreground_active=lambda: False,
            breaker=CrashLoopBreaker(),
        )
        actions = asyncio.run(rec.reconcile_once())
        assert evicted == [], "the last warm lane must never be voluntarily evicted"
        assert any(
            a["action"] == "held" and "disruption_budget" in a["detail"]
            for a in actions
        )

    def test_budget_allows_eviction_when_another_lane_is_warm(self):
        from core.brain.lane_admission import ActiveLane, QoSClass
        from core.runtime.lane_reconciler import disruption_budget_blocks

        lanes = [
            ActiveLane("cortex", QoSClass.GUARANTEED, 20.0, model_path="/m/cortex"),
            ActiveLane("trainer", QoSClass.BEST_EFFORT, 12.0, model_path="/m/trainer"),
        ]
        assert disruption_budget_blocks("/m/trainer", lanes) is None

    def test_mlx_last_warm_lane_helper(self, monkeypatch):
        from core.brain.llm import mlx_client as mc

        class _Fake:
            def __init__(self, alive):
                self._alive = alive

            def is_alive(self):
                return self._alive

        only = _Fake(True)
        monkeypatch.setattr(mc, "_CLIENTS", {"/m/only": only})
        assert mc._lane_is_last_warm(only) is True

        other = _Fake(True)
        monkeypatch.setattr(mc, "_CLIENTS", {"/m/only": only, "/m/other": other})
        assert mc._lane_is_last_warm(only) is False

        dead = _Fake(False)
        monkeypatch.setattr(mc, "_CLIENTS", {"/m/only": only, "/m/dead": dead})
        assert mc._lane_is_last_warm(only) is True, "dead lanes do not count as warm"
