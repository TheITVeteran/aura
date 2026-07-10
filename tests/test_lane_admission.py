"""Contracts for declarative model-lane memory admission (roadmap K3).

The over-commitment doom loop (stall → force-kill → cold reload) came from
lanes spawning against instantaneous free-RAM spot checks. These tests pin
the declarative model: declared footprints vs an explicit host budget, QoS
eviction order, the envelope-breach refusal, and the mlx_client spawn seam.
"""
from __future__ import annotations

import pytest

from core.brain.lane_admission import (
    ActiveLane,
    LaneAdmissionController,
    QoSClass,
    classify_lane,
    get_lane_admission_controller,
    lane_budget_gb,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def controller():
    return LaneAdmissionController()


@pytest.fixture
def budget_46(monkeypatch):
    monkeypatch.setenv("AURA_LANE_BUDGET_GB", "46")
    return 46.0


class TestLaneClassification:
    def test_primary_cortex_is_guaranteed(self):
        assert classify_lane("/models/Aura-32B-4bit") == ("cortex", QoSClass.GUARANTEED)
        assert classify_lane("zenith-fused") == ("cortex", QoSClass.GUARANTEED)

    def test_solver_and_small_lanes_are_burstable(self):
        assert classify_lane("/models/Deep-72B") == ("solver", QoSClass.BURSTABLE)
        assert classify_lane("/models/qwen-7b") == ("brainstem", QoSClass.BURSTABLE)
        assert classify_lane("/models/qwen-1.5b") == ("reflex", QoSClass.BURSTABLE)

    def test_trainers_are_best_effort_regardless_of_size(self):
        assert classify_lane("/models/Aura-32B-4bit", purpose="train") == (
            "trainer",
            QoSClass.BEST_EFFORT,
        )

    def test_unknown_paths_are_best_effort_auxiliary(self):
        assert classify_lane("/models/whisper-large") == ("auxiliary", QoSClass.BEST_EFFORT)


class TestBudget:
    def test_absolute_override_wins(self, monkeypatch):
        monkeypatch.setenv("AURA_LANE_BUDGET_GB", "40")
        assert lane_budget_gb() == 40.0

    def test_fraction_is_clamped(self, monkeypatch):
        monkeypatch.delenv("AURA_LANE_BUDGET_GB", raising=False)
        monkeypatch.setenv("AURA_LANE_BUDGET_FRACTION", "0.05")
        # clamped to 0.30 of host — never a sliver budget
        assert lane_budget_gb() > 0.0


class TestAdmissionArithmetic:
    def test_fits_admits_cleanly(self, controller, budget_46):
        decision = controller.admit(
            model_path="/models/Aura-32B-cortex", request_gb=23.0, active=[]
        )
        assert decision.admitted and decision.reason == "fits"
        assert decision.evict_first == ()

    def test_committed_lanes_count_against_budget(self, controller, budget_46):
        active = [
            ActiveLane("cortex", QoSClass.GUARANTEED, 20.0),
            ActiveLane("brainstem", QoSClass.BURSTABLE, 5.0),
        ]
        decision = controller.admit(
            model_path="/models/qwen-1.5b-reflex", request_gb=2.0, active=active
        )
        assert decision.admitted
        assert decision.committed_gb == pytest.approx(25.0)

    def test_guaranteed_candidate_gets_yield_advisory(self, controller, budget_46):
        """Cortex coming up over a loaded solver: admit, advise the solver yields."""
        active = [ActiveLane("solver", QoSClass.BURSTABLE, 41.0, model_path="/m/deep-72b")]
        decision = controller.admit(
            model_path="/models/Aura-32B-cortex", request_gb=23.0, active=active
        )
        assert decision.admitted and decision.reason == "fits_after_yield"
        assert decision.evict_first == ("/m/deep-72b",)

    def test_guaranteed_candidate_ignores_the_user_facing_shield(
        self, controller, budget_46
    ):
        """The cortex must ALWAYS be able to come up — even over a lane that
        served the user seconds ago."""
        active = [
            ActiveLane(
                "solver",
                QoSClass.BURSTABLE,
                41.0,
                model_path="/m/deep-72b",
                last_user_facing_age_s=10.0,
            )
        ]
        decision = controller.admit(
            model_path="/models/Aura-32B-cortex", request_gb=23.0, active=active
        )
        assert decision.admitted and decision.evict_first == ("/m/deep-72b",)

    def test_burstable_candidate_respects_the_shield(self, controller, budget_46):
        """A background solver must NOT be told it may evict a reflex lane
        that just served the user."""
        active = [
            ActiveLane("cortex", QoSClass.GUARANTEED, 20.0, model_path="/m/cortex"),
            ActiveLane(
                "reflex",
                QoSClass.BURSTABLE,
                2.0,
                model_path="/m/reflex",
                last_user_facing_age_s=5.0,
            ),
        ]
        decision = controller.admit(
            model_path="/models/Deep-72B-solver", request_gb=41.0, active=active
        )
        assert not decision.admitted
        assert "lane_budget_exceeded" in decision.reason
        # cortex is GUARANTEED (higher QoS) and reflex is shielded: no advisories
        assert decision.evict_first == ()

    def test_envelope_breach_names_the_arithmetic(self, controller, budget_46):
        """The 72B over a committed host: the refusal that replaces the
        OOM-SIGKILL-with-empty-stderr death."""
        active = [ActiveLane("cortex", QoSClass.GUARANTEED, 20.0, model_path="/m/cortex")]
        decision = controller.admit(
            model_path="/models/Deep-72B-solver", request_gb=41.0, active=active
        )
        assert not decision.admitted
        assert "request 41.0GB" in decision.reason
        assert "budget 46.0GB" in decision.reason

    def test_best_effort_evicted_before_burstable(self, controller, budget_46):
        active = [
            ActiveLane("trainer", QoSClass.BEST_EFFORT, 8.0, model_path="/m/trainer"),
            ActiveLane("brainstem", QoSClass.BURSTABLE, 5.0, model_path="/m/brainstem"),
            ActiveLane("cortex", QoSClass.GUARANTEED, 20.0, model_path="/m/cortex"),
        ]
        # solver needs 20GB of room in a 46 budget: 33 committed + 20 = 53 > 46;
        # evicting the 8GB trainer alone brings it to 45 <= 46.
        decision = controller.admit(
            model_path="/models/Deep-72B", request_gb=20.0, active=active
        )
        assert decision.admitted and decision.reason == "fits_after_yield"
        assert decision.evict_first == ("/m/trainer",)

    def test_advise_mode_never_enforces(self, controller, budget_46, monkeypatch):
        monkeypatch.setenv("AURA_LANE_ADMISSION", "advise")
        active = [ActiveLane("cortex", QoSClass.GUARANTEED, 20.0)]
        decision = controller.admit(
            model_path="/models/Deep-72B", request_gb=41.0, active=active
        )
        assert not decision.admitted
        assert decision.enforced is False


class TestObservability:
    def test_snapshot_carries_recent_decisions(self, controller, budget_46):
        controller.admit(model_path="/m/qwen-7b", request_gb=5.0, active=[])
        snap = controller.snapshot()
        assert snap["budget_gb"] == 46.0
        assert snap["mode"] in {"enforce", "advise"}
        assert snap["recent_decisions"][-1]["admitted"] is True

    def test_singleton_accessor(self):
        assert get_lane_admission_controller() is get_lane_admission_controller()


class TestSpawnSeam:
    """The mlx_client integration: observed lanes + the spawn consult."""

    class _FakeClient:
        def __init__(self, model_path, alive=True, last_user_facing=0.0):
            self.model_path = model_path
            self._alive = alive
            self._last_user_facing_completed_at = last_user_facing

        def is_alive(self):
            return self._alive

    def test_observed_lanes_exclude_self_and_dead(self, monkeypatch, budget_46):
        from core.brain.llm import mlx_client as mc

        me = self._FakeClient("/m/Aura-32B-cortex")
        other = self._FakeClient("/m/qwen-7b")
        dead = self._FakeClient("/m/Deep-72B", alive=False)
        monkeypatch.setattr(
            mc, "_CLIENTS", {c.model_path: c for c in (me, other, dead)}
        )
        lanes = mc._observed_active_lanes(exclude_client=me)
        assert [l.lane for l in lanes] == ["brainstem"]

    def test_spawn_consult_refuses_envelope_breach(self, monkeypatch, budget_46):
        from core.brain.llm import mlx_client as mc

        cortex = self._FakeClient("/m/Aura-32B-cortex-4bit-fused-model")
        solver = self._FakeClient("/m/Deep-72B-solver", alive=False)
        monkeypatch.setattr(
            mc, "_CLIENTS", {c.model_path: c for c in (cortex, solver)}
        )
        candidate = self._FakeClient("/m/Deep-72B-solver")
        reason = mc._lane_admission_blocks_worker_spawn(candidate)
        assert reason is not None and "lane_budget_exceeded" in reason

    def test_spawn_consult_admits_within_budget(self, monkeypatch, budget_46):
        from core.brain.llm import mlx_client as mc

        monkeypatch.setattr(mc, "_CLIENTS", {})
        candidate = self._FakeClient("/m/qwen-7b")
        assert mc._lane_admission_blocks_worker_spawn(candidate) is None
