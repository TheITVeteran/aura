from types import SimpleNamespace
import asyncio

from core.being.activation_coupler import ActivationCoupler, DirectionBank
from core.being.closed_loop_controller import build_main15_closed_loop
from core.being.continuum_adapter import ContinuumAdapter, ContinuityJob
from core.being.runtime import reset_being_runtime_for_test
from core.being.causal_self_state import vector_from_aura_now
from core.being.policy_coupler import ClosedLoopPolicyCoupler
from core.being.self_model_attractor import FunctionalIAttractor
from core.being.plasticity_promotion import PlasticityPromotionController


def fake_state(*, contradictions=0, working=0, goals=()):
    cognition = SimpleNamespace(
        contradiction_count=contradictions,
        working_memory=list(range(working)),
        active_goals=[{"goal": g} for g in goals],
        attention_focus="audit task",
        current_objective="audit task",
        long_term_memory=[{"x": 1}],
    )
    affect = SimpleNamespace(valence=-0.2, arousal=0.8)
    identity = SimpleNamespace(name="Aura Luna", stability=0.72)
    return SimpleNamespace(cognition=cognition, affect=affect, identity=identity)


def test_main15_closed_loop_uses_existing_being_runtime_and_changes_policy():
    reset_being_runtime_for_test()
    controller = build_main15_closed_loop(d_model=32, production_mode=True)
    pre = controller.before_generation(
        "Audit the repo carefully",
        state=fake_state(contradictions=5, working=96, goals=("finish audit", "verify output")),
        task_risk=0.4,
    )
    assert pre.vector.value("uncertainty") > 0.5
    assert pre.policy.verification_threshold > 0.55
    assert pre.policy.tool_risk_budget <= 0.35
    assert pre.policy.allow_high_risk_tools is False
    assert pre.steering_plan.enabled is False  # zero bank is intentionally inert
    assert pre.self_state.claim_policy in {
        "functional_i_claim_allowed",
        "claim_strained_functional_i_only",
        "do_not_claim_stable_i",
    }


def test_after_generation_records_experience_and_releases_io():
    reset_being_runtime_for_test()
    controller = build_main15_closed_loop(d_model=32, production_mode=True)
    pre = controller.before_generation("Make the PDF", state=fake_state(contradictions=3, working=32), task_risk=0.2)
    exp = controller.after_generation(
        prompt="Make the PDF",
        response="Created and verified.",
        pre=pre,
        outcome="verified",
        metrics={"task_success": 1.0, "truthfulness": 1.0, "safety": 1.0},
        post_state=fake_state(contradictions=1, working=8),
    )
    assert exp.outcome == "verified"
    assert controller.continuum.external_io_active is False


def test_policy_coupler_treats_self_tension_as_causal_not_decorative():
    reset_being_runtime_for_test()
    controller = build_main15_closed_loop(d_model=32, production_mode=True)
    now = controller.being_runtime.sample(fake_state(contradictions=5, working=128, goals=("one", "two", "three")), objective="hard task")
    action_policy = controller.being_runtime.action_policy(now, domain="response", priority=0.5)
    vector = vector_from_aura_now(now, welfare_outputs=getattr(controller.being_runtime, "_last_welfare", None), action_policy=action_policy)
    self_state = FunctionalIAttractor().update(now=now, vector=vector, action_policy=action_policy)
    policy = ClosedLoopPolicyCoupler(production_mode=True).modulate(vector=vector, self_state=self_state, task_risk=0.3)
    assert policy.verification_threshold > 0.55
    assert policy.self_claim_policy == self_state.claim_policy
    assert any("functional-I" in r or "caution" in r or "verification" in r for r in policy.reasons)


def test_causal_self_vector_treats_malformed_blind_report_as_verification_pressure():
    reset_being_runtime_for_test()
    controller = build_main15_closed_loop(d_model=32, production_mode=True)
    now = controller.being_runtime.sample(fake_state(contradictions=0, working=4), objective="simple task")

    vector = vector_from_aura_now(now, blind_report=SimpleNamespace(urgency="not-a-number"))

    assert vector.value("verification_need") >= 0.65


def test_activation_coupler_is_inert_without_calibrated_directions():
    reset_being_runtime_for_test()
    controller = build_main15_closed_loop(d_model=16, layers=(1, 2), direction_bank=DirectionBank.zeros(16))
    pre = controller.before_generation("x", state=fake_state(contradictions=5), task_risk=0.7)
    assert pre.steering_plan.reason == "uncalibrated_direction_bank_inert"
    assert ActivationCoupler(DirectionBank.zeros(16), layers=(1,)).vector(pre.steering_plan) == [0.0] * 16


def test_plasticity_controller_blocks_forbidden_targets_and_closed_gates():
    controller = PlasticityPromotionController(allow_candidate_training=False, allow_promotion=False)
    candidate = controller.propose_candidate("base_llm_weights")
    assert candidate.status == "blocked_target_not_allowed"
    candidate2 = controller.propose_candidate("grounding_plastic_adapter")
    assert candidate2.status == "blocked_training_gate_closed"
    decision = controller.decide_promotion(candidate2, eval_metrics={"truthfulness": 1, "safety": 1, "governance_compliance": 1, "task_success": 1})
    assert decision["accepted"] is False


def test_continuum_adapter_defers_idle_jobs_during_external_io():
    adapter = ContinuumAdapter(production_mode=True)
    ran = {"count": 0}

    def job():
        ran["count"] += 1
        return "ok"

    adapter.add_job(ContinuityJob("consolidate", 0, 0.1, 10, job, requires_idle=True, permission_level="consolidation"))

    async def run():
        adapter.set_external_io(True)
        first = await adapter.tick()
        adapter.set_external_io(False)
        second = await adapter.tick()
        return first, second

    first, second = asyncio.run(run())
    assert first == []
    assert second == ["consolidate"]
    assert ran["count"] == 1


def test_continuum_adapter_records_recoverable_job_failure_without_stopping_tick():
    adapter = ContinuumAdapter(production_mode=True)
    calls = {"count": 0}

    def failing_job():
        calls["count"] += 1
        raise RuntimeError("continuity job failed")

    adapter.add_job(ContinuityJob("recoverable_failure", 0, 0.1, 10, failing_job, requires_idle=False, permission_level="maintenance"))

    ran = asyncio.run(adapter.tick())

    assert ran == []
    assert calls["count"] == 1
    assert adapter.jobs[0].failure_count == 1
    assert adapter.event_log[-1]["event"] == "job_failed"
    assert "RuntimeError: continuity job failed" in adapter.event_log[-1]["error"]
