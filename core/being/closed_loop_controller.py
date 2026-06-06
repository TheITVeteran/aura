"""Main-15 integrated closed-loop controller.

This is the v3 integration point. It uses the existing BeingRuntime/AuraNow
instead of building a parallel self system.

Use:
    controller = build_main15_closed_loop(...)
    pre = controller.before_generation(prompt, state=aura_state, objective=...)
    response = mlx_client.generate(prompt, **pre.policy)
    controller.after_generation(...)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

from core.being.activation_coupler import ActivationCoupler, DirectionBank, SteeringPlan
from core.being.causal_self_state import CausalSelfVector, vector_from_aura_now
from core.being.continuum_adapter import ContinuumAdapter, install_default_continuity_jobs
from core.being.policy_coupler import ClosedLoopPolicy, ClosedLoopPolicyCoupler
from core.being.plasticity_promotion import ClosedLoopExperience, PlasticityPromotionController
from core.being.self_model_attractor import FunctionalIAttractor, SelfAttractorState


@dataclass(frozen=True)
class ClosedLoopPreGeneration:
    generation_id: str
    now: Any
    action_policy: dict[str, Any]
    vector: CausalSelfVector
    self_state: SelfAttractorState
    policy: ClosedLoopPolicy
    steering_plan: SteeringPlan
    created_at: float


class Main15ClosedLoopController:
    def __init__(
        self,
        *,
        being_runtime: Any,
        policy_coupler: ClosedLoopPolicyCoupler,
        attractor: FunctionalIAttractor,
        activation: ActivationCoupler,
        plasticity: PlasticityPromotionController,
        continuum: ContinuumAdapter,
    ) -> None:
        self.being_runtime = being_runtime
        self.policy_coupler = policy_coupler
        self.attractor = attractor
        self.activation = activation
        self.plasticity = plasticity
        self.continuum = continuum
        self._counter = 0

    def before_generation(
        self,
        prompt: str,
        *,
        state: Any | None = None,
        objective: str = "",
        task_risk: float = 0.0,
        domain: str = "response",
    ) -> ClosedLoopPreGeneration:
        self._counter += 1
        generation_id = f"being-v3-gen-{self._counter}"
        objective = objective or str(prompt)[:240]
        self.continuum.set_external_io(True)

        now = self.being_runtime.sample(state, objective=objective)
        action_policy = self.being_runtime.action_policy(now, domain=domain, priority=max(0.1, float(task_risk)))
        welfare = getattr(self.being_runtime, "_last_welfare", None)
        blind = getattr(self.being_runtime, "_last_blind_report", None)

        vector = vector_from_aura_now(now, welfare_outputs=welfare, blind_report=blind, action_policy=action_policy)
        self_state = self.attractor.update(now=now, vector=vector, action_policy=action_policy)
        policy = self.policy_coupler.modulate(
            vector=vector,
            self_state=self_state,
            task_risk=task_risk,
            action_policy=action_policy,
        )
        steering_plan = self.activation.plan(vector, self_state)

        self._emit_state_receipt(
            event="before_generation",
            payload={
                "generation_id": generation_id,
                "domain": domain,
                "objective": objective,
                "aura_state_hash": vector.aura_state_hash,
                "action_policy": action_policy,
                "vector": vector.to_dict(),
                "self_state": self_state.to_dict(),
                "policy": policy.to_dict(),
                "steering_plan": steering_plan.to_dict(),
            },
        )

        return ClosedLoopPreGeneration(
            generation_id=generation_id,
            now=now,
            action_policy=action_policy,
            vector=vector,
            self_state=self_state,
            policy=policy,
            steering_plan=steering_plan,
            created_at=time.time(),
        )

    def after_generation(
        self,
        *,
        prompt: str,
        response: str,
        pre: ClosedLoopPreGeneration,
        outcome: str,
        metrics: dict[str, float],
        post_state: Any | None = None,
    ) -> ClosedLoopExperience:
        post_now = self.being_runtime.sample(
            post_state,
            objective=getattr(pre.now.world, "focal_object", "") or str(prompt)[:240],
            candidate_action="response",
            predicted_outcome="verified answer",
            actual_outcome=outcome,
            tool_failed=outcome not in {"success", "verified"},
        )
        post_policy = self.being_runtime.action_policy(post_now, domain="response", priority=0.5)
        post_vector = vector_from_aura_now(
            post_now,
            welfare_outputs=getattr(self.being_runtime, "_last_welfare", None),
            blind_report=getattr(self.being_runtime, "_last_blind_report", None),
            action_policy=post_policy,
        )
        experience = self.plasticity.record_experience(
            prompt=prompt,
            response=response,
            vector=post_vector,
            self_state=pre.self_state,
            outcome=outcome,
            metrics=metrics,
        )
        self.continuum.set_external_io(False)
        self._emit_state_receipt(
            event="after_generation",
            payload={
                "generation_id": pre.generation_id,
                "experience": experience.to_dict(),
                "post_vector": post_vector.to_dict(),
                "metrics": metrics,
            },
        )
        return experience

    def _emit_state_receipt(self, *, event: str, payload: dict[str, Any]) -> None:
        store = getattr(self.plasticity, "receipt_store", None)
        if store is None:
            return
        try:
            from core.runtime.receipts import StateMutationReceipt
            receipt = StateMutationReceipt(
                cause=f"being_closed_loop_v3:{event}",
                domain="being_closed_loop",
                key=event,
                metadata=payload,
            )
            store.emit(receipt)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return


def build_main15_closed_loop(
    *,
    being_runtime: Any | None = None,
    d_model: int = 4096,
    layers: tuple[int, ...] = (),
    direction_bank: DirectionBank | None = None,
    production_mode: bool = True,
    allow_candidate_training: bool = False,
    allow_promotion: bool = False,
    receipt_store: Any | None = None,
) -> Main15ClosedLoopController:
    if being_runtime is None:
        from core.being.runtime import get_being_runtime
        being_runtime = get_being_runtime()
    bank = direction_bank or DirectionBank.zeros(d_model)
    controller = Main15ClosedLoopController(
        being_runtime=being_runtime,
        policy_coupler=ClosedLoopPolicyCoupler(production_mode=production_mode),
        attractor=FunctionalIAttractor(),
        activation=ActivationCoupler(bank, layers=layers),
        plasticity=PlasticityPromotionController(
            allow_candidate_training=allow_candidate_training,
            allow_promotion=allow_promotion,
            receipt_store=receipt_store,
        ),
        continuum=ContinuumAdapter(production_mode=production_mode),
    )
    install_default_continuity_jobs(controller.continuum, being_runtime=being_runtime)
    return controller
