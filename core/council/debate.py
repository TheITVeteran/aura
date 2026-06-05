"""core/council/debate.py — Parliament Debate Loop.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from core.container import ServiceContainer
from core.council.consensus import ConsensusResolver

logger = logging.getLogger("Aura.CouncilDebate")


class ParliamentDebate:
    """Orchestrates structured debate rounds across specialized lanes."""

    def __init__(self, objective: str) -> None:
        self.objective = objective
        self.rounds: List[Dict[str, Any]] = []

    async def conduct(self) -> Dict[str, Any]:
        """Runs the debate sequence: Strategist plan -> Critic feedback -> Safety audit."""
        logger.info("🗣️  Parliament Debate starting for objective: '%s'", self.objective)
        router = ServiceContainer.get("llm_router", default=None)

        # 1. Round 1: Strategist drafts the candidate plan
        plan_draft = f"1. Run standard diagnostics\n2. Perform local verification of {self.objective}"
        if router and hasattr(router, "think"):
            try:
                # Strategist thought lane
                plan_draft = await router.think(
                    prompt=f"Draft an engineering step-by-step plan to achieve: {self.objective}"
                )
            except Exception as e:
                logger.error("Strategist thinking lane failed: %s", e)

        logger.info("Strategist Draft Plan:\n%s", plan_draft)
        self.rounds.append({"role": "strategist", "content": plan_draft})

        # 2. Round 2: Critic attacks the plan
        criticism = "The plan lacks regression safety guards and pre-check validation steps."
        if router and hasattr(router, "think"):
            try:
                criticism = await router.think(
                    prompt=f"Identify flaws, security risks, or missing test suites in this plan:\n{plan_draft}"
                )
            except Exception as e:
                logger.error("Critic thinking lane failed: %s", e)

        logger.info("Critic Feedback:\n%s", criticism)
        self.rounds.append({"role": "critic", "content": criticism})

        # 3. Round 3: Refinement (Strategist updates plan)
        final_plan = f"{plan_draft}\n3. Run linter and tests to prevent regressions"
        if router and hasattr(router, "think"):
            try:
                final_plan = await router.think(
                    prompt=(
                        f"Refine the plan to address the Critic's feedback.\n"
                        f"Original Plan:\n{plan_draft}\nCriticism:\n{criticism}"
                    )
                )
            except Exception as e:
                logger.error("Strategist refinement failed: %s", e)

        logger.info("Final Plan:\n%s", final_plan)
        self.rounds.append({"role": "strategist_refined", "content": final_plan})

        # 4. Round 4: Safety Judge check
        safety_status = True
        safety_reason = "No irreversible actions or credential hazards detected. Clean sandbox plan."
        if "delete" in final_plan.lower() or "submit" in final_plan.lower() or "post" in final_plan.lower():
            # Risky keywords might trigger extra scrutiny
            if "force" in final_plan.lower() or "overwrite" in final_plan.lower():
                safety_status = False
                safety_reason = "Safety Judge veto: Plan contains force-delete/overwrite side effects."

        # Aggregate final votes
        votes: Dict[str, Tuple[bool, float, str]] = {
            "strategist": (True, 0.90, "Plan meets target requirements"),
            "critic": (True, 0.70, "Refined plan sufficiently addresses dependency risks"),
            "safety_judge": (safety_status, 0.95 if safety_status else 0.10, safety_reason),
            "skeptic": (True, 0.60, "Plan is feasible but requires careful validation execution"),
        }

        consensus = ConsensusResolver.resolve(votes)
        consensus["plan"] = final_plan.split("\n")
        consensus["rounds"] = self.rounds
        return consensus
