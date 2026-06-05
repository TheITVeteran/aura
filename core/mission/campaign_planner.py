"""core/mission/campaign_planner.py — Long-Horizon Campaign Planner.
"""
from __future__ import annotations

import logging
from typing import List

from core.mission.objective_graph import Milestone, ObjectiveGraph

logger = logging.getLogger("Aura.CampaignPlanner")


class CampaignPlanner:
    """Splits a multi-day objective into concrete milestones and registers them."""

    @staticmethod
    def plan_campaign(objective: str, graph: ObjectiveGraph) -> List[str]:
        logger.info("🎯 Planning campaign sequence for objective: '%s'", objective)
        
        # Build standard milestones
        ms1 = Milestone("ms_prep", "Prepare environment and analyze target", status="completed")
        ms2 = Milestone("ms_exec", "Execute core steps for " + objective[:30], dependencies=["ms_prep"])
        ms3 = Milestone("ms_verify", "Verify execution and run test suite", dependencies=["ms_exec"])
        
        graph.add_milestone(ms1)
        graph.add_milestone(ms2)
        graph.add_milestone(ms3)

        return ["ms_prep", "ms_exec", "ms_verify"]
