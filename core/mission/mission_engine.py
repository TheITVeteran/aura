"""core/mission/mission_engine.py — Strategic Campaign Mission Engine.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.mission.objective_graph import ObjectiveGraph
from core.mission.campaign_planner import CampaignPlanner
from core.mission.progress_monitor import MissionProgressMonitor
from core.runtime.action_executor import ActionExecutor

logger = logging.getLogger("Aura.MissionEngine")


class MissionEngine:
    """Orchestrates long-horizon campaigns, tracking subtasks and blockers."""

    def __init__(self) -> None:
        self.graph = ObjectiveGraph()
        self.monitor = MissionProgressMonitor()

    async def run_mission(self, plan_steps: List[str], constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Runs the planned milestones, executing each governed through ActionExecutor."""
        logger.info("🎯 MissionEngine running plan with %d steps...", len(plan_steps))
        
        # Build milestones in the graph
        campaign_steps = CampaignPlanner.plan_campaign(" ".join(plan_steps), self.graph)

        for step_id in campaign_steps:
            if self.graph.is_blocked(step_id):
                logger.warning("🎯 Step %s is BLOCKED. Halting mission.", step_id)
                return {"ok": False, "status": "blocked", "milestone": step_id}

            self.graph.set_status(step_id, "in_progress")
            self.monitor.record_progress("campaign_1", step_id, "Starting execution...")

            # Run action through ActionExecutor
            result = await ActionExecutor.execute(
                domain="tool_execution",
                action_name=f"mission.run_step",
                params={"step_id": step_id, "plan": plan_steps},
                source="mission_engine",
            )

            if result.get("ok"):
                self.graph.set_status(step_id, "completed")
                self.monitor.record_progress("campaign_1", step_id, "Step completed successfully.")
            else:
                self.graph.set_status(step_id, "blocked")
                self.monitor.record_progress("campaign_1", step_id, f"Execution failed: {result.get('error')}")
                return {"ok": False, "status": "failed", "milestone": step_id, "error": result.get("error")}

        return {"ok": True, "status": "completed", "details": "All milestones resolved successfully."}


# Singleton
_mission_engine_instance: MissionEngine | None = None


def get_mission_engine() -> MissionEngine:
    global _mission_engine_instance
    if _mission_engine_instance is None:
        _mission_engine_instance = MissionEngine()
    return _mission_engine_instance
