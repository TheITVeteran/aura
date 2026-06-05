"""core/body/cloud_body.py — Controlled Cloud Body."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("Aura.CloudBody")


class CloudBody:
    """Monitors active remote nodes, storage buckets, network bounds, and costs."""

    def __init__(self) -> None:
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.cost_limit = 100.00  # $100 budget cap
        self.current_cost = 0.00
        self.active_jobs: List[Dict[str, Any]] = []
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True
        logger.info("Controlled Cloud Body subsystem active.")

    def register_node(self, node_id: str, region: str, cost_per_hour: float) -> None:
        self.workers[node_id] = {
            "region": region,
            "cost_per_hour": cost_per_hour,
            "status": "online",
        }
        logger.info("Registered worker node %s in region %s", node_id, region)

    def request_compute_allocation(self, task_name: str, estimated_hours: float, node_id: str) -> bool:
        """Grants compute allocation if it falls within the budget and safety constraints."""
        node = self.workers.get(node_id)
        if not node:
            logger.error("Node %s not found for allocation", node_id)
            return False

        cost = node["cost_per_hour"] * estimated_hours
        if self.current_cost + cost > self.cost_limit:
            logger.warning(
                "🚫 CloudBody: Allocation denied. Predicted cost $%.2f exceeds remaining budget $%.2f",
                cost, self.cost_limit - self.current_cost
            )
            return False

        self.current_cost += cost
        job = {
            "task_name": task_name,
            "node_id": node_id,
            "cost": cost,
            "status": "running",
        }
        self.active_jobs.append(job)
        logger.info("Allocated job %s to node %s (cost=$%.2f)", task_name, node_id, cost)
        return True

    def get_body_status(self) -> Dict[str, Any]:
        return {
            "workers_online": len(self.workers),
            "budget_limit": self.cost_limit,
            "current_cost": self.current_cost,
            "active_jobs_count": len(self.active_jobs),
            "active_jobs": self.active_jobs,
        }
