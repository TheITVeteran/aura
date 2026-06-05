"""core/mission/progress_monitor.py — Mission Progress Monitor and Logger.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List

logger = logging.getLogger("Aura.MissionProgressMonitor")


class MissionProgressMonitor:
    """Monitors, records, and logs campaign state changes over time."""

    def __init__(self) -> None:
        self.logs: List[Dict[str, Any]] = []

    def record_progress(self, mission_id: str, milestone_id: str, details: str) -> None:
        log_entry = {
            "timestamp": time.time(),
            "mission_id": mission_id,
            "milestone_id": milestone_id,
            "details": details,
        }
        self.logs.append(log_entry)
        logger.info("⏱️  [PROGRESS] Mission: %s, Milestone: %s -> %s", mission_id, milestone_id, details)

    def get_logs(self, mission_id: str) -> List[Dict[str, Any]]:
        return [g for g in self.logs if g["mission_id"] == mission_id]
