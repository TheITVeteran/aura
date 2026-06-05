"""core/council/minority_report.py — Council Disagreement Audit Logger.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger("Aura.MinorityReport")


@dataclass
class MinorityDisagreement:
    timestamp: float
    mission_id: str
    dissenting_role: str
    dissent_content: str
    risk_level: str  # "low", "medium", "critical"
    consensus_decision: str


class MinorityReportStore:
    """Manages files tracking dissenting opinions of council members."""

    def __init__(self, log_path: Optional[Path] = None) -> None:
        if log_path is None:
            # Default to data dir
            from core.config import config
            self.log_path = config.paths.log_dir / "minority_reports.jsonl"
        else:
            self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record_dissent(self, disagreement: MinorityDisagreement) -> None:
        """Appends a new dissent entry to the log file."""
        logger.warning("📝 Minority Report filed by %s on %s!", disagreement.dissenting_role, disagreement.mission_id)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(disagreement)) + "\n")
        except OSError as e:
            logger.error("Failed to write minority report: %s", e)

    def list_dissents(self) -> List[MinorityDisagreement]:
        if not self.log_path.exists():
            return []
        entries = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        entries.append(MinorityDisagreement(**data))
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to read minority reports: %s", e)
        return entries


# Singleton
_store_instance: MinorityReportStore | None = None


def get_minority_report_store() -> MinorityReportStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = MinorityReportStore()
    return _store_instance
