"""core/world/perception_hub.py — Ingestion Perception Hub.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.world.connectors.web_connector import WebConnector
from core.world.connectors.papers_connector import PapersConnector
from core.world.connectors.github_connector import GitHubConnector
from core.world.connectors.data_connector import DataConnector
from core.world.source_reliability import get_source_reliability_monitor

logger = logging.getLogger("Aura.PerceptionHub")


class PerceptionHub:
    """Orchestrates all planetary perception feeds and handles ingestion pipelines."""

    def __init__(self) -> None:
        self.connectors = [
            WebConnector(),
            PapersConnector(),
            GitHubConnector(),
            DataConnector(),
        ]
        self.monitor = get_source_reliability_monitor()
        self.raw_feed: List[Dict[str, Any]] = []

    async def perceive(self, query: str) -> List[Dict[str, Any]]:
        """Queries all active connectors, scoring source reliability and formatting claims."""
        logger.info("📡 PerceptionHub initiating scan for: '%s'", query)
        new_items: List[Dict[str, Any]] = []

        for conn in self.connectors:
            try:
                results = await conn.fetch(query)
                for item in results:
                    source_domain = item.get("source", "unknown")
                    reliability = self.monitor.get_score(source_domain)
                    
                    # Update item with calibrated reliability score
                    item["reliability"] = reliability
                    # Check duplication
                    if not any(x.get("url") == item.get("url") for x in self.raw_feed):
                        new_items.append(item)
                        self.raw_feed.append(item)
            except Exception as e:
                logger.error("Error running connector %s: %s", type(conn).__name__, e)

        logger.info("📡 Ingested %d new items into Perception Hub.", len(new_items))
        return new_items


# Singleton
_hub_instance: PerceptionHub | None = None


def get_perception_hub() -> PerceptionHub:
    global _hub_instance
    if _hub_instance is None:
        _hub_instance = PerceptionHub()
    return _hub_instance
