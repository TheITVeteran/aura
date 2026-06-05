"""core/world/connectors/data_connector.py — Governed Public Dataset Connector.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from core.runtime.errors import record_degradation
from core.runtime.network_gateway import get_network_gateway

logger = logging.getLogger("Aura.Perception.DataConnector")
_CONNECTOR_RECOVERABLE_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class DataConnector:
    """Connector for public datasets, economic indicators, and geospatial feeds."""

    def __init__(self) -> None:
        self.domain = "data"

    async def fetch(self, query: str) -> List[Dict[str, Any]]:
        logger.info("📊 Ingesting public databases for: '%s'", query)
        gateway = get_network_gateway()
        results = []

        try:
            url = "https://api.weather.gov/alerts/active?area=CA"
            response = gateway.request(method="GET", url=url, timeout=5.0, source="data_connector")
            if response.get("ok"):
                content = response.get("content") or b""
                text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
                results.append({
                    "title": "US Government Weather Alerts",
                    "source": "weather.gov",
                    "content": text[:2000],
                    "url": url,
                    "confidence": 0.98,
                })
        except _CONNECTOR_RECOVERABLE_ERRORS as e:
            record_degradation(
                "data_connector",
                e,
                action="returned no public dataset perception items after governed fetch failure",
                extra={"query": query[:200]},
            )
            logger.debug("Public dataset query degraded: %s", e)
        return results
