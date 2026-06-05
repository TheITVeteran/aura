"""core/world/connectors/web_connector.py — Governed Web Search Connector.
"""
from __future__ import annotations

import logging
import urllib.parse
from typing import Any, Dict, List

from core.runtime.errors import record_degradation
from core.runtime.network_gateway import get_network_gateway

logger = logging.getLogger("Aura.Perception.WebConnector")
_CONNECTOR_RECOVERABLE_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class WebConnector:
    """Connector for querying the public web and parsing content."""

    def __init__(self) -> None:
        self.domain = "web"

    async def fetch(self, query: str) -> List[Dict[str, Any]]:
        """Query search engines and return structured results."""
        logger.info("🌐 Fetching web info for: '%s'", query)
        
        gateway = get_network_gateway()
        results = []

        try:
            encoded = urllib.parse.urlencode({"q": query, "format": "json"})
            response = gateway.request(
                method="GET",
                url=f"https://api.duckduckgo.com/?{encoded}",
                timeout=5.0,
                source="web_connector",
            )
            if response.get("ok"):
                content = response.get("content") or b""
                text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
                results.append({
                    "title": f"DuckDuckGo Search: {query}",
                    "source": "duckduckgo.com",
                    "content": text[:2000],
                    "url": f"https://duckduckgo.com/?q={query}",
                    "confidence": 0.80,
                })
        except _CONNECTOR_RECOVERABLE_ERRORS as e:
            record_degradation(
                "web_connector",
                e,
                action="returned no web perception items after governed fetch failure",
                extra={"query": query[:200]},
            )
            logger.debug("Web search query degraded: %s", e)
        return results
