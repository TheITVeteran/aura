"""core/world/connectors/papers_connector.py — Governed Academic Literature Connector.
"""
from __future__ import annotations

import logging
import urllib.parse
from typing import Any, Dict, List

from core.runtime.errors import record_degradation
from core.runtime.network_gateway import get_network_gateway

logger = logging.getLogger("Aura.Perception.PapersConnector")
_CONNECTOR_RECOVERABLE_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class PapersConnector:
    """Connector for querying academic libraries and repositories (arXiv, PubMed)."""

    def __init__(self) -> None:
        self.domain = "papers"

    async def fetch(self, query: str) -> List[Dict[str, Any]]:
        logger.info("📚 Fetching scientific literature for: '%s'", query)
        gateway = get_network_gateway()
        results = []

        try:
            encoded = urllib.parse.urlencode({"search_query": f"all:{query}", "max_results": "3"})
            url = f"http://export.arxiv.org/api/query?{encoded}"
            response = gateway.request(method="GET", url=url, timeout=5.0, source="papers_connector")
            if response.get("ok"):
                content = response.get("content") or b""
                text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
                results.append({
                    "title": f"arXiv Search Result: {query}",
                    "source": "arxiv.org",
                    "content": text[:2000],
                    "url": url,
                    "confidence": 0.92,
                })
        except _CONNECTOR_RECOVERABLE_ERRORS as e:
            record_degradation(
                "papers_connector",
                e,
                action="returned no literature perception items after governed fetch failure",
                extra={"query": query[:200]},
            )
            logger.debug("Academic literature query degraded: %s", e)
        return results
