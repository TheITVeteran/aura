"""core/world/connectors/github_connector.py — Governed GitHub Repository Connector.
"""
from __future__ import annotations

import logging
import urllib.parse
from typing import Any, Dict, List

from core.runtime.errors import record_degradation
from core.runtime.network_gateway import get_network_gateway

logger = logging.getLogger("Aura.Perception.GitHubConnector")
_CONNECTOR_RECOVERABLE_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class GitHubConnector:
    """Connector for querying GitHub API for repositories, files, and issues."""

    def __init__(self) -> None:
        self.domain = "github"

    async def fetch(self, query: str) -> List[Dict[str, Any]]:
        logger.info("🐙 Querying GitHub repos for: '%s'", query)
        gateway = get_network_gateway()
        results = []

        try:
            encoded = urllib.parse.urlencode({"q": query, "sort": "stars"})
            url = f"https://api.github.com/search/repositories?{encoded}"
            response = gateway.request(method="GET", url=url, timeout=5.0, source="github_connector")
            if response.get("ok"):
                content = response.get("content") or b""
                text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
                results.append({
                    "title": f"GitHub Search Result: {query}",
                    "source": "github.com",
                    "content": text[:2000],
                    "url": url,
                    "confidence": 0.90,
                })
        except _CONNECTOR_RECOVERABLE_ERRORS as e:
            record_degradation(
                "github_connector",
                e,
                action="returned no GitHub perception items after governed fetch failure",
                extra={"query": query[:200]},
            )
            logger.debug("GitHub connector query degraded: %s", e)
        return results
