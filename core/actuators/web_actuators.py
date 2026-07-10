"""core/actuators/web_actuators.py
================================
Actuators for searching the web and fetching URLs.
"""

import asyncio
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.actuators.actuator_registry import ActuatorResult, BaseActuator


def run_async_in_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


class WebSearchActuator(BaseActuator):
    requires_authority = True

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information using a search query."

    def validate_params(self, params: dict[str, Any]) -> bool:
        if not isinstance(params, dict) or "query" not in params:
            return False
        return isinstance(params["query"], str) and len(params["query"].strip()) > 0

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not params.get("_aura_authorized"):
            return ActuatorResult(False, "Web search requires ActuatorRegistry/AuthorityGateway authorization.", {})
        if not self.validate_params(params):
            return ActuatorResult(False, "Invalid search query parameter.", {})

        query = params["query"]
        num_results = int(params.get("num_results", 5))
        deep = bool(params.get("deep", False))

        from core.search import ResearchSearchPipeline
        pipeline = ResearchSearchPipeline()
        
        async def _run():
            return await pipeline.search(query, num_results=num_results, deep=deep)

        try:
            res = run_async_in_sync(_run())
            success = res.get("ok", False)
            if success:
                res = dict(res)
                res.setdefault("summary", res.get("answer") or res.get("message") or "")
                res.setdefault(
                    "sources",
                    res.get("citations")
                    or res.get("chunks")
                    or res.get("results")
                    or ([] if not res.get("source") else [{"url": res.get("source")}]),
                )
            msg = res.get("summary") or res.get("message") or ("Search executed." if success else "Search failed.")
            return ActuatorResult(
                success=success,
                message=msg,
                updates={"search_results": res}
            )
        except (ImportError, RuntimeError, TimeoutError, OSError, AttributeError, TypeError, ValueError) as e:
            return ActuatorResult(False, f"Web search failed: {e}", {})


class WebFetchActuator(BaseActuator):
    requires_authority = True

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch the page content (text/HTML) of a target URL."

    def validate_params(self, params: dict[str, Any]) -> bool:
        if not isinstance(params, dict) or "url" not in params:
            return False
        url = params["url"]
        if not isinstance(url, str) or not url.startswith("http"):
            return False
            
        # Check allowlist domains if any
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        # Clean port if any
        if ":" in domain:
            domain = domain.split(":")[0]
            
        allowed_domains = {"wikipedia.org", "python.org", "github.com", "pypi.org", "stackoverflow.com", "w3schools.com"}
        allowed = False
        for allowed_domain in allowed_domains:
            if domain == allowed_domain or domain.endswith("." + allowed_domain):
                allowed = True
                break
        return allowed

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not params.get("_aura_authorized"):
            return ActuatorResult(False, "Web fetch requires ActuatorRegistry/AuthorityGateway authorization.", {})
        if not self.validate_params(params):
            return ActuatorResult(False, "Invalid URL or domain not in safety allowlist.", {})

        url = params["url"]

        from core.skills.sovereign_browser import SovereignBrowserSkill
        skill = SovereignBrowserSkill()

        async def _run():
            return await skill.execute({"mode": "browse", "url": url}, {})

        try:
            res = run_async_in_sync(_run())
            success = res.get("ok", False)
            msg = res.get("message") or ("Content fetched successfully." if success else "Fetch failed.")
            return ActuatorResult(
                success=success,
                message=msg,
                updates={"fetch_results": res}
            )
        except (ImportError, RuntimeError, TimeoutError, OSError, AttributeError, TypeError, ValueError) as e:
            return ActuatorResult(False, f"Web fetch failed: {e}", {})
