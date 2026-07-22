"""core/actuators/web_actuators.py
================================
Actuators for searching the web and fetching URLs.

Hardening (CP126): fetch URLs are validated as exact HTTPS with no userinfo,
matched against a configurable domain allowlist, and — critically — every
resolved IP is checked so an allowlisted hostname cannot be rebound to a
loopback / private / link-local / cloud-metadata address (SSRF). Search inputs
are bounded, the async bridge carries a deadline, results are shape-checked,
and the authority/capability context is forwarded to the browser skill.
"""

import asyncio
import ipaddress
import os
import socket
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.actuators.actuator_registry import ActuatorResult, BaseActuator

_DEFAULT_BRIDGE_DEADLINE_S = 60.0
_MAX_QUERY_CHARS = 2048
_MIN_RESULTS = 1
_MAX_RESULTS = 25

_DEFAULT_FETCH_ALLOWLIST = {
    "wikipedia.org", "python.org", "github.com", "pypi.org",
    "stackoverflow.com", "w3schools.com",
}


def run_async_in_sync(coro, *, deadline_s: float = _DEFAULT_BRIDGE_DEADLINE_S):
    """Run a coroutine from sync code with a bounded deadline.

    NOTE: when called from within a running loop this still blocks the calling
    thread on a worker loop; the deadline caps that wait so a hung network call
    cannot stall the caller indefinitely. A fully non-blocking path requires an
    async actuator interface (tracked separately).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(asyncio.wait_for(coro, timeout=deadline_s))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, asyncio.wait_for(coro, timeout=deadline_s))
        return future.result(timeout=deadline_s + 5.0)


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        num = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, num))


def _allowed_fetch_domains() -> set[str]:
    extra = os.environ.get("AURA_WEB_FETCH_ALLOWLIST", "")
    names = {p.strip().lower() for p in extra.split(",") if p.strip()}
    return _DEFAULT_FETCH_ALLOWLIST | names


def _ip_is_public(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
        or ip_text == "169.254.169.254"  # cloud metadata (already link-local, explicit)
    )


def validate_fetch_url(url: Any) -> tuple[str | None, str]:
    """Exact-HTTPS + allowlist + resolved-IP SSRF policy. Returns (url, error)."""
    if not isinstance(url, str) or not url.strip():
        return None, "url is missing"
    parsed = urllib.parse.urlparse(url.strip())
    scheme = parsed.scheme.lower()
    allow_http = str(os.environ.get("AURA_WEB_FETCH_ALLOW_HTTP", "")).strip().lower() in {"1", "true", "yes", "on"}
    if scheme != "https" and not (allow_http and scheme == "http"):
        return None, f"scheme '{scheme}' not allowed (https required)"
    if parsed.username or parsed.password or "@" in (parsed.netloc or ""):
        return None, "url must not contain credentials (userinfo)"
    host = (parsed.hostname or "").lower()
    if not host:
        return None, "url has no host"
    allowed = _allowed_fetch_domains()
    if not any(host == d or host.endswith("." + d) for d in allowed):
        return None, f"host '{host}' is not in the fetch allowlist"
    # Bind the allowlisted name to its resolved addresses — reject if ANY
    # resolves to a non-public target (defeats DNS rebinding to internal hosts).
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except (socket.gaierror, OSError, ValueError) as exc:
        return None, f"host '{host}' did not resolve: {exc}"
    addrs = {info[4][0] for info in infos}
    if not addrs:
        return None, f"host '{host}' resolved to no addresses"
    for addr in addrs:
        if not _ip_is_public(addr):
            return None, f"host '{host}' resolves to a non-public address ({addr})"
    return url.strip(), ""


class WebSearchActuator(BaseActuator):
    requires_authority = True
    _pipeline = None  # canonical, reused across searches

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

    def _get_pipeline(self):
        if WebSearchActuator._pipeline is None:
            from core.search import ResearchSearchPipeline
            WebSearchActuator._pipeline = ResearchSearchPipeline()
        return WebSearchActuator._pipeline

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not params.get("_aura_authorized"):
            return ActuatorResult(False, "Web search requires ActuatorRegistry/AuthorityGateway authorization.", {})
        if not self.validate_params(params):
            return ActuatorResult(False, "Invalid search query parameter.", {})

        query = params["query"].strip()[:_MAX_QUERY_CHARS]
        num_results = _clamp_int(params.get("num_results", 5), 5, _MIN_RESULTS, _MAX_RESULTS)
        deep = bool(params.get("deep", False))
        deadline = float(params.get("deadline_s", _DEFAULT_BRIDGE_DEADLINE_S))

        pipeline = self._get_pipeline()

        async def _run():
            return await pipeline.search(query, num_results=num_results, deep=deep)

        try:
            res = run_async_in_sync(_run(), deadline_s=deadline)
            if not isinstance(res, dict):
                return ActuatorResult(False, "Search returned a malformed (non-dict) result.", {})
            success = bool(res.get("ok", False))
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
            return ActuatorResult(success=success, message=msg, updates={"search_results": res})
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
        validated, _err = validate_fetch_url(params["url"])
        return validated is not None

    def execute(self, params: dict[str, Any]) -> ActuatorResult:
        if not params.get("_aura_authorized"):
            return ActuatorResult(False, "Web fetch requires ActuatorRegistry/AuthorityGateway authorization.", {})

        validated_url, url_err = validate_fetch_url(params.get("url"))
        if validated_url is None:
            return ActuatorResult(False, f"Invalid URL or domain not in safety allowlist: {url_err}", {})

        deadline = float(params.get("deadline_s", _DEFAULT_BRIDGE_DEADLINE_S))

        from core.skills.sovereign_browser import SovereignBrowserSkill
        skill = SovereignBrowserSkill()

        # Forward the authority/capability context so the browser skill runs
        # under the same end-to-end authorization, not an empty context.
        skill_context = {
            "_aura_authorized": True,
            "_capability_token_id": params.get("_capability_token_id"),
            "source": "web_fetch_actuator",
        }

        async def _run():
            return await skill.execute({"mode": "browse", "url": validated_url}, skill_context)

        try:
            res = run_async_in_sync(_run(), deadline_s=deadline)
            if not isinstance(res, dict):
                return ActuatorResult(False, "Fetch returned a malformed (non-dict) result.", {})
            success = bool(res.get("ok", False))
            msg = res.get("message") or ("Content fetched successfully." if success else "Fetch failed.")
            return ActuatorResult(success=success, message=msg, updates={"fetch_results": res, "requested_url": validated_url})
        except (ImportError, RuntimeError, TimeoutError, OSError, AttributeError, TypeError, ValueError) as e:
            return ActuatorResult(False, f"Web fetch failed: {e}", {})
