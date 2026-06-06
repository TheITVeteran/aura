"""core/capabilities/browser_controller.py — General Browser Automation
========================================================================
General browser automation through the most reliable available adapter.

Prefers: AppleScript tab control > system 'open' > direct HTTP fetch.

Includes a readability pipeline to strip boilerplate from web pages
before summarization, producing clean ArticleExtract objects.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import quote_plus, urlparse

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.network_gateway import get_network_gateway

if TYPE_CHECKING:
    from core.capabilities.host_automation import AutomationReceipt

logger = logging.getLogger("Aura.BrowserController")


@dataclass
class ArticleExtract:
    """Clean extracted article from a web page."""
    url: str
    title: str = ""
    author: str = ""
    date: str = ""
    body: str = ""
    source_domain: str = ""
    word_count: int = 0
    extracted_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "author": self.author,
            "body": self.body[:500] + "..." if len(self.body) > 500 else self.body,
            "source": self.source_domain,
            "words": self.word_count,
        }


class BrowserController:
    """General browser automation.

    Uses AppleScript for Chrome/Safari tab control.
    Falls back to system 'open' command for basic URL opening.
    Uses NetworkGateway for content extraction (not UI-dependent).
    """

    def __init__(self) -> None:
        self._preferred_browser: str = "Google Chrome"
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        # Detect preferred browser
        try:
            registry = ServiceContainer.get("app_registry", default=None)
            if registry:
                pref = registry.get_preferred_browser()
                if pref:
                    self._preferred_browser = pref.name
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("browser_controller.start_registry", exc)

        ServiceContainer.register_instance("browser_controller", self, required=False)
        self._started = True
        logger.info("BrowserController ONLINE (preferred: %s)", self._preferred_browser)

    async def open_url(self, url: str, new_tab: bool = True) -> "AutomationReceipt":
        """Open a URL in the preferred browser."""
        from core.capabilities.host_automation import AutomationReceipt, AppleScriptRunner

        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        browser = self._preferred_browser
        if "chrome" in browser.lower():
            if new_tab:
                script = f'tell application "Google Chrome" to open location "{url}"'
            else:
                script = f'''
                    tell application "Google Chrome"
                        set URL of active tab of front window to "{url}"
                    end tell
                '''
        elif "safari" in browser.lower():
            if new_tab:
                script = f'''
                    tell application "Safari"
                        tell front window
                            set newTab to make new tab with properties {{URL:"{url}"}}
                        end tell
                        activate
                    end tell
                '''
            else:
                script = f'''
                    tell application "Safari"
                        set URL of current tab of front window to "{url}"
                    end tell
                '''
        else:
            # Generic fallback
            script = f'open location "{url}"'

        receipt = await AppleScriptRunner.run(script, timeout=10.0)
        receipt.action = "open_url"
        receipt.target = url[:200]
        return receipt

    async def open_multiple_tabs(self, urls: List[str]) -> List["AutomationReceipt"]:
        """Open multiple URLs in separate tabs."""
        receipts = []
        for i, url in enumerate(urls[:10]):  # Cap at 10 tabs
            receipt = await self.open_url(url, new_tab=True)
            receipts.append(receipt)
            if i < len(urls) - 1:
                await asyncio.sleep(0.3)  # Brief delay between tabs
        return receipts

    async def get_open_tabs(self) -> List[Dict[str, str]]:
        """List all open tabs in the preferred browser."""
        from core.capabilities.host_automation import AppleScriptRunner

        browser = self._preferred_browser
        if "chrome" in browser.lower():
            script = '''
                tell application "Google Chrome"
                    set tabList to {}
                    repeat with w in windows
                        repeat with t in tabs of w
                            set end of tabList to (URL of t) & "|" & (title of t)
                        end repeat
                    end repeat
                    return tabList as text
                end tell
            '''
        elif "safari" in browser.lower():
            script = '''
                tell application "Safari"
                    set tabList to {}
                    repeat with w in windows
                        repeat with t in tabs of w
                            set end of tabList to (URL of t) & "|" & (name of t)
                        end repeat
                    end repeat
                    return tabList as text
                end tell
            '''
        else:
            return []

        receipt = await AppleScriptRunner.run(script, timeout=5.0)
        if not receipt.success or not receipt.result:
            return []

        tabs = []
        raw = str(receipt.result)
        for entry in raw.split(", "):
            parts = entry.split("|", 1)
            if len(parts) == 2:
                tabs.append({"url": parts[0].strip(), "title": parts[1].strip()})
            elif parts[0].strip().startswith("http"):
                tabs.append({"url": parts[0].strip(), "title": ""})
        return tabs

    async def search_and_open(
        self, query: str, count: int = 3
    ) -> "AutomationReceipt":
        """Search the web and open top results in browser tabs."""
        from core.capabilities.host_automation import AutomationReceipt

        start = time.time()
        # Use DuckDuckGo search
        search_url = f"https://duckduckgo.com/?q={quote_plus(query)}"

        # Open the search page
        receipt = await self.open_url(search_url, new_tab=True)

        # Also try to fetch search results programmatically for opening
        try:
            results = await self._fetch_search_results(query, count)
            if results:
                urls = [r["url"] for r in results[:count]]
                await self.open_multiple_tabs(urls)
                receipt.result = json.dumps(results[:count])
        except (RuntimeError, OSError) as e:
            record_degradation("browser_controller.programmatic_search", e)
            logger.debug("Programmatic search failed: %s", e)

        receipt.action = "search_and_open"
        receipt.target = query[:200]
        receipt.duration_ms = (time.time() - start) * 1000
        return receipt

    async def _fetch_search_results(self, query: str, count: int = 5) -> List[Dict[str, str]]:
        """Fetch search results from DuckDuckGo Lite (HTML scraping)."""
        url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        try:
            response = await get_network_gateway().request_async(
                "GET",
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                },
                timeout=10,
                read_only=True,
                source="browser_controller.fetch_search_results",
            )
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or response.get("status_code")))
            html = bytes(response.get("content", b"")).decode("utf-8", errors="replace")

            # Extract links from DuckDuckGo Lite results
            results = []
            for match in re.finditer(r'<a[^>]+href="(https?://[^"]+)"[^>]*class="result-link"[^>]*>([^<]+)</a>', html):
                link_url = match.group(1)
                title = match.group(2).strip()
                if not any(d in link_url for d in ("duckduckgo.com", "duck.co")):
                    results.append({"url": link_url, "title": title})
                    if len(results) >= count:
                        break

            # Fallback: extract any external links
            if not results:
                for match in re.finditer(r'href="(https?://(?!duckduckgo)[^"]+)"', html):
                    link_url = match.group(1)
                    results.append({"url": link_url, "title": urlparse(link_url).netloc})
                    if len(results) >= count:
                        break

            return results
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            record_degradation("browser_controller.search_fetch", e)
            logger.debug("Search fetch failed: %s", e)
            return []

    async def extract_article_text(self, url: str) -> ArticleExtract:
        """Fetch a URL and extract clean article text.

        Uses a readability pipeline to strip boilerplate (nav, footer,
        sidebar, ads) and return just the article content.
        """
        extract = ArticleExtract(url=url, source_domain=urlparse(url).netloc)

        try:
            response = await get_network_gateway().request_async(
                "GET",
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                },
                timeout=15,
                read_only=True,
                source="browser_controller.extract_article_text",
            )
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error") or response.get("status_code")))
            html = bytes(response.get("content", b"")).decode("utf-8", errors="replace")

            # Extract title
            title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
            if title_match:
                extract.title = title_match.group(1).strip()[:200]

            # Extract author
            author_match = re.search(
                r'(?:name|property)=["\'](?:author|article:author)["\'][^>]*content=["\']([^"\']+)',
                html, re.IGNORECASE,
            )
            if author_match:
                extract.author = author_match.group(1).strip()[:100]

            # Extract date
            date_match = re.search(
                r'(?:name|property)=["\'](?:date|article:published_time|datePublished)["\'][^>]*content=["\']([^"\']+)',
                html, re.IGNORECASE,
            )
            if date_match:
                extract.date = date_match.group(1).strip()[:50]

            # Extract article body using readability heuristics
            body = self._extract_readable_text(html)
            extract.body = body
            extract.word_count = len(body.split())

        except (OSError, RuntimeError, TypeError, ValueError) as e:
            extract.body = f"[Extraction failed: {e}]"
            record_degradation("browser_controller.article_extract", e)
            logger.debug("Article extraction failed for %s: %s", url, e)

        return extract

    def _extract_readable_text(self, html: str) -> str:
        """Extract readable text from HTML using heuristic cleanup.

        Strips: nav, header, footer, sidebar, script, style, ads.
        Keeps: article, main, p tags, headings.
        """
        # Remove script, style, nav, footer, header
        cleaned = re.sub(
            r"<(script|style|nav|footer|header|aside|iframe|noscript)[^>]*>.*?</\1>",
            "", html, flags=re.DOTALL | re.IGNORECASE,
        )

        # Try to find <article> or <main> content
        article_match = re.search(
            r"<(?:article|main)[^>]*>(.*?)</(?:article|main)>",
            cleaned, re.DOTALL | re.IGNORECASE,
        )
        if article_match:
            cleaned = article_match.group(1)

        # Extract text from paragraphs and headings
        paragraphs = []
        for match in re.finditer(r"<(?:p|h[1-6]|li|blockquote)[^>]*>(.*?)</(?:p|h[1-6]|li|blockquote)>", cleaned, re.DOTALL | re.IGNORECASE):
            text = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            text = re.sub(r"\s+", " ", text)
            if len(text) > 20:  # Skip very short fragments
                paragraphs.append(text)

        # If paragraph extraction yielded nothing, fall back to stripping all tags
        if not paragraphs:
            stripped = re.sub(r"<[^>]+>", " ", cleaned)
            stripped = re.sub(r"\s+", " ", stripped).strip()
            # Take the middle portion (skip headers/footers)
            words = stripped.split()
            if len(words) > 100:
                start = len(words) // 10
                end = len(words) * 9 // 10
                paragraphs = [" ".join(words[start:end])]
            else:
                paragraphs = [stripped]

        body = "\n\n".join(paragraphs)
        # Truncate to reasonable size
        if len(body) > 10000:
            body = body[:10000] + "\n\n[...truncated...]"

        return body

    async def get_page_content(self, url: str) -> str:
        """Get clean text content from a URL."""
        extract = await self.extract_article_text(url)
        return extract.body

    def get_status(self) -> Dict[str, Any]:
        return {
            "preferred_browser": self._preferred_browser,
            "started": self._started,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[BrowserController] = None


def get_browser_controller() -> BrowserController:
    global _instance
    if _instance is None:
        _instance = BrowserController()
    return _instance


__all__ = [
    "BrowserController",
    "ArticleExtract",
    "get_browser_controller",
]
