"""Governed browser/web agent surface.

Uses Playwright when installed for interactive pages and falls back to httpx
for simple fetch/read tasks.  Callers remain responsible for Will/capability
receipts before external network use.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from core.runtime.network_gateway import get_network_gateway


@dataclass(frozen=True)
class WebAgentResult:
    url: str
    title: str
    text: str
    engine: str
    ok: bool
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "title": self.title, "text": self.text, "engine": self.engine, "ok": self.ok, "error": self.error}


class BrowserWebAgent:
    async def execute(self, params: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
        url = str(params.get("url") or params.get("target") or "").strip()
        if not url:
            return WebAgentResult(url="", title="", text="", engine="none", ok=False, error="url required").to_dict()
        timeout_ms = int(params.get("timeout_ms") or 15000)
        return (await self.read(url, timeout_ms=timeout_ms)).to_dict()

    async def read(self, url: str, *, timeout_ms: int = 15000) -> WebAgentResult:
        try:
            return await self._read_playwright(url, timeout_ms=timeout_ms)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            try:
                return await self._read_httpx(url, error_prefix=f"playwright_unavailable:{exc}")
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc2:
                return WebAgentResult(url=url, title="", text="", engine="none", ok=False, error=repr(exc2))

    async def _read_playwright(self, url: str, *, timeout_ms: int) -> WebAgentResult:
        from playwright.async_api import async_playwright  # type: ignore

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            title = await page.title()
            text = await page.locator("body").inner_text(timeout=timeout_ms)
            await browser.close()
            return WebAgentResult(url=url, title=title, text=text[:12000], engine="playwright", ok=True)

    async def _read_httpx(self, url: str, *, error_prefix: str = "") -> WebAgentResult:
        resp = await asyncio.to_thread(
            get_network_gateway().request,
            "GET",
            url,
            timeout=15.0,
            source="skills.toolbox_web_agent.read",
        )
        if not resp.get("ok"):
            return WebAgentResult(
                url=url,
                title="",
                text="",
                engine="network_gateway",
                ok=False,
                error=resp.get("error") or f"HTTP {resp.get('status_code')}",
            )
        html = bytes(resp.get("content") or b"").decode("utf-8", errors="replace")
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
        text = re.sub(r"<(script|style).*?</\1>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return WebAgentResult(url=url, title=title, text=text[:12000], engine="httpx", ok=True, error=error_prefix)


__all__ = ["BrowserWebAgent", "WebAgentResult"]
