#!/usr/bin/env python3
"""Dynamic Web-Browsing Task Runner.

This script executes a live browser navigation task using Aura's PhantomBrowser
and verifies that she can dynamic-browse, navigate links, click elements,
and extract content/facts from webpages (or local mock services).
"""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from core.phantom_browser import PhantomBrowser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DynamicBrowsingRunner")


async def run_browsing_task(
    start_url: str,
    target_link_text: str | None = None,
    expected_content_keywords: list[str] | None = None,
    click_selector: str | None = None,
) -> dict[str, Any]:
    """Execute a dynamic browsing session using the live PhantomBrowser."""
    logger.info("Starting dynamic browsing task for URL: %s", start_url)
    browser = PhantomBrowser(visible=False)
    
    try:
        # Initialize browser
        ready = await browser.ensure_ready()
        if not ready:
            status = browser.get_status()
            logger.error("Failed to initialize browser. Status: %s", status)
            return {
                "ok": False,
                "error": "Browser initialization failed",
                "status": status,
            }
        
        # 1. Browse start URL
        success = await browser.browse(start_url)
        if not success:
            return {"ok": False, "error": f"Failed to navigate to {start_url}"}
        
        # 2. Extract initial content
        initial_content = await browser.read_content()
        logger.info("Successfully navigated to start page. Content length: %d", len(initial_content))
        
        # 3. Handle optional dynamic interactions
        if target_link_text:
            logger.info("Attempting to click link with text: '%s'", target_link_text)
            clicked = await browser.click(text_match=target_link_text)
            if not clicked:
                logger.warning("Failed to click link using text match. Attempting link traversal via extraction.")
                links = await browser.get_links()
                for link in links:
                    if target_link_text.lower() in link.get("text", "").lower():
                        logger.info("Found matching link URL: %s. Direct traversing.", link["url"])
                        await browser.browse(link["url"])
                        clicked = True
                        break
            if not clicked:
                return {"ok": False, "error": f"Could not navigate to target link: '{target_link_text}'"}
                
        elif click_selector:
            logger.info("Attempting to click selector: '%s'", click_selector)
            clicked = await browser.click(selector=click_selector)
            if not clicked:
                return {"ok": False, "error": f"Could not click selector: '{click_selector}'"}

        # 4. Extract final content
        final_content = await browser.read_content()
        logger.info("Final page content length: %d", len(final_content))
        
        # 5. Verify keywords
        verification_results = {}
        all_matched = True
        if expected_content_keywords:
            for kw in expected_content_keywords:
                matched = kw.lower() in final_content.lower()
                verification_results[kw] = matched
                if not matched:
                    all_matched = False
                    logger.warning("Verification failed: keyword '%s' not found in final content.", kw)
        
        return {
            "ok": all_matched,
            "start_url": start_url,
            "verification": verification_results,
            "content_snippet": final_content[:800],
        }
        
    except Exception as e:
        logger.exception("An error occurred during dynamic browsing: %s", e)
        return {"ok": False, "error": str(e)}
    finally:
        await browser.close()
        logger.info("Phantom Browser closed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./run_dynamic_browsing_task.py <url> [target_link_text] [expected_keywords_comma_separated]")
        sys.exit(1)
        
    url = sys.argv[1]
    link_text = sys.argv[2] if len(sys.argv) > 2 else None
    kws = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    
    res = asyncio.run(run_browsing_task(url, link_text, kws))
    print("\nResult:")
    print(res)
    sys.exit(0 if res["ok"] else 1)
