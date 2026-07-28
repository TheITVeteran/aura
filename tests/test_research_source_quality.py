"""An ad redirect is not an article, and a nav bar is not reporting.

Measured live. Asked for "3 recent articles about AI", the document she wrote
cited these as her three sources:

  1. AI fundamentals | OpenAI — https://openai.com/academy/what-is-ai/
  2. OpenAI | Research & Deployment — https://openai.com/
  3. Claude Ai - Amazing AI Assistant — https://duckduckgo.com/y.js?ad_domain=
     ai%2Dpro.org&ad_provider=bingv7aa&ad_type=txad&click_metadata=...

The third is a DuckDuckGo AD REDIRECT, and its 600-character tracking URL was
printed into the document as a citation. The second is a product homepage.

And what she wrote as the synthesis was the site's navigation bar:

  "Taken together, the reporting points to this: AI fundamentals | OpenAI Skip
   to main content Research Products Business Developers Company Foundation
   (opens in a new window) Log in Try ChatGPT (opens in a new window)..."
"""

from __future__ import annotations

from core.skills.desktop_task import DesktopTaskSkill


def test_the_live_ad_redirect_is_rejected():
    ad = (
        "https://duckduckgo.com/y.js?ad_domain=ai%2Dpro.org&ad_provider=bingv7aa"
        "&ad_type=txad&click_metadata=xyhqpbOr3aW5cxvxdz2Ep"
    )
    assert DesktopTaskSkill._is_article_url(ad) is False


def test_click_trackers_and_search_pages_are_rejected():
    for url in (
        "https://www.bing.com/aclick?ld=e8EOOiDV7bPONO3m0q4lz0wzVUCUxOUD",
        "https://www.google.com/search?q=AI",
        "https://googleadservices.com/pagead/aclk?sa=L",
    ):
        assert DesktopTaskSkill._is_article_url(url) is False, url


def test_bare_product_homepages_are_not_articles():
    for url in ("https://openai.com/", "https://chatgpt.com/", "https://example.com"):
        assert DesktopTaskSkill._is_article_url(url) is False, url


def test_real_articles_are_kept():
    for url in (
        "https://openai.com/academy/what-is-ai/",
        "https://www.nature.com/articles/d41586-026-01234-5",
        "https://www.reuters.com/technology/some-story-2026-04-10/",
    ):
        assert DesktopTaskSkill._is_article_url(url) is True, url


def test_navigation_furniture_is_stripped_from_the_article_text():
    raw = (
        "AI fundamentals | OpenAI Skip to main content Research Products Business "
        "Developers Company Foundation (opens in a new window) Log in Try ChatGPT "
        "(opens in a new window) OpenAI April 10, 2026 Artificial intelligence "
        "systems learn patterns from data rather than following explicit rules."
    )
    cleaned = DesktopTaskSkill._strip_page_chrome(raw)
    for chrome in ("Skip to main content", "opens in a new window", "Try ChatGPT", "Log in"):
        assert chrome not in cleaned, f"nav furniture survived: {chrome!r}"
    # The actual reporting must survive.
    assert "Artificial intelligence systems learn patterns from data" in cleaned


def test_stripping_never_empties_real_prose():
    prose = "Researchers reported a measurable gain on held-out tasks this quarter."
    assert DesktopTaskSkill._strip_page_chrome(prose) == prose
