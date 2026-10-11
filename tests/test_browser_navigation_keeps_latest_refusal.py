"""A denied destination is not sent elsewhere or explained by an older HTTP failure."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.capabilities.browser_authority import BrowserVerdict
from core.capabilities.phantom_browser import PhantomBrowser, why_it_would_not_load
from core.skills.sovereign_browser_going import its_archived_copy, the_archived_copy


@pytest.mark.asyncio
async def test_new_policy_refusal_retires_the_old_network_failure(monkeypatch):
    from core.capabilities import phantom_browser

    browser = PhantomBrowser(principal="owner")
    browser._last_navigation = {"ok": False, "reason": "http_429"}
    monkeypatch.setattr(phantom_browser, "authorize_browser_action",
                        lambda action, **kwargs: BrowserVerdict(False, "destination not authorized", action, "owner"))
    assert not await browser.browse("http://127.0.0.1:54321/fixture")
    assert why_it_would_not_load(browser) == "policy_refused:destination not authorized"


@pytest.mark.asyncio
async def test_authority_refusal_is_not_retried_at_an_archive():
    skill = SimpleNamespace(_safe_browse=AsyncMock(), _say_out_loud=AsyncMock())
    browser = SimpleNamespace(_last_verdict={"allowed": False, "reason": "standing directive"})
    assert await the_archived_copy(skill, browser, "https://example.org/") == ""
    skill._safe_browse.assert_not_called()


@pytest.mark.parametrize("url", ["http://127.0.0.1:54321/fixture", "https://[::1]/fixture", "http://localhost/fixture",
                                "https://192.168.1.2/private", "https://169.254.169.254/latest/metadata",
                                "https://internal.local/", "https://user:password@example.org/", "file:///tmp/scene"])
def test_private_or_credential_bearing_destinations_have_no_public_archive_copy(url):
    assert its_archived_copy(url) == ""


def test_public_pages_still_have_an_archival_destination():
    assert its_archived_copy("https://example.org/game").endswith("/https://example.org/game")
