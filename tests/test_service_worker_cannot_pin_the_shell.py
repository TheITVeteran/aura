"""The service worker must not be able to pin a window to one revision forever.

Measured live 2026-08-03. Bryan's desktop window was controlled by a worker
bound to a revision from hours earlier. It had survived four runtime restarts
and three revision changes, four `aura-runtime-shell-*` caches had piled up,
and `fetch('/static/aura.js', {cache: 'no-store'})` from inside the page
returned the OLD file. He was reading "Conversation lane initializing" while
/api/health reported conversation_ready: true.

The trap: this worker is bound to the revision in its own script URL, and the
only thing that could move a tab onto a newer worker was page JS — which this
same worker was serving out of its frozen cache. Reloading could not escape
it, because the reload was answered from that cache.

The listener's own heading said "Network-first with cache fallback". The code
under it was cache-first. Network-first is what makes the pin escapable: the
server ignores the _aura_runtime query, so an old worker asking for its own
revision URL still receives whatever the runtime is serving now.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SERVICE_WORKER = (
    Path(__file__).resolve().parents[1] / "interface" / "static" / "service-worker.js"
)


@pytest.fixture(scope="module")
def source() -> str:
    return SERVICE_WORKER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def fetch_listener(source: str) -> str:
    start = source.index("self.addEventListener('fetch'")
    end = source.index("self.addEventListener('message'", start)
    return source[start:end]


class TestShellAssetsAreServedNetworkFirst:
    def test_the_network_is_consulted_before_the_cache(self, fetch_listener):
        revision_block = fetch_listener[fetch_listener.index("if (revisionBound)"):]
        network_at = revision_block.index("fetchWithinShellBudget")
        cache_at = revision_block.index("cache.match(revisionUrl)")
        assert network_at < cache_at, (
            "cache-first is what pinned a window to a dead revision; the cache "
            "must only answer when the network could not"
        )

    def test_a_fresh_response_replaces_the_cached_copy(self, fetch_listener):
        assert "cache.put(revisionUrl, response.clone())" in fetch_listener

    def test_the_cache_still_answers_when_the_runtime_is_down(self, fetch_listener):
        revision_block = fetch_listener[fetch_listener.index("if (revisionBound)"):]
        assert "catch" in revision_block, "a network failure must not blank the shell"
        assert "cache.match(revisionUrl)" in revision_block, (
            "offline must still serve the cached shell — a stale shell beats no shell"
        )

    def test_the_network_attempt_is_bounded(self, source):
        assert "SHELL_NETWORK_BUDGET_MS" in source
        assert "AbortController" in source, (
            "an unbounded shell fetch turns a wedged runtime into a window that "
            "never paints"
        )
        budget = re.search(r"const SHELL_NETWORK_BUDGET_MS = (\d+);", source)
        assert budget, "the budget must be a named constant"
        assert 0 < int(budget.group(1)) <= 5000, (
            "the shell comes from this machine; a longer wait only delays the window"
        )
        assert "clearTimeout(timer)" in source, "the abort timer must always be cleared"


class TestTheExistingContractsStillHold:
    def test_install_does_not_swap_assets_under_a_live_page(self, source):
        install_block = source.split("self.addEventListener('install'", 1)[1].split(
            "self.addEventListener('activate'", 1
        )[0]
        assert "self.skipWaiting()" not in install_block

    def test_api_traffic_is_never_intercepted(self, fetch_listener):
        assert "url.pathname.startsWith('/api/')" in fetch_listener
        assert "url.pathname.startsWith('/ws/')" in fetch_listener

    def test_the_worker_script_itself_is_never_served_from_cache(self, fetch_listener):
        assert "url.pathname !== '/static/service-worker.js'" in fetch_listener, (
            "the worker script is the one thing the browser must always revalidate; "
            "caching it would make the pin unescapable"
        )
