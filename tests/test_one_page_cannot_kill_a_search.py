"""Fetching one source must never destroy the whole search.

Typed into the live UI: "Can you look up what the current weather is in
Lisbon right now?"

The search SUCCEEDED — hits returned, "Tool web_search execution completed:
True", "Causal Link Recorded: web_search -> Success". Then one optional page
fetch called playwright's page.goto, which raised its own Error class. The
handler listed ImportError, ConnectionError, OSError, RuntimeError,
TimeoutError and AttributeError — playwright.Error is none of those — so the
exception escaped, the tool result became ok=False, and the person was told:

    I hit an error before I could finish that thought — the model lane was
    unavailable and the fallback was rate-limited.

None of which happened. The model lane was fine; a web page didn't load.

The pipeline already knows how to synthesize from snippets when no page can
be fetched — that path existed and was simply never reached. Enrichment is
optional by construction, so both fetch paths now swallow anything and skip
the hit.
"""

import inspect

import pytest

from core.search import research_pipeline

pytestmark = pytest.mark.unit


def _source_of(name: str) -> str:
    return inspect.getsource(getattr(research_pipeline.ResearchSearchPipeline, name))


@pytest.mark.parametrize("method", ["_fetch_page_with_browser", "_fetch_page"])
def test_a_single_page_fetch_swallows_anything(method: str):
    """A narrow except list is how the last one escaped."""
    source = _source_of(method)
    assert "except Exception" in source, (
        f"{method} must not let one page's failure escape into the search"
    )
    assert "record_degradation" in source, "silence is not allowed either"
    assert "return None" in source


def test_the_snippet_fallback_still_exists():
    """The reason skipping a page is safe: there is a path without pages."""
    source = inspect.getsource(research_pipeline.ResearchSearchPipeline.search)
    assert "Fallback Synthesis" in source
