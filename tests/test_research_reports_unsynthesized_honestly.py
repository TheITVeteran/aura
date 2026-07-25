"""Research that gathered evidence but could not synthesize it is not "complete".

Live 2026-07-25: deep research ran, fetched 5 sources, then called the model
for synthesis. Background inference was queued behind foreground headroom, so
generate() returned instantly and empty. The pipeline logged
``Deep research complete: 1 loops, 1 queries, 5 sources, 0.0s`` and returned an
empty answer, and the caller logged "returned an empty answer" — which reads as
"the web had nothing", when in fact it had five sources and the local model was
simply busy.

Two different failures. Only one of them is about the web, and conflating them
sends the next debugging hour in the wrong direction.
"""
from __future__ import annotations

import pytest

from core.skills.deep_research import run_deep_research, synthesize_answer
from core.skills.deep_research import ResearchState, SearchResult

pytestmark = pytest.mark.unit


class Brain:
    """Returns whatever the test says the model produced."""

    def __init__(self, response, error=None):
        self._response = response
        self._error = error
        self.calls = 0

    async def generate(self, prompt, options=None):
        self.calls += 1
        payload = {"response": self._response}
        if self._error:
            payload["error"] = self._error
        return payload


async def _search_fn(query):
    return {
        "ok": True,
        "content": f"content about {query}",
        "sources": [{"url": f"https://example.invalid/{query}", "title": "src"}],
    }


class TestSynthesisStatus:
    @pytest.mark.asyncio
    async def test_a_real_answer_is_marked_ok(self):
        state = ResearchState(original_question="q")
        state.search_results.append(SearchResult(query="q", content="c"))

        state = await synthesize_answer(state, Brain("a genuine synthesized answer"))

        assert state.synthesis_status == "ok"
        assert state.final_answer == "a genuine synthesized answer"

    @pytest.mark.asyncio
    async def test_an_empty_generation_is_not_an_answer(self):
        state = ResearchState(original_question="q")

        state = await synthesize_answer(state, Brain(""))

        assert state.synthesis_status == "no_text"
        assert state.final_answer == ""

    @pytest.mark.asyncio
    async def test_whitespace_is_not_an_answer(self):
        state = await synthesize_answer(ResearchState(original_question="q"), Brain("  \n "))
        assert state.synthesis_status == "no_text"

    @pytest.mark.asyncio
    async def test_the_reason_is_carried_not_guessed(self):
        state = await synthesize_answer(
            ResearchState(original_question="q"),
            Brain("", error="background inference queued: foreground_headroom_reserved"),
        )
        assert "foreground_headroom_reserved" in state.synthesis_detail

    @pytest.mark.asyncio
    async def test_a_missing_reason_still_says_something_true(self):
        state = await synthesize_answer(ResearchState(original_question="q"), Brain(""))
        assert state.synthesis_detail == "the model returned no text"


class TestPipelineReporting:
    @pytest.mark.asyncio
    async def test_the_live_shape_reports_unsynthesized_with_its_sources(self, caplog):
        """5 sources gathered, model unavailable — must not claim completion."""
        with caplog.at_level("WARNING"):
            result = await run_deep_research(
                "what changed?", Brain(""), _search_fn, max_loops=1
            )

        assert result["synthesis_status"] == "no_text"
        assert result["answer"] == ""
        assert result["sources"], "evidence gathered must still be returned"
        assert "could not synthesize" in caplog.text
        assert "Deep research complete" not in caplog.text

    @pytest.mark.asyncio
    async def test_a_successful_run_still_reports_completion(self):
        result = await run_deep_research(
            "what changed?", Brain("a synthesized answer"), _search_fn, max_loops=1
        )

        assert result["synthesis_status"] == "ok"
        assert result["answer"] == "a synthesized answer"
