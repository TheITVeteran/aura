"""A file excerpt must contain what was asked about, not the file's first page.

LIVE DEFECT, 2026-08-10. Asked "in your own source there is a file
core/soma/resilience_engine.py — read it and tell me what happens to the
depletion value when record_success is called; quote the line that changes it,
or tell me plainly that no line does", she answered with source from an
entirely different module, failed her own reply gate on
``incomplete_code_response``, and shipped the draft anyway.

Everything upstream had worked. The path was extracted, the file was read, and
the log recorded "Chat preflight: loaded 1 referenced file(s) into context."
What reached her was the first 5,461 characters of a 20,428 character file —
and ``def record_success`` begins at character 5,815. The excerpt stopped 354
characters short of the only region the question was about, said "truncated",
and she generated the rest.

Position is not relevance.
"""

from __future__ import annotations

import pytest
from tests.chat_lane_support import chat_lane_source


LIVE_MESSAGE = (
    "in your own source there is a file core/soma/resilience_engine.py. read "
    "it and tell me what happens to the depletion value when record_success "
    "is called - quote the line that changes it, or tell me plainly that no "
    "line does."
)


def _block(message: str) -> str:
    from core.conversation.chat_preflight import (
        build_file_context_block,
        extract_file_references,
    )

    return build_file_context_block(
        extract_file_references(message), query=message
    )


def test_the_live_question_now_reaches_its_own_answer() -> None:
    block = _block(LIVE_MESSAGE)

    assert "def record_success" in block
    # The line that actually answers "what happens to depletion".
    assert "depletion_release" in block


def test_excerpt_is_line_numbered_so_a_quote_is_checkable() -> None:
    import re

    block = _block(LIVE_MESSAGE)

    assert re.search(r"^\d+\t", block, re.MULTILINE)


def test_omitted_ranges_are_named_not_merely_flagged() -> None:
    """"truncated" told her something was missing, never which part."""
    block = _block(LIVE_MESSAGE)

    assert "not included in this excerpt" in block


def test_the_header_forbids_reconstructing_an_omitted_region() -> None:
    block = _block(LIVE_MESSAGE)

    assert "NOT read" in block
    assert "instead of" in block


def test_budget_is_still_respected() -> None:
    from core.conversation.chat_preflight import (
        FILE_READ_BUDGET,
        MAX_FILES_PER_TURN,
        load_referenced_files,
    )

    loaded = load_referenced_files(
        ["core/soma/resilience_engine.py"], query=LIVE_MESSAGE
    )
    assert loaded
    per_file_budget = max(1024, FILE_READ_BUDGET // max(1, MAX_FILES_PER_TURN))
    # Rendering adds line numbers and omission markers on top of the raw lines
    # it selected, so allow for that rather than asserting the raw ceiling.
    assert len(loaded[0][1]) < per_file_budget * 2


def test_every_named_term_gets_a_window() -> None:
    """Round-robin, not greedy: the first term must not eat the budget.

    Seating `record_success` alone was not enough — the line that answers the
    question sits ~60 lines further down, inside the region a greedy first
    term had already spent the budget on.
    """
    from core.conversation.chat_preflight import _relevant_excerpt

    lines = [f"line {index}\n" for index in range(400)]
    lines[10] = "def alpha_function():\n"
    lines[300] = "    beta_marker = 1\n"

    excerpt = _relevant_excerpt(
        lines, query="alpha_function and beta_marker", budget=2000
    )

    assert "alpha_function" in excerpt
    assert "beta_marker" in excerpt


def test_a_question_naming_nothing_findable_falls_back_to_the_head() -> None:
    """"what does this file do" is correctly answered by the top of it."""
    from core.conversation.chat_preflight import _relevant_excerpt

    lines = [f"line {index}\n" for index in range(400)]

    excerpt = _relevant_excerpt(lines, query="what does this do", budget=300)

    assert "1\tline 0" in excerpt


def test_a_small_file_is_returned_whole(tmp_path, monkeypatch) -> None:
    """The fix must not start excerpting files that already fit."""
    from core.conversation import chat_preflight

    target = tmp_path / "small.py"
    target.write_text("alpha = 1\nbeta = 2\ngamma = 3\n", encoding="utf-8")
    monkeypatch.setattr(
        chat_preflight, "_resolve_safely", lambda ref: target
    )

    loaded = chat_preflight.load_referenced_files(["small.py"], query="beta")

    assert loaded[0][1] == "alpha = 1\nbeta = 2\ngamma = 3\n"
    assert "not included" not in loaded[0][1]


@pytest.mark.parametrize("query", ["", "beta_marker", "what does this do"])
def test_content_with_no_line_structure_is_never_dropped(query: str) -> None:
    """The regression this fix introduced, and the reason head-prefix existed.

    A minified bundle, a one-line JSON blob, or any file without newlines has
    no whole line that fits a budget. The first draft of the relevance excerpt
    returned an EMPTY string for those — losing the file silently, which is
    strictly worse than the head prefix it replaced.
    """
    from core.conversation.chat_preflight import _relevant_excerpt

    single_line = ["x" * 10_000]

    excerpt = _relevant_excerpt(single_line, query=query, budget=1024)

    assert excerpt, "a file with no newlines must not vanish"
    assert len(excerpt) < 1400
    assert "truncated" in excerpt


def test_query_terms_ignore_the_question_scaffolding() -> None:
    from core.conversation.chat_preflight import _excerpt_query_terms

    terms = _excerpt_query_terms(LIVE_MESSAGE)

    assert "record_success" in terms
    assert "depletion" in terms
    for scaffold in ("read", "tell", "quote", "value", "happens", "source"):
        assert scaffold not in terms, scaffold


def test_the_route_passes_the_message_as_the_query() -> None:
    """A relevance excerpt with no query is just the head again."""
    source = chat_lane_source()
    # Anchor on the invocation, not the import list.
    index = source.find("operation_name=\"referenced_file_context\"")
    assert index != -1, "referenced-file preflight call site not found"
    call = source[max(0, index - 600) : index]

    assert "build_file_context_block" in call
    assert "query=body.message" in call
