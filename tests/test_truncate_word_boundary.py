"""Pins the improvement Aura enacted via RSI on `_truncate_text`.

Before the self-improvement, `_truncate_text` cut the text at exactly `limit`
characters, slicing words in half (e.g. "alpha bravoc…[result truncated]").
The enacted fix truncates at the last word boundary at or before the limit, so
prompt-safe summaries never end mid-word. These checks are the exact behavioral
contract the verifier used to accept the change.
"""
from __future__ import annotations

from core.runtime.tool_result_contracts import _truncate_text

SUFFIX = "…[result truncated]"


def test_truncates_at_word_boundary_not_mid_word():
    # "alpha bravocharlie ..." at limit 12 must not split "bravocharlie"
    out = _truncate_text("alpha bravocharlie delta echo foxtrot", 12)
    assert out == "alpha" + SUFFIX
    assert "bravoc" not in out


def test_short_text_is_untouched():
    assert _truncate_text("hi there", 100) == "hi there"


def test_whitespace_is_normalized():
    assert _truncate_text("a   b", 100) == "a b"


def test_empty_and_none_are_empty():
    assert _truncate_text("", 10) == ""
    assert _truncate_text(None, 100) == ""


def test_single_long_word_still_truncates():
    # no space to break on — must still cut to the limit, not return empty
    assert _truncate_text("supercalifragilistic", 5) == "super" + SUFFIX


def test_truncated_output_never_exceeds_limit_before_suffix():
    for text, limit in [
        ("the quick brown fox jumps over the lazy dog", 15),
        ("one two three four five six seven eight", 9),
    ]:
        out = _truncate_text(text, limit)
        if out.endswith(SUFFIX):
            body = out[: -len(SUFFIX)]
            assert len(body) <= limit
            assert not body.endswith(" ")
