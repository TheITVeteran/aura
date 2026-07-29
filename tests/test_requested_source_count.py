"""She read five articles when Bryan asked for three.

Two bugs, one measurement. The live log for "find 3 recent articles about
orcas" said:

    Deep research gathered 5 source(s) over 1 quer(ies) in 0.0s
    Deep research complete: 1 loops, 1 queries, 5 sources, 82.4s

Gathering rounds to zero — the entire cost of that step is the local model
READING what was gathered. So two of those five articles were pure latency
for material nobody asked for, and the document cited more sources than the
request wanted.

And the count itself was wrong: "3 RECENT articles" matched no pattern,
because the regex demanded the number sit directly against the noun. It fell
through to 1, so a request for three sources was validated against one. Only
the adjective "different" had ever been allowed through.

There were also two copies of the parser — the visible-tab count and the
research count — carrying the same regex with different fallbacks, so fixing
a phrasing in one left the other wrong. There is one now.
"""

from __future__ import annotations

import pytest

from core.skills.desktop_task import DesktopTaskSkill


class TestTheCountComesFromTheRequest:
    @pytest.mark.parametrize(
        "objective,expected",
        [
            ("find 3 recent articles about orcas online", 3),
            ("find 2 articles about orcas", 2),
            ("find 5 credible sources about orcas", 5),
            ("pull up 4 top stories about orcas", 4),
            ("find three different recent articles", 3),
            ("find several articles about orcas", 3),
            ("find a couple of good articles about orcas", 2),
        ],
    )
    def test_the_number_asked_for_is_the_number_used(self, objective, expected):
        assert DesktopTaskSkill._requested_research_source_count(objective) == expected

    def test_an_adjective_does_not_break_the_count(self):
        """The measured bug: "3 recent articles" parsed as 1."""
        assert (
            DesktopTaskSkill._requested_research_source_count(
                "find 3 recent articles about orcas online"
            )
            == 3
        )

    def test_no_number_asked_means_no_number_claimed(self):
        assert (
            DesktopTaskSkill._requested_research_source_count(
                "write a summary about orcas"
            )
            == 0
        )

    def test_the_count_is_bounded(self):
        """A request for forty sources is not a licence to read forty."""
        assert (
            DesktopTaskSkill._requested_research_source_count(
                "find 40 articles about orcas"
            )
            <= 5
        )


class TestOneParserNotTwo:
    """They had diverged: same regex, different fallbacks."""

    @pytest.mark.parametrize(
        "objective",
        [
            "open 3 recent articles about orcas",
            "pull up two different sources about orcas",
            "show me several stories about orcas",
        ],
    )
    def test_both_counts_agree_on_the_same_phrase(self, objective):
        visible = DesktopTaskSkill._requested_visible_source_count(objective)
        research = DesktopTaskSkill._requested_research_source_count(objective)
        assert visible == research, (visible, research)

    def test_the_visible_count_still_needs_an_opening_verb(self):
        """"find 3 articles and write a PDF" opens no tabs."""
        assert (
            DesktopTaskSkill._requested_visible_source_count(
                "find 3 recent articles about orcas and write a PDF"
            )
            == 0
        )


class TestTheFetchFollowsTheRequest:
    def test_the_search_asks_for_what_was_requested_plus_one_spare(self):
        """The spare covers a dead link — the reason a margin existed at all.
        Five-for-three was not a margin, it was two extra articles read."""
        import inspect

        source = inspect.getsource(DesktopTaskSkill)
        assert "_requested_research_source_count(objective)" in source
        assert "min(5, requested + 1) if requested else 5" in source
