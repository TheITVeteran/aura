"""Manner words say WHERE to look, not WHAT to look for.

Measured live. "find 3 recent articles about orcas online" was searched as the
literal phrase "orcas online" — which is a wireless ISP on Orcas Island,
Washington. The PDF Aura wrote was a competent, well-sourced summary of that
company's vacation-hold feature and password-expiry policy. Every mechanism
worked; the topic was wrong.
"""

from __future__ import annotations

import pytest

from core.skills.desktop_task import DesktopTaskSkill


@pytest.mark.parametrize(
    "objective,expected",
    [
        (
            "Then find 3 recent articles about orcas online, read them, and "
            "write a synthesis into a PDF",
            "orcas",
        ),
        ("find 3 recent articles about ocean warming online", "ocean warming"),
        ("search for climate news on the internet", "climate news"),
        ("find articles about orcas", "orcas"),
    ],
)
def test_the_topic_excludes_where_to_look(objective, expected):
    assert DesktopTaskSkill._extract_search_query(objective) == expected


def test_online_inside_a_name_is_not_stripped():
    """"the online safety act" is a topic, not a manner."""
    assert (
        DesktopTaskSkill._extract_search_query("look up the online safety act")
        == "the online safety act"
    )
