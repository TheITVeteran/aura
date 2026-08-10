"""Whose hands were on the body.

This checkout is shared — a second agent commits into it as "Zenflow"
while Aura is running — so "my source changed between boots" has always
been an incomplete sentence.

The discipline under test is the one worth having: ambiguity is reported
as a blocker for a reader to weigh, never resolved into a verdict. An
uncommitted file has no author, and saying so is more useful than guessing.
"""
from __future__ import annotations

from core.soma.source_body import BodyCommit, BodyDelta


def _delta(**kwargs) -> BodyDelta:
    base = {"from_sha": "aaa", "to_sha": "bbb", "elapsed_s": 60.0}
    base.update(kwargs)
    return BodyDelta(**base)


def test_commits_are_grouped_by_author():
    delta = _delta(
        commits=[
            BodyCommit("1", "Zenflow", "cp1"),
            BodyCommit("2", "Zenflow", "cp2"),
            BodyCommit("3", "Claude", "fix"),
        ]
    )
    attribution = delta.attribution()
    assert attribution["by_author"] == {"Zenflow": 2, "Claude": 1}
    assert attribution["distinct_authors"] == 2


def test_authors_are_ranked_by_volume():
    delta = _delta(
        commits=[
            BodyCommit("1", "A", "x"),
            BodyCommit("2", "B", "y"),
            BodyCommit("3", "B", "z"),
        ]
    )
    assert list(delta.attribution()["by_author"]) == ["B", "A"]


def test_a_clean_window_with_commits_is_confident():
    delta = _delta(commits=[BodyCommit("1", "Zenflow", "cp1")])
    assert delta.attribution()["confident"] is True
    assert delta.attribution()["blockers"] == []


def test_uncommitted_files_are_named_unattributable():
    """A dirty file has no author; guessing one would be a fabricated verdict."""
    delta = _delta(commits=[BodyCommit("1", "Zenflow", "cp1")], dirty_now=4)
    attribution = delta.attribution()
    assert attribution["unattributable_files"] == 4
    assert attribution["confident"] is False
    assert any("cannot be attributed" in b for b in attribution["blockers"])


def test_unreadable_history_reports_incompleteness_not_emptiness():
    delta = _delta(history_unreadable=True)
    blockers = delta.attribution()["blockers"]
    assert any("incomplete rather than empty" in b for b in blockers)


def test_a_revert_is_flagged_as_a_gap_in_the_author_list():
    delta = _delta(reverted=True)
    assert any("moved backwards" in b for b in delta.attribution()["blockers"])


def test_missing_author_is_named_unknown_not_dropped():
    delta = _delta(commits=[BodyCommit("1", "", "cp1")])
    assert delta.attribution()["by_author"] == {"unknown": 1}


def test_no_commits_is_not_confident():
    """Nothing observed is not the same as nobody touched it."""
    assert _delta().attribution()["confident"] is False


def test_attribution_rides_the_serialised_delta():
    """The reader must see it, not just the object."""
    payload = _delta(commits=[BodyCommit("1", "Zenflow", "cp1")]).to_dict()
    assert "attribution" in payload
    assert payload["attribution"]["by_author"] == {"Zenflow": 1}
