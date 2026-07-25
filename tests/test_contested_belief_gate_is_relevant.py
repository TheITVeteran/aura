"""A contest about one subject must not block writes about every other subject.

Live 2026-07-25: 71 autonomous knowledge writes deferred in one hour on
``epistemic_reconciliation_required:2`` — two contested claims gating every
unrelated fact and concept she learned. The freshness window added earlier
fixed contests that never aged out; it did not make the gate about relevance,
so two fresh contests still wedged the whole knowledge lane.

A contest is evidence that ONE subject is unsettled. It is not evidence that
everything is. The gate stays fully closed on the contested subject and fails
closed whenever relevance cannot be judged.
"""
from __future__ import annotations

import pytest

from core.executive.executive_core import ExecutiveCore, Intent

pytestmark = pytest.mark.unit


def _intent(goal="", **payload):
    return Intent(goal=goal, payload=dict(payload))


def _epistemic(*keys):
    return {"contested": len(keys), "contested_keys": list(keys)}


class TestRelevance:
    def test_a_write_about_the_contested_subject_is_still_gated(self):
        assert ExecutiveCore._intent_touches_contested_topic(
            _intent(goal="update what I know about mlx memory limits"),
            _epistemic("runtime:mlx memory limits"),
        )

    def test_an_unrelated_write_is_not_gated(self):
        assert not ExecutiveCore._intent_touches_contested_topic(
            _intent(goal="record that Kyoto is the old capital of Japan"),
            _epistemic("runtime:mlx memory limits"),
        )

    def test_the_namespace_alone_counts_as_a_match(self):
        assert ExecutiveCore._intent_touches_contested_topic(
            _intent(goal="a note about the runtime lane"),
            _epistemic("runtime:mlx memory limits"),
        )

    def test_payload_content_is_searched_not_just_the_goal(self):
        assert ExecutiveCore._intent_touches_contested_topic(
            _intent(goal="", content="a long note discussing mlx memory limits"),
            _epistemic("runtime:mlx memory limits"),
        )

    def test_a_short_token_cannot_match_by_accident(self):
        """Namespaces like 'ai' would otherwise match nearly any sentence."""
        assert not ExecutiveCore._intent_touches_contested_topic(
            _intent(goal="the rain in spain"),
            _epistemic("ai:x"),
        )


class TestFailsClosed:
    def test_contests_with_no_keys_keep_the_old_global_behaviour(self):
        """An authority that predates key reporting must not silently open up."""
        assert ExecutiveCore._intent_touches_contested_topic(
            _intent(goal="anything at all"), {"contested": 2, "contested_keys": []}
        )

    def test_an_intent_with_nothing_to_judge_stays_gated(self):
        assert ExecutiveCore._intent_touches_contested_topic(
            _intent(goal=""), _epistemic("runtime:mlx memory limits")
        )

    def test_missing_keys_field_entirely_stays_gated(self):
        assert ExecutiveCore._intent_touches_contested_topic(
            _intent(goal="anything"), {"contested": 1}
        )


class TestTheAuthorityReportsKeys:
    def test_summary_names_the_fresh_contested_beliefs(self):
        from core.constitution import BeliefAuthority

        authority = BeliefAuthority()
        authority.review_update("runtime", "resident model", "32B")
        authority.review_update("runtime", "resident model", "7B")

        summary = authority.summary()
        assert summary["fresh_contested"] == 1, "the conflict must actually contest"
        assert summary["fresh_contested_keys"] == ["runtime:resident_model"], (
            "a contested count with no keys can only gate globally"
        )

    def test_the_key_list_is_bounded(self):
        from core.constitution import BeliefAuthority

        authority = BeliefAuthority()
        for i in range(80):
            authority.review_update("bulk", f"claim {i}", "a")
            authority.review_update("bulk", f"claim {i}", "b")

        assert len(authority.summary().get("fresh_contested_keys") or []) <= 32
