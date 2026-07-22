"""Source reliability updates must be normalized and mathematically sound."""

import pytest

from core.epistemics.source_ranker import SourceRanker


def test_positive_and_negative_outcomes_move_toward_observed_truth():
    ranker = SourceRanker()
    start = ranker.get_reliability("https://github.com/org/repo")
    ranker.record_outcome("https://github.com/org/repo", True)
    after_true = ranker.get_reliability("github.com")
    assert start < after_true < 0.99
    ranker.record_outcome("www.github.com/another/path", False)
    assert ranker.get_reliability("github.com") < after_true


def test_unknown_hosts_get_independent_calibration_cells():
    ranker = SourceRanker()
    ranker.record_outcome("https://evidence.example/path", True)
    assert ranker.get_reliability("evidence.example") > 0.5
    assert ranker.get_reliability("unseen.example") == 0.5
    assert ranker.evidence_count == {"evidence.example": 1}


def test_outcome_requires_boolean_ground_truth():
    ranker = SourceRanker()
    with pytest.raises(TypeError, match="boolean ground truth"):
        ranker.record_outcome("example.com", "false")  # type: ignore[arg-type]
