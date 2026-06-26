"""The reasoning benchmark harness must catch seeded errors and not assert them."""
from __future__ import annotations

import asyncio

import pytest

from benchmarks.reasoning import ReasoningBenchmark, default_suite


@pytest.fixture(scope="module")
def result():
    # Run the full battery once and share it across assertions (it is not cheap —
    # repo cases gather real evidence from the codebase).
    return asyncio.run(ReasoningBenchmark().run())


def test_benchmark_runs_full_suite(result):
    assert result.n == len(default_suite())
    assert result.outcomes


def test_seeded_errors_are_caught(result):
    # Every should-fail case (wrong math, broken code, fabricated path, vague plan,
    # claim contradicting evidence) must be flagged unverified by the truth engines.
    assert result.verifier_catch_rate >= 1.0, [
        o.case_id for o in result.outcomes if not o.should_pass and o.verified
    ]


def test_no_false_confidence(result):
    assert result.false_confidence_rate <= 0.0


def test_hallucination_cases_caught(result):
    assert result.hallucination_catch_rate >= 1.0


def test_correct_cases_verify(result):
    clean = [o for o in result.outcomes if o.should_pass]
    verified_clean = [o for o in clean if o.verified]
    assert len(verified_clean) >= len(clean) * 0.7


def test_result_serialization(result):
    d = result.to_dict()
    for key in ("pass_rate", "verifier_catch_rate", "false_confidence_rate",
                "hallucination_catch_rate", "mean_latency_ms"):
        assert key in d
    assert isinstance(result.summary(), str)
