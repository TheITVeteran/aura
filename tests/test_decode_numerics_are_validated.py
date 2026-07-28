"""Decode-path math must not raise, and must say when it did nothing.

CP126 (high), core/brain/llm/contrastive_decoding.py — two findings.

    "NumPy primitives do not validate empty or non-finite logits. max,
     exponentiation, and normalization run on arbitrary arrays without
     checking nonempty rank, finite values, or positive normalizer. Empty
     arrays raise and NaN/Infinity can produce invalid masks or output;
     steering_combine_np has no fail-open exception envelope."

    "Fail-open degradation is not surfaced to the generation receipt.
     Processors return original logits or None with only debug logging.
     Callers receive no status indicating contrastive/steering coverage,
     failure count, fallback reason, or whether advertised reasoning
     steering causally affected output."

Both are reachable and they fail differently. ``np.max`` on a zero-size
array raises ValueError, which kills the generation. A single NaN makes the
softmax NaN everywhere, so the plausibility mask selects nothing and every
token becomes -inf — that one does not raise, it produces garbage text, and
it is harder to notice because it looks like a bad answer rather than a bug.

The second finding is why the first went unnoticed: a processor that
silently returns its input is indistinguishable from one that is working,
so "reasoning steering is applied" was an unfalsifiable claim.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.brain.llm.contrastive_decoding import (
    contrastive_combine_np,
    decode_health,
    plausible_mask_np,
    reset_decode_health,
    steering_combine_np,
)

GOOD = np.array([2.0, 1.0, 0.5, -1.0])
AMATEUR = np.array([0.1, 0.2, 0.3, 0.4])


@pytest.fixture(autouse=True)
def _clean_health():
    reset_decode_health()
    yield
    reset_decode_health()


class TestNothingOnTheDecodePathRaises:
    @pytest.mark.parametrize(
        ("smart", "amateur"),
        [
            (np.array([]), np.array([])),
            (np.array([1.0, np.nan, 2.0]), np.array([1.0, 2.0, 3.0])),
            (np.array([1.0, 2.0, 3.0]), np.array([np.inf, 1.0, 2.0])),
            (np.array([np.inf, 1.0]), np.array([1.0, 2.0])),
            (GOOD, np.array([1.0, 2.0])),          # shape mismatch
            (np.array([-np.inf, -np.inf]), np.array([1.0, 2.0])),
        ],
    )
    def test_contrastive_combine_never_raises(self, smart, amateur):
        result = contrastive_combine_np(smart, amateur)
        assert isinstance(result, np.ndarray)

    @pytest.mark.parametrize(
        ("logits", "bias"),
        [
            (np.array([]), {0: 1.0}),
            (np.array([np.nan, 1.0]), {0: 1.0}),
            (np.array([np.inf, 1.0]), {0: 1.0}),
            (GOOD, {0: float("nan")}),
            (GOOD, {0: float("inf")}),
            (GOOD, {"not-an-int": 1.0}),
            (GOOD, {999999: 1.0}),
        ],
    )
    def test_steering_combine_never_raises(self, logits, bias):
        """steering_combine_np previously had no envelope at all."""
        result = steering_combine_np(logits, bias)
        assert isinstance(result, np.ndarray)

    def test_the_empty_array_case_that_used_to_raise(self):
        """np.max on a zero-size array is a ValueError, not a bad answer."""
        assert contrastive_combine_np(np.array([]), np.array([])).size == 0


class TestARefusalLeavesTheCallerUntouched:
    def test_bad_input_returns_the_original_logits(self):
        smart = np.array([1.0, np.nan, 2.0])
        result = contrastive_combine_np(smart, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(result, smart)

    def test_bad_steering_returns_the_original_logits(self):
        logits = np.array([np.nan, 1.0])
        np.testing.assert_array_equal(steering_combine_np(logits, {0: 1.0}), logits)

    def test_an_unusable_bias_does_not_corrupt_the_distribution(self):
        result = steering_combine_np(GOOD, {0: float("nan")})
        assert np.all(np.isfinite(result))
        np.testing.assert_array_equal(result, GOOD)


class TestHealthyInputStillWorks:
    """Over-refusal would silently disable the feature, which is the state
    the coverage receipt exists to make visible."""

    def test_contrastive_changes_the_distribution(self):
        result = contrastive_combine_np(GOOD, AMATEUR)
        assert not np.array_equal(result, GOOD)
        assert np.any(np.isfinite(result))

    def test_steering_shifts_a_plausible_token(self):
        result = steering_combine_np(GOOD, {0: 1.5})
        assert result[0] > GOOD[0]

    def test_the_plausibility_mask_selects_something(self):
        mask = plausible_mask_np(GOOD, 0.1)
        assert mask.any()

    def test_a_healthy_call_is_counted_as_applied(self):
        contrastive_combine_np(GOOD, AMATEUR)
        assert decode_health()["applied"] == 1


class TestCoverageIsReportable:
    """The second finding: a processor that silently returns its input is
    indistinguishable from one that is working."""

    def test_noops_are_counted_with_reasons(self):
        contrastive_combine_np(np.array([]), np.array([]))
        steering_combine_np(np.array([np.nan, 1.0]), {0: 1.0})
        health = decode_health()
        assert health["noops"] == 2
        assert health["reasons"]["contrastive"] == 1
        assert health["reasons"]["steering"] == 1

    def test_coverage_reflects_the_mix(self):
        contrastive_combine_np(GOOD, AMATEUR)          # applied
        contrastive_combine_np(GOOD, np.array([1.0]))  # shape mismatch
        health = decode_health()
        assert health["calls"] == 2
        assert health["coverage"] == pytest.approx(0.5)

    def test_the_last_reason_is_retained(self):
        contrastive_combine_np(GOOD, np.array([1.0]))
        assert "shape_mismatch" in decode_health()["last_reason"]

    def test_zero_calls_reports_zero_coverage_not_full(self):
        """An unexercised processor has not achieved 100% coverage."""
        health = decode_health()
        assert health["calls"] == 0
        assert health["coverage"] == 0.0

    def test_the_report_is_serializable(self):
        contrastive_combine_np(GOOD, AMATEUR)
        payload = decode_health()
        assert payload["schema"] == "aura.contrastive_decode_health.v1"
        assert set(payload) >= {"applied", "noops", "calls", "coverage", "reasons"}
