"""Measurement contracts for the recurrent-memory language canary."""

from __future__ import annotations

from tools import run_mathematics_memory_decode_canary as canary


def test_canary_separates_ordinary_and_matched_wire_baselines() -> None:
    assert canary.ARMS[:3] == (
        "ordinary_base",
        "matched_wire_base",
        "treatment",
    )
    assert len(canary.ARMS) == len(set(canary.ARMS)) == 8


def test_claim_boundary_is_bound_by_identity_not_a_stale_model_name() -> None:
    assert "model_identity" in canary.CLAIM_BOUNDARY
    assert "1.5B" not in canary.CLAIM_BOUNDARY
    assert "32B" not in canary.CLAIM_BOUNDARY
    assert "WOW" in canary.CLAIM_BOUNDARY


def test_summary_does_not_charge_structural_prefill_as_generation() -> None:
    rows = [
        {
            "arm": "treatment",
            "correct": True,
            "parsed": True,
            "prompt_tokens": 10,
            "generated_tokens": 7,
            "wire_prefill_tokens": 5,
            "latency_ms": 3,
        },
        {
            "arm": "treatment",
            "correct": False,
            "parsed": True,
            "prompt_tokens": 12,
            "generated_tokens": 9,
            "wire_prefill_tokens": 5,
            "latency_ms": 5,
        },
    ]

    summary = canary._summary(rows, "treatment")  # noqa: SLF001

    assert summary["exact_accuracy"] == 0.5
    assert summary["mean_generated_tokens"] == 8
    assert summary["mean_wire_prefill_tokens"] == 5
