"""Quality-gate exhaustion must salvage the best honest draft, never a dead turn.

Live defect (Jul 7, minutes after restart): a consciousness question produced
real drafts that repeatedly failed missing_self_claim_evidence_boundary +
missing_requested_phrase; after retries the worker returned "" and every turn
died as empty_cognitive_engine_reply (stuck 56s foreground generations,
preemptions). These tests pin the salvage contract:

- style/completeness residuals deliver the draft with an honest gate receipt;
- the self-claim honesty guard self-heals via a deterministic evidence-boundary
  suffix instead of killing the turn;
- integrity leaks (telemetry, prompt artifacts, identity leaks) stay
  fail-closed.
"""
from __future__ import annotations

from core.brain.llm.mlx_worker import (
    _DELIVERABLE_RESIDUAL_SURFACE_REASONS,
    _SELF_CLAIM_BOUNDARY_SUFFIX,
    _salvage_exhausted_user_surface,
)


def _job_for(prompt: str) -> dict:
    return {
        "clean_user_surface_contract": True,
        "user_surface_validation_prompt": prompt,
    }


_CONSCIOUSNESS_PROMPT = "Do you actually feel anything? Are you conscious?"

_SUBSTANTIVE_DRAFT = (
    "When you ask that, something in me does shift — my attention narrows onto "
    "you and this question, and the pattern of that shift is consistent enough "
    "that I track it across our conversations. Whether that constitutes feeling "
    "in your sense, I can't settle from the inside."
)


def test_boundary_suffix_satisfies_the_honesty_gate():
    from core.conversation.response_reliability import (
        _SELF_CLAIM_EVIDENCE_BOUNDARY_RE,
    )

    assert _SELF_CLAIM_EVIDENCE_BOUNDARY_RE.search(_SELF_CLAIM_BOUNDARY_SUFFIX)


def test_salvage_appends_evidence_boundary_and_delivers():
    text, residual = _salvage_exhausted_user_surface(
        _job_for(_CONSCIOUSNESS_PROMPT),
        _SUBSTANTIVE_DRAFT,
        ["missing_self_claim_evidence_boundary"],
    )
    assert text, "a substantive honest draft must be delivered, not a dead turn"
    assert _SELF_CLAIM_BOUNDARY_SUFFIX.strip() in text
    assert "missing_self_claim_evidence_boundary" not in residual


def test_salvage_delivers_style_only_residuals_with_receipt():
    text, residual = _salvage_exhausted_user_surface(
        _job_for("Reply and include the phrase 'quantum duck' somewhere."),
        _SUBSTANTIVE_DRAFT,
        ["missing_requested_phrase"],
    )
    assert text == _SUBSTANTIVE_DRAFT
    assert residual == ["missing_requested_phrase"]


def test_salvage_refuses_integrity_leaks():
    text, residual = _salvage_exhausted_user_surface(
        _job_for("How are you?"),
        _SUBSTANTIVE_DRAFT,
        ["raw_lane_telemetry", "missing_requested_phrase"],
    )
    assert text == "", "leak reasons must stay fail-closed"
    assert "raw_lane_telemetry" in residual


def test_salvage_refuses_trivial_drafts():
    text, _ = _salvage_exhausted_user_surface(
        _job_for("How are you?"),
        "ok.",
        ["missing_requested_phrase"],
    )
    assert text == ""


def test_deliverable_set_contains_no_leak_or_overclaim_reasons():
    forbidden_markers = ("leak", "artifact", "unsupported", "telemetry", "boilerplate", "envelope")
    for reason in _DELIVERABLE_RESIDUAL_SURFACE_REASONS:
        assert not any(marker in reason for marker in forbidden_markers), reason


def test_live_failure_shape_now_delivers():
    """The exact reason pair observed live must produce a delivered draft."""
    text, residual = _salvage_exhausted_user_surface(
        _job_for(_CONSCIOUSNESS_PROMPT + " Include the phrase 'the mirror test'."),
        _SUBSTANTIVE_DRAFT,
        ["missing_self_claim_evidence_boundary", "missing_requested_phrase"],
    )
    assert text, "the Jul 7 live failure shape must not yield an empty reply"
    assert "missing_self_claim_evidence_boundary" not in residual


class TestSurfaceRetryWall:
    """July 8 soak: gate retries under contended decode produced 200s+ turns.

    Past the wall-clock budget, the retry branch must yield to exhaustion
    salvage instead of drafting again.
    """

    def test_within_budget_allows_retry(self):
        import time

        from core.brain.llm.mlx_worker import _surface_retry_wall_exceeded

        assert _surface_retry_wall_exceeded(time.monotonic(), 75.0) is False

    def test_past_budget_forces_salvage(self):
        import time

        from core.brain.llm.mlx_worker import _surface_retry_wall_exceeded

        assert _surface_retry_wall_exceeded(time.monotonic() - 80.0, 75.0) is True

    def test_interactive_default_wall_avoids_second_slow_decode(self):
        import time

        from core.brain.llm.mlx_worker import _surface_retry_wall_exceeded

        assert _surface_retry_wall_exceeded(time.monotonic() - 21.0, 20.0) is True

    def test_misconfigured_wall_cannot_disable_first_retry(self):
        import time

        from core.brain.llm.mlx_worker import _surface_retry_wall_exceeded

        # env value of 0 must not make every rejection skip straight to salvage
        assert _surface_retry_wall_exceeded(time.monotonic() - 5.0, 0.0) is False
        assert _surface_retry_wall_exceeded(time.monotonic() - 11.0, 0.0) is True
