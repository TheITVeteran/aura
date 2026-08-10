"""Ruts that no per-step check can see.

Every step in these histories succeeds. That is exactly why the two existing
gateway guards miss them: they veto an action that failed twice, and a loop of
successful actions never trips that.

The waiting tests are the important ones. The prior art shipped loop detection
and then had to fix it because it killed agents legitimately polling a
long-running build. Shipping their bug before their fix would waste the reading.
"""
from __future__ import annotations

import pytest

from core.agency.stuck_detector import (
    AgentStep,
    Remedy,
    StuckDetector,
    StuckPattern,
)

pytestmark = pytest.mark.unit


def _tool(action, observation="done", *, is_error=False, args="", progress=None):
    return AgentStep(
        action=action, arguments=args, observation=observation,
        is_error=is_error, kind="tool", progress_marker=progress,
    )


def _message(text="thinking out loud"):
    return AgentStep(action="message", observation=text, kind="message")


def _wait(action="poll_build", observation="still running", progress=None):
    return AgentStep(
        action=action, observation=observation, kind="wait", progress_marker=progress,
    )


# ── a healthy run is not stuck ──────────────────────────────────────────────


def test_an_empty_history_is_not_stuck():
    assert not StuckDetector().check([])


def test_varied_work_is_not_stuck():
    steps = [_tool("read", "a"), _tool("edit", "b"), _tool("test", "c"),
             _tool("commit", "d")]

    assert not StuckDetector().check(steps)


def test_the_same_action_with_different_results_is_progress():
    """Running the same tool on different inputs is normal work."""
    steps = [_tool("read", f"file {i} contents") for i in range(8)]

    assert not StuckDetector().check(steps)


def test_repetition_below_the_threshold_is_tolerated():
    steps = [_tool("ls", "same")] * 3

    assert not StuckDetector(repeat_threshold=4).check(steps)


# ── repeated action / identical observation ────────────────────────────────


def test_the_same_action_returning_the_same_thing_is_stuck():
    steps = [_tool("ls", "file_a\nfile_b")] * 4

    verdict = StuckDetector().check(steps)

    assert verdict.stuck
    assert verdict.pattern is StuckPattern.REPEATED_ACTION_OBSERVATION


def test_a_successful_loop_is_caught_even_though_no_step_failed():
    """The gap the existing gateway guards leave: every step returns success."""
    steps = [_tool("ls", "ok", is_error=False)] * 5

    assert StuckDetector().check(steps).stuck


def test_volatile_detail_does_not_defeat_the_match():
    """A result differing only by a timestamp is the same result."""
    steps = [
        _tool("status", f"2026-08-09T10:00:0{i} nothing to commit") for i in range(4)
    ]

    assert StuckDetector().check(steps).stuck


def test_the_verdict_names_the_offending_action():
    steps = [_tool("ls", "same")] * 4

    verdict = StuckDetector().check(steps)

    assert "ls" in verdict.evidence
    assert "ls" in verdict.detail


# ── repeated errors ────────────────────────────────────────────────────────


def test_the_same_failure_three_times_is_stuck():
    steps = [_tool("build", "ModuleNotFoundError: foo", is_error=True)] * 3

    verdict = StuckDetector().check(steps)

    assert verdict.pattern is StuckPattern.REPEATED_ACTION_ERROR


def test_the_error_verdict_says_retrying_will_not_help():
    steps = [_tool("build", "same error", is_error=True)] * 3

    assert "same way" in StuckDetector().check(steps).detail


def test_different_errors_are_not_a_rut():
    steps = [
        _tool("build", "error one", is_error=True),
        _tool("build", "error two", is_error=True),
        _tool("build", "error three", is_error=True),
    ]

    assert not StuckDetector().check(steps).stuck


# ── oscillation ────────────────────────────────────────────────────────────


def test_ping_ponging_between_two_actions_is_stuck():
    steps = [_tool("edit", "written"), _tool("test", "failed")] * 3

    verdict = StuckDetector().check(steps)

    assert verdict.pattern is StuckPattern.OSCILLATION


def test_the_oscillation_verdict_names_both_actions():
    steps = [_tool("edit", "written"), _tool("test", "failed")] * 3

    verdict = StuckDetector().check(steps)

    assert set(verdict.evidence) == {"edit", "test"}


def test_two_cycles_is_not_yet_oscillation():
    steps = [_tool("edit", "written"), _tool("test", "failed")] * 2

    assert not StuckDetector(oscillation_cycles=3).check(steps).stuck


def test_a_three_way_rotation_is_not_reported_as_oscillation():
    """A, B, C, A, B, C is a cycle but not the two-move ping-pong pattern."""
    steps = [_tool("a", "1"), _tool("b", "2"), _tool("c", "3")] * 2

    verdict = StuckDetector().check(steps)

    assert verdict.pattern is not StuckPattern.OSCILLATION


# ── monologue ──────────────────────────────────────────────────────────────


def test_consecutive_messages_with_no_work_is_stuck():
    steps = [_tool("read", "ok"), _message("a"), _message("b"), _message("c")]

    verdict = StuckDetector().check(steps)

    assert verdict.pattern is StuckPattern.MONOLOGUE


def test_a_message_after_real_work_is_not_a_monologue():
    steps = [_message("a"), _tool("read", "ok"), _message("b")]

    assert not StuckDetector().check(steps).stuck


def test_only_trailing_messages_count():
    steps = [_message("a"), _message("b"), _tool("read", "ok"), _message("c")]

    assert not StuckDetector().check(steps).stuck


# ── context overflow is its own thing ──────────────────────────────────────


def test_repeated_context_window_errors_are_their_own_pattern():
    steps = [
        _tool("chat", "Error: context window exceeded (32768 tokens)", is_error=True),
        _tool("chat", "Error: context window exceeded (32768 tokens)", is_error=True),
    ]

    verdict = StuckDetector().check(steps)

    assert verdict.pattern is StuckPattern.REPEATED_CONTEXT_OVERFLOW


def test_the_overflow_remedy_is_not_a_nudge():
    """No amount of rephrasing helps; the window has to get smaller."""
    steps = [
        _tool("chat", "context length exceeded", is_error=True),
        _tool("chat", "context length exceeded", is_error=True),
    ]

    verdict = StuckDetector().check(steps)

    assert verdict.remedy is Remedy.FORCE_NEW_STRATEGY


def test_overflow_outranks_generic_repeated_error():
    steps = [_tool("chat", "context window token limit", is_error=True)] * 4

    assert StuckDetector().check(steps).pattern is StuckPattern.REPEATED_CONTEXT_OVERFLOW


# ── waiting is not stuck: the inherited incident ───────────────────────────


def test_polling_a_long_running_build_is_not_stuck():
    """The documented production bug in the prior art. Waiting IS repeating,
    and it is also correct."""
    steps = [_wait("poll_build", "still running")] * 10

    assert not StuckDetector().check(steps).stuck


def test_a_wait_does_not_break_up_a_genuine_rut():
    """Interleaving a poll must not launder a real loop."""
    steps = []
    for _ in range(4):
        steps.append(_tool("ls", "same"))
        steps.append(_wait("poll_build", "still running"))

    assert StuckDetector().check(steps).stuck


def test_a_changing_progress_marker_means_it_is_not_a_repeat():
    """Identical action and output, but the build advanced."""
    steps = [
        _tool("check_build", "running", progress=f"line {i}") for i in range(6)
    ]

    assert not StuckDetector().check(steps).stuck


def test_a_frozen_progress_marker_is_still_a_rut():
    steps = [_tool("check_build", "running", progress="line 42")] * 5

    assert StuckDetector().check(steps).stuck


# ── recovery escalates ─────────────────────────────────────────────────────


def test_the_first_finding_is_a_nudge():
    detector = StuckDetector()

    assert detector.check([_tool("ls", "same")] * 4).remedy is Remedy.NUDGE


def test_repeated_findings_escalate_rather_than_repeating_a_nudge():
    """One nudge, then escalate. Re-nudging is itself a loop, which would be a
    poor look for the loop detector."""
    detector = StuckDetector()
    steps = [_tool("ls", "same")] * 4

    remedies = [detector.check(steps).remedy for _ in range(4)]

    assert remedies == [
        Remedy.NUDGE,
        Remedy.FORCE_NEW_STRATEGY,
        Remedy.ASK_HUMAN,
        Remedy.ASK_HUMAN,
    ]


def test_escalation_never_ends_in_an_unrecoverable_halt():
    """Their other fix: a hard error state the agent could not be talked out of
    turned a recoverable rut into a dead session."""
    detector = StuckDetector()
    steps = [_tool("ls", "same")] * 4

    for _ in range(10):
        verdict = detector.check(steps)

    assert verdict.remedy is Remedy.ASK_HUMAN
    assert verdict.stuck


def test_resetting_after_real_progress_returns_to_a_nudge():
    detector = StuckDetector()
    steps = [_tool("ls", "same")] * 4
    for _ in range(3):
        detector.check(steps)

    detector.reset()

    assert detector.check(steps).remedy is Remedy.NUDGE


def test_interventions_are_counted():
    detector = StuckDetector()
    steps = [_tool("ls", "same")] * 4

    detector.check(steps)
    detector.check(steps)

    assert detector.interventions == 2


# ── guards ─────────────────────────────────────────────────────────────────


def test_the_window_bounds_what_is_examined():
    """Old repetition that has since been escaped is not a current rut."""
    steps = [_tool("ls", "same")] * 4 + [_tool(f"step{i}", f"out{i}") for i in range(20)]

    assert not StuckDetector(window=10).check(steps).stuck


@pytest.mark.parametrize("kwargs", [
    {"window": 1},
    {"repeat_threshold": 1},
    {"error_threshold": 0},
    {"monologue_threshold": 1},
    {"oscillation_cycles": 1},
])
def test_thresholds_that_cannot_describe_a_repetition_are_refused(kwargs):
    with pytest.raises(ValueError):
        StuckDetector(**kwargs)


def test_a_verdict_is_falsy_when_not_stuck():
    assert not StuckDetector().check([_tool("a", "1"), _tool("b", "2")])


def test_describe_is_readable_in_both_states():
    detector = StuckDetector()

    assert detector.check([]).describe() == "not stuck"
    assert "ls" in detector.check([_tool("ls", "same")] * 4).describe()
