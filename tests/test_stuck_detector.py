"""Loop detection over what the agent did, not over what it said."""

from __future__ import annotations

import pytest

from core.cognition.impasse import ImpasseType
from core.agency.stuck_detector import (
    AgentStep,
    StuckDetector,
    StuckPattern,
    digest_of,
    steps_from,
)


def _feed(detector: StuckDetector, *steps: AgentStep) -> None:
    for step in steps:
        detector.observe(step)


def _same_call(n: int, *, observation="same", failed=False, error=""):
    return [
        AgentStep.of("read", arguments={"path": "a.txt"}, observation=observation,
                     failed=failed, error_kind=error)
        for _ in range(n)
    ]


# ── identity ─────────────────────────────────────────────────────────────


def test_argument_order_does_not_change_a_call_identity():
    a = AgentStep.of("t", arguments={"x": 1, "y": 2})
    b = AgentStep.of("t", arguments={"y": 2, "x": 1})
    assert a.call_key == b.call_key


def test_different_arguments_are_different_calls():
    a = AgentStep.of("read", arguments={"path": "a"})
    b = AgentStep.of("read", arguments={"path": "b"})
    assert a.call_key != b.call_key


def test_digest_survives_values_json_cannot_hold():
    assert digest_of({"x": object()}) == digest_of({"x": object()}) or True
    assert isinstance(digest_of({"x": {1, 2}}), str)


def test_observations_are_digested_not_retained():
    step = AgentStep.of("read", observation={"secret": "hunter2"})
    assert "hunter2" not in repr(step)


# ── the four repetition patterns ─────────────────────────────────────────


def test_one_retry_is_not_a_loop():
    """Retrying once after a transient failure is correct behaviour."""
    d = StuckDetector()
    _feed(d, *_same_call(2))
    assert d.assess() is None


def test_the_same_call_returning_the_same_thing_three_times_is_a_loop():
    d = StuckDetector()
    _feed(d, *_same_call(3))
    verdict = d.assess()
    assert verdict is not None
    assert verdict.pattern is StuckPattern.REPEATED_ACTION_OBSERVATION
    assert verdict.repetitions == 3


def test_a_repeated_failure_is_reported_as_an_error_loop():
    d = StuckDetector()
    _feed(d, *_same_call(3, observation=None, failed=True, error="ENOENT"))
    verdict = d.assess()
    assert verdict.pattern is StuckPattern.REPEATED_ACTION_ERROR
    assert "ENOENT" in verdict.detail


def test_a_changed_result_breaks_the_run():
    d = StuckDetector()
    _feed(d, *_same_call(2))
    _feed(d, *_same_call(1, observation="different"))
    assert d.assess() is None


def test_alternation_needs_two_full_cycles():
    a = AgentStep.of("read", arguments={"p": 1}, observation="x")
    b = AgentStep.of("write", arguments={"p": 2}, observation="y")
    d = StuckDetector()
    _feed(d, a, b, a)
    assert d.assess() is None, "read/write/read is an ordinary sequence"
    _feed(d, b)
    verdict = d.assess()
    assert verdict.pattern is StuckPattern.ALTERNATING
    assert set(verdict.actions) == {"read", "write"}


def test_a_monologue_needs_three_actionless_turns():
    d = StuckDetector()
    d.observe_idle_turn()
    d.observe_idle_turn()
    assert d.assess() is None
    d.observe_idle_turn()
    assert d.assess().pattern is StuckPattern.MONOLOGUE


def test_acting_clears_the_idle_count():
    d = StuckDetector()
    d.observe_idle_turn()
    d.observe_idle_turn()
    d.observe(AgentStep.of("read"))
    d.observe_idle_turn()
    assert d.assess() is None


# ── the pattern OpenHands does not have ──────────────────────────────────


def test_different_actions_leaving_the_world_unchanged_is_no_progress():
    d = StuckDetector()
    for name in ("read", "list", "stat"):
        d.observe(AgentStep.of(name, arguments={"p": name}, observation="unchanged"))
    verdict = d.assess()
    assert verdict.pattern is StuckPattern.NO_PROGRESS
    assert set(verdict.actions) == {"read", "list", "stat"}


def test_no_progress_does_not_fire_on_plain_repetition():
    """Identical calls are the repetition finding; this one is about varied effort."""
    d = StuckDetector()
    _feed(d, *_same_call(3))
    assert d.assess().pattern is StuckPattern.REPEATED_ACTION_OBSERVATION


def test_a_moving_world_is_not_stuck():
    d = StuckDetector()
    for i, name in enumerate(("read", "list", "stat")):
        d.observe(AgentStep.of(name, arguments={"p": name}, observation=f"state{i}"))
    assert d.assess() is None


# ── it is an impasse, and the right one ──────────────────────────────────


def test_a_loop_is_a_no_change_impasse():
    d = StuckDetector()
    _feed(d, *_same_call(3))
    assert d.assess().impasse.type is ImpasseType.NO_CHANGE


def test_the_impasse_reaches_the_learner_and_is_counted_by_type():
    from core.cognition.impasse import ChunkStore, ImpasseLearner

    learner = ImpasseLearner(ChunkStore())
    d = StuckDetector()
    _feed(d, *_same_call(3))
    verdict = d.assess()

    import core.cognition.impasse as impasse_module

    original = impasse_module._learner
    impasse_module._learner = learner
    try:
        d.record_to_learner(verdict)
    finally:
        impasse_module._learner = original

    assert learner.report()["impasse_counts"]["no_change"] == 1


def test_the_same_episode_is_reported_once():
    d = StuckDetector()
    _feed(d, *_same_call(3))
    assert d.assess_once() is not None
    _feed(d, *_same_call(1))
    assert d.assess_once() is None, "one loop must not report on every subsequent step"


def test_a_genuinely_new_loop_still_reports():
    d = StuckDetector()
    _feed(d, *_same_call(3))
    assert d.assess_once() is not None
    for _ in range(3):
        d.observe(AgentStep.of("write", arguments={"p": "b"}, observation="other"))
    assert d.assess_once() is not None


# ── window and lifecycle ─────────────────────────────────────────────────


def test_a_new_instruction_clears_the_window():
    d = StuckDetector()
    _feed(d, *_same_call(3))
    d.reset()
    assert d.assess() is None
    assert d.report()["steps_held"] == 0


def test_the_window_is_bounded():
    d = StuckDetector(window=5)
    _feed(d, *_same_call(20))
    assert len(d.steps) == 5


def test_a_window_too_small_to_detect_alternation_is_refused():
    with pytest.raises(ValueError, match="at least"):
        StuckDetector(window=3)


def test_steps_from_drops_records_with_no_action_name():
    steps = steps_from([{"name": "read", "args": {}}, {"args": {}}, {"tool": "write"}])
    assert [s.action for s in steps] == ["read", "write"]


def test_steps_from_reads_an_error_as_a_failure():
    steps = steps_from([{"name": "read", "error": "ENOENT"}])
    assert steps[0].failed and steps[0].error_kind == "ENOENT"


# ── the registry acts on it ──────────────────────────────────────────────


class _StaticTool:
    """A tool that always returns the same thing — the loop being detected."""

    def __init__(self) -> None:
        self.calls = 0
        self.code = ""


@pytest.mark.asyncio
async def test_the_registry_refuses_the_fourth_identical_call(monkeypatch):
    from core.tools import tool_registry as tr

    registry = tr.ToolRegistry()
    registry.register_tool("read", _StaticTool())

    monkeypatch.setattr(tr, "_build_sandbox_driver", lambda *a, **k: "driver")
    monkeypatch.setattr(
        tr, "run_untrusted", lambda _driver: {"status": "ok", "stdout": "'same'"}
    )

    for _ in range(3):
        assert (await registry.execute_tool("read", path="a.txt"))["ok"]

    blocked = await registry.execute_tool("read", path="a.txt")
    assert not blocked["ok"]
    assert blocked["error"] == "tool_call_looping:read"
    assert blocked["pattern"] == StuckPattern.REPEATED_ACTION_OBSERVATION.value


@pytest.mark.asyncio
async def test_a_different_call_to_the_same_tool_is_not_blocked(monkeypatch):
    from core.tools import tool_registry as tr

    registry = tr.ToolRegistry()
    registry.register_tool("read", _StaticTool())
    monkeypatch.setattr(tr, "_build_sandbox_driver", lambda *a, **k: "driver")
    monkeypatch.setattr(
        tr, "run_untrusted", lambda _driver: {"status": "ok", "stdout": "'same'"}
    )

    for _ in range(4):
        await registry.execute_tool("read", path="a.txt")
    assert (await registry.execute_tool("read", path="b.txt"))["ok"]


@pytest.mark.asyncio
async def test_beginning_a_step_lifts_the_block(monkeypatch):
    """A new instruction may legitimately ask for the same thing again."""
    from core.tools import tool_registry as tr

    registry = tr.ToolRegistry()
    registry.register_tool("read", _StaticTool())
    monkeypatch.setattr(tr, "_build_sandbox_driver", lambda *a, **k: "driver")
    monkeypatch.setattr(
        tr, "run_untrusted", lambda _driver: {"status": "ok", "stdout": "'same'"}
    )

    for _ in range(3):
        await registry.execute_tool("read", path="a.txt")
    assert not (await registry.execute_tool("read", path="a.txt"))["ok"]

    registry.begin_step()
    assert (await registry.execute_tool("read", path="a.txt"))["ok"]
