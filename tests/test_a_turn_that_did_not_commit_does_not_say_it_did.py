"""The cycle declared success before it had any.

Every exit from the state-commit loop was a bare `break` — a bypassed commit, a
missing repository, exhausted version-conflict retries, a RuntimeError. Response
extraction then built a Thought with `confidence=0.9` and the reasoning line
"Phase-based cognitive cycle completed successfully", whatever had happened to
durable state.

Around that sat four more of the same shape:

- Foreground closure fired BEFORE persistence, so external lifecycle state
  could say the turn completed while durable cognitive state had no record of
  it.
- The version-conflict retry rebased onto the latest state and then copied
  goals, initiatives, focus, phenomenal state and modifiers back out of the
  per-turn snapshot — overwriting the concurrent writer whose commit caused the
  conflict in the first place.
- Extraction took the last message if its role was "assistant". A duplicate or
  suppressed user append leaves an older assistant message at the end, so a
  previous answer went out again as this turn's.
- Three learning systems received reward 1.0 for the existence of an
  assistant-shaped message, with no user feedback, no correctness check and no
  persistence outcome.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain.cognitive_engine import CognitiveEngine


# ─────────────────────────────── the response belongs to this turn


def test_an_assistant_message_from_an_earlier_turn_is_not_this_answer():
    memory = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]

    # The mark is taken when THIS turn starts: everything already there is old.
    assert CognitiveEngine._turn_response_message(memory, mark=len(memory)) is None


def test_an_assistant_message_produced_this_turn_is_the_answer():
    memory = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]
    mark = len(memory)
    memory.append({"role": "user", "content": "second question"})
    memory.append({"role": "assistant", "content": "second answer"})

    found = CognitiveEngine._turn_response_message(memory, mark=mark)

    assert found is not None
    assert found["content"] == "second answer"


def test_a_turn_that_ended_on_a_user_message_produced_no_answer():
    memory = [{"role": "user", "content": "question"}]

    assert CognitiveEngine._turn_response_message(memory, mark=0) is None


@pytest.mark.parametrize("memory", [None, [], ["junk"], [None]])
def test_malformed_working_memory_yields_no_answer(memory):
    assert CognitiveEngine._turn_response_message(memory, mark=0) is None


# ─────────────────────────────── confidence follows the evidence


def test_a_committed_turn_keeps_the_full_confidence():
    assert (
        CognitiveEngine._cycle_confidence(
            commit_outcome="committed", degraded_subsystems=0
        )
        == 0.9
    )


@pytest.mark.parametrize(
    "outcome",
    ["version_conflict_exhausted", "no_state_repository", "not_attempted", "error:X"],
)
def test_an_uncommitted_turn_is_less_confident(outcome):
    committed = CognitiveEngine._cycle_confidence(
        commit_outcome="committed", degraded_subsystems=0
    )
    uncommitted = CognitiveEngine._cycle_confidence(
        commit_outcome=outcome, degraded_subsystems=0
    )

    assert uncommitted < committed


def test_test_isolation_is_not_a_failure():
    """A bypassed commit under test isolation is by design, not a fault."""
    assert CognitiveEngine._cycle_confidence(
        commit_outcome="bypassed_test_isolation", degraded_subsystems=0
    ) == CognitiveEngine._cycle_confidence(
        commit_outcome="committed", degraded_subsystems=0
    )


def test_degraded_subsystems_lower_confidence():
    clean = CognitiveEngine._cycle_confidence(
        commit_outcome="committed", degraded_subsystems=0
    )
    degraded = CognitiveEngine._cycle_confidence(
        commit_outcome="committed", degraded_subsystems=3
    )

    assert degraded < clean


def test_confidence_stays_inside_its_bounds():
    worst = CognitiveEngine._cycle_confidence(
        commit_outcome="version_conflict_exhausted", degraded_subsystems=99
    )

    assert 0.3 <= worst <= 0.95


# ─────────────────────────────── a rebase keeps the concurrent writer's work


def _cognition(**fields):
    base = {
        "active_goals": [],
        "pending_initiatives": [],
        "attention_focus": None,
        "phenomenal_state": None,
        "modifiers": {},
    }
    base.update(fields)
    return SimpleNamespace(**base)


def test_a_field_this_turn_did_not_touch_keeps_the_latest_value():
    baseline = {
        "active_goals": ["old goal"],
        "pending_initiatives": [],
        "attention_focus": None,
        "phenomenal_state": None,
        "modifiers": {},
    }
    worked = _cognition(active_goals=["old goal"])
    # The concurrent writer whose commit caused the conflict.
    rebased = _cognition(active_goals=["someone else's goal"])

    CognitiveEngine._reapply_turn_changes(rebased, worked, baseline)

    assert rebased.active_goals == ["someone else's goal"]


def test_a_field_this_turn_changed_wins():
    baseline = {
        "active_goals": ["old goal"],
        "pending_initiatives": [],
        "attention_focus": None,
        "phenomenal_state": None,
        "modifiers": {},
    }
    worked = _cognition(active_goals=["old goal", "new goal from this turn"])
    rebased = _cognition(active_goals=["someone else's goal"])

    CognitiveEngine._reapply_turn_changes(rebased, worked, baseline)

    assert rebased.active_goals == ["old goal", "new goal from this turn"]


def test_modifiers_follow_the_same_rule():
    baseline = {
        "active_goals": [],
        "pending_initiatives": [],
        "attention_focus": None,
        "phenomenal_state": None,
        "modifiers": {"a": 1},
    }
    worked = _cognition(modifiers={"a": 1})
    rebased = _cognition(modifiers={"a": 1, "b": 2})

    CognitiveEngine._reapply_turn_changes(rebased, worked, baseline)

    assert rebased.modifiers == {"a": 1, "b": 2}


def test_a_scalar_this_turn_set_is_reapplied():
    baseline = {
        "active_goals": [],
        "pending_initiatives": [],
        "attention_focus": None,
        "phenomenal_state": None,
        "modifiers": {},
    }
    worked = _cognition(attention_focus="this turn's focus")
    rebased = _cognition(attention_focus="concurrent focus")

    CognitiveEngine._reapply_turn_changes(rebased, worked, baseline)

    assert rebased.attention_focus == "this turn's focus"


# ─────────────────────────────── closure waits for the commit


def test_closure_is_notified_after_the_commit_loop():
    import ast
    import inspect

    import core.brain.cognitive_engine as engine_mod

    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    closure_line = None
    commit_line = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "complete_foreground_turn"
        ):
            closure_line = node.lineno
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "commit_outcome" for t in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and node.value.value == "committed"
        ):
            commit_line = node.lineno

    assert closure_line is not None and commit_line is not None
    assert commit_line < closure_line, (
        "foreground closure still fires before persistence"
    )


def test_an_uncommitted_foreground_turn_withholds_closure():
    import inspect

    import core.brain.cognitive_engine as engine_mod

    source = inspect.getsource(engine_mod)

    assert "withheld foreground closure because cognitive state did not commit" in source


def test_the_thought_carries_the_commit_outcome():
    import inspect

    import core.brain.cognitive_engine as engine_mod

    source = inspect.getsource(engine_mod)

    assert '"state_commit_outcome": commit_outcome' in source
    assert 'reasoning=["Phase-based cognitive cycle completed successfully."]' not in source


def test_the_reward_is_not_maximal_for_an_uncommitted_turn():
    import inspect

    import core.brain.cognitive_engine as engine_mod

    source = inspect.getsource(engine_mod)

    assert "_cycle_reward = 1.0 if commit_outcome in {" in source
    assert "reward=_cycle_reward," in source


# ─────────────────── the deadline reaches past phase execution


def test_the_commit_loop_is_bounded_by_the_cycle_deadline():
    """asyncio.timeout wrapped phase execution only. Repository reads,
    advisors, augmentors and this commit loop all ran outside it, so a
    configured cycle timeout was never the end-to-end budget it reads as."""
    import inspect

    import core.brain.cognitive_engine as engine_mod

    source = inspect.getsource(engine_mod)

    assert "_commit_budget = max(0.0, cycle_deadline_at - time.monotonic())" in source
    assert "timeout=_commit_budget," in source


def test_an_expired_cycle_skips_the_commit_rather_than_waiting():
    import inspect

    import core.brain.cognitive_engine as engine_mod

    source = inspect.getsource(engine_mod)

    assert 'commit_outcome = "cycle_deadline_expired"' in source
    assert 'commit_outcome = "commit_timeout"' in source


def test_a_deadline_outcome_lowers_confidence_like_any_other_failure():
    from core.brain.cognitive_engine import CognitiveEngine

    for outcome in ("cycle_deadline_expired", "commit_timeout"):
        assert CognitiveEngine._cycle_confidence(
            commit_outcome=outcome, degraded_subsystems=0
        ) < CognitiveEngine._cycle_confidence(
            commit_outcome="committed", degraded_subsystems=0
        )


# ─────────────────── the retry keeps the contract, the phase loop keeps failures


def test_the_deep_downshift_forwards_the_context():
    """A retry that drops the desktop flags, the live-mind evidence and the
    scoped request metadata is a different request."""
    import inspect

    import core.brain.cognitive_engine as engine_mod

    source = inspect.getsource(engine_mod)

    assert "context=dict(context or {})," in source
    assert (
        "return await self.think(objective, mode=ThinkingMode.FAST, origin=origin, **kwargs)"
        not in source
    )


def test_the_phase_loop_catches_what_phases_actually_raise():
    """It caught sqlite3.Error and OSError only, so a RuntimeError or a
    malformed phase return escaped the cognitive API entirely."""
    import inspect

    import core.brain.cognitive_engine as engine_mod

    source = inspect.getsource(engine_mod)

    assert "except (sqlite3.Error, *_COGNITIVE_ENGINE_RECOVERABLE_ERRORS) as e:" in source


# ─────────────────── her prior position is not the person's words


def test_the_spine_injection_is_not_spliced_into_the_objective():
    """The prompt builder appends the whole objective as role=user, so a
    prepended prior position was persisted as something the PERSON said."""
    import inspect

    import core.brain.cognitive_engine as engine_mod

    source = inspect.getsource(engine_mod)

    assert 'objective = check.injection + "\\n\\n" + objective' not in source
    assert '"type": "spine_prior_position"' in source
    assert '"role": "system",' in source
