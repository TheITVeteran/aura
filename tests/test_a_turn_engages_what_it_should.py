"""Whether a turn is actually hers, or the base model wearing her name.

The question that prompted this — "are we sure voice and chat invoke everything
they should, personality too?" — had no answer anywhere in the system, and the
reason it had no answer is the interesting part.

The persona pass is applied through
``ServiceContainer.get("personality_engine", default=None)``. When the service
is absent, the whole pass is skipped: no exception, no degradation, no record.
The reply then ships in the flat register of the base model, every layer that
computed her disposition discarded at the last inch — and the turn still
reports a proven full-mind path, because personality was never in the required
subsystem list. Its absence looked exactly like its presence.

The fix is visibility, not another gate. The persona pass now records its own
absence, and a turn carries the list of conversational organs it did not
engage: personality, affect, episodic and semantic memory, identity continuity,
felt state, the honesty governor, the knowledge graph, the event bus.

Requiring them was the first attempt and it was wrong. A missing persona pass
makes a reply flat — it does not make it wrong — and refusing a correct answer
for being flat is the over-blocking failure this codebase has already paid for
twice. Thirty-three tests said so immediately. Flat and true beats refused, so
these are reported and never fatal.

Voice needs no separate treatment, and that is worth asserting rather than
assuming: it replays the real HTTP chat handler through a synthetic ASGI scope,
so it inherits this contract exactly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from interface.routes.chat import (
    _EXPECTED_TURN_ORGANS,
    _absent_turn_organs,
    _collect_expected_turn_organs,
    _collect_live_chat_required_subsystems,
)

SOURCE = Path("interface/routes/chat.py")


# ── What a turn cannot be hers without ─────────────────────────────────────

@pytest.mark.parametrize(
    "subsystem",
    [
        "kernel",
        "cognitive_engine",
        "inference",
        "memory",
        "tool_governance",
        "substrate_voice",
    ],
)
def test_the_required_contract_names_it(subsystem: str) -> None:
    assert subsystem in _collect_live_chat_required_subsystems()


def test_personality_is_reported_rather_than_required() -> None:
    """A missing persona pass makes a reply flat; it does not make it wrong.

    The instinct is to require it. Requiring it refuses a correct answer for
    being flat, which is the over-blocking failure already paid for twice —
    most recently a gate that replaced 900 characters of real answer with an
    apology because confidence read "degraded". Flat and true beats refused,
    so the absence is recorded and visible instead of fatal.
    """
    assert "personality_engine" in _collect_expected_turn_organs()
    assert "personality" not in _collect_live_chat_required_subsystems()


def test_a_missing_persona_pass_is_recorded() -> None:
    """Shipping the base model's register as her voice must not be silent."""
    src = SOURCE.read_text(encoding="utf-8")
    assert "personality_engine absent; reply shaped by nothing" in src


# ── What a turn should have engaged ───────────────────────────────────────

def test_every_expected_organ_states_why_it_matters() -> None:
    """A list of names teaches nobody what its absence costs."""
    for name, why in _EXPECTED_TURN_ORGANS:
        assert name and why
        assert len(why) > 15, f"{name} has no real explanation"


@pytest.mark.parametrize(
    "organ",
    [
        "personality_engine",
        "episodic_memory",
        "semantic_memory",
        "soul",
        "soma",
        "affect_engine",
        "data_honesty_governor",
        "knowledge_graph",
        "event_bus",
    ],
)
def test_the_expected_set_covers_the_conversational_organs(organ: str) -> None:
    assert organ in _collect_expected_turn_organs()


def test_absent_organs_are_reported_with_their_reason() -> None:
    absent = _absent_turn_organs({"episodic_memory": False, "soul": True})
    assert len(absent) == 1
    assert "episodic_memory" in absent[0]
    assert "remembered" in absent[0]


def test_a_fully_engaged_turn_reports_nothing_absent() -> None:
    assert _absent_turn_organs(dict.fromkeys(dict(_EXPECTED_TURN_ORGANS), True)) == []


def test_probing_never_raises_outside_a_runtime() -> None:
    """Nothing is registered in a bare process; that is a report, not a crash."""
    engaged = _collect_expected_turn_organs()
    assert set(engaged) == set(dict(_EXPECTED_TURN_ORGANS))
    assert all(isinstance(value, bool) for value in engaged.values())


def test_the_expected_tier_is_not_fatal() -> None:
    """Promoting these would recreate a failure already paid for once."""
    src = SOURCE.read_text(encoding="utf-8")
    required = src[src.index("def _collect_live_chat_required_subsystems") :]
    required = required[: required.index("def _assess_live_mind_snapshot")]
    for organ, _why in _EXPECTED_TURN_ORGANS:
        assert f'"{organ}":' not in required, (
            f"{organ} became a hard requirement; a warming organ would now "
            "refuse an otherwise good answer"
        )


def test_the_turn_contract_carries_the_engagement() -> None:
    """Reported on the turn, or it is a probe nobody reads."""
    src = SOURCE.read_text(encoding="utf-8")
    assert '"expected_organs": _expected_organs,' in src
    assert '"absent_expected_organs": _absent_turn_organs(_expected_organs),' in src
    # Counted once per turn, beside the required-subsystem collection.
    assert "_note_organ_engagement(_expected_organs)" in src


# ── Voice inherits it rather than reimplementing it ───────────────────────

def _function_body(name: str) -> str:
    """The source of exactly one function, bounded by the AST.

    These assertions used to take a 4000-CHARACTER window from the start of
    the function name. That window silently spanned whatever followed, so
    the tests passed on code belonging to other functions — and when the
    voice turn was refactored into a 590-char wrapper the window stopped
    reaching the properties it was supposed to check, failing while
    everything it protects was intact. A test scoped by character count is
    a test about formatting.
    """
    import ast

    src = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, (ast.AsyncFunctionDef, ast.FunctionDef)) and item.name == name
    )
    return ast.get_source_segment(src, node) or ""


#: Voice does not implement a turn; it delegates to the one shared surface
#: handler. That IS the property — a second implementation is what would
#: drift.
_SHARED_SURFACE_HANDLER = "run_governed_surface_chat_turn"


def test_voice_replays_the_real_chat_handler() -> None:
    """Voice is a presentation surface, not a second cognition lane.

    If it built its own request path it would drift, and the contract
    asserted above would cover only half the ways she speaks.
    """
    voice = _function_body("run_governed_voice_chat_turn")
    assert _SHARED_SURFACE_HANDLER in voice, (
        "the voice turn no longer delegates to the shared surface handler; a "
        "second request path is exactly the drift this guards against"
    )

    shared = _function_body(_SHARED_SURFACE_HANDLER)
    assert '"path": "/api/chat"' in shared
    assert "validate_runtime_security_request(request)" in shared
    assert "_require_internal(request)" in shared


def test_the_voice_surface_is_declared_not_disguised() -> None:
    shared = _function_body(_SHARED_SURFACE_HANDLER)
    assert "x-aura-surface" in shared
    assert "x-aura-require-cognitiveengine" in shared


def test_the_voice_wrapper_names_its_surface() -> None:
    """Delegating must not lose WHICH surface is speaking."""
    voice = _function_body("run_governed_voice_chat_turn")
    assert 'surface="voice"' in voice


# ── A gap that persists is a different fact from a gap on one turn ─────────

def _engagement(**overrides: bool) -> dict[str, bool]:
    engaged = dict.fromkeys(dict(_EXPECTED_TURN_ORGANS), True)
    engaged.update(overrides)
    return engaged


@pytest.fixture(autouse=True)
def _clean_streaks():
    from interface.routes.chat import reset_organ_engagement_streaks_for_test

    reset_organ_engagement_streaks_for_test()
    yield
    reset_organ_engagement_streaks_for_test()


def test_one_missing_turn_is_not_escalated() -> None:
    """Boot, a restart, a lane cycling — absence for a turn is noise."""
    from interface.routes.chat import _note_organ_engagement

    assert _note_organ_engagement(_engagement(personality_engine=False)) == []


def test_a_persistent_gap_becomes_a_defect() -> None:
    """Reporting alone changes nothing if she sounds flat on every turn."""
    from interface.routes.chat import _CHRONIC_ABSENCE_TURNS, _note_organ_engagement

    became = []
    for _ in range(_CHRONIC_ABSENCE_TURNS):
        became = _note_organ_engagement(_engagement(personality_engine=False))
    assert became == ["personality_engine"]


def test_a_permanent_gap_escalates_once_not_every_turn() -> None:
    """A real signal repeated every turn is a storm nobody reads."""
    from interface.routes.chat import _CHRONIC_ABSENCE_TURNS, _note_organ_engagement

    escalations = 0
    for _ in range(_CHRONIC_ABSENCE_TURNS * 3):
        escalations += len(_note_organ_engagement(_engagement(personality_engine=False)))
    assert escalations == 1


def test_an_organ_coming_back_clears_its_streak() -> None:
    from interface.routes.chat import _CHRONIC_ABSENCE_TURNS, _note_organ_engagement

    for _ in range(_CHRONIC_ABSENCE_TURNS - 1):
        _note_organ_engagement(_engagement(soul=False))
    _note_organ_engagement(_engagement())
    for _ in range(_CHRONIC_ABSENCE_TURNS - 1):
        assert _note_organ_engagement(_engagement(soul=False)) == []


def test_organs_are_tracked_independently() -> None:
    from interface.routes.chat import _CHRONIC_ABSENCE_TURNS, _note_organ_engagement

    for _ in range(_CHRONIC_ABSENCE_TURNS - 1):
        _note_organ_engagement(_engagement(soul=False, soma=False))
    became = _note_organ_engagement(_engagement(soul=False, soma=False))
    assert set(became) == {"soul", "soma"}


def test_a_fully_engaged_turn_escalates_nothing() -> None:
    from interface.routes.chat import _CHRONIC_ABSENCE_TURNS, _note_organ_engagement

    for _ in range(_CHRONIC_ABSENCE_TURNS * 2):
        assert _note_organ_engagement(_engagement()) == []


def test_escalation_never_refuses_the_turn() -> None:
    """The whole point: chronic absence is loud, and still not a gate."""
    from interface.routes.chat import _note_organ_engagement

    src = SOURCE.read_text(encoding="utf-8")
    fn = src[src.index("def _note_organ_engagement") :]
    fn = fn[: fn.index("def reset_organ_engagement_streaks_for_test")]
    assert "return" in fn
    assert "raise" not in fn, "a reporting tier must not throw"
    assert _note_organ_engagement(_engagement(soul=False)) == []
