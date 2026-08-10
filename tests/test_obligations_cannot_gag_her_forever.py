"""A gate that only its own blocked actions can open is a gag, not a rule.

Measured live 2026-08-10, after 1008 ambient ticks and 267 observations:

    seconds_since_spoke: None

She had never once spoken proactively. The log showed 44 proactive initiations
generated and 44 suppressed — a 100% suppression rate — every one carrying the
same reason:

    Constitutional preflight suppressed spontaneous emission for jarvis
    (temporal_obligation_active:Find the most obscure fact about xenobiology
     concepts.)

Rule 5 defers autonomous intents whenever ``obligation_pressure > 0.0``, and
the pressure is ``pending*0.25 + goals*0.2 + commitments*0.2``, so ONE
outstanding goal closes the gate. The obligations were five stale autonomous
goals persisted in continuity.json with no timestamps and no expiry. Clearing
them requires finishing autonomous work; the gate defers autonomous work.

Nothing was misconfigured. The win was structurally impossible.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from core.goals.goal_text import is_actionable_goal_text, normalize_goal_text

EXECUTIVE = Path(__file__).resolve().parents[1] / "core/executive/executive_core.py"
SOURCE = EXECUTIVE.read_text(encoding="utf-8")


def _temporal_rule_blocks() -> list[str]:
    """Every action-type set guarded by the temporal-obligation rule.

    Anchored structurally rather than on the reason string: an earlier version
    of this helper matched up to the literal ``temporal_obligation_active``,
    and the explanatory comment added beside the fix contains that same
    literal, so the match stopped inside the prose and found no action set at
    all. A test that reads source has to be robust to the source describing
    itself.
    """
    blocks: list[str] = []
    for chunk in SOURCE.split('temporal["obligation_pressure"] > 0.0')[1:]:
        marker = "intent.action_type in {"
        if marker not in chunk:
            continue
        blocks.append(chunk.split(marker, 1)[1].split("}", 1)[0])
    return blocks


def test_the_rule_is_still_enforced_somewhere():
    assert _temporal_rule_blocks(), "the temporal-obligation rule vanished entirely"


def test_speaking_is_not_deferred_by_unfinished_background_work():
    """Saying something spawns nothing and finishes when the sentence ends.

    Deferring it because a research goal is open is a category error, and it
    is what produced a permanent gag.
    """
    for action_set in _temporal_rule_blocks():
        assert "EMIT_MESSAGE" not in action_set, (
            "EMIT_MESSAGE is deferred by temporal obligations again; "
            "an obligation that only autonomous work can clear will gag her"
        )


def test_work_that_genuinely_competes_is_still_deferred():
    """The rule must keep doing its job for actions that start more work."""
    for action_set in _temporal_rule_blocks():
        for competing in ("SPAWN_TASK", "TOOL_CALL", "REFLECT"):
            assert competing in action_set, f"{competing} stopped being deferred"


def test_a_goal_stringified_on_the_way_in_is_read_back_not_read_aloud():
    """Three of five persisted goals were reprs of dicts.

    They passed every actionability check, counted as outstanding obligations,
    and the first was offered to the person as the reason she could not act.
    """
    repr_goal = (
        "{'id': 'db847edb9427', 'name': '[AUTONOMOUS INITIATIVE] "
        "enhance_memory_retention', 'objective': 'enhance memory retention'}"
    )
    assert normalize_goal_text(repr_goal) == "enhance memory retention"

    json_goal = '{"id": "abc", "objective": "Research user privacy"}'
    assert normalize_goal_text(json_goal) == "Research user privacy"

    # A mapping carrying no goal text is not a goal.
    assert normalize_goal_text("{}") == ""
    assert not is_actionable_goal_text("{}")


def test_ordinary_goal_text_is_untouched():
    plain = "Find the most obscure fact about xenobiology concepts."
    assert normalize_goal_text(plain) == plain
    # Something that merely starts with a brace must not be mangled.
    assert normalize_goal_text("{not actually a dict") == "{not actually a dict"


def test_the_live_store_would_no_longer_hold_the_gate_shut():
    """Against the real persisted record, if it is still present.

    Skipped rather than failed when absent: this asserts a property of the
    fix, and a machine without that file is not evidence against it.
    """
    record = Path.home() / ".aura/data/continuity.json"
    if not record.exists():
        return
    data = json.loads(record.read_text(encoding="utf-8"))
    goals = data.get("active_goal_details") or []
    # Whatever survives must at least be readable as language, not as a repr.
    for goal in goals:
        text = normalize_goal_text(goal)
        assert not (text.startswith("{") and "'id'" in text), (
            f"a serialized mapping is still standing in for a goal: {text[:80]}"
        )
