"""A want settled while she could not act is done when she can, without being argued again.

Deciding tonight to walk tomorrow, a person does not reopen the question in the
morning. Her slate was rebuilt every cycle and nothing on it was ever settled.
"""
from __future__ import annotations

import asyncio
import random

import pytest

from core.conation import what_she_wants
from core.conation.engine import ConationEngine
from core.conation.what_she_wants import WhatSheWants, _conation
from core.runtime import overt_action_loop
from core.runtime.atomic_writer import flush_writes_behind
from core.runtime.overt_action_loop import OvertActionLoop

TIDE, LOGS = "topic:tide pools", "topic:log files"
LABELS = {TIDE: "tide pools", LOGS: "log files"}


def _a_life(record) -> None:
    rng = random.Random(7)
    for turn in range(24):
        key, paid = (TIDE, 0.6) if turn % 2 == 0 else (LOGS, -0.5)
        record.went(key, "web_search", label=LABELS[key])
        record.met(["self", key], paid + rng.uniform(-0.2, 0.2), reachable=[key], labels=LABELS)
    for held in _conation(record).dynamics._satiation.values():
        held.last_update -= 600.0


@pytest.fixture
def record(tmp_path):
    held = what_she_wants.reset_for_test(tmp_path / "wants.json", engine=ConationEngine())
    _a_life(held)
    yield held
    what_she_wants.reset_for_test(tmp_path / "after.json", engine=ConationEngine())


def test_what_she_most_wants_is_settled_once_and_stays_settled(record):
    first = record.decide()
    assert first["thing"] == TIDE and first["way"] == "web_search"
    record.met(["self", LOGS], 0.9, reachable=[LOGS], labels=LABELS)
    assert record.decide()["thing"] == TIDE
    assert record.decided()["decided_turn"] == first["decided_turn"]


def test_it_is_still_decided_after_a_restart(record, tmp_path):
    record.decide()
    flush_writes_behind()
    again = WhatSheWants(tmp_path / "wants.json", engine=ConationEngine())
    assert again.decided()["thing"] == TIDE


def test_a_decision_is_let_go_when_she_has_come_to_dislike_the_thing(record):
    record.decide()
    for _ in range(40):
        record.met(["self", TIDE], -0.8, reachable=[TIDE], labels=LABELS)
        record.met(["self", LOGS], 0.6, reachable=[LOGS], labels=LABELS)
    assert record.known(TIDE) is False
    assert record.decided() is None


class _NeverAsked:
    async def synthesize(self, state):
        raise AssertionError("what she had decided was argued again")


def _loop(monkeypatch, *, busy: bool) -> OvertActionLoop:
    loop = OvertActionLoop(synthesizer=_NeverAsked(), state_provider=lambda: object())
    monkeypatch.setattr(OvertActionLoop, "enabled", staticmethod(lambda: True))
    monkeypatch.setattr(loop, "_background_reason", lambda: "user_active" if busy else "")
    monkeypatch.setattr(loop, "_authorize_fallback", lambda initiative: "will-1")
    return loop


def test_while_she_cannot_act_what_she_wants_is_settled_for_later(record, monkeypatch):
    loop = _loop(monkeypatch, busy=True)
    skipped = asyncio.run(loop.run_once())
    assert "background_policy" in str(skipped)
    assert record.decided()["thing"] == TIDE


def test_when_she_is_free_she_does_what_she_decided_without_a_new_contest(record, monkeypatch):
    record.decide()
    loop = _loop(monkeypatch, busy=False)
    acted_on: list[dict] = []

    async def goal_for(initiative):
        return {}

    async def execute(initiative, *, goal, selection, will_receipt_id):
        acted_on.append({"initiative": initiative, "skill": selection.skill, "params": selection.params, "will": will_receipt_id})
        result = overt_action_loop.OvertActionResult(action_id="a1", status="verified", skill=selection.skill, verified=True)
        overt_action_loop._note_want(initiative, result)
        return result

    monkeypatch.setattr(loop, "_goal_for_initiative", goal_for)
    monkeypatch.setattr(loop, "_execute_initiative", execute)
    asyncio.run(loop.run_once())
    assert len(acted_on) == 1
    assert acted_on[0]["initiative"]["metadata"]["want"]["thing"] == TIDE
    assert acted_on[0]["skill"] == "web_search" and acted_on[0]["will"] == "will-1"
    assert record.decided() is None
