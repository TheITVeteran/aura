"""The rest of what a want is in a life: the doing, the picture, the quiet, the pace, one record.

Bryan, 13 October: deciding on a walk, getting up and going without being
told, and enjoying the whole of it. Each test is one part of that.
"""
from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import pytest

from core.conation import what_she_wants
from core.conation.engine import ConationEngine
from core.conation.what_she_wants import REACHABLE_KINDS
from core.runtime.autonomy_conductor import AutonomyConductor
from core.runtime.overt_action_loop import OvertActionLoop, OvertActionResult

A, B, C = "topic:tide pools", "topic:limpet", "topic:invoices"
LABELS = {A: "tide pools", B: "limpet", C: "invoices"}


@pytest.fixture
def record(tmp_path):
    held = what_she_wants.reset_for_test(tmp_path / "wants.json", engine=ConationEngine())
    yield held
    what_she_wants.reset_for_test(tmp_path / "after.json", engine=ConationEngine())


def _rested(record) -> None:
    for held in record._conation().dynamics._satiation.values():
        held.last_update -= 600.0


def test_what_she_has_under_way_is_lived_turn_by_turn_and_not_only_at_its_end(record):
    record.began(A, "drawing-1", "Draw the zones of a tide pool", label="tide pools")
    for _ in range(6):
        record.met(["self"], 0.5)
    doing = record.thing(A)
    assert doing.contacts == 6
    assert doing.during.times == 6 and doing.during.liking == pytest.approx(0.5)
    record.ended(A, "drawing-1", finished=True)
    record.met(["self"], 0.5)
    assert record.thing(A).contacts == 6


def test_she_can_want_what_she_has_only_pictured(record):
    """The limpet earned nothing of its own, and it comes with what she likes."""
    rng = random.Random(2)
    for _ in range(16):
        record.met([A], 0.6 + rng.uniform(-0.1, 0.1), reachable=[A], labels=LABELS)
    for _ in range(4):
        record.met([A, B], 0.6, reachable=[A, B], labels=LABELS)
        record.met([C], 0.0, reachable=[C], labels=LABELS)
    assert abs(record.thing(B).liking) < 0.1
    assert record.pictured(B) > 0.4 > record.pictured(C)
    _rested(record)
    pull = {want.thing: want for want in record.wants()}
    assert pull[B].wanting > 0.05
    assert C not in pull or pull[B].wanting > pull[C].wanting
    assert "pictured from its company" in pull[B].why


def test_she_acts_sooner_when_her_life_has_been_paying(record):
    assert record.vigor() == pytest.approx(1.0)
    for _ in range(40):
        record.met(["self"], 0.5)
    lively = record.vigor()
    for _ in range(120):
        record.met(["self"], -0.5)
    assert lively > 1.2 and record.vigor() < 0.8


def test_the_time_between_her_acts_follows_it(record):
    conductor = AutonomyConductor.__new__(AutonomyConductor)
    conductor.jobs = {"overt_action_cycle": SimpleNamespace(interval_s=120.0)}
    for _ in range(40):
        record.met(["self"], 0.5)
    conductor._pace_her_acts()
    assert conductor.jobs["overt_action_cycle"].interval_s < 100.0


def test_what_she_does_on_a_want_of_her_own_is_not_announced(monkeypatch):
    loop = OvertActionLoop(state_provider=lambda: object())
    announced: list[str] = []
    monkeypatch.setattr(loop, "_emit_visible_trace", lambda result: announced.append(result.source))
    monkeypatch.setattr(loop, "_emit_receipts", lambda result, raw: None)
    monkeypatch.setattr(loop, "_record_life_trace", lambda result, raw: None)
    monkeypatch.setattr(loop, "_update_goal", lambda result: None)
    for source in ("want", "goal_engine"):
        result = OvertActionResult(action_id=source, status="verified", source=source, verified=True, started_at=1.0)
        asyncio.run(loop._finish(result, raw_result={}))
    assert announced == ["goal_engine"]


def test_a_game_she_played_is_on_the_record_her_talk_is_on(record):
    assert "game" in REACHABLE_KINDS
    record.went("game:zap to it", "play", label="zap to it")
    record.met(["self"], 0.7)
    game = record.thing("game:zap to it")
    assert game.reachable and game.ways["play"].times == 1 and game.liking > 0.0
