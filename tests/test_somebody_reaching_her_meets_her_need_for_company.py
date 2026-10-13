"""A message from somebody meets her need for company in the store that is read.

Live, 13 October: the incoming path credited the drive engine only, her
heartbeat read the motivation budgets, and "social is depleted (80% urgency)"
won her workspace 69 times in three hours of conversation.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from core.orchestrator.mixins import incoming_logic
from core.phases import motivation_update
from core.phases.motivation_update import met_by_somebody


def _with_social_at(monkeypatch, level: float):
    budgets = {"social": {"level": level, "capacity": 100.0}}
    repo = SimpleNamespace(_current=SimpleNamespace(motivation=SimpleNamespace(budgets=budgets)))
    monkeypatch.setattr(motivation_update, "get_runtime_service", lambda name, default=None: repo)
    return budgets["social"]


def test_being_reached_meets_as_much_of_the_need_as_is_unmet(monkeypatch):
    social = _with_social_at(monkeypatch, 20.0)
    assert met_by_somebody(15.0) == pytest.approx(12.0)
    assert social["level"] == pytest.approx(32.0)


def test_a_conversation_brings_her_out_of_depletion(monkeypatch):
    social = _with_social_at(monkeypatch, 20.0)
    for _turn in range(12):
        met_by_somebody(15.0)
    assert social["level"] > 85.0


def test_a_need_already_met_takes_nothing_more(monkeypatch):
    social = _with_social_at(monkeypatch, 100.0)
    assert met_by_somebody(15.0) == 0.0
    assert social["level"] == 100.0


def test_with_no_state_registered_nothing_is_credited(monkeypatch):
    monkeypatch.setattr(motivation_update, "get_runtime_service", lambda name, default=None: None)
    assert met_by_somebody(15.0) == 0.0


def test_the_incoming_path_credits_it_on_a_message():
    assert "met_by_somebody(15.0)" in inspect.getsource(incoming_logic)
