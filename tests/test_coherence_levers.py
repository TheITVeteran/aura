"""Documented levers must exist, and must actually move the number.

A documented control that silently does nothing is worse than an undocumented
one: it gets pulled during an incident and the absence of an effect is read as
evidence about the system rather than about the lever.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from core.brain.llm import continuity_ledger
from core.brain.llm.context_assembler import ContextAssembler

DOC = pathlib.Path(__file__).resolve().parent.parent / "docs" / "COHERENCE_LEVERS.md"


def _documented_levers() -> set[str]:
    return set(re.findall(r"AURA_[A-Z0-9_]+", DOC.read_text(encoding="utf-8")))


def _source_levers() -> set[str]:
    import inspect

    text = inspect.getsource(continuity_ledger) + inspect.getsource(ContextAssembler)
    return set(re.findall(r'"(AURA_CONTINUITY[A-Z0-9_]*)"', text))


def test_every_implemented_lever_is_documented():
    missing = _source_levers() - _documented_levers()
    assert not missing, f"undocumented levers: {sorted(missing)}"


def test_every_documented_lever_exists_in_source():
    documented = {n for n in _documented_levers() if n.startswith("AURA_CONTINUITY")}
    phantom = documented - _source_levers()
    assert not phantom, f"documented but not implemented: {sorted(phantom)}"


@pytest.mark.parametrize(
    "floor,ceiling,ramp,depth,expected",
    [
        (1800, 4800, 40, 0, 1800),
        (1800, 4800, 40, 40, 4800),
        (1800, 4800, 40, 200, 4800),
        (400, 400, 40, 46, 400),      # the old, broken behaviour, on purpose
        (2000, 9000, 25, 25, 9000),
    ],
)
def test_budget_levers_move_the_number(monkeypatch, floor, ceiling, ramp, depth, expected):
    monkeypatch.setenv("AURA_CONTINUITY_FLOOR_CHARS", str(floor))
    monkeypatch.setenv("AURA_CONTINUITY_CEILING_CHARS", str(ceiling))
    monkeypatch.setenv("AURA_CONTINUITY_RAMP_TURNS", str(ramp))
    assert ContextAssembler._continuity_budget_chars(depth) == expected


def test_ledger_budget_lever_moves_the_cap(monkeypatch):
    monkeypatch.setenv("AURA_CONTINUITY_LEDGER_CHARS", "800")
    assert continuity_ledger.ledger_budget_chars() == 800

    ledger = continuity_ledger.ContinuityLedger()
    ledger.observe(
        [{"role": "user", "content": f"I have always loved topic {i} quite a lot."}
         for i in range(80)]
    )
    assert len(ledger.render()) <= 800


def test_salience_weight_lever_changes_what_survives(monkeypatch):
    monkeypatch.setenv("AURA_CONTINUITY_W_DISCLOSURE", "99")
    entry = continuity_ledger.LedgerEntry(
        kind="disclosure", text="x", speaker="user", first_turn=1, last_turn=1
    )
    high = entry.salience(10)
    monkeypatch.setenv("AURA_CONTINUITY_W_DISCLOSURE", "0.1")
    assert entry.salience(10) < high


def test_a_malformed_lever_falls_back_instead_of_crashing(monkeypatch):
    monkeypatch.setenv("AURA_CONTINUITY_LEDGER_CHARS", "not-a-number")
    assert continuity_ledger.ledger_budget_chars() == 3200


def test_levers_are_read_at_call_time_not_import_time(monkeypatch):
    """A reboot must not be required to change one."""
    first = continuity_ledger.ledger_budget_chars()
    monkeypatch.setenv("AURA_CONTINUITY_LEDGER_CHARS", str(first + 500))
    assert continuity_ledger.ledger_budget_chars() == first + 500
