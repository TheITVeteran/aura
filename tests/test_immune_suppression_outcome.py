"""Regulatory cells must be scored on whether suppressing was RIGHT.

CP126 b694c436: suppression on protected tissue incremented `successes` and
raised fitness the instant it happened — for suppressing at all, not for
suppressing correctly. Silencing a genuine threat that happens to sit on
protected tissue scored exactly like preventing an autoimmune response, so
the regulatory lineage was selected for quietness rather than judgement.
"""
from __future__ import annotations

import time

import pytest

from core.adaptation.adaptive_immunity import CellKind, get_adaptive_immune_system


@pytest.fixture()
def immune():
    system = get_adaptive_immune_system()
    system._pending_suppressions.clear()
    return system


def _regulatory(immune):
    """A regulatory cell in the live population — created if none exists.

    Skipping here would have hidden every assertion below, which is the
    failure mode this whole campaign is about.
    """
    for cell in immune._cells:
        if cell.kind == CellKind.REGULATORY:
            return cell
    import numpy as np

    from core.adaptation.adaptive_immunity import ImmuneCell

    cell = ImmuneCell(
        cell_id="test-regulatory",
        lineage_id="test-regulatory",
        kind=CellKind.REGULATORY,
        receptor=np.zeros(immune.expansion_engine.current_dim, dtype=np.float32),
        subsystem_scope="memory",
        persistence=0.72,
    )
    immune._cells.append(cell)
    return cell


def _park(immune, cell, *, subsystem="memory", danger=0.5, at=None):
    immune._pending_suppressions[cell.cell_id].append(
        {
            "subsystem": subsystem,
            "danger": danger,
            "suppression": 0.5,
            "at": time.time() if at is None else at,
        }
    )


def test_suppression_alone_earns_no_success(immune):
    """The act is not the outcome."""
    cell = _regulatory(immune)
    before = cell.successes

    _park(immune, cell)

    assert cell.successes == before


def test_a_louder_recurrence_means_the_suppression_was_wrong(immune):
    cell = _regulatory(immune)
    before_successes = cell.successes
    before_fitness = cell.fitness
    _park(immune, cell, subsystem="memory", danger=0.4)

    immune._settle_suppressions("memory", 0.9)

    assert cell.successes == before_successes      # no credit
    assert cell.fitness < before_fitness           # penalised
    assert cell.cell_id not in immune._pending_suppressions


def test_staying_quiet_through_the_window_earns_the_credit(immune):
    cell = _regulatory(immune)
    before = cell.successes
    _park(
        immune, cell, subsystem="memory", danger=0.4,
        at=time.time() - immune._suppression_verdict_window_s - 1,
    )

    immune._settle_suppressions("other_subsystem", 0.1)

    assert cell.successes == before + 1
    assert cell.cell_id not in immune._pending_suppressions


def test_a_quieter_return_is_not_counted_as_failure(immune):
    """Coming back calmer is consistent with the suppression having held."""
    cell = _regulatory(immune)
    before = cell.fitness
    _park(immune, cell, subsystem="memory", danger=0.8)

    immune._settle_suppressions("memory", 0.2)

    assert cell.fitness == before                  # unresolved, not penalised
    assert immune._pending_suppressions[cell.cell_id]


def test_a_different_subsystem_does_not_settle_the_verdict(immune):
    cell = _regulatory(immune)
    _park(immune, cell, subsystem="memory", danger=0.4)

    immune._settle_suppressions("network", 0.99)

    assert immune._pending_suppressions[cell.cell_id]


def test_settling_is_safe_when_the_cell_is_gone(immune):
    immune._pending_suppressions["not-a-real-cell"].append(
        {"subsystem": "memory", "danger": 0.4, "suppression": 0.5, "at": time.time()}
    )

    immune._settle_suppressions("memory", 0.9)      # must not raise

    assert "not-a-real-cell" not in immune._pending_suppressions


def test_the_immediate_credit_is_gone():
    import inspect

    from core.adaptation import adaptive_immunity as mod

    source = inspect.getsource(mod.AdaptiveImmuneSystem)
    assert "dominant_regulatory.successes += 1" not in source
