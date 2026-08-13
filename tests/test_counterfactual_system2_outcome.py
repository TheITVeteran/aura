"""Counterfactual search teaches only from the action that actually executes."""

from __future__ import annotations

import pytest

from core.cognition.outcome_ledger import OutcomeLedger
from core.consciousness.counterfactual_engine import CounterfactualEngine
from core.container import ServiceContainer
from core.reasoning.action_value import ActionValueModel
from core.reasoning.native_system2 import NativeSystem2Engine


@pytest.mark.asyncio
async def test_selected_counterfactual_opens_and_resolves_measured_outcome(
    monkeypatch, tmp_path
):
    ledger = OutcomeLedger(db_path=str(tmp_path / "outcomes.db"))
    monkeypatch.setattr(
        "core.cognition.outcome_ledger.get_outcome_ledger", lambda: ledger
    )

    # Keep the test hermetic: caller scores drive this search, and the singleton
    # must not read the user's ordinary outcome database.
    import core.reasoning.action_value as action_value_module

    monkeypatch.setattr(action_value_module, "_model", ActionValueModel({}))
    engine = NativeSystem2Engine()
    monkeypatch.setattr(ServiceContainer, "_services", dict(ServiceContainer._services))
    monkeypatch.setattr(ServiceContainer, "_aliases", dict(ServiceContainer._aliases))
    ServiceContainer.register_instance("native_system2", engine)

    counterfactual = CounterfactualEngine()
    candidates = await counterfactual.deliberate(
        [
            {"type": "learn", "description": "study the evidence", "params": {}},
            {"type": "rest", "description": "pause", "params": {}},
        ],
        {"curiosity": 0.9, "valence": 0.1},
    )

    assert ledger.pending() == [], "deliberation alone opened outcome evidence"
    selected = counterfactual.select(candidates)
    assert selected is not None
    assert selected.system2_outcome_receipt_id
    assert len(ledger.pending()) == 1

    counterfactual.record_outcome(selected, actual_hedonic_change=0.4, candidates=candidates)

    assert ledger.pending() == []
    stats = ledger.measured_action_stats()
    assert stats
    assert next(iter(stats.values()))["mean"] == pytest.approx(0.7)
