"""The heartbeat appraises through the hierarchical agency ladder — no longer an island.

Proves the live cognitive loop turns its real internal signals into a Situation, dispatches
it through the one control ladder, and that the dispatch does causal work (opens an
outcome-ledger receipt). This is the action-side of unity: the agency ladder, nociception,
and the outcome ledger are exercised by the heartbeat instead of sitting idle.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from core.cognitive_loop import CognitiveLoop


def _loop():
    # Bypass __init__ (which spins a ProcessPoolExecutor) — we only exercise one method.
    loop = CognitiveLoop.__new__(CognitiveLoop)
    loop.orchestrator = SimpleNamespace()
    return loop


def test_heartbeat_appraisal_dispatches_and_records_a_receipt(monkeypatch, tmp_path):
    import core.cognition.outcome_ledger as ol
    monkeypatch.setattr(
        ol,
        "_ledger",
        ol.OutcomeLedger(db_path=str(tmp_path / "aura_test_hb_ledger.db")),
    )

    # Reset agency singleton so it uses the (ledger-enabled) default.
    import core.agency.hierarchical_agency as ha
    monkeypatch.setattr(ha, "_agency", None)

    from core.affect.nociception import DamageChannel, get_nociception_engine
    noci = get_nociception_engine()
    noci.reset()
    noci.register_damage(DamageChannel.GOVERNANCE_BREACH, 0.95)  # acute → reflex tier

    fe_state = SimpleNamespace(arousal=0.5, dominant_action="explore", valence=-0.3)

    before = len(ol.get_outcome_ledger().pending())
    asyncio.run(_loop()._appraise_through_agency(fe_state, spl=None))
    after = len(ol.get_outcome_ledger().pending())

    assert after > before, "heartbeat appraisal did not open an outcome-ledger receipt"
    # the most recent receipt is an agency dispatch crediting a tier
    latest = ol.get_outcome_ledger().pending()[-1]
    assert latest["action"].startswith("agency:")
    noci.reset()


def test_appraisal_is_fail_open(monkeypatch):
    # Even if the agency module blows up, the heartbeat must not raise.
    import core.agency.hierarchical_agency as ha

    calls = []

    def _boom():
        calls.append("called")
        raise RuntimeError("agency down")

    monkeypatch.setattr(ha, "get_hierarchical_agency", _boom)
    fe_state = SimpleNamespace(arousal=0.1, dominant_action="idle", valence=0.0)
    # should swallow the error, not propagate
    asyncio.run(_loop()._appraise_through_agency(fe_state, spl=None))
    assert calls == ["called"]
