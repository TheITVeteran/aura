"""Internal maintenance writes must carry their own governed scope.

Every consequential write goes through the file-write gateway, and the
gateway refuses one that arrives outside a governed context. Two live
subsystems were doing exactly that, and the consequences were not merely
cosmetic:

* ``adaptation.heuristic_synthesizer.state`` — raised
  GovernanceViolationError out of ``curiosity_explorer``, which is
  fail-closed, so an ordinary heuristic save became a CRITICAL SERVICE
  FAILURE and drove the felt existential threat to 0.99 at boot
  (2026-07-25 launch log).
* ``brain.cognitive.integrity_check.audit_log`` — the belief-integrity
  audit trail was refused and silently lost (2026-07-18 launch log). An
  audit log that disappears under governance is worse than no audit log.

These run under production governance, where an unscoped write RAISES.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture()
def production_governance(monkeypatch):
    """The mode where an unscoped consequential write hard-fails."""
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "production")
    yield


class TestHeuristicSynthesizerSave:
    def test_save_survives_production_governance(self, production_governance):
        from core.adaptation.heuristic_synthesizer import HeuristicSynthesizer

        with tempfile.TemporaryDirectory() as tmp:
            synth = HeuristicSynthesizer()
            synth.heuristics_path = Path(tmp) / "heuristics.json"
            synth._active_heuristics = [{"rule": "prefer measured evidence", "hits": 3}]

            synth._save()  # must not raise

            written = json.loads(synth.heuristics_path.read_text(encoding="utf-8"))
            assert written["heuristics"][0]["rule"] == "prefer measured evidence"
            assert "updated_at" in written

    def test_save_declares_its_scope_at_the_call_site(self):
        import inspect

        from core.adaptation.heuristic_synthesizer import HeuristicSynthesizer

        source = inspect.getsource(HeuristicSynthesizer._save)
        assert "local_internal_governed_scope" in source


class TestIntegrityAuditLog:
    def test_audit_log_declares_its_scope_at_the_call_site(self):
        import inspect

        from core.brain.cognitive.integrity_check import IntegrityGuard

        source = inspect.getsource(IntegrityGuard._write_audit_log)
        assert "local_internal_governed_scope" in source
        assert "append_text" in source
