"""Sensorimotor grounding — action, prediction, verified reality, surprise.

Pins: filesystem predicates verify actual world-state (never the tool's
claim); a claimed success without the predicted effect records the
confabulated-action fault; the outcome ledger receives expectation →
observation with prediction error.
"""
from __future__ import annotations

from core.grounding.sensorimotor_loop import (
    build_predicate,
    ground_result,
    open_grounding,
)


class TestPredicates:
    def test_write_predicate_verifies_real_file(self, tmp_path):
        target = tmp_path / "note.txt"
        predicate = build_predicate(
            "file_operation",
            {"action": "write", "path": str(target), "content": "dinosaur facts"},
        )
        assert predicate is not None
        verified, detail = predicate()
        assert verified is False and "absent" in detail
        target.write_text("dinosaur facts here")
        verified, detail = predicate()
        assert verified is True and "exists" in detail

    def test_mkdir_and_delete_predicates(self, tmp_path):
        folder = tmp_path / "Notes"
        made = build_predicate("file_operation", {"action": "mkdir", "path": str(folder)})
        assert made()[0] is False
        folder.mkdir()
        assert made()[0] is True

        gone = build_predicate("file_operation", {"action": "delete", "path": str(folder)})
        assert gone()[0] is False
        folder.rmdir()
        assert gone()[0] is True

    def test_unknown_tools_stay_ungrounded(self):
        assert build_predicate("native_chat", {"message": "hi"}) is None
        assert build_predicate("file_operation", {"action": "write"}) is None  # no path


class TestClaimRealityDiff:
    def test_confabulated_action_records_the_fault(self, tmp_path, monkeypatch):
        from core.resilience.fault_taxonomy import FaultRegistry

        registry = FaultRegistry()
        monkeypatch.setattr(
            "core.resilience.fault_taxonomy.get_fault_registry", lambda: registry,
        )
        target = tmp_path / "ghost.txt"
        predicate = build_predicate(
            "file_operation", {"action": "write", "path": str(target), "content": "x" * 40},
        )
        # The tool CLAIMS success; the file was never written.
        ground_result("file_operation", {}, {"ok": True}, predicate, None)
        assert registry.fault_count("ACTION-CLAIM-MISMATCH") == 1
        defn = registry.get_definition("ACTION-CLAIM-MISMATCH")
        assert defn is not None and defn.severity.name == "CRITICAL"

    def test_verified_success_records_no_fault(self, tmp_path, monkeypatch):
        from core.resilience.fault_taxonomy import FaultRegistry

        registry = FaultRegistry()
        monkeypatch.setattr(
            "core.resilience.fault_taxonomy.get_fault_registry", lambda: registry,
        )
        target = tmp_path / "real.txt"
        predicate = build_predicate(
            "file_operation", {"action": "write", "path": str(target), "content": "hello"},
        )
        target.write_text("hello world")
        ground_result("file_operation", {}, {"ok": True}, predicate, None)
        assert registry.fault_count("ACTION-CLAIM-MISMATCH") == 0


class TestLedgerCoupling:
    def test_expectation_opens_and_reality_resolves(self, tmp_path, monkeypatch):
        opened, resolved = [], []

        class _Ledger:
            def open(self, action, expected, **kw):
                opened.append((action, expected))
                return "rcpt-test"

            def resolve(self, receipt_id, observed, **kw):
                resolved.append((receipt_id, observed))

        monkeypatch.setattr(
            "core.cognition.outcome_ledger.get_outcome_ledger", lambda: _Ledger(),
        )
        target = tmp_path / "out.txt"
        predicate, receipt = open_grounding(
            "file_operation", {"action": "write", "path": str(target), "content": "abc"},
        )
        assert receipt == "rcpt-test"
        assert opened and opened[0][0].startswith("sensorimotor:file_operation")
        target.write_text("abc content")
        ground_result("file_operation", {}, {"ok": True}, predicate, receipt)
        assert resolved == [("rcpt-test", 1.0)]

    def test_engine_wiring(self):
        import inspect

        from core import capability_engine

        src = inspect.getsource(capability_engine)
        assert "open_grounding" in src and "ground_result" in src
