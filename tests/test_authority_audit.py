"""Tests for the authority audit receipt system.

Proves that:
  1. Receipt IDs are generated deterministically from authorize() params
  2. Effects with valid receipt_ids match correctly
  3. Effects WITHOUT receipt_ids are flagged as unmatched
  4. Effects with WRONG receipt_ids are flagged as unmatched
  5. BLOCK verdicts do NOT produce matching receipts (only ALLOW/CONSTRAIN/CRITICAL_PASS)
  6. Coverage ratio is computed correctly
  7. The full authorize→effect→verify chain works end-to-end
"""

import pytest
from core.consciousness.authority_audit import AuthorityAudit, _make_receipt_id
from core.consciousness.substrate_authority import (
    SubstrateAuthority, ActionCategory, AuthorizationDecision,
)


def _make_healthy_authority():
    """Authority with healthy refs that will ALLOW."""
    class HealthyField:
        def get_coherence(self): return 0.8

    class HealthySomatic:
        def evaluate(self, c, s, p):
            class V:
                approach_score = 0.3
                confidence = 0.3
                budget_available = True
            return V()

    class HealthyChem:
        class Chem:
            def __init__(self, eff):
                self.effective = eff
                self.surges = []

            def surge(self, amount):
                self.surges.append(amount)
        chemicals = {
            "cortisol": Chem(0.3), "gaba": Chem(0.5),
            "dopamine": Chem(0.5), "norepinephrine": Chem(0.4),
        }
        def on_frustration(self, amount):
            self.last_frustration = amount

    auth = SubstrateAuthority()
    auth._field_ref = HealthyField()
    auth._somatic_ref = HealthySomatic()
    auth._neurochemical_ref = HealthyChem()
    return auth


class TestReceiptIdGeneration:

    def test_receipt_id_is_deterministic(self):
        id1 = _make_receipt_id(1000.0, "test", "content")
        id2 = _make_receipt_id(1000.0, "test", "content")
        assert id1 == id2

    def test_receipt_id_differs_by_source(self):
        id1 = _make_receipt_id(1000.0, "source_a", "content")
        id2 = _make_receipt_id(1000.0, "source_b", "content")
        assert id1 != id2

    def test_receipt_id_differs_by_content(self):
        id1 = _make_receipt_id(1000.0, "test", "content_a")
        id2 = _make_receipt_id(1000.0, "test", "content_b")
        assert id1 != id2

    def test_authorize_returns_receipt_id(self):
        auth = _make_healthy_authority()
        verdict = auth.authorize("test", "test_source", ActionCategory.RESPONSE, 0.5)
        assert verdict.receipt_id != ""
        assert len(verdict.receipt_id) == 16  # sha256[:16]


class TestExactProvenanceMatching:

    def test_effect_with_valid_receipt_matches(self):
        audit = AuthorityAudit()
        rid = _make_receipt_id(1000.0, "test", "content")
        audit.record_receipt(rid, "content", "test", "RESPONSE", 0.5, "ALLOW")
        audit.record_effect("response", "test", "content", receipt_id=rid)

        report = audit.verify()
        assert report["matched_effects"] == 1
        assert report["unmatched_effects"] == 0
        assert report["coverage_ratio"] == 1.0
        assert report["verdict"] == "CLEAN"

    def test_effect_without_receipt_is_unmatched(self):
        audit = AuthorityAudit()
        audit.record_effect("response", "test", "content", receipt_id=None)

        report = audit.verify()
        assert report["unmatched_effects"] == 1
        assert report["coverage_ratio"] == 0.0
        assert report["verdict"] == "UNMATCHED_EFFECTS_FOUND"

    def test_effect_with_wrong_receipt_is_unmatched(self):
        audit = AuthorityAudit()
        real_rid = _make_receipt_id(1000.0, "real", "content")
        fake_rid = "aaaa_not_a_real_id"
        audit.record_receipt(real_rid, "content", "real", "RESPONSE", 0.5, "ALLOW")
        audit.record_effect("response", "test", "content", receipt_id=fake_rid)

        report = audit.verify()
        assert report["unmatched_effects"] == 1

    def test_block_receipt_does_not_match_effect(self):
        """BLOCKED verdicts should NOT produce valid matching receipts."""
        audit = AuthorityAudit()
        rid = _make_receipt_id(1000.0, "test", "content")
        audit.record_receipt(rid, "content", "test", "RESPONSE", 0.5, "BLOCK")
        audit.record_effect("response", "test", "content", receipt_id=rid)

        report = audit.verify()
        assert report["unmatched_effects"] == 1  # BLOCK receipt doesn't count

    def test_constrain_receipt_matches(self):
        audit = AuthorityAudit()
        rid = _make_receipt_id(1000.0, "test", "content")
        audit.record_receipt(rid, "content", "test", "RESPONSE", 0.5, "CONSTRAIN")
        audit.record_effect("response", "test", "content", receipt_id=rid)

        report = audit.verify()
        assert report["matched_effects"] == 1
        assert report["verdict"] == "CLEAN"

    def test_critical_pass_receipt_matches(self):
        audit = AuthorityAudit()
        rid = _make_receipt_id(1000.0, "test", "content")
        audit.record_receipt(rid, "content", "test", "RESPONSE", 0.5, "CRITICAL_PASS")
        audit.record_effect("response", "test", "content", receipt_id=rid)

        report = audit.verify()
        assert report["matched_effects"] == 1


class TestEndToEndAuthorizeToEffect:

    def test_full_chain_produces_clean_audit(self):
        """authorize() → receipt_id → record_effect(receipt_id) → verify() = CLEAN"""
        from core.consciousness.authority_audit import get_audit
        import importlib
        import core.consciousness.authority_audit as audit_mod
        # Use fresh audit instance
        audit = AuthorityAudit()
        audit_mod._instance = audit

        auth = _make_healthy_authority()
        verdict = auth.authorize("hello world", "user", ActionCategory.RESPONSE, 0.5)
        assert verdict.decision == AuthorizationDecision.ALLOW
        assert verdict.receipt_id != ""

        # Simulate the effect recording that would happen at the output point
        audit.record_effect("response", "user", "hello world", receipt_id=verdict.receipt_id)

        report = audit.verify()
        assert report["total_receipts"] >= 1
        assert report["total_effects"] == 1
        assert report["matched_effects"] == 1
        assert report["unmatched_effects"] == 0
        assert report["verdict"] == "CLEAN"

        # Restore
        audit_mod._instance = None

    def test_multiple_actions_all_matched(self):
        from core.consciousness.authority_audit import AuthorityAudit
        import core.consciousness.authority_audit as audit_mod
        audit = AuthorityAudit()
        audit_mod._instance = audit

        auth = _make_healthy_authority()

        actions = [
            ("respond to user", "user", ActionCategory.RESPONSE),
            ("store memory", "memory", ActionCategory.MEMORY_WRITE),
            ("run tool", "agency", ActionCategory.TOOL_EXECUTION),
        ]

        for content, source, category in actions:
            verdict = auth.authorize(content, source, category, 0.5)
            assert verdict.decision in (AuthorizationDecision.ALLOW, AuthorizationDecision.CONSTRAIN)
            audit.record_effect(category.name.lower(), source, content, receipt_id=verdict.receipt_id)

        report = audit.verify()
        assert report["total_effects"] == 3
        assert report["matched_effects"] == 3
        assert report["unmatched_effects"] == 0
        assert report["verdict"] == "CLEAN"

        audit_mod._instance = None

    def test_one_unmatched_effect_breaks_clean(self):
        """If even ONE effect is unmatched, verdict is not CLEAN."""
        from core.consciousness.authority_audit import AuthorityAudit
        import core.consciousness.authority_audit as audit_mod
        audit = AuthorityAudit()
        audit_mod._instance = audit

        auth = _make_healthy_authority()
        v = auth.authorize("legit", "user", ActionCategory.RESPONSE, 0.5)
        audit.record_effect("response", "user", "legit", receipt_id=v.receipt_id)

        # Rogue effect with no receipt
        audit.record_effect("tool_execution", "rogue", "unauthorized", receipt_id=None)

        report = audit.verify()
        assert report["unmatched_effects"] == 1
        assert report["verdict"] == "UNMATCHED_EFFECTS_FOUND"

        audit_mod._instance = None


class TestCoverageRatio:

    def test_empty_system_coverage(self):
        audit = AuthorityAudit()
        report = audit.verify()
        assert report["coverage_ratio"] == 0.0
        assert report["verdict"] == "CLEAN"  # no effects = clean

    def test_half_coverage(self):
        audit = AuthorityAudit()
        rid = _make_receipt_id(1.0, "a", "x")
        audit.record_receipt(rid, "x", "a", "R", 0.5, "ALLOW")
        audit.record_effect("response", "a", "x", receipt_id=rid)     # matched
        audit.record_effect("response", "b", "y", receipt_id=None)    # unmatched

        report = audit.verify()
        assert report["coverage_ratio"] == 0.5
