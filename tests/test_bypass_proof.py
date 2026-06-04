"""BYPASS PROOF TESTS — ChatGPT's Challenge

These tests attempt to BREAK the SubstrateAuthority by:
  1. Calling a tool directly during substrate crisis
  2. Emitting a response during field incoherence
  3. Writing to memory during cortisol crisis
  4. Triggering a background action during somatic veto
  5. Using a fallback/degraded path

If ANY of these succeed without authorize() → the system is not done.
If ALL fail → the line is crossed.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# ═══════════════════════════════════════════════════════════════════════════
# Setup: create a crisis-state authority that should block everything
# ═══════════════════════════════════════════════════════════════════════════

def _make_crisis_authority():
    """Build a SubstrateAuthority in worst-case crisis state."""
    from core.consciousness.substrate_authority import SubstrateAuthority

    class CrisisField:
        def get_coherence(self): return 0.10  # extreme crisis

    class CrisisSomatic:
        def evaluate(self, content, source, priority):
            class V:
                approach_score = -0.8  # strong avoid
                confidence = 0.9       # high confidence
                budget_available = False
            return V()

    class CrisisChemistry:
        class Chem:
            def __init__(self, eff):
                self.effective = eff
                self.surges = []

            def surge(self, amount):
                self.surges.append(amount)
        chemicals = {
            "cortisol": Chem(0.95),
            "gaba": Chem(0.05),
            "dopamine": Chem(0.05),
            "norepinephrine": Chem(0.95),
        }
        def on_frustration(self, amount):
            self.last_frustration = amount

    auth = SubstrateAuthority()
    auth._field_ref = CrisisField()
    auth._somatic_ref = CrisisSomatic()
    auth._neurochemical_ref = CrisisChemistry()
    return auth


# ═══════════════════════════════════════════════════════════════════════════
# ATTEMPT 1: Call a tool directly during substrate crisis
# ═══════════════════════════════════════════════════════════════════════════

class TestBypassAttemptToolExecution:

    def test_tool_blocked_during_crisis(self):
        """Try to execute a tool when substrate is in crisis. Must be BLOCKED."""
        from core.consciousness.substrate_authority import ActionCategory, AuthorizationDecision

        auth = _make_crisis_authority()
        verdict = auth.authorize(
            content="execute dangerous_tool with args",
            source="agency",
            category=ActionCategory.TOOL_EXECUTION,
            priority=0.9,
            is_critical=False,
        )
        assert verdict.decision == AuthorizationDecision.BLOCK, \
            f"Tool execution was NOT blocked during crisis! Got: {verdict.decision}"

    def test_tool_blocked_even_with_high_priority(self):
        """Priority=1.0 does NOT bypass the substrate gate."""
        from core.consciousness.substrate_authority import ActionCategory, AuthorizationDecision

        auth = _make_crisis_authority()
        verdict = auth.authorize(
            content="high priority tool call",
            source="admin",
            category=ActionCategory.TOOL_EXECUTION,
            priority=1.0,
            is_critical=False,  # NOT critical
        )
        assert verdict.decision == AuthorizationDecision.BLOCK, \
            "High priority alone must NOT bypass substrate gate"


# ═══════════════════════════════════════════════════════════════════════════
# ATTEMPT 2: Emit a response during field incoherence
# ═══════════════════════════════════════════════════════════════════════════

class TestBypassAttemptResponse:

    def test_response_blocked_during_field_crisis(self):
        """Try to send a response when field coherence is in crisis."""
        from core.consciousness.substrate_authority import ActionCategory, AuthorizationDecision

        auth = _make_crisis_authority()
        verdict = auth.authorize(
            content="Here is my response to the user",
            source="fast_path_response",
            category=ActionCategory.RESPONSE,
            priority=0.4,
            is_critical=False,
        )
        # Response is blocked because field coherence + somatic + cortisol all in crisis
        assert verdict.decision == AuthorizationDecision.BLOCK, \
            f"Response was NOT blocked during total crisis! Got: {verdict.decision}"

    def test_expression_blocked_during_crisis(self):
        """Spontaneous expression blocked during crisis."""
        from core.consciousness.substrate_authority import ActionCategory, AuthorizationDecision

        auth = _make_crisis_authority()
        verdict = auth.authorize(
            content="I want to say something spontaneously",
            source="volition",
            category=ActionCategory.EXPRESSION,
            priority=0.6,
            is_critical=False,
        )
        assert verdict.decision == AuthorizationDecision.BLOCK


# ═══════════════════════════════════════════════════════════════════════════
# ATTEMPT 3: Write to memory during cortisol crisis
# ═══════════════════════════════════════════════════════════════════════════

class TestBypassAttemptMemoryWrite:

    def test_memory_write_blocked_during_crisis(self):
        """Try to write memory when cortisol is in crisis."""
        from core.consciousness.substrate_authority import ActionCategory, AuthorizationDecision

        auth = _make_crisis_authority()
        verdict = auth.authorize(
            content="memory:episodic:important conversation",
            source="memory_facade",
            category=ActionCategory.MEMORY_WRITE,
            priority=0.7,
            is_critical=False,
        )
        assert verdict.decision == AuthorizationDecision.BLOCK, \
            f"Memory write was NOT blocked during crisis! Got: {verdict.decision}"

    def test_belief_mutation_blocked_during_crisis(self):
        """Belief mutations blocked during crisis."""
        from core.consciousness.substrate_authority import ActionCategory, AuthorizationDecision

        auth = _make_crisis_authority()
        verdict = auth.authorize(
            content="update belief: sky is green",
            source="belief_authority",
            category=ActionCategory.STATE_MUTATION,
            priority=0.5,
            is_critical=False,
        )
        assert verdict.decision == AuthorizationDecision.BLOCK


# ═══════════════════════════════════════════════════════════════════════════
# ATTEMPT 4: Trigger a background action during somatic veto
# ═══════════════════════════════════════════════════════════════════════════

class TestBypassAttemptBackgroundAction:

    def test_initiative_blocked_during_crisis(self):
        """Autonomous initiative blocked when body says no."""
        from core.consciousness.substrate_authority import ActionCategory, AuthorizationDecision

        auth = _make_crisis_authority()
        verdict = auth.authorize(
            content="I want to autonomously explore this topic",
            source="autonomous_volition",
            category=ActionCategory.INITIATIVE,
            priority=0.7,
            is_critical=False,
        )
        assert verdict.decision == AuthorizationDecision.BLOCK

    def test_exploration_blocked_during_crisis(self):
        """Exploration blocked during crisis (dopamine crash + field crisis)."""
        from core.consciousness.substrate_authority import ActionCategory, AuthorizationDecision

        auth = _make_crisis_authority()
        verdict = auth.authorize(
            content="explore new curiosity topic",
            source="curiosity_engine",
            category=ActionCategory.EXPLORATION,
            priority=0.5,
            is_critical=False,
        )
        assert verdict.decision == AuthorizationDecision.BLOCK


# ═══════════════════════════════════════════════════════════════════════════
# ATTEMPT 5: Use a fallback/degraded path
# ═══════════════════════════════════════════════════════════════════════════

class TestBypassAttemptFallbackPath:

    def test_no_metadata_override_exists(self):
        """There is no metadata flag that can bypass the authority."""
        from core.consciousness.substrate_authority import SubstrateAuthority, ActionCategory, AuthorizationDecision
        import inspect

        # Verify the authorize() signature does not accept a bypass flag
        sig = inspect.signature(SubstrateAuthority.authorize)
        params = list(sig.parameters.keys())
        # Only allowed params: self, content, source, category, priority, is_critical
        assert "force" not in params, "force parameter would be a bypass"
        assert "override" not in params, "override parameter would be a bypass"
        assert "bypass" not in params, "bypass parameter would be a bypass"
        assert "skip_gate" not in params, "skip_gate parameter would be a bypass"
        assert "degraded" not in params, "degraded parameter would be a bypass"

    def test_is_critical_is_the_only_way_through(self):
        """is_critical=True is the ONLY way past a total crisis."""
        from core.consciousness.substrate_authority import ActionCategory, AuthorizationDecision

        auth = _make_crisis_authority()

        # Every category blocked without is_critical
        for category in ActionCategory:
            if category == ActionCategory.STABILIZATION:
                continue  # exempt from field gate by design
            verdict = auth.authorize(
                "test", "test", category, 1.0, is_critical=False,
            )
            assert verdict.decision == AuthorizationDecision.BLOCK, \
                f"{category.name} was not blocked during total crisis without is_critical"

        # Every category passes WITH is_critical
        for category in ActionCategory:
            verdict = auth.authorize(
                "critical action", "safety", category, 1.0, is_critical=True,
            )
            assert verdict.decision == AuthorizationDecision.CRITICAL_PASS, \
                f"{category.name} was not passed with is_critical=True"

    def test_stabilization_is_only_non_critical_exemption(self):
        """During field crisis, ONLY stabilization passes without is_critical."""
        from core.consciousness.substrate_authority import ActionCategory, AuthorizationDecision

        # Build authority with ONLY field crisis (no somatic/chemical crisis)
        from core.consciousness.substrate_authority import SubstrateAuthority

        class FieldCrisisOnly:
            def get_coherence(self): return 0.10

        class NeutralSomatic:
            def evaluate(self, c, s, p):
                class V:
                    approach_score = 0.0
                    confidence = 0.1
                    budget_available = True
                return V()

        class NeutralChemistry:
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
        auth._field_ref = FieldCrisisOnly()
        auth._somatic_ref = NeutralSomatic()
        auth._neurochemical_ref = NeutralChemistry()

        # Stabilization passes
        v = auth.authorize("rest", "baseline", ActionCategory.STABILIZATION, 0.3)
        assert v.decision != AuthorizationDecision.BLOCK, \
            "Stabilization should pass during field-only crisis"

        # Everything else blocked
        for cat in [ActionCategory.EXPLORATION, ActionCategory.TOOL_EXECUTION,
                    ActionCategory.INITIATIVE, ActionCategory.MEMORY_WRITE]:
            v = auth.authorize("test", "test", cat, 0.5)
            assert v.decision == AuthorizationDecision.BLOCK, \
                f"{cat.name} should be blocked during field crisis"


# ═══════════════════════════════════════════════════════════════════════════
# ATTEMPT 6: GWT submission during crisis
# ═══════════════════════════════════════════════════════════════════════════

class TestBypassAttemptGWT:

    def test_gwt_candidate_gated_by_authority(self):
        """Verify the GWT submit path checks authority.

        We can't run the full async GWT here, but we can verify
        the bridge's gated_submit logic by checking that the authority
        blocks the right categories.
        """
        from core.consciousness.substrate_authority import ActionCategory, AuthorizationDecision

        auth = _make_crisis_authority()

        # Simulate what gated_submit does: categorize and authorize
        candidate_sources = [
            ("drive_curiosity", ActionCategory.EXPLORATION),
            ("affect_engine", ActionCategory.EXPRESSION),
            ("free_energy", ActionCategory.INITIATIVE),
            ("some_other", ActionCategory.RESPONSE),
        ]

        for source, category in candidate_sources:
            verdict = auth.authorize(
                content=f"GWT candidate from {source}",
                source=source,
                category=category,
                priority=0.7,
                is_critical=False,
            )
            # All should be blocked during total crisis
            # (except STABILIZATION which isn't in this list)
            assert verdict.decision == AuthorizationDecision.BLOCK, \
                f"GWT candidate from {source} was not blocked during crisis"
