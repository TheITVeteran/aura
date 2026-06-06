"""core/executive/executive_kernel.py
Unified Executive Kernel directing action selection and decision logs.
"""
from typing import Dict, List, Any, Optional
import logging

from core.executive.action_arbitrator import ActionArbitrator
from core.executive.attention_controller import AttentionController
from core.executive.conflict_resolver import ExecutiveConflictResolver
from core.executive.inhibition_system import ActionInhibitor
from core.executive.permission_router import PermissionRouter
from core.executive.decision_receipt import DecisionReceiptCompiler

logger = logging.getLogger("Executive.ExecutiveKernel")


class DeliberationEngine:
    """Executes reasoning loops and updates pending action list."""

    async def deliberate(self, state: Any) -> None:
        """Assembles logical plans and appends planned actions to state."""
        # Simple heuristic: if goals are active, convert to action intents
        for g in state.cognition.current_goals:
            if g.get("status") in ["pending", "resumed", "unblocked"]:
                g["status"] = "in_progress"
                # Queue a placeholder gesture or checkup command representing task work
                state.cognition.pending_actions.append({
                    "channel": "gesture",
                    "params": {"gesture": f"process_goal_{g.get('id')}"}
                })


class ExecutiveKernel:
    """Canonical single-will executive controller for Aura."""

    def __init__(self):
        self.arbitrator = ActionArbitrator()
        self.attention = AttentionController()
        self.resolver = ExecutiveConflictResolver()
        self.inhibitor = ActionInhibitor()
        self.router = PermissionRouter()
        self.receipt_compiler = DecisionReceiptCompiler()

    async def evaluate_tick_decisions(self, state: Any) -> None:
        """Called during the life loop tick to resolve competing goals."""
        # Clean duplicate goal nodes
        state.cognition.current_goals = self.resolver.resolve_goal_clashes(state.cognition.current_goals)

        # Sort pending actions
        state.cognition.pending_actions = self.arbitrator.arbitrate(state.cognition.pending_actions)

        # Focus attention
        state.cognition.active_attention = await self.attention.focus_attention(state)
        
        logger.info("Executive Kernel tick completed. Focus: %s", state.cognition.active_attention)
