"""Standing-objective validity: the one authority for durable volition ingress.

Live evidence (July 2026): a check-in question ("Ok. Once more. You with me?")
and later a NetHack framebuffer spent days as CURRENT IMPERATIVE at urgency
0.98 because every organ trusted whatever text reached current_objective.
Quarantine-after-the-fact lost the race — the executive loop recreated the
goal faster than provenance repair abandoned it. These tests pin the refusal
at every ingress: predicate, actionable-goal gate, restore sanitizer,
proposal governance, arbiter quarantine, and goal-store repair.
"""
from __future__ import annotations

import pytest

from core.goals.standing_objective import (
    is_valid_standing_objective,
    standing_objective_rejection_reason,
)

CHAT_TURN = "Ok. Once more. You with me?"
NETHACK_RENDER = (
    "---+------------                    \n"
    "                                            |..............|          \n"
    "                                            |.....x........|          \n"
    "                                            |....@.........|          \n"
    "                                            ----------------          \n"
)
EMBODIED_CONTRACT = (
    "[EMBODIED CONTROL CONTRACT]\n"
    "- Output exactly one action marker and no conversational prose.\n"
    "- Marker format: [ACTION:execute_nethack_action] <action>\n"
    "- Valid actions: h, j, k, l, y, u, b, n, s, i\n"
)

VALID_GOALS = (
    "Investigate the correlation between memory pressure and tick latency",
    "Consolidate the latent-cortex evidence into the tracker document",
    "Research retrieval-augmented consolidation strategies for episodic memory",
)


class TestRejectionReasons:
    def test_chat_turn_rejected(self):
        assert standing_objective_rejection_reason(CHAT_TURN) == (
            "ephemeral_conversation_turn"
        )

    def test_screen_render_rejected(self):
        assert standing_objective_rejection_reason(NETHACK_RENDER) == (
            "nonlinguistic_render"
        )

    def test_control_contract_rejected(self):
        assert standing_objective_rejection_reason(EMBODIED_CONTRACT) == (
            "control_contract_scaffold"
        )

    def test_action_marker_alone_rejected(self):
        assert standing_objective_rejection_reason(
            "Respond with [ACTION:execute_nethack_action] j to descend"
        ) == "control_contract_scaffold"

    def test_overlong_raw_text_rejected(self):
        text = "Investigate the runtime because " + "it matters greatly " * 40
        assert standing_objective_rejection_reason(text) == "overlong_raw_text"

    def test_empty_rejected(self):
        assert standing_objective_rejection_reason("") == "empty"
        assert standing_objective_rejection_reason(None) == "empty"

    @pytest.mark.parametrize("goal", VALID_GOALS)
    def test_legitimate_goals_pass(self, goal):
        assert is_valid_standing_objective(goal), goal

    def test_dict_goal_uses_description(self):
        assert not is_valid_standing_objective({"description": NETHACK_RENDER})
        assert is_valid_standing_objective({"description": VALID_GOALS[0]})


class TestActionableGoalGate:
    def test_actionable_goal_text_rejects_all_leaked_classes(self):
        from core.goals.goal_text import is_actionable_goal_text

        assert not is_actionable_goal_text(CHAT_TURN)
        assert not is_actionable_goal_text(NETHACK_RENDER)
        assert not is_actionable_goal_text(EMBODIED_CONTRACT)
        assert is_actionable_goal_text(VALID_GOALS[0])


class TestRestoreSanitizer:
    def _state_with_objective(self, objective: str, origin: str):
        from core.state.aura_state import AuraState

        state = AuraState.default()
        state.cognition.current_objective = objective
        state.cognition.current_origin = origin
        return state

    def test_render_purged_regardless_of_origin(self):
        state = self._state_with_objective(NETHACK_RENDER, "embodied_motor_reflex")
        state.cognition.attention_focus = NETHACK_RENDER
        state.cognition.sanitize_restored_autonomy_state()
        assert state.cognition.current_objective is None
        assert state.cognition.current_origin == "system"

    def test_contract_scaffold_purged_from_goals_and_modifiers(self):
        state = self._state_with_objective(EMBODIED_CONTRACT, "embodied_motor_reflex")
        state.cognition.active_goals = [
            {"description": EMBODIED_CONTRACT, "source": "embodied"},
            {"description": VALID_GOALS[0], "source": "volition"},
        ]
        state.cognition.modifiers["executive_objective"] = NETHACK_RENDER
        state.cognition.modifiers["executive_hysteresis"] = {
            "active": True,
            "committed_objective": NETHACK_RENDER,
        }
        state.cognition.sanitize_restored_autonomy_state()
        descriptions = [g["description"] for g in state.cognition.active_goals]
        assert EMBODIED_CONTRACT not in descriptions
        assert VALID_GOALS[0] in descriptions
        assert "executive_objective" not in state.cognition.modifiers
        assert "executive_hysteresis" not in state.cognition.modifiers

    def test_valid_objective_survives_restore(self):
        state = self._state_with_objective(VALID_GOALS[0], "volition")
        state.cognition.sanitize_restored_autonomy_state()
        assert state.cognition.current_objective == VALID_GOALS[0]


class TestProposalGovernance:
    @pytest.mark.asyncio
    async def test_invalid_standing_objective_rejected_before_governance(self):
        from core.runtime.proposal_governance import (
            propose_governed_initiative_to_state,
        )
        from core.state.aura_state import AuraState

        state = AuraState.default()
        for bad in (CHAT_TURN, NETHACK_RENDER, EMBODIED_CONTRACT):
            _, decision = await propose_governed_initiative_to_state(
                state,
                bad,
                source="executive_closure",
                kind="executive_closure",
                urgency=0.98,
            )
            assert decision["action"] in {"rejected", "quarantined"}, bad
            assert (
                decision["reason"].startswith("standing_objective_invalid")
                or decision["reason"] == "transient_foreground_projection"
            ), decision


class TestArbiterQuarantine:
    def test_arbiter_quarantines_leaked_classes(self):
        from core.agency.initiative_arbiter import _is_quarantined_initiative

        for bad in (CHAT_TURN, NETHACK_RENDER, EMBODIED_CONTRACT):
            assert _is_quarantined_initiative({"goal": bad}), bad
        assert not _is_quarantined_initiative({"goal": VALID_GOALS[0]})


class TestGoalStoreRepair:
    @pytest.mark.asyncio
    async def test_quarantine_abandons_invalid_standing_objectives(self, tmp_path):
        from core.goals.goal_engine import GoalEngine

        engine = GoalEngine(db_path=str(tmp_path / "goal_lifecycle.db"))
        try:
            created: dict[str, str] = {}
            for text in (CHAT_TURN, NETHACK_RENDER, EMBODIED_CONTRACT, VALID_GOALS[0]):
                record = await engine.add_goal(
                    name=text[:60],
                    objective=text,
                    source="executive_authority",
                    status="in_progress",
                )
                created[text] = record["id"]
            quarantined = engine.quarantine_transient_foreground_goals()
            assert len(quarantined) >= 3
            for text in (CHAT_TURN, NETHACK_RENDER, EMBODIED_CONTRACT):
                goal = engine.get_goal(created[text])
                assert goal is not None
                assert goal["status"] == "abandoned", text
            healthy = engine.get_goal(created[VALID_GOALS[0]])
            assert healthy is not None
            assert healthy["status"] != "abandoned"
        finally:
            engine.close()


class TestSelfModelBeliefSanitation:
    def test_restored_chat_turn_projection_is_scrubbed(self):
        from core.self_model import SelfModel

        beliefs = {
            "executive_closure": {
                "selected_objective": CHAT_TURN,
                "background_commitment": NETHACK_RENDER,
                "attention_focus": "Phenomenal Surge",
            },
            "unrelated": "kept",
        }
        cleaned = SelfModel._sanitize_restored_beliefs(beliefs)
        closure = cleaned["executive_closure"]
        assert closure["selected_objective"] == ""
        assert closure["background_commitment"] == ""
        assert closure["attention_focus"] == "Phenomenal Surge"
        assert cleaned["unrelated"] == "kept"

    def test_valid_objective_survives_belief_restore(self):
        from core.self_model import SelfModel

        beliefs = {
            "executive_closure": {"selected_objective": VALID_GOALS[0]}
        }
        cleaned = SelfModel._sanitize_restored_beliefs(beliefs)
        assert cleaned["executive_closure"]["selected_objective"] == VALID_GOALS[0]

    def test_malformed_beliefs_degrade_to_empty(self):
        from core.self_model import SelfModel

        assert SelfModel._sanitize_restored_beliefs("not a dict") == {}
