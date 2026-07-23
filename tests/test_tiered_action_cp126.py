"""CP126 contract tests for the tiered action controller.

Each test names the finding it pins. The theme is the same one throughout:
a tier verdict that does not withhold anything is not a control system.
"""
from __future__ import annotations

import math

import pytest

from core.advanced_cognition.schemas import ActionCandidate, Observation
from core.advanced_cognition.tiered_action import (
    ActionTier,
    TieredActionController,
)


def _obs(**state) -> Observation:
    return Observation(domain="test", state=dict(state) or {"k": 1})


def _tactical_proof() -> dict:
    return {"search_ranking": [{"action_id": "a"}], "prediction": {"risk": 0.1}}


def _deliberative_proof() -> dict:
    return {
        **_tactical_proof(),
        "alternatives_considered": ["a", "b"],
        "approval": {"by": "test"},
    }


def _reflective_proof() -> dict:
    return {
        **_deliberative_proof(),
        "proof_obligations": ["tests_pass"],
        "postmortem_owner": "architect",
    }


# --- 1b5e9bae: selection must be a decision, not candidates[0] ---------------


def test_selection_is_not_caller_ordering():
    controller = TieredActionController()
    costly = ActionCandidate("costly", "act", expected_cost=0.2, tags=("safe",))
    cheap = ActionCandidate("cheap", "act", expected_cost=0.01, tags=("safe",))

    first = controller.choose_tier(_obs(), [costly, cheap], risk=0.1, uncertainty=0.1)
    second = controller.choose_tier(_obs(), [cheap, costly], risk=0.1, uncertainty=0.1)

    assert first.selected["action_id"] == "cheap"
    assert second.selected["action_id"] == "cheap"


def test_irreversible_action_is_ineligible_at_cheap_tiers():
    controller = TieredActionController()
    risky = ActionCandidate("wipe", "act", reversible=False, tags=("safe",))
    safe = ActionCandidate("read", "act", reversible=True, tags=("safe",))

    decision = controller.choose_tier(_obs(), [risky, safe], risk=0.1, uncertainty=0.1)

    assert decision.selected["action_id"] == "read"
    assert any(row["action"]["action_id"] == "wipe" for row in decision.ineligible)


def test_elevated_authority_is_not_reflex_eligible():
    controller = TieredActionController()
    privileged = ActionCandidate("sudo", "act", authority_tier=4)

    decision = controller.choose_tier(_obs(), [privileged], risk=0.1, uncertainty=0.1)

    assert decision.selected is None
    assert decision.abstained
    assert decision.blocked_reason == "no_eligible_candidates"


# --- 9bffa715: non-finite risk must not fall through to reflex --------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "high", None])
def test_unusable_risk_escalates_instead_of_reaching_reflex(bad):
    controller = TieredActionController()
    action = ActionCandidate("a", "act", tags=("safe",))

    decision = controller.choose_tier(_obs(), [action], risk=bad, uncertainty=0.1)

    assert decision.tier >= ActionTier.DELIBERATIVE
    assert decision.input_faults
    assert decision.inputs["risk"] == 1.0
    if isinstance(bad, float) and math.isnan(bad):
        assert "non-finite" in decision.input_faults[0]


def test_unusable_uncertainty_escalates():
    controller = TieredActionController()
    decision = TieredActionController().choose_tier(
        _obs(), [ActionCandidate("a", "act")], risk=0.0, uncertainty=float("nan")
    )
    assert decision.tier >= ActionTier.DELIBERATIVE
    assert controller.LATENCY_BUDGET_MS[decision.tier] >= 10_000


def test_out_of_range_inputs_are_clamped_and_reported():
    decision = TieredActionController().choose_tier(
        _obs(), [ActionCandidate("a", "act", tags=("safe",))], risk=-5.0, uncertainty=0.1
    )
    assert decision.inputs["risk"] == 0.0
    assert any("out of range" in fault for fault in decision.input_faults)


# --- 1c45593a: the System 2 requirement must actually withhold the action ----


def test_system2_requirement_withholds_the_action():
    controller = TieredActionController()
    action = ActionCandidate("a", "act", tags=("safe",))

    decision = controller.choose_tier(_obs(), [action], risk=0.9, uncertainty=0.9)

    assert decision.requires_system2
    assert not decision.system2_satisfied
    assert decision.selected is None
    assert not decision.released
    assert decision.blocked_reason == "system2_proof_missing"
    assert "approval" in decision.missing_proof


def test_supplied_proof_releases_the_action():
    controller = TieredActionController()
    action = ActionCandidate("a", "act", tags=("safe",))

    decision = controller.choose_tier(
        _obs(),
        [action],
        risk=0.9,
        uncertainty=0.9,
        system2_evidence=_deliberative_proof(),
    )

    assert decision.system2_satisfied
    assert decision.released
    assert decision.selected["action_id"] == "a"


def test_empty_and_false_evidence_do_not_satisfy_the_requirement():
    controller = TieredActionController()
    action = ActionCandidate("a", "act", tags=("safe",))
    evidence = {**_deliberative_proof(), "approval": False, "search_ranking": []}

    decision = controller.choose_tier(
        _obs(), [action], risk=0.9, uncertainty=0.9, system2_evidence=evidence
    )

    assert not decision.system2_satisfied
    assert set(decision.missing_proof) == {"approval", "search_ranking"}


def test_self_modification_needs_obligations_and_a_postmortem_owner():
    controller = TieredActionController()
    action = ActionCandidate("patch", "patch_module", tags=("self_modify",), reversible=False)

    partial = controller.choose_tier(
        _obs(), [action], risk=0.2, uncertainty=0.2,
        self_modification=True, system2_evidence=_deliberative_proof(),
    )
    assert partial.tier is ActionTier.REFLECTIVE
    assert not partial.released
    assert set(partial.missing_proof) == {"proof_obligations", "postmortem_owner"}

    full = controller.choose_tier(
        _obs(), [action], risk=0.2, uncertainty=0.2,
        self_modification=True, system2_evidence=_reflective_proof(),
    )
    assert full.released


def test_self_modify_tag_alone_reaches_the_reflective_tier():
    decision = TieredActionController().choose_tier(
        _obs(), [ActionCandidate("p", "patch", tags=("self_modify",))],
        risk=0.0, uncertainty=0.0,
    )
    assert decision.tier is ActionTier.REFLECTIVE


def test_execution_helper_refuses_an_unreleased_decision():
    controller = TieredActionController()
    decision = controller.choose_tier(
        _obs(), [ActionCandidate("a", "act", tags=("safe",))], risk=0.9, uncertainty=0.9
    )
    calls = []

    outcome = controller.execute_within_budget(decision, lambda action: calls.append(action))

    assert calls == []
    assert outcome["executed"] is False
    assert outcome["error"] == "system2_proof_missing"


def test_execution_helper_runs_and_measures_a_released_decision():
    controller = TieredActionController()
    decision = controller.choose_tier(
        _obs(), [ActionCandidate("a", "act", tags=("safe",))],
        risk=0.9, uncertainty=0.9, system2_evidence=_deliberative_proof(),
    )

    outcome = controller.execute_within_budget(decision, lambda action: action["action_id"])

    assert outcome["ok"] and outcome["executed"]
    assert outcome["result"] == "a"
    assert outcome["budget_ms"] == controller.LATENCY_BUDGET_MS[decision.tier]
    assert outcome["elapsed_ms"] >= 0.0


# --- 838b9e38: budgets must be enforced, not merely reported ----------------


def test_late_deliberation_evidence_is_refused():
    controller = TieredActionController()
    seen = {}

    def slow_system2(context):
        seen.update(context)
        # Blow through the tactical budget deterministically.
        import time as _time

        _time.sleep((context["latency_budget_ms"] + 40) / 1000.0)
        return _deliberative_proof()

    decision = controller.choose_tier(
        _obs(),
        [ActionCandidate("a", "act", tags=("safe",))],
        risk=0.5,
        uncertainty=0.1,
        system2=slow_system2,
    )

    assert decision.tier is ActionTier.TACTICAL
    assert seen["deadline_monotonic"] > 0
    assert seen["required_proof"]
    assert not decision.system2_satisfied
    assert decision.selected is None
    assert decision.system2_latency_ms >= decision.latency_budget_ms


def test_prompt_deliberation_evidence_is_accepted():
    controller = TieredActionController()

    decision = controller.choose_tier(
        _obs(),
        [ActionCandidate("a", "act", tags=("safe",))],
        risk=0.5,
        uncertainty=0.1,
        system2=lambda context: _tactical_proof(),
    )

    assert decision.tier is ActionTier.TACTICAL
    assert decision.released


def test_failing_deliberation_callback_does_not_release():
    def boom(context):
        raise RuntimeError("deliberation unavailable")

    decision = TieredActionController().choose_tier(
        _obs(), [ActionCandidate("a", "act", tags=("safe",))],
        risk=0.5, uncertainty=0.1, system2=boom,
    )
    assert not decision.released
    assert decision.selected is None


def test_overrunning_execution_is_reported():
    controller = TieredActionController()
    decision = controller.choose_tier(
        _obs(), [ActionCandidate("a", "act", tags=("safe",))], risk=0.1, uncertainty=0.1
    )
    assert decision.latency_budget_ms <= 50

    def slow(action):
        import time as _time

        _time.sleep((decision.latency_budget_ms + 30) / 1000.0)
        return "done"

    outcome = controller.execute_within_budget(decision, slow)
    assert outcome["overran"] is True
    assert outcome["elapsed_ms"] > outcome["budget_ms"]


# --- 5abe6e13: an empty candidate list is an abstention ---------------------


def test_no_candidates_is_an_explicit_abstention():
    decision = TieredActionController().choose_tier(
        _obs(), [], risk=0.1, uncertainty=0.1
    )

    assert decision.abstained
    assert decision.selected is None
    assert decision.blocked_reason == "no_candidates"
    assert "abstain" in decision.reason
    assert not decision.released
    assert decision.to_dict()["released"] is False


def test_malformed_candidates_are_reported_not_swallowed():
    decision = TieredActionController().choose_tier(
        _obs(), [{"not_a_field": 1}], risk=0.1, uncertainty=0.1
    )
    assert decision.abstained
    assert decision.ineligible and "malformed" in decision.ineligible[0]["reason"]


# --- 846edf27: the decision id must bind everything that drove it ----------


def test_decision_id_separates_novelty_and_self_modification():
    controller = TieredActionController()
    action = ActionCandidate("a", "act", tags=("safe",))
    base = controller.choose_tier(_obs(), [action], risk=0.1, uncertainty=0.1, novelty=0.0)
    novel = controller.choose_tier(_obs(), [action], risk=0.1, uncertainty=0.1, novelty=0.9)
    selfmod = controller.choose_tier(
        _obs(), [action], risk=0.1, uncertainty=0.1, self_modification=True
    )

    assert len({base.decision_id, novel.decision_id, selfmod.decision_id}) == 3


def test_decision_id_binds_action_details_beyond_the_id():
    controller = TieredActionController()
    cheap = ActionCandidate("a", "act", expected_cost=0.01, tags=("safe",))
    same_id_costlier = ActionCandidate("a", "act", expected_cost=0.9, tags=("safe",))

    one = controller.choose_tier(_obs(), [cheap], risk=0.1, uncertainty=0.1)
    two = controller.choose_tier(_obs(), [same_id_costlier], risk=0.1, uncertainty=0.1)

    assert one.decision_id != two.decision_id


def test_decision_id_is_stable_for_identical_inputs():
    controller = TieredActionController()
    action = ActionCandidate("a", "act", tags=("safe",))
    obs = _obs()
    assert (
        controller.choose_tier(obs, [action], risk=0.1, uncertainty=0.1).decision_id
        == controller.choose_tier(obs, [action], risk=0.1, uncertainty=0.1).decision_id
    )


# --- integration: the gate honours the tier --------------------------------


def test_gate_withholds_self_modification_without_obligations(tmp_path):
    from core.advanced_cognition.integration import AdvancedCognitionRuntime

    runtime = AdvancedCognitionRuntime(state_dir=tmp_path / "adv")
    payload = runtime.observe_state("repo", {"file": "core/x.py"}, confidence=0.8)
    gate = runtime.pre_action_gate(
        payload["observation"],
        [{"action_id": "patch", "kind": "patch_module", "tags": ("self_modify",),
          "reversible": False, "authority_tier": 4}],
    )

    assert gate["tier"]["tier_name"] == "reflective"
    assert gate["tier"]["system2_satisfied"] is False
    assert gate["allowed"] is False
    assert gate["selected"] is None
