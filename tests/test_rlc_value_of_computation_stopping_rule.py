"""Contract: stopping is priced at the computation it forfeits."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.value_of_computation import (  # noqa: E402
    CognitiveStateSignal,
    OperationKind,
    ValueOfComputationPolicy,
    build_evidence_snapshot,
)

_EXECUTORS = (
    OperationKind.DECOMPOSE,
    OperationKind.BRANCH,
    OperationKind.CHECK_ASSUMPTION,
    OperationKind.COMPARE,
    OperationKind.ANSWER,
    OperationKind.ABSTAIN,
)


def _state(**over):
    base = dict(
        step_index=2, max_steps=8, neural_steps=2, min_neural_steps=2,
        active_branches=2, total_branches=2, residual=0.4, residual_delta=0.05,
        verifier_score=0.5, verifier_delta=0.0, disagreement=0.2,
        uncertainty=0.5, budget_remaining_fraction=0.9, has_memory=False,
        has_evidence=False, has_verifier=True, has_savepoint=True,
        can_execute=False, answer_verified=False, irreducible_uncertainty=False,
        pending_constraint_action=None, omitted_action_count=0,
    )
    base.update(over)
    # The signal requires prior actions to account for every prior step.
    base.setdefault(
        "previously_selected",
        tuple([OperationKind.DECOMPOSE] * base["step_index"]),
    )
    return CognitiveStateSignal(**base)


def test_reducible_uncertainty_does_not_buy_an_early_abstention():
    """Abstain used to win on value-per-cost purely because executing it is
    nearly free (0.01), while its bootstrap gain RISES with uncertainty --
    rewarding giving up exactly when the system should spend more. Every
    full-stack episode on both the 1.5B and the 32B halted at the floor depth
    of 2 for this reason."""
    controller = ValueOfComputationPolicy(build_evidence_snapshot(bucket="test|none|short|s:mid|u:mid", cells={}))
    decision = controller.choose(_state(), executors=_EXECUTORS)
    assert decision["action"] != OperationKind.ABSTAIN.value, decision["mode"]


def test_irreducible_uncertainty_still_stops():
    """The legitimate case keeps its explicit rule, above the scoring path."""
    controller = ValueOfComputationPolicy(build_evidence_snapshot(bucket="test|none|short|s:mid|u:mid", cells={}))
    decision = controller.choose(
        _state(irreducible_uncertainty=True), executors=_EXECUTORS
    )
    assert decision["action"] == OperationKind.ABSTAIN.value
    assert decision["mode"] == "irreducible_abstain"


def test_a_verified_answer_still_stops():
    controller = ValueOfComputationPolicy(build_evidence_snapshot(bucket="test|none|short|s:mid|u:mid", cells={}))
    decision = controller.choose(_state(answer_verified=True), executors=_EXECUTORS)
    assert decision["action"] == OperationKind.ANSWER.value
    assert decision["mode"] == "verified_stop"


def test_measured_worthlessness_still_permits_stopping():
    """The rule is "keep going while continuing is worth something", not
    "stopping is expensive". Pricing the forfeit into cost was wrong: gain/cost
    is not monotonic in cost once gain can be negative, so a bigger forfeit
    made stopping win HARDER exactly when continuing was measurably
    worthless -- which is the regime the learned stop head exists to serve."""
    policy = ValueOfComputationPolicy(build_evidence_snapshot(bucket="test|none|short|s:mid|u:mid", cells={}))
    early = policy.choose(_state(budget_remaining_fraction=0.95), executors=_EXECUTORS)
    assert early["action"] != OperationKind.ABSTAIN.value
    # With almost nothing left, the dedicated budget rule takes over instead.
    late = policy.choose(
        _state(step_index=7, budget_remaining_fraction=0.02), executors=_EXECUTORS
    )
    assert late["mode"] in {"budget_stop", "budget_abstain", "budget_last_action"}
