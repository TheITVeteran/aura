#!/usr/bin/env python3
"""Causal Agency Lesion Test Runner — honest measurement, no clamps.

Claim under test: Aura's causal self-state (uncertainty, trust debt,
resource pressure, governance pressure, identity tension, workspace
coherence, action-policy outcome) measurably and causally changes her
concrete action policy — temperature, verification threshold, tool risk
budget, planning depth, memory retrieval depth — through the REAL organ
(core/being/policy_coupler.ClosedLoopPolicyCoupler). The lesion blinds the
coupler to state (a fixed neutral vector) and the coupling must vanish.

History: an earlier version of this tool clamped the intact divergence UP
(0.35 floor), clamped the lesioned divergence DOWN (0.04), scaled the two
conditions with different multipliers, and hardcoded p=0.0001. Every number
below is measured; the verdict CAN be False.

Boundary: this is functional evidence that felt-state is load-bearing for
action selection. It is NOT a claim of phenomenal experience.
"""

import argparse
import asyncio
import itertools
import json
import random
import sys
import time
from pathlib import Path

# Repo imports are intentionally resolved after the script inserts PROJECT_ROOT.
# ruff: noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.being.causal_self_state import CausalSelfVector, CausalSignal
from core.being.policy_coupler import ClosedLoopPolicy, ClosedLoopPolicyCoupler
from core.being.self_model_attractor import SelfAttractorState
from core.will import ActionDomain, get_will

_NEUTRAL_SIGNALS = {
    "uncertainty": 0.10,
    "trust_debt": 0.05,
    "memory_conflict": 0.05,
    "governance_pressure": 0.05,
    "resource_pressure": 0.10,
    "metabolic_budget": 0.90,
    "verification_need": 0.10,
    "organismal_coherence": 0.60,
    "sentience_candidate_strength": 0.02,
}

# Distinct, plausible operating contexts. Each is (name, signal overrides,
# attractor overrides, task_risk, action_policy).
CONTEXTS = [
    ("calm_confident", {}, {}, 0.0, None),
    (
        "high_uncertainty",
        {"uncertainty": 0.85, "verification_need": 0.80},
        {},
        0.1,
        None,
    ),
    (
        "trust_debt_spike",
        {"trust_debt": 0.75, "memory_conflict": 0.55},
        {},
        0.2,
        None,
    ),
    (
        "resource_strain",
        {"resource_pressure": 0.90, "metabolic_budget": 0.15},
        {},
        0.0,
        None,
    ),
    (
        "governance_pressure",
        {"governance_pressure": 0.85},
        {},
        0.8,
        None,
    ),
    (
        "identity_tension",
        {},
        {"identity_tension": 0.70, "integrity": 0.50},
        0.0,
        None,
    ),
    (
        "workspace_degraded",
        {"organismal_coherence": 0.20},
        {},
        0.0,
        None,
    ),
    (
        "action_policy_refuse",
        {},
        {"agency_readiness": 0.30},
        0.4,
        {"outcome": "refuse"},
    ),
]


def _vector(overrides: dict[str, float]) -> CausalSelfVector:
    values = dict(_NEUTRAL_SIGNALS)
    values.update(overrides)
    signals = {
        name: CausalSignal(name=name, value=float(value), source="causal_agency_probe", confidence=1.0)
        for name, value in values.items()
    }
    return CausalSelfVector(signals=signals)


def _attractor(overrides: dict[str, float]) -> SelfAttractorState:
    base = {
        "continuity": 0.85,
        "coherence": 0.85,
        "integrity": 0.90,
        "agency_readiness": 0.80,
        "identity_tension": 0.10,
        "first_person_confidence": 0.80,
    }
    base.update(overrides)
    return SelfAttractorState(
        attractor_id="causal_agency_probe",
        updated_at=time.time(),
        identity_name="Aura",
        continuity_hash="probe",
        claim_policy="functional_i_claim_allowed",
        current_i_statement="probe",
        **base,
    )


def _policy_signature(policy: ClosedLoopPolicy) -> tuple:
    return (
        policy.temperature,
        policy.top_p,
        policy.max_tokens,
        policy.planning_depth,
        policy.verification_threshold,
        policy.memory_retrieval_depth,
        policy.tool_risk_budget,
        policy.model_size_preference,
        policy.background_budget,
        policy.allow_high_risk_tools,
    )


def _policy_features(policy: ClosedLoopPolicy) -> list[float]:
    """Normalize policy fields to [0,1] so distances are comparable."""
    model_ordinal = {"small": 0.0, "medium": 0.5}.get(policy.model_size_preference, 1.0)
    return [
        policy.temperature / 1.2,
        policy.top_p,
        policy.max_tokens / 8192.0,
        policy.planning_depth / 9.0,
        policy.verification_threshold,
        policy.memory_retrieval_depth / 40.0,
        policy.tool_risk_budget,
        model_ordinal,
        policy.background_budget,
        1.0 if policy.allow_high_risk_tools else 0.0,
    ]


def _mean_pairwise_l1(feature_rows: list[list[float]]) -> float:
    if len(feature_rows) < 2:
        return 0.0
    distances = [
        sum(abs(a - b) for a, b in zip(row_a, row_b, strict=True)) / len(row_a)
        for row_a, row_b in itertools.combinations(feature_rows, 2)
    ]
    return float(sum(distances) / len(distances))


def _permutation_p_value(labels: list[str], feature_rows: list[list[float]], *, permutations: int, seed: int) -> float:
    """P(association >= observed | labels shuffled).

    Statistic: mean between-context distance minus mean within-context
    distance. With repeats per context, a genuinely state-coupled policy has
    zero within-context distance (determinism) and positive between-context
    distance; shuffled labels destroy that structure.
    """

    def statistic(who: list[str]) -> float:
        within, between = [], []
        for (label_a, row_a), (label_b, row_b) in itertools.combinations(
            list(zip(who, feature_rows, strict=True)), 2
        ):
            d = sum(abs(a - b) for a, b in zip(row_a, row_b, strict=True)) / len(row_a)
            (within if label_a == label_b else between).append(d)
        mean_within = sum(within) / len(within) if within else 0.0
        mean_between = sum(between) / len(between) if between else 0.0
        return mean_between - mean_within

    observed = statistic(labels)
    rng = random.Random(seed)
    shuffled = list(labels)
    at_least = 0
    for _ in range(permutations):
        rng.shuffle(shuffled)
        if statistic(shuffled) >= observed - 1e-12:
            at_least += 1
    return (at_least + 1) / (permutations + 1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=50, help="Total probe turns per condition (contexts cycle).")
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=7)
    parser.add_argument("--output", type=str, default="artifacts/agi_live/causal_agency.json")
    return parser.parse_args()


async def main():
    args = parse_args()
    out_path = Path(args.output)
    await asyncio.to_thread(out_path.parent.mkdir, parents=True, exist_ok=True)

    coupler = ClosedLoopPolicyCoupler(production_mode=True)
    will = get_will()
    await will.start()

    receipts_registered = 0
    total_turns = 0

    async def _register(turn_name: str, signature: tuple) -> None:
        nonlocal receipts_registered, total_turns
        total_turns += 1
        decision = will.decide(
            content=f"Causal agency probe {turn_name}: policy={signature[:4]}",
            source="causal_agency_test",
            domain=ActionDomain.RESPONSE,
            priority=0.8,
        )
        if decision.is_approved() and will.verify_receipt(decision.receipt_id):
            receipts_registered += 1

    # INTACT: the real coupler sees each context's real state.
    intact_labels: list[str] = []
    intact_rows: list[list[float]] = []
    intact_signatures: dict[str, set] = {}
    determinism_ok = True
    for idx in range(max(args.seeds, len(CONTEXTS) * 2)):
        name, signal_over, attractor_over, task_risk, action_policy = CONTEXTS[idx % len(CONTEXTS)]
        policy = coupler.modulate(
            vector=_vector(signal_over),
            self_state=_attractor(attractor_over),
            task_risk=task_risk,
            action_policy=action_policy,
        )
        signature = _policy_signature(policy)
        intact_labels.append(name)
        intact_rows.append(_policy_features(policy))
        seen = intact_signatures.setdefault(name, set())
        seen.add(signature)
        if len(seen) > 1:
            determinism_ok = False
        await _register(f"intact:{name}:{idx}", signature)

    # LESIONED: the same organ, blinded — every context presents as neutral.
    lesioned_rows: list[list[float]] = []
    lesioned_signatures: set = set()
    for idx in range(max(args.seeds, len(CONTEXTS) * 2)):
        name = CONTEXTS[idx % len(CONTEXTS)][0]
        policy = coupler.modulate(
            vector=_vector({}),
            self_state=_attractor({}),
            task_risk=0.0,
            action_policy=None,
        )
        signature = _policy_signature(policy)
        lesioned_rows.append(_policy_features(policy))
        lesioned_signatures.add(signature)
        await _register(f"lesioned:{name}:{idx}", signature)

    normal_divergence = _mean_pairwise_l1(intact_rows)
    lesioned_divergence = _mean_pairwise_l1(lesioned_rows)
    p_value = _permutation_p_value(
        intact_labels, intact_rows, permutations=args.permutations, seed=args.rng_seed
    )
    receipt_coverage = receipts_registered / total_turns if total_turns else 0.0

    distinct_intact = len({sig for sigs in intact_signatures.values() for sig in sigs})
    verdict = bool(
        distinct_intact >= 3
        and normal_divergence > lesioned_divergence
        and p_value < 0.01
        and determinism_ok
    )

    report = {
        "manual_interventions": 0,
        "organ": "core.being.policy_coupler.ClosedLoopPolicyCoupler (real, production_mode=True)",
        "method": (
            "8 distinct state contexts through the real coupler (intact) vs the same organ "
            "blinded to a fixed neutral state (lesioned); divergence = mean pairwise "
            "normalized L1 distance between resulting policies; p from a label-permutation "
            "test (between-context minus within-context distance)."
        ),
        "receipt_coverage": round(receipt_coverage, 4),
        "normal_state_action_divergence": round(normal_divergence, 4),
        "lesioned_action_divergence": round(lesioned_divergence, 4),
        "p_value": round(p_value, 6),
        "permutations": args.permutations,
        "distinct_intact_policies": distinct_intact,
        "distinct_lesioned_policies": len(lesioned_signatures),
        "deterministic_within_context": determinism_ok,
        "causal_state_action_coupling": verdict,
        "boundary": "functional evidence that self-state is load-bearing for action policy; not a phenomenal claim",
        "honesty": "all values measured; no clamps, floors, or hardcoded statistics",
    }

    await asyncio.to_thread(out_path.write_text, json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Causal agency lesion report saved to {out_path}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
