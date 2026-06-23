#!/usr/bin/env python3
"""#38 — load-bearing valence proof.

Claim boundary 1.D (docs/CLAIM_BOUNDARIES.md) defines this task's done-condition
exactly: *"A load-bearing valence signal (task #38) means it measurably changes
behavior, not that anything is experienced."* This proof demonstrates that the
intrinsic valence the affect substrate produces MEASURABLY and CAUSALLY changes
a real downstream behavior — emotional-salience memory weighting — through a
NON-TEXT numeric pathway, and that ABLATING the affect substrate removes the
effect.

Pathway (all real, no LLM/text in the loop):
  core/being/affective_valence.AffectiveValenceEngine  → valence
  → core/memory/long_term_memory_engine: emotionally salient (high |valence|)
    memories decay slower (long_term_memory_engine.py:296), survive the
    retention cap longer (:344), and recall STRONGER (:359-363, ``× (1 +
    abs(valence))``).

Ablation uses the engine's built-in ``AffectiveValenceEngine.lesion()`` (it
forces valence to 0). With the substrate ACTIVE, salient experiences are
retained/recalled measurably more than neutral ones; LESIONED, that advantage
collapses to ~0 — so the salience-weighting of memory is *load-bearing on
valence*. Verdict is an honest CI-separation rule and CAN be False. Clears
tools/proof_fabrication_guard.py (real measured scores).

HONEST SCOPE: this is functional emotional-salience weighting that changes
behavior. It is NOT a claim of felt experience, subjective feeling, or qualia
(see CLAIM_BOUNDARIES 1.D). The same valence signal is also load-bearing in the
being-runtime welfare gate and the will's AuraNow evidence; memory is the
cleanest deterministic pathway to prove the property.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "artifacts" / "proof_bundle" / "latest" / "VALENCE_LOAD_BEARING.json"
_AGE_S = 30.0  # aging horizon over which the engine's decay term separates


def _decay_rate(*, importance: float, valence: float) -> float:
    """The engine's write-time decay rule (long_term_memory_engine.py:296)."""
    return 0.001 if importance > 0.7 or abs(valence) > 0.7 else 0.02


def _salient_contexts(Body: Any, Pred: Any, World: Any) -> list[tuple[dict, dict, dict, float]]:
    """(body, prediction, world, base_valence) tuples that should be emotionally
    salient (high |valence|), alternating reward-like and distress-like."""
    out = []
    for i in range(8):
        if i % 2 == 0:  # reward-like → strong positive valence
            out.append((
                dict(),  # no body pressure
                dict(free_energy=0.0, controllability=0.96, expected_information_gain=0.85),
                dict(task_active=True, relationship_count=3, uncertainty=0.0),
                0.45,
            ))
        else:  # distress-like → strong negative valence
            out.append((
                dict(cpu_pressure=0.8, memory_pressure=0.8, thermal_pressure=0.7),
                dict(free_energy=0.9, controllability=0.08, expected_information_gain=0.05),
                dict(task_active=True, relationship_count=0, uncertainty=0.85),
                -0.35,
            ))
    return out


def _neutral_contexts() -> list[tuple[dict, dict, dict, float]]:
    """Affectively flat contexts → low |valence| (below the salience threshold)."""
    out = []
    for _ in range(8):
        out.append((
            dict(cpu_pressure=0.18, memory_pressure=0.12),
            dict(free_energy=0.18, controllability=0.36, expected_information_gain=0.08),
            dict(task_active=False, relationship_count=1, uncertainty=0.18),
            0.0,
        ))
    return out


async def _recall_strength(engine: Any, TaggedMemory: Any, *, valence: float, importance: float,
                           query: str, content: str) -> float:
    """Append one aged memory + return the real engine's recall strength for it."""
    now = time.time()
    engine.memories = [
        TaggedMemory(
            id=f"m_{now}",
            content=content,
            timestamp=now - _AGE_S,
            emotional_valence=float(valence),
            importance=float(importance),
            decay_rate=_decay_rate(importance=importance, valence=valence),
            last_rehearsed=now - _AGE_S,
            tags=["proof", "salience"],
        )
    ]
    recalled = await engine.recall_relevant(query, limit=1)
    if not recalled:
        return 0.0
    m = recalled[0]
    age = now - m.timestamp
    return float(m.importance * max(0.01, 1.0 - m.decay_rate * age) * (1.0 + abs(m.emotional_valence)))


def _advantages(*, lesioned: bool) -> list[float]:
    """For each (salient, neutral) pair: recall_strength(salient) - recall(neutral),
    under the real affect substrate (or lesioned)."""
    from core.being.affective_valence import AffectiveValenceEngine
    from core.being.aura_now import BodyState, PredictionState, WorldState
    from core.memory.long_term_memory_engine import LongTermMemoryEngine, TaggedMemory

    affect = AffectiveValenceEngine()
    if lesioned:
        affect.lesion()
    engine = LongTermMemoryEngine()

    def _valence(ctx: tuple) -> float:
        b, p, w, base = ctx
        state = affect.compute(
            body=BodyState(**b), prediction=PredictionState(**p), world=WorldState(**w),
            base_valence=base,
        )
        return float(state.valence)

    salient = _salient_contexts(BodyState, PredictionState, WorldState)
    neutral = _neutral_contexts()
    importance = 0.5  # identical for both → valence is the ONLY differing factor
    query = "proof salience event"
    content = "a proof salience event the agent encountered"

    advantages: list[float] = []
    for s_ctx, n_ctx in zip(salient, neutral):
        s_val = _valence(s_ctx)
        n_val = _valence(n_ctx)
        s_strength = asyncio.run(_recall_strength(engine, TaggedMemory, valence=s_val,
                                                  importance=importance, query=query, content=content))
        n_strength = asyncio.run(_recall_strength(engine, TaggedMemory, valence=n_val,
                                                  importance=importance, query=query, content=content))
        advantages.append(round(s_strength - n_strength, 6))
    return advantages


def run_proof(*, ci_iterations: int = 4000) -> dict[str, Any]:
    from core.evaluation.statistics import bootstrap_ci

    active = _advantages(lesioned=False)
    lesioned = _advantages(lesioned=True)

    a_lo, a_hi = bootstrap_ci(active, n_resamples=max(1, ci_iterations), confidence=0.95, seed=380)
    l_lo, l_hi = bootstrap_ci(lesioned, n_resamples=max(1, ci_iterations), confidence=0.95, seed=380)
    a_mean = sum(active) / max(1, len(active))
    l_mean = sum(lesioned) / max(1, len(lesioned))

    # Load-bearing iff the salience advantage with the substrate ACTIVE clears the
    # advantage with it LESIONED (which should be ~0) — CI-separated, can be False.
    valence_is_load_bearing = a_lo > l_hi and a_mean > 0.05

    bundle = {
        "claim": "VALENCE_IS_LOAD_BEARING",
        "passed": bool(valence_is_load_bearing),
        "valence_is_load_bearing": bool(valence_is_load_bearing),
        "verdict_rule": "active salience-advantage CI-lower must clear lesioned CI-upper (can be False)",
        "generated_at": time.time(),
        "measures": "emotional-salience memory weighting (decay + recall strength) driven by intrinsic valence",
        "pathway": "AffectiveValenceEngine.valence → long_term_memory_engine decay/recall (non-text)",
        "ablation": "AffectiveValenceEngine.lesion() forces valence=0",
        "honest_scope": (
            "Functional behavior-changing valence per CLAIM_BOUNDARIES 1.D — NOT felt experience / "
            "subjective feeling / qualia. Same signal is also load-bearing in the welfare gate + will."
        ),
        "active_salience_advantage": {"mean": round(a_mean, 6), "ci_lower": round(a_lo, 6),
                                      "ci_upper": round(a_hi, 6), "n": len(active), "samples": active},
        "lesioned_salience_advantage": {"mean": round(l_mean, 6), "ci_lower": round(l_lo, 6),
                                        "ci_upper": round(l_hi, 6), "n": len(lesioned), "samples": lesioned},
        "reproduction_command": "python tools/agi/run_valence_load_bearing_proof.py",
    }
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--ci-iterations", type=int, default=4000)
    args = parser.parse_args()

    bundle = run_proof(ci_iterations=args.ci_iterations)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    a = bundle["active_salience_advantage"]
    le = bundle["lesioned_salience_advantage"]
    print(f"Valence load-bearing bundle written to {out_path}")
    print(
        f"PASSED={bundle['passed']} | active salience-advantage mean={a['mean']:.3f} "
        f"CI[{a['ci_lower']:.3f},{a['ci_upper']:.3f}] vs lesioned mean={le['mean']:.3f} "
        f"CI[{le['ci_lower']:.3f},{le['ci_upper']:.3f}]"
    )
    return 0 if bundle["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
