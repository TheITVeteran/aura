#!/usr/bin/env python3
"""#48 — substrate state as a DIRECT (non-text) causal constraint on cognition.

Claim boundary 4.B: Aura's grounding is *substantially text-mediated* and full
grounding is not claimed. This task REDUCES text-mediation by demonstrating that
substrate affect state shapes the actual generation DYNAMICS (sampling
temperature, token budget, repetition pressure) DIRECTLY — a numeric causal
constraint — rather than only being DESCRIBED to the model as a
``[Affect: valence=…]`` prompt string.

Two real pathways, no LLM/text in the loop:

  A. Affective circumplex (core/affect/affective_circumplex) → generation
     temperature / max_tokens / repetition_penalty. This is the LIVE CONSUMED
     pathway (core/brain/inference_gate.py applies it to non-strict user-facing
     generations). Driving the affect axes shifts the numeric sampling params.

  B. The affect engine's direct generation-control surface
     (core/being/affective_valence.generation_controls): across diverse affect
     contexts the controls VARY when the substrate is active, and are CONSTANT
     (flat) when the substrate is LESIONED — so the contribution is ablatable.

Verdict (honest, can be False): substrate state is a direct causal constraint iff
the affect-active control spread clearly exceeds the lesioned spread (~0) AND the
consumed circumplex params measurably respond to affect. Clears
tools/proof_fabrication_guard.py. Runs offline (no model).

HONEST SCOPE: this proves the affect→generation pathway is direct/non-text and
ablatable — it REDUCES, it does not ELIMINATE, text-mediation (valence is still
also surfaced in prompts elsewhere), and it carries NO phenomenal claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "artifacts" / "proof_bundle" / "latest" / "TEXT_MEDIATION.json"


def _contexts(Body: Any, Pred: Any, World: Any) -> list[tuple[dict, dict, dict, float]]:
    """Diverse affect-inducing contexts (body, prediction, world, base_valence)."""
    return [
        (dict(), dict(controllability=0.96, expected_information_gain=0.9, free_energy=0.0),
         dict(task_active=True, relationship_count=3, uncertainty=0.0), 0.45),   # curious/reward
        (dict(cpu_pressure=0.85, memory_pressure=0.8, thermal_pressure=0.7),
         dict(controllability=0.08, expected_information_gain=0.05, free_energy=0.9),
         dict(task_active=True, relationship_count=0, uncertainty=0.85), -0.35),  # distress
        (dict(), dict(controllability=0.6, expected_information_gain=0.2, free_energy=0.1),
         dict(task_active=False, relationship_count=2, uncertainty=0.1), 0.1),    # content
        (dict(cpu_pressure=0.1), dict(controllability=0.4, expected_information_gain=0.02, free_energy=0.15),
         dict(task_active=False, relationship_count=0, uncertainty=0.2), 0.0),    # bored/flat
        (dict(), dict(controllability=0.9, expected_information_gain=0.6, free_energy=0.05),
         dict(task_active=True, relationship_count=5, uncertainty=0.05), 0.3),    # engaged
        (dict(memory_pressure=0.6, latency_pressure=0.5),
         dict(controllability=0.2, expected_information_gain=0.1, free_energy=0.7),
         dict(task_active=True, relationship_count=1, uncertainty=0.7), -0.2),    # strained
    ]


def _pathway_b(*, lesioned: bool) -> dict[str, list[float]]:
    """Affect engine → generation_controls across contexts (active vs lesioned)."""
    from core.being.affective_valence import AffectiveValenceEngine, generation_controls
    from core.being.aura_now import BodyState, PredictionState, WorldState

    engine = AffectiveValenceEngine()
    if lesioned:
        engine.lesion()
    temp_deltas: list[float] = []
    token_scales: list[float] = []
    for b, p, w, base in _contexts(BodyState, PredictionState, WorldState):
        affect = engine.compute(body=BodyState(**b), prediction=PredictionState(**p),
                                world=WorldState(**w), base_valence=base)
        controls = generation_controls(affect)
        temp_deltas.append(float(controls["temperature_delta"]))
        token_scales.append(float(controls["max_tokens_scale"]))
    return {"temperature_delta": temp_deltas, "max_tokens_scale": token_scales}


def _pathway_a_circumplex() -> dict[str, Any]:
    """Live consumed pathway: affect axes → circumplex generation params."""
    from core.affect.affective_circumplex import AffectiveCircumplex

    def _params(valence_delta: float, arousal_delta: float) -> dict[str, float]:
        c = AffectiveCircumplex()
        c._cached = None
        if valence_delta or arousal_delta:
            c.apply_event(valence_delta, arousal_delta)
        c._cached = None
        p = c.get_llm_params()
        return {"temperature": float(p["temperature"]), "max_tokens": float(p["max_tokens"]),
                "repetition_penalty": float(p.get("repetition_penalty", p.get("rep_penalty", 0.0)))}

    positive = _params(0.45, 0.4)
    negative = _params(-0.45, -0.4)
    neutral = _params(0.0, 0.0)
    temp_spread = max(positive["temperature"], negative["temperature"], neutral["temperature"]) - \
        min(positive["temperature"], negative["temperature"], neutral["temperature"])
    token_spread = max(positive["max_tokens"], negative["max_tokens"], neutral["max_tokens"]) - \
        min(positive["max_tokens"], negative["max_tokens"], neutral["max_tokens"])
    return {
        "positive_affect": positive, "negative_affect": negative, "neutral_affect": neutral,
        "temperature_spread": round(temp_spread, 4), "max_tokens_spread": round(token_spread, 4),
        "responds_to_affect": bool(temp_spread > 0.02 or token_spread > 1.0),
    }


def run_proof() -> dict[str, Any]:
    active = _pathway_b(lesioned=False)
    lesioned = _pathway_b(lesioned=True)

    active_temp_spread = round(pstdev(active["temperature_delta"]) if len(active["temperature_delta"]) > 1 else 0.0, 6)
    lesioned_temp_spread = round(pstdev(lesioned["temperature_delta"]) if len(lesioned["temperature_delta"]) > 1 else 0.0, 6)
    active_token_spread = round(pstdev(active["max_tokens_scale"]) if len(active["max_tokens_scale"]) > 1 else 0.0, 6)
    lesioned_token_spread = round(pstdev(lesioned["max_tokens_scale"]) if len(lesioned["max_tokens_scale"]) > 1 else 0.0, 6)

    circumplex = _pathway_a_circumplex()

    # The affect substrate directly shapes generation controls iff: active controls
    # genuinely VARY across affect contexts while the LESIONED substrate flattens
    # them, AND the live consumed circumplex pathway responds to affect.
    substrate_is_direct_causal = bool(
        active_temp_spread > 0.02
        and lesioned_temp_spread < 1e-6
        and circumplex["responds_to_affect"]
    )

    bundle = {
        "claim": "SUBSTRATE_STATE_IS_DIRECT_CAUSAL_NOT_TEXT",
        "passed": substrate_is_direct_causal,
        "substrate_is_direct_causal": substrate_is_direct_causal,
        "verdict_rule": (
            "affect-active control spread > 0.02 AND lesioned spread ~0 AND the consumed "
            "circumplex pathway responds to affect (can be False)"
        ),
        "generated_at": time.time(),
        "measures": "affect substrate → generation sampling dynamics (temperature/tokens/rep), non-text",
        "honest_scope": (
            "Affect→generation is a DIRECT, non-text, ablatable causal pathway. This REDUCES "
            "text-mediation; it does NOT eliminate it (valence is also surfaced in prompts elsewhere; "
            "full grounding not claimed, CLAIM_BOUNDARIES 4.B). NO phenomenal claim."
        ),
        "pathway_b_affect_engine": {
            "active_temperature_delta_spread": active_temp_spread,
            "lesioned_temperature_delta_spread": lesioned_temp_spread,
            "active_max_tokens_scale_spread": active_token_spread,
            "lesioned_max_tokens_scale_spread": lesioned_token_spread,
            "active_temperature_deltas": active["temperature_delta"],
            "lesioned_temperature_deltas": lesioned["temperature_delta"],
            "ablation": "AffectiveValenceEngine.lesion() flattens the controls across all contexts",
        },
        "pathway_a_circumplex_consumed": circumplex,
        "reproduction_command": "python tools/agi/run_text_mediation_proof.py",
    }
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    bundle = run_proof()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    pb = bundle["pathway_b_affect_engine"]
    ca = bundle["pathway_a_circumplex_consumed"]
    print(f"Text-mediation (#48) bundle written to {out_path}")
    print(
        f"PASSED={bundle['passed']} | affect-engine control spread active="
        f"{pb['active_temperature_delta_spread']:.3f} vs lesioned={pb['lesioned_temperature_delta_spread']:.3f} "
        f"| circumplex temp_spread={ca['temperature_spread']:.3f} tokens_spread={ca['max_tokens_spread']:.0f} "
        f"(responds={ca['responds_to_affect']})"
    )
    return 0 if bundle["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
