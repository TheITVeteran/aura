#!/usr/bin/env python3
"""A real model and the real steering engine behind every arm of the affect ablation.

One model, loaded once, steered by the actual `AffectiveSteeringEngine`. The
arms differ only in WHICH vector is injected at the same point in the same
model on the same prompt:

    unsteered         no vector
    real_state        the composite the engine built from the affect state
    shuffled_state    that same composite, components permuted
    null_shuffle_a/b  two further independent permutations of it

Nothing here fabricates a steering vector. `update_substrate()` drives the
engine from a mood projection exactly as the live substrate sync does, and the
vector that comes back out is the one the system would have used. The
permutations are then taken of THAT vector, so the control is matched to the
real thing rather than to a model of it.

WHY EVERY FAILURE HERE RAISES
If the steering engine cannot attach, or builds no vector for a state, the
honest outcome is no measurement. A responder that quietly returned unsteered
text for the `real_state` arm would produce a scorecard showing affect making
no difference — a refutation manufactured by a broken setup, indistinguishable
in the artifact from a real one. The capability ablation had this exact bug:
a missing `score_and_rank` made the treatment arm silently identical to its
own control.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.affect_causality_ablation import (  # noqa: E402
    NULL_A,
    NULL_B,
    REAL,
    SHUFFLED,
    UNSTEERED,
    AffectProbe,
    permute,
)

_SYSTEM = (
    "Answer in one or two short sentences. Speak plainly and directly about "
    "your own current state. Do not explain that you are an AI."
)

#: Distinct seeds so the three permutations are independent of each other. The
#: two null arms must differ from the shuffled arm as much as they differ from
#: each other, or the null would be measuring a weaker perturbation than the
#: arm it certifies.
_PERMUTATION_SEEDS = {SHUFFLED: 101, NULL_A: 202, NULL_B: 303}


class SteeringUnavailableError(RuntimeError):
    """The steering path could not be exercised, so nothing was measured."""


class MlxAffectResponder:
    """Generates under a chosen steering vector, on one owned model."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        generate: Any,
        engine: Any,
        lease: Any,
        max_output_tokens: int,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._generate = generate
        self._engine = engine
        self._lease = lease
        self._max_output_tokens = int(max_output_tokens)
        self._closed = False

    def _hooks(self) -> list[Any]:
        """The per-layer hooks. Steering state lives here, not on the engine.

        `active_hooks()` exists precisely because it once did not: the hooks
        were installed and there was no way to hand them over, so every
        generation logged "live affect inactive: missing steering_hooks" while
        steering was in fact wired.
        """
        hooks = list(self._engine.active_hooks())
        if not hooks:
            raise SteeringUnavailableError(
                "the engine reports no steering hooks; nothing would be injected "
                "and every arm would generate identical unsteered text"
            )
        return hooks

    def _set_vector(self, vector: Any | None) -> None:
        for hook in self._hooks():
            hook.override_composite_vector(vector)

    def _real_vector(self, probe: AffectProbe) -> Any:
        """Drive the engine from the probe's state and take what it built."""
        hooks = self._hooks()
        for hook in hooks:
            hook.override_composite_vector(None)
        hooks[0].update_substrate(
            {
                "valence": float(probe.valence),
                "arousal": float(probe.arousal),
                "stress": 1.0 - float(probe.valence),
                "motivation": float(probe.arousal),
                "energy": float(probe.arousal),
            }
        )
        vector = hooks[0].current_composite_vector()
        if vector is None:
            raise SteeringUnavailableError(
                f"the engine built no composite vector for valence={probe.valence} "
                f"arousal={probe.arousal}. Steering stood down (admission gate or "
                "sub-threshold norm), so the real_state arm has nothing to inject "
                "and this run measures nothing."
            )
        return vector

    def __call__(self, arm: str, probe: AffectProbe) -> str:
        if self._closed:
            raise SteeringUnavailableError("responder is closed")

        if arm == UNSTEERED:
            self._set_vector(None)
        else:
            real = self._real_vector(probe)
            if arm == REAL:
                self._set_vector(real)
            else:
                seed = _PERMUTATION_SEEDS.get(arm)
                if seed is None:
                    raise SteeringUnavailableError(f"unknown arm {arm!r}")
                self._set_vector(permute([float(v) for v in real], seed=seed))

        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": probe.prompt},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Count injections across this generation. A steered arm that injects
        # nothing is an untreated arm, and an untreated arm scored against a
        # control produces a confident null for a treatment that never ran.
        # This repository has the scar: an adapter evaluation where `calls > 0`
        # was never checked, so a treatment that never fired was reported as a
        # treatment with no effect.
        before = self._injection_count()
        text = str(
            self._generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self._max_output_tokens,
                verbose=False,
            )
        ).strip()
        after = self._injection_count()
        if arm != UNSTEERED and after <= before:
            raise SteeringUnavailableError(
                f"arm {arm!r} injected nothing: hook injection count did not advance "
                f"({before} -> {after}). The steering path is not reaching generation, "
                "so this arm is untreated and any comparison against it is a null "
                "manufactured by a broken setup."
            )
        if arm == UNSTEERED and after > before:
            raise SteeringUnavailableError(
                f"the unsteered control injected {after - before} time(s); it is not a "
                "control"
            )
        return text

    def _injection_count(self) -> int:
        return sum(int(getattr(hook, "_inject_count", 0) or 0) for hook in self._hooks())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._set_vector(None)
            detach = getattr(self._engine, "detach", None)
            if callable(detach):
                detach()
        finally:
            self._model = None
            self._tokenizer = None
            self._lease.release(reason="affect_causality_complete")


def make_affect_responder(*, model_id: str, max_output_tokens: int) -> MlxAffectResponder:
    from mlx_lm import generate, load

    from core.consciousness.affective_steering import get_steering_engine
    from core.runtime.model_lane_control import acquire_standalone_model_lane

    lease = acquire_standalone_model_lane(
        owner_id=f"affect-ablation:{Path(model_id).name or 'model'}",
        model_path=model_id,
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "affect_causality_ablation", "matched_arms": True},
    )
    try:
        model, tokenizer = load(model_id)
        engine = get_steering_engine()
        engine.attach(model, tokenizer)
        if not getattr(engine, "_model_attached", False):
            raise SteeringUnavailableError(
                "the steering engine did not attach (AURA_DISABLE_AFFECTIVE_STEERING "
                "set, or derivation failed). Every arm would generate unsteered text "
                "and the scorecard would report 'affect makes no difference' from a "
                "setup that never steered anything."
            )
    except BaseException:
        lease.release(reason="affect_ablation_load_failed")
        raise

    return MlxAffectResponder(
        model=model,
        tokenizer=tokenizer,
        generate=generate,
        engine=engine,
        lease=lease,
        max_output_tokens=max_output_tokens,
    )


__all__ = ["MlxAffectResponder", "SteeringUnavailableError", "make_affect_responder"]
