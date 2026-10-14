"""A turn whose answer is worked out, and so is not steered by how she feels.

Her state reaches the forward pass on a reply somebody is waiting for
(core/consciousness/fusion_certificate.py). On a turn with one right answer
that push is distortion. Both sides that decide how much steering a reply
gets read this: her mind when it asks (core/brain/cognitive_engine.py) and
the worker when nobody asked (core/brain/llm/mlx_worker_surface_quality.py).
"""
from __future__ import annotations

from typing import Any

__all__ = ["has_one_right_answer"]


def has_one_right_answer(user_message: Any) -> bool:
    """Whether this turn has one right answer that affect must not bend.

    Substrate steering exists to give her replies her own voice. It works by
    pushing an affect direction into the residual stream, and on a turn whose
    answer is a fact rather than a feeling that push is pure distortion.

    Measured live on the desktop surface 2026-07-26. "What is 17 minus 8, and
    then times 3?" — a question the bare 32B answers without effort — came back
    twice, from a healthy resident cortex:

        "Not too broad. Some skills serve me better than others.Did you pay
         attention in class? Hey, look at this - ätze! I got chocolate on my
         shirt."

        "Five thousand: So first you break it down. Mental operations can
         generate digits well outside the world's population. Imagination
         defaults to scalar scaling when pushed into math without boundaries."

    The second is the tell: not noise, but the answer being pulled toward
    whatever the affect vector encodes — talking ABOUT scaling and imagination
    instead of subtracting eight from seventeen.
    """
    try:
        from core.brain.executable_reasoning import (
            ReasoningObjectiveRole,
            classify_reasoning_objective_role,
        )
        from core.conversation.response_reliability import (
            asks_for_a_number,
            requires_reasoning_lane,
        )
    except ImportError:
        # not a failure: with no classifier, this turn does not get routed
        # to the reasoning lane, which is the default it would have had.
        return False
    try:
        if (
            classify_reasoning_objective_role(str(user_message or ""))
            is ReasoningObjectiveRole.EXPOSITORY
        ):
            return False
        return bool(
            asks_for_a_number(user_message) or requires_reasoning_lane(user_message)
        )
    except (RuntimeError, TypeError, ValueError):
        # not a failure: a classification that will not run leaves the turn
        # on the ordinary lane, which is the default.
        return False
