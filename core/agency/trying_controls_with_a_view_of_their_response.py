"""Deliver finite control experiments while retaining time to observe them."""
from __future__ import annotations

import statistics
from typing import Any

from core.agency.a_response_worth_measuring_again import response_worth_repeating
from core.agency.controls_ready_for_the_motion_model import has_a_measured_response
from core.agency.countering_motion_as_it_appears import CounteringMotion
from core.agency.keeping_a_trial_observable import control_before_the_next_trial
from core.agency.the_key_against_its_going import FIRST, against_its_going
from core.agency.which_one_answers_to_her import PUSHED_FOR_S, REALLY_MOVES, RESPONSE_S, SETTLE_S
from core.perception.what_moves_in_the_picture import TOLD_FROM_STANDING_PX


async def try_controls(hands: Any, run: Any, hers: Any, at: float, moves: Any, *, hold: Any, seconds: float) -> None:
    """Repeat unknown controls when there is headroom, using measured control when waiting would lose it."""
    identified = hers is not None and moves is not None and hers.kind is not None and hers.thing(moves) is not None
    if identified and hers.carried and not hers.keys_known(run.keys):
        elapsed = at - run.held_since if run.held_since is not None else 0.
        remaining = max(0., seconds - elapsed) if run.held_trying and elapsed < seconds else seconds
        measured_delays = run.motion_responses if len(run.motion_responses) >= 3 else run.responses
        delay = max(RESPONSE_S, statistics.median(measured_delays) if measured_delays else RESPONSE_S)
        observation = control_before_the_next_trial(hers, moves, at, run.keys, remaining + SETTLE_S + delay)
        if observation is not None:
            await hold(hands, run, hers, observation.key, at, trying=False)
            run.tried_at = run.held_since
            return
    if at - run.tried_at < seconds:
        return
    if not hasattr(run, "counter_trial"):
        run.counter_trial = CounteringMotion()
    key = run.counter_trial.choose(run.keys, getattr(moves, "things", {}).values(), at,
                                   held=getattr(run, "held", ""), held_since=run.held_since,
                                   epoch=getattr(getattr(hers, "identification", None), "epoch", 0),
                                   settle_s=SETTLE_S, step_s=seconds, steps=FIRST,
                                   wait_s=4 * len(run.keys) * seconds,
                                   against=lambda keys, things: against_its_going(keys, things, 2.0 * TOLD_FROM_STANDING_PX),
                                   measured=(lambda keys, things: response_worth_repeating(keys, things, hers, moves.shape,
                                                 at, run.held_since, seconds=seconds, steps=FIRST,
                                                 effect_over=REALLY_MOVES / PUSHED_FOR_S))
                                             if hers is not None and hasattr(moves, "shape") else None)
    if key is None:
        if identified:
            if run.held_trying and run.held:
                key = ""
            else:
                unknown = [control for control in run.keys if not has_a_measured_response(hers, control)]
                if not unknown:
                    return
                key = unknown[0]
            run.trying += 1
        elif run.keys:
            turn = run.trying % (2 * len(run.keys))
            run.trying += 1
            key = run.keys[turn // 2] if turn % 2 == 0 else ""
        else:
            return
    await hold(hands, run, hers, key, at, trying=True)
    run.tried_at = run.held_since
