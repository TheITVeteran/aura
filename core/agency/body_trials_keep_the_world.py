"""Try body controls without restarting their world or entering a text field.

Offline 2026-10-12 a reset was tried as movement, and a control said to confirm
input was tried among walking and jumping. Both change what the experiment
is acting in. They stay available to the interface, outside body trials.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from core.verify import invariant

_RESET = re.compile(r"\b(?:resets?|resetting|restarts?|restarted|restarting|retry|retries|retrying)\b|"
                    r"\b(?:plays?|playing|starts?|starting|begins?|beginning)\b.{0,40}\bagain\b|\bstarts? over\b", re.I)
_INPUT = re.compile(r"\b(?:confirms?|confirming|submits?|submitting|enters?|entering|types?|typing|sends?|sending)\s+"
                    r"(?:(?:the|your|an?)\s+)?"
                    r"(?:input|text|code|password|name|answer|form)\b", re.I)


def keys_for_body_trials(keys: Sequence[str], inputs: Mapping[str, str], *, body_known: bool) -> list[str]:
    """Keep unknown keys to try; leave controls explicitly bound to reset or text input outside this experiment."""
    if not body_known:
        return list(keys)
    return [key for key in keys if not (_RESET.search(inputs.get(key, "")) or _INPUT.search(inputs.get(key, "")))]


@invariant("agency.body_trials_preserve_their_interaction_phase", scope="agency",
           owner="core/agency/body_trials_keep_the_world.py", observational=False)
def _body_trial_phase_invariant() -> tuple:
    keys = ["a", "b", "c", "d"]
    inputs = {"a": "move", "b": "play this stage again", "c": "confirms input"}
    assert keys_for_body_trials(keys, inputs, body_known=True) == ["a", "d"]
    assert keys_for_body_trials(keys, inputs, body_known=False) == keys
    assert inputs["b"] == "play this stage again", "interface control was removed from the guide"
    return ()
