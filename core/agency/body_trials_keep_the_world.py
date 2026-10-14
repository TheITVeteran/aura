"""Try body controls without restarting their world or entering a text field.

Offline 2026-10-12 a reset was tried as movement, and a control said to confirm
input was tried among walking and jumping. Both change what the experiment
is acting in. They stay available to the interface, outside body trials.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

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


def keys_with_current_sources(guide: Any, kept: Sequence[str], named: Sequence[str], guided: Sequence[str]) -> list[str]:
    """Local instructions resolve controls before remembered names or unqualified prose."""
    from core.cognition.a_guide_to_a_place import PAGE, SCREEN, TOLD
    from core.cognition.what_its_code_says import CODE

    local = any(getattr(control, 'source', '') in (CODE, SCREEN, TOLD, PAGE)
                for control in (getattr(guide, 'controls', None) or {}).values())
    return list(dict.fromkeys(guided if local else [*kept, *named, *guided]))


@invariant('agency.current_controls_keep_their_sources', scope='agency',
           owner='core/agency/body_trials_keep_the_world.py', observational=False)
def _control_source_invariant() -> tuple:
    from core.cognition.a_guide_to_a_place import COUNSEL, SCREEN, Guide

    guide = Guide()
    guide.take_in(SCREEN, 'Press LEFT to move.')
    guide.take_in(COUNSEL, 'Press J to jump.')
    assert guide.keys_for_play() == ['left']
    assert keys_with_current_sources(guide, ['unverified'], ['another'], guide.keys_for_play()) == ['left']
    guide.controls.clear()
    assert keys_with_current_sources(guide, ['unverified'], ['another'], ['a']) == ['unverified', 'another', 'a']
    return ()


@invariant("agency.body_trials_preserve_their_interaction_phase", scope="agency",
           owner="core/agency/body_trials_keep_the_world.py", observational=False)
def _body_trial_phase_invariant() -> tuple:
    keys = ["a", "b", "c", "d"]
    inputs = {"a": "move", "b": "play this stage again", "c": "confirms input"}
    assert keys_for_body_trials(keys, inputs, body_known=True) == ["a", "d"]
    assert keys_for_body_trials(keys, inputs, body_known=False) == keys
    assert inputs["b"] == "play this stage again", "interface control was removed from the guide"
    return ()
