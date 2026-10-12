"""Motor measurements do not invoke controls named for resetting or entering text."""
import pytest

from core.agency.how_a_game_works import STEERED, Answer, HowItWorks

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("purpose", ["reset the world", "restart this stage", "play this level again", "start over",
                                      "confirms input", "submit the code", "enter an answer", "type your name"])
def test_explicit_interface_purposes_do_not_become_motor_trials(purpose):
    how = HowItWorks(me=Answer(STEERED, .7), inputs={"go": "move forward", "other": purpose})
    assert how.keys_for_body_trials(["go", "other", "unseen"]) == ["go", "unseen"]
    assert how.inputs["other"] == purpose


def test_unknown_body_roles_preserve_the_candidate_inputs():
    how = HowItWorks(inputs={"q": "reset the world"})
    assert how.keys_for_body_trials(["q", "x"]) == ["q", "x"]


def test_the_registered_body_trial_phase_invariant():
    from core.agency.body_trials_keep_the_world import _body_trial_phase_invariant

    assert _body_trial_phase_invariant() == ()


@pytest.mark.asyncio
async def test_the_real_play_constructor_receives_only_body_trial_inputs(monkeypatch):
    from core.agency import playing_as_it_happens as play
    from core.cognition.a_guide_to_a_place import THE_GUIDE, TOLD, Control, Guide

    guide = Guide(place="A new interactive floor")
    guide.controls = {key: Control(key, act, TOLD) for key, act in
                      (("up", "move forward"), ("x", "play level again"), ("r", "confirms input"))}
    constructed = []

    class ConstructedError(Exception):
        pass

    def construct(**kwargs):
        constructed.append(kwargs["keys"])
        raise ConstructedError

    async def look():
        raise AssertionError("constructor probe should not issue input or look")

    monkeypatch.setattr(play, "_Run", construct)
    token = THE_GUIDE.set(guide)
    try:
        with pytest.raises(ConstructedError):
            await play._play_as_it_happens(look, None, keys=["up", "x", "r", "unknown"], seconds=0)
    finally:
        THE_GUIDE.reset(token)
    assert constructed == [["up", "unknown"]]
