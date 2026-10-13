"""A parsed body control retains its source when a lower-priority reading arrives."""
from core.cognition.a_guide_to_a_place import PROGRAM, SCREEN, Guide
from core.cognition.reading_the_code import take_in_the_code
from core.cognition.what_its_code_says import CODE
from core.perception.how_a_program_plays import AKey, HowItPlays
from core.perception.reading_a_program import ProgramRead


def guide_with_a_parsed_body_control():
    guide = Guide()
    guide.take_in_program(ProgramRead(keys={'j': ''}, plays=HowItPlays(
        keys={'j': AKey('j', 'moves', 'Rover.moveBy(1, 0)', 'Rover.update')},
        hers={'class': 'Rover'})), at=1.)
    return guide


def test_a_model_reading_cannot_overwrite_a_parsed_body_control():
    guide = guide_with_a_parsed_body_control()
    take_in_the_code(guide, {'controls': [{'control': 'J', 'does': 'jump'}]})
    assert guide.controls['j'].source == CODE and guide.controls['j'].act == 'move'
    assert guide.keys_for_play() == ['j']


def test_lower_priority_text_cannot_change_a_parsed_controls_role():
    guide = guide_with_a_parsed_body_control()
    for source in (PROGRAM, SCREEN):
        guide.take_in(source, ['Press J to quit.'], at=2.)
        assert guide.controls['j'].source == CODE and guide.controls['j'].act == 'move'
        assert guide.keys_for_play() == ['j']


def test_unresolved_code_keeps_an_available_screen_instruction():
    guide = Guide()
    guide.take_in(SCREEN, ['Press J to jump.'], at=1.)
    guide.take_in_program(ProgramRead(keys={'j': ''}, plays=HowItPlays(
        keys={'j': AKey('j', 'does something', 'opaque()', 'an unknown function')})), at=2.)
    assert guide.controls['j'].source == SCREEN and guide.controls['j'].act == 'jump'


def test_the_source_priority_has_a_registered_measured_claim():
    from core.cognition.what_its_code_says import _parsed_control_priority_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _parsed_control_priority_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'parsed_controls_keep_their_source_priority')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
