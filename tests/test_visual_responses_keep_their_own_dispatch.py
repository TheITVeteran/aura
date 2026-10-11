"""A visual response measures one owned input, independently of goal success."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from core.agency.deliberate_action import (
    ActionOption,
    Deliberation,
    Expectation,
    _unmeasured_response_invariant,
    confirm,
)
from core.cognition.reading_the_rules import Frame, Rules
from core.runtime.skill_contract import PredicateState
from core.skills.screen_pursuit_on_a_page import it_was_answered
from core.skills.screen_step_evidence import (
    _response_dispatch_invariant,
    owned_response_state,
    settle_step,
)


def evidence(surface="editor"):
    before = {"text": "Filters", "clickable": ['click "Filters"'], "regions": [], "surface_id": surface,
              "capture_epoch": "a", "capture_at": 1., "bounds": [0, 0, 400, 300]}
    pending = {"observed_before": before, "dispatch_completed": 2., "dispatch_started_monotonic": 10.,
               "dispatch_completed_monotonic": 11., "dispatch_sequence_before": 4, "verification_started": 0.}
    after = {"text": "Filters", "surface_id": surface, "_capture_epoch": "b", "capture_at": 3.,
             "bounds": [0, 0, 400, 300], "answered": True,
             "answer_window": {"sequence": 5, "started": 10.1, "finished": 10.9,
                               "captured_through": 11.1, "delivered": True}}
    return pending, after


def decision(expectation=Expectation()):
    return Deliberation(goal="prepare the workspace", situation="Filters", episode_id="trial",
                        chosen=ActionOption('click "Filters"', expectation=expectation))


def stores():
    resolved, recorded = [], []
    return SimpleNamespace(resolve=lambda *a: resolved.append(a)), SimpleNamespace(record_outcome=lambda *a: recorded.append(a)), resolved, recorded


@pytest.mark.parametrize("surface", ["image editor", "document chooser", "circuit builder", "inventory"])
def test_same_words_can_have_an_owned_visual_response(surface):
    pending, after = evidence(surface)
    state = owned_response_state(pending, after)
    spine, graph, resolved, recorded = stores()
    attempt = confirm(decision(), "Filters", "Filters", change_state=state, spine=spine, graph=graph)
    assert attempt.verdict.held is True and attempt.verdict.observed_change is True
    assert attempt.progressed is None
    assert len(resolved) == len(recorded) == 1


@pytest.mark.parametrize("expectation", [Expectation(contains=("Saved",)), Expectation(absent=("Filters",))])
def test_visual_response_cannot_satisfy_a_missing_result(expectation):
    spine, graph, _, _ = stores()
    attempt = confirm(decision(expectation), "Filters", "Filters", change_state=PredicateState.SATISFIED,
                      spine=spine, graph=graph)
    assert attempt.verdict.held is False and attempt.verdict.observed_change is True


def test_owned_visual_response_cannot_satisfy_a_wrong_predicted_arrangement():
    from core.perception.what_is_there import Arrangement, Cell

    reading = Arrangement(rows=1, columns=1, cells=(Cell(row=0, column=0, says="actual", at=(0., 0.)),))
    claim = Expectation(becomes=object(), becomes_holds=lambda _after: False)
    spine, graph, _, _ = stores()
    attempt = confirm(decision(claim), "same", "same", seen_before=reading, seen_after=reading,
                      change_state=PredicateState.SATISFIED, spine=spine, graph=graph)
    assert attempt.verdict.held is False and attempt.verdict.observed_change is True


@pytest.mark.parametrize("damage", ["surface", "bounds", "epoch", "capture", "stream_capture", "early", "late",
                                   "unfinished", "refused", "batch", "sequence_missing", "not_answered", "nan"])
def test_unowned_or_incomplete_response_is_unknown(damage):
    pending, after = evidence()
    window = after["answer_window"]
    changes = {"surface": lambda: after.update(surface_id="elsewhere"),
               "bounds": lambda: after.update(bounds=[0, 0, 800, 600]),
               "epoch": lambda: after.update(_capture_epoch="a"),
               "capture": lambda: after.update(capture_at=2.),
               "stream_capture": lambda: window.update(captured_through=10.8),
               "early": lambda: window.update(started=9.),
               "late": lambda: window.update(finished=12.),
               "unfinished": lambda: window.update(finished=None),
               "refused": lambda: window.update(delivered=False),
               "batch": lambda: window.update(sequence=6),
               "sequence_missing": lambda: window.pop("sequence"),
               "not_answered": lambda: after.update(answered=None),
               "nan": lambda: window.update(started=float("nan"))}
    changes[damage]()
    assert owned_response_state(pending, after) is PredicateState.UNKNOWN


def test_unknown_response_creates_no_false_episode_or_consequence_grade():
    spine, graph, resolved, recorded = stores()
    attempt = confirm(decision(), "score 0", "score 100", toward="score 100",
                      change_state=PredicateState.UNKNOWN, spine=spine, graph=graph)
    assert attempt.verdict.held is None and attempt.verdict.observed_change is None
    assert attempt.verdict.measured is False and attempt.progressed is None
    assert resolved == recorded == []
    assert "awaits a measured response" in attempt.as_evidence()


def test_an_unowned_positive_cannot_override_unknown_in_a_learning_consumer():
    pending, after = evidence()
    after["answer_window"]["sequence"] += 1
    assert owned_response_state(pending, after) is PredicateState.UNKNOWN
    assert it_was_answered(None, after) is None


def test_a_measured_negative_vetoes_unrelated_changed_words():
    spine, graph, _, _ = stores()
    attempt = confirm(decision(), "Filters clock 12", "Filters clock 13",
                      change_state=PredicateState.UNSATISFIED, spine=spine, graph=graph)
    assert attempt.verdict.held is False and attempt.verdict.observed_change is False


def test_legacy_unstreamed_callers_remain_explicitly_supported():
    assert owned_response_state({}, {"text": "somewhere"}) is None
    assert owned_response_state({}, {"answer_window": None}) is PredicateState.UNKNOWN


def test_step_receipt_uses_visual_response_when_words_and_inventory_are_identical():
    pending, after = evidence()
    frame = Frame("Click Filters", act="click things", thing="Filters")
    rules = Rules(frames={frame.key: frame})
    attempt = rules.begin('click "Filters"', pending["observed_before"])
    pending.update(deliberation=decision(), step_bindings=((rules, attempt),))
    assert settle_step(pending, after, ['click "Filters"'])
    assert rules.passed(frame)


def test_step_receipt_cannot_be_completed_by_a_stale_positive_window():
    pending, after = evidence()
    frame = Frame("Click Filters", act="click things", thing="Filters")
    rules = Rules(frames={frame.key: frame})
    attempt = rules.begin('click "Filters"', pending["observed_before"])
    pending.update(deliberation=decision(), step_bindings=((rules, attempt),))
    after["answer_window"]["started"] = 9.
    assert settle_step(pending, after, ['click "Filters"'])
    assert not rules.passed(frame) and rules.receipts[-1].state is PredicateState.UNKNOWN
    assert frame.key not in rules.unanswered_steps


def test_early_unknown_waits_for_the_response_interval(monkeypatch):
    pending, after = evidence()
    pending.update(deliberation=decision(), verification_started=10.)
    after["answered"] = None
    monkeypatch.setattr("core.skills.screen_step_evidence.time.monotonic", lambda: 10.5)
    assert not settle_step(pending, after, ['click "Filters"'])
    monkeypatch.setattr("core.skills.screen_step_evidence.time.monotonic", lambda: 13.1)
    assert settle_step(pending, deepcopy(after), ['click "Filters"'])


def test_owned_dispatch_invariant_is_executable():
    assert _response_dispatch_invariant() == ()
    assert _unmeasured_response_invariant() == ()


@pytest.mark.parametrize("name", ["response_streams_keep_owned_capture_windows", "responses_belong_to_one_delivered_dispatch",
                                 "unmeasured_responses_are_not_graded", "object_candidates_require_measured_extents",
                                 "completed_guidance_cannot_publish"])
def test_registered_capability_claims_run_their_actual_canaries(name):
    from core.organism.model_validation import Evidence, get_suite, install_runtime_validation

    install_runtime_validation()
    suite = get_suite()
    test = next(test for test in suite.tests() if test.name == name)
    assert test.predict(None) is True
    claim = next(claim for claim in suite.claims() if claim.test == name)
    assert claim.evidence is Evidence.MEASURED_SYNTHETIC
    assert "live demo performance remains independently unproved" in claim.evidence_note


def test_unknown_placement_does_not_suppress_every_future_transfer():
    from core.agency.putting_things_in_place import RESTS_AFTER, PuttingInPlace

    move = 'drag "candidate" to "workspace"'
    carrying = PuttingInPlace(carrying_said=True)
    for number in range(RESTS_AFTER + 1):
        receipt = SimpleNamespace(move=move, state=PredicateState.UNKNOWN, attempt_id=str(number))
        carrying.carried(move, True, receipt=receipt)
    assert carrying.carries_unanswered == 0 and move not in carrying.carried_to
    assert all(value is PredicateState.UNKNOWN for value in carrying.placement_states.values())


@pytest.mark.asyncio
async def test_settling_sees_a_fast_owned_pixel_response_with_identical_ocr(monkeypatch):
    from core.skills import screen_pursuit
    from core.skills.screen_pursuit_looking import _settled_after

    pending, after = evidence()
    after["settled"] = True
    reads = []

    async def read(_app):
        reads.append(True)
        return after

    monkeypatch.setattr(screen_pursuit, "read_screen", read)
    seen, changed = await _settled_after(pending["observed_before"], "editor", patience=.8,
                                       response_state=lambda observation: owned_response_state(pending, observation))
    assert seen is after and changed is True and len(reads) == 1


@pytest.mark.asyncio
async def test_settling_does_not_claim_a_stale_positive_or_autonomous_text_as_an_answer(monkeypatch):
    from core.skills import screen_pursuit
    from core.skills.screen_pursuit_looking import _settled_after

    pending, after = evidence()
    after.update(settled=True, text="Filters clock 13")
    after["answer_window"]["sequence"] += 1

    async def read(_app):
        return after

    monkeypatch.setattr(screen_pursuit, "read_screen", read)
    seen, changed = await _settled_after(pending["observed_before"], "editor", patience=.35,
                                       response_state=lambda observation: owned_response_state(pending, observation))
    assert seen is after and changed is None
