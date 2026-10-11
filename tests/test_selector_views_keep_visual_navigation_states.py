"""Owned selector contents distinguish same-word screens without tracking gameplay frames."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.what_i_can_do_here import WhatWorksHere, a_click_on
from core.agency.where_things_lead import (
    MOST_SCREENS,
    MOST_SELECTOR_VARIANTS,
    WhereThingsLead,
    screen_words,
)
from core.cognition.procedure_binding import RevealedChoices
from core.cognition.reading_the_rules import EvidenceSnapshot, StepAttempt, StepReceipt
from core.cognition.selector_view import UNKNOWN_SELECTOR_VIEW, SelectorSignature, SelectorView
from core.perception.drawn_objects import drawn_objects, measured_objects
from core.perception.observed_transfer import ScreenReading, make_snapshot
from core.runtime.skill_contract import PredicateState
from core.skills.screen_step_evidence import screen_evidence

WORDS = (a_click_on("Library"), a_click_on("Category"), a_click_on("Close"))


def _outline(image, x, y, width, height):
    image[y:y + 2, x:x + width] = 190
    image[y + height - 2:y + height, x:x + width] = 190
    image[y:y + height, x:x + 2] = 190
    image[y:y + height, x + width - 2:x + width] = 190


def _sprite(image, x, y, variant=0):
    image[y:y + 20, x:x + 25] = (120 + variant * 7, 100, 40)
    image[y + 4:y + 9, x + 3:x + 20] = (70, 190 - variant * 5, 160)


def _reading(*, variant=0, epoch="after", captured=2., background=0, empty=False, extra=False):
    image = np.empty((240, 500, 3), np.uint8)
    image[:] = (20, 30, 50)
    _outline(image, 15, 15, 170, 190)
    _outline(image, 260, 15, 180, 190)
    image[155:167, 35:110] = (150, 150, 190)
    writing = [{"text": "Category", "x": 35 / 500, "y": 155 / 240, "width": 75 / 500,
                "height": 12 / 240, "center_x": 72.5 / 500, "center_y": 161 / 240}]
    _sprite(image, 285, 45, background)
    if not empty:
        _sprite(image, 35, 45, variant)
        _sprite(image, 100, 85, variant)
    if extra:
        _sprite(image, 45, 110, variant)
    reading = ScreenReading({"objects": drawn_objects(image, writing=writing, capture_epoch=epoch),
                             "layout": writing, "shapes": [], "text": "Library Category Close",
                             "surface_id": "editor", "bounds": [0, 0, 500, 240],
                             "capture_at": captured, "_capture_epoch": epoch, "settled": True},
                            picture_snapshot=make_snapshot(image))
    objects = measured_objects(reading)
    assert len(objects) == len(reading["objects"])
    return reading, (*WORDS, *(a_click_on(name) for name in objects))


def _owned_view(variant=0):
    before, before_moves = _reading(empty=True, epoch="before", captured=1.)
    after, after_moves = _reading(variant=variant)
    prior, current = screen_evidence(before, before_moves), screen_evidence(after, after_moves)
    choices = RevealedChoices.from_effect(prior, current, "open", "choose", opened_move=a_click_on("Category"))
    assert choices is not None
    attempt = StepAttempt(1, "open", a_click_on("Category"), EvidenceSnapshot.of(prior))
    receipt = StepReceipt(attempt, EvidenceSnapshot.of(current), PredicateState.SATISFIED, True, "owned reveal")
    view = SelectorView.from_effect(receipt, choices)
    assert view is not None and len(view.signature.sources) == 2
    assert isinstance(view.current(after, after_moves), SelectorView)
    return view, after, after_moves, receipt, choices


def test_stable_capture_epochs_do_not_create_selector_states_and_anonymous_labels_are_not_words():
    view, after, moves, _, _ = _owned_view()
    again, again_moves = _reading(epoch="later", captured=3.)
    current = view.current(again, again_moves)
    assert isinstance(current, SelectorView) and current.signature == view.signature
    assert current.epoch != view.epoch
    assert screen_words(moves) == screen_words(WORDS)
    leads = WhereThingsLead()
    first = leads.which(screen_words(moves), view.current(after, moves))
    assert leads.which(screen_words(again_moves), current) == first
    assert len(leads.screens) == 1


def test_same_ocr_palettes_have_separate_offerings_and_navigation_edges():
    one, _, first_moves, _, _ = _owned_view(0)
    two, _, next_moves, _, _ = _owned_view(1)
    assert one.signature != two.signature and screen_words(first_moves) == screen_words(next_moves)
    leads = WhereThingsLead()
    leads.looked(first_moves, selector_view=one)
    first = leads._here
    leads.in_order(first_moves)
    leads.looked(next_moves, selector_view=two)
    leads.acted(a_click_on("Category"), True)
    second = leads._here
    assert first != second and leads.to[f"{first}|{a_click_on('Category')}"] == second
    leads.in_order(next_moves)
    leads.looked(first_moves, selector_view=one)
    leads.acted(a_click_on("Category"), True)
    assert leads.to[f"{second}|{a_click_on('Category')}"] == first


def test_background_motion_outside_owned_container_preserves_the_selector_view():
    view, _, _, _, _ = _owned_view()
    animated, moves = _reading(background=5, epoch="moving", captured=3.)
    current = view.current(animated, moves)
    assert isinstance(current, SelectorView) and current.signature == view.signature


def test_owned_dismissal_retires_a_selector_while_missing_contents_alone_do_not():
    from core.skills.screen_step_evidence import settle_step

    view, before, moves, _, choices = _owned_view()
    before["layout"].append({"text": "Close", "center_x": .2, "center_y": .75})
    after, after_moves = _reading(empty=True, epoch="closed", captured=3.)
    assert view.closed_by(a_click_on("Close"), before, after)
    assert not view.closed_by(a_click_on("Category"), before, after)
    assert not view.closed_by(a_click_on("Close"), before, before)
    before_at = screen_evidence(before, moves)
    owner = SimpleNamespace(selector_view=view, revealed_choices=choices)

    def pending(answered):
        return {"observed_before": before_at, "watched": before,
                "dispatch_completed": 2.5, "verification_started": 0.,
                "dispatch_started_monotonic": 10., "dispatch_completed_monotonic": 11.,
                "dispatch_sequence_before": 0, "selector_before": ((owner, view),),
                "deliberation": SimpleNamespace(chosen=SimpleNamespace(name=a_click_on("Close")))}

    after["answer_window"] = {"sequence": 1, "started": 10.1, "finished": 10.9,
                               "captured_through": 11.1, "delivered": True}
    window = after.pop("answer_window")
    assert settle_step(pending(None), after, after_moves)
    assert owner.selector_view is view and owner.revealed_choices is choices
    after["answer_window"] = window
    after["answered"] = None
    assert settle_step(pending(None), after, after_moves)
    assert owner.selector_view is view and owner.revealed_choices is choices
    after["answered"] = True
    assert settle_step(pending(True), after, after_moves)
    assert owner.selector_view is None and owner.revealed_choices is None


def test_an_empty_still_open_palette_or_outside_dismissal_cannot_retire_its_view():
    view, before, _, _, _ = _owned_view()
    before["layout"].append({"text": "Close", "center_x": .2, "center_y": .75})
    empty, _ = _reading(empty=True, epoch="empty", captured=3.)
    empty["layout"].append({"text": "Close", "center_x": .2, "center_y": .75})
    assert not view.closed_by(a_click_on("Close"), before, empty)
    empty["layout"].pop()
    before["layout"][-1]["center_x"] = .95
    assert not view.closed_by(a_click_on("Close"), before, empty)


def test_omitted_public_objects_cannot_claim_dismissal_of_unchanged_private_pixels():
    view, before, _, _, _ = _owned_view()
    before["layout"].append({"text": "Close", "center_x": .2, "center_y": .75})
    forged = ScreenReading({**before, "_capture_epoch": "later", "capture_at": 3., "objects": [],
                            "layout": [region for region in before["layout"] if region.get("text") != "Close"]},
                           picture_snapshot=before.picture_snapshot)
    assert not view.closed_by(a_click_on("Close"), before, forged)


@pytest.mark.parametrize("change", ["changed", "missing", "extra", "occluded", "unsettled", "viewport", "surface", "old"])
def test_unowned_changed_or_unobserved_contents_are_unknown_and_do_not_mint_nodes(change):
    view, original, moves, _, _ = _owned_view()
    if change in {"changed", "extra"}:
        current, moves = _reading(variant=int(change == "changed"), extra=change == "extra", epoch="next", captured=3.)
    else:
        current = ScreenReading(dict(original), picture_snapshot=original.picture_snapshot)
        if change == "missing":
            current["objects"] = current["objects"][1:]
        elif change == "occluded":
            current["occluded"] = True
        elif change == "unsettled":
            current["settled"] = False
        elif change == "viewport":
            current["bounds"] = [0, 0, 500, 260]
        elif change == "surface":
            current["surface_id"] = "elsewhere"
        else:
            current["capture_at"] = .5
    leads = WhereThingsLead()
    leads.looked(moves, selector_view=view)
    before = leads.as_memory()
    invalid = view.current(current, moves)
    assert invalid is UNKNOWN_SELECTOR_VIEW
    leads.looked(moves, selector_view=invalid)
    leads.acted(a_click_on("Category"), True)
    assert leads._here is None and leads.as_memory() == before
    assert not leads.went and not leads.to


@pytest.mark.parametrize("state", [PredicateState.UNKNOWN, PredicateState.UNSATISFIED])
def test_unconfirmed_effect_cannot_create_a_view(state):
    _, _, _, receipt, choices = _owned_view()
    assert SelectorView.from_effect(StepReceipt(receipt.attempt, receipt.after, state, True, "unconfirmed"), choices) is None


def test_wrong_receipt_and_uncertain_response_cannot_create_a_view():
    _, _, _, receipt, choices = _owned_view()
    wrong = StepAttempt(2, "different", receipt.attempt.move, receipt.before)
    assert SelectorView.from_effect(StepReceipt(wrong, receipt.after, receipt.state, True, "wrong owner"), choices) is None
    assert SelectorView.from_effect(StepReceipt(receipt.attempt, receipt.after, receipt.state, None, "unmeasured"), choices) is None


def test_selector_variant_and_total_screen_limits_return_unknown_without_merging():
    leads = WhereThingsLead()
    for variant in range(MOST_SELECTOR_VARIANTS):
        view, _, moves, _, _ = _owned_view(variant)
        assert leads.which(screen_words(moves), view) is not None
    view, _, moves, _, _ = _owned_view(MOST_SELECTOR_VARIANTS)
    assert leads.which(screen_words(moves), view) is None
    assert len(leads.screens) == MOST_SELECTOR_VARIANTS
    ordinary = WhereThingsLead()
    for number in range(MOST_SCREENS):
        assert ordinary.which(frozenset({f"identity{number}", f"place{number}"})) == number
    assert ordinary.which(frozenset({"unseen", "bounded"})) is None
    assert len(ordinary.screens) == MOST_SCREENS


def test_retained_variants_rehydrate_by_valid_signatures_after_graph_renumbering():
    one, _, moves, _, _ = _owned_view()
    two, _, _, _, _ = _owned_view(1)
    leads = WhereThingsLead()
    first = leads.which(screen_words(moves), one)
    second = leads.which(screen_words(moves), two)
    leads.went[f"{second}|{a_click_on('Category')}"] = {"on": 9}
    leads.to[f"{second}|{a_click_on('Category')}"] = first
    leads.offered[first] = frozenset(moves)
    kept = leads.as_memory()
    assert kept["selector_signatures"]["0"] == two.signature.as_memory()
    again = WhereThingsLead.from_memory(kept)
    assert again.which(screen_words(moves), two) == 0
    assert again.which(screen_words(moves), one) == 1
    assert again.to[f"0|{a_click_on('Category')}"] == 1
    # Absence of a current owned view cannot reuse a retained visual identity.
    assert again.which(screen_words(moves)) not in again.selector_signatures


def test_legacy_memory_stays_lexical_and_damaged_selector_memory_is_not_downgraded():
    leads = WhereThingsLead.from_memory({"screens": [["library", "category"]], "went": {"0|left": {"moved": 2}}})
    assert leads.which(frozenset({"library", "category"})) == 0 and not leads.selector_signatures
    view, _, moves, _, _ = _owned_view()
    leads.which(screen_words(moves), view)
    kept = leads.as_memory()
    held = next(iter(kept["selector_signatures"].values()))
    held["sources"][0][1] = "not a measured witness"
    assert not WhereThingsLead.from_memory(kept).screens
    assert SelectorSignature.from_memory(held) is None


def test_missing_variant_families_and_duplicate_retained_visual_identities_are_rejected():
    view, _, moves, _, _ = _owned_view()
    leads = WhereThingsLead()
    leads.which(screen_words(moves), view)
    kept = leads.as_memory()
    kept["families"] = {}
    assert not WhereThingsLead.from_memory(kept).screens
    kept = leads.as_memory()
    kept["screens"].append(kept["screens"][0])
    kept["selector_signatures"]["1"] = kept["selector_signatures"]["0"]
    kept["families"]["1"] = 0
    assert not WhereThingsLead.from_memory(kept).screens


def test_unknown_selector_probe_and_normal_wordless_gameplay_keep_their_boundaries():
    from core.agency.where_things_lead import WORDLESS, _unknown_selector_invariant

    assert _unknown_selector_invariant() == ()
    leads = WhereThingsLead()
    assert leads.which(screen_words([a_click_on("SCORE 00150")])) == WORDLESS
    shape = 'click "the shape at 50% across, 80% down"'
    leads.looked((shape,))
    leads.looked((shape,))
    leads.acted("left", True)
    assert WORDLESS in leads.played and not leads.selector_signatures


@pytest.mark.parametrize("other", ["absent", "same", "different", "unknown"])
def test_normal_screen_intake_passes_validated_selector_after_owned_context(monkeypatch, other):
    from core.cognition.a_guide_to_a_place import THE_GUIDE
    from core.skills import screen_pursuit_bearings as bearings
    from core.skills import screen_pursuit_looking as looking

    view, reading, moves, _, _ = _owned_view()
    events = []
    procedure = SimpleNamespace(observe_context=lambda evidence: events.append("context"),
                                current_selector_view=lambda observation, clickable: (events.append("validated")
                                                                                     or view.current(observation, clickable)),
                                asks_to_carry=lambda: False)
    additional = None
    if other != "absent":
        another = _owned_view(1)[0] if other == "different" else view
        additional = SimpleNamespace(observe_context=lambda evidence: events.append("plan_context"),
                                     next_step=lambda: None,
                                     current_selector_view=lambda observation, clickable: (events.append("plan_validated")
                                         or (UNKNOWN_SELECTOR_VIEW if other == "unknown" else another.current(observation, clickable))))
    guide = SimpleNamespace(rules=procedure, plan=additional, reading=SimpleNamespace(acts=lambda: ()))
    monkeypatch.setattr(bearings, "things_to_click", lambda *_: moves)
    monkeypatch.setattr(bearings, "what_it_says", lambda *_: "Library Category Close")
    monkeypatch.setattr(bearings, "where_to_click", lambda *_: None)
    for name in ("_where_the_way_on_was", "_what_paused_it", "_the_place_its_words_name"):
        monkeypatch.setattr(looking, name, lambda observation, clickable, *_: clickable)
    monkeypatch.setattr(looking, "_what_she_has_built", lambda *_: None)
    monkeypatch.setattr(looking, "_as_the_guide_reads_it", lambda clickable, *_, **__: clickable)
    monkeypatch.setattr(looking, "_her_bearings", lambda *_: SimpleNamespace(leaving_alone=()))
    monkeypatch.setattr(WhatWorksHere, "asked_for_by", lambda *_args, **_kwargs: None)
    here = WhatWorksHere()
    here.looked_at(WORDS)
    prior = here.leads._here
    token = THE_GUIDE.set(guide)
    try:
        looking._take_in_the_screen(here, reading, None, False)
        assert events == (["context", "validated"] if other == "absent" else
                          ["context", "plan_context", "validated", "plan_validated"])
        current = here.leads._here
        if other in {"different", "unknown"}:
            assert current is None and len(here.leads.screens) == 1
            here.tried(a_click_on("Category"), True)
            assert not here.leads.to
            return
        assert current != prior and here.leads.selector_signatures[current] == view.signature
        here.tried(a_click_on("Category"), True)
        assert here.leads.to[f"{prior}|{a_click_on('Category')}"] == current
    finally:
        THE_GUIDE.reset(token)
