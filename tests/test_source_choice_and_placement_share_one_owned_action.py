"""One declared source-use action may prove only its prebound obligations."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from core.cognition.a_plan_to_an_end import Plan
from core.cognition.action_refinement import _bound_refinement_invariant, choose_and_carry
from core.cognition.reading_the_rules import Frame, Rules
from core.runtime.skill_contract import PredicateState, SemanticPredicate

NAME = "the object at 15.00% across, 25.00% down"
MOVE = f'drag "{NAME}" to "workspace"'


def observation(epoch, labels=(), *, identity="a" * 64, object_visible=True):
    out = {"text": " ".join(labels), "surface_id": "editor", "bounds": [0, 0, 400, 300],
           "capture_epoch": str(epoch), "capture_at": float(epoch), "work_epoch": "work-1",
           "clickable": [f'click "{label}"' for label in labels], "regions": []}
    if object_visible:
        out["clickable"].append(f'click "{NAME}"')
        out["regions"].append({"text": NAME, "x": .1, "y": .2, "width": .1, "height": .1,
                               "center_x": .15, "center_y": .25, "role": "object", "source": "measured_pixels",
                               "candidate_id": identity, "container_bbox": [.05, .1, .4, .7],
                               "capture_epoch": str(epoch)})
    return out


def revealed_rules(entity="component", *, cls=Rules, carry_thing=None):
    choice = Frame(f"Choose an item of {entity}", act="click things", thing=f"an item of {entity}")
    opener = Frame(f"Open the {entity} library", act="click things", thing=f"{entity} library", order=1)
    carry = Frame(f"Place the {entity} in the workspace", act="carry", thing=carry_thing or entity,
                  where="workspace", order=2)
    rules = cls(frames={frame.key: frame for frame in (choice, opener, carry)})
    before = observation(1, [f"{entity} library"], object_visible=False)
    current = observation(2, [f"{entity} library", "workspace"])
    rules.begin(f'click "{entity} library"', before, step_key=opener.key)
    rules.tried(f'click "{entity} library"', True, after=current)
    assert rules.next_step() is choice and rules.revealed_choices is not None
    return rules, choice, carry, current


def binding_for(rules, current):
    candidate = rules.refinement_of(MOVE, current)
    assert candidate is not None
    return candidate.with_transfer("physical-1", source_at=(.15, .25), destination_at=(.75, .7))


def outcome(binding, **extra):
    out = observation(3, ["workspace"], object_visible=False)
    out.update(extra)
    out["refinement"] = {"state": "verified", "attempt_id": binding.attempt_id,
                         "move": binding.action.move, "source": binding.action.source,
                         "destination": binding.action.destination, "source_at": list(binding.source_at),
                         "destination_at": list(binding.destination_at)}
    return out


def test_registered_refinement_requires_the_exact_bound_transfer():
    assert _bound_refinement_invariant() == ()


@pytest.mark.parametrize("entity", ["component", "asset", "card"])
def test_current_source_choice_and_placement_are_bound_before_the_same_physical_action(entity):
    rules, choice, carry, current = revealed_rules(entity)
    binding = binding_for(rules, current)
    assert rules.step_of(MOVE) is choice and rules.the_step_to_do([MOVE]) is choice
    assert rules.begin(MOVE, current) is None
    assert rules.tried(MOVE, True, after=outcome(binding)) is None
    assert not rules.passed(choice) and not rules.passed(carry)
    primary = rules.begin(MOVE, current, refinement=binding)
    pending = rules.pending_refinement
    assert primary is pending.attempts[0]
    assert [attempt.step_key for attempt in pending.attempts] == [choice.key, carry.key]
    assert [attempt.refinement_role for attempt in pending.attempts] == ["choose_value", "place_value"]
    assert all(attempt.refinement_id == binding.attempt_id for attempt in pending.attempts)
    assert len({attempt.attempt_id for attempt in pending.attempts}) == 2
    assert not rules.passed(choice) and not rules.passed(carry)
    receipt = rules.tried(MOVE, True, after=outcome(binding))
    assert receipt.state is PredicateState.SATISFIED
    assert all(item.state is PredicateState.SATISFIED for item in rules.receipts[-2:])
    assert rules.passed(choice) and rules.passed(carry)
    assert rules.pending_step is None and rules.pending_refinement is None


@pytest.mark.parametrize("damage", ["missing", "different_attempt", "different_move", "different_source", "different_destination",
                                  "different_pickup", "different_drop", "surface", "bounds", "work", "epoch", "time"])
def test_a_generic_or_foreign_transfer_cannot_certify_the_bound_bundle(damage):
    rules, choice, carry, current = revealed_rules()
    binding = binding_for(rules, current)
    rules.begin(MOVE, current, refinement=binding)
    after = outcome(binding, placement={"state": "verified"})
    if damage == "missing":
        after.pop("refinement")
    elif damage.startswith("different_"):
        key = {"different_attempt": "attempt_id", "different_move": "move", "different_source": "source",
               "different_destination": "destination", "different_pickup": "source_at", "different_drop": "destination_at"}[damage]
        after["refinement"][key] = [0., 0.] if key.endswith("_at") else "another"
    else:
        key = {"surface": "surface_id", "bounds": "bounds", "work": "work_epoch", "epoch": "capture_epoch", "time": "capture_at"}[damage]
        after[key] = {"surface": "other", "bounds": [0, 0, 800, 600], "work": "work-2", "epoch": "2", "time": 1.}[damage]
    receipt = rules.tried(MOVE, True, after=after)
    assert receipt.state is PredicateState.UNKNOWN
    assert not rules.passed(choice) and not rules.passed(carry)
    assert rules.pending_step is None and rules.pending_refinement is None


@pytest.mark.parametrize("changed,state", [(None, PredicateState.UNKNOWN), (False, PredicateState.UNSATISFIED)])
def test_refused_or_unmeasured_delivery_releases_every_owned_obligation(changed, state):
    rules, choice, carry, current = revealed_rules()
    binding = binding_for(rules, current)
    rules.begin(MOVE, current, refinement=binding)
    receipt = rules.tried(MOVE, changed, after={})
    assert receipt.state is state and all(item.state is state for item in rules.receipts[-2:])
    assert not rules.passed(choice) and not rules.passed(carry)
    assert rules.pending_step is None and rules.pending_refinement is None


def test_both_declarations_remain_frozen_and_one_missing_effect_prevents_bundle_credit():
    rules, choice, carry, current = revealed_rules()
    choice_effect = SemanticPredicate("selection_confirmed", "after.selection.confirmed", expected=True)
    carry_effect = SemanticPredicate("attached", "after.attachment.confirmed", expected=True)
    rules.effects = {choice.key: [choice_effect], carry.key: [carry_effect]}
    binding = binding_for(rules, current)
    rules.begin(MOVE, current, refinement=binding)
    held = rules.pending_refinement
    current["regions"][0]["candidate_id"] = "b" * 64
    rules.effects.clear()
    with pytest.raises(FrozenInstanceError):
        held.attempts[0].refinement_role = "changed"
    assert "selection_confirmed" in held.attempts[0].predicates
    assert "attached" in held.attempts[1].predicates
    assert held.attempts[0].before.evidence["regions"][0]["candidate_id"] == "a" * 64
    receipt = rules.tried(MOVE, True, after=outcome(binding, selection={"confirmed": True}))
    assert receipt.state is PredicateState.UNKNOWN
    assert not rules.passed(choice) and not rules.passed(carry)


def test_an_explicit_failed_own_effect_prevents_both_obligations_from_advancing():
    rules, choice, carry, current = revealed_rules()
    rules.effects[choice.key] = [SemanticPredicate("selection_confirmed", "after.selection.confirmed", expected=True)]
    binding = binding_for(rules, current)
    rules.begin(MOVE, current, refinement=binding)
    receipt = rules.tried(MOVE, True, after=outcome(binding, selection={"confirmed": False}))
    assert receipt.state is PredicateState.UNSATISFIED
    assert not rules.passed(choice) and not rules.passed(carry)


def test_the_shared_occupancy_predicate_cannot_erase_either_plans_expected_phrase():
    rules, choice, carry, current = revealed_rules(cls=Plan)
    rules.expects = {choice.key: "selection ready", carry.key: "attachment ready"}
    binding = binding_for(rules, current)
    rules.begin(MOVE, current, refinement=binding)
    after = outcome(binding)
    after["text"] = "selection ready"
    receipt = rules.tried(MOVE, True, after=after)
    assert receipt.state is PredicateState.UNSATISFIED
    assert not rules.passed(choice) and not rules.passed(carry)
    assert rules.awaiting == "" and rules._pending_changed is None


def test_plan_waiting_and_later_abandonment_keep_and_release_the_same_bundle():
    rules, choice, carry, current = revealed_rules(cls=Plan)
    binding = binding_for(rules, current)
    rules.begin(MOVE, current, refinement=binding)
    assert rules.tried(MOVE, True) is None
    assert rules.awaiting == choice.key
    assert rules.states[choice.key] is rules.states[carry.key] is PredicateState.UNKNOWN
    assert rules.pending_refinement is not None
    rules.tried(MOVE, None, after={})
    assert rules.awaiting == "" and rules._pending_changed is None
    assert rules.pending_refinement is None and rules.pending_step is None


def test_plan_can_settle_both_frozen_declarations_after_a_late_owned_observation():
    rules, choice, carry, current = revealed_rules(cls=Plan)
    rules.expects = {choice.key: "selection ready", carry.key: "attachment ready"}
    binding = binding_for(rules, current)
    rules.begin(MOVE, current, refinement=binding)
    assert rules.tried(MOVE, True) is None
    after = outcome(binding)
    after["text"] = "selection ready; attachment ready"
    receipt = rules.tried(MOVE, True, after=after)
    assert receipt.state is PredicateState.SATISFIED
    assert rules.passed(choice) and rules.passed(carry)
    assert rules.awaiting == "" and rules._pending_changed is None and not rules.stuck


@pytest.mark.parametrize("owned", [True, False])
def test_plans_normal_late_observation_settles_every_bound_obligation_together(owned):
    plan, choice, carry, current = revealed_rules(cls=Plan)
    binding = binding_for(plan, current)
    plan.begin(MOVE, current, refinement=binding)
    assert plan.tried(MOVE, True) is None
    after = outcome(binding)
    if not owned:
        after["refinement"]["attempt_id"] = "a different action"
    assert not plan.saw(after)
    assert plan.passed(choice) is owned and plan.passed(carry) is owned
    assert plan.states[choice.key] is plan.states[carry.key]
    assert plan.awaiting == "" and plan._pending_changed is None
    assert plan.pending_step is None and plan.pending_refinement is None
    assert len(plan.receipts[-2:]) == 2


def test_a_late_plan_observation_cannot_replace_the_prebound_refinement_effects():
    plan, choice, carry, current = revealed_rules(cls=Plan)
    binding = binding_for(plan, current)
    plan.begin(MOVE, current, refinement=binding)
    assert plan.tried(MOVE, True) is None
    with pytest.raises(ValueError):
        plan.saw(outcome(binding), effects=(SemanticPredicate("late", "after.refinement.state", expected="verified"),))
    assert not plan.passed(choice) and not plan.passed(carry)
    assert plan.pending_refinement is not None


@pytest.mark.parametrize("cls", [Rules, Plan])
def test_source_use_does_not_mint_an_unrelated_selector_reveal_for_the_next_choice(cls):
    rules, choice, carry, current = revealed_rules(cls=cls)
    later = Frame("Choose another item of component", act="click things", thing="an item of component", order=3)
    rules.frames[later.key] = later
    original = rules.revealed_choices
    binding = binding_for(rules, current)
    rules.begin(MOVE, current, refinement=binding)
    rules.tried(MOVE, True, after=outcome(binding))
    assert rules.next_step() is later and rules.revealed_choices is original
    assert not rules.revealed_choices.offers(f'click "{NAME}"', current, later.key)


@pytest.mark.parametrize("damage", ["entity", "literal", "category_only", "intervening", "unnamed_destination", "wrong_destination",
                                  "changed_object", "missing_object", "foreign_capture", "work"])
def test_only_an_owned_same_entity_source_and_immediately_dependent_placement_can_refine(damage):
    rules, choice, carry, current = revealed_rules()
    if damage == "entity":
        rules.frames[carry.key] = replace(carry, thing="asset")
    elif damage == "literal":
        rules.frames[choice.key] = replace(choice, sentence='Click "an item of component"')
    elif damage == "category_only":
        rules.revealed_choices = replace(rules.revealed_choices, object_witnesses=())
    elif damage == "intervening":
        other = Frame("Confirm the selection", act="click things", thing="Confirm", order=1.5)
        rules.frames[other.key] = other
    elif damage == "unnamed_destination":
        rules.frames[carry.key] = replace(carry, where="")
    elif damage == "wrong_destination":
        rules.frames[carry.key] = replace(carry, where="tray")
    elif damage == "changed_object":
        current["regions"][0]["candidate_id"] = "b" * 64
    elif damage == "missing_object":
        current["regions"] = []
    elif damage == "foreign_capture":
        current["regions"][0]["capture_epoch"] = "other"
    else:
        # A work change invalidates the explicit prebound proposal at begin.
        binding = binding_for(rules, current)
        current["work_epoch"] = "work-2"
        with pytest.raises(ValueError):
            rules.begin(MOVE, current, refinement=binding)
        return
    assert choose_and_carry(rules, MOVE, current) is None


def test_a_source_clause_is_not_a_new_entity_and_a_measured_type_choice_can_bind_an_instance():
    rules, choice, carry, current = revealed_rules(carry_thing="a component from the library")
    revised = replace(choice, thing="the type of component you wish to use",
                      sentence="Choose the type of component you wish to use")
    rules.frames = {revised.key if key == choice.key else key: revised if key == choice.key else frame
                    for key, frame in rules.frames.items()}
    rules.revealed_choices = replace(rules.revealed_choices, for_step=revised.key)
    assert choose_and_carry(rules, MOVE, current) is not None


def test_a_bundle_cannot_be_rebound_and_cannot_own_a_different_observed_action():
    rules, choice, carry, current = revealed_rules()
    binding = binding_for(rules, current)
    rules.begin(MOVE, current, refinement=binding)
    with pytest.raises(RuntimeError):
        rules.begin(MOVE, current, refinement=binding)
    with pytest.raises(RuntimeError):
        rules.tried('click "Run"', True, after=outcome(binding))
    assert not rules.passed(choice) and not rules.passed(carry)
    assert rules.pending_refinement is not None


@pytest.mark.parametrize("cls", [Rules, Plan])
def test_a_step_missing_on_two_screens_does_not_block_the_owned_source_use(cls):
    rules, choice, carry, current = revealed_rules(cls=cls)
    unavailable = Frame("Click the introduction", act="click things", thing="introduction", order=-1)
    rules.frames[unavailable.key] = unavailable
    assert rules.the_step_to_do(['click "another screen"']) is None
    assert rules.refinement_of(MOVE, current) is None
    assert rules.the_step_to_do([MOVE, 'click "different screen"']) is choice
    assert rules.next_step() is unavailable and not rules.passed(unavailable)
    binding = binding_for(rules, current)
    rules.begin(MOVE, current, refinement=binding)
    receipt = rules.tried(MOVE, True, after=outcome(binding))
    assert receipt.state is PredicateState.SATISFIED
    assert rules.passed(choice) and rules.passed(carry) and not rules.passed(unavailable)
    assert rules.the_step_to_do(['click "introduction"']) is unavailable


def test_a_selector_reveal_can_bind_the_next_held_choice_after_an_unavailable_step():
    choice = Frame("Choose an item of component", act="click things", thing="an item of component", order=1)
    unavailable = Frame("Click the introduction", act="click things", thing="introduction")
    opener = Frame("Open the component library", act="click things", thing="component library", order=2)
    carry = Frame("Place the component in the workspace", act="carry", thing="component", where="workspace", order=3)
    rules = Rules(frames={frame.key: frame for frame in (unavailable, choice, opener, carry)})
    rules.missing_on[unavailable.key] = {1, 2}
    before = observation(1, ["component library"], object_visible=False)
    rules.observe_context(before)
    assert rules.reaches_next('click "component library"')
    current = observation(2, ["component library", "workspace"])
    rules.begin('click "component library"', before, step_key=opener.key)
    rules.tried('click "component library"', True, after=current)
    assert rules.revealed_choices is not None and rules.revealed_choices.for_step == choice.key
    assert rules.refinement_of(MOVE, current) is not None
    assert rules.step_of(f'click "{NAME}"') is choice
