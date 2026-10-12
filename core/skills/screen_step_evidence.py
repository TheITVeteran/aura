"""Bind procedural obligations to the action and observation that can answer them.

Pixel evidence stays private to the action path; procedure receipts contain only
detached JSON data. Input delivery, screen response and a specific effect are
separate facts, including when a plan is replaced while the input is in flight.
"""
from __future__ import annotations

import math
import time
from typing import Any

from core.runtime.skill_contract import PredicateOperator, PredicateState, SemanticPredicate
from core.verify.invariants import invariant


def owned_response_state(pending: dict[str, Any], observation: dict[str, Any]) -> PredicateState | None:
    """Bind streamed response evidence to this dispatch window, never an earlier input."""
    window = observation.get("answer_window")
    if not isinstance(window, dict):
        return PredicateState.UNKNOWN if "answer_window" in observation else None
    before = pending.get("observed_before") or {}
    epoch = observation.get("_capture_epoch")
    surface = observation.get("surface_id") or observation.get("scoped_to")
    before_at, after_at = before.get("capture_at"), observation.get("capture_at")
    completed = pending.get("dispatch_completed")
    sequence = window.get("sequence")
    preceding = pending.get("dispatch_sequence_before")
    if (not epoch or epoch == before.get("capture_epoch") or not before.get("capture_epoch")
            or not surface or surface != before.get("surface_id")
            or list(observation.get("bounds") or ()) != before.get("bounds")
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in (before_at, after_at, completed))
            or not before_at < completed < after_at
            or type(sequence) is not int or type(preceding) is not int or sequence != preceding + 1):
        return PredicateState.UNKNOWN
    start, end = pending.get("dispatch_started_monotonic"), pending.get("dispatch_completed_monotonic")
    at, finished, captured = window.get("started"), window.get("finished"), window.get("captured_through")
    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, end, at, finished, captured))
            or not start <= at <= finished <= end or captured < finished or window.get("delivered") is not True):
        return PredicateState.UNKNOWN
    answer = observation.get("answered")
    if answer is True:
        return PredicateState.SATISFIED
    if answer is False:
        return PredicateState.UNSATISFIED
    return PredicateState.UNKNOWN


def screen_evidence(observation: dict[str, Any], clickable: Any = ()) -> dict[str, Any]:
    """A small detached observation for deterministic procedure evaluation."""
    return {
        "text": str(observation.get("text") or ""),
        "clickable": [str(move) for move in clickable],
        "surface_id": str(observation.get("surface_id") or observation.get("scoped_to") or ""),
        "capture_epoch": str(observation.get("_capture_epoch") or ""),
        "work_epoch": str(observation.get("_work_epoch") or ""),
        "capture_at": observation.get("capture_at"),
        "bounds": list(observation.get("bounds") or ()),
        "regions": [{key: region.get(key) for key in ("text", "x", "y", "width", "height", "center_x", "center_y",
                                                     "role", "source", "candidate_id", "container_bbox", "capture_epoch")}
                    for region in (*tuple(observation.get("objects") or ()), *tuple(observation.get("layout") or ()),
                                   *tuple(observation.get("shapes") or ()))
                    if isinstance(region, dict)],
    }


@invariant("agency.responses_belong_to_one_delivered_dispatch", scope="agency",
           owner="core/skills/screen_step_evidence.py", observational=False)
def _response_dispatch_invariant() -> tuple:
    before = {"surface_id": "editor", "capture_epoch": "a", "capture_at": 1., "bounds": [0, 0, 400, 300]}
    pending = {"observed_before": before, "dispatch_completed": 2., "dispatch_started_monotonic": 10.,
               "dispatch_completed_monotonic": 11., "dispatch_sequence_before": 0}
    after = {"surface_id": "editor", "_capture_epoch": "b", "capture_at": 3., "bounds": [0, 0, 400, 300],
             "answered": True, "answer_window": {"sequence": 1, "started": 10.1, "finished": 10.9,
                                                "captured_through": 11.1, "delivered": True}}
    assert owned_response_state(pending, after) is PredicateState.SATISFIED
    assert owned_response_state(pending, {**after, "surface_id": "elsewhere"}) is PredicateState.UNKNOWN
    assert owned_response_state(pending, {**after, "answer_window": {**after["answer_window"], "sequence": 2}}) is PredicateState.UNKNOWN
    return ()


def transfer_effects() -> tuple[SemanticPredicate, ...]:
    return (SemanticPredicate(
        predicate_id="object_reached_destination", evidence_path="after.placement.state",
        operator=PredicateOperator.EQUALS, expected="verified",
        repair_hint="Observe the source and destination again; repair this transfer before continuing.",
    ),)


def current_controls(observation: dict[str, Any], drawn_where: Any, can_do: Any = None) -> tuple[str, ...]:
    """Give settlement and decision the same currently measured input candidates."""
    from core.agency.what_i_can_do_here import a_click_on
    from core.cognition.a_guide_to_a_place import THE_GUIDE
    from core.perception.drawn_objects import measured_objects

    from .screen_pursuit_bearings import things_to_click

    clickable = things_to_click(observation, drawn_where)
    guide = THE_GUIDE.get()
    procedures = (getattr(guide, "rules", None), getattr(guide, "plan", None))
    carrying = getattr(can_do, "carrying_said", False) or any(
        callable(asks := getattr(procedure, "asks_to_carry", None)) and asks()
        for procedure in procedures)
    if drawn_where and carrying:
        clickable = tuple(dict.fromkeys((*clickable, *(a_click_on(name) for name in measured_objects(observation)))))
    return clickable


def bind_step(run: Any) -> None:
    """Freeze the procedure instances and their own step binding before dispatch."""
    from core.agency.putting_things_in_place import what_is_carried
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    before = run.pending.get("observed_before") or {}
    effects = transfer_effects() if what_is_carried(run.key) else ()
    bindings = []
    procedures = [procedure for procedure in (getattr(guide, "rules", None), getattr(guide, "plan", None))
                  if procedure is not None]
    run.pending["selector_before"] = tuple((procedure, getattr(procedure, "selector_view", None))
                                           for procedure in procedures)
    if any(procedure.pending_step is not None for procedure in procedures):
        raise RuntimeError("the earlier delivered requirement still needs its observation")
    proposals = [(procedure, refine(run.key, before) if callable(refine := getattr(procedure, "refinement_of", None)) else None)
                 for procedure in procedures]
    refined = [(procedure, proposal) for procedure, proposal in proposals if proposal is not None]
    prepared = _prepare_refined_transfer(run, before, [proposal for _, proposal in refined]) if refined else None
    try:
        for procedure in procedures:
            proposal = next((proposal for owner, proposal in refined if owner is procedure), None)
            kwargs = {}
            if proposal is not None:
                kwargs["refinement"] = proposal.with_transfer(prepared.attempt_id, source_at=prepared.source_at,
                                                              destination_at=prepared.destination_at)
            attempt = procedure.begin(run.key, before, effects=effects, **kwargs)
            if attempt is not None:
                bindings.append((procedure, attempt))
                run.pending["step_bindings"] = tuple(bindings)
    except BaseException:
        abandon_bound_step(run.pending, run.key)
        raise
    run.pending["step_bindings"] = tuple(bindings)
    run.pending["step_settled"] = False
    run.pending["verification_started"] = time.monotonic()
    run.pending["verification_epochs"] = set()


def _prepare_refined_transfer(run: Any, before: dict[str, Any], proposals: list[Any]) -> Any:
    """Authorize a joint obligation only from this capture's verified source pixels."""
    from core.perception.drawn_objects import measured_objects
    from core.perception.observed_transfer import bind_transfer

    from .screen_pursuit_bearings import where_to_click

    current = run.observation
    if (before.get("capture_epoch") != current.get("_capture_epoch")
            or before.get("capture_at") != current.get("capture_at")
            or before.get("surface_id") != str(current.get("surface_id") or current.get("scoped_to") or "")
            or tuple(before.get("bounds") or ()) != tuple(current.get("bounds") or ())
            or str(before.get("work_epoch") or "") != str(current.get("_work_epoch") or "")):
        raise ValueError("a refined action needs the same capture used to bind its obligations")
    sources = measured_objects(current)
    first = proposals[0]
    source = sources.get(first.source)
    if source is None:
        raise ValueError("the refined source has no current measured pixels")
    witness = (source["candidate_id"], tuple(source[key] for key in ("x", "y", "width", "height")),
               tuple(source.get("container_bbox") or ()))
    if any(proposal.move != run.key or proposal.source != first.source or proposal.destination != first.destination
           or proposal.source_witness != witness or proposal.surface_id != before.get("surface_id")
           or proposal.bounds != tuple(before.get("bounds") or ())
           or proposal.capture_epoch != before.get("capture_epoch") or proposal.capture_at != before.get("capture_at")
           or proposal.work_epoch != str(before.get("work_epoch") or "") for proposal in proposals):
        raise ValueError("the refined obligations disagree with the measured source")
    start, end = where_to_click(current, first.source), where_to_click(current, first.destination)
    if start is None or end is None:
        raise ValueError("a refined transfer requires both current endpoints")
    intent = bind_transfer(run.key, current, source=first.source, destination=first.destination,
                           source_at=start, destination_at=end)
    if intent.source_at is None or intent.destination_at is None:
        raise ValueError("a refined transfer needs bounded endpoint coordinates")
    run.pending["prepared_transfer"] = intent
    return intent


def abandon_bound_step(pending: dict[str, Any], move: str) -> None:
    """Release owned obligations after refused or uncertain delivery, without success credit."""
    try:
        for procedure, attempt in pending.get("step_bindings", ()):
            if procedure.pending_step is attempt:
                try:
                    procedure.tried(move, None, before=attempt.before.evidence, after={}, step_key=attempt.step_key)
                except (RuntimeError, TypeError, ValueError) as exc:
                    from core.runtime.errors import record_degradation

                    record_degradation("screen_pursuit", exc, severity="info",
                                       action="released an unmeasured procedure attempt after failed delivery")
                finally:
                    if procedure.pending_step is attempt:
                        procedure.pending_step = None
                        procedure.pending_refinement = None
    finally:
        pending["step_bindings"] = ()
        pending["step_settled"] = True
        pending.pop("prepared_transfer", None)


def current_work_is_measured(can_do: Any, observation: dict[str, Any]) -> bool:
    """Revalidate retained work on this capture before using it for completion."""
    from core.agency.putting_things_in_place import PuttingInPlace

    if not isinstance(can_do, PuttingInPlace):
        return False
    can_do.observe_placements(observation, independent=True)
    return can_do.placed_count() > 0


@invariant("agency.creation_completion_requires_current_work", scope="agency",
           owner="core/skills/screen_step_evidence.py", observational=False)
def _creation_completion_invariant() -> tuple:
    from core.agency.what_i_can_do_here import WhatWorksHere

    remembered = WhatWorksHere(carried_to={'drag "part" to "canvas"': True},
                               did_something={'click "Next background"': 500})
    assert not current_work_is_measured(remembered, {"text": "An account says it was made"})
    assert not current_work_is_measured(None, {"text": "Create your own scene"})
    return ()


def artifact_ready_for(move: str, can_do: Any = None) -> bool:
    """A construction task's execution controls need currently measured work, including unmatched controls."""
    from core.agency.what_i_can_do_here import what_is_clicked
    from core.cognition.a_guide_to_a_place import THE_GUIDE
    from core.cognition.procedure_binding import consumes_artifact

    label = what_is_clicked(move)
    guide = THE_GUIDE.get()
    procedures = [procedure for procedure in (getattr(guide, "rules", None), getattr(guide, "plan", None))
                  if procedure is not None]
    constructing = any(frame.act == "carry" for procedure in procedures for frame in procedure.steps())
    if not constructing or not label or not consumes_artifact(label):
        return True
    measured = getattr(can_do, "placed_count", None)
    return callable(measured) and measured() > 0


@invariant("agency.construction_execution_requires_current_placement", scope="agency",
           owner="core/skills/screen_step_evidence.py", observational=False)
def _construction_readiness_invariant() -> tuple:
    from core.agency.what_i_can_do_here import WhatWorksHere
    from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide
    from core.cognition.reading_the_rules import Frame, Rules

    step = Frame("Place a component in the workspace", act="carry", thing="component", where="workspace")
    guide = Guide(rules=Rules(frames={step.key: step}), built="An old account says it was built")
    can_do = WhatWorksHere(carried_to={'drag "component" to "workspace"': True})
    token = THE_GUIDE.set(guide)
    try:
        assert not artifact_ready_for('click "Run"', can_do)
        assert not artifact_ready_for('click "Test circuit"', can_do)
        assert artifact_ready_for('click "Library"', can_do)
    finally:
        THE_GUIDE.reset(token)
    return ()


def admissible_step(move: str, *, reaching: Any = (), can_do: Any = None) -> bool:
    """Known later requirements wait for their prerequisites, independent of scores."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE
    from core.cognition.reading_the_rules import the_step_that_holds

    guide = THE_GUIDE.get()
    if not artifact_ready_for(move, can_do):
        return False
    for procedure in (getattr(guide, "rules", None), getattr(guide, "plan", None)):
        if procedure is None:
            continue
        if procedure.pending_step is not None:
            return False
        step, needed = procedure.step_of(move), the_step_that_holds(procedure)
        if (step is not None and needed is not None and step.order > needed.order and move not in reaching
                and not procedure.reaches_next(move)):
            return False
    return True


def settle_step(pending: dict[str, Any], observation: dict[str, Any], clickable: Any, can_do: Any = None) -> bool:
    """Answer delivered obligations once, before the screen can propose a new plan."""
    if pending.get("step_settled"):
        return True
    previous = pending.get("deliberation")
    chosen = getattr(previous, "chosen", None)
    if chosen is None:
        return True
    after = screen_evidence(observation, clickable)
    before = pending.get("observed_before") or {}
    intent = pending.get("transfer_intent")
    effects = transfer_effects() if intent is not None else ()
    if intent is not None:
        from core.perception.observed_transfer import verify_transfer

        receipt = verify_transfer(intent, observation)
        pending["transfer_receipt"] = receipt
        if receipt.state is PredicateState.UNKNOWN:
            epochs = pending.setdefault("verification_epochs", set())
            if after.get("capture_epoch") and after["capture_epoch"] != before.get("capture_epoch"):
                epochs.add(after["capture_epoch"])
            if len(epochs) < 2 and time.monotonic() - pending.get("verification_started", 0.0) < 30.0:
                return False
        if can_do is not None:
            can_do.carried(chosen.name, True, receipt=receipt)
        if receipt.state is not PredicateState.UNKNOWN:
            after["placement"] = {"state": "verified" if receipt.state is PredicateState.SATISFIED else "absent"}
        prepared = pending.get("prepared_transfer")
        if (prepared is not None and intent.attempt_id == prepared.attempt_id and intent.move == prepared.move
                and intent.before is prepared.before and intent.source_at == prepared.source_at
                and intent.destination_at == prepared.destination_at and receipt.attempt_id == prepared.attempt_id
                and receipt.state is PredicateState.SATISFIED):
            from core.agency.putting_things_in_place import what_is_carried

            source, destination = what_is_carried(prepared.move)
            # Publish only the physical receipt of the source-use action bound
            # before dispatch. A different successful carry cannot settle it.
            after["refinement"] = {"state": "verified", "attempt_id": prepared.attempt_id, "move": prepared.move,
                                   "source": source, "destination": destination,
                                   "source_at": list(prepared.source_at), "destination_at": list(prepared.destination_at)}
    # Response cannot be claimed across a different surface or an old capture.
    fresh = (bool(before.get("capture_epoch"))
             and bool(after.get("capture_epoch"))
             and before.get("capture_epoch") != after.get("capture_epoch")
             and bool(before.get("surface_id")) and before.get("surface_id") == after.get("surface_id")
             and before.get("bounds") == after.get("bounds"))
    before_at, after_at = before.get("capture_at"), after.get("capture_at")
    fresh = fresh and (type(before_at) in (int, float) and type(after_at) in (int, float)
                       and before_at < after_at and after_at > pending.get("dispatch_completed", float("inf")))
    changed = (before.get("text") != after.get("text")
               or before.get("clickable") != after.get("clickable")
               or before.get("regions") != after.get("regions")
               or after.get("placement", {}).get("state") == "verified") if fresh else None
    response = owned_response_state(pending, observation)
    if (fresh and response is PredicateState.UNKNOWN and intent is None
            and time.monotonic() - pending.get("verification_started", 0.0) < 3.0):
        # An early still is not a measured failure. Allow the stream's bounded
        # response interval to finish before releasing this unobserved trial.
        return False
    if fresh and response is not None and intent is None:
        changed = None if response is PredicateState.UNKNOWN else response is PredicateState.SATISFIED
    if intent is not None and pending["transfer_receipt"].state is PredicateState.UNKNOWN:
        changed = None
    elif response is None and after.get("placement", {}).get("state") != "verified" and changed is not None:
        from .screen_pursuit_on_a_page import it_was_answered

        changed = it_was_answered(changed, observation)
    if changed is True and fresh and response is PredicateState.SATISFIED:
        from core.cognition.selector_view import SelectorView

        watched = pending.get("watched") or {}
        for procedure, view in pending.get("selector_before", ()):
            if (isinstance(view, SelectorView) and procedure.selector_view is view
                    and view.closed_by(chosen.name, watched, observation)):
                procedure.selector_view = None
                procedure.revealed_choices = None
    for procedure, attempt in pending.get("step_bindings", ()):
        procedure.tried(chosen.name, changed, before=before, after=after,
                        step_key=attempt.step_key, effects=effects)
    pending["step_settled"] = True
    pending["step_after"] = after
    return True


def trial_context(pending: dict[str, Any], observation: dict[str, Any], clickable: Any) -> dict[str, Any]:
    return {"before": pending.get("observed_before") or {},
            "after": pending.get("step_after") or screen_evidence(observation, clickable),
            "transfer": pending.get("transfer_receipt")}
