"""Contract tests: incremental FINAL_ANSWER completion detection (CP180).

The early-stop predicate must fire exactly when a single marker's JSON
object closes and parses — never on open objects, prose, multiple markers,
or non-object payloads — and the full-state analysis must agree with the
strict parser on validity so decode receipts and scoring share one truth.
"""
from __future__ import annotations

from core.brain.llm.latent_cortex.answer_contract import (
    contract_answer_state,
    is_contract_complete,
)


def test_no_marker_is_incomplete():
    state = contract_answer_state("Let me reason step by step about this.")
    assert state["marker_count"] == 0
    assert not state["complete"] and not state["valid"]
    assert not is_contract_complete("plain reasoning text")


def test_open_object_is_incomplete_until_it_closes():
    partial = 'Reasoning done.\nFINAL_ANSWER: {"node": 6, "path": [1, '
    assert not is_contract_complete(partial)
    assert contract_answer_state(partial)["reason"] == "object_open"
    closed = partial + "2]}"
    state = contract_answer_state(closed)
    assert state["complete"] and state["valid"]
    assert state["parsed"] == {"node": 6, "path": [1, 2]}
    assert is_contract_complete(closed)


def test_braces_inside_strings_do_not_close_the_object():
    tricky = 'FINAL_ANSWER: {"expr": "f(x) = {x}", "value'
    assert not is_contract_complete(tricky)
    done = tricky + '": 3}'
    state = contract_answer_state(done)
    assert state["complete"] and state["valid"]
    assert state["parsed"]["expr"] == "f(x) = {x}"


def test_escaped_quotes_inside_strings_are_handled():
    text = 'FINAL_ANSWER: {"quote": "she said \\"hi\\" today", "n": 1}'
    state = contract_answer_state(text)
    assert state["complete"] and state["valid"]
    assert state["parsed"]["n"] == 1


def test_nested_objects_complete_only_at_top_level_close():
    partial = 'FINAL_ANSWER: {"outer": {"inner": {"deep": 1}}'
    assert not is_contract_complete(partial)
    assert is_contract_complete(partial + "}")


def test_multiple_markers_never_complete():
    text = 'FINAL_ANSWER: {"a": 1}\nFINAL_ANSWER: {"a": 2}'
    state = contract_answer_state(text)
    assert state["marker_count"] == 2
    assert not state["complete"]
    assert state["reason"] == "multiple_markers"


def test_non_object_payload_never_completes():
    assert not is_contract_complete("FINAL_ANSWER: 42")
    assert not is_contract_complete('FINAL_ANSWER: [1, 2, 3]')
    assert (
        contract_answer_state("FINAL_ANSWER: 42")["reason"]
        == "marker_line_has_no_object"
        or not contract_answer_state("FINAL_ANSWER: 42")["complete"]
    )


def test_trailing_content_after_object_is_complete_but_invalid():
    """Stopping AT completion prevents this shape; when it already exists
    (the model kept talking), completeness is true but the strict parser's
    terminality rule makes it invalid — and the state says both."""
    text = 'FINAL_ANSWER: {"node": 6}\nWait, actually let me reconsider.'
    state = contract_answer_state(text)
    assert state["complete"] is True
    assert state["valid"] is False
    assert state["reason"].startswith("parser_rejected")


def test_invalid_json_in_closed_braces_does_not_complete():
    text = "FINAL_ANSWER: {'node': 6}"  # single quotes: not strict JSON
    state = contract_answer_state(text)
    assert not state["complete"]
    assert state["reason"] == "object_invalid_json"


def test_marker_line_break_before_object_is_rejected():
    text = "FINAL_ANSWER:\nnow thinking about the object {\"a\": 1}"
    state = contract_answer_state(text)
    assert not state["complete"]
    assert state["reason"] == "marker_line_has_no_object"


def test_agreement_with_strict_parser_on_valid_answers():
    from core.brain.llm.latent_cortex.frontier_tasks import parse_final_answer

    text = 'Steps done.\nFINAL_ANSWER: {"value": 1}'
    state = contract_answer_state(text)
    assert state["valid"] is True
    assert state["parsed"] == parse_final_answer(text)


def test_streaming_simulation_stops_exactly_once_at_completion():
    full = 'Thinking...\nFINAL_ANSWER: {"count": 2, "witness": [1, 5]}'
    fire_points = [
        index
        for index in range(1, len(full) + 1)
        if is_contract_complete(full[:index])
    ]
    # Fires at the closing brace character and stays fired to the end.
    assert fire_points
    assert fire_points[0] == len(full)  # the final char IS the close brace
    assert fire_points == list(range(fire_points[0], len(full) + 1))
