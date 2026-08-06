"""A reply must say which architecture produced it.

The quick lane, a canonical pre-rendered floor, the full phase pipeline and
reactive recovery all emit fluent text with a self-described response_path.
Only one of them is the cognitive architecture a demo is taken to demonstrate.
These tests pin the difference to something derived rather than declared.
"""

from __future__ import annotations

import asyncio

import pytest

from core.verify.turn_receipt import (
    record_model_generation,
    record_phase,
    record_response_path,
    recent_receipts,
    recording_turn,
    reset_turn_receipts_for_test,
    current_receipt,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_turn_receipts_for_test()
    yield
    reset_turn_receipts_for_test()


PHASES = ["Perception", "Affect", "Qualia", "Workspace", "Planning"]


def test_a_turn_that_ran_every_phase_says_so():
    with recording_turn("t", phases_available=PHASES) as receipt:
        for phase in PHASES:
            record_phase(phase)
        record_response_path("full_phase_pipeline", model_generation=True)

    assert receipt.full_pipeline_ran
    assert receipt.phases_skipped == ()
    assert receipt.coverage == 1.0


def test_the_quick_lane_cannot_report_a_full_pipeline():
    """The finding this file exists for.

    _direct_desktop_quick_reply returns before a single phase executes. The
    reply is fine; what it is not is evidence that affect, qualia, Phi, the
    workspace or planning had anything to do with it.
    """

    with recording_turn("t", phases_available=PHASES) as receipt:
        record_response_path("desktop_quick_reply", model_generation=True)

    assert not receipt.full_pipeline_ran
    assert set(receipt.phases_skipped) == set(PHASES)
    assert receipt.coverage == 0.0
    assert receipt.model_generation is True


def test_a_canonical_floor_records_that_the_model_never_ran():
    """Pre-rendered text is not a model output and must not read like one."""

    with recording_turn("t", phases_available=PHASES) as receipt:
        record_response_path(
            "cognitive_engine_self_condition_grounding", model_generation=False
        )

    payload = receipt.as_dict()
    assert payload["model_generation"] is False
    assert payload["full_pipeline_ran"] is False
    assert payload["response_path"] == "cognitive_engine_self_condition_grounding"


def test_full_pipeline_is_derived_and_cannot_be_asserted():
    with recording_turn("t", phases_available=PHASES) as receipt:
        record_response_path("full_phase_pipeline", model_generation=True)
        # Naming the path is not the same as having run it.
        assert not receipt.full_pipeline_ran
        for phase in PHASES[:-1]:
            record_phase(phase)
        assert not receipt.full_pipeline_ran, "one missing phase is not a full pipeline"
        record_phase(PHASES[-1])
        assert receipt.full_pipeline_ran


def test_a_partial_pipeline_names_what_it_skipped():
    with recording_turn("t", phases_available=PHASES) as receipt:
        record_phase("Perception")
        record_phase("Affect")

    assert receipt.phases_skipped == ("Qualia", "Workspace", "Planning")
    assert 0.0 < receipt.coverage < 1.0


def test_an_engine_with_no_phases_is_never_a_full_pipeline():
    """Zero of zero is not completeness, and vacuous truth would report success."""

    with recording_turn("t", phases_available=[]) as receipt:
        record_response_path("desktop_quick_reply", model_generation=True)
    assert not receipt.full_pipeline_ran


def test_an_unresolved_path_is_visible_rather_than_defaulted():
    with recording_turn("t", phases_available=PHASES) as receipt:
        pass
    assert receipt.response_path == "unresolved"
    assert receipt.model_generation is False


def test_the_receipt_closes_even_when_the_turn_raises():
    with pytest.raises(RuntimeError):
        with recording_turn("t", phases_available=PHASES):
            record_phase("Perception")
            raise RuntimeError("phase blew up")

    logged = recent_receipts()
    assert len(logged) == 1
    assert logged[0]["phases_executed"] == ["Perception"]
    assert logged[0]["full_pipeline_ran"] is False
    assert current_receipt() is None


def test_concurrent_turns_do_not_write_into_each_others_receipts():
    async def turn(name: str, phases: list[str]) -> dict:
        with recording_turn(name, phases_available=PHASES) as receipt:
            for phase in phases:
                record_phase(phase)
                await asyncio.sleep(0.01)
            record_response_path(f"path_{name}", model_generation=True)
            return receipt.as_dict()

    async def main():
        return await asyncio.gather(
            turn("a", PHASES),
            turn("b", ["Perception"]),
        )

    first, second = asyncio.run(main())
    assert first["full_pipeline_ran"] is True
    assert second["full_pipeline_ran"] is False
    assert second["phases_executed"] == ["Perception"]
    assert first["response_path"] == "path_a"
    assert second["response_path"] == "path_b"


def test_recording_outside_a_turn_is_harmless():
    record_phase("Perception")
    record_response_path("nowhere", model_generation=True)
    record_model_generation()
    assert recent_receipts() == []


def test_recent_receipts_are_bounded():
    for index in range(200):
        with recording_turn(str(index), phases_available=PHASES):
            record_response_path("desktop_quick_reply", model_generation=True)
    assert len(recent_receipts(limit=1000)) <= 64


# ---------------------------------------------------------------------------
# Through the real engine
# ---------------------------------------------------------------------------


def test_engine_attaches_a_path_receipt_to_the_thought():
    """End to end: the evidence travels with the answer."""

    from core.brain.cognitive_engine import _attach_turn_receipt

    class Bare:
        metadata: dict = {}

    thought = Bare()
    thought.metadata = {}
    with recording_turn("t", phases_available=PHASES) as receipt:
        record_response_path("desktop_quick_reply", model_generation=True)
    _attach_turn_receipt(thought, receipt)

    assert "turn_receipt" in thought.metadata
    assert thought.metadata["turn_receipt"]["full_pipeline_ran"] is False
    assert thought.metadata["turn_receipt"]["response_path"] == "desktop_quick_reply"


def test_attaching_to_a_thought_without_metadata_is_harmless():
    from core.brain.cognitive_engine import _attach_turn_receipt

    class NoMetadata:
        pass

    with recording_turn("t", phases_available=PHASES) as receipt:
        pass
    _attach_turn_receipt(NoMetadata(), receipt)  # must not raise
