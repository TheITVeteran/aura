"""A bandit that only sees its wins cannot learn which arm loses.

CP126 d4a5bb97 on core/brain/latent_cortex_service.py. Controller learning
sat entirely inside the success path: every failed, timed-out,
invalid-receipt and quality-rejected episode returned through
``_record_failure`` without recording an outcome at all.

So the execution controller's statistics were drawn only from the episodes
that worked. An expensive arm that fails most of the time looked, to the
component whose entire job is choosing between arms, exactly like an
expensive arm that always works — because its failures were never counted
and its occasional successes were.

These tests drive the real refusal path on a real service instance and read
the real controller. No source-text assertions: the previous version of this
fix "passed" an AST check while the behaviour was unchanged.
"""
from __future__ import annotations

import pytest

from core.brain.latent_cortex_service import LatentCortexService


class _RecordingController:
    """Stands in for the execution controller, recording what it is taught."""

    def __init__(self) -> None:
        self.outcomes: list[dict] = []

    def record_outcome(self, **kwargs) -> bool:
        self.outcomes.append(kwargs)
        return True


@pytest.fixture
def service():
    return LatentCortexService.__new__(LatentCortexService)


@pytest.fixture
def controller(monkeypatch):
    recorder = _RecordingController()
    monkeypatch.setattr(
        "core.brain.llm.latent_cortex.execution_controller.get_execution_controller",
        lambda: recorder,
    )
    return recorder


def _reset(service) -> None:
    service._failure_streak = 0
    service._last_refusal = ""
    service._ok_episodes = 0
    service._last_success_at = 0.0
    service._controller_outcome_recorded_for = None


def _prepare(service, controller, *, decision_id: str = "dec-1") -> None:
    _reset(service)
    service._last_allocation = {
        "execution_controller": {
            "decision_id": decision_id,
            "bucket": "hard/reasoning",
            "arm": "recurrent_depth_8",
        }
    }


def test_a_refused_episode_teaches_the_controller_its_arm_failed(service, controller):
    _prepare(service, controller)

    service._record_failure("generation_gate_busy")

    assert controller.outcomes, (
        "the episode failed and the bandit was told nothing; the arm's "
        "statistics still contain only its successes"
    )
    outcome = controller.outcomes[0]
    assert outcome["success"] is False
    assert outcome["arm"] == "recurrent_depth_8"
    assert outcome["bucket"] == "hard/reasoning"
    assert outcome["decision_id"] == "dec-1"


def test_the_failure_is_not_graded_as_an_answer(service, controller):
    """An execution failure is evidence about the arm, not about correctness.

    ``checked=True`` would claim an independent verifier graded this answer
    and found it wrong. Nothing graded it — it never produced one.
    """
    _prepare(service, controller)

    service._record_failure("model_unavailable")

    assert controller.outcomes[0]["checked"] is False
    assert controller.outcomes[0]["verified_score"] == 0.0


def test_one_decision_records_one_outcome(service, controller):
    """Refusal paths nest; a doubly-counted loss is as wrong as an uncounted one."""
    _prepare(service, controller)

    service._record_failure("generation_lease_unavailable:TimeoutError")
    service._record_failure("adaptive_compute_enforcement_failed")

    assert len(controller.outcomes) == 1


def test_a_new_decision_is_recorded_on_its_own_merits(service, controller):
    """De-duplication is per decision, not a one-shot latch for the process."""
    _prepare(service, controller, decision_id="dec-1")
    service._record_failure("first episode failed")

    _prepare(service, controller, decision_id="dec-2")
    service._last_allocation["execution_controller"]["arm"] = "base"
    service._record_failure("second episode failed too")

    assert [row["decision_id"] for row in controller.outcomes] == ["dec-1", "dec-2"]
    assert [row["arm"] for row in controller.outcomes] == ["recurrent_depth_8", "base"]


def test_a_refusal_before_any_arm_was_chosen_records_nothing(service, controller):
    """Pre-dispatch refusals have no arm to blame; inventing one is a false trial."""
    _reset(service)
    service._last_allocation = {}

    service._record_failure("latent_cortex_disabled")

    assert controller.outcomes == []


def test_a_decision_without_an_id_records_nothing(service, controller):
    """An outcome that cannot be bound to its decision can credit any arm."""
    _reset(service)
    service._last_allocation = {"execution_controller": {"bucket": "b", "arm": "a"}}

    service._record_failure("invalid_decode_token_override")

    assert controller.outcomes == []


def test_the_refusal_still_returns_its_receipt_when_the_controller_explodes(
    service, monkeypatch
):
    """Teaching the bandit is bookkeeping. It must never take the refusal down."""

    class _Broken:
        def record_outcome(self, **kwargs):
            raise RuntimeError("controller store is unavailable")

    monkeypatch.setattr(
        "core.brain.llm.latent_cortex.execution_controller.get_execution_controller",
        lambda: _Broken(),
    )
    _prepare(service, _Broken())

    receipt = service._record_failure("generation_gate_busy", stage="dispatch")

    assert receipt["ok"] is False
    assert receipt["reason"] == "generation_gate_busy"
    assert receipt["refusal_receipt"]["stage"] == "dispatch"


def test_the_bookkeeping_key_never_reaches_a_published_receipt(service, controller):
    """The decision dict is spread into receipts; it must stay clean."""
    _prepare(service, controller)

    service._record_failure("generation_gate_busy")

    decision = service._last_allocation["execution_controller"]
    assert "_outcome_recorded" not in decision
    assert set(decision) == {"decision_id", "bucket", "arm"}
