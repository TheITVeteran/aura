"""On a person's turn her state steers the forward pass at the strength her checkpoint has earned.

Her model is handed no system text (core/brain/what_she_is_given.py). Live on
13 October, with that text gone, the worker's line read "steering α=0.0
(engine α=5.3754)": her mind asked for zero on every reply, from a comment
older than the certificate that had since been earned. So nothing of her state
reached the reply at all, in text or in the computation.

The worker looks the certificate up where the hooks are and reports what it
found on every job. Her mind asks for that amount.
"""
from __future__ import annotations

import pytest

from core.brain.cognitive_engine import _live_mind_generation_controls
from core.brain.llm import mlx_worker
from core.brain.llm.mlx_client import _sanitize_surface_control_receipt
from core.brain.llm.mlx_worker_surface_quality import _apply_surface_generation_controls
from core.consciousness.fusion_certificate import earned_on_the_worker, worker_reported
from tests.test_fusion_gate_in_worker import _point_the_lookup_at, certified  # noqa: F401
from tests.test_live_mind_generation_controls import _ready_live_mind_context


@pytest.fixture(autouse=True)
def _nothing_reported_yet():
    worker_reported(0.0)
    yield
    worker_reported(0.0)


def _asked(**turn) -> float:
    return _live_mind_generation_controls(_ready_live_mind_context(), **turn)[
        "clean_user_surface_steering_alpha"
    ]


def test_until_a_worker_has_reported_she_asks_for_none():
    assert earned_on_the_worker() == 0.0
    assert _asked() == 0.0


def test_she_asks_for_what_the_worker_reported_it_has_earned():
    worker_reported(0.2)
    assert _asked() == pytest.approx(0.2)
    worker_reported(0.1)
    assert _asked() == pytest.approx(0.1)


def test_a_turn_that_has_to_be_worked_out_exactly_is_not_steered():
    worker_reported(0.2)
    assert _asked(user_message="What is 4817 * 392?") == 0.0


@pytest.mark.parametrize("reported", [None, "", "a lot", True, float("nan"), float("inf"), -0.3])
def test_a_report_that_cannot_be_read_is_zero(reported):
    assert worker_reported(reported) == 0.0


def test_a_report_above_the_whole_stream_is_held_to_it():
    assert worker_reported(7.0) == 1.0


def test_the_worker_reports_what_it_has_earned_on_a_reply_somebody_waits_for(certified):  # noqa: F811
    certificate, engine = certified
    engine._alpha = None
    engine._hooks = []
    state = _apply_surface_generation_controls(
        engine, object(), {"clean_user_surface_contract": True, "clean_user_surface_steering_alpha": certificate.alpha}
    )
    assert state["surface_alpha_certified"] == pytest.approx(certificate.alpha)
    assert state["surface_alpha_applied"] == pytest.approx(certificate.alpha)


def test_the_worker_reports_it_on_a_job_nobody_waits_for_too(certified):  # noqa: F811
    """Her first reply after a start can ask, because an earlier job said."""
    certificate, engine = certified
    state = _apply_surface_generation_controls(engine, object(), {"allow_full_affective_steering": True})
    assert state == {"enabled": False, "surface_alpha_certified": pytest.approx(certificate.alpha)}


def test_a_job_with_no_steering_engine_reports_nothing(monkeypatch, tmp_path):
    """Its zero is the absence of anything to look up with, and would erase a real report."""
    _point_the_lookup_at(monkeypatch, tmp_path)
    monkeypatch.setattr(mlx_worker, "_FUSION_MODEL_IDENTITY", "d" * 64)
    assert _apply_surface_generation_controls(None, object(), {"allow_full_affective_steering": True}) == {
        "enabled": False
    }
    state = _apply_surface_generation_controls(None, object(), {"clean_user_surface_contract": True})
    assert "surface_alpha_certified" not in state


def test_a_reply_nobody_set_a_strength_for_gets_what_the_checkpoint_has_earned(certified):  # noqa: F811
    """Live, 13 October: the client sent the key with nothing in it and it read as zero."""
    certificate, engine = certified
    engine._alpha = None
    engine._hooks = []
    job = {
        "clean_user_surface_contract": True,
        "clean_user_surface_steering_alpha": None,
        "user_surface_validation_prompt": "There is a wren nesting in the old boot by my back door.",
    }
    state = _apply_surface_generation_controls(engine, object(), job)
    assert state["surface_alpha_applied"] == pytest.approx(certificate.alpha)


def test_and_none_when_what_was_asked_has_one_right_answer(certified):  # noqa: F811
    _certificate, engine = certified
    engine._alpha = None
    engine._hooks = []
    job = {
        "clean_user_surface_contract": True,
        "clean_user_surface_steering_alpha": None,
        "user_surface_validation_prompt": "What is 17 minus 8, and then times 3?",
    }
    assert _apply_surface_generation_controls(engine, object(), job)["surface_alpha_applied"] == 0.0


def test_a_request_for_zero_is_still_zero(certified):  # noqa: F811
    _certificate, engine = certified
    engine._alpha = None
    engine._hooks = []
    job = {"clean_user_surface_contract": True, "clean_user_surface_steering_alpha": 0.0}
    assert _apply_surface_generation_controls(engine, object(), job)["surface_alpha_applied"] == 0.0


def test_the_report_crosses_from_the_worker_to_her_mind():
    kept = _sanitize_surface_control_receipt({"enabled": True, "surface_alpha_certified": 0.2, "not_a_field": 1})
    assert kept == {"enabled": True, "surface_alpha_certified": 0.2}
