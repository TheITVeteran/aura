"""Tests that ResearchLab does REAL falsification, not confirmation theatre.

The old pipeline rigged a guaranteed-positive effect so every hypothesis validated and
none could ever be refuted. These tests pin the fix: a true checkable claim validates,
a FALSE one is refuted with a counterexample (impossible before), an unverifiable topic
is honestly inconclusive (never fabricated-validated), and no result is fabricated.
"""
from __future__ import annotations

import inspect

import pytest

from core.lab.experiment_designer import ExperimentDesigner
from core.lab.hypothesis_engine import HypothesisEngine
from core.lab.result_interpreter import ResultInterpreter
from core.lab.simulation_runner import SimulationRunner


def _spec(claim: str) -> dict:
    return {"name": "exp_test", "hypothesis_id": "h1", "claim": claim}


async def test_true_claim_validates_via_real_verifier():
    result = await SimulationRunner().run_sim(_spec("is n^5 - n divisible by 30 for all n"))
    assert result["status"] == "proven"
    assert result["validated"] is True
    assert result["fabricated"] is False
    assert result["method"] == "exact_falsification"


async def test_false_claim_is_refuted_with_counterexample():
    """The rigged runner could NEVER produce this outcome — refutation is the proof of fix."""
    result = await SimulationRunner().run_sim(_spec("is n^5 - n divisible by 7 for all n"))
    assert result["status"] == "refuted"
    assert result["refuted"] is True
    assert result["validated"] is False
    assert result["counterexample"] == 2


async def test_unverifiable_topic_is_inconclusive_not_fabricated():
    result = await SimulationRunner().run_sim(_spec("dark matter is made of axions"))
    assert result["validated"] is False
    assert result["inconclusive"] is True
    assert result["fabricated"] is False


async def test_no_randomness_in_runner_source():
    """Guard against regression to the rigged random-noise simulator."""
    src = inspect.getsource(SimulationRunner)
    assert "random" not in src
    assert "Simulate a positive effect" not in src


def test_interpreter_can_refute():
    interp = ResultInterpreter()
    hyp = HypothesisEngine().generate_hypothesis("n^5 - n divisibility")
    refuted = interp.interpret(hyp, {
        "status": "refuted", "validated": False, "refuted": True,
        "counterexample": 2, "rendered": "Counterexample at n=2.",
    })
    assert refuted["validated"] is False and refuted["refuted"] is True
    assert refuted["new_confidence"] == 0.0
    assert "REFUTED" in refuted["conclusion"]


def test_interpreter_inconclusive_is_not_validated():
    interp = ResultInterpreter()
    hyp = HypothesisEngine().generate_hypothesis("vague topic")
    out = interp.interpret(hyp, {"status": "conjecture", "validated": False, "refuted": False})
    assert out["validated"] is False
    assert "Inconclusive" in out["conclusion"] or "inconclusive" in out["conclusion"].lower()


def test_designer_carries_checkable_claim():
    hyp = HypothesisEngine().generate_hypothesis("topic")
    spec = ExperimentDesigner().design_experiment(hyp, [], claim="is n^3 - n divisible by 6")
    assert spec["claim"] == "is n^3 - n divisible by 6"
    assert spec["method"] == "exact_falsification"


async def test_full_research_cycle_refutes_false_topic():
    from core.lab.research_lab import ResearchLab

    lab = ResearchLab()
    out = await lab.execute_cycle("is n^5 - n divisible by 7 for all n")
    assert out["ok"] is True
    assert out["refuted"] is True
    assert out["validated"] is False
    assert out["fabricated"] is False
