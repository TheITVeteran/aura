"""tests/test_lesion_inference.py — a large delta is not a strong claim.

Operationally: this measures whether a lesion result is classified by what it
can support — wiring, mechanism, or capability — rather than by the size of its
delta.

The case that motivates it is in this repository's own scorecard.
`without_system2` reports 1.000 → 0.000 on `strict_proof_exact_answer_rate`,
and the lesioned component IS the strict-proof solver. Removing the only thing
that can emit an answer in that format and observing the format stop appearing
is a true statement about wiring. The number is large, real, correctly
measured, and inferentially almost empty — the worst combination available,
because size reads as strength.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.evaluation.lesion_inference import (
    LICENSE,
    InferenceClass,
    LesionClaim,
    summarise,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORECARD = PROJECT_ROOT / "artifacts" / "ablation" / "ablation_scorecard.json"


def _claim(**kwargs) -> LesionClaim:
    defaults = dict(
        condition="without_x",
        subsystem="core.x",
        metric_name="x_rate",
        delta=1.0,
        metric_has_other_producers=True,
        metric_is_task_success=False,
        tasks_solvable_without_component=False,
    )
    defaults.update(kwargs)
    return LesionClaim(**defaults)


def test_a_metric_only_the_component_produces_is_tautological_at_any_delta():
    """The System 2 case. A perfect 1.0 that could not have been anything else."""
    claim = _claim(
        condition="without_system2",
        metric_name="strict_proof_exact_answer_rate",
        delta=1.0,
        metric_has_other_producers=False,
        metric_is_task_success=True,
    )

    assert claim.inference_class is InferenceClass.TAUTOLOGICAL
    assert claim.supports_earns_its_cost is False


def test_task_success_over_tasks_only_this_component_can_do_is_not_capability():
    """A tautology wearing a task battery is still a tautology.

    This is the subtle one: dressing a wiring check in the vocabulary of task
    success is exactly how a scorecard starts sounding like a capability
    result.
    """
    claim = _claim(
        metric_is_task_success=True,
        metric_has_other_producers=True,
        tasks_solvable_without_component=False,
    )
    assert claim.inference_class is InferenceClass.MECHANISTIC
    assert claim.supports_earns_its_cost is False


def test_capability_requires_task_success_on_tasks_solvable_without_it():
    claim = _claim(
        metric_is_task_success=True,
        metric_has_other_producers=True,
        tasks_solvable_without_component=True,
        delta=0.18,
    )
    assert claim.inference_class is InferenceClass.CAPABILITY
    assert claim.supports_earns_its_cost is True


def test_a_capability_condition_with_no_delta_supports_nothing():
    """A zero result is a valid finding and is not support for the claim."""
    claim = _claim(
        metric_is_task_success=True,
        metric_has_other_producers=True,
        tasks_solvable_without_component=True,
        delta=0.0,
    )
    assert claim.inference_class is InferenceClass.CAPABILITY
    assert claim.supports_earns_its_cost is False


def test_a_mechanistic_result_is_real_and_still_short_of_capability():
    claim = _claim(metric_has_other_producers=True, delta=0.4)
    assert claim.inference_class is InferenceClass.MECHANISTIC
    assert "one layer short" in LICENSE[InferenceClass.MECHANISTIC]


def test_the_summary_counts_capability_claims_not_load_bearing_ones():
    """Three large deltas can contain zero pieces of capability evidence."""
    summary = summarise(
        [
            _claim(condition="a", metric_has_other_producers=False, delta=1.0),
            _claim(condition="b", metric_has_other_producers=False, delta=0.5),
            _claim(condition="c", metric_has_other_producers=False, delta=0.085),
        ]
    )
    assert summary["conditions"] == 3
    assert summary["capability_claims"] == 0
    assert summary["by_inference_class"]["tautological"] == 3
    assert "cannot support a capability claim" in summary["honest_summary"]


def test_every_class_has_a_license_sentence():
    for member in InferenceClass:
        assert LICENSE[member].strip()


@pytest.mark.skipif(not SCORECARD.is_file(), reason="scorecard not generated")
def test_the_committed_scorecard_states_what_it_licenses():
    """The artifact must carry the classification, not just the deltas.

    Before this, the scorecard's headline was `all_conditions_load_bearing:
    true` over three wiring checks — which reads exactly like three capability
    results and is not one.
    """
    report = json.loads(SCORECARD.read_text(encoding="utf-8"))
    assert "inference" in report, "the scorecard reports deltas with no inference class"
    inference = report["inference"]
    assert inference["schema"] == "aura.lesion_inference.v1"
    assert set(inference["by_inference_class"]) == {str(m) for m in InferenceClass}
    for claim in report.get("claims", []):
        assert claim["inference_class"] in {str(m) for m in InferenceClass}
        assert claim["licenses"].strip()
