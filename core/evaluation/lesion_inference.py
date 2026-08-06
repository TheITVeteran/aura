"""core/evaluation/lesion_inference.py — what a lesion result licenses you to say.

Operationally: this measures how much a lesion's delta constrains the claim
"this subsystem contributes to the system's capability", by asking whether the
metric could have moved for any reason other than the subsystem being useful.

The criticism it answers is precise and fair:

    Lesion a direct connection, assert that its downstream value disappears.
    These establish that the code follows its equations. They do not establish
    that the equations are correct, that the variables correspond to what
    their names imply, or that the result improves real task performance.

Aura's own ablation scorecard shows the shape. `without_system2` reports
1.000 → 0.000 on `strict_proof_exact_answer_rate`, and the ablated component
IS the strict-proof solver. Removing the only thing that can emit an answer in
that format and observing that the format stops appearing is a true statement
about wiring and tells you nothing about whether System 2 makes Aura better at
anything. The number is large, real, correctly measured, and inferentially
almost empty — which is the worst combination available, because size reads as
strength.

Three kinds, and the distinction is about the METRIC, not the effect size:

    tautological   the metric is definitionally produced by the lesioned
                   component. Removing it must zero the metric; a delta here
                   is a wiring check.
    mechanistic    the metric is a property of the component's own output
                   (divergence, entropy, count). Real, and still one layer
                   away from capability — it says the organ does something,
                   not that the something helps.
    capability     the metric is task success, scored against an answer key
                   the component did not write, on tasks solvable without it.
                   The only kind that supports "earns its cost".

Nothing here scores a lesion as good or bad. It classifies what may be
concluded, so an artifact cannot present a wiring check in the vocabulary of a
capability result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class InferenceClass(StrEnum):
    TAUTOLOGICAL = "tautological"
    MECHANISTIC = "mechanistic"
    CAPABILITY = "capability"


#: What each class licenses, in the words an artifact should use.
LICENSE: dict[InferenceClass, str] = {
    InferenceClass.TAUTOLOGICAL: (
        "the component is wired to the metric it produces. This is a wiring "
        "check: the delta could not have been anything else, so it is not "
        "evidence about capability, usefulness, or architecture."
    ),
    InferenceClass.MECHANISTIC: (
        "the component measurably changes its own output. Real and worth "
        "knowing, and one layer short of capability — it establishes the organ "
        "does something, not that the something helps."
    ),
    InferenceClass.CAPABILITY: (
        "the component measurably changes TASK SUCCESS, scored against an "
        "answer key it did not write, on tasks that are solvable without it. "
        "This is the only class that supports an earns-its-cost claim."
    ),
}


@dataclass(frozen=True)
class LesionClaim:
    """One lesion result and the strongest sentence it supports."""

    condition: str
    subsystem: str
    metric_name: str
    delta: float
    #: Could the metric be produced by anything other than the lesioned
    #: component? False makes the result tautological regardless of delta.
    metric_has_other_producers: bool
    #: Is the metric task success against an independent answer key?
    metric_is_task_success: bool = False
    #: Were the tasks solvable without this component at all? A capability
    #: claim over tasks only this component can do is a tautology wearing a
    #: task battery.
    tasks_solvable_without_component: bool = False

    @property
    def inference_class(self) -> InferenceClass:
        if not self.metric_has_other_producers:
            return InferenceClass.TAUTOLOGICAL
        if self.metric_is_task_success and self.tasks_solvable_without_component:
            return InferenceClass.CAPABILITY
        return InferenceClass.MECHANISTIC

    @property
    def supports_earns_its_cost(self) -> bool:
        return self.inference_class is InferenceClass.CAPABILITY and self.delta > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "subsystem": self.subsystem,
            "metric": self.metric_name,
            "delta": self.delta,
            "inference_class": str(self.inference_class),
            "licenses": LICENSE[self.inference_class],
            "supports_earns_its_cost": self.supports_earns_its_cost,
        }


def summarise(claims: list[LesionClaim]) -> dict[str, Any]:
    """Portfolio view: how much of a scorecard is actually load-bearing evidence.

    The number that matters is `capability_claims`. A scorecard of twelve
    conditions all reporting large deltas can contain zero of them, and
    reporting "12/12 load-bearing" over that is how a wiring diagram gets
    presented as a result.
    """
    by_class: dict[str, int] = {str(k): 0 for k in InferenceClass}
    for claim in claims:
        by_class[str(claim.inference_class)] += 1
    capability = [c for c in claims if c.supports_earns_its_cost]
    return {
        "schema": "aura.lesion_inference.v1",
        "conditions": len(claims),
        "by_inference_class": by_class,
        "capability_claims": len(capability),
        "earns_its_cost_supported_by": [c.condition for c in capability],
        "honest_summary": (
            f"{len(capability)} of {len(claims)} conditions measure task success "
            "on tasks solvable without the component; the rest establish wiring "
            "or mechanism and cannot support a capability claim."
        ),
    }


__all__ = ["InferenceClass", "LICENSE", "LesionClaim", "summarise"]
