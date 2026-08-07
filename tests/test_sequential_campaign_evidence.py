from __future__ import annotations

from copy import deepcopy

import pytest

from core.brain.llm.latent_cortex.campaign_journal import CampaignPlan
from core.brain.llm.latent_cortex.sequential_campaign_evidence import (
    SequentialCampaignEvidenceError,
    build_sequential_look_certificate,
    cumulative_task_ids,
    sequential_task_look_assignments,
)


def _plan() -> CampaignPlan:
    tasks = []
    cells = []
    for domain in ("coding", "mathematics"):
        for ordinal in range(1, 5):
            task_id = f"{domain}-{ordinal}"
            tasks.append({"task_id": task_id, "domain": domain})
            for arm in ("base_vanilla", "adapter_rlc"):
                cells.append(
                    {
                        "task_id": task_id,
                        "domain": domain,
                        "arm": arm,
                    }
                )
    return CampaignPlan.build(
        "sequential-evidence-test",
        cells,
        metadata={
            "arms": ["base_vanilla", "adapter_rlc"],
            "task_manifest": {"tasks": tasks},
            "execution_config": {
                "sequential_look_observations_per_domain": [2, 4],
                "exact_statistical_power": {
                    "schema": "aura.latent_cortex.exact_group_sequential_power.v1",
                    "looks": [
                        {
                            "look": 1,
                            "family_alpha": {"numerator": 1, "denominator": 200},
                        },
                        {
                            "look": 2,
                            "family_alpha": {"numerator": 9, "denominator": 200},
                        },
                    ],
                },
            },
        },
    )


def _records(plan: CampaignPlan, look: int):
    task_ids = cumulative_task_ids(plan, look)
    records = []
    for cell_id in plan.cell_ids:
        definition = plan.cell_definition(cell_id)
        if definition["task_id"] in task_ids:
            records.append(
                {
                    "cell_id": cell_id,
                    "definition": definition,
                    "commit": {
                        "result_sha256": "a" * 64,
                        "verification_sha256": "b" * 64,
                    },
                }
            )
    return records


def _grade(plan: CampaignPlan, look: int, verdict: str):
    task_count = len(cumulative_task_ids(plan, look))
    alpha = ({"numerator": 1, "denominator": 200} if look == 1 else {"numerator": 9, "denominator": 200})
    return {
        "plan_sha256": plan.plan_sha256,
        "expected_task_count": task_count,
        "expected_cell_count": task_count * 2,
        "statistical_policy": {"alpha": alpha},
        "verdict": verdict,
    }


def test_task_look_assignment_is_balanced_and_cumulative():
    plan = _plan()
    assignments = sequential_task_look_assignments(plan)
    assert list(assignments.values()).count(1) == 4
    assert list(assignments.values()).count(2) == 4
    assert len(cumulative_task_ids(plan, 1)) == 4
    assert len(cumulative_task_ids(plan, 2)) == 8


def test_look_certificate_chains_and_classifies_only_exact_grade_verdicts():
    plan = _plan()
    first_grade = _grade(plan, 1, "inconclusive")
    first = build_sequential_look_certificate(
        plan=plan,
        look=1,
        committed_records=_records(plan, 1),
        production_grade=first_grade,
        independent_grade=first_grade,
        previous_certificate_sha256=None,
    )
    assert first["decision"] == "continue"

    final_grade = _grade(plan, 2, "gain_preverified")
    final = build_sequential_look_certificate(
        plan=plan,
        look=2,
        committed_records=_records(plan, 2),
        production_grade=final_grade,
        independent_grade=final_grade,
        previous_certificate_sha256=first["certificate_sha256"],
    )
    assert final["decision"] == "positive_boundary_crossed"
    assert final["previous_certificate_sha256"] == first["certificate_sha256"]


def test_look_certificate_rejects_independent_or_scope_tampering():
    plan = _plan()
    grade = _grade(plan, 1, "inconclusive")
    independent = deepcopy(grade)
    independent["verdict"] = "gain_preverified"
    with pytest.raises(
        SequentialCampaignEvidenceError,
        match="sequential_independent_grade_mismatch",
    ):
        build_sequential_look_certificate(
            plan=plan,
            look=1,
            committed_records=_records(plan, 1),
            production_grade=grade,
            independent_grade=independent,
            previous_certificate_sha256=None,
        )

    records = _records(plan, 1)
    records[0]["cell_id"] = "cell-" + "f" * 64
    with pytest.raises(
        SequentialCampaignEvidenceError,
        match="sequential_record_invalid",
    ):
        build_sequential_look_certificate(
            plan=plan,
            look=1,
            committed_records=records,
            production_grade=grade,
            independent_grade=grade,
            previous_certificate_sha256=None,
        )

    records = _records(plan, 1)[:-1]
    with pytest.raises(
        SequentialCampaignEvidenceError,
        match="sequential_record_scope_incomplete",
    ):
        build_sequential_look_certificate(
            plan=plan,
            look=1,
            committed_records=records,
            production_grade=grade,
            independent_grade=grade,
            previous_certificate_sha256=None,
        )
