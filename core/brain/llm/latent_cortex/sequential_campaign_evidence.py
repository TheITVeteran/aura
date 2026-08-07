"""Tamper-evident look certificates for paired RLC campaigns."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final

from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignPlan,
    canonical_json_bytes,
)

SEQUENTIAL_LOOK_CERTIFICATE_SCHEMA: Final = (
    "aura.latent_cortex.sequential_look_certificate.v1"
)
SEQUENTIAL_POWER_SCHEMA: Final = (
    "aura.latent_cortex.exact_group_sequential_power.v1"
)


class SequentialCampaignEvidenceError(ValueError):
    """A sequential look cannot be reconstructed from frozen evidence."""


def _fail(code: str) -> None:
    raise SequentialCampaignEvidenceError(code)


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sequential_task_look_assignments(plan: CampaignPlan) -> dict[str, int]:
    """Derive each task's signed look from manifest order and domain boundary."""

    if not isinstance(plan, CampaignPlan):
        _fail("sequential_plan_invalid")
    metadata = plan.to_dict().get("metadata")
    if not isinstance(metadata, Mapping):
        _fail("sequential_plan_metadata_invalid")
    execution = metadata.get("execution_config")
    task_manifest = metadata.get("task_manifest")
    if not isinstance(execution, Mapping) or not isinstance(task_manifest, Mapping):
        _fail("sequential_plan_metadata_invalid")
    looks = execution.get("sequential_look_observations_per_domain")
    tasks = task_manifest.get("tasks")
    if (
        not isinstance(looks, list)
        or not looks
        or any(type(value) is not int or value <= 0 for value in looks)
        or any(current <= previous for previous, current in zip(looks, looks[1:], strict=False))
        or not isinstance(tasks, list)
        or not tasks
    ):
        _fail("sequential_plan_contract_invalid")
    domain_ordinals: Counter[str] = Counter()
    assignments: dict[str, int] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            _fail("sequential_task_manifest_invalid")
        task_id = task.get("task_id")
        domain = task.get("domain")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in assignments
            or not isinstance(domain, str)
            or not domain
        ):
            _fail("sequential_task_manifest_invalid")
        domain_ordinals[domain] += 1
        ordinal = domain_ordinals[domain]
        assigned = next(
            (index for index, boundary in enumerate(looks, 1) if ordinal <= boundary),
            None,
        )
        if assigned is None:
            _fail("sequential_task_outside_terminal_look")
        assignments[task_id] = assigned
    if any(count != looks[-1] for count in domain_ordinals.values()):
        _fail("sequential_terminal_look_unbalanced")
    return assignments


def cumulative_task_ids(plan: CampaignPlan, look: int) -> frozenset[str]:
    if type(look) is not int or look <= 0:
        _fail("sequential_look_invalid")
    assignments = sequential_task_look_assignments(plan)
    if look > max(assignments.values()):
        _fail("sequential_look_invalid")
    return frozenset(task_id for task_id, assigned in assignments.items() if assigned <= look)


def build_sequential_look_certificate(
    *,
    plan: CampaignPlan,
    look: int,
    committed_records: Sequence[Mapping[str, Any]],
    production_grade: Mapping[str, Any],
    independent_grade: Mapping[str, Any],
    previous_certificate_sha256: str | None,
) -> dict[str, Any]:
    """Build one deterministic decision certificate from independently equal trees."""

    task_ids = cumulative_task_ids(plan, look)
    metadata = plan.to_dict()["metadata"]
    power = metadata["execution_config"].get("exact_statistical_power")
    if (
        not isinstance(power, Mapping)
        or power.get("schema") != SEQUENTIAL_POWER_SCHEMA
        or not isinstance(power.get("looks"), list)
        or look > len(power["looks"])
    ):
        _fail("sequential_power_receipt_invalid")
    look_receipt = power["looks"][look - 1]
    if not isinstance(look_receipt, Mapping) or look_receipt.get("look") != look:
        _fail("sequential_power_receipt_invalid")
    if previous_certificate_sha256 is not None and not _is_sha256(
        previous_certificate_sha256
    ):
        _fail("sequential_previous_certificate_invalid")
    if look == 1 and previous_certificate_sha256 is not None:
        _fail("sequential_previous_certificate_unexpected")
    if look > 1 and previous_certificate_sha256 is None:
        _fail("sequential_previous_certificate_required")

    records = tuple(committed_records)
    expected_cell_ids = {
        cell_id
        for cell_id in plan.cell_ids
        if plan.cell_definition(cell_id).get("task_id") in task_ids
    }
    expected_cells = len(expected_cell_ids)
    observed_task_ids: set[str] = set()
    observed_cell_ids: set[str] = set()
    record_receipts: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            _fail("sequential_record_invalid")
        cell_id = record.get("cell_id")
        definition = record.get("definition")
        commit = record.get("commit")
        if (
            not isinstance(cell_id, str)
            or cell_id not in expected_cell_ids
            or cell_id in observed_cell_ids
            or not isinstance(definition, Mapping)
            or dict(definition) != plan.cell_definition(cell_id)
            or definition.get("task_id") not in task_ids
            or not isinstance(commit, Mapping)
            or not _is_sha256(commit.get("result_sha256"))
            or not _is_sha256(commit.get("verification_sha256"))
        ):
            _fail("sequential_record_invalid")
        observed_cell_ids.add(cell_id)
        observed_task_ids.add(definition["task_id"])
        record_receipts.append(
            {
                "cell_id": cell_id,
                "result_sha256": commit["result_sha256"],
                "verification_sha256": commit["verification_sha256"],
            }
        )
    if (
        observed_task_ids != set(task_ids)
        or observed_cell_ids != expected_cell_ids
        or len(observed_cell_ids) != expected_cells
        or len(records) != expected_cells
    ):
        _fail("sequential_record_scope_incomplete")

    production = dict(production_grade)
    independent = dict(independent_grade)
    if canonical_json_bytes(production) != canonical_json_bytes(independent):
        _fail("sequential_independent_grade_mismatch")
    expected_alpha = look_receipt.get("family_alpha")
    policy = production.get("statistical_policy")
    if (
        production.get("plan_sha256") != plan.plan_sha256
        or production.get("expected_task_count") != len(task_ids)
        or production.get("expected_cell_count") != expected_cells
        or not isinstance(policy, Mapping)
        or policy.get("alpha") != expected_alpha
    ):
        _fail("sequential_grade_contract_invalid")
    verdict = production.get("verdict")
    terminal = look == len(power["looks"])
    if verdict == "gain_preverified":
        decision = "positive_boundary_crossed"
    elif verdict == "gain_refuted":
        decision = "refutation_boundary_crossed"
    elif terminal:
        decision = "terminal_inconclusive"
    else:
        decision = "continue"
    sorted_receipts = sorted(record_receipts, key=lambda item: item["cell_id"])
    material = {
        "schema": SEQUENTIAL_LOOK_CERTIFICATE_SCHEMA,
        "campaign_name": plan.campaign_name,
        "plan_sha256": plan.plan_sha256,
        "look": look,
        "terminal_look": terminal,
        "previous_certificate_sha256": previous_certificate_sha256,
        "look_power_receipt": dict(look_receipt),
        "cumulative_task_count": len(task_ids),
        "cumulative_cell_count": expected_cells,
        "cumulative_task_ids_sha256": _sha256(sorted(task_ids)),
        "record_receipts": sorted_receipts,
        "record_receipts_sha256": _sha256(sorted_receipts),
        "production_grade": production,
        "production_grade_sha256": _sha256(production),
        "independent_grade": independent,
        "independent_grade_sha256": _sha256(independent),
        "independent_semantic_parity": True,
        "decision": decision,
        "claim_status": "candidate_external_final_verifier_required",
    }
    return {
        **material,
        "certificate_sha256": _sha256(material),
    }


__all__ = [
    "SEQUENTIAL_LOOK_CERTIFICATE_SCHEMA",
    "SequentialCampaignEvidenceError",
    "build_sequential_look_certificate",
    "cumulative_task_ids",
    "sequential_task_look_assignments",
]
