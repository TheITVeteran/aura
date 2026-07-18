"""Immutable planning and grading for resident RLC attribution campaigns.

This module is deliberately model-free. The executable producer records one
cell per task and arm in :mod:`campaign_journal`; this module freezes those
cells and derives every verdict from replayed committed evidence.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Never, cast

from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.experiments import (
    CONJECTURE,
    PROVEN,
    REFUTED,
    PairedObservation,
    grade_paired_treatment_vs_control,
)
from core.brain.llm.latent_cortex.frontier_tasks import (
    FRONTIER_DOMAINS,
    FrontierTask,
    build_task_commitment,
    build_task_manifest,
)

CAMPAIGN_SCHEMA = "aura.latent_cortex.resident_paired_campaign.v1"
GRADE_SCHEMA = "aura.latent_cortex.resident_paired_grade.v1"
CONTAMINATION_AUDIT_SCHEMA = "aura.latent_cortex.contamination_audit.v2"

BASE_VANILLA = "base_vanilla"
BASE_RLC = "base_rlc"
ADAPTER_VANILLA = "adapter_vanilla"
ADAPTER_RLC = "adapter_rlc"
BASE_EQUAL_COMPUTE = "base_equal_compute"
ADAPTER_EQUAL_COMPUTE = "adapter_equal_compute"

PRIMARY_ARMS = (BASE_VANILLA, BASE_RLC, ADAPTER_VANILLA, ADAPTER_RLC)
FULL_ARMS = (*PRIMARY_ARMS, BASE_EQUAL_COMPUTE, ADAPTER_EQUAL_COMPUTE)

_MIN_DOMAIN_TRIALS = 20
_BOOTSTRAP_RESAMPLES = 20_000


class PairedCampaignError(ValueError):
    """Stable fail-closed campaign planning or grading error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise PairedCampaignError(code)


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _contamination_audit_valid(
    audit: Any,
    *,
    task_manifest_sha256: Any,
) -> bool:
    if not isinstance(audit, Mapping) or set(audit) != {
        "schema",
        "task_manifest_sha256",
        "status",
        "overlap_count",
        "auditor_independence",
        "corpora",
        "methods",
        "signature",
    }:
        return False
    signature = audit.get("signature")
    if not isinstance(signature, Mapping) or set(signature) != {
        "algorithm",
        "key_id",
        "signature_b64",
        "signed_payload_sha256",
        "public_key_der_b64",
        "trust_root_sha256",
        "verified",
    }:
        return False
    corpora = audit.get("corpora")
    methods = audit.get("methods")
    if (
        audit.get("schema") != CONTAMINATION_AUDIT_SCHEMA
        or audit.get("task_manifest_sha256") != task_manifest_sha256
        or audit.get("status") != "passed_zero_overlap"
        or audit.get("overlap_count") != 0
        or audit.get("auditor_independence") != "external"
        or not isinstance(corpora, list)
        or not corpora
        or any(
            not isinstance(record, Mapping)
            or set(record) != {"name", "snapshot_sha256"}
            or not isinstance(record.get("name"), str)
            or not isinstance(record.get("snapshot_sha256"), str)
            or len(cast(str, record.get("snapshot_sha256"))) != 64
            for record in corpora
        )
        or not isinstance(methods, list)
        or not {"exact_prompt", "normalized_prompt", "token_fivegram"}.issubset(
            methods
        )
        or signature.get("algorithm") != "ed25519"
        or signature.get("verified") is not True
        or any(
            not isinstance(signature.get(key), str)
            for key in (
                "key_id",
                "signature_b64",
                "signed_payload_sha256",
                "public_key_der_b64",
                "trust_root_sha256",
            )
        )
    ):
        return False
    body = {key: value for key, value in audit.items() if key != "signature"}
    signed_payload = canonical_json_bytes(body)
    if signature.get("signed_payload_sha256") != hashlib.sha256(
        signed_payload
    ).hexdigest():
        return False
    try:
        public_der = base64.b64decode(
            cast(str, signature.get("public_key_der_b64")), validate=True
        )
        signature_bytes = base64.b64decode(
            cast(str, signature.get("signature_b64")), validate=True
        )
    except (TypeError, ValueError, binascii.Error):
        return False
    trust_sha256 = hashlib.sha256(public_der).hexdigest()
    if (
        signature.get("trust_root_sha256") != trust_sha256
        or signature.get("key_id") != trust_sha256
    ):
        return False
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        public_key = serialization.load_der_public_key(public_der)
        if not isinstance(public_key, Ed25519PublicKey):
            return False
        public_key.verify(signature_bytes, signed_payload)
    except (TypeError, ValueError, InvalidSignature):
        return False
    return True


def _arm_execution_order(campaign_name: str, arms: tuple[str, ...]) -> tuple[str, ...]:
    """Freeze a randomized block order while honoring equal-compute dependencies."""

    prerequisites = {
        BASE_EQUAL_COMPUTE: {BASE_RLC},
        ADAPTER_EQUAL_COMPUTE: {ADAPTER_RLC},
    }
    pending = set(arms)
    completed: set[str] = set()
    ordered: list[str] = []
    while pending:
        ready = [
            arm
            for arm in pending
            if prerequisites.get(arm, set()).issubset(completed)
        ]
        if not ready:
            _fail("campaign_arm_dependencies_invalid")
        selected = min(
            ready,
            key=lambda arm: hashlib.sha256(
                f"{campaign_name}:arm-execution-order:{arm}".encode()
            ).digest(),
        )
        ordered.append(selected)
        pending.remove(selected)
        completed.add(selected)
    return tuple(ordered)


def build_campaign_plan(
    campaign_name: str,
    tasks: Sequence[FrontierTask],
    *,
    model_identity: Mapping[str, Any],
    adapter_identity: Mapping[str, Any],
    execution_config: Mapping[str, Any],
    contamination_audit: Mapping[str, Any] | None = None,
    arms: Sequence[str] = FULL_ARMS,
    claim_eligible: bool = False,
) -> CampaignPlan:
    """Freeze all public tasks and arm cells before model execution."""

    if not tasks or not all(isinstance(task, FrontierTask) for task in tasks):
        _fail("campaign_tasks_invalid")
    normalized_arms = tuple(arms)
    if (
        not normalized_arms
        or len(set(normalized_arms)) != len(normalized_arms)
        or any(arm not in FULL_ARMS for arm in normalized_arms)
        or not set(PRIMARY_ARMS).issubset(normalized_arms)
    ):
        _fail("campaign_arms_invalid")
    if type(claim_eligible) is not bool:
        _fail("campaign_claim_eligibility_invalid")

    manifest = build_task_manifest(tasks)
    commitment = build_task_commitment(manifest)
    audit = {} if contamination_audit is None else dict(contamination_audit)
    audit_valid = _contamination_audit_valid(
        audit,
        task_manifest_sha256=manifest.manifest_sha256,
    )
    if claim_eligible and not audit_valid:
        _fail("campaign_contamination_audit_required")
    task_by_id = {task.task_id: task for task in tasks}
    ordered_tasks = [task_by_id[record.task_id] for record in manifest.tasks]
    execution_order = _arm_execution_order(campaign_name, normalized_arms)
    task_execution_ordinals: dict[tuple[str, str], int] = {}
    for arm in normalized_arms:
        arm_tasks = sorted(
            ordered_tasks,
            key=lambda task: hashlib.sha256(
                f"{campaign_name}:task-order:{arm}:{task.task_id}".encode()
            ).digest(),
        )
        task_execution_ordinals.update(
            {
                (arm, task.task_id): ordinal
                for ordinal, task in enumerate(arm_tasks)
            }
        )
    cells: list[dict[str, Any]] = []
    for task_ordinal, task in enumerate(ordered_tasks):
        for arm in normalized_arms:
            cells.append(
                {
                    "arm": arm,
                    "domain": task.domain,
                    "execution_ordinal_within_arm": task_execution_ordinals[
                        (arm, task.task_id)
                    ],
                    "task_id": task.task_id,
                    "task_ordinal": task_ordinal,
                    "task_payload_sha256": task.public.task_payload_sha256,
                }
            )
    metadata = {
        "schema": CAMPAIGN_SCHEMA,
        "claim_eligible": claim_eligible,
        "claim_scope": (
            "resident same-checkpoint causal attribution preflight"
            if not claim_eligible
            else "resident same-checkpoint causal attribution"
        ),
        "external_frontier_claim_eligible": False,
        "producer_independence": "producer_local_not_external",
        "arms": list(normalized_arms),
        "arm_execution_order": list(execution_order),
        "model_identity": dict(model_identity),
        "adapter_identity": dict(adapter_identity),
        "execution_config": dict(execution_config),
        "task_manifest": manifest.to_dict(),
        "task_commitment": commitment.to_dict(),
        "contamination_audit": audit,
        "contamination_trust_root_sha256": (
            audit["signature"]["trust_root_sha256"] if audit_valid else None
        ),
    }
    return CampaignPlan.build(campaign_name, cells, metadata=metadata)


def _strict_result_row(
    record: Mapping[str, Any],
    *,
    plan: CampaignPlan,
    tasks_by_id: Mapping[str, Mapping[str, Any]],
    issuer_tasks_by_id: Mapping[str, FrontierTask],
    model_identity: Mapping[str, Any],
    adapter_identity: Mapping[str, Any],
    execution_config: Mapping[str, Any],
) -> tuple[str, str, str, bool, int]:
    if not isinstance(record, Mapping):
        _fail("campaign_record_invalid")
    definition = record.get("definition")
    result = record.get("result")
    verification = record.get("verification")
    commit = record.get("commit")
    cell_id = record.get("cell_id")
    if not isinstance(cell_id, str) or cell_id not in plan.cell_ids:
        _fail("campaign_record_cell_invalid")
    if not isinstance(definition, Mapping):
        _fail("campaign_record_shape_invalid")
    if not isinstance(result, Mapping):
        _fail("campaign_record_shape_invalid")
    if not isinstance(verification, Mapping):
        _fail("campaign_record_shape_invalid")
    if not isinstance(commit, Mapping):
        _fail("campaign_record_shape_invalid")
    definition = cast(Mapping[str, Any], definition)
    result = cast(Mapping[str, Any], result)
    verification = cast(Mapping[str, Any], verification)
    commit = cast(Mapping[str, Any], commit)
    expected_definition = plan.cell_definition(cell_id)
    if dict(definition) != expected_definition:
        _fail("campaign_record_definition_mismatch")
    task_id = expected_definition["task_id"]
    domain = expected_definition["domain"]
    arm = expected_definition["arm"]
    correct = verification.get("correct")
    layer_apps = result.get("layer_apps")
    if not all(isinstance(value, str) and value for value in (task_id, domain, arm)):
        _fail("campaign_record_identity_invalid")
    if arm not in FULL_ARMS:
        _fail("campaign_record_arm_invalid")
    if result.get("arm") != arm:
        _fail("campaign_result_arm_mismatch")
    text = result.get("text")
    output_sha256 = result.get("output_sha256")
    if (
        not isinstance(text, str)
        or not isinstance(output_sha256, str)
        or hashlib.sha256(text.encode("utf-8")).hexdigest() != output_sha256
    ):
        _fail("campaign_result_output_mismatch")
    runtime_identity = result.get("runtime_model_identity")
    runtime_bundle = model_identity.get("runtime_bundle")
    implementation_sha256 = execution_config.get("implementation_sha256")
    planned_personality = model_identity.get("personality_adapter")
    planned_effective_stack = model_identity.get("effective_stack_sha256")
    if (
        not isinstance(runtime_identity, Mapping)
        or not isinstance(runtime_bundle, Mapping)
        or not isinstance(implementation_sha256, Mapping)
        or runtime_identity.get("worker_model_path") != model_identity.get("model_path")
        or runtime_identity.get("worker_model_parameter_count")
        != runtime_bundle.get("logical_parameter_count")
        or runtime_identity.get("worker_model_parameter_count_basis")
        != runtime_bundle.get("logical_parameter_count_basis")
        or runtime_identity.get("worker_weight_fingerprint")
        != model_identity.get("fingerprint")
        or runtime_identity.get("worker_weight_fingerprint_method")
        != model_identity.get("method")
        or runtime_identity.get("worker_weight_file_count")
        != model_identity.get("files")
        or runtime_identity.get("worker_runtime_bundle_sha256")
        != runtime_bundle.get("bundle_sha256")
        or runtime_identity.get("worker_load_boundary_verified") is not True
        or runtime_identity.get("worker_source_sha256")
        != implementation_sha256.get("tools/run_latent_cortex_paired_campaign.py")
        or (
            planned_personality is not None
            and runtime_identity.get("worker_personality_adapter")
            != planned_personality
        )
        or (
            planned_effective_stack is not None
            and runtime_identity.get("worker_effective_stack_sha256")
            != planned_effective_stack
        )
    ):
        _fail("campaign_runtime_model_identity_mismatch")
    adapter_receipt = adapter_identity.get("identity_receipt")
    if not isinstance(adapter_receipt, Mapping):
        _fail("campaign_adapter_identity_invalid")
    if arm.startswith("adapter_"):
        if (
            result.get("adapter_identity_sha256")
            != adapter_receipt.get("composite_identity_sha256")
            or result.get("adapter_wrapped_projections")
            != adapter_receipt.get("wrapped_projection_count")
            or not isinstance(result.get("runtime_adapter_identity"), Mapping)
            or dict(cast(Mapping[str, Any], result["runtime_adapter_identity"]))
            != dict(adapter_receipt)
        ):
            _fail("campaign_adapter_activation_mismatch")
    elif (
        result.get("adapter_identity_sha256") is not None
        or result.get("adapter_wrapped_projections") != 0
        or result.get("runtime_adapter_identity") is not None
    ):
        _fail("campaign_base_arm_adapter_contaminated")
    if type(correct) is not bool:
        _fail("campaign_record_verification_invalid")
    if type(layer_apps) is not int or layer_apps <= 0:
        _fail("campaign_record_compute_invalid")
    task = tasks_by_id.get(cast(str, task_id))
    issuer_task = issuer_tasks_by_id.get(cast(str, task_id))
    score = verification.get("score_receipt")
    if task is None or issuer_task is None or not isinstance(score, Mapping):
        _fail("campaign_score_binding_invalid")
    independent_score = issuer_task.score(text).to_dict()
    if (
        dict(score) != independent_score
        or independent_score.get("correct") is not correct
        or verification.get("answer_commitment_sha256")
        != task.get("answer_commitment_sha256")
    ):
        _fail("campaign_score_binding_invalid")
    if commit.get("result_sha256") != _sha256(dict(result)):
        _fail("campaign_result_commitment_mismatch")
    if commit.get("verification_sha256") != _sha256(dict(verification)):
        _fail("campaign_verification_commitment_mismatch")
    return (
        cast(str, task_id),
        cast(str, domain),
        cast(str, arm),
        cast(bool, correct),
        layer_apps,
    )


def _paired_claim(
    rows: Mapping[str, Mapping[str, tuple[str, bool, int]]],
    *,
    treatment: str,
    control: str,
    require_compute: bool,
    compute_tolerance: float,
) -> dict[str, Any]:
    by_domain: dict[str, list[PairedObservation]] = defaultdict(list)
    for task_id in sorted(rows):
        arms = rows[task_id]
        if treatment not in arms or control not in arms:
            _fail("campaign_comparison_incomplete")
        treatment_domain, treatment_success, treatment_cost = arms[treatment]
        control_domain, control_success, control_cost = arms[control]
        if treatment_domain != control_domain:
            _fail("campaign_task_domain_drift")
        by_domain[treatment_domain].append(
            PairedObservation(
                task_id=task_id,
                family=treatment_domain,
                treatment_success=treatment_success,
                control_success=control_success,
                treatment_layer_apps=treatment_cost,
                control_layer_apps=control_cost,
            )
        )
    claim = grade_paired_treatment_vs_control(
        f"{treatment}_vs_{control}",
        f"{treatment} improves over {control}",
        dict(by_domain),
        alpha=0.05,
        minimum_effect=0.02,
        compute_tolerance=compute_tolerance,
        require_compute=require_compute,
    ).to_dict()
    claim.pop("graded_at", None)
    return cast(dict[str, Any], claim)


def _interaction_interval(values: list[int]) -> tuple[float, float, float]:
    """Deterministic percentile interval for the 2x2 interaction effect."""

    if not values:
        return 0.0, 0.0, 1.0
    mean = sum(values) / len(values)
    if len(set(values)) == 1:
        pvalue = 2.0 ** (-len(values)) if values[0] > 0 else 1.0
        return float(values[0]), float(values[0]), pvalue
    import numpy as np

    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260718)
    samples: Any = np.empty((_BOOTSTRAP_RESAMPLES,), dtype=np.float64)
    for start in range(0, _BOOTSTRAP_RESAMPLES, 500):
        count = min(500, _BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, len(data), size=(count, len(data)))
        samples[start : start + count] = data[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    # One-sided random-sign permutation over task-level interaction values.
    permuted: Any = np.empty((_BOOTSTRAP_RESAMPLES,), dtype=np.float64)
    for start in range(0, _BOOTSTRAP_RESAMPLES, 500):
        count = min(500, _BOOTSTRAP_RESAMPLES - start)
        signs = rng.choice((-1.0, 1.0), size=(count, len(data)))
        permuted[start : start + count] = (signs * data).mean(axis=1)
    pvalue = (1.0 + float((permuted >= mean).sum())) / (len(permuted) + 1.0)
    return float(low), float(high), pvalue


def grade_campaign(
    records: Iterable[Mapping[str, Any]],
    *,
    plan: CampaignPlan,
    issuer_tasks: Sequence[FrontierTask],
    trusted_contamination_root_sha256: str | None = None,
) -> dict[str, Any]:
    """Grade a complete replayed campaign; never infer absent cells."""

    if not isinstance(plan, CampaignPlan):
        _fail("campaign_plan_invalid")
    plan_document = plan.to_dict()
    metadata = plan_document.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("schema") != CAMPAIGN_SCHEMA:
        _fail("campaign_plan_metadata_invalid")
    raw_arms = metadata.get("arms")
    task_manifest = metadata.get("task_manifest")
    model_identity = metadata.get("model_identity")
    adapter_identity = metadata.get("adapter_identity")
    execution_config = metadata.get("execution_config")
    if (
        not isinstance(raw_arms, list)
        or not isinstance(task_manifest, Mapping)
        or not isinstance(model_identity, Mapping)
        or not isinstance(adapter_identity, Mapping)
        or not isinstance(execution_config, Mapping)
    ):
        _fail("campaign_plan_metadata_invalid")
    raw_tasks = task_manifest.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        _fail("campaign_plan_tasks_invalid")
    tasks_by_id: dict[str, Mapping[str, Any]] = {}
    for task in raw_tasks:
        if not isinstance(task, Mapping) or not isinstance(task.get("task_id"), str):
            _fail("campaign_plan_tasks_invalid")
        task_id = cast(str, task["task_id"])
        if task_id in tasks_by_id:
            _fail("campaign_plan_task_duplicate")
        tasks_by_id[task_id] = cast(Mapping[str, Any], task)
    if (
        not isinstance(issuer_tasks, Sequence)
        or not issuer_tasks
        or any(not isinstance(task, FrontierTask) for task in issuer_tasks)
    ):
        _fail("campaign_issuer_tasks_invalid")
    issuer_tasks_by_id = {task.task_id: task for task in issuer_tasks}
    if set(issuer_tasks_by_id) != set(tasks_by_id) or any(
        task.public.to_dict() != dict(tasks_by_id[task_id])
        for task_id, task in issuer_tasks_by_id.items()
    ):
        _fail("campaign_issuer_tasks_mismatch")
    expected_task_count = len(tasks_by_id)
    arms = tuple(raw_arms)
    if not set(PRIMARY_ARMS).issubset(arms) or any(arm not in FULL_ARMS for arm in arms):
        _fail("expected_arms_invalid")
    claim_eligible = metadata.get("claim_eligible")
    if type(claim_eligible) is not bool:
        _fail("campaign_claim_eligibility_invalid")
    contamination_audit = metadata.get("contamination_audit")
    planned_contamination_root = metadata.get(
        "contamination_trust_root_sha256"
    )
    if claim_eligible and not _contamination_audit_valid(
        contamination_audit,
        task_manifest_sha256=task_manifest.get("manifest_sha256"),
    ):
        _fail("campaign_contamination_audit_required")
    if claim_eligible and (
        not isinstance(trusted_contamination_root_sha256, str)
        or len(trusted_contamination_root_sha256) != 64
        or trusted_contamination_root_sha256 != planned_contamination_root
        or not isinstance(contamination_audit, Mapping)
        or not isinstance(contamination_audit.get("signature"), Mapping)
        or contamination_audit["signature"].get("trust_root_sha256")
        != trusted_contamination_root_sha256
    ):
        _fail("campaign_contamination_trust_root_required")
    arm_execution_order = metadata.get("arm_execution_order")
    if (
        not isinstance(arm_execution_order, list)
        or len(arm_execution_order) != len(arms)
        or set(arm_execution_order) != set(arms)
    ):
        _fail("campaign_plan_arm_order_invalid")
    planned_domains: set[str] = set()
    for task in tasks_by_id.values():
        domain = task.get("domain")
        payload_sha256 = task.get("task_payload_sha256")
        if not isinstance(domain, str) or not isinstance(payload_sha256, str):
            _fail("campaign_plan_tasks_invalid")
        planned_domains.add(domain)
    expected_pairs = {(task_id, arm) for task_id in tasks_by_id for arm in arms}
    observed_pairs: set[tuple[str, str]] = set()
    execution_ordinals: dict[str, set[int]] = defaultdict(set)
    for cell_id in plan.cell_ids:
        definition = plan.cell_definition(cell_id)
        cell_task_id = definition.get("task_id")
        arm = definition.get("arm")
        domain = definition.get("domain")
        payload_sha256 = definition.get("task_payload_sha256")
        execution_ordinal = definition.get("execution_ordinal_within_arm")
        task = (
            tasks_by_id.get(cell_task_id)
            if isinstance(cell_task_id, str)
            else None
        )
        if (
            not isinstance(cell_task_id, str)
            or task is None
            or arm not in arms
            or domain != task.get("domain")
            or payload_sha256 != task.get("task_payload_sha256")
            or type(execution_ordinal) is not int
            or not 0 <= execution_ordinal < expected_task_count
        ):
            _fail("campaign_plan_cell_invalid")
        pair = (cell_task_id, cast(str, arm))
        if pair in observed_pairs:
            _fail("campaign_plan_cell_duplicate")
        observed_pairs.add(pair)
        execution_ordinals[cast(str, arm)].add(execution_ordinal)
    if observed_pairs != expected_pairs or any(
        execution_ordinals[arm] != set(range(expected_task_count)) for arm in arms
    ):
        _fail("campaign_plan_coverage_invalid")
    if claim_eligible and (
        arms != FULL_ARMS
        or planned_domains != set(FRONTIER_DOMAINS)
    ):
        _fail("campaign_claim_eligibility_invalid")
    rows: dict[str, dict[str, tuple[str, bool, int]]] = defaultdict(dict)
    observed_cell_ids: set[str] = set()
    for record in records:
        task_id, domain, arm, correct, layer_apps = _strict_result_row(
            record,
            plan=plan,
            tasks_by_id=tasks_by_id,
            issuer_tasks_by_id=issuer_tasks_by_id,
            model_identity=cast(Mapping[str, Any], model_identity),
            adapter_identity=cast(Mapping[str, Any], adapter_identity),
            execution_config=cast(Mapping[str, Any], execution_config),
        )
        cell_id = cast(str, record["cell_id"])
        if cell_id in observed_cell_ids:
            _fail("duplicate_campaign_cell_result")
        observed_cell_ids.add(cell_id)
        if arm not in arms:
            _fail("unexpected_campaign_arm")
        if arm in rows[task_id]:
            _fail("duplicate_task_arm_result")
        rows[task_id][arm] = (domain, correct, layer_apps)
    expected_cells = expected_task_count * len(arms)
    observed_cells = sum(len(task_arms) for task_arms in rows.values())
    complete = (
        len(rows) == expected_task_count
        and observed_cells == expected_cells
        and all(set(task_arms) == set(arms) for task_arms in rows.values())
    )
    if not complete:
        body = {
            "schema": GRADE_SCHEMA,
            "verdict": "incomplete",
            "claim_tier": CONJECTURE,
            "expected_task_count": expected_task_count,
            "expected_cell_count": expected_cells,
            "observed_task_count": len(rows),
            "observed_cell_count": observed_cells,
            "frontier_claim_eligible": False,
            "same_checkpoint_gain_claim_eligible": claim_eligible,
            "plan_sha256": plan.plan_sha256,
            "reasons": ["campaign_incomplete"],
        }
        return {**body, "grade_sha256": _sha256(body)}

    comparisons = {
        "base_rlc_gain": _paired_claim(
            rows,
            treatment=BASE_RLC,
            control=BASE_VANILLA,
            require_compute=False,
            compute_tolerance=1.0,
        ),
        "adapter_rlc_gain": _paired_claim(
            rows,
            treatment=ADAPTER_RLC,
            control=ADAPTER_VANILLA,
            require_compute=False,
            compute_tolerance=1.0,
        ),
        "adapter_effect_under_rlc": _paired_claim(
            rows,
            treatment=ADAPTER_RLC,
            control=BASE_RLC,
            require_compute=False,
            compute_tolerance=1.0,
        ),
        "adapter_effect_under_vanilla": _paired_claim(
            rows,
            treatment=ADAPTER_VANILLA,
            control=BASE_VANILLA,
            require_compute=False,
            compute_tolerance=1.0,
        ),
    }
    if BASE_EQUAL_COMPUTE in arms:
        comparisons["base_equal_compute"] = _paired_claim(
            rows,
            treatment=BASE_RLC,
            control=BASE_EQUAL_COMPUTE,
            require_compute=True,
            compute_tolerance=0.20,
        )
    if ADAPTER_EQUAL_COMPUTE in arms:
        comparisons["adapter_equal_compute"] = _paired_claim(
            rows,
            treatment=ADAPTER_RLC,
            control=ADAPTER_EQUAL_COMPUTE,
            require_compute=True,
            compute_tolerance=0.20,
        )

    interactions: list[int] = []
    domain_counts: dict[str, int] = defaultdict(int)
    for task_arms in rows.values():
        domain = task_arms[BASE_VANILLA][0]
        domain_counts[domain] += 1
        interactions.append(
            (int(task_arms[ADAPTER_RLC][1]) - int(task_arms[ADAPTER_VANILLA][1]))
            - (int(task_arms[BASE_RLC][1]) - int(task_arms[BASE_VANILLA][1]))
        )
    interaction_mean = sum(interactions) / len(interactions)
    interaction_low, interaction_high, interaction_p = _interaction_interval(interactions)
    underpowered = sorted(
        domain for domain, count in domain_counts.items() if count < _MIN_DOMAIN_TRIALS
    )
    required_claims = ["adapter_rlc_gain", "adapter_effect_under_rlc"]
    if ADAPTER_EQUAL_COMPUTE in arms:
        required_claims.append("adapter_equal_compute")
    statistically_proven = (
        not underpowered
        and all(comparisons[name]["tier"] == PROVEN for name in required_claims)
        and interaction_low > 0.02
        and interaction_p < 0.05
        and not comparisons["adapter_effect_under_vanilla"]["evidence"].get("regressed_families")
    )
    refuted = comparisons["adapter_rlc_gain"]["tier"] == REFUTED or interaction_high <= 0.0
    if statistically_proven and claim_eligible:
        verdict, tier, reasons = "gain_proven", PROVEN, []
    elif statistically_proven:
        verdict, tier, reasons = (
            "gain_observed_preflight",
            CONJECTURE,
            ["campaign_not_claim_eligible"],
        )
    elif underpowered:
        verdict, tier, reasons = (
            "incomplete_underpowered",
            CONJECTURE,
            [f"underpowered_domain:{domain}" for domain in underpowered],
        )
    elif refuted:
        verdict, tier, reasons = "gain_refuted", REFUTED, ["gain_gate_failed"]
    else:
        verdict, tier, reasons = "inconclusive", CONJECTURE, ["gain_not_proven"]
    body = {
        "schema": GRADE_SCHEMA,
        "verdict": verdict,
        "claim_tier": tier,
        "expected_task_count": expected_task_count,
        "expected_cell_count": expected_cells,
        "observed_task_count": len(rows),
        "observed_cell_count": observed_cells,
        "plan_sha256": plan.plan_sha256,
        "domain_counts": dict(sorted(domain_counts.items())),
        "comparisons": comparisons,
        "interaction": {
            "method": "task-paired 2x2 difference-in-differences bootstrap and sign permutation",
            "mean": round(interaction_mean, 6),
            "interval_95": [round(interaction_low, 6), round(interaction_high, 6)],
            "one_sided_permutation_p": interaction_p,
            "resamples": _BOOTSTRAP_RESAMPLES,
        },
        "frontier_claim_eligible": False,
        "same_checkpoint_gain_claim_eligible": claim_eligible,
        "reasons": reasons,
    }
    return {**body, "grade_sha256": _sha256(body)}


__all__ = [
    "ADAPTER_EQUAL_COMPUTE",
    "ADAPTER_RLC",
    "ADAPTER_VANILLA",
    "BASE_EQUAL_COMPUTE",
    "BASE_RLC",
    "BASE_VANILLA",
    "FULL_ARMS",
    "PRIMARY_ARMS",
    "PairedCampaignError",
    "build_campaign_plan",
    "grade_campaign",
]
