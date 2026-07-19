from __future__ import annotations

import base64
import hashlib
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.exact_paired_grade import (
    exact_campaign_power_plan,
)
from core.brain.llm.latent_cortex.frontier_tasks import (
    FRONTIER_DOMAINS,
    build_task_manifest,
    generate_task_battery,
)
from core.brain.llm.latent_cortex.paired_campaign import (
    ADAPTER_EQUAL_COMPUTE,
    ADAPTER_RLC,
    ADAPTER_VANILLA,
    BASE_EQUAL_COMPUTE,
    BASE_RLC,
    BASE_VANILLA,
    FULL_ARMS,
    PairedCampaignError,
    build_campaign_plan,
    grade_campaign,
)

MODEL_PATH = "/sealed/resident-32b"
RUNNER_SHA256 = "c" * 64
ADAPTER_SHA256 = "b" * 64
MODEL_SHA256 = "a" * 64
MODEL_BUNDLE_SHA256 = "d" * 64
TEST_PUBLIC_DER = Ed25519PrivateKey.from_private_bytes(
    bytes(range(32))
).public_key().public_bytes(
    encoding=serialization.Encoding.DER,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
)
TEST_TRUST_ROOT_SHA256 = hashlib.sha256(TEST_PUBLIC_DER).hexdigest()
TEST_POLICY_SHA256 = "f" * 64


def _campaign_trust():
    return {
        "prelaunch_verified": True,
        "externally_custodied": True,
        "policy_sha256": TEST_POLICY_SHA256,
        "unsigned_plan_sha256": "1" * 64,
    }


def _signed_contamination_audit(task_manifest_sha256: str):
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public_der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    trust_sha256 = hashlib.sha256(public_der).hexdigest()
    body = {
        "schema": "aura.latent_cortex.contamination_audit.v2",
        "task_manifest_sha256": task_manifest_sha256,
        "status": "passed_zero_overlap",
        "overlap_count": 0,
        "auditor_independence": "external",
        "corpora": [{"name": "held-out-corpus", "snapshot_sha256": "e" * 64}],
        "methods": ["exact_prompt", "normalized_prompt", "token_fivegram"],
    }
    signed_payload = canonical_json_bytes(body)
    return {
        **body,
        "signature": {
            "algorithm": "ed25519",
            "key_id": trust_sha256,
            "signature_b64": base64.b64encode(
                private_key.sign(signed_payload)
            ).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(signed_payload).hexdigest(),
            "public_key_der_b64": base64.b64encode(public_der).decode("ascii"),
            "trust_root_sha256": trust_sha256,
            "verified": True,
        },
    }


def _plan():
    tasks = generate_task_battery([101], domains=("mathematics", "coding"), difficulty=2)
    kwargs = {
        "model_identity": {"checkpoint_fingerprint": "a" * 64},
        "adapter_identity": {"composite_identity_sha256": "b" * 64},
        "execution_config": {"max_steps": 8, "decode_max_tokens": 256},
    }
    return tasks, build_campaign_plan("paired-test", tasks, **kwargs)


def _grade_plan(*, claim_eligible: bool = False):
    tasks = generate_task_battery(
        range(20),
        domains=FRONTIER_DOMAINS,
        difficulty=2,
    )
    manifest = build_task_manifest(tasks)
    contamination_audit = _signed_contamination_audit(manifest.manifest_sha256)
    adapter_receipt = {
        "composite_identity_sha256": ADAPTER_SHA256,
        "wrapped_projection_count": 64,
    }
    plan = build_campaign_plan(
        "paired-grade-test",
        tasks,
        model_identity={
            "model_path": MODEL_PATH,
            "fingerprint": MODEL_SHA256,
            "method": "sha256",
            "files": 4,
            "runtime_bundle": {
                "logical_parameter_count": 32_763_876_352,
                "logical_parameter_count_basis": "architecture_config_logical",
                "bundle_sha256": MODEL_BUNDLE_SHA256,
            },
        },
        adapter_identity={
            "identity_receipt": adapter_receipt,
        },
        execution_config={
            "max_steps": 8,
            "decode_max_tokens": 256,
            "worker_task_material": "public_manifest_only",
            "answer_reveal_protocol": "sealed_outputs_then_issuer_reveal_v1",
            "worker_origin_protocol": "detached_supervisor_staged_arm_import_v3",
            "worker_origin_attempt_slots": 3,
            "generation_seed_count": 20,
            "generation_seed_min_entropy_bits": 60,
            "generation_seed_policy": "external_issuer_uniform_63bit",
            "generation_seed_disclosure": "post_seal_answer_reveal",
            "implementation_sha256": {
                "tools/run_latent_cortex_paired_campaign.py": RUNNER_SHA256,
            },
        },
        contamination_audit=contamination_audit,
        campaign_trust=_campaign_trust() if claim_eligible else None,
        claim_eligible=False,
    )
    if claim_eligible:
        document = plan.to_dict()
        metadata = document["metadata"]
        metadata["claim_eligible"] = True
        metadata["claim_scope"] = "resident same-checkpoint causal attribution"
        plan = type(plan).build(
            document["campaign_name"],
            [
                plan.cell_definition(cell_id)
                for cell_id in plan.cell_ids
            ],
            metadata=metadata,
        )
    return plan, tasks


def _records(plan, tasks, *, gain: bool = True):
    rows = []
    metadata = plan.to_dict()["metadata"]
    task_records = {
        task["task_id"]: task for task in metadata["task_manifest"]["tasks"]
    }
    issuer_tasks = {task.task_id: task for task in tasks}
    for cell_id in plan.cell_ids:
        definition = plan.cell_definition(cell_id)
        arm = definition["arm"]
        outcomes = {
            BASE_VANILLA: not gain,
            BASE_RLC: not gain,
            ADAPTER_VANILLA: not gain,
            ADAPTER_RLC: gain,
            BASE_EQUAL_COMPUTE: not gain,
            ADAPTER_EQUAL_COMPUTE: not gain,
        }
        issuer_task = issuer_tasks[definition["task_id"]]
        text = (
            "FINAL_ANSWER: "
            + json.dumps(
                issuer_task.reveal_for_verifier()["expected"],
                sort_keys=True,
                separators=(",", ":"),
            )
            if outcomes[arm]
            else "synthetic answer intentionally lacks the terminal marker"
        )
        result = {
            "arm": arm,
            "text": text,
            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "layer_apps": 10_000,
            "adapter_identity_sha256": (
                ADAPTER_SHA256 if arm.startswith("adapter_") else None
            ),
            "adapter_wrapped_projections": (
                64 if arm.startswith("adapter_") else 0
            ),
            "runtime_model_identity": {
                "worker_model_path": MODEL_PATH,
                "worker_model_parameter_count": 32_763_876_352,
                "worker_model_parameter_count_basis": (
                    "architecture_config_logical"
                ),
                "worker_source_sha256": RUNNER_SHA256,
                "worker_weight_fingerprint": MODEL_SHA256,
                "worker_weight_fingerprint_method": "sha256",
                "worker_weight_file_count": 4,
                "worker_runtime_bundle_sha256": MODEL_BUNDLE_SHA256,
                "worker_load_boundary_verified": True,
            },
            "runtime_adapter_identity": (
                metadata["adapter_identity"]["identity_receipt"]
                if arm.startswith("adapter_")
                else None
            ),
        }
        task = task_records[definition["task_id"]]
        score = issuer_task.score(text).to_dict()
        correct = score["correct"]
        verification = {
            "correct": correct,
            "score_receipt": score,
            "answer_commitment_sha256": task["answer_commitment_sha256"],
        }
        rows.append(
            {
                "cell_id": cell_id,
                "definition": definition,
                "result": result,
                "verification": verification,
                "commit": {
                    "result_sha256": hashlib.sha256(
                        canonical_json_bytes(result)
                    ).hexdigest(),
                    "verification_sha256": hashlib.sha256(
                        canonical_json_bytes(verification)
                    ).hexdigest(),
                },
            }
        )
    return rows


def test_plan_freezes_every_task_arm_and_is_deterministic():
    tasks, first = _plan()
    _, second = _plan()

    assert first.plan_sha256 == second.plan_sha256
    assert len(first.cell_ids) == len(tasks) * len(FULL_ARMS)
    metadata = first.to_dict()["metadata"]
    assert metadata["external_frontier_claim_eligible"] is False
    assert metadata["task_manifest"]["task_count"] == len(tasks)
    assert metadata["task_commitment"]["task_count"] == len(tasks)
    assert set(metadata["arm_execution_order"]) == set(FULL_ARMS)
    assert metadata["arm_execution_order"].index(BASE_EQUAL_COMPUTE) > metadata[
        "arm_execution_order"
    ].index(BASE_RLC)
    assert metadata["arm_execution_order"].index(ADAPTER_EQUAL_COMPUTE) > metadata[
        "arm_execution_order"
    ].index(ADAPTER_RLC)
    for arm in FULL_ARMS:
        ordinals = sorted(
            first.cell_definition(cell_id)["execution_ordinal_within_arm"]
            for cell_id in first.cell_ids
            if first.cell_definition(cell_id)["arm"] == arm
        )
        assert ordinals == list(range(len(tasks)))


def test_strong_2x2_gain_without_powered_noninferiority_stays_inconclusive():
    plan, tasks = _grade_plan()
    grade = grade_campaign(
        _records(plan, tasks, gain=True),
        plan=plan,
        issuer_tasks=tasks,
        trusted_contamination_root_sha256=TEST_TRUST_ROOT_SHA256,
        trusted_campaign_policy_sha256=TEST_POLICY_SHA256,
    )

    assert grade["verdict"] == "inconclusive"
    assert grade["claim_tier"] == "CONJECTURE"
    assert grade["reasons"] == ["gain_not_proven"]
    assert grade["interaction"]["lower"]["numerator"] > 0
    assert grade["frontier_claim_eligible"] is False
    assert grade["same_checkpoint_gain_claim_eligible"] is False
    assert (
        grade["interaction"]["one_sided_exact_sign_flip_p"]["numerator"] > 0
    )
    assert (
        grade["comparisons"]["adapter_effect_under_vanilla"]["evidence"][
            "all_families_noninferior"
        ]
        is False
    )


def test_regressing_adapter_is_refuted():
    plan, tasks = _grade_plan()
    grade = grade_campaign(
        _records(plan, tasks, gain=False),
        plan=plan,
        issuer_tasks=tasks,
        trusted_contamination_root_sha256=TEST_TRUST_ROOT_SHA256,
        trusted_campaign_policy_sha256=TEST_POLICY_SHA256,
    )

    assert grade["verdict"] == "gain_refuted"
    assert grade["claim_tier"] == "REFUTED"
    assert grade["interaction"]["upper"]["numerator"] <= 0


def test_missing_cell_stays_incomplete():
    plan, tasks = _grade_plan()
    rows = _records(plan, tasks, gain=True)
    grade = grade_campaign(
        rows[:-1],
        plan=plan,
        issuer_tasks=tasks,
        trusted_contamination_root_sha256=TEST_TRUST_ROOT_SHA256,
        trusted_campaign_policy_sha256=TEST_POLICY_SHA256,
    )

    assert grade["verdict"] == "incomplete"
    assert grade["reasons"] == ["campaign_incomplete"]


def test_positive_preflight_without_noninferiority_cannot_emit_gain_verdict():
    plan, tasks = _grade_plan(claim_eligible=False)
    grade = grade_campaign(
        _records(plan, tasks, gain=True),
        plan=plan,
        issuer_tasks=tasks,
    )

    assert grade["verdict"] == "inconclusive"
    assert grade["claim_tier"] == "CONJECTURE"
    assert grade["reasons"] == ["gain_not_proven"]


@pytest.mark.parametrize("trusted_root", [None, "0" * 64])
def test_claim_grade_requires_out_of_band_pinned_contamination_root(trusted_root):
    plan, tasks = _grade_plan(claim_eligible=True)

    with pytest.raises(
        PairedCampaignError,
        match="campaign_contamination_trust_root_required",
    ):
        grade_campaign(
            _records(plan, tasks),
            plan=plan,
            issuer_tasks=tasks,
            trusted_contamination_root_sha256=trusted_root,
            trusted_campaign_policy_sha256=TEST_POLICY_SHA256,
        )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda row: row["definition"].__setitem__("domain", "not-a-domain"),
            "campaign_record_definition_mismatch",
        ),
        (
            lambda row: row["verification"].__setitem__(
                "correct", not row["verification"]["correct"]
            ),
            "campaign_score_binding_invalid",
        ),
        (
            lambda row: row["result"].__setitem__("layer_apps", 9_999),
            "campaign_result_commitment_mismatch",
        ),
        (
            lambda row: row["result"]["runtime_model_identity"].__setitem__(
                "worker_model_parameter_count", 1
            ),
            "campaign_runtime_model_identity_mismatch",
        ),
        (
            lambda row: row["result"]["runtime_model_identity"].__setitem__(
                "worker_runtime_bundle_sha256", "0" * 64
            ),
            "campaign_runtime_model_identity_mismatch",
        ),
    ],
)
def test_grader_rejects_unbound_or_mutated_evidence(mutation, reason):
    plan, tasks = _grade_plan()
    rows = _records(plan, tasks)
    mutation(rows[0])

    with pytest.raises(PairedCampaignError, match=reason):
        grade_campaign(
            rows,
            plan=plan,
            issuer_tasks=tasks,
            trusted_contamination_root_sha256=TEST_TRUST_ROOT_SHA256,
            trusted_campaign_policy_sha256=TEST_POLICY_SHA256,
        )


def test_grader_rejects_bool_aliases_in_canonical_evidence():
    plan, tasks = _grade_plan()
    rows = _records(plan, tasks)
    ordinal_row = next(
        row
        for row in rows
        if row["definition"]["execution_ordinal_within_arm"] == 1
    )
    ordinal_row["definition"]["execution_ordinal_within_arm"] = True
    with pytest.raises(
        PairedCampaignError,
        match="campaign_record_definition_mismatch",
    ):
        grade_campaign(rows, plan=plan, issuer_tasks=tasks)

    rows = _records(plan, tasks)
    base_row = next(
        row
        for row in rows
        if row["definition"]["arm"] == BASE_VANILLA
    )
    base_row["result"]["adapter_wrapped_projections"] = False
    base_row["commit"]["result_sha256"] = hashlib.sha256(
        canonical_json_bytes(base_row["result"])
    ).hexdigest()
    with pytest.raises(
        PairedCampaignError,
        match="campaign_base_arm_adapter_contaminated",
    ):
        grade_campaign(rows, plan=plan, issuer_tasks=tasks)


def test_grader_rejects_malformed_plan_coverage():
    original, tasks = _grade_plan()
    document = original.to_dict()
    definitions = [
        original.cell_definition(cell_id) for cell_id in original.cell_ids[:-1]
    ]
    malformed = type(original).build(
        document["campaign_name"],
        definitions,
        metadata=document["metadata"],
    )

    with pytest.raises(PairedCampaignError, match="campaign_plan_coverage_invalid"):
        grade_campaign(
            [],
            plan=malformed,
            issuer_tasks=tasks,
            trusted_contamination_root_sha256=TEST_TRUST_ROOT_SHA256,
            trusted_campaign_policy_sha256=TEST_POLICY_SHA256,
        )


def test_independent_scorer_rejects_self_consistent_forged_correctness():
    plan, tasks = _grade_plan()
    rows = _records(plan, tasks)
    row = rows[0]
    row["verification"]["correct"] = not row["verification"]["correct"]
    row["verification"]["score_receipt"]["correct"] = row["verification"][
        "correct"
    ]
    row["commit"]["verification_sha256"] = hashlib.sha256(
        canonical_json_bytes(row["verification"])
    ).hexdigest()

    with pytest.raises(PairedCampaignError, match="campaign_score_binding_invalid"):
        grade_campaign(
            rows,
            plan=plan,
            issuer_tasks=tasks,
            trusted_contamination_root_sha256=TEST_TRUST_ROOT_SHA256,
            trusted_campaign_policy_sha256=TEST_POLICY_SHA256,
        )


def test_claim_eligible_plan_requires_bound_contamination_audit():
    tasks = generate_task_battery(
        range(20), domains=FRONTIER_DOMAINS, difficulty=2
    )
    with pytest.raises(
        PairedCampaignError, match="campaign_contamination_audit_required"
    ):
        build_campaign_plan(
            "missing-contamination-audit",
            tasks,
            model_identity={"checkpoint_fingerprint": "a" * 64},
            adapter_identity={"composite_identity_sha256": "b" * 64},
            execution_config={"max_steps": 8},
            claim_eligible=True,
        )


def test_claim_eligible_plan_rejects_tampered_audit_signature():
    tasks = generate_task_battery(
        range(20), domains=FRONTIER_DOMAINS, difficulty=2
    )
    manifest = build_task_manifest(tasks)
    audit = _signed_contamination_audit(manifest.manifest_sha256)
    audit["overlap_count"] = 1

    with pytest.raises(
        PairedCampaignError, match="campaign_contamination_audit_required"
    ):
        build_campaign_plan(
            "tampered-contamination-audit",
            tasks,
            model_identity={"checkpoint_fingerprint": "a" * 64},
            adapter_identity={"composite_identity_sha256": "b" * 64},
            execution_config={"max_steps": 8},
            contamination_audit=audit,
            claim_eligible=True,
        )


def test_claim_eligible_plan_requires_prelaunch_role_trust():
    tasks = generate_task_battery([7], domains=("mathematics",), difficulty=1)
    manifest = build_task_manifest(tasks)
    audit = _signed_contamination_audit(manifest.manifest_sha256)

    with pytest.raises(
        PairedCampaignError, match="campaign_prelaunch_trust_required"
    ):
        build_campaign_plan(
            "missing-role-trust",
            tasks,
            model_identity={"checkpoint_fingerprint": "a" * 64},
            adapter_identity={"composite_identity_sha256": "b" * 64},
            execution_config={"max_steps": 8},
            contamination_audit=audit,
            claim_eligible=True,
        )


def test_claim_eligible_plan_rejects_worker_visible_generation_seeds():
    tasks = generate_task_battery([7], domains=("mathematics",), difficulty=1)
    manifest = build_task_manifest(tasks)
    audit = _signed_contamination_audit(manifest.manifest_sha256)
    execution_config = {
        "worker_task_material": "public_manifest_only",
        "answer_reveal_protocol": "sealed_outputs_then_issuer_reveal_v1",
        "generation_seeds": [7],
        "generation_seed_count": 1,
        "generation_seed_min_entropy_bits": 60,
        "generation_seed_policy": "external_issuer_uniform_63bit",
        "generation_seed_disclosure": "post_seal_answer_reveal",
    }
    unsigned = build_campaign_plan(
        "seed-leak",
        tasks,
        model_identity={"checkpoint_fingerprint": "a" * 64},
        adapter_identity={"composite_identity_sha256": "b" * 64},
        execution_config=execution_config,
        contamination_audit=audit,
        claim_eligible=False,
    )
    trust = {
        "prelaunch_verified": True,
        "externally_custodied": True,
        "policy_sha256": "c" * 64,
        "unsigned_plan_sha256": unsigned.plan_sha256,
    }

    with pytest.raises(
        PairedCampaignError, match="campaign_answer_blinding_required"
    ):
        build_campaign_plan(
            "seed-leak",
            tasks,
            model_identity={"checkpoint_fingerprint": "a" * 64},
            adapter_identity={"composite_identity_sha256": "b" * 64},
            execution_config=execution_config,
            contamination_audit=audit,
            campaign_trust=trust,
            claim_eligible=True,
        )


def test_claim_eligible_plan_rejects_exact_but_underpowered_receipt():
    tasks = generate_task_battery(
        range(20),
        domains=FRONTIER_DOMAINS,
        difficulty=2,
    )
    manifest = build_task_manifest(tasks)
    audit = _signed_contamination_audit(manifest.manifest_sha256)
    power = exact_campaign_power_plan(
        domain_count=len(FRONTIER_DOMAINS),
        comparison_count=6,
        arm_count=len(FULL_ARMS),
        planned_observations_per_domain=20,
    )
    execution_config = {
        "worker_task_material": "public_manifest_only",
        "answer_reveal_protocol": "sealed_outputs_then_issuer_reveal_v1",
        "worker_origin_protocol": "detached_supervisor_staged_arm_import_v3",
        "worker_origin_attempt_slots": 3,
        "generation_seed_count": 20,
        "generation_seed_min_entropy_bits": 60,
        "generation_seed_policy": "external_issuer_uniform_63bit",
        "generation_seed_disclosure": "post_seal_answer_reveal",
        "domains": list(FRONTIER_DOMAINS),
        "exact_statistical_power": power,
    }

    assert power["powered_for_zero_loss_noninferiority"] is False
    with pytest.raises(
        PairedCampaignError,
        match="campaign_exact_power_required",
    ):
        build_campaign_plan(
            "underpowered-claim",
            tasks,
            model_identity={"checkpoint_fingerprint": "a" * 64},
            adapter_identity={"composite_identity_sha256": "b" * 64},
            execution_config=execution_config,
            contamination_audit=audit,
            campaign_trust=_campaign_trust(),
            claim_eligible=True,
        )


def test_claim_grade_requires_out_of_band_campaign_policy_pin():
    plan, tasks = _grade_plan(claim_eligible=True)

    with pytest.raises(
        PairedCampaignError, match="campaign_prelaunch_trust_required"
    ):
        grade_campaign(
            [],
            plan=plan,
            issuer_tasks=tasks,
            trusted_contamination_root_sha256=TEST_TRUST_ROOT_SHA256,
        )


def test_grader_rejects_forged_underpowered_claim_plan():
    plan, tasks = _grade_plan(claim_eligible=True)

    with pytest.raises(
        PairedCampaignError,
        match="campaign_exact_power_required",
    ):
        grade_campaign(
            [],
            plan=plan,
            issuer_tasks=tasks,
            trusted_contamination_root_sha256=TEST_TRUST_ROOT_SHA256,
            trusted_campaign_policy_sha256=TEST_POLICY_SHA256,
        )


def test_grader_rejects_planned_adapter_identity_reported_without_loaded_receipt():
    plan, tasks = _grade_plan()
    rows = _records(plan, tasks)
    row = next(
        item for item in rows if item["definition"]["arm"] == ADAPTER_RLC
    )
    row["result"]["runtime_adapter_identity"] = {
        **row["result"]["runtime_adapter_identity"],
        "wrapped_projection_count": 63,
    }
    row["commit"]["result_sha256"] = hashlib.sha256(
        canonical_json_bytes(row["result"])
    ).hexdigest()

    with pytest.raises(PairedCampaignError, match="campaign_adapter_activation_mismatch"):
        grade_campaign(
            rows,
            plan=plan,
            issuer_tasks=tasks,
            trusted_contamination_root_sha256=TEST_TRUST_ROOT_SHA256,
            trusted_campaign_policy_sha256=TEST_POLICY_SHA256,
        )
