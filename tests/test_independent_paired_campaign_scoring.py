from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from core.brain.llm.latent_cortex import paired_campaign as production_campaign
from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.exact_paired_grade import (
    _bound_payload,
    exact_campaign_power_plan,
)
from core.brain.llm.latent_cortex.exact_paired_statistics import (
    Rational,
    certified_rational_effect_bounds,
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
    build_campaign_plan,
    grade_campaign,
)
from tests.fixtures.latent_frontier import _trial_accounting
from tools import independent_paired_campaign_scoring as independent_kernel
from tools.independent_paired_campaign_scoring import (
    _Q,
    IndependentScoringError,
    _canonical_bytes,
    _counts_at_most,
    _effect_bounds,
    _exact_campaign_power_plan,
    _holm,
    _proportion_bound,
    _q_payload,
    _sign_flip,
    _verify_ed25519,
    _within_compute,
    independent_grade_campaign,
)


def test_independent_sft_projection_count_is_manifest_bound() -> None:
    paths = [f"model.layers.40.self_attn.q_proj.{index}" for index in range(24)]
    identity = {
        "format": independent_kernel.RESIDENT_RECURRENT_SFT_MANIFEST_SCHEMA,
        "manifest": {
            "schema": independent_kernel.RESIDENT_RECURRENT_SFT_MANIFEST_SCHEMA,
            "lora": {"wrapped_projections": 24, "projection_paths": paths},
        },
    }
    receipt = {"schema": independent_kernel.RESIDENT_RECURRENT_SFT_RECEIPT_SCHEMA}

    assert independent_kernel._independent_adapter_projection_count(identity, receipt) == 24
    identity["manifest"]["lora"]["wrapped_projections"] = 23
    assert independent_kernel._independent_adapter_projection_count(identity, receipt) is None


MODEL_PATH = "/sealed/resident-32b"
MODEL_SHA256 = "a" * 64
ADAPTER_SHA256 = "b" * 64
RUNNER_SHA256 = "c" * 64
MODEL_BUNDLE_SHA256 = "d" * 64
POLICY_SHA256 = "f" * 64


def _plan_and_tasks():
    tasks = generate_task_battery(
        [701, 702],
        domains=("mathematics", "coding"),
        difficulty=2,
    )
    plan = build_campaign_plan(
        "independent-exact-parity",
        tasks,
        model_identity={
            "model_path": MODEL_PATH,
            "fingerprint": MODEL_SHA256,
            "method": "sha256",
            "files": 4,
            "runtime_bundle": {
                "logical_parameter_count": 32_763_876_352,
                "logical_parameter_count_basis": ("architecture_config_logical"),
                "bundle_sha256": MODEL_BUNDLE_SHA256,
            },
        },
        adapter_identity={
            "identity_receipt": {
                "composite_identity_sha256": ADAPTER_SHA256,
                "wrapped_projection_count": 64,
            }
        },
        execution_config={
            "max_steps": 8,
            "decode_max_tokens": 256,
            "implementation_sha256": {
                "tools/run_latent_cortex_paired_campaign.py": RUNNER_SHA256,
            },
        },
        claim_eligible=False,
    )
    return plan, tasks


def _records(plan, tasks):
    rows = []
    metadata = plan.to_dict()["metadata"]
    task_records = {task["task_id"]: task for task in metadata["task_manifest"]["tasks"]}
    issuer_tasks = {task.task_id: task for task in tasks}
    task_ordinals = {task_id: ordinal for ordinal, task_id in enumerate(sorted(issuer_tasks))}
    for cell_id in plan.cell_ids:
        definition = plan.cell_definition(cell_id)
        arm = definition["arm"]
        ordinal = task_ordinals[definition["task_id"]]
        outcomes = {
            BASE_VANILLA: False,
            BASE_RLC: ordinal % 2 == 0,
            ADAPTER_VANILLA: ordinal % 3 == 0,
            ADAPTER_RLC: True,
            BASE_EQUAL_COMPUTE: False,
            ADAPTER_EQUAL_COMPUTE: ordinal % 3 == 0,
        }
        issuer_task = issuer_tasks[definition["task_id"]]
        if outcomes[arm]:
            text = "FINAL_ANSWER: " + json.dumps(
                issuer_task.reveal_for_verifier()["expected"],
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        else:
            text = "synthetic incorrect answer"
        task = task_records[definition["task_id"]]
        resource_accounting, information_accounting = _trial_accounting(task["task_payload_sha256"])
        result = {
            "arm": arm,
            "text": text,
            "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "layer_apps": 10_000,
            "adapter_identity_sha256": (ADAPTER_SHA256 if arm.startswith("adapter_") else None),
            "adapter_wrapped_projections": (64 if arm.startswith("adapter_") else 0),
            "runtime_model_identity": {
                "worker_model_path": MODEL_PATH,
                "worker_model_parameter_count": 32_763_876_352,
                "worker_model_parameter_count_basis": ("architecture_config_logical"),
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
            "episode_receipt": (
                {
                    "budget": {
                        "resource_accounting": resource_accounting,
                        "information_accounting": information_accounting,
                    }
                }
                if arm.endswith("_rlc")
                else {}
            ),
            "resource_accounting": resource_accounting,
            "information_accounting": information_accounting,
        }
        score = issuer_task.score(text).to_dict()
        verification = {
            "correct": score["correct"],
            "score_receipt": score,
            "answer_commitment_sha256": task_records[definition["task_id"]][
                "answer_commitment_sha256"
            ],
        }
        rows.append(
            {
                "cell_id": cell_id,
                "definition": definition,
                "result": result,
                "verification": verification,
                "commit": {
                    "result_sha256": hashlib.sha256(canonical_json_bytes(result)).hexdigest(),
                    "verification_sha256": hashlib.sha256(
                        canonical_json_bytes(verification)
                    ).hexdigest(),
                },
            }
        )
    return rows


def _signed_contamination_audit(task_manifest_sha256: str):
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public_der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    trust_root = hashlib.sha256(public_der).hexdigest()
    body = {
        "schema": "aura.latent_cortex.contamination_audit.v2",
        "task_manifest_sha256": task_manifest_sha256,
        "status": "passed_zero_overlap",
        "overlap_count": 0,
        "auditor_independence": "external",
        "corpora": [
            {
                "name": "held-out-corpus",
                "snapshot_sha256": "e" * 64,
            }
        ],
        "methods": [
            "exact_prompt",
            "normalized_prompt",
            "token_fivegram",
        ],
    }
    payload = canonical_json_bytes(body)
    return (
        {
            **body,
            "signature": {
                "algorithm": "ed25519",
                "key_id": trust_root,
                "signature_b64": base64.b64encode(private_key.sign(payload)).decode("ascii"),
                "signed_payload_sha256": hashlib.sha256(payload).hexdigest(),
                "public_key_der_b64": base64.b64encode(public_der).decode("ascii"),
                "trust_root_sha256": trust_root,
                "verified": True,
            },
        },
        trust_root,
    )


def _claim_plan_and_tasks(power_receipt):
    tasks = generate_task_battery(
        [811],
        domains=FRONTIER_DOMAINS,
        difficulty=2,
    )
    manifest = build_task_manifest(tasks)
    audit, trust_root = _signed_contamination_audit(manifest.manifest_sha256)
    plan = build_campaign_plan(
        "independent-claim-eligibility",
        tasks,
        model_identity={
            "model_path": MODEL_PATH,
            "fingerprint": MODEL_SHA256,
            "method": "sha256",
            "files": 4,
            "runtime_bundle": {
                "logical_parameter_count": 32_763_876_352,
                "logical_parameter_count_basis": ("architecture_config_logical"),
                "bundle_sha256": MODEL_BUNDLE_SHA256,
            },
        },
        adapter_identity={
            "identity_receipt": {
                "composite_identity_sha256": ADAPTER_SHA256,
                "wrapped_projection_count": 64,
            }
        },
        execution_config={
            "max_steps": 8,
            "decode_max_tokens": 256,
            "worker_task_material": "public_manifest_only",
            "answer_reveal_protocol": "sealed_outputs_then_issuer_reveal_v1",
            "worker_origin_protocol": "detached_supervisor_staged_arm_import_v3",
            "worker_origin_attempt_slots": 3,
            "generation_seed_count": 1,
            "generation_seed_min_entropy_bits": 60,
            "generation_seed_policy": "external_issuer_uniform_63bit",
            "generation_seed_disclosure": "post_seal_answer_reveal",
            "domains": list(FRONTIER_DOMAINS),
            "exact_statistical_power": power_receipt,
            "implementation_sha256": {
                "tools/run_latent_cortex_paired_campaign.py": RUNNER_SHA256,
            },
        },
        contamination_audit=audit,
        campaign_trust={
            "prelaunch_verified": True,
            "externally_custodied": True,
            "policy_sha256": POLICY_SHA256,
            "unsigned_plan_sha256": "1" * 64,
        },
        claim_eligible=True,
    )
    return plan, tasks, trust_root


def _tiny_power_receipt():
    return {
        "schema": "aura.latent_cortex.exact_noninferiority_power.v1",
        "certified": True,
        "global_bound_family_count": 50,
        "margin": {"numerator": 1, "denominator": 50},
        "minimum_observations": 1,
        "selected_lower": {"numerator": 0, "denominator": 1},
        "selected_upper": {"numerator": 1, "denominator": 1},
        "prior_observations": None,
        "prior_lower": None,
        "prior_upper": None,
        "precision_bits": 40,
        "resource_max_observations": 4_096,
        "domain_count": 7,
        "comparison_count": 6,
        "planned_observations_per_domain": 1,
        "planned_total_tasks": 7,
        "planned_total_cells": 42,
        "powered_for_zero_loss_noninferiority": True,
    }


def test_rational_reduces_and_rejects_bool():
    assert _Q(18, 24) == _Q(3, 4)
    assert _Q(0, 99) == _Q(0, 1)

    with pytest.raises(
        IndependentScoringError,
        match="independent_rational_type_invalid",
    ):
        _Q(True, 2)


def test_holm_ties_use_exact_probability_then_ascii_name():
    adjusted, entries = _holm(
        {
            "zeta": _Q(1, 20),
            "alpha": _Q(2, 40),
            "middle": _Q(1, 10),
        }
    )

    assert [entry["hypothesis"] for entry in entries] == [
        "alpha",
        "zeta",
        "middle",
    ]
    assert adjusted["alpha"] == _Q(3, 20)
    assert adjusted["zeta"] == _Q(3, 20)
    assert adjusted["middle"] == _Q(3, 20)


def test_compute_tolerance_is_exact_at_boundary_and_rejects_bool():
    assert _within_compute(120, 100, _Q(1, 5)) is True
    assert _within_compute(80, 100, _Q(1, 5)) is True
    assert _within_compute(121, 100, _Q(1, 5)) is False
    assert _within_compute(79, 100, _Q(1, 5)) is False

    with pytest.raises(
        IndependentScoringError,
        match="independent_compute_invalid",
    ):
        _within_compute(True, 100, _Q(1, 5))


def test_clopper_pearson_dyadic_witnesses_are_outward():
    family_count = 17
    bounds = _effect_bounds(7, 2, 3, family_count)

    assert bounds["certified"] is True
    assert bounds["family_count"] == family_count
    for component in bounds["components"]:
        if component["tail_kind"] == "exact-boundary":
            assert component["tail_probability"] is None
            assert component["adjacent_bound"] is None
            continue
        selected = _Q(**component["tail_probability"])
        adjacent = _Q(**component["adjacent_tail_probability"])
        component_alpha = _Q(**component["component_alpha"])
        assert _counts_at_most(
            (selected.numerator, selected.denominator),
            component_alpha,
        )
        assert not _counts_at_most(
            (adjacent.numerator, adjacent.denominator),
            component_alpha,
        )


def test_clopper_pearson_endpoint_certificate_matches_direct_builder():
    component = _proportion_bound(
        "win_lower",
        0,
        12,
        "lower",
        _Q(1, 80),
        40,
    )

    assert component["bound"] == _q_payload(_Q(0, 1))
    assert component["tail_kind"] == "exact-boundary"
    assert component["certified"] is True


def test_twenty_tie_fixture_cannot_satisfy_strict_noninferiority_at_budget_50():
    bounds = _effect_bounds(0, 0, 20, 50)

    assert _Q(**bounds["lower"]).numerator * 50 <= -(_Q(**bounds["lower"]).denominator)


def test_513_no_loss_observations_can_certify_strict_noninferiority():
    bounds = _effect_bounds(0, 0, 513, 50)
    production = certified_rational_effect_bounds(
        0,
        0,
        513,
        family_count=50,
        family_alpha=Rational(1, 20),
        precision_bits=40,
    )

    lower = _Q(**bounds["lower"])
    assert bounds["observations"] == 513
    assert -lower.denominator < 50 * lower.numerator
    assert bounds == _bound_payload(production)


def test_exact_sign_flip_supports_more_than_512_observations():
    probability, certificate = _sign_flip([1] * 513)

    assert certificate["observations"] == 513
    assert certificate["observed_sum"] == 513
    assert certificate["total_assignments"] == 1 << 513
    assert probability == _Q(1, 1 << 513)


def test_standard_library_ed25519_verifier_accepts_valid_and_rejects_tamper():
    private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    public_der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    message = b"independent external trust root"
    signature = private_key.sign(message)

    assert _verify_ed25519(public_der, signature, message) is True
    tampered = bytearray(signature)
    tampered[0] ^= 1
    assert _verify_ed25519(public_der, bytes(tampered), message) is False


def test_claim_eligibility_is_independently_derived_for_incomplete_grade(
    monkeypatch,
):
    power_receipt = _tiny_power_receipt()
    monkeypatch.setattr(
        production_campaign,
        "exact_campaign_power_plan",
        lambda **_kwargs: power_receipt,
    )
    monkeypatch.setattr(
        independent_kernel,
        "_exact_campaign_power_plan",
        lambda **_kwargs: power_receipt,
    )
    plan, tasks, trust_root = _claim_plan_and_tasks(power_receipt)

    independent = independent_grade_campaign(
        [],
        plan=plan,
        issuer_tasks=tasks,
        trusted_contamination_root_sha256=trust_root,
        trusted_campaign_policy_sha256=POLICY_SHA256,
    )

    assert independent["semantic_grade"]["same_checkpoint_gain_claim_eligible"] is True
    assert independent["semantic_grade"]["verdict"] == "incomplete"

    document = plan.to_dict()
    document["metadata"]["execution_config"]["exact_statistical_power"]["planned_total_cells"] += 1
    drifted = type(plan).build(
        document["campaign_name"],
        [plan.cell_definition(cell_id) for cell_id in plan.cell_ids],
        metadata=document["metadata"],
    )
    with pytest.raises(
        IndependentScoringError,
        match="independent_claim_power_invalid",
    ):
        independent_grade_campaign(
            [],
            plan=drifted,
            issuer_tasks=tasks,
            trusted_contamination_root_sha256=trust_root,
            trusted_campaign_policy_sha256=POLICY_SHA256,
        )

    with pytest.raises(
        IndependentScoringError,
        match="independent_claim_trust_invalid",
    ):
        independent_grade_campaign(
            [],
            plan=plan,
            issuer_tasks=tasks,
            trusted_contamination_root_sha256="0" * 64,
            trusted_campaign_policy_sha256=POLICY_SHA256,
        )


def test_independent_power_boundary_matches_production_byte_for_byte():
    for observations, powered in ((410, False), (411, True)):
        production = exact_campaign_power_plan(
            domain_count=7,
            comparison_count=6,
            arm_count=6,
            planned_observations_per_domain=observations,
        )
        independent = _exact_campaign_power_plan(
            domain_count=7,
            comparison_count=6,
            arm_count=6,
            planned_observations_per_domain=observations,
        )
        assert independent == production
        assert _canonical_bytes(independent) == canonical_json_bytes(production)
        assert independent["powered_for_zero_loss_noninferiority"] is powered


def test_independent_complete_semantic_tree_matches_production_byte_for_byte():
    plan, tasks = _plan_and_tasks()
    records = _records(plan, tasks)

    production = grade_campaign(records, plan=plan, issuer_tasks=tasks)
    independent = independent_grade_campaign(
        records,
        plan=plan,
        issuer_tasks=tasks,
    )

    assert independent["semantic_grade"] == production
    assert _canonical_bytes(independent["semantic_grade"]) == canonical_json_bytes(production)
    assert (
        independent["implementation_sha256"]
        == hashlib.sha256(
            Path(independent_grade_campaign.__code__.co_filename).read_bytes()
        ).hexdigest()
    )
    assert (
        independent["semantic_grade_canonical_sha256"]
        == hashlib.sha256(canonical_json_bytes(production)).hexdigest()
    )


def test_independent_kernel_reconstructs_inner_resource_ledger_after_outer_rehash():
    plan, tasks = _plan_and_tasks()
    records = _records(plan, tasks)
    result = records[0]["result"]
    result["resource_accounting"]["totals"]["transformer_layer_apps"] += 1
    records[0]["commit"]["result_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()

    with pytest.raises(
        IndependentScoringError,
        match="independent_resource_accounting_invalid",
    ):
        independent_grade_campaign(records, plan=plan, issuer_tasks=tasks)


def test_independent_valid_incomplete_tree_matches_production_byte_for_byte():
    plan, tasks = _plan_and_tasks()
    records = _records(plan, tasks)[:-1]

    production = grade_campaign(records, plan=plan, issuer_tasks=tasks)
    independent = independent_grade_campaign(
        records,
        plan=plan,
        issuer_tasks=tasks,
    )

    assert production["verdict"] == "incomplete"
    assert independent["semantic_grade"] == production
    assert _canonical_bytes(independent["semantic_grade"]) == canonical_json_bytes(production)


def test_independent_rejects_bool_compute_in_raw_evidence():
    plan, tasks = _plan_and_tasks()
    records = _records(plan, tasks)
    records[0]["result"]["layer_apps"] = True
    records[0]["commit"]["result_sha256"] = hashlib.sha256(
        canonical_json_bytes(records[0]["result"])
    ).hexdigest()

    with pytest.raises(
        IndependentScoringError,
        match="independent_result_invalid",
    ):
        independent_grade_campaign(
            records,
            plan=plan,
            issuer_tasks=tasks,
        )


def test_independent_rejects_bool_aliases_in_plan_and_adapter_evidence():
    plan, tasks = _plan_and_tasks()
    records = _records(plan, tasks)
    ordinal_row = next(
        row for row in records if row["definition"]["execution_ordinal_within_arm"] == 1
    )
    ordinal_row["definition"]["execution_ordinal_within_arm"] = True
    with pytest.raises(
        IndependentScoringError,
        match="independent_record_definition_mismatch",
    ):
        independent_grade_campaign(
            records,
            plan=plan,
            issuer_tasks=tasks,
        )

    records = _records(plan, tasks)
    base_row = next(row for row in records if row["definition"]["arm"] == BASE_VANILLA)
    base_row["result"]["adapter_wrapped_projections"] = False
    base_row["commit"]["result_sha256"] = hashlib.sha256(
        canonical_json_bytes(base_row["result"])
    ).hexdigest()
    with pytest.raises(
        IndependentScoringError,
        match="independent_base_arm_adapter_contaminated",
    ):
        independent_grade_campaign(
            records,
            plan=plan,
            issuer_tasks=tasks,
        )
