"""Contract tests: independent raw-output/compute verification of a paired
campaign from raw disk artifacts only.

The verifier must regenerate the task battery from the plan's declared
generation parameters (a doctored plan cannot smuggle tasks), replay the
hash-chained journal, re-grade from committed evidence, agree with an
honest published grade, and fail closed on tampered plans, journals, or
forged grades.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignJournal,
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.frontier_tasks import (
    CURRENT_REGISTRY_VERSION,
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
from core.brain.llm.latent_cortex.sequential_campaign_evidence import (
    build_sequential_look_certificate,
)
from tests.fixtures.latent_frontier import _trial_accounting
from tools import run_latent_cortex_paired_campaign as campaign_runner_module
from tools import verify_paired_campaign_evidence as verifier_module
from tools.independent_paired_campaign_scoring import (
    independent_grade_campaign,
)
from tools.verify_paired_campaign_evidence import (
    GRADE_FILE,
    JOURNAL_FILE,
    MANIFEST_FILE,
    PLAN_FILE,
    verify_campaign_evidence,
)

MODEL_PATH = "/sealed/resident-32b"
RUNNER_SHA256 = "c" * 64
ADAPTER_SHA256 = "b" * 64
MODEL_SHA256 = "a" * 64
MODEL_BUNDLE_SHA256 = "d" * 64
SEEDS = (515151,)
DOMAINS = ("mathematics", "coding")
DIFFICULTY = 1


def _tasks():
    return generate_task_battery(
        SEEDS,
        domains=DOMAINS,
        difficulty=DIFFICULTY,
        registry_version=CURRENT_REGISTRY_VERSION,
    )


def _plan(tasks, *, answer_reveal: bool = False):
    return build_campaign_plan(
        "independent-verify-test",
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
            "identity_receipt": {
                "composite_identity_sha256": ADAPTER_SHA256,
                "wrapped_projection_count": 64,
            },
        },
        execution_config={
            "max_steps": 8,
            "decode_max_tokens": 256,
            "difficulty": DIFFICULTY,
            "task_registry_version": CURRENT_REGISTRY_VERSION,
            "domains": list(DOMAINS),
            "implementation_sha256": {
                "tools/run_latent_cortex_paired_campaign.py": RUNNER_SHA256,
            },
            **(
                {
                    "worker_task_material": "public_manifest_only",
                    "answer_reveal_protocol": "sealed_outputs_then_issuer_reveal_v1",
                    "generation_seed_count": len(SEEDS),
                    "generation_seed_min_entropy_bits": min(
                        seed.bit_length() for seed in SEEDS
                    ),
                    "generation_seed_policy": "external_issuer_uniform_63bit",
                    "generation_seed_disclosure": "post_seal_answer_reveal",
                }
                if answer_reveal
                else {"generation_seeds": list(SEEDS)}
            ),
        },
    )


def _record_material(plan, tasks, cell_id, *, gain: bool = True):
    metadata = plan.to_dict()["metadata"]
    task_records = {
        task["task_id"]: task for task in metadata["task_manifest"]["tasks"]
    }
    issuer_tasks = {task.task_id: task for task in tasks}
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
    if outcomes[arm]:
        text = "FINAL_ANSWER: " + json.dumps(
            issuer_task.reveal_for_verifier()["expected"],
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        text = "synthetic answer intentionally lacks the terminal marker"
    task = task_records[definition["task_id"]]
    resource_accounting, information_accounting = _trial_accounting(
        task["task_payload_sha256"]
    )
    result = {
        "arm": arm,
        "text": text,
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "layer_apps": 10_000,
        "adapter_identity_sha256": (
            ADAPTER_SHA256 if arm.startswith("adapter_") else None
        ),
        "adapter_wrapped_projections": 64 if arm.startswith("adapter_") else 0,
        "runtime_model_identity": {
            "worker_model_path": MODEL_PATH,
            "worker_model_parameter_count": 32_763_876_352,
            "worker_model_parameter_count_basis": "architecture_config_logical",
            "worker_source_sha256": RUNNER_SHA256,
            "worker_weight_fingerprint": MODEL_SHA256,
            "worker_weight_fingerprint_method": "sha256",
            "worker_weight_file_count": 4,
            "worker_runtime_bundle_sha256": MODEL_BUNDLE_SHA256,
            "worker_load_boundary_verified": True,
        },
        "runtime_adapter_identity": (
            plan.to_dict()["metadata"]["adapter_identity"]["identity_receipt"]
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
        "answer_commitment_sha256": task["answer_commitment_sha256"],
    }
    commit = {
        "result_sha256": hashlib.sha256(
            canonical_json_bytes(result)
        ).hexdigest(),
        "verification_sha256": hashlib.sha256(
            canonical_json_bytes(verification)
        ).hexdigest(),
    }
    return result, verification, commit


def _build_campaign_dir(
    tmp_path: Path, *, gain: bool = True, answer_reveal: bool = False
):
    tasks = _tasks()
    plan = _plan(tasks, answer_reveal=answer_reveal)
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / PLAN_FILE).write_bytes(
        canonical_json_bytes(plan.to_dict()) + b"\n"
    )
    with CampaignJournal(campaign_dir / JOURNAL_FILE, plan) as journal:
        for cell_id in plan.cell_ids:
            result, verification, commit = _record_material(
                plan, tasks, cell_id, gain=gain
            )
            attempt_id = journal.start_cell(cell_id)
            journal.record_arm_result(cell_id, attempt_id, result)
            journal.record_verified(cell_id, attempt_id, verification)
            journal.commit_cell(cell_id, attempt_id, commit)
        records = journal.committed_records()
        manifest = journal.finalize(campaign_dir / MANIFEST_FILE)
    if answer_reveal:
        sealed = campaign_runner_module._seal_output_manifest(campaign_dir, plan)
        reveal = campaign_runner_module._admit_answer_reveal(
            type(
                "Args",
                (),
                {
                    "campaign_dir": str(campaign_dir),
                    "answer_reveal_attestation": "",
                },
            )(),
            plan,
            tasks,
            sealed,
        )
        assert reveal is not None
    return campaign_dir, plan, tasks, records, manifest


def _publish_grade(campaign_dir, plan, tasks, records, manifest) -> dict:
    grade = grade_campaign(records, plan=plan, issuer_tasks=tasks)
    material = dict(grade)
    material.pop("grade_sha256", None)
    material["campaign_manifest_sha256"] = manifest["manifest_sha256"]
    sealed_path = campaign_dir / campaign_runner_module.SEALED_OUTPUT_MANIFEST_FILE
    reveal_path = campaign_dir / campaign_runner_module.ANSWER_REVEAL_FILE
    if sealed_path.exists() and reveal_path.exists():
        material["sealed_output_manifest_sha256"] = json.loads(
            sealed_path.read_bytes()
        )["manifest_sha256"]
        material["answer_reveal_sha256"] = json.loads(reveal_path.read_bytes())[
            "reveal_sha256"
        ]
    final = {
        **material,
        "grade_sha256": hashlib.sha256(
            canonical_json_bytes(material)
        ).hexdigest(),
    }
    (campaign_dir / GRADE_FILE).write_bytes(
        canonical_json_bytes(final) + b"\n"
    )
    return final


def _build_sequential_verifier_fixture(tmp_path: Path, monkeypatch):
    tasks = [
        {"task_id": f"{domain}-{ordinal}", "domain": domain}
        for domain in ("coding", "mathematics")
        for ordinal in (1, 2)
    ]
    cells = [
        {**task, "arm": arm}
        for task in tasks
        for arm in ("base_vanilla", "adapter_rlc")
    ]
    power_looks = [
        {"look": 1, "family_alpha": {"numerator": 1, "denominator": 200}},
        {"look": 2, "family_alpha": {"numerator": 9, "denominator": 200}},
    ]
    plan = CampaignPlan.build(
        "sequential-independent-verifier-test",
        cells,
        metadata={
            "task_manifest": {"tasks": tasks},
            "execution_config": {
                "sequential_look_observations_per_domain": [1, 2],
                "exact_statistical_power": {
                    "schema": "aura.latent_cortex.exact_group_sequential_power.v1",
                    "looks": power_looks,
                },
            },
        },
    )
    records = []
    for index, cell_id in enumerate(plan.cell_ids):
        records.append(
            {
                "cell_id": cell_id,
                "definition": plan.cell_definition(cell_id),
                "commit": {
                    "result_sha256": f"{index:064x}",
                    "verification_sha256": f"{index + 100:064x}",
                },
            }
        )

    def fake_grade(records, *, plan, family_alpha, included_task_ids, **_kwargs):
        alpha = {
            "numerator": family_alpha.numerator,
            "denominator": family_alpha.denominator,
        }
        return {
            "plan_sha256": plan.plan_sha256,
            "expected_task_count": len(included_task_ids),
            "expected_cell_count": len(records),
            "statistical_policy": {"alpha": alpha},
            "verdict": "inconclusive",
        }

    def fake_independent(
        records,
        *,
        plan,
        family_alpha,
        included_task_ids,
        **_kwargs,
    ):
        grade = {
            "plan_sha256": plan.plan_sha256,
            "expected_task_count": len(included_task_ids),
            "expected_cell_count": len(records),
            "statistical_policy": {"alpha": dict(family_alpha)},
            "verdict": "inconclusive",
        }
        return {
            "semantic_grade": grade,
            "semantic_grade_canonical_sha256": hashlib.sha256(
                canonical_json_bytes(grade)
            ).hexdigest(),
            "implementation_sha256": "e" * 64,
        }

    monkeypatch.setattr(verifier_module, "grade_campaign", fake_grade)
    monkeypatch.setattr(
        verifier_module,
        "independent_grade_campaign",
        fake_independent,
    )
    look_dir = tmp_path / verifier_module.SEQUENTIAL_LOOK_DIR
    look_dir.mkdir()
    previous_sha256 = None
    certificates = []
    for look, power in enumerate(power_looks, 1):
        task_ids = {
            task["task_id"]
            for task in tasks
            if int(task["task_id"].rsplit("-", 1)[1]) <= look
        }
        scoped = [
            record
            for record in records
            if record["definition"]["task_id"] in task_ids
        ]
        alpha = power["family_alpha"]
        production = {
            "plan_sha256": plan.plan_sha256,
            "expected_task_count": len(task_ids),
            "expected_cell_count": len(scoped),
            "statistical_policy": {"alpha": alpha},
            "verdict": "inconclusive",
        }
        certificate = build_sequential_look_certificate(
            plan=plan,
            look=look,
            committed_records=scoped,
            production_grade=production,
            independent_grade=production,
            previous_certificate_sha256=previous_sha256,
        )
        (look_dir / f"look-{look:03d}.json").write_bytes(
            canonical_json_bytes(certificate) + b"\n"
        )
        certificates.append(certificate)
        previous_sha256 = certificate["certificate_sha256"]
    return plan, records, tasks, certificates


def _rewrite_self_hashed_certificate(path: Path, mutate) -> None:
    certificate = json.loads(path.read_bytes())
    material = dict(certificate)
    material.pop("certificate_sha256")
    mutate(material)
    certificate = {
        **material,
        "certificate_sha256": hashlib.sha256(
            canonical_json_bytes(material)
        ).hexdigest(),
    }
    path.write_bytes(canonical_json_bytes(certificate) + b"\n")


def test_independent_verifier_reconstructs_complete_sequential_chain(
    tmp_path: Path,
    monkeypatch,
):
    plan, records, tasks, certificates = _build_sequential_verifier_fixture(
        tmp_path,
        monkeypatch,
    )

    failures, detail = verifier_module._verify_sequential_look_chain(
        tmp_path,
        plan=plan,
        records=records,
        tasks=tasks,
        trusted_contamination_root_sha256=None,
        trusted_campaign_policy_sha256=None,
    )

    assert failures == []
    assert detail["verified"] is True
    assert detail["verified_look_count"] == 2
    assert detail["certificate_head_sha256"] == certificates[-1][
        "certificate_sha256"
    ]
    assert detail["decisions"] == ["continue", "terminal_inconclusive"]


@pytest.mark.parametrize("artifact_failure", ["missing", "extra"])
def test_sequential_verifier_rejects_artifact_set_drift(
    tmp_path: Path,
    monkeypatch,
    artifact_failure: str,
):
    plan, records, tasks, _certificates = _build_sequential_verifier_fixture(
        tmp_path,
        monkeypatch,
    )
    look_dir = tmp_path / verifier_module.SEQUENTIAL_LOOK_DIR
    if artifact_failure == "missing":
        (look_dir / "look-002.json").unlink()
    else:
        (look_dir / "look-003.json").write_bytes(b"{}\n")

    failures, detail = verifier_module._verify_sequential_look_chain(
        tmp_path,
        plan=plan,
        records=records,
        tasks=tasks,
        trusted_contamination_root_sha256=None,
        trusted_campaign_policy_sha256=None,
    )

    assert detail["verified"] is False
    assert any("artifact set differs" in failure for failure in failures)


@pytest.mark.parametrize(
    ("mutation", "difference_path"),
    [
        (
            lambda material: material.__setitem__(
                "decision",
                "positive_boundary_crossed",
            ),
            "decision",
        ),
        (
            lambda material: material.__setitem__(
                "previous_certificate_sha256",
                "0" * 64,
            ),
            "previous_certificate_sha256",
        ),
        (
            lambda material: material["look_power_receipt"].__setitem__(
                "family_alpha",
                {"numerator": 1, "denominator": 100},
            ),
            "look_power_receipt",
        ),
    ],
)
def test_sequential_verifier_rejects_self_rehashed_semantic_tampering(
    tmp_path: Path,
    monkeypatch,
    mutation,
    difference_path: str,
):
    plan, records, tasks, _certificates = _build_sequential_verifier_fixture(
        tmp_path,
        monkeypatch,
    )
    _rewrite_self_hashed_certificate(
        tmp_path / verifier_module.SEQUENTIAL_LOOK_DIR / "look-002.json",
        mutation,
    )

    failures, detail = verifier_module._verify_sequential_look_chain(
        tmp_path,
        plan=plan,
        records=records,
        tasks=tasks,
        trusted_contamination_root_sha256=None,
        trusted_campaign_policy_sha256=None,
    )

    assert detail["verified"] is False
    assert any(difference_path in failure for failure in failures)


def test_sequential_verifier_rejects_self_rehashed_cell_substitution(
    tmp_path: Path,
    monkeypatch,
):
    plan, records, tasks, _certificates = _build_sequential_verifier_fixture(
        tmp_path,
        monkeypatch,
    )

    def substitute_cell(material):
        material["record_receipts"][0]["cell_id"] = "cell-" + "f" * 64
        material["record_receipts_sha256"] = hashlib.sha256(
            canonical_json_bytes(material["record_receipts"])
        ).hexdigest()

    _rewrite_self_hashed_certificate(
        tmp_path / verifier_module.SEQUENTIAL_LOOK_DIR / "look-002.json",
        substitute_cell,
    )

    failures, detail = verifier_module._verify_sequential_look_chain(
        tmp_path,
        plan=plan,
        records=records,
        tasks=tasks,
        trusted_contamination_root_sha256=None,
        trusted_campaign_policy_sha256=None,
    )

    assert detail["verified"] is False
    assert any("record_receipts" in failure for failure in failures)


def test_sequential_verifier_rejects_independent_kernel_divergence(
    tmp_path: Path,
    monkeypatch,
):
    plan, records, tasks, _certificates = _build_sequential_verifier_fixture(
        tmp_path,
        monkeypatch,
    )
    honest_kernel = verifier_module.independent_grade_campaign

    def divergent_kernel(*args, **kwargs):
        result = honest_kernel(*args, **kwargs)
        semantic_grade = copy.deepcopy(result["semantic_grade"])
        semantic_grade["verdict"] = "gain_preverified"
        return {
            **result,
            "semantic_grade": semantic_grade,
            "semantic_grade_canonical_sha256": hashlib.sha256(
                canonical_json_bytes(semantic_grade)
            ).hexdigest(),
        }

    monkeypatch.setattr(
        verifier_module,
        "independent_grade_campaign",
        divergent_kernel,
    )

    failures, detail = verifier_module._verify_sequential_look_chain(
        tmp_path,
        plan=plan,
        records=records,
        tasks=tasks,
        trusted_contamination_root_sha256=None,
        trusted_campaign_policy_sha256=None,
    )

    assert detail["verified"] is False
    assert any("look grades differ" in failure for failure in failures)


def test_honest_campaign_passes_independent_verification(tmp_path: Path):
    campaign_dir, plan, tasks, records, manifest = _build_campaign_dir(tmp_path)
    published = _publish_grade(campaign_dir, plan, tasks, records, manifest)
    verdict = verify_campaign_evidence(campaign_dir)
    assert verdict["passed"], verdict["failures"]
    assert verdict["recomputed_verdict"] == published["verdict"]
    assert verdict["committed_records"] == len(plan.cell_ids)
    assert verdict["task_count"] == len(tasks)
    assert "sequential_looks" not in verdict


def test_two_phase_output_and_answer_reveal_pass_independent_verification(
    tmp_path: Path,
):
    campaign_dir, plan, tasks, records, manifest = _build_campaign_dir(
        tmp_path, answer_reveal=True
    )
    _publish_grade(campaign_dir, plan, tasks, records, manifest)

    verdict = verify_campaign_evidence(campaign_dir)

    assert verdict["passed"], verdict["failures"]
    assert "generation_seeds" not in plan.to_dict()["metadata"]["execution_config"]
    assert verdict["generation"]["seed_source"] == "post_seal_answer_reveal"
    assert verdict["answer_reveal"]["verified"] is True
    assert verdict["answer_reveal"]["issuer_attested"] is False


def test_two_phase_verifier_rejects_changed_post_seal_answer(tmp_path: Path):
    campaign_dir, plan, tasks, records, manifest = _build_campaign_dir(
        tmp_path, answer_reveal=True
    )
    _publish_grade(campaign_dir, plan, tasks, records, manifest)
    reveal_path = campaign_dir / campaign_runner_module.ANSWER_REVEAL_FILE
    reveal = json.loads(reveal_path.read_bytes())
    reveal["payload"]["answers"][0]["answer_payload"]["expected"] = {}
    reveal_path.write_bytes(canonical_json_bytes(reveal) + b"\n")

    verdict = verify_campaign_evidence(campaign_dir)

    assert verdict["passed"] is False
    assert any("answer reveal" in reason for reason in verdict["failures"])


def test_independent_verifier_reconstructs_runner_protocol_identity():
    assert (
        verifier_module._campaign_protocol_sha256()
        == campaign_runner_module._campaign_protocol_sha256()
    )


def test_unpublished_grade_still_recomputes_verdict(tmp_path: Path):
    campaign_dir, *_rest = _build_campaign_dir(tmp_path)
    verdict = verify_campaign_evidence(campaign_dir)
    assert verdict["passed"]
    assert verdict["published_verdict"] is None
    assert isinstance(verdict["recomputed_verdict"], str)


def test_dishonest_plan_generation_claim_is_caught(tmp_path: Path):
    """A plan whose DECLARED generation seeds cannot reproduce its embedded
    task manifest is a lie the journal cannot see (it only binds the plan
    hash) — the independent regeneration check must expose it."""
    tasks = _tasks()
    dishonest_plan = build_campaign_plan(
        "independent-verify-test",
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
            "identity_receipt": {
                "composite_identity_sha256": ADAPTER_SHA256,
                "wrapped_projection_count": 64,
            },
        },
        execution_config={
            "max_steps": 8,
            "decode_max_tokens": 256,
            "difficulty": DIFFICULTY,
            "task_registry_version": CURRENT_REGISTRY_VERSION,
            "generation_seeds": [999999],  # the lie: cannot reproduce tasks
            "domains": list(DOMAINS),
            "implementation_sha256": {
                "tools/run_latent_cortex_paired_campaign.py": RUNNER_SHA256,
            },
        },
    )
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    (campaign_dir / PLAN_FILE).write_bytes(
        canonical_json_bytes(dishonest_plan.to_dict()) + b"\n"
    )
    with CampaignJournal(campaign_dir / JOURNAL_FILE, dishonest_plan) as journal:
        for cell_id in dishonest_plan.cell_ids:
            result, verification, commit = _record_material(
                dishonest_plan, tasks, cell_id
            )
            attempt_id = journal.start_cell(cell_id)
            journal.record_arm_result(cell_id, attempt_id, result)
            journal.record_verified(cell_id, attempt_id, verification)
            journal.commit_cell(cell_id, attempt_id, commit)
    verdict = verify_campaign_evidence(campaign_dir)
    assert not verdict["passed"]
    assert any("task manifest mismatch" in f for f in verdict["failures"])


def test_posthoc_plan_tampering_fails_closed_via_journal_binding(
    tmp_path: Path,
):
    import pytest

    from core.brain.llm.latent_cortex.campaign_journal import (
        CampaignJournalError,
    )

    campaign_dir, plan, tasks, records, manifest = _build_campaign_dir(tmp_path)
    _publish_grade(campaign_dir, plan, tasks, records, manifest)
    document = json.loads((campaign_dir / PLAN_FILE).read_bytes())
    document["metadata"]["execution_config"]["generation_seeds"] = [999999]
    (campaign_dir / PLAN_FILE).write_bytes(
        canonical_json_bytes(document) + b"\n"
    )
    with pytest.raises(CampaignJournalError):
        verify_campaign_evidence(campaign_dir)


def test_forged_grade_verdict_is_caught(tmp_path: Path):
    campaign_dir, plan, tasks, records, manifest = _build_campaign_dir(
        tmp_path, gain=False
    )
    published = _publish_grade(campaign_dir, plan, tasks, records, manifest)
    forged = dict(published)
    forged["verdict"] = "gain_proven"
    (campaign_dir / GRADE_FILE).write_bytes(
        canonical_json_bytes(forged) + b"\n"
    )
    verdict = verify_campaign_evidence(campaign_dir)
    assert not verdict["passed"]
    assert any("fully agree" in f for f in verdict["failures"])
    assert any("grade_sha256" in f for f in verdict["failures"])


def test_rehashed_forged_comparison_detail_is_caught(tmp_path: Path):
    """A self-consistent grade hash cannot hide altered evidence fields."""

    campaign_dir, plan, tasks, records, manifest = _build_campaign_dir(tmp_path)
    published = _publish_grade(campaign_dir, plan, tasks, records, manifest)
    forged = json.loads(json.dumps(published))
    forged["comparisons"]["adapter_rlc_gain"]["evidence"]["pooled"][
        "treatment_wins"
    ] += 1
    forged.pop("grade_sha256")
    forged["grade_sha256"] = hashlib.sha256(
        canonical_json_bytes(forged)
    ).hexdigest()
    (campaign_dir / GRADE_FILE).write_bytes(
        canonical_json_bytes(forged) + b"\n"
    )

    verdict = verify_campaign_evidence(campaign_dir)

    assert not verdict["passed"]
    assert any("fully agree" in failure for failure in verdict["failures"])


def test_production_grader_divergence_is_caught_by_independent_kernel(
    tmp_path: Path,
    monkeypatch,
):
    campaign_dir, plan, tasks, records, manifest = _build_campaign_dir(
        tmp_path, gain=False
    )
    _publish_grade(campaign_dir, plan, tasks, records, manifest)
    broken = grade_campaign(records, plan=plan, issuer_tasks=tasks)
    broken["verdict"] = "gain_proven"
    broken["claim_tier"] = "PROVEN"
    monkeypatch.setattr(verifier_module, "grade_campaign", lambda *a, **k: broken)

    verdict = verify_campaign_evidence(campaign_dir)

    assert not verdict["passed"]
    assert any(
        "semantic grade trees differ" in failure
        for failure in verdict["failures"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda grade: grade["statistical_policy"].__setitem__(
            "minimum_domain_observations",
            21,
        ),
        lambda grade: grade["comparisons"]["adapter_rlc_gain"]["evidence"][
            "pooled"
        ].__setitem__("treatment_wins", 999),
        lambda grade: grade["interaction"]["sign_flip"].__setitem__(
            "threshold",
            999,
        ),
        lambda grade: grade["reasons"].append("forged_deep_reason"),
    ],
)
def test_any_deep_production_grade_divergence_fails_full_tree_parity(
    tmp_path: Path,
    monkeypatch,
    mutation,
):
    campaign_dir, plan, tasks, records, manifest = _build_campaign_dir(tmp_path)
    _publish_grade(campaign_dir, plan, tasks, records, manifest)
    broken = copy.deepcopy(
        grade_campaign(records, plan=plan, issuer_tasks=tasks)
    )
    mutation(broken)
    monkeypatch.setattr(
        verifier_module,
        "grade_campaign",
        lambda *args, **kwargs: broken,
    )

    verdict = verify_campaign_evidence(campaign_dir)

    assert verdict["passed"] is False
    parity_failures = [
        failure
        for failure in verdict["failures"]
        if "semantic grade trees differ" in failure
    ]
    assert len(parity_failures) == 1
    assert "$." in parity_failures[0]


def test_independent_semantic_tree_hash_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch,
):
    campaign_dir, *_rest = _build_campaign_dir(tmp_path)
    real_kernel = verifier_module.independent_grade_campaign

    def mismatched_hash(*args, **kwargs):
        result = real_kernel(*args, **kwargs)
        return {
            **result,
            "semantic_grade_canonical_sha256": "0" * 64,
        }

    monkeypatch.setattr(
        verifier_module,
        "independent_grade_campaign",
        mismatched_hash,
    )

    verdict = verify_campaign_evidence(campaign_dir)

    assert verdict["passed"] is False
    assert any(
        "semantic grade hash does not match" in failure
        for failure in verdict["failures"]
    )


def test_canonical_artifact_rejects_duplicate_json_keys(tmp_path: Path):
    artifact = tmp_path / "duplicate.json"
    artifact.write_bytes(b'{"schema":"first","schema":"second"}\n')

    with pytest.raises(ValueError, match="duplicate JSON key"):
        verifier_module._canonical_artifact(artifact, role="test artifact")


def test_independent_kernel_has_no_production_grading_imports():
    source_path = Path(independent_grade_campaign.__code__.co_filename)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden = {
        "core.brain.llm.latent_cortex.paired_campaign",
        "core.brain.llm.latent_cortex.experiments",
        "core.brain.llm.latent_cortex.frontier_tasks",
    }
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not (imported & forbidden)


def test_tampered_journal_fails_closed(tmp_path: Path):
    import pytest

    from core.brain.llm.latent_cortex.campaign_journal import (
        CampaignJournalError,
    )

    campaign_dir, *_rest = _build_campaign_dir(tmp_path)
    journal_path = campaign_dir / JOURNAL_FILE
    lines = journal_path.read_bytes().splitlines(keepends=True)
    tampered = lines[2].replace(b'"layer_apps":10000', b'"layer_apps":10')
    assert tampered != lines[2]
    journal_path.write_bytes(b"".join(lines[:2] + [tampered] + lines[3:]))
    with pytest.raises(CampaignJournalError):
        verify_campaign_evidence(campaign_dir)
