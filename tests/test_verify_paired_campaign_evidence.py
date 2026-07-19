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
import hashlib
import json
from pathlib import Path

from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignJournal,
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


def _plan(tasks):
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
            "generation_seeds": list(SEEDS),
            "domains": list(DOMAINS),
            "implementation_sha256": {
                "tools/run_latent_cortex_paired_campaign.py": RUNNER_SHA256,
            },
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
    }
    task = task_records[definition["task_id"]]
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


def _build_campaign_dir(tmp_path: Path, *, gain: bool = True):
    tasks = _tasks()
    plan = _plan(tasks)
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
    return campaign_dir, plan, tasks, records, manifest


def _publish_grade(campaign_dir, plan, tasks, records, manifest) -> dict:
    grade = grade_campaign(records, plan=plan, issuer_tasks=tasks)
    material = dict(grade)
    material.pop("grade_sha256", None)
    material["campaign_manifest_sha256"] = manifest["manifest_sha256"]
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


def test_honest_campaign_passes_independent_verification(tmp_path: Path):
    campaign_dir, plan, tasks, records, manifest = _build_campaign_dir(tmp_path)
    published = _publish_grade(campaign_dir, plan, tasks, records, manifest)
    verdict = verify_campaign_evidence(campaign_dir)
    assert verdict["passed"], verdict["failures"]
    assert verdict["recomputed_verdict"] == published["verdict"]
    assert verdict["committed_records"] == len(plan.cell_ids)
    assert verdict["task_count"] == len(tasks)


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
        "independent kernel" in failure for failure in verdict["failures"]
    )


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
