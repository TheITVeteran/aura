"""Contracts for the frozen-checkpoint reconciliation sweep.

The sweep decides whether a twelve-hour resident run is worth launching, so
its journal, resumption, and grading must be correct before it consumes any
32B time.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS = REPO_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_rlc_reconciliation_sweep as sweep  # noqa: E402


def test_script_and_package_imports_share_typed_fault_identity() -> None:
    from tools import run_rlc_reconciliation_sweep as packaged

    assert packaged is sweep
    assert packaged.EpisodeFault is sweep.EpisodeFault


def test_cell_status_is_published_before_model_execution(tmp_path: Path) -> None:
    task = SimpleNamespace(task_id="task-1", domain="mathematics")

    sweep._mark_cell_started(
        tmp_path,
        arm="complete_system_recurrent_composed",
        task=task,
        index=2,
        task_count=7,
        planned_cells=49,
        committed_cells=14,
        elapsed_s=90.5,
    )

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "executing_cell"
    assert status["arm"] == "complete_system_recurrent_composed"
    assert status["task_id"] == "task-1"
    assert status["domain"] == "mathematics"
    assert status["arm_progress"] == "3/7"
    assert status["committed_cells"] == 14


def test_complete_system_progress_status_is_live_and_content_blind(tmp_path: Path) -> None:
    callback = sweep._complete_system_progress_callback(
        tmp_path,
        arm="complete_system_closed_book",
        task_id="task-secret",
    )

    callback(
        {
            "schema": "aura.rlc.complete_system_progress.v1",
            "phase": "first_rlc_engine",
            "elapsed_s": 12.5,
            "engine_stage": "recurrent_schedule",
            "spent_layer_apps": 4096,
            "objective": "must never be published",
            "candidate_text": "must never be published",
            "answer_key": {"value": 7},
        }
    )

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "executing_cell"
    assert status["arm"] == "complete_system_closed_book"
    assert status["task_id"] == "task-secret"
    progress = status["complete_system_progress"]
    assert progress == {
        "elapsed_s": 12.5,
        "engine_stage": "recurrent_schedule",
        "phase": "first_rlc_engine",
        "schema": "aura.rlc.complete_system_progress.v1",
        "spent_layer_apps": 4096,
    }


def test_complete_system_phase_tracker_times_work_and_is_observer_safe(
    monkeypatch,
) -> None:
    from tools import rlc_complete_system_closed_book as complete_system

    ticks = iter([10.0, 11.0, 12.0, 15.0, 16.0, 20.0])
    monkeypatch.setattr(complete_system.time, "monotonic", lambda: next(ticks))
    observed = []
    tracker = complete_system._CompleteSystemPhaseTracker(observed.append)
    tracker.begin("amplifier_generation", generation_index=1)
    tracker.end("amplifier_generation", generation_index=1)

    assert tracker.receipt() == {"amplifier_generation": 4.0, "total": 10.0}
    assert [row["phase"] for row in observed] == [
        "amplifier_generation_started",
        "amplifier_generation_completed",
    ]
    assert observed[-1]["phase_duration_s"] == 4.0

    monkeypatch.setattr(complete_system.time, "monotonic", lambda: 21.0)
    failing = complete_system._CompleteSystemPhaseTracker(
        lambda _payload: (_ for _ in ()).throw(RuntimeError("observer failed"))
    )
    failing.begin("promotion")
    failing.end("promotion")


def test_subset_self_test_binds_only_the_selected_domains(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    selected_domains = "mathematics,calibration,misleading_premise"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_rlc_reconciliation_sweep.py",
            "--model",
            "unused-by-self-test",
            "--out-dir",
            str(tmp_path),
            "--campaign-stage",
            "component",
            "--domains",
            selected_domains,
            "--per-domain",
            "1",
            "--arms",
            "complete_system_closed_book",
            "--self-test",
        ],
    )

    assert sweep.main() == 0
    commitment = json.loads((tmp_path / "task_commitment.json").read_text())
    assert commitment["domains"] == selected_domains.split(",")
    output = capsys.readouterr().out
    self_test = json.loads(output[output.index("{") :])
    assert self_test["domains"] == selected_domains.split(",")
    assert self_test["execution_arms"] == [
        "vanilla",
        "complete_system_closed_book",
        "complete_system_adaptation_ablation",
        "complete_system_executable_ablation",
    ]


def _write_evidence_manifest(
    out_dir: Path,
    tasks,
    *,
    required_arms: list[str],
    requested_arms: list[str] | None = None,
    campaign_stage: str = "certificate",
) -> None:
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    implementation = sweep._implementation_manifest()
    task_manifest = ft.build_task_manifest(tasks)
    commitment = ft.build_task_commitment(task_manifest)
    fingerprints = {name: "a" * 64 for name in required_arms}
    arm_tokens = {name: 512 for name in required_arms}
    task_tokens = {
        name: {
            task.task_id: sweep._task_decode_max_tokens(task, arm_tokens[name]) for task in tasks
        }
        for name in required_arms
    }
    (out_dir / "decode_fingerprint.json").write_text(
        json.dumps(
            {
                "schema": sweep.EVIDENCE_MANIFEST_SCHEMA,
                "decode_fingerprint": fingerprints,
                "arm_max_tokens": arm_tokens,
                "completion_budget_policy": sweep.COMPLETION_BUDGET_POLICY,
                "task_max_tokens": task_tokens,
                "requested_arms": requested_arms or list(required_arms),
                "required_arms": required_arms,
                "expected_task_ids": [task.task_id for task in tasks],
                "task_commitment_sha256": commitment.commitment_sha256,
                "task_registry_version": task_manifest.registry_version,
                "campaign_stage": campaign_stage,
                "domains": list(dict.fromkeys(task.domain for task in tasks)),
                "implementation_files": implementation,
                "implementation_sha256": sweep._implementation_sha256(implementation),
            }
        ),
        encoding="utf-8",
    )


def test_the_full_stack_arm_enables_every_pillar_that_was_built():
    """Every run before 2026-08-07 measured recurrence with hidden-state
    optimization, fast weights and adaptive halting switched OFF, and reported
    the result as a property of "the recurrent path". The program's own claim
    is that reasoning is a unified system, so the arm under test has to be the
    union, not one component of it."""
    cfg = sweep._build_config(8, 16, "applied", 512, profile="full")
    assert cfg.latent_opt.enabled is True, "hidden-state optimization"
    assert cfg.fast_weights.enabled is True, "temporary fast weights"
    assert cfg.local_repair_enabled is True, "local repair"
    assert cfg.answer_replacement_enabled is True, "evidence-bound acceptance"
    assert cfg.generative_verifier_enabled is True
    assert cfg.counterfactual_verifier_enabled is True
    assert cfg.prefix_stability_enabled is True
    assert cfg.decode_contract == "final_answer_v1"
    assert cfg.decode_contract_grace_tokens == 0
    assert cfg.verifier_probe_contract == "final_answer_v1"
    assert cfg.local_repair_max_attempts == 2
    assert cfg.verifier_accept_non_regression is True
    # Adaptive halting: the depth is a ceiling, not a floor.
    assert cfg.recurrence.min_steps == 2
    assert cfg.recurrence.max_steps == 8
    assert cfg.recurrence.fixed_depth is False


def test_output_memory_diagnostic_is_explicit_and_source_bound():
    ordinary = sweep._build_config(8, 16, "applied", 512, profile="full")
    diagnostic = sweep._build_config(
        8,
        16,
        "applied",
        512,
        profile="full",
        output_memory_diagnostic=True,
    )
    assert ordinary.fast_weights.output_memory_diagnostic_enabled is False
    assert diagnostic.fast_weights.output_memory_diagnostic_enabled is True

    common = {
        "model": "checkpoint",
        "n_slots": 16,
        "max_tokens": 512,
        "episode_wall_s": 720.0,
        "seed": 7,
        "per_domain": 1,
    }
    assert sweep.decode_fingerprint(**common) != sweep.decode_fingerprint(
        **common,
        output_memory_diagnostic=True,
    )


def test_the_mechanism_arm_stays_an_ablation():
    """The stripped configuration is retained, but only underneath the full
    arm, and it must not silently acquire the pillars."""
    cfg = sweep._build_config(4, 16, "suppressed", 512, profile="mechanism")
    assert cfg.latent_opt.enabled is False
    assert cfg.fast_weights.enabled is False
    assert cfg.local_repair_enabled is False
    assert cfg.answer_replacement_enabled is False
    # Forced depth: no early halting, which is what makes it an ablation.
    assert cfg.recurrence.min_steps == cfg.recurrence.max_steps == 4
    assert cfg.recurrence.fixed_depth is True
    assert cfg.verifier_accept_non_regression is False


def test_frontier_verifiers_enforce_each_tasks_public_response_shape():
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    task = ft.generate_task_battery([20260808], difficulty=2)[0]
    verifier = sweep._episode_verifier(task)
    oracle = sweep._OracleTaskVerifier(task)

    assert verifier.response_contract == task.public.response_contract
    assert oracle.response_contract == task.public.response_contract
    assert (
        verifier.evaluate('FINAL_ANSWER: {"wrong_key": 1}')["checks"]["response_contract"]["valid"]
        is False
    )


def test_the_battery_leads_with_the_unified_system_not_the_ablation():
    by_name = {a.name: a for a in sweep.ARMS}
    assert by_name["vanilla"].profile == "ordinary"
    assert by_name["complete_system_closed_book"].profile == "complete_closed_book"
    assert (
        by_name["complete_system_recurrent_composed"].profile
        == "complete_closed_book_recurrent_composed"
    )
    assert (
        by_name["complete_system_recurrent_initial_control"].profile
        == "complete_closed_book_recurrent_initial_control"
    )
    assert (
        by_name["complete_system_recurrent_depth_lesion"].profile
        == "complete_closed_book_recurrent_depth_lesion"
    )
    assert by_name["full_stack"].profile == "full"
    assert by_name["full_stack_oracle"].profile == "full_oracle"
    assert by_name["rlc_mechanism"].profile == "mechanism"
    # The control and the unified arm must share a decode budget, or the
    # contrast measures the budget instead of the system.
    assert by_name["vanilla"].max_tokens == by_name["full_stack"].max_tokens
    # The oracle arm is a diagnostic ceiling and is never promotable.
    assert "oracle" in by_name["full_stack_oracle"].name


def test_completion_budget_is_task_aware_and_identical_across_arms():
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260809], difficulty=1)
    budgets = {task.domain: sweep._task_decode_max_tokens(task, 320) for task in tasks}

    assert budgets["coding"] == 768
    assert budgets["long_horizon_planning"] == 640
    assert all(
        budget == 512
        for domain, budget in budgets.items()
        if domain not in {"coding", "long_horizon_planning"}
    )
    for task in tasks:
        treatment = sweep._task_decode_max_tokens(task, 320)
        control = sweep._task_decode_max_tokens(task, 320)
        assert treatment == control
        assert sweep._task_decode_max_tokens(task, 1024) == 1024


def test_evidence_manifest_rejects_task_token_policy_drift(tmp_path: Path):
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260809], difficulty=1)
    _write_evidence_manifest(tmp_path, tasks, required_arms=["vanilla"])
    path = tmp_path / "decode_fingerprint.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["task_max_tokens"]["vanilla"][tasks[0].task_id] += 1
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert "task_token_budget_policy_mismatch" in sweep._manifest_integrity_issues(
        manifest,
        tasks,
    )


def test_implementation_identity_binds_extracted_complete_system_modules():
    manifest = sweep._implementation_manifest()

    assert "tools/rlc_complete_system_closed_book.py" in manifest
    assert "tools/rlc_integrated_recurrent_producer.py" in manifest
    assert "tools/rlc_reconciliation_evidence.py" in manifest
    assert "core/brain/reasoning_amplifier_v2.py" in manifest


def test_complete_system_request_expands_to_controls_without_narrower_treatments():
    expanded = sweep._expand_requested_arms({"complete_system_closed_book"})
    assert [arm.name for arm in expanded] == [
        "vanilla",
        "vanilla_equal_compute",
        "complete_system_closed_book",
        "complete_system_adaptation_ablation",
        "complete_system_executable_ablation",
        "vanilla_resource_dominating",
    ]


def test_recurrent_composed_request_keeps_the_uncomposed_engine_as_a_control():
    expanded = sweep._expand_requested_arms(
        {"complete_system_recurrent_composed"},
        campaign_stage="certificate",
    )
    assert [arm.name for arm in expanded] == [
        "vanilla",
        "vanilla_equal_compute",
        "complete_system_closed_book",
        "complete_system_recurrent_composed",
        "complete_system_recurrent_initial_control",
        "complete_system_recurrent_depth_lesion",
        "complete_system_adaptation_ablation",
        "complete_system_executable_ablation",
        "vanilla_resource_dominating",
    ]


def test_recurrent_package_identity_is_bound_into_the_decode_fingerprint():
    common = {
        "model": "checkpoint",
        "n_slots": 16,
        "max_tokens": 512,
        "episode_wall_s": 720.0,
        "seed": 7,
        "per_domain": 1,
    }
    composed_a = sweep.decode_fingerprint(
        **common,
        arm="complete_system_recurrent_composed",
        integrated_recurrent_package_sha256="a" * 64,
    )
    composed_b = sweep.decode_fingerprint(
        **common,
        arm="complete_system_recurrent_composed",
        integrated_recurrent_package_sha256="b" * 64,
    )
    assert composed_a != composed_b


def test_composed_profiles_invoke_their_causally_matched_recurrent_producer(
    monkeypatch: pytest.MonkeyPatch,
):
    from tools import rlc_integrated_recurrent_producer as producer

    calls = []

    def fake_produce(**kwargs):
        calls.append(kwargs)
        return {
            "source": kwargs["source"],
            "recurrence_depth": kwargs["recurrence_depth"],
        }, {"receipt": True}

    monkeypatch.setattr(producer, "produce_integrated_recurrent_candidate", fake_produce)
    trained_controller = object()
    common = {
        "loaded": SimpleNamespace(controller=trained_controller),
        "model": object(),
        "tokenizer": object(),
        "task": object(),
        "public_tokens": [1, 2, 3],
        "max_tokens": 2048,
        "initial_controller": object(),
    }
    assert sweep._integrated_candidates_for_profile(
        profile="complete_closed_book",
        **common,
    ) == []
    assert calls == []
    assert sweep._integrated_candidates_for_profile(
        profile="complete_closed_book_recurrent_composed",
        **common,
    ) == [
        {"source": producer.TRAINED_DEPTH_ONE_SOURCE, "recurrence_depth": 1},
        {"source": producer.SOURCE_NAME, "recurrence_depth": None},
    ]
    assert len(calls) == 2
    assert calls[0]["public_tokens"] == [1, 2, 3]
    assert calls[0]["max_tokens"] == 2048
    assert calls[0]["source"] == producer.TRAINED_DEPTH_ONE_SOURCE
    assert calls[0]["controller"] is trained_controller
    assert calls[0]["recurrence_depth"] == 1
    assert calls[1]["source"] == producer.SOURCE_NAME
    assert calls[1]["controller"] is trained_controller
    assert calls[1]["recurrence_depth"] is None
    assert sweep._integrated_candidates_for_profile(
        profile="complete_closed_book_recurrent_initial_control",
        **common,
    ) == [
        {
            "source": producer.INITIAL_CONTROL_DEPTH_ONE_SOURCE,
            "recurrence_depth": 1,
        },
        {"source": producer.INITIAL_CONTROL_SOURCE, "recurrence_depth": None},
    ]
    assert calls[-2]["source"] == producer.INITIAL_CONTROL_DEPTH_ONE_SOURCE
    assert calls[-2]["controller"] is common["initial_controller"]
    assert calls[-2]["recurrence_depth"] == 1
    assert calls[-1]["source"] == producer.INITIAL_CONTROL_SOURCE
    assert calls[-1]["controller"] is common["initial_controller"]
    assert calls[-1]["recurrence_depth"] is None
    assert sweep._integrated_candidates_for_profile(
        profile="complete_closed_book_recurrent_depth_lesion",
        **common,
    ) == [{"source": producer.DEPTH_LESION_SOURCE, "recurrence_depth": 1}]
    assert calls[-1]["source"] == producer.DEPTH_LESION_SOURCE
    assert calls[-1]["controller"] is trained_controller
    assert calls[-1]["recurrence_depth"] == 1


def test_composed_adjudication_requires_learned_parameters_and_recurrent_depth():
    task_ids = {f"task-{index}" for index in range(8)}

    def rows(correct: bool):
        return {task_id: {"correct": correct} for task_id in task_ids}

    scored = {
        "complete_system_recurrent_composed": rows(True),
        "complete_system_recurrent_initial_control": rows(False),
        "complete_system_recurrent_depth_lesion": rows(False),
        "complete_system_closed_book": rows(False),
        "vanilla": rows(False),
    }
    result = sweep._adjudicate_composed_recurrent_tissue(
        scored=scored,
        expected_task_ids=task_ids,
        complete=True,
        resource_target_control_proven=True,
    )

    assert result["bounded_learned_tissue_positive"] is True
    assert result["recurrent_depth_positive"] is True
    assert result["powered_causal_result"] is True
    assert result["wow_signal_authorized"] is False
    assert result["fusion_authorized"] is False


def test_composed_campaign_decision_cannot_be_replaced_by_legacy_engine():
    decision, target = sweep._primary_campaign_decision(
        generic_decision="proceed_to_checkpoint_phase",
        selected_arm_names={"complete_system_recurrent_composed", "vanilla"},
        composed_adjudication={"decision": "no_bounded_learned_tissue_gain"},
    )
    assert decision == "no_bounded_learned_tissue_gain"
    assert target == "composed_recurrent_tissue"


def test_legacy_campaign_keeps_its_own_decision():
    decision, target = sweep._primary_campaign_decision(
        generic_decision="proceed_to_checkpoint_phase",
        selected_arm_names={"complete_system_closed_book", "vanilla"},
        composed_adjudication={"decision": "no_bounded_learned_tissue_gain"},
    )
    assert decision == "proceed_to_checkpoint_phase"
    assert target == "legacy_reconciliation_engine"


def test_composed_adjudication_does_not_mislabel_architecture_only_gain_as_learning():
    task_ids = {"task-a", "task-b"}
    correct = {task_id: {"correct": True} for task_id in task_ids}
    wrong = {task_id: {"correct": False} for task_id in task_ids}
    result = sweep._adjudicate_composed_recurrent_tissue(
        scored={
            "complete_system_recurrent_composed": correct,
            "complete_system_recurrent_initial_control": correct,
            "complete_system_recurrent_depth_lesion": wrong,
            "complete_system_closed_book": wrong,
            "vanilla": wrong,
        },
        expected_task_ids=task_ids,
        complete=True,
        resource_target_control_proven=True,
    )

    assert result["bounded_learned_tissue_positive"] is False
    assert result["decision"] == "no_bounded_learned_tissue_gain"


def test_composed_schedule_finishes_each_causal_task_block_before_the_next():
    selected = sweep._expand_requested_arms(
        {"complete_system_recurrent_composed"},
        campaign_stage="component",
    )
    tasks = (SimpleNamespace(task_id="one"), SimpleNamespace(task_id="two"))
    schedule = sweep._execution_schedule(selected, tasks)

    first_task_arms = [arm.name for arm, _index, task in schedule if task.task_id == "one"]
    assert first_task_arms == [arm.name for arm in selected]
    assert [task.task_id for _arm, _index, task in schedule[: len(selected)]] == [
        "one"
    ] * len(selected)
    assert schedule[len(selected)][2].task_id == "two"


def test_noncomposed_schedule_preserves_arm_major_resume_order():
    selected = sweep._expand_requested_arms(
        {"complete_system_closed_book"},
        campaign_stage="component",
    )
    tasks = (SimpleNamespace(task_id="one"), SimpleNamespace(task_id="two"))
    schedule = sweep._execution_schedule(selected, tasks)

    assert [(arm.name, task.task_id) for arm, _index, task in schedule[:2]] == [
        (selected[0].name, "one"),
        (selected[0].name, "two"),
    ]


def test_composed_futility_stops_only_on_a_fully_paired_regression(
    monkeypatch: pytest.MonkeyPatch,
):
    task = SimpleNamespace(task_id="task-a", domain="mathematics")
    correctness = {
        "trained": False,
        "initial": True,
        "uncomposed": False,
        "lesion": False,
        "vanilla": True,
    }
    monkeypatch.setattr(
        "core.brain.llm.latent_cortex.frontier_tasks.score_task",
        lambda _task, text: SimpleNamespace(correct=correctness[text]),
    )
    cells = [
        {
            "task_id": task.task_id,
            "arm": "complete_system_recurrent_composed",
            "text": "trained",
            "error": "",
        },
        {
            "task_id": task.task_id,
            "arm": "complete_system_recurrent_initial_control",
            "text": "initial",
            "error": "",
        },
        {
            "task_id": task.task_id,
            "arm": "complete_system_closed_book",
            "text": "uncomposed",
            "error": "",
        },
        {
            "task_id": task.task_id,
            "arm": "complete_system_recurrent_depth_lesion",
            "text": "lesion",
            "error": "",
        },
        {
            "task_id": task.task_id,
            "arm": "vanilla",
            "text": "vanilla",
            "error": "",
        },
    ]

    assert sweep._composed_task_terminal_futility(task=task, cells=cells[:-1]) is None
    receipt = sweep._composed_task_terminal_futility(task=task, cells=cells)
    assert receipt is not None
    assert receipt["fatal_regression_contrasts"] == [
        "vanilla_floor",
        "learned_parameters",
    ]
    assert receipt["positive_claim_authority"] is False
    assert receipt["receipt_sha256"] == sweep._canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def test_composed_futility_does_not_stop_on_ties_or_positive_lifts(
    monkeypatch: pytest.MonkeyPatch,
):
    task = SimpleNamespace(task_id="task-a", domain="mathematics")
    monkeypatch.setattr(
        "core.brain.llm.latent_cortex.frontier_tasks.score_task",
        lambda _task, text: SimpleNamespace(correct=text == "trained"),
    )
    cells = [
        {"task_id": task.task_id, "arm": arm, "text": text, "error": ""}
        for arm, text in (
            ("complete_system_recurrent_composed", "trained"),
            ("complete_system_recurrent_initial_control", "initial"),
            ("complete_system_closed_book", "uncomposed"),
            ("complete_system_recurrent_depth_lesion", "lesion"),
            ("vanilla", "vanilla"),
        )
    ]

    assert sweep._composed_task_terminal_futility(task=task, cells=cells) is None


def test_staged_campaigns_do_not_pay_for_the_full_certificate_early():
    component = {
        arm.name
        for arm in sweep._expand_requested_arms(
            {"complete_system_closed_book"}, campaign_stage="component"
        )
    }
    pilot = {
        arm.name
        for arm in sweep._expand_requested_arms(
            {"complete_system_closed_book"}, campaign_stage="pilot"
        )
    }
    certificate = {
        arm.name
        for arm in sweep._expand_requested_arms(
            {"complete_system_closed_book"}, campaign_stage="certificate"
        )
    }

    assert component == {
        "vanilla",
        "complete_system_closed_book",
        "complete_system_adaptation_ablation",
        "complete_system_executable_ablation",
    }
    assert pilot == {
        "vanilla",
        "vanilla_equal_compute",
        "complete_system_closed_book",
        "complete_system_adaptation_ablation",
        "complete_system_executable_ablation",
    }
    assert certificate == {
        "vanilla",
        "vanilla_equal_compute",
        "complete_system_closed_book",
        "complete_system_adaptation_ablation",
        "complete_system_executable_ablation",
        "vanilla_resource_dominating",
    }


def test_stage_admission_separates_architecture_gain_from_neural_attribution():
    common = {
        "campaign_stage": "component",
        "complete": True,
        "floor_holds": True,
        "treatment_lifts": 5,
        "treatment_regressions": 0,
        "adaptation_lifts": 5,
        "adaptation_regressions": 0,
        "producer_lifts": 5,
        "producer_regressions": 0,
        "gain_domains": 3,
        "exact_promotions": 5,
        "completion_limited_cells": 0,
        "treatment_correct": 2,
        "preliminary_control_correct": None,
    }
    inert = sweep._next_stage_admission(
        **common,
        latent_accepted_steps=0,
        fast_weight_applications=0,
    )
    moving = sweep._next_stage_admission(
        **common,
        latent_accepted_steps=1,
        fast_weight_applications=0,
    )

    assert inert["admitted"] is True
    assert inert["reasons"] == []
    assert inert["architecture_admitted"] is True
    assert inert["adaptive_neural_admitted"] is False
    assert inert["adaptive_neural_reasons"] == ["neural_tissue_zero_yield"]
    assert moving["admitted"] is True
    assert moving["adaptive_neural_admitted"] is True
    assert moving["next_stage"] == "pilot"
    assert moving["paired_sign_test_p"] == 0.03125


def test_pilot_must_beat_preliminary_control_before_certificate():
    verdict = sweep._next_stage_admission(
        campaign_stage="pilot",
        complete=True,
        floor_holds=True,
        treatment_lifts=5,
        treatment_regressions=0,
        adaptation_lifts=5,
        adaptation_regressions=0,
        producer_lifts=5,
        producer_regressions=0,
        gain_domains=3,
        exact_promotions=5,
        latent_accepted_steps=1,
        fast_weight_applications=0,
        completion_limited_cells=0,
        treatment_correct=2,
        preliminary_control_correct=2,
    )

    assert verdict["admitted"] is False
    assert verdict["reasons"] == ["pilot_did_not_beat_preliminary_control"]


def test_terminal_certificate_admits_evidence_without_inventing_another_stage():
    verdict = sweep._next_stage_admission(
        campaign_stage="certificate",
        complete=True,
        floor_holds=True,
        treatment_lifts=5,
        treatment_regressions=0,
        adaptation_lifts=0,
        adaptation_regressions=0,
        producer_lifts=5,
        producer_regressions=0,
        gain_domains=3,
        exact_promotions=5,
        latent_accepted_steps=0,
        fast_weight_applications=0,
        completion_limited_cells=0,
        treatment_correct=5,
        preliminary_control_correct=0,
    )

    assert verdict["admitted"] is True
    assert verdict["architecture_admitted"] is True
    assert verdict["advance_authorized"] is False
    assert verdict["next_stage"] == "none"
    assert verdict["reasons"] == []
    assert verdict["adaptive_neural_admitted"] is False


def test_complete_system_config_runs_the_same_neural_pillars_as_full_stack():
    config = sweep._build_config(
        8,
        16,
        "suppressed",
        512,
        profile="complete_closed_book",
    )
    assert config.objective_program_enabled is True
    assert config.latent_opt.enabled is True
    assert config.fast_weights.enabled is True
    assert config.local_repair_enabled is True
    assert config.answer_replacement_enabled is True
    assert config.recurrence.fixed_depth is False


def test_composed_causal_controls_share_the_exact_complete_engine_config():
    treatment = sweep._build_config(
        8,
        16,
        "suppressed",
        512,
        profile="complete_closed_book_recurrent_composed",
    )
    initialization = sweep._build_config(
        8,
        16,
        "suppressed",
        512,
        profile="complete_closed_book_recurrent_initial_control",
    )
    depth_lesion = sweep._build_config(
        8,
        16,
        "suppressed",
        512,
        profile="complete_closed_book_recurrent_depth_lesion",
    )

    assert initialization == treatment
    assert depth_lesion == treatment
    assert treatment.recurrence.fixed_depth is False
    assert treatment.latent_opt.enabled is True
    assert treatment.fast_weights.enabled is True
    assert treatment.objective_program_enabled is True
    assert treatment.decode_incumbent_policy == "vanilla_incumbent"
    assert treatment.decode_repetition_penalty == 1.0


def test_adaptation_ablation_preserves_the_outer_system_but_removes_neural_updates():
    config = sweep._build_config(
        8,
        16,
        "suppressed",
        512,
        profile="complete_closed_book_adaptation_ablation",
    )

    assert config.recurrence.fixed_depth is False
    assert config.local_repair_enabled is True
    assert config.answer_replacement_enabled is True
    assert config.objective_program_enabled is True
    assert config.latent_opt.enabled is False
    assert config.fast_weights.enabled is False
    assert config.verifier_accept_non_regression is False


def test_executable_ablation_preserves_adaptive_neural_tissue():
    config = sweep._build_config(
        8,
        16,
        "suppressed",
        512,
        profile="complete_closed_book_executable_ablation",
    )

    assert config.recurrence.fixed_depth is False
    assert config.local_repair_enabled is True
    assert config.answer_replacement_enabled is True
    assert config.objective_program_enabled is False
    assert config.verified_objective_teacher_enabled is True
    assert config.latent_opt.enabled is True
    assert config.fast_weights.enabled is True
    assert config.verifier_accept_non_regression is True


def test_requesting_a_treatment_always_expands_to_both_controls():
    expanded = sweep._expand_requested_arms({"full_stack"})
    assert [arm.name for arm in expanded] == [
        "vanilla",
        "vanilla_equal_compute",
        "full_stack",
    ]

    controls_only = sweep._expand_requested_arms({"vanilla"})
    assert [arm.name for arm in controls_only] == ["vanilla"]

    with pytest.raises(ValueError, match="unknown arms"):
        sweep._expand_requested_arms({"not_a_real_arm"})


def test_equal_compute_random_streams_are_task_bound_and_reproducible():
    first = sweep._equal_compute_seed(20260809, "task-a", 0)

    assert first == sweep._equal_compute_seed(20260809, "task-a", 0)
    assert first != sweep._equal_compute_seed(20260809, "task-b", 0)
    assert first != sweep._equal_compute_seed(20260809, "task-a", 1)
    assert first != sweep._equal_compute_seed(20260810, "task-a", 0)


def test_resource_dominating_control_spends_until_every_target_dimension_is_met(
    monkeypatch,
    tmp_path: Path,
):
    from core.brain.llm.latent_cortex.resource_accounting import (
        ModelComputeProfile,
        ResourceLedger,
        build_information_receipt,
        policy_sha256,
    )
    from core.brain.llm.latent_cortex.value_of_computation import (
        build_evidence_snapshot,
    )

    profile = ModelComputeProfile(
        model_type="fixture",
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        head_dim=4,
    )
    target = ResourceLedger(profile)
    target.charge(
        "target",
        transformer_layer_apps=40,
        attention_query_key_pairs=40,
        output_head_tokens=4,
        tensor_element_reads=40,
        tensor_element_writes=40,
        verifier_calls=4,
        verifier_input_bytes=2,
        verifier_output_bytes=2,
        tool_calls=2,
        tool_input_bytes=8,
        tool_result_bytes=2,
    )
    encoded_tokens = b"[1]"
    evidence_payload = json.dumps(
        build_evidence_snapshot(bucket="fixture|none|short|s:mid|u:mid", cells={}),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    tokenizer = SimpleNamespace(chat_template="")
    tokenizer_type = type(tokenizer)

    information = build_information_receipt(
        sources=[
            {
                "source_id": "rendered_model_input",
                "kind": "model_input_tokens",
                "content_sha256": hashlib.sha256(encoded_tokens).hexdigest(),
                "byte_count": len(encoded_tokens),
                "token_count": 1,
            },
            {
                "source_id": "value_controller_evidence",
                "kind": "controller_evidence",
                "content_sha256": hashlib.sha256(evidence_payload).hexdigest(),
                "byte_count": len(evidence_payload),
                "token_count": 0,
            },
        ],
        policies={
            "tokenizer": policy_sha256(
                {
                    "module": tokenizer_type.__module__,
                    "qualname": tokenizer_type.__qualname__,
                    "chat_template_sha256": hashlib.sha256(b"").hexdigest(),
                }
            ),
            "verifier": "0" * 64,
            "tools": policy_sha256({"policy": "no_external_tools_inside_rlc_v1"}),
            "nonparametric_memory": policy_sha256(
                {
                    "policy": "context_only_prompt_tail_recall_v1",
                    "active_source_receipt_sha256": "none",
                }
            ),
        },
    )
    calls = 0

    def fake_vanilla(*args, **kwargs):
        nonlocal calls
        calls += 1
        ledger = ResourceLedger(profile)
        ledger.charge(
            "sample",
            transformer_layer_apps=10,
            attention_query_key_pairs=10,
            output_head_tokens=1,
            tensor_element_reads=10,
            tensor_element_writes=10,
        )
        value = calls
        return (
            f'FINAL_ANSWER: {{"value":{value}}}',
            [value],
            "contract_complete",
            ledger.to_receipt(),
        )

    class FakeVerifier:
        def __init__(self):
            self.score = 0.0

        def __call__(self, text):
            self.score = 1.0 if '"value":2' in text else 0.0
            return self.score

        def to_receipt(self):
            return {"score": self.score, "source": "fixture"}

    verifier_path = inspect.getsourcefile(FakeVerifier)
    verifier_policy = policy_sha256(
        {
            "module": FakeVerifier.__module__,
            "qualname": FakeVerifier.__qualname__,
            "source_sha256": hashlib.sha256(Path(verifier_path).read_bytes()).hexdigest(),
        }
    )
    information = build_information_receipt(
        sources=information["sources"],
        policies={**information["policies"], "verifier": verifier_policy},
    )

    monkeypatch.setattr(sweep, "_run_vanilla", fake_vanilla)
    monkeypatch.setattr(sweep, "_episode_verifier", lambda task: FakeVerifier())
    from tools import rlc_complete_system_closed_book as complete_system

    def fake_equal_tool_control(*args, cycle_index, task, **kwargs):
        ledger = ResourceLedger(profile)
        ledger.charge(
            "equal_tool",
            transformer_layer_apps=10,
            attention_query_key_pairs=10,
            output_head_tokens=1,
            tensor_element_reads=10,
            tensor_element_writes=10,
            tool_calls=1,
            tool_input_bytes=4,
            tool_result_bytes=1,
        )
        resource = ledger.to_receipt()
        text = f'FINAL_ANSWER: {{"value":{10 + cycle_index}}}'
        receipt = {
            "schema": "aura.rlc.equal_tool_ordinary_control.v1",
            "task_id": task.task_id,
            "cycle_index": cycle_index,
            "answer_key_used": False,
            "latent_recurrence_used": False,
            "generated_tokens": 1,
            "final_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "executable_operations": [
                {
                    "status": "executed",
                    "program_bytes": 4,
                    "result_bytes": 1,
                    "sandbox_ok": True,
                    "refused": False,
                    "timed_out": False,
                    "process_launched": True,
                    "network_denied": True,
                }
            ],
            "resource_accounting_sha256": resource["receipt_sha256"],
        }
        return text, resource, 1, receipt

    monkeypatch.setattr(
        complete_system,
        "_run_equal_tool_ordinary_control",
        fake_equal_tool_control,
    )
    task = SimpleNamespace(
        task_id="task-a",
        domain="fixture",
        public=SimpleNamespace(prompt="fixture", response_contract='{"value":int}'),
    )
    text, resource, observed_information, certificate, samples, generated, receipt = (
        sweep._run_vanilla_resource_dominating(
            SimpleNamespace(),
            tokenizer,
            [1],
            32,
            task=task,
            target_resource=target.to_receipt(),
            target_information=information,
            treatment_acquisition=None,
            campaign_seed=7,
            incumbent_text='FINAL_ANSWER: {"value":0}',
            max_samples=6,
        )
    )

    assert calls == 2
    assert samples == generated == 4
    assert receipt["equal_tool_cycle_count"] == 2
    assert text == 'FINAL_ANSWER: {"value":2}'
    assert resource["estimated_flops"] >= target.to_receipt()["estimated_flops"]
    assert observed_information == information
    assert certificate["admitted"] is True
    assert receipt["selected_index"] == 3
    assert receipt["sample_count"] == 4
    assert receipt["equal_tool_cycle_limit"] == 6
    mismatched_sources = [dict(source) for source in information["sources"]]
    prompt_source = next(
        source for source in mismatched_sources if source["source_id"] == "rendered_model_input"
    )
    prompt_source["content_sha256"] = hashlib.sha256(b"[2]").hexdigest()
    mismatched_information = build_information_receipt(
        sources=mismatched_sources,
        policies=information["policies"],
    )
    with pytest.raises(RuntimeError, match="prompt differs"):
        sweep._run_vanilla_resource_dominating(
            SimpleNamespace(),
            tokenizer,
            [1],
            32,
            task=task,
            target_resource=target.to_receipt(),
            target_information=mismatched_information,
            treatment_acquisition=None,
            campaign_seed=7,
            max_samples=4,
        )
    path, digest = sweep._persist_runtime_receipt(
        tmp_path,
        arm=sweep.RESOURCE_DOMINATING_CONTROL_ARM,
        task_id="task-a",
        receipt=receipt,
    )
    cell = {
        "runtime_receipt_path": path,
        "runtime_receipt_sha256": digest,
        "resource_accounting": resource,
        "information_accounting": observed_information,
        "resource_dominance_certificate": certificate,
        "text": text,
    }
    assert (
        sweep._resource_dominating_control_receipt_issues(
            tmp_path,
            cell,
            task=task,
        )
        == []
    )
    persisted = tmp_path / path
    tampered = json.loads(persisted.read_text())
    tampered["candidates"][0]["verifier_score"] = 0.5
    tampered["candidates"][0]["verifier_receipt"] = {
        "score": 0.5,
        "source": "fixture",
    }
    body = {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    tampered["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    persisted.write_text(json.dumps(tampered, indent=1, sort_keys=True) + "\n")
    cell["runtime_receipt_sha256"] = hashlib.sha256(
        json.dumps(
            tampered,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert "resource_control_verifier_evidence_mismatch" in (
        sweep._resource_dominating_control_receipt_issues(
            tmp_path,
            cell,
            task=task,
        )
    )

    persisted = tmp_path / path
    tampered = json.loads(persisted.read_text())
    tampered["candidates"][1]["text"] = 'FINAL_ANSWER: {"value":999}'
    persisted.write_text(json.dumps(tampered), encoding="utf-8")
    issues = sweep._resource_dominating_control_receipt_issues(tmp_path, cell)
    assert "resource_control_runtime_receipt_digest_mismatch" in issues
    assert "resource_control_candidate_evidence_invalid" in issues

    persisted.write_text('{"invalid_number":NaN}', encoding="utf-8")
    assert sweep._resource_dominating_control_receipt_issues(tmp_path, cell) == [
        "resource_control_runtime_receipt_noncanonical"
    ]


def test_equal_tool_cycle_plan_is_target_derived_and_rejects_impossible_bytes():
    setup = {"totals": {"tool_calls": 0, "tool_input_bytes": 0, "tool_result_bytes": 0}}
    target = {"totals": {"tool_calls": 3, "tool_input_bytes": 9, "tool_result_bytes": 4}}

    assert sweep._equal_tool_cycle_limit(setup, target, max_samples=12) == 12
    assert sweep._equal_tool_cycle_limit(setup, target, max_samples=4) == 4

    calls_met_bytes_short = {
        "totals": {"tool_calls": 3, "tool_input_bytes": 9, "tool_result_bytes": 1}
    }
    assert sweep._equal_tool_cycle_limit(
        calls_met_bytes_short,
        target,
        max_samples=7,
    ) == 7

    target_met = {
        "totals": {"tool_calls": 3, "tool_input_bytes": 9, "tool_result_bytes": 4}
    }
    assert sweep._equal_tool_cycle_limit(target_met, target, max_samples=7) == 0

    impossible = {"totals": {"tool_calls": 0, "tool_input_bytes": 1, "tool_result_bytes": 0}}
    with pytest.raises(RuntimeError, match="without a measured tool call"):
        sweep._equal_tool_cycle_limit(setup, impossible, max_samples=4)


def test_resource_control_refuses_treatment_derived_symbolic_context(monkeypatch, tmp_path: Path):
    from core.brain import cortex_compute_acquisition
    from core.brain.llm.latent_cortex.resource_accounting import (
        ModelComputeProfile,
        ResourceLedger,
        build_information_receipt,
        policy_sha256,
    )
    from core.brain.llm.latent_cortex.value_of_computation import (
        build_evidence_snapshot,
    )
    from tools.rlc_complete_system_closed_book import (
        _contextual_continuation_objective,
    )

    objective = "Return a checked result."
    text = "Verifier result: ok=true."
    context = {
        "source": "capability.symbolic_formalize",
        "text": text,
        "context_role": "evidence_observation",
        "instruction_authority": False,
        "evidence_id": "evidence-" + "a" * 24,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "retrieval_receipt_sha256": "b" * 64,
        "evidence_kind": "governed_tool_observation",
        "evidence_origin": "core.brain.cortex_compute_acquisition",
        "source_version": "aura.rlc.compute_acquisition.v1",
    }
    continuation_objective = _contextual_continuation_objective(objective, [context])

    class Tokenizer:
        chat_template = "fixture"

        def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
            assert add_generation_prompt is True
            content = messages[0]["content"]
            assert content in {objective, continuation_objective}
            if tokenize:
                return [11, 12, 13] if content == continuation_objective else [7, 8]
            return "rendered"

    class FakeVerifier:
        def __init__(self):
            self.score = 1.0

        def __call__(self, candidate):
            return self.score

        def to_receipt(self):
            return {"score": self.score, "source": "fixture"}

    tokenizer = Tokenizer()
    verifier_path = inspect.getsourcefile(FakeVerifier)
    evidence_payload = json.dumps(
        build_evidence_snapshot(bucket="fixture|none|short|s:mid|u:mid", cells={}),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    encoded_tokens = b"[11,12,13]"
    information = build_information_receipt(
        sources=[
            {
                "source_id": "rendered_model_input",
                "kind": "model_input_tokens",
                "content_sha256": hashlib.sha256(encoded_tokens).hexdigest(),
                "byte_count": len(encoded_tokens),
                "token_count": 3,
            },
            {
                "source_id": "value_controller_evidence",
                "kind": "controller_evidence",
                "content_sha256": hashlib.sha256(evidence_payload).hexdigest(),
                "byte_count": len(evidence_payload),
                "token_count": 0,
            },
        ],
        policies={
            "tokenizer": policy_sha256(
                {
                    "module": Tokenizer.__module__,
                    "qualname": Tokenizer.__qualname__,
                    "chat_template_sha256": hashlib.sha256(b"fixture").hexdigest(),
                }
            ),
            "verifier": policy_sha256(
                {
                    "module": FakeVerifier.__module__,
                    "qualname": FakeVerifier.__qualname__,
                    "source_sha256": hashlib.sha256(Path(verifier_path).read_bytes()).hexdigest(),
                }
            ),
            "tools": policy_sha256({"policy": "no_external_tools_inside_rlc_v1"}),
            "nonparametric_memory": policy_sha256(
                {
                    "policy": "context_only_prompt_tail_recall_v1",
                    "active_source_receipt_sha256": "none",
                }
            ),
        },
    )
    profile = ModelComputeProfile(
        model_type="fixture",
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        head_dim=4,
    )
    target = ResourceLedger(profile)
    target.charge(
        "treatment",
        transformer_layer_apps=1,
        tool_calls=1,
        tool_input_bytes=1,
        tool_result_bytes=1,
    )
    compute_body = {
        "schema": "aura.rlc.compute_acquisition.v1",
        "action": "formalize",
    }
    compute_receipt = {
        **compute_body,
        "receipt_sha256": hashlib.sha256(
            json.dumps(compute_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }

    async def fake_acquire(**kwargs):
        assert kwargs["objective"] == objective
        assert kwargs["first_text"] == "candidate"
        return SimpleNamespace(context=(context,), receipt=compute_receipt)

    def fake_vanilla(*args, **kwargs):
        ledger = ResourceLedger(profile)
        ledger.charge("sample", transformer_layer_apps=2)
        return 'FINAL_ANSWER: {"value":1}', [1], "contract_complete", ledger.to_receipt()

    monkeypatch.setattr(cortex_compute_acquisition, "acquire_cognitive_compute", fake_acquire)
    monkeypatch.setattr(sweep, "_episode_verifier", lambda task: FakeVerifier())
    monkeypatch.setattr(sweep, "_run_vanilla", fake_vanilla)
    task = SimpleNamespace(
        task_id="task-context",
        domain="fixture",
        public=SimpleNamespace(prompt=objective),
    )
    acquisition = {
        "status": "completed_new_context",
        "request": {"action": "formalize"},
        "compute": compute_receipt,
        "timeout_s": 12.0,
        "input_candidate": "candidate",
        "input_candidate_sha256": hashlib.sha256(b"candidate").hexdigest(),
        "acquired_context": [context],
        "continuation_objective": continuation_objective,
        "continuation_objective_sha256": hashlib.sha256(
            continuation_objective.encode()
        ).hexdigest(),
    }
    with pytest.raises(RuntimeError, match="treatment-derived acquisition"):
        sweep._run_vanilla_resource_dominating(
            SimpleNamespace(),
            tokenizer,
            [11, 12, 13],
            32,
            task=task,
            target_resource=target.to_receipt(),
            target_information=information,
            treatment_acquisition=acquisition,
            campaign_seed=9,
            max_samples=2,
        )


def test_decode_identity_binds_committed_task_difficulty():
    common = {
        "model": "/models/Qwen-1.5B",
        "n_slots": 8,
        "max_tokens": 224,
        "episode_wall_s": 120.0,
        "seed": 20260808,
        "per_domain": 2,
        "arm": "vanilla",
        "implementation_sha256": "a" * 64,
    }

    easy = sweep.decode_fingerprint(difficulty=1, **common)
    standard = sweep.decode_fingerprint(difficulty=2, **common)

    assert easy != standard
    assert standard == sweep.decode_fingerprint(difficulty=2, **common)


def test_decode_identity_binds_fast_weight_site():
    common = {
        "model": "/models/Qwen-1.5B",
        "n_slots": 8,
        "max_tokens": 224,
        "episode_wall_s": 120.0,
        "seed": 20260808,
        "per_domain": 2,
        "arm": "complete_system_closed_book",
        "implementation_sha256": "a" * 64,
    }

    early_attention = sweep.decode_fingerprint(
        fast_weight_target="o_proj",
        fast_weight_layer_placement="early",
        **common,
    )
    late_attention = sweep.decode_fingerprint(
        fast_weight_target="o_proj",
        fast_weight_layer_placement="late",
        **common,
    )
    late_mlp = sweep.decode_fingerprint(
        fast_weight_target="down_proj",
        fast_weight_layer_placement="late",
        **common,
    )

    assert len({early_attention, late_attention, late_mlp}) == 3


def test_decode_identity_binds_the_task_registry():
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    assert sweep.CLAIM_TASK_REGISTRY_VERSION == ft.CONTAMINATION_SAFE_REGISTRY_VERSION
    common = {
        "model": "/models/Qwen-32B",
        "n_slots": 16,
        "max_tokens": 320,
        "episode_wall_s": 720.0,
        "seed": 20260808,
        "per_domain": 1,
        "difficulty": 2,
        "arm": "vanilla",
        "implementation_sha256": "a" * 64,
    }
    safe = sweep.decode_fingerprint(
        task_registry_version=sweep.CLAIM_TASK_REGISTRY_VERSION,
        **common,
    )
    legacy = sweep.decode_fingerprint(
        task_registry_version="2026.07.18.1",
        **common,
    )
    assert safe != legacy


def test_evidence_manifest_rejects_task_registry_drift(tmp_path: Path):
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery(
        [20260831],
        difficulty=2,
        registry_version=ft.CONTAMINATION_SAFE_REGISTRY_VERSION,
    )
    _write_evidence_manifest(
        tmp_path,
        tasks,
        required_arms=["vanilla"],
    )
    manifest_path = tmp_path / "decode_fingerprint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task_registry_version"] = ft.REGISTRY_VERSION
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert "task_registry_version_mismatch" in sweep._manifest_integrity_issues(
        manifest,
        tasks,
    )


def test_contract_neutral_diagnostic_repairs_shape_without_inventing_values():
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    task = ft.generate_task_battery(
        [20260831],
        domains=("misleading_premise",),
        difficulty=1,
    )[0]
    expected = task.reveal_for_verifier()["expected"]
    fenced = "```json\n" + json.dumps(expected) + "\n```"

    strict = ft.score_task(task, fenced)
    neutral, normalized = sweep._contract_neutral_score(task, fenced)

    assert strict.correct is False
    assert neutral.correct is True
    assert normalized is True

    wrong = dict(expected)
    wrong["actual_score"] += 1
    neutral_wrong, _normalized = sweep._contract_neutral_score(
        task,
        "```json\n" + json.dumps(wrong) + "\n```",
    )
    assert neutral_wrong.correct is False


def test_config_carries_the_arm_policy_and_validates():
    for steps, policy in ((4, "applied"), (1, "suppressed")):
        config = sweep._build_config(steps, 16, policy, 320, profile="mechanism")
        assert config.validate() == []
        assert config.terminal_instruction_policy == policy
        assert config.recurrence.max_steps == steps
        # The bridge stays off in every arm so the only prefix difference
        # between arms is the disposition being measured.
        assert config.decode_bridge_policy == "none"


def test_journal_resumes_and_ignores_a_torn_final_line(tmp_path: Path):
    path = tmp_path / "journal.jsonl"
    journal = sweep.Journal(path)
    journal.append({"event": "CELL", "arm": "vanilla", "task_id": "t1", "text": "x"})
    journal.append({"event": "CELL", "arm": "vanilla", "task_id": "t2", "text": "y"})
    # Simulate a hard kill mid-write.
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "CELL", "arm": "vanil')

    resumed = sweep.Journal(path)
    assert ("vanilla", "t1") in resumed.done
    assert ("vanilla", "t2") in resumed.done
    assert len(resumed.done) == 2
    assert len(resumed.cells()) == 2


def test_grade_refuses_to_credit_a_harness_fault_as_a_wrong_answer(tmp_path: Path):
    """A crashed cell is an error, never a scored zero."""
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    task = tasks[0]

    journal = sweep.Journal(tmp_path / "journal.jsonl")
    journal.append(
        {
            "event": "CELL",
            "arm": "rlc_asrun",
            "task_id": task.task_id,
            "domain": task.domain,
            "text": "",
            "error": "RuntimeError: worker died",
        }
    )
    verdict = sweep.grade(tmp_path, [task])
    bucket = verdict["arms"]["rlc_asrun"]
    assert bucket["errors"] == 1
    assert bucket["total"] == 1
    # The failure did not become a graded observation.
    assert bucket["correct"] == 0
    assert bucket["reasons"] == {}


def test_grade_decides_against_the_ordinary_decode_not_against_itself(tmp_path: Path):
    """Parity is measured against vanilla. A recurrent arm cannot grade itself."""
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")

    for task in tasks:
        reveal = task.reveal_for_verifier()
        correct_line = "FINAL_ANSWER: " + json.dumps(reveal["expected"])
        # Vanilla answers everything correctly; the recurrent arm answers none.
        journal.append(
            {
                "event": "CELL",
                "arm": "vanilla",
                "task_id": task.task_id,
                "domain": task.domain,
                "text": correct_line,
                "error": "",
            }
        )
        journal.append(
            {
                "event": "CELL",
                "arm": "rlc_asrun",
                "task_id": task.task_id,
                "domain": task.domain,
                "text": 'FINAL_ANSWER: {"wrong": 1}',
                "error": "",
            }
        )

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["vanilla_correct"] == len(tasks)
    assert verdict["best_recurrent_correct"] == 0
    assert verdict["reaches_parity_with_ordinary_decode"] is False
    # The product claims an incumbent floor. Falling below a right vanilla
    # answer therefore invalidates that contract; it is not a negative result
    # about the complete engine.
    assert verdict["decision"] == "invalid_full_stack_violated_vanilla_incumbent"
    assert verdict["paired_vanilla_floor"]["holds"] is False
    # A negative sweep never authorizes downstream work.
    assert verdict["claims"]["fusion_authorized"] is False
    assert verdict["claims"]["reasoning_gain_proven"] is False


def test_grade_promotes_only_when_a_recurrent_arm_reaches_parity(tmp_path: Path):
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")
    for task in tasks:
        reveal = task.reveal_for_verifier()
        correct_line = "FINAL_ANSWER: " + json.dumps(reveal["expected"])
        for arm in ("vanilla", "rlc_nodisp"):
            journal.append(
                {
                    "event": "CELL",
                    "arm": arm,
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "text": correct_line,
                    "error": "",
                }
            )
    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["reaches_parity_with_ordinary_decode"] is True
    assert verdict["decision"] == "proceed_to_checkpoint_phase"
    # Parity on a frozen path still proves no gain and authorizes no fusion.
    assert verdict["claims"]["fusion_authorized"] is False


class _StubTokenizer:
    """Enough tokenizer to make the disposition text real tokens."""

    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):  # noqa: D102, ARG002
        return [1 + (ord(ch) % 100) for ch in text[:64]]

    def decode(self, tokens):  # noqa: D102
        return "".join(chr(65 + (int(t) % 26)) for t in tokens)


def _tiny_model():
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    args = ModelArgs(
        model_type="qwen2",
        hidden_size=64,
        num_hidden_layers=8,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def test_disposition_injection_is_real_when_applied_and_absent_when_suppressed():
    """The load-bearing claim: the arms differ by exactly this injection.

    With no tokenizer the engine cannot encode the disposition at all, so a
    test that omits one proves nothing -- both arms trivially report zero.
    """
    import mlx.core as mx

    # Seed explicitly. The tiny model has random weights, so whether it emits
    # EOS on the first token depends on global MLX RNG state -- which earlier
    # tests in the same process have already advanced. Run alone this passed;
    # run inside the suite it raised "episode produced no answer", failing a
    # 29-minute gate for a reason that had nothing to do with the code under
    # test. The assertions below are about prefix composition and never about
    # the generated text, so determinism is all this needs.
    mx.random.seed(20260807)
    model = _tiny_model()

    applied_config = sweep._build_config(2, 4, "applied", 8, decode_contract="none")
    _, applied = sweep._run_rlc(model, applied_config, [1, 2, 3, 4, 5], _StubTokenizer())

    suppressed_config = sweep._build_config(2, 4, "suppressed", 8, decode_contract="none")
    _, suppressed = sweep._run_rlc(model, suppressed_config, [1, 2, 3, 4, 5], _StubTokenizer())

    applied_prefix = applied["decode_prefix_composition"]
    suppressed_prefix = suppressed["decode_prefix_composition"]

    assert applied_prefix["terminal_instruction_tokens"] > 0
    assert suppressed_prefix["terminal_instruction_tokens"] == 0
    assert applied["decode_prefix_token_count"] > 0
    assert suppressed["decode_prefix_token_count"] == 0
    # Neither arm configured a bridge policy, so neither may claim one.
    assert applied["decode_bridge_applied"] is False
    assert suppressed["decode_bridge_applied"] is False
    assert applied["decode_bridge_token_count"] == 0


def test_run_rlc_passes_task_domain_and_cognitive_context_to_engine(monkeypatch):
    captured = {}

    class _Receipt:
        @staticmethod
        def to_dict():
            return {"decode_termination": "contract_complete", "honest_flags": []}

    class _Result:
        ok = True
        text = "FINAL_ANSWER: {}"
        tokens = [1]
        reason = "complete"
        receipt = _Receipt()

    class _CapturingEngine:
        def __init__(self, *args, **kwargs):  # noqa: D107, ARG002
            pass

        def reason(self, **kwargs):
            captured.update(kwargs)
            return _Result()

    import core.brain.llm.latent_cortex.engine as engine_module

    monkeypatch.setattr(engine_module, "LatentCortexEngine", _CapturingEngine)
    model = _tiny_model()
    config = sweep._build_config(2, 4, "suppressed", 8, decode_contract="none")
    context = [{"source": "test", "text": "bounded observation"}]

    def progress(_payload):
        return None

    sweep._run_rlc(
        model,
        config,
        [1, 2, 3],
        _StubTokenizer(),
        objective="test objective",
        domain="scientific_inference",
        cognitive_context=context,
        progress=progress,
    )

    assert captured["domain"] == "scientific_inference"
    assert captured["cognitive_context"] == context
    assert captured["messages"] == [{"role": "user", "content": "test objective"}]
    assert captured["progress"] is progress


def test_complete_system_promotion_preserves_incumbent_until_verified_improvement():
    class _Verifier:
        @staticmethod
        def evaluate(text, _record=False):  # noqa: ARG004
            score = {"incumbent": 0.6, "candidate": 0.9}.get(text, 0.0)
            return {
                "score": score,
                "applicable_checks": ["response_contract", "deterministic_router"],
                "checks": {
                    "response_contract": {"valid": True},
                    "atomic_decomposition": {"valid": True},
                    "deterministic_router": {
                        "valid": True,
                        "receipt": {
                            "routes": (
                                [
                                    {
                                        "verifier": "exact_objective_program",
                                        "outcome": "verified",
                                    }
                                ]
                                if text == "candidate"
                                else []
                            )
                        },
                    },
                    "arithmetic": {"failures": []},
                    "code": {"failures": []},
                },
            }

    retained, retained_receipt = sweep._promotion_assessment(
        verifier=_Verifier(),
        incumbent_text="incumbent",
        candidate_text="candidate",
        candidate_verified=False,
    )
    assert retained == "incumbent"
    assert retained_receipt["decision"] == "retain"
    assert retained_receipt["reason"] == "candidate_not_verified"

    promoted, promoted_receipt = sweep._promotion_assessment(
        verifier=_Verifier(),
        incumbent_text="incumbent",
        candidate_text="candidate",
        candidate_verified=True,
        authority="public_objective_deterministic_execution",
    )
    assert promoted == "candidate"
    assert promoted_receipt["decision"] == "replace"
    assert promoted_receipt["answer_key_used"] is False
    assert promoted_receipt["no_regression_guaranteed"] is True

    proxy_only, proxy_receipt = sweep._promotion_assessment(
        verifier=_Verifier(),
        incumbent_text="incumbent",
        candidate_text="candidate",
        candidate_verified=True,
    )
    assert proxy_only == "incumbent"
    assert proxy_receipt["reason"] == "candidate_lacks_exact_public_objective_proof"

    consensus, consensus_receipt = sweep._promotion_assessment(
        verifier=_Verifier(),
        incumbent_text="incumbent",
        candidate_text="candidate",
        candidate_verified=True,
        authority="independent_executable_consensus",
    )
    assert consensus == "incumbent"
    assert consensus_receipt["decision"] == "retain"
    assert consensus_receipt["reason"] == ("probabilistic_consensus_not_promotion_authority")
    assert consensus_receipt["promotion_is_probabilistic"] is True
    assert consensus_receipt["ground_truth_verified"] is False
    assert consensus_receipt["no_regression_guaranteed"] is False


def test_complete_system_selection_preserves_exact_rlc_output_over_weaker_amplifier():
    from tools.rlc_complete_system_closed_book import (
        _select_complete_system_promotion_candidate,
    )

    class _Verifier:
        @staticmethod
        def evaluate(text, _record=False):  # noqa: ARG004
            exact = text == "exact-rlc"
            return {
                "score": 1.0 if exact else 0.25,
                "applicable_checks": ["response_contract", "deterministic_router"],
                "checks": {
                    "response_contract": {"valid": True},
                    "atomic_decomposition": {"valid": True},
                    "deterministic_router": {
                        "valid": True,
                        "receipt": {
                            "routes": (
                                [
                                    {
                                        "verifier": "exact_objective_program",
                                        "outcome": "verified",
                                    }
                                ]
                                if exact
                                else []
                            )
                        },
                    },
                    "arithmetic": {"failures": []},
                    "code": {"failures": []},
                },
            }

    selected, receipt = _select_complete_system_promotion_candidate(
        verifier=_Verifier(),
        rlc_text="exact-rlc",
        amplifier_text="weaker-amplifier",
        amplifier_consensus_programs=0,
        amplifier_consensus_strategies=0,
    )

    assert selected == "exact-rlc"
    assert receipt["selected_source"] == "rlc_final"
    assert receipt["authority"] == "public_objective_deterministic_execution"
    promoted, promotion = sweep._promotion_assessment(
        verifier=_Verifier(),
        incumbent_text="incumbent",
        candidate_text=selected,
        candidate_verified=True,
        authority=receipt["authority"],
    )
    assert promoted == "exact-rlc"
    assert promotion["decision"] == "replace"


def _run_with_dead_engine(model, config, reason: str, termination: str):
    class _DeadResult:
        ok = False
        tokens: list[int] = []
        text = ""

        def __init__(self):
            self.reason = reason

        class receipt:  # noqa: N801
            @staticmethod
            def to_dict():
                return {"decode_termination": termination}

    class _DeadEngine:
        def __init__(self, *args, **kwargs):
            pass

        def reason(self, **kwargs):  # noqa: ARG002
            return _DeadResult()

    import core.brain.llm.latent_cortex.engine as engine_module

    original = engine_module.LatentCortexEngine
    engine_module.LatentCortexEngine = _DeadEngine
    try:
        return sweep._run_rlc(model, config, [1, 2, 3], _StubTokenizer())
    finally:
        engine_module.LatentCortexEngine = original


def test_infrastructure_failure_raises_instead_of_scoring_zero():
    """A broken harness must not be gradeable as a wrong answer.

    The first live sweep died on the engine's default 120s episode wall clock,
    which is smaller than these episodes need -- the 2026-08-06 campaign's
    median recurrent episode ran 298s.
    """
    model = _tiny_model()
    config = sweep._build_config(2, 4, "applied", 8, decode_contract="none")

    with pytest.raises(sweep.EpisodeFault) as excinfo:
        _run_with_dead_engine(model, config, "latent_phase_failed:ValueError:boom", "not_reached")
    assert "latent_phase_failed" in str(excinfo.value)


def test_a_model_that_cannot_finish_its_answer_is_scored_not_excluded():
    """An unfinished decode is the arm failing to answer, which is a result.

    CP420S12 settled this: bounded abstentions and incomplete decodes are
    scored as incorrect policy observations, while cancellation, latent-phase,
    worker and invariant failures stay fatal. The 2026-08-06 base_rlc arm
    carried nine such policy failures out of 28, so excluding them would
    flatter the recurrent path rather than measure it.
    """
    model = _tiny_model()
    config = sweep._build_config(2, 4, "applied", 8, decode_contract="none")

    text, receipt = _run_with_dead_engine(
        model,
        config,
        "decode_incomplete:contract_irrecoverable",
        "contract_irrecoverable",
    )
    assert text == ""
    assert receipt["decode_termination"] == "contract_irrecoverable"

    assert sweep._is_policy_failure("decode_incomplete:x", "contract_irrecoverable")
    assert sweep._is_policy_failure("", "token_limit_contract_incomplete")
    # Infrastructure never counts as policy, even when it mentions a budget.
    assert not sweep._is_policy_failure("latent_phase_failed:budget_exhausted", "")
    assert not sweep._is_policy_failure("worker_died", "budget_exhausted")


def test_a_faulted_arm_makes_the_sweep_inconclusive_not_negative(tmp_path: Path):
    """An arm that did not run has not lost. It has not been measured."""
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")
    for task in tasks:
        reveal = task.reveal_for_verifier()
        journal.append(
            {
                "event": "CELL",
                "arm": "vanilla",
                "task_id": task.task_id,
                "domain": task.domain,
                "text": "FINAL_ANSWER: " + json.dumps(reveal["expected"]),
                "error": "",
            }
        )
        journal.append(
            {
                "event": "CELL",
                "arm": "rlc_asrun",
                "task_id": task.task_id,
                "domain": task.domain,
                "text": "",
                "error": "EpisodeFault: episode produced no answer",
            }
        )

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["arms_complete"] is False
    assert verdict["faulted_arms"]["rlc_asrun"] == len(tasks)
    assert verdict["decision"] == "inconclusive_arms_carry_harness_faults"
    # Crucially it does NOT report the recurrent path as below vanilla.
    assert verdict["reaches_parity_with_ordinary_decode"] is False
    assert verdict["arms"]["rlc_asrun"]["correct"] == 0


def test_mutual_failure_is_not_parity(tmp_path: Path):
    """0 >= 0 satisfies the parity inequality. It must not satisfy the gate.

    A battery the ordinary decode cannot score on has not measured recurrence
    at all, and promoting on it would advance a model that answered nothing.
    """
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")

    for task in tasks:
        for arm in ("vanilla", "rlc_asrun"):
            journal.append(
                {
                    "event": "CELL",
                    "arm": arm,
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "text": "I am not able to answer this.",
                    "error": "",
                }
            )

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["vanilla_correct"] == 0
    assert verdict["best_recurrent_correct"] == 0
    assert verdict["battery_informative"] is False
    assert verdict["reaches_parity_with_ordinary_decode"] is False
    assert verdict["decision"] == "inconclusive_battery_uninformative_ordinary_decode_scored_zero"
    assert verdict["claims"]["fusion_authorized"] is False


def test_one_solved_control_task_makes_the_battery_informative(tmp_path: Path):
    """The floor is structural: a baseline exists, or it does not."""
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")

    for index, task in enumerate(tasks):
        reveal = task.reveal_for_verifier()
        correct_line = "FINAL_ANSWER: " + json.dumps(reveal["expected"])
        # Exactly one control task is solved, by both arms.
        text = correct_line if index == 0 else "no answer"
        for arm in ("vanilla", "rlc_asrun"):
            journal.append(
                {
                    "event": "CELL",
                    "arm": arm,
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "text": text,
                    "error": "",
                }
            )

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["vanilla_correct"] == 1
    assert verdict["battery_informative"] is True
    assert verdict["reaches_parity_with_ordinary_decode"] is True
    assert verdict["decision"] == "proceed_to_checkpoint_phase"


def test_strict_lift_from_zero_baseline_is_informative_not_mutual_failure(tmp_path: Path):
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([202608071], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")
    for task in tasks:
        correct = "FINAL_ANSWER: " + json.dumps(task.reveal_for_verifier()["expected"])
        for arm, text in (
            ("vanilla", "I am not able to answer this."),
            ("rlc_asrun", correct),
        ):
            journal.append(
                {
                    "event": "CELL",
                    "arm": arm,
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "text": text,
                    "error": "",
                }
            )

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["vanilla_correct"] == 0
    assert verdict["best_recurrent_correct"] == len(tasks)
    assert verdict["battery_informative"] is True
    assert verdict["reaches_parity_with_ordinary_decode"] is True
    assert verdict["decision"] == "proceed_to_checkpoint_phase"


def test_a_cell_from_a_superseded_decode_configuration_is_re_run(tmp_path: Path):
    """The defect that cost two restarts: a resumed run reused cells produced
    under an older decode configuration, so the control and the treatment were
    compared across different rules. Configuration identity travels with the
    cell, and a mismatch is treated as absent rather than as evidence."""
    current = sweep.decode_fingerprint(
        model="/models/resident",
        n_slots=16,
        max_tokens=320,
        episode_wall_s=720.0,
        seed=20260807,
        per_domain=4,
    )
    superseded = sweep.decode_fingerprint(
        model="/models/resident",
        n_slots=16,
        max_tokens=192,  # the only difference, and it changes every answer
        episode_wall_s=720.0,
        seed=20260807,
        per_domain=4,
    )
    assert current != superseded

    path = tmp_path / "journal.jsonl"
    stale = sweep.Journal(path, superseded)
    stale.append(
        {
            "event": "CELL",
            "arm": "vanilla",
            "task_id": "task-a",
            "domain": "mathematics",
            "decode_fingerprint": superseded,
            "text": "FINAL_ANSWER: {}",
            "error": "",
        }
    )

    resumed = sweep.Journal(path, current)
    assert resumed.done == set(), "a superseded cell must not count as committed"
    assert resumed.superseded == 1
    assert resumed.cells() == [], "and must not be graded"

    # Under its own fingerprint it is still perfectly good evidence.
    replayed = sweep.Journal(path, superseded)
    assert replayed.done == {("vanilla", "task-a")}


def test_a_source_change_retires_every_cell_from_the_old_engine(tmp_path: Path):
    common = dict(
        model="/models/resident",
        n_slots=16,
        max_tokens=320,
        episode_wall_s=720.0,
        seed=20260807,
        per_domain=4,
        arm="full_stack",
    )
    old = sweep.decode_fingerprint(implementation_sha256="a" * 64, **common)
    current = sweep.decode_fingerprint(implementation_sha256="b" * 64, **common)
    assert old != current

    path = tmp_path / "journal.jsonl"
    sweep.Journal(path).append(
        {
            "event": "CELL",
            "arm": "full_stack",
            "task_id": "task-a",
            "decode_fingerprint": old,
            "text": "FINAL_ANSWER: {}",
            "error": "",
        }
    )
    resumed = sweep.Journal(path, {"full_stack": current})
    assert resumed.done == set()
    assert resumed.superseded == 1


def test_an_unfingerprinted_journal_is_still_readable(tmp_path: Path):
    """Older runs and unit fixtures carry no fingerprint; they are admitted."""
    path = tmp_path / "journal.jsonl"
    journal = sweep.Journal(path)
    journal.append(
        {
            "event": "CELL",
            "arm": "vanilla",
            "task_id": "task-a",
            "domain": "mathematics",
            "text": "FINAL_ANSWER: {}",
            "error": "",
        }
    )
    assert sweep.Journal(path).done == {("vanilla", "task-a")}
    assert sweep.Journal(path).superseded == 0


def test_no_recurrent_arm_is_not_a_verdict_about_recurrence(tmp_path: Path):
    """Grading a vanilla-only run reported recurrent_path_below_ordinary_decode
    off a -1 sentinel. A conclusion about the recurrent path requires having
    run one."""
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")
    for task in tasks:
        reveal = task.reveal_for_verifier()
        journal.append(
            {
                "event": "CELL",
                "arm": "vanilla",
                "task_id": task.task_id,
                "domain": task.domain,
                "text": "FINAL_ANSWER: " + json.dumps(reveal["expected"]),
                "error": "",
            }
        )

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["vanilla_correct"] == len(tasks)
    assert verdict["best_recurrent_correct"] == -1
    assert verdict["reaches_parity_with_ordinary_decode"] is False
    assert verdict["decision"] == "inconclusive_no_recurrent_arm_measured"
    assert verdict["claims"]["fusion_authorized"] is False


def test_complete_system_is_the_only_claimant_when_narrower_engine_arm_is_present(
    tmp_path: Path,
):
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")
    for index, task in enumerate(tasks):
        correct = "FINAL_ANSWER: " + json.dumps(task.reveal_for_verifier()["expected"])
        wrong = 'FINAL_ANSWER: {"wrong": 1}'
        for arm, text in (
            ("vanilla", correct if index == 0 else wrong),
            ("vanilla_equal_compute", correct if index == 0 else wrong),
            ("full_stack", correct),
            ("complete_system_closed_book", correct if index == 0 else wrong),
        ):
            journal.append(
                {
                    "event": "CELL",
                    "arm": arm,
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "text": text,
                    "error": "",
                    "answer_replacement_decision": "retain",
                }
            )

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["arms"]["full_stack"]["correct"] == len(tasks)
    assert verdict["best_recurrent_arm"] == "complete_system_closed_book"
    assert verdict["best_recurrent_correct"] == 1


def test_per_arm_fingerprints_retire_only_the_arm_that_changed(tmp_path: Path):
    """Production passes a per-arm mapping. Raising one arm's budget must not
    discard the arms whose configuration is untouched."""
    common = dict(
        model="/models/resident",
        n_slots=16,
        episode_wall_s=720.0,
        seed=20260807,
        per_domain=4,
    )
    vanilla_fp = sweep.decode_fingerprint(max_tokens=512, arm="vanilla", **common)
    long_512 = sweep.decode_fingerprint(max_tokens=512, arm="vanilla_long", **common)
    long_1024 = sweep.decode_fingerprint(max_tokens=1024, arm="vanilla_long", **common)

    path = tmp_path / "journal.jsonl"
    writer = sweep.Journal(path)
    for arm, fp in (("vanilla", vanilla_fp), ("vanilla_long", long_512)):
        writer.append(
            {
                "event": "CELL",
                "arm": arm,
                "task_id": "task-a",
                "domain": "mathematics",
                "decode_fingerprint": fp,
                "text": "FINAL_ANSWER: {}",
                "error": "",
            }
        )

    resumed = sweep.Journal(path, {"vanilla": vanilla_fp, "vanilla_long": long_1024})
    assert resumed.done == {("vanilla", "task-a")}
    assert resumed.superseded == 1

    # An arm dropped from the configuration stops counting as evidence.
    narrowed = sweep.Journal(path, {"vanilla": vanilla_fp})
    assert narrowed.done == {("vanilla", "task-a")}
    assert narrowed.superseded == 1


def test_latency_is_reported_beside_accuracy(tmp_path: Path):
    """A unified system that answers better but takes ten minutes has not been
    shown to be deployable. The program's own standard requires equal-latency
    and equal-compute comparisons, so cost travels with the verdict."""
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")
    for task in tasks:
        reveal = task.reveal_for_verifier()
        correct = "FINAL_ANSWER: " + json.dumps(reveal["expected"])
        journal.append(
            {
                "event": "CELL",
                "arm": "vanilla",
                "task_id": task.task_id,
                "domain": task.domain,
                "text": correct,
                "error": "",
                "latency_s": 40.0,
            }
        )
        journal.append(
            {
                "event": "CELL",
                "arm": "full_stack",
                "task_id": task.task_id,
                "domain": task.domain,
                "text": correct,
                "error": "",
                "latency_s": 400.0,
                "steps_taken": 3,
                "halted_early": True,
            }
        )

    verdict = sweep.grade(tmp_path, tasks)
    full = verdict["arms"]["full_stack"]
    assert full["latency_median_s"] == 400.0
    assert full["steps_median"] == 3
    # Adaptive halting is the latency lever; its use must be visible.
    assert full["halted_early_fraction"] == 1.0
    # Ten times the cost for the same score is a reportable result.
    assert verdict["latency_ratio_vs_ordinary_decode"]["full_stack"] == 10.0
    assert verdict["latency_ratio_vs_ordinary_decode"]["vanilla"] == 1.0


def test_equal_compute_control_can_never_be_named_the_recurrent_winner(tmp_path: Path):
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")
    for index, task in enumerate(tasks):
        correct = "FINAL_ANSWER: " + json.dumps(task.reveal_for_verifier()["expected"])
        wrong = 'FINAL_ANSWER: {"wrong": 1}'
        for arm, text in (
            ("vanilla", correct if index == 0 else wrong),
            ("vanilla_equal_compute", correct),
            ("full_stack", correct if index == 0 else wrong),
        ):
            journal.append(
                {
                    "event": "CELL",
                    "arm": arm,
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "text": text,
                    "error": "",
                    "answer_replacement_decision": "retain" if arm == "full_stack" else "",
                }
            )

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["best_recurrent_arm"] == "full_stack"
    assert verdict["best_recurrent_correct"] == 1
    assert verdict["vanilla_equal_compute_correct"] == len(tasks)
    assert verdict["beats_equal_compute_control"] is False
    assert verdict["resource_matched_control_proven"] is False
    assert verdict["control_contracts"]["vanilla_equal_compute"] == {
        "artifact_compatible_name": True,
        "selection": "best_of_3_self_consistency",
        "resource_matched": False,
        "claim_authority": "preliminary_only",
        "required_claim_successor": "digest_bound_paired_resource_certificate",
    }


def test_best_of_three_win_is_not_misreported_as_equal_compute(tmp_path: Path):
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    journal = sweep.Journal(tmp_path / "journal.jsonl")
    for index, task in enumerate(tasks):
        correct = "FINAL_ANSWER: " + json.dumps(task.reveal_for_verifier()["expected"])
        wrong = 'FINAL_ANSWER: {"wrong": 1}'
        for arm, text in (
            ("vanilla", correct if index == 0 else wrong),
            ("vanilla_equal_compute", correct if index == 0 else wrong),
            ("full_stack", correct),
        ):
            journal.append(
                {
                    "event": "CELL",
                    "arm": arm,
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "text": text,
                    "error": "",
                    "answer_replacement_decision": "replace" if arm == "full_stack" else "",
                }
            )

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["outscored_preliminary_best_of_3"] is True
    assert verdict["beats_equal_compute_control"] is False
    assert verdict["resource_matched_control_proven"] is False


def test_complete_system_can_earn_a_conservative_resource_advantaged_win(
    tmp_path: Path,
):
    from core.brain.llm.latent_cortex import frontier_tasks as ft
    from core.brain.llm.latent_cortex.resource_accounting import (
        ModelComputeProfile,
        ResourceLedger,
        build_information_receipt,
        certify_control_resource_dominance,
        policy_sha256,
    )

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    profile = ModelComputeProfile(
        model_type="fixture",
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        head_dim=4,
    )
    treatment = ResourceLedger(profile)
    treatment.charge(
        "treatment",
        transformer_layer_apps=10,
        tensor_element_reads=10,
        tensor_element_writes=10,
        verifier_calls=1,
        verifier_input_bytes=10,
        verifier_output_bytes=10,
    )
    control = ResourceLedger(profile)
    control.charge(
        "control",
        transformer_layer_apps=20,
        tensor_element_reads=20,
        tensor_element_writes=20,
        verifier_calls=2,
        verifier_input_bytes=20,
        verifier_output_bytes=20,
    )
    information = build_information_receipt(
        sources=[
            {
                "source_id": "prompt",
                "kind": "model_input_tokens",
                "content_sha256": hashlib.sha256(b"same").hexdigest(),
                "byte_count": 4,
                "token_count": 1,
            }
        ],
        policies={"verifier": policy_sha256({"kind": "candidate_local"})},
    )
    control_setup = ResourceLedger(profile).to_receipt()
    control_aggregate = ResourceLedger.aggregate([control_setup, control.to_receipt()]).to_receipt()
    certificate = certify_control_resource_dominance(
        treatment_resource=treatment.to_receipt(),
        control_resource=control_aggregate,
        treatment_information=information,
        control_information=information,
    )
    journal = sweep.Journal(tmp_path / "journal.jsonl")
    for index, task in enumerate(tasks):
        correct = "FINAL_ANSWER: " + json.dumps(task.reveal_for_verifier()["expected"])
        wrong = 'FINAL_ANSWER: {"wrong": 1}'
        for arm, text in (
            ("vanilla", correct if index == 0 else wrong),
            ("vanilla_equal_compute", correct if index == 0 else wrong),
            ("complete_system_closed_book", correct),
            (sweep.RESOURCE_DOMINATING_CONTROL_ARM, wrong),
        ):
            runtime_fields = {}
            if arm == "complete_system_closed_book":
                path, digest = sweep._persist_runtime_receipt(
                    tmp_path,
                    arm=arm,
                    task_id=task.task_id,
                    receipt={
                        "complete_system_closed_book": {
                            "objective": task.public.prompt,
                            "cognitive_acquisition": {"status": "not_requested"},
                        }
                    },
                )
                runtime_fields = {
                    "runtime_receipt_path": path,
                    "runtime_receipt_sha256": digest,
                }
            if arm == sweep.RESOURCE_DOMINATING_CONTROL_ARM:
                verifier = sweep._episode_verifier(task)
                verifier_score = float(verifier(wrong))
                receipt_body = {
                    "schema": "aura.rlc.resource_dominating_control.v1",
                    "task_id": task.task_id,
                    "campaign_seed": 7,
                    "sample_limit": 1,
                    "sample_count": 1,
                    "equal_tool_cycle_count": 0,
                    "equal_tool_cycle_limit": 0,
                    "generated_tokens": 1,
                    "setup_resource_accounting": control_setup,
                    "control_acquisition": {
                        "schema": "aura.rlc.resource_control_acquisition.v1",
                        "status": "not_required",
                    },
                    "candidates": [
                        {
                            "sample_index": 0,
                            "control_path": "ordinary_sample",
                            "text": wrong,
                            "text_sha256": hashlib.sha256(wrong.encode()).hexdigest(),
                            "verifier_score": verifier_score,
                            "verifier_receipt": verifier.to_receipt(),
                            "resource_accounting": control.to_receipt(),
                            "equal_tool_resource_accounting": None,
                            "generated_tokens": 1,
                            "equal_tool_receipt": None,
                        }
                    ],
                    "selected_index": 0,
                    "selected_text_sha256": hashlib.sha256(wrong.encode()).hexdigest(),
                    "resource_accounting": control_aggregate,
                    "information_accounting": information,
                    "resource_dominance_certificate": certificate,
                }
                runtime_receipt = {
                    **receipt_body,
                    "receipt_sha256": policy_sha256(receipt_body),
                }
                path, digest = sweep._persist_runtime_receipt(
                    tmp_path,
                    arm=arm,
                    task_id=task.task_id,
                    receipt=runtime_receipt,
                )
                runtime_fields = {
                    "runtime_receipt_path": path,
                    "runtime_receipt_sha256": digest,
                }
            journal.append(
                {
                    "event": "CELL",
                    "arm": arm,
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "text": text,
                    "error": "",
                    "answer_replacement_decision": (
                        "replace" if arm == "complete_system_closed_book" else ""
                    ),
                    "resource_accounting": (
                        control_aggregate
                        if arm == sweep.RESOURCE_DOMINATING_CONTROL_ARM
                        else treatment.to_receipt()
                    ),
                    "information_accounting": information,
                    "resource_dominance_certificate": (
                        certificate if arm == sweep.RESOURCE_DOMINATING_CONTROL_ARM else None
                    ),
                    **runtime_fields,
                }
            )

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["resource_advantaged_control_proven"] is True, (
        verdict.get("resource_dominance_issues"),
        verdict.get("mechanism_issues"),
        verdict.get("complete"),
    )
    assert verdict["outscored_resource_advantaged_control"] is True
    assert verdict["decision"] == "proceed_to_checkpoint_phase"


def test_manifest_requires_every_control_and_treatment_cell(tmp_path: Path):
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    required = ["vanilla", "vanilla_equal_compute", "full_stack"]
    _write_evidence_manifest(
        tmp_path,
        tasks,
        required_arms=required,
        requested_arms=["full_stack"],
    )
    task = tasks[0]
    correct = "FINAL_ANSWER: " + json.dumps(task.reveal_for_verifier()["expected"])
    journal = sweep.Journal(tmp_path / "journal.jsonl")
    for arm in ("vanilla", "vanilla_equal_compute", "full_stack"):
        journal.append(
            {
                "event": "CELL",
                "arm": arm,
                "task_id": task.task_id,
                "domain": task.domain,
                "decode_fingerprint": "a" * 64,
                "text": correct,
                "error": "",
            }
        )

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["coverage_complete"] is False
    assert set(verdict["missing_cells"]) == {
        "vanilla",
        "vanilla_equal_compute",
        "full_stack",
    }
    assert verdict["decision"] == "inconclusive_campaign_incomplete"


def test_manifest_tampering_invalidates_the_campaign(tmp_path: Path):
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    required = ["vanilla", "vanilla_equal_compute", "full_stack"]
    _write_evidence_manifest(
        tmp_path,
        tasks,
        required_arms=required,
        requested_arms=["full_stack"],
    )
    path = tmp_path / "decode_fingerprint.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["task_commitment_sha256"] = "0" * 64
    manifest["implementation_sha256"] = "1" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")

    verdict = sweep.grade(tmp_path, tasks)
    assert verdict["evidence_manifest_valid"] is False
    assert verdict["decision"] == "inconclusive_evidence_manifest_invalid"
    assert verdict["evidence_manifest_issues"] == [
        "implementation_identity_mismatch",
        "task_commitment_mismatch",
    ]


def test_claim_manifest_cannot_omit_stage_required_control(tmp_path: Path):
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    _write_evidence_manifest(
        tmp_path,
        tasks,
        required_arms=["vanilla", "full_stack"],
        requested_arms=["full_stack"],
    )
    verdict = sweep.grade(tmp_path, tasks)
    assert "campaign_stage_arm_set_mismatch" in verdict["evidence_manifest_issues"]
    assert verdict["decision"] == "inconclusive_evidence_manifest_invalid"


def test_component_manifest_does_not_require_expensive_stage_controls(tmp_path: Path):
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    tasks = ft.generate_task_battery([20260807], difficulty=2)
    required = [
        "vanilla",
        "complete_system_closed_book",
        "complete_system_adaptation_ablation",
        "complete_system_executable_ablation",
    ]
    _write_evidence_manifest(
        tmp_path,
        tasks,
        required_arms=required,
        requested_arms=["complete_system_closed_book"],
        campaign_stage="component",
    )

    manifest = json.loads((tmp_path / "decode_fingerprint.json").read_text())
    assert sweep._manifest_integrity_issues(manifest, tasks) == []


def test_full_stack_receipt_must_measure_every_claimed_mechanism(tmp_path: Path):
    from core.brain.llm.latent_cortex.causal_receipt import build_causal_receipt
    from core.brain.llm.latent_cortex.incumbent_artifact import (
        build_incumbent_artifact,
    )
    from core.brain.llm.latent_cortex.runtime_integrity import (
        bind_worker_runtime_integrity,
    )
    from tests.fixtures.rlc_runtime_integrity import (
        complete_worker_identity,
        engine_runtime_integrity,
    )

    incumbent = build_incumbent_artifact(
        input_tokens=[1, 2, 3],
        output_tokens=[4, 5],
        output_text="answer",
        checkpoint_fingerprint="a" * 64,
        checkpoint_fingerprint_method="sha256",
        max_tokens=16,
        n_layers=8,
        termination="contract_complete",
    ).receipt
    episode_id = "full-stack-evidence-test"
    input_sha256 = "9" * 64
    worker_identity = complete_worker_identity(model_path="/models/test-32b")
    runtime_integrity = bind_worker_runtime_integrity(
        engine_runtime_integrity(
            episode_id=episode_id,
            input_tokens_sha256=input_sha256,
            checkpoint_fingerprint="a" * 64,
            checkpoint_file_count=1,
        ),
        worker_identity=worker_identity,
    )
    valid_receipt = {
        "episode_id": episode_id,
        "request_payload_sha256": "8" * 64,
        "input_tokens_sha256": input_sha256,
        "input_token_count": 3,
        "runtime_identity": {"identity_bound": True},
        "worker_identity": worker_identity,
        "runtime_integrity": runtime_integrity,
        "worker_boot_id": worker_identity["worker_boot_id"],
        "worker_pid": worker_identity["worker_pid"],
        "worker_model_path": worker_identity["worker_model_path"],
        "worker_source_sha256": worker_identity["worker_source_sha256"],
        "n_slots": 16,
        "cognitive_slots": [],
        "steps_taken": 3,
        "n_branches": 2,
        "branch_isolation": {
            "certified": True,
            "candidates": [
                {"role": "planner"},
                {"role": "critic"},
            ],
        },
        "branch_exchange": {
            "schema": "aura.rlc.branch_exchange_trace.v1",
            "exchanges": [],
        },
        "recurrent_grounding": {"bound": True},
        "loop_stability": {"stable": True},
        "kv_state_tree": {"root_sha256": "7" * 64},
        "selected_branch": 0,
        "verifier_preflight": {"verifier_admitted": True},
        "verifier_fusion": {"measured": True},
        "neural_uncertainty": {"measured": True},
        "mistake_locator": {"measured": True},
        "update_acceptance": {"measured": True},
        "verified_best_state": {"measured": True},
        "latent_opt_mode": "gradient",
        "latent_opt_applied": True,
        "latent_opt_attempts": 4,
        "latent_opt_steps": 1,
        "fast_weight_learning": {"disposition": "not_admitted_high_confidence_evidence_absent"},
        "value_of_computation": {"continue": False},
        "cognitive_action_trace": [{"action": "halt"}],
        "diagnostic_action_selection": {"selected": "verify"},
        "local_repair": {"requests": []},
        "answer_replacement": {
            "decision": "retain",
            "baseline_decode": {
                "text_sha256": incumbent["output"]["text_sha256"],
                "tokens_sha256": incumbent["output"]["tokens_sha256"],
                "token_count": incumbent["output"]["token_count"],
            },
        },
        "incumbent_artifact": incumbent,
        "checkpoint_fingerprint": "a" * 64,
        "checkpoint_fingerprint_method": "sha256",
        "checkpoint_file_count": 1,
        "params_unchanged": True,
        "fast_weights_applied": False,
        "budget": {"spent_layer_apps": 64},
        "halting_reason": "fixed_point",
        "halting": {"stopped": True},
        "terminal_disposition": {"language": {"output_text_sha256": "6" * 64}},
        "decode_generated_tokens": 2,
        "decode_termination": "contract_complete",
    }
    valid_receipt["causal_receipt"] = build_causal_receipt(valid_receipt)
    assert valid_receipt["causal_receipt"]["missing_required_stages"] == []
    evidence = sweep._full_stack_evidence(valid_receipt)
    assert evidence["issues"] == []
    assert evidence["valid"] is True
    path, digest = sweep._persist_runtime_receipt(
        tmp_path,
        arm="full_stack",
        task_id="task-a",
        receipt=valid_receipt,
    )
    cell = {
        "runtime_receipt_path": path,
        "runtime_receipt_sha256": digest,
        "full_stack_evidence": evidence,
    }
    assert sweep._runtime_receipt_issues(tmp_path, cell) == []

    receipt_path = tmp_path / path
    tampered = dict(valid_receipt)
    tampered["n_slots"] = 8
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert sweep._runtime_receipt_issues(tmp_path, cell) == [
        "runtime_receipt_digest_mismatch",
        "runtime_receipt_summary_mismatch",
    ]

    invalid = dict(valid_receipt)
    invalid.pop("local_repair")
    invalid["fast_weight_learning"] = {"disposition": "rejected_verifier_unavailable"}
    invalid["causal_receipt"] = build_causal_receipt(invalid)
    evidence = sweep._full_stack_evidence(invalid)
    assert evidence["valid"] is False
    assert evidence["issues"] == [
        "fast_weight_verifier_unavailable",
        "local_repair_policy_not_measured",
    ]

    ablated = json.loads(json.dumps(valid_receipt))
    ablated.update(
        {
            "latent_opt_mode": "",
            "latent_opt_applied": False,
            "latent_opt_attempts": 0,
            "latent_opt_steps": 0,
            "fast_weight_learning": {},
            "fast_weights_applied": False,
            "fast_weight_optimization_attempts": 0,
        }
    )
    ablated["causal_receipt"] = build_causal_receipt(ablated)
    ablation_evidence = sweep._full_stack_evidence(
        ablated,
        adaptive_neural_expected=False,
    )
    assert ablation_evidence["valid"] is True
    assert ablation_evidence["issues"] == []
    assert ablation_evidence["adaptive_neural_expected"] is False

    contaminated = json.loads(json.dumps(ablated))
    contaminated["latent_opt_attempts"] = 1
    contaminated["causal_receipt"] = build_causal_receipt(contaminated)
    contaminated_evidence = sweep._full_stack_evidence(
        contaminated,
        adaptive_neural_expected=False,
    )
    assert contaminated_evidence["valid"] is False
    assert contaminated_evidence["issues"] == [
        "latent_optimization_ablation_contaminated"
    ]


def test_complete_system_receipt_requires_acquisition_amplifier_and_promotion(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        sweep,
        "_full_stack_evidence",
        lambda receipt, **_kwargs: {"valid": True, "issues": []},
    )
    objective = "Return a JSON object with value 1."
    response_contract = '{"value":int}'
    candidate_text = 'FINAL_ANSWER: {"value":1}'
    from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier

    candidate_evaluation = EpisodeTaskVerifier(
        objective,
        response_contract=response_contract,
    ).evaluate(candidate_text, _record=False)
    from tools.rlc_complete_system_closed_book import (
        _candidate_quality_assessment,
        _select_complete_system_promotion_candidate,
    )

    candidate_quality = _candidate_quality_assessment(candidate_evaluation)
    _, candidate_selection = _select_complete_system_promotion_candidate(
        verifier=EpisodeTaskVerifier(
            objective,
            response_contract=response_contract,
        ),
        rlc_text=candidate_text,
        amplifier_text=candidate_text,
        amplifier_consensus_programs=0,
        amplifier_consensus_strategies=0,
    )
    from core.brain.llm.latent_cortex.resource_accounting import (
        ModelComputeProfile,
        ResourceLedger,
        build_information_receipt,
    )

    ledger = ResourceLedger(
        ModelComputeProfile(
            model_type="fixture",
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            head_dim=4,
        )
    )
    ledger.charge(
        "fixture_complete_system",
        transformer_layer_apps=10,
        attention_query_key_pairs=20,
        output_head_tokens=2,
        verifier_calls=4,
        verifier_input_bytes=32,
        verifier_output_bytes=16,
    )
    information = build_information_receipt(
        sources=[
            {
                "source_id": "public_prompt",
                "kind": "prompt",
                "content_sha256": hashlib.sha256(objective.encode()).hexdigest(),
                "byte_count": len(objective.encode()),
                "token_count": 8,
            }
        ],
        policies={"closed_book": hashlib.sha256(b"closed_book").hexdigest()},
    )
    receipt = {
        "complete_system_closed_book": {
            "schema": "aura.rlc.complete_system_closed_book.v1",
            "contract": "same_information_no_memory_rag_web_or_answer_key",
            "task_id": "fixture-task",
            "objective": objective,
            "objective_sha256": hashlib.sha256(objective.encode()).hexdigest(),
            "response_contract": response_contract,
            "single_model_owner": True,
            "adaptive_neural_enabled": True,
            "executable_reasoning_enabled": True,
            "first_rlc_runtime": {"valid": True, "issues": []},
            "first_rlc_receipt": None,
            "rlc_rounds": 1,
            "cognitive_acquisition": {
                "status": "not_requested",
                "continuation_executed": False,
                "closed_book_external_sources_withheld": True,
            },
            "reasoning_amplifier": {
                "mode": "normal",
                "strategy_used": "self_consistency",
                "num_candidates": 3,
            },
            "amplifier_verified": candidate_quality["proxy_admitted"],
            "amplifier_candidate": {
                "schema": "aura.rlc.closed_book_amplifier_candidate.v1",
                "text": candidate_text,
                "text_sha256": hashlib.sha256(candidate_text.encode()).hexdigest(),
                "evaluation": candidate_evaluation,
                "quality_assessment": candidate_quality,
            },
            "integrated_candidates": [],
            "promotion_candidate_selection": candidate_selection,
            "amplifier_verifier_calls": 4,
            "in_process_generation_calls": 1,
            "seed_candidate_count": 2,
            "resource_accounting": ledger.to_receipt(),
            "information_accounting": information,
            "phase_latency_s": {
                "first_rlc": 3.0,
                "amplifier_generation": 2.0,
                "reasoning_amplifier": 4.0,
                "promotion": 0.1,
                "total": 7.2,
            },
            "promotion": {
                "schema": "aura.rlc.closed_book_promotion.v1",
                "decision": "replace",
                "answer_key_used": False,
                "authority": "candidate_quality_proxy_not_ground_truth",
                "ground_truth_verified": False,
                "no_regression_guaranteed": False,
                "candidate_text_sha256": hashlib.sha256(candidate_text.encode()).hexdigest(),
                "final_text_sha256": hashlib.sha256(candidate_text.encode()).hexdigest(),
            },
        }
    }
    evidence = sweep._complete_system_evidence(receipt)
    assert evidence["valid"] is True
    assert evidence["issues"] == []
    assert evidence["amplifier_candidates"] == 3
    assert evidence["amplifier_ground_truth_verified"] is False
    assert evidence["no_regression_guaranteed"] is False
    assert evidence["estimated_flops"] == ledger.estimated_flops()
    assert evidence["resource_accounting_sha256"] == ledger.to_receipt()["receipt_sha256"]
    assert evidence["information_accounting_sha256"] == information["receipt_sha256"]
    assert evidence["phase_latency_s"]["total"] == 7.2

    missing_latency = json.loads(json.dumps(receipt))
    del missing_latency["complete_system_closed_book"]["phase_latency_s"]
    assert "complete_system_phase_latency_absent" in sweep._complete_system_evidence(
        missing_latency
    )["issues"]


    from tools import rlc_reconciliation_evidence
    from tools.rlc_complete_system_closed_book import (
        _contextual_continuation_objective,
    )

    monkeypatch.setattr(
        rlc_reconciliation_evidence,
        "full_stack_evidence",
        lambda candidate, **_kwargs: {"valid": True, "issues": []},
    )
    from core.brain.llm.latent_cortex.cognitive_acquisition import (
        build_acquisition_receipt,
    )
    from core.brain.llm.latent_cortex.epistemic_state import canonical_sha256

    request_body = {
        "schema": "aura.rlc.cognitive_acquisition.v1",
        "action": "formalize",
        "action_step": 0,
        "objective_sha256": hashlib.sha256(objective.encode()).hexdigest(),
        "first_answer_sha256": hashlib.sha256(candidate_text.encode()).hexdigest(),
        "transition_sha256": "a" * 64,
        "retrieval_query": objective,
        "retrieval_query_sha256": hashlib.sha256(objective.encode()).hexdigest(),
        "before_inventory": [],
        "before_inventory_sha256": canonical_sha256(()),
        "max_acquisitions": 1,
        "max_continuation_rounds": 1,
        "worker_performed_io": False,
    }
    request = {**request_body, "request_sha256": canonical_sha256(request_body)}
    compute_body = {
        "schema": "aura.rlc.compute_acquisition.v1",
        "action": "formalize",
        "objective_sha256": hashlib.sha256(objective.encode()).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_text.encode()).hexdigest(),
        "task_type": "general",
        "status": "measured",
        "verifier": {},
        "sandbox": None,
        "guard": {},
        "admitted": True,
    }
    compute_receipt = {
        **compute_body,
        "receipt_sha256": canonical_sha256(compute_body),
    }
    context_text = "Verifier result: ok=true."
    context = {
        "source": "capability.symbolic_formalize",
        "text": context_text,
        "context_role": "evidence_observation",
        "instruction_authority": False,
        "evidence_id": "evidence-" + "c" * 24,
        "content_sha256": hashlib.sha256(context_text.encode()).hexdigest(),
        "retrieval_receipt_sha256": compute_receipt["receipt_sha256"],
        "evidence_kind": "governed_tool_observation",
        "evidence_origin": "core.brain.cortex_compute_acquisition",
        "source_version": "aura.rlc.compute_acquisition.v1",
    }
    continuation_objective = _contextual_continuation_objective(
        objective,
        [context],
    )
    ingress_receipt = {
        "schema": "aura.rlc.compute_ingress.v1",
        "compute": compute_receipt,
        "absent_sources": [],
    }
    acquisition_receipt = build_acquisition_receipt(
        request,
        acquired_context=[context],
        ingress_receipt=ingress_receipt,
        elapsed_s=0.1,
    )
    contextual_information = build_information_receipt(
        sources=[
            {
                "source_id": "rendered_model_input",
                "kind": "model_input_tokens",
                "content_sha256": "e" * 64,
                "byte_count": 10,
                "token_count": 3,
            },
            {
                "source_id": "value_controller_evidence",
                "kind": "controller_evidence",
                "content_sha256": "f" * 64,
                "byte_count": 10,
                "token_count": 0,
            },
        ],
        policies={"closed_book": hashlib.sha256(b"closed_book").hexdigest()},
    )
    contextual_receipt = json.loads(json.dumps(receipt))
    contextual_system = contextual_receipt["complete_system_closed_book"]
    contextual_system["first_rlc_receipt"] = {}
    contextual_system["rlc_rounds"] = 2
    contextual_system["information_accounting"] = contextual_information
    contextual_system["phase_latency_s"]["cognitive_acquisition"] = 0.1
    contextual_system["phase_latency_s"]["continuation_rlc"] = 1.0
    contextual_system["phase_latency_s"]["total"] = 8.3
    contextual_system["cognitive_acquisition"] = {
        "status": "completed_new_context",
        "request": request,
        "receipt": acquisition_receipt,
        "compute": compute_receipt,
        "ingress_receipt": ingress_receipt,
        "continuation_executed": True,
        "closed_book_external_sources_withheld": True,
        "timeout_s": 12.0,
        "input_candidate": candidate_text,
        "input_candidate_sha256": hashlib.sha256(candidate_text.encode()).hexdigest(),
        "acquired_context": [context],
        "continuation_objective": continuation_objective,
        "continuation_objective_sha256": hashlib.sha256(
            continuation_objective.encode()
        ).hexdigest(),
    }
    contextual_evidence = sweep._complete_system_evidence(contextual_receipt)
    assert contextual_evidence["valid"] is True
    assert contextual_evidence["issues"] == []
    contextual_system["cognitive_acquisition"]["continuation_objective_sha256"] = "0" * 64
    assert (
        "cognitive_continuation_objective_digest_mismatch"
        in (sweep._complete_system_evidence(contextual_receipt)["issues"])
    )

    path, digest = sweep._persist_runtime_receipt(
        tmp_path,
        arm="complete_system_closed_book",
        task_id="task-a",
        receipt=receipt,
    )
    cell = {
        "runtime_receipt_path": path,
        "runtime_receipt_sha256": digest,
        "full_stack_evidence": {"valid": True, "issues": []},
        "complete_system_evidence": evidence,
        "text": candidate_text,
    }
    assert sweep._runtime_receipt_issues(tmp_path, cell) == []
    cell["text"] = 'FINAL_ANSWER: {"value":2}'
    assert sweep._runtime_receipt_issues(tmp_path, cell) == ["complete_system_final_text_mismatch"]

    del receipt["complete_system_closed_book"]["reasoning_amplifier"]
    invalid = sweep._complete_system_evidence(receipt)
    assert invalid["valid"] is False
    assert "amplifier_candidates_absent" in invalid["issues"]
    assert "amplifier_strategy_not_executed" in invalid["issues"]

    del receipt["complete_system_closed_book"]["resource_accounting"]
    invalid = sweep._complete_system_evidence(receipt)
    assert "complete_system_resource_accounting_invalid" in invalid["issues"]
    assert "complete_system_resource_accounting_incomplete" in invalid["issues"]


def test_complete_system_admits_a_bound_recurrent_candidate_without_answer_key():
    from core.brain.llm.latent_cortex.frontier_tasks import generate_task_battery
    from core.brain.llm.latent_cortex.resource_accounting import (
        ModelComputeProfile,
        ResourceLedger,
    )
    from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier
    from tools.rlc_complete_system_closed_book import (
        _normalize_integrated_candidates,
        _select_complete_system_promotion_candidate,
        build_integrated_candidate,
    )
    from tools.rlc_integrated_recurrent_producer import (
        SOURCE_NAME,
        TRAINED_DEPTH_ONE_SOURCE,
    )

    task = generate_task_battery(
        [131973299],
        domains=["novel_algorithms"],
        difficulty=2,
    )[0]
    correct = (
        'FINAL_ANSWER: {"sequence":[58,51,30,22,85,87,94],'
        '"checksum":1943}'
    )
    ledger = ResourceLedger(
        ModelComputeProfile(
            model_type="fixture",
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            head_dim=4,
        )
    )
    ledger.charge(
        "unified_recurrent_decode",
        transformer_layer_apps=10,
        attention_query_key_pairs=20,
        output_head_tokens=2,
    )
    source_body = {
        "schema": "aura.rlc.integrated_recurrent_producer.v1",
        "source": TRAINED_DEPTH_ONE_SOURCE,
        "task_id": task.task_id,
        "text_sha256": hashlib.sha256(correct.encode()).hexdigest(),
        "resource_accounting_sha256": ledger.to_receipt()["receipt_sha256"],
        "same_public_information": True,
        "answer_key_used": False,
    }
    source_receipt = {
        **source_body,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                source_body,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest(),
    }
    integrated = build_integrated_candidate(
        source=TRAINED_DEPTH_ONE_SOURCE,
        task_id=task.task_id,
        text=correct,
        resource_accounting=ledger.to_receipt(),
        source_receipt=source_receipt,
        source_receipt_sha256=source_receipt["receipt_sha256"],
    )
    wrong = 'FINAL_ANSWER: {"sequence":[],"checksum":0}'
    deep_source_body = {
        **source_body,
        "source": SOURCE_NAME,
        "text_sha256": hashlib.sha256(wrong.encode()).hexdigest(),
    }
    deep_source_receipt = {
        **deep_source_body,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                deep_source_body,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest(),
    }
    deep_integrated = build_integrated_candidate(
        source=SOURCE_NAME,
        task_id=task.task_id,
        text=wrong,
        resource_accounting=ledger.to_receipt(),
        source_receipt=deep_source_receipt,
        source_receipt_sha256=deep_source_receipt["receipt_sha256"],
    )
    assert _normalize_integrated_candidates(
        [integrated, deep_integrated],
        task_id=task.task_id,
    ) == [integrated, deep_integrated]

    selected, receipt = _select_complete_system_promotion_candidate(
        verifier=EpisodeTaskVerifier(
            task.public.prompt,
            response_contract=task.public.response_contract,
        ),
        rlc_text='FINAL_ANSWER: {"sequence":[],"checksum":0}',
        amplifier_text='FINAL_ANSWER: {"sequence":[],"checksum":0}',
        amplifier_consensus_programs=0,
        amplifier_consensus_strategies=0,
        integrated_candidates=[integrated, deep_integrated],
    )
    assert selected == correct
    assert receipt["selected_source"] == TRAINED_DEPTH_ONE_SOURCE
    assert receipt["authority"] == "public_objective_deterministic_execution"
    assert [row["source"] for row in receipt["candidates"]] == [
        "rlc_final",
        TRAINED_DEPTH_ONE_SOURCE,
        SOURCE_NAME,
        "reasoning_amplifier",
    ]

    tampered = json.loads(json.dumps(integrated))
    tampered["text"] = 'FINAL_ANSWER: {"sequence":[],"checksum":0}'
    with pytest.raises(ValueError, match="receipt differs"):
        _normalize_integrated_candidates([tampered], task_id=task.task_id)
    with pytest.raises(ValueError, match="task identity differs"):
        _normalize_integrated_candidates([integrated], task_id="different-task")
    with pytest.raises(ValueError, match="source is duplicated"):
        _normalize_integrated_candidates([integrated, integrated], task_id=task.task_id)
    for reserved in ("rlc_final", "reasoning_amplifier"):
        with pytest.raises(ValueError, match="source is invalid"):
            build_integrated_candidate(
                source=reserved,
                task_id=task.task_id,
                text=correct,
                resource_accounting=ledger.to_receipt(),
                source_receipt={},
                source_receipt_sha256="a" * 64,
            )


def test_complete_system_replaces_only_the_bound_incumbent_resource_placeholder():
    from core.brain.llm.latent_cortex.resource_accounting import (
        ModelComputeProfile,
        ResourceLedger,
    )
    from tools.rlc_complete_system_closed_book import (
        _aggregate_complete_system_resources,
    )

    profile = ModelComputeProfile(
        model_type="fixture",
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        head_dim=4,
    )
    incumbent = ResourceLedger(profile)
    incumbent.charge("vanilla", transformer_layer_apps=7)
    rlc = ResourceLedger(profile)
    rlc.charge("recurrence", transformer_layer_apps=11)
    rlc.mark_unknown("bound_incumbent_generation")
    amplifier = ResourceLedger(profile)
    amplifier.charge("decode", transformer_layer_apps=13)
    integrated = ResourceLedger(profile)
    integrated.charge("unified_recurrent_decode", transformer_layer_apps=17)

    combined = _aggregate_complete_system_resources(
        incumbent_resource=incumbent.to_receipt(),
        rlc_resources=[rlc.to_receipt()],
        amplifier_resources=[amplifier.to_receipt()],
        integrated_resources=[integrated.to_receipt()],
    ).to_receipt()
    assert combined["accounting_complete"] is True
    assert combined["unknown_operations"] == []
    assert combined["totals"]["transformer_layer_apps"] == 48

    rlc_without_placeholder = ResourceLedger(profile)
    rlc_without_placeholder.charge("recurrence", transformer_layer_apps=11)
    with pytest.raises(ValueError, match="lacks a bound incumbent generation"):
        _aggregate_complete_system_resources(
            incumbent_resource=incumbent.to_receipt(),
            rlc_resources=[rlc_without_placeholder.to_receipt()],
            amplifier_resources=[amplifier.to_receipt()],
        )


def test_executable_accounting_requires_a_real_sandbox_invocation_and_counts_bytes():
    from core.brain.llm.latent_cortex.resource_accounting import (
        ModelComputeProfile,
        ResourceLedger,
    )
    from tools.rlc_complete_system_closed_book import (
        _charge_executable_operations,
    )

    ledger = ResourceLedger(
        ModelComputeProfile(
            model_type="fixture",
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            head_dim=4,
        )
    )
    recorded = _charge_executable_operations(
        ledger,
        [
            {
                "schema": "aura.executable_reasoning.v1",
                "status": "not_applicable",
                "program_chars": 0,
                "program_bytes": 0,
            },
            {
                "schema": "aura.executable_reasoning.v1",
                "status": "deadline_exhausted",
                "program_chars": 0,
                "program_bytes": 0,
            },
            {
                "schema": "aura.executable_reasoning.v1",
                "status": "sandbox_execution_failed",
                "program_chars": 1,
                "program_bytes": len("\u03c0".encode("utf-8")),
                "program_sha256": "a" * 64,
                "candidate_sha256": "",
                "strategy": "direct",
                "sandbox": {
                    "ok": False,
                    "stdout_total_bytes": 2,
                    "stderr_total_bytes": 3,
                    "isolation": {"network_denied": True},
                },
            },
        ],
        prefix="fixture",
    )

    receipt = ledger.to_receipt()
    assert receipt["totals"]["tool_calls"] == 1
    assert receipt["totals"]["tool_input_bytes"] == 2
    assert receipt["totals"]["tool_result_bytes"] == 5
    assert len(recorded) == 1
    assert recorded[0]["program_bytes"] == 2


def test_executable_accounting_preserves_every_sandbox_attempt():
    from core.brain.llm.latent_cortex.resource_accounting import (
        ModelComputeProfile,
        ResourceLedger,
    )
    from tools.rlc_complete_system_closed_book import (
        _charge_executable_operations,
    )

    ledger = ResourceLedger(
        ModelComputeProfile(
            model_type="fixture",
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            head_dim=4,
        )
    )
    recorded = _charge_executable_operations(
        ledger,
        [
            {
                "schema": "aura.executable_reasoning.v1",
                "status": "candidate_ready",
                "strategy": "direct",
                "candidate_sha256": "c" * 64,
                "attempts": [
                    {
                        "attempt": 1,
                        "status": "refused",
                        "program_sha256": "a" * 64,
                        "program_bytes": 9,
                        "sandbox": {
                            "ok": False,
                            "refused": True,
                            "timed_out": False,
                            "stdout_total_bytes": 0,
                            "stderr_total_bytes": 0,
                            "isolation": {"isolation_level": "unavailable"},
                        },
                    },
                    {
                        "attempt": 2,
                        "status": "executed",
                        "program_sha256": "b" * 64,
                        "program_bytes": 11,
                        "sandbox": {
                            "ok": True,
                            "refused": False,
                            "timed_out": False,
                            "stdout_total_bytes": 7,
                            "stderr_total_bytes": 0,
                            "isolation": {
                                "isolation_level": "kernel:seatbelt",
                                "network_denied": True,
                            },
                        },
                    },
                ],
            }
        ],
        prefix="fixture",
    )

    assert ledger.to_receipt()["totals"]["tool_calls"] == 2
    assert [row["status"] for row in recorded] == ["refused", "executed"]
    assert recorded[0]["process_launched"] is False
    assert recorded[1]["network_denied"] is True


def test_candidate_quality_separates_proxy_admission_from_exact_public_proof():
    from core.brain.llm.latent_cortex.task_verifiers import EpisodeTaskVerifier
    from tools.rlc_complete_system_closed_book import _candidate_quality_assessment

    proxy_objective = "Return a JSON object with value 1."
    proxy = EpisodeTaskVerifier(
        proxy_objective,
        response_contract='{"value":int}',
    ).evaluate('FINAL_ANSWER: {"value":1}', _record=False)
    proxy_assessment = _candidate_quality_assessment(proxy)
    assert proxy_assessment["proxy_admitted"] is True
    assert proxy_assessment["ground_truth_verified"] is False

    exact_objective = (
        "Start at the given value and apply each operation modulo 19: start=5. Operations: +3, *2."
    )
    exact = EpisodeTaskVerifier(
        exact_objective,
        response_contract='{"residue":int}',
    ).evaluate('FINAL_ANSWER: {"residue":16}', _record=False)
    exact_assessment = _candidate_quality_assessment(exact)
    assert exact_assessment["proxy_admitted"] is True
    assert exact_assessment["ground_truth_verified"] is True


def test_oracle_diagnostic_is_admitted_but_only_answer_keys_task_outputs():
    from core.brain.llm.latent_cortex import frontier_tasks as ft
    from core.brain.llm.latent_cortex.blind_review import run_decoy_preflight

    task = ft.generate_task_battery([202608081], difficulty=2)[0]
    verifier = sweep._oracle_verifier(task)
    preflight = run_decoy_preflight(
        verifier,
        episode_id="oracle-admission-test",
        objective_sha256="a" * 64,
    )
    assert preflight["verifier_admitted"] is True

    expected = json.dumps(task.reveal_for_verifier()["expected"])
    assert verifier("FINAL_ANSWER: " + expected) == 1.0
    assert verifier('FINAL_ANSWER: {"definitely": "wrong"}') == 0.0


def test_unpromoted_full_stack_output_must_be_byte_identical_to_incumbent(tmp_path: Path):
    from core.brain.llm.latent_cortex import frontier_tasks as ft

    task = ft.generate_task_battery([20260807], difficulty=2)[0]
    expected = json.dumps(task.reveal_for_verifier()["expected"])
    journal = sweep.Journal(tmp_path / "journal.jsonl")
    journal.append(
        {
            "event": "CELL",
            "arm": "vanilla",
            "task_id": task.task_id,
            "domain": task.domain,
            "text": "FINAL_ANSWER: " + expected,
            "error": "",
        }
    )
    journal.append(
        {
            "event": "CELL",
            "arm": "full_stack",
            "task_id": task.task_id,
            "domain": task.domain,
            "text": "Reasoning complete.\nFINAL_ANSWER: " + expected,
            "error": "",
            "answer_replacement_decision": "retain",
        }
    )

    verdict = sweep.grade(tmp_path, [task])
    assert verdict["paired_vanilla_floor"]["holds"] is False
    assert verdict["paired_vanilla_floor"]["right_to_wrong_regressions"] == []
    assert verdict["paired_vanilla_floor"]["unpromoted_byte_divergences"]
    assert verdict["decision"] == "invalid_full_stack_violated_vanilla_incumbent"


def test_the_operator_can_reclaim_the_machine_between_cells(tmp_path: Path):
    """The host cannot hold two 32B models, so the campaign and the live
    instance are exclusive. The campaign must therefore be able to leave on
    request rather than requiring long contiguous blocks."""
    assert sweep.yield_requested(tmp_path) is False
    (tmp_path / sweep.YIELD_SENTINEL).touch()
    assert sweep.yield_requested(tmp_path) is True
    # Committed work survives a yield: identity travels with the cell, so a
    # resumed run re-admits exactly the cells it already paid for.
    (tmp_path / sweep.YIELD_SENTINEL).unlink()
    assert sweep.yield_requested(tmp_path) is False


def test_the_product_arm_keeps_ordinary_decode_as_the_incumbent():
    """The stack scored HALF of plain greedy decode because the product arm ran
    decode_incumbent_policy="latent": the recurrent path owned the answer
    unconditionally, so ordinary decode's answer was never a candidate. Adding
    verifiers to that cannot help -- selection cannot exceed the best candidate
    in the pool, and the good one was not in it.

    The deployed system runs "vanilla_incumbent": every subsystem still
    executes and is receipted, the answer decodes from the clean prompt root,
    and a latent answer takes over only when a gain gate promotes it. That is
    monotonic by construction."""
    full = sweep._build_config(8, 16, "applied", 512, profile="full")
    assert full.decode_incumbent_policy == "vanilla_incumbent"
    # The acceptance rule is what lets a latent answer win at all.
    assert full.answer_replacement_enabled is True

    # The ablation deliberately keeps the latent path owning the answer, so a
    # degraded episode cannot silently serve vanilla and read as a result.
    mech = sweep._build_config(4, 16, "suppressed", 512, profile="mechanism")
    assert mech.decode_incumbent_policy == "latent"
    assert mech.answer_replacement_enabled is False
