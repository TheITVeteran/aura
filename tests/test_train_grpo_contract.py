"""Scientific and persistence contracts for the GRPO trainer."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import tools.train_grpo as train_grpo
from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.grpo import GRPOConfig, GRPOTelemetry, group_advantages
from core.learning.grpo_training_state import (
    GRPOCheckpointError,
    canonical_json_bytes,
    sha256_bytes,
)
from core.learning.recurrent_grpo_artifact_schema import (
    recurrent_training_adequacy_report,
)
from core.learning.verified_transition_trainer import (
    VerifiedTransitionTrainingScheduleEntry,
)
from tools.train_grpo import (
    GRPO_DATASET_SCHEMA,
    _advantage_report_with_verifier_rate,
    _answer_channel_report_from_verdicts,
    _answer_contract_instruction,
    _assert_exact_adapter_keys,
    _build_task_split,
    _calibration_admission_report,
    _calibration_reward_observations,
    _calibration_token_budget,
    _dataset_payload,
    _load_execution_spec,
    _point_estimate_delta,
    _publish_adapter_snapshot,
    _publish_immutable_bytes,
    _publish_immutable_tensor_snapshot,
    _record_recurrent_step_failure,
    _render,
    _resolve_model_path,
    _scheduled_verified_training_task,
    _shape_recurrent_rewards_from_ce_trails,
    _should_halt_for_no_learning_signal,
    _signal_admission_report,
    _stable_seed,
    _task_gold_answer_text,
    _training_source_files,
    _verified_campaign_halt_is_resumable,
    completion_logprob,
    evaluate_heldout,
    evaluate_recurrent_heldout,
    sample_recurrent_group,
)
from tools.train_grpo import (
    main as train_grpo_main,
)


@dataclass(frozen=True)
class _Task:
    task_id: str
    prompt: str = "prompt"
    domain: str = "logic"
    depth: int = 4
    knowledge: str = "parametric"
    grader: str = "boolean"
    expected: bool = True
    metadata: dict = field(default_factory=dict)


def test_stable_seed_has_a_fixed_process_independent_value():
    assert _stable_seed(7, "cell", 4) == 3478236081
    assert _stable_seed(7, "cell", 4) != _stable_seed(7, "cell", 5)


def test_model_path_resolves_from_authenticated_main_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    main = tmp_path / "main"
    worktree = tmp_path / "worktree"
    gitdir = main / ".git" / "worktrees" / "spark"
    model = main / "training" / "fused-model" / "resident"
    gitdir.mkdir(parents=True)
    model.mkdir(parents=True)
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {gitdir}\n", encoding="ascii")
    monkeypatch.setattr(train_grpo, "REPO_ROOT", worktree)
    monkeypatch.chdir(worktree)

    assert _resolve_model_path("training/fused-model/resident") == model.resolve()


def test_recurrent_source_inventory_matches_identity_contract() -> None:
    from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
        REQUIRED_SOURCE_ROLES,
    )

    source_files = _training_source_files(
        train_grpo.REPO_ROOT / "core/learning/recurrence_curriculum.py",
        recurrent=True,
    )

    assert set(source_files) == REQUIRED_SOURCE_ROLES
    assert all(path.is_file() for path in source_files.values())


def test_pre_stage_recovery_runs_after_exact_restore_and_before_training_loop():
    source = inspect.getsource(train_grpo_main)
    restore = source.index("resumed = load_grpo_checkpoint(")
    reconcile = source.index("measurement_chain_store.reconcile_interrupted_admissions(")
    recovery_gate = source.index(
        "if pre_stage_recovery_halt is not None:\n            training_allowed = False",
        reconcile,
    )
    mutation_loop = source.index(
        "while training_allowed and step < args.max_steps:",
        recovery_gate,
    )

    assert restore < reconcile < recovery_gate < mutation_loop


def test_dataset_identity_binds_split_order_and_task_bytes():
    first = _Task("first")
    second = _Task("second", prompt="different")
    payload = _dataset_payload([first], [second], seed=11)

    assert payload["schema"] == GRPO_DATASET_SCHEMA
    digest = sha256_bytes(canonical_json_bytes(payload))
    swapped = _dataset_payload([second], [first], seed=11)
    assert sha256_bytes(canonical_json_bytes(swapped)) != digest
    assert _dataset_payload([first], [second], seed=11) == payload


def test_recurrent_training_adequacy_requires_full_distributed_real_updates():
    tasks = [f"task-{index}" for index in range(8)]
    receipts = [
        {
            "step": index + 1,
            "task_id": task_id,
            "step_kind": (
                "verified_optimizer_update" if index in {0, 4} else "verified_rejected_group"
            ),
            "policy_after_sha256": hashlib.sha256(task_id.encode("ascii")).hexdigest(),
        }
        for index, task_id in enumerate(tasks)
    ]

    report = recurrent_training_adequacy_report(
        step_receipts=receipts,
        scheduled_task_ids=tasks,
        max_steps=8,
        eval_every=4,
        evaluation_steps=[4, 8],
        learning_signal={"learning_signal": True},
    )

    assert report["admitted"] is True
    assert report["minimum_optimizer_updates"] == 2
    assert [window["optimizer_updates"] for window in report["update_windows"]] == [1, 1]


def test_recurrent_training_adequacy_rejects_end_loaded_or_incomplete_dose():
    tasks = [f"task-{index}" for index in range(8)]
    receipts = [
        {
            "step": index + 1,
            "task_id": task_id,
            "step_kind": (
                "verified_optimizer_update" if index in {6, 7} else "verified_rejected_group"
            ),
            "policy_after_sha256": hashlib.sha256(task_id.encode("ascii")).hexdigest(),
        }
        for index, task_id in enumerate(tasks[:-1])
    ]

    report = recurrent_training_adequacy_report(
        step_receipts=receipts,
        scheduled_task_ids=tasks,
        max_steps=8,
        eval_every=4,
        evaluation_steps=[8],
        learning_signal={"learning_signal": True},
    )

    assert report["admitted"] is False
    assert {
        "one_complete_pass",
        "exact_task_schedule",
        "minimum_optimizer_updates",
        "distributed_update_activity",
        "evaluation_schedule_complete",
    }.issubset(report["failed_checks"])


def test_verified_training_task_and_seed_come_only_from_provider_schedule():
    class Provider:
        @staticmethod
        def training_schedule_entry(*, sequence: int):
            return VerifiedTransitionTrainingScheduleEntry(
                campaign_sequence=sequence,
                task_id="second",
                trainer_sample_seed=991,
            )

    first = _Task("first")
    second = _Task("second")
    task, seed = _scheduled_verified_training_task(
        Provider(),
        {first.task_id: first, second.task_id: second},
        campaign_sequence=3,
    )

    assert task is second
    assert seed == 991


def test_verified_training_rejects_schedule_sequence_or_task_substitution():
    class WrongSequence:
        @staticmethod
        def training_schedule_entry(*, sequence: int):
            return VerifiedTransitionTrainingScheduleEntry(
                campaign_sequence=sequence + 1,
                task_id="task",
                trainer_sample_seed=1,
            )

    class UnknownTask:
        @staticmethod
        def training_schedule_entry(*, sequence: int):
            return VerifiedTransitionTrainingScheduleEntry(
                campaign_sequence=sequence,
                task_id="foreign",
                trainer_sample_seed=1,
            )

    with pytest.raises(RuntimeError, match="different schedule sequence"):
        _scheduled_verified_training_task(
            WrongSequence(), {"task": _Task("task")}, campaign_sequence=0
        )
    with pytest.raises(RuntimeError, match="outside the frozen dataset"):
        _scheduled_verified_training_task(
            UnknownTask(), {"task": _Task("task")}, campaign_sequence=0
        )


def test_grpo_can_bind_the_broad_recurrence_training_registry():
    train, holdout, source = _build_task_split(
        task_source="recurrence_curriculum",
        domains=["khop", "code_trace"],
        depths=[2, 4],
        train_per_cell=2,
        holdout_per_cell=1,
        seed=101,
    )

    assert len(train) == 8
    assert len(holdout) == 4
    assert source.name == "recurrence_curriculum.py"
    assert {task.metadata["source"] for task in train} == {"recurrence_curriculum"}
    assert {task.task_id for task in train}.isdisjoint({task.task_id for task in holdout})


def test_grpo_rejects_an_unbound_task_registry():
    with pytest.raises(ValueError, match="unsupported task source"):
        _build_task_split(
            task_source="frontier_evaluation",
            domains=["mathematics"],
            depths=[2],
            train_per_cell=1,
            holdout_per_cell=1,
            seed=1,
        )


def test_render_adds_contract_scaffold_without_answer_value_leakage():
    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            return messages[0]["content"]

    task = _Task(
        "contract",
        prompt="Find the node.",
        expected={"node": 9, "trace": [1, 2, 9]},
    )

    rendered = _render(Tokenizer(), task)

    assert rendered.endswith("Find the node.")
    assert "FINAL_ANSWER: {JSON object}" in rendered
    assert "node, trace" in rendered
    assert "9" not in rendered
    assert "[1, 2, 9]" not in rendered


def test_answer_contract_instruction_tolerates_non_mapping_expected():
    instruction = _answer_contract_instruction(_Task("plain", expected=True))

    assert "FINAL_ANSWER: {JSON object}" in instruction
    assert "Use exactly these JSON keys" not in instruction


def test_adapter_resume_requires_the_exact_trainable_keyset():
    expected = {"a.lora_a": object(), "a.lora_b": object()}
    _assert_exact_adapter_keys(expected, dict(expected))

    with pytest.raises(GRPOCheckpointError, match="keyset differs"):
        _assert_exact_adapter_keys(expected, {"a.lora_a": object()})
    with pytest.raises(GRPOCheckpointError, match="keyset differs"):
        _assert_exact_adapter_keys(expected, {**expected, "foreign": object()})


def test_point_estimate_delta_does_not_manufacture_missing_comparisons():
    assert _point_estimate_delta(None, {"overall": 0.7}) is None
    assert _point_estimate_delta({"overall": 0.5}, None) is None
    assert _point_estimate_delta({"overall": 0.5}, {"overall": 0.625}) == 0.125


def test_calibration_cannot_reintroduce_a_short_reasoning_budget():
    assert _calibration_token_budget(320, 0) == 320
    assert _calibration_token_budget(320, 320) == 320
    with pytest.raises(ValueError, match="must equal training"):
        _calibration_token_budget(320, 128)


def test_calibration_replay_preserves_group_outcome_counts():
    assert _calibration_reward_observations(
        {"completions": 4, "correct": 2}
    ) == (1.0, 1.0, 0.0, 0.0)
    with pytest.raises(GRPOCheckpointError, match="counts are invalid"):
        _calibration_reward_observations({"completions": 2, "correct": 3})


def test_bounded_halts_leave_verified_campaign_open_for_exact_resume():
    assert _verified_campaign_halt_is_resumable("wall_clock_budget")
    assert _verified_campaign_halt_is_resumable("interrupted")
    assert not _verified_campaign_halt_is_resumable("max_steps")
    assert not _verified_campaign_halt_is_resumable("training_adequacy_failed")


def test_calibration_progress_replays_exact_prefix_and_remaining_budget(
    tmp_path,
):
    path = tmp_path / "calibration-progress.json"
    identity = {
        "schema": "test.recurrent_calibration_identity.v1",
        "protocol_sha256": "a" * 64,
    }
    first = {
        "family": "logic",
        "difficulty": 2,
        "probe_index": 0,
        "task_id": "logic-2-0",
        "seed": 7,
        "pass_rate": 0.5,
        "answer_channel": _answer_channel_report_from_verdicts(
            [{"correct": True}, {"correct": False}]
        ),
        "elapsed_s": 1.25,
    }
    second = {
        **first,
        "family": "code",
        "task_id": "code-2-0",
        "seed": 11,
        "pass_rate": 1.0,
        "answer_channel": _answer_channel_report_from_verdicts(
            [{"correct": True}, {"correct": True}]
        ),
        "elapsed_s": 0.25,
    }

    initial = train_grpo._CalibrationProgressJournal.open(
        path,
        identity=identity,
        budget_s=10.0,
    )
    initial.append(first)

    resumed = train_grpo._CalibrationProgressJournal.open(
        path,
        identity=identity,
        budget_s=10.0,
    )
    assert resumed.saved_count == 1
    assert 0.0 < resumed.remaining_s <= 8.75
    with pytest.raises(GRPOCheckpointError, match="exact probe prefix"):
        resumed.replay(
            family="wrong",
            difficulty=2,
            probe_index=0,
            task_id="logic-2-0",
            seed=7,
        )
    assert resumed.replay(
        family="logic",
        difficulty=2,
        probe_index=0,
        task_id="logic-2-0",
        seed=7,
    ) == first
    resumed.append(second)
    resumed.finish()

    completed = train_grpo._CalibrationProgressJournal.open(
        path,
        identity=identity,
        budget_s=10.0,
    )
    assert completed.prior_complete is True
    assert completed.replay(
        family="logic",
        difficulty=2,
        probe_index=0,
        task_id="logic-2-0",
        seed=7,
    ) == first
    assert completed.replay(
        family="code",
        difficulty=2,
        probe_index=0,
        task_id="code-2-0",
        seed=11,
    ) == second
    assert completed.permits_new_probe() is False
    completed.finish()

    with pytest.raises(GRPOCheckpointError, match="identity differs"):
        train_grpo._CalibrationProgressJournal.open(
            path,
            identity={**identity, "protocol_sha256": "b" * 64},
            budget_s=10.0,
        )


def test_recurrent_calibration_requires_measured_learnable_signal():
    calibration = {
        "learnable": [],
        "unexplored": ["khop@2", "khop@4"],
        "partial": True,
        "probes": [{"pass_rate": 0.0}],
        "answer_channel": {
            "completions": 4,
            "parseable": 0,
            "unparseable": 4,
            "correct": 0,
            "parseable_fraction": 0.0,
        },
    }

    admitted = _calibration_admission_report(
        calibration,
        allow_unexplored_frontier=False,
    )

    assert admitted["training_admitted"] is False
    assert admitted["reason"] == "answer_channel_blocked"
    assert "before resident GRPO" in admitted["required_next_gate"]


def test_standard_calibration_can_keep_exploring_unmeasured_cells():
    calibration = {
        "learnable": [],
        "unexplored": ["logic@2"],
        "partial": True,
        "probes": [{"pass_rate": 0.0}],
        "answer_channel": {
            "completions": 4,
            "parseable": 4,
            "unparseable": 0,
            "correct": 0,
            "parseable_fraction": 1.0,
        },
    }

    admitted = _calibration_admission_report(
        calibration,
        allow_unexplored_frontier=True,
    )

    assert admitted["training_admitted"] is True
    assert admitted["reason"] == "unexplored_frontier_allowed"


def test_training_halts_when_grpo_has_no_learning_signal():
    telemetry = GRPOTelemetry()
    config = GRPOConfig(group_size=4, max_degenerate_fraction=0.5)
    for _ in range(3):
        telemetry.observe(group_advantages([0.0, 0.0, 0.0, 0.0]))

    assert _should_halt_for_no_learning_signal(telemetry, config, min_groups=4) is None

    telemetry.observe(group_advantages([0.0, 0.0, 0.0, 0.0]))
    verdict = _should_halt_for_no_learning_signal(telemetry, config, min_groups=4)

    assert verdict is not None
    assert verdict["learning_signal"] is False
    assert "too_hard" in verdict["diagnosis"]


def test_zero_group_signal_report_is_terminal_and_diagnostic():
    report = _signal_admission_report(
        GRPOTelemetry().verdict(GRPOConfig()),
        step_receipts=[],
    )

    assert report["learning_signal"] is False
    assert report["groups"] == 0
    assert "no_training_groups_observed" in report["diagnosis"]
    assert "calibration admission receipt" in report["required_next_gate"]


def test_no_signal_halt_reports_answer_channel_blocker():
    telemetry = GRPOTelemetry()
    config = GRPOConfig(group_size=4, max_degenerate_fraction=0.5)
    receipts = []
    for step in range(4):
        report = group_advantages([0.0, 0.0, 0.0, 0.0])
        telemetry.observe(report)
        receipts.append(
            {
                "step": step + 1,
                "answer_channel": {
                    "completions": 4,
                    "parseable": 0,
                    "unparseable": 4,
                    "correct": 0,
                    "parseable_fraction": 0.0,
                    "correct_fraction": 0.0,
                    "grade_reasons": {"unparseable": 4},
                },
                "advantage_report": report,
            }
        )

    verdict = _should_halt_for_no_learning_signal(
        telemetry,
        config,
        min_groups=4,
        step_receipts=receipts,
    )

    assert verdict is not None
    assert verdict["schema"] == "aura.grpo_signal_admission.v1"
    assert verdict["answer_channel"]["completions"] == 16
    assert verdict["answer_channel"]["parseable_fraction"] == 0.0
    assert "answer_channel_blocked" in verdict["diagnosis"]
    assert "decode contract" in verdict["required_next_gate"]


def test_answer_channel_report_counts_parseability_separately_from_correctness():
    report = _answer_channel_report_from_verdicts(
        [
            {"correct": False, "parsed": None, "reason": "unparseable"},
            {"correct": False, "parsed": {"node": 3}},
            {"correct": True, "parsed": {"node": 4}},
        ]
    )

    assert report["completions"] == 3
    assert report["parseable"] == 2
    assert report["correct"] == 1
    assert report["parseable_fraction"] == pytest.approx(0.6667)
    assert report["grade_reasons"] == {"correct": 1, "incorrect": 1, "unparseable": 1}


def test_recurrent_trajectory_credit_preserves_verifier_rewards():
    verifier = [0.0, 0.0, 0.0]
    shaped = _shape_recurrent_rewards_from_ce_trails(
        verifier,
        [
            [2.2, 1.1, 0.3],
            [1.5, 1.5, 1.5],
            [0.3, 1.1, 2.2],
        ],
        shaping_weight=0.25,
    )
    report = group_advantages(shaped["shaped_rewards"])

    assert report["degenerate"] is False
    assert shaped["rows"][0]["final_reward"] == 0.0
    assert shaped["rows"][0]["shaping"] > 0.0
    assert shaped["rows"][2]["shaping"] < 0.0
    assert len(shaped["ce_trails"]) == 3
    assert len(shaped["score_trails"]) == 3


def test_recurrent_trajectory_credit_keeps_telemetry_mean_as_verifier_rate():
    verifier_report = group_advantages([1.0, 1.0, 1.0, 1.0])
    shaped_report = group_advantages([1.12, 1.05, 0.98, 0.91])

    report = _advantage_report_with_verifier_rate(
        shaped_report,
        verifier_report,
    )
    telemetry = GRPOTelemetry()
    telemetry.observe(report)

    assert report["trajectory_shaped"] is True
    assert report["mean_reward"] == 1.0
    assert report["shaped_mean_reward"] > 1.0
    assert report["degenerate"] is False
    assert telemetry.state()["reward_sum"] == 1.0


def test_degenerate_trajectory_credit_does_not_masquerade_as_partial_reward():
    verifier_report = group_advantages([0.0, 0.0, 0.0, 0.0])
    shaped_report = group_advantages([0.03, 0.03, 0.03, 0.03])

    report = _advantage_report_with_verifier_rate(
        shaped_report,
        verifier_report,
    )
    telemetry = GRPOTelemetry()
    telemetry.observe(report)
    verdict = _signal_admission_report(
        telemetry.verdict(GRPOConfig(group_size=4, max_degenerate_fraction=0.5)),
        step_receipts=[
            {
                "answer_channel": {
                    "completions": 4,
                    "parseable": 4,
                    "unparseable": 0,
                    "correct": 0,
                    "parseable_fraction": 1.0,
                    "correct_fraction": 0.0,
                    "grade_reasons": {"incorrect": 4},
                },
                "advantage_report": report,
            }
        ],
    )

    assert report["mean_reward"] == 0.0
    assert report["shaped_mean_reward"] == 0.03
    assert report["all_wrong"] is True
    assert report["uniform_partial"] is False
    assert telemetry.state()["reward_sum"] == 0.0
    assert "trajectory_credit_constant" in verdict["diagnosis"]


def test_task_gold_answer_text_prefers_bound_answer_contract():
    class RecurrenceStyle:
        answer = 'FINAL_ANSWER: {"node":3}'

        @property
        def expected(self):
            raise AssertionError("answer text should be the training contract")

    class VerifiableStyle:
        answer = None
        expected = {"value": 1}

    assert _task_gold_answer_text(RecurrenceStyle()) == 'FINAL_ANSWER: {"node":3}'
    assert _task_gold_answer_text(VerifiableStyle()) == 'FINAL_ANSWER: {"value":1}'


def test_execution_mode_requires_and_strictly_loads_recurrent_spec(tmp_path):
    assert _load_execution_spec("standard", None) is None
    with pytest.raises(ValueError, match="only applies"):
        _load_execution_spec("standard", str(tmp_path / "unused.json"))
    with pytest.raises(ValueError, match="requires"):
        _load_execution_spec("recurrent", None)

    spec = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution", "critical_audit"),
        recurrent_steps=3,
    )
    path = tmp_path / "execution_spec.json"
    path.write_text(json.dumps(spec.to_dict()), encoding="ascii")

    loaded = _load_execution_spec("recurrent", str(path))
    assert loaded == spec
    assert loaded.sha256 == spec.sha256

    payload = spec.to_dict()
    payload["unknown"] = True
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError, match="unknown"):
        _load_execution_spec("recurrent", str(path))


def test_recurrent_failure_receipt_is_immutable_and_latest_is_bound(tmp_path):
    receipt = _record_recurrent_step_failure(
        tmp_path,
        protocol_sha256="a" * 64,
        dataset_sha256="b" * 64,
        execution_spec_sha256="c" * 64,
        attempted_step=3,
        last_durable_step=1,
        phase="exact_adjoint",
        task_id="logic-3",
        sample_seed=41,
        samples=(),
        error=RuntimeError("clip admission failed"),
    )

    payload = json.loads(receipt.read_text(encoding="ascii"))
    latest = json.loads((tmp_path / "latest_failure.json").read_text(encoding="ascii"))
    assert payload["attempted_step"] == 3
    assert payload["last_durable_step"] == 1
    assert payload["volatile_completed_steps"] == 1
    assert payload["error"]["type"] == "RuntimeError"
    assert latest["receipt"] == str(receipt.relative_to(tmp_path))
    assert latest["receipt_sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()


def test_recurrent_group_resamples_inadmissible_cached_behavior(monkeypatch):
    from core.learning import recurrent_grpo
    from core.learning.recurrent_grpo import RecurrentSamplingAdmissionError

    class Tokenizer:
        eos_token_id = None

        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            return messages[0]["content"]

        @staticmethod
        def encode(text, **_kwargs):
            return [1, 2, 3]

        @staticmethod
        def decode(tokens):
            return ",".join(str(token) for token in tokens)

    class FakeRejectedSample:
        max_abs_logprob_drift = 8.8
        mean_abs_logprob_drift = 0.03
        clipped_token_fraction = 0.01

        @staticmethod
        def receipt():
            return {"schema": "test.rejected"}

    class FakeAdmittedSample:
        behavior_admitted = True

        def __init__(self, seed):
            self.seed = seed
            self.tokens = (seed % 7, seed % 11)

    calls = []

    def fake_sample_completion(_model, _prompt, **kwargs):
        calls.append(kwargs["seed"])
        if len(calls) == 1:
            raise RecurrentSamplingAdmissionError(FakeRejectedSample())
        return FakeAdmittedSample(kwargs["seed"])

    monkeypatch.setattr(
        recurrent_grpo,
        "sample_recurrent_completion",
        fake_sample_completion,
    )

    _prompt, samples, completions = sample_recurrent_group(
        object(),
        Tokenizer(),
        _Task("resample", prompt="solve"),
        spec=object(),
        size=2,
        max_tokens=8,
        seed=17,
    )

    assert len(calls) == 3
    assert len(samples) == len(completions) == 2
    assert all(sample.behavior_admitted for sample in samples)


def test_recurrent_group_exhaustion_reports_rejected_receipts(monkeypatch):
    from core.learning import recurrent_grpo
    from core.learning.recurrent_grpo import RecurrentSamplingAdmissionError

    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            return messages[0]["content"]

        @staticmethod
        def encode(text, **_kwargs):
            return [1, 2, 3]

        @staticmethod
        def decode(tokens):
            return ",".join(str(token) for token in tokens)

    class FakeRejectedSample:
        max_abs_logprob_drift = 9.0
        mean_abs_logprob_drift = 0.04
        clipped_token_fraction = 0.02

        @staticmethod
        def receipt():
            return {"schema": "test.rejected", "reason": "ppo_drift"}

    def always_reject(*_args, **_kwargs):
        raise RecurrentSamplingAdmissionError(FakeRejectedSample())

    monkeypatch.setattr(recurrent_grpo, "sample_recurrent_completion", always_reject)

    with pytest.raises(RuntimeError, match="sampling exhausted") as captured:
        sample_recurrent_group(
            object(),
            Tokenizer(),
            _Task("exhaust", prompt="solve"),
            spec=object(),
            size=2,
            max_tokens=8,
            seed=17,
        )

    message = str(captured.value)
    assert "aura.recurrent_group_sampling_exhausted.v1" in message
    assert '"admitted":0' in message
    assert '"requested":2' in message


def test_adapter_snapshot_is_atomically_published_as_real_safetensors(tmp_path):
    mx = pytest.importorskip("mlx.core")
    target = tmp_path / "grpo_adapters.safetensors"

    _publish_adapter_snapshot(target, {"layer.lora_a": mx.array([1.0, 2.0])})

    loaded = mx.load(str(target))
    assert set(loaded) == {"layer.lora_a"}
    assert bool(mx.array_equal(loaded["layer.lora_a"], mx.array([1.0, 2.0])))
    assert not list(tmp_path.glob(".*.tmp.safetensors"))


def test_initial_adapter_snapshot_is_immutable_and_inspectable(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from core.learning.verified_transition_policy_probe import (
        inspect_initial_adapter_snapshot,
        inspect_initial_optimizer_snapshot,
    )

    target = tmp_path / "initial_adapter.safetensors"
    tensors = {
        "layer.lora_a": mx.array([[1.0, 2.0]]),
        "layer.lora_b": mx.array([[3.0], [4.0]]),
    }
    _publish_immutable_tensor_snapshot(
        target,
        tensors,
        role="initial recurrent adapter snapshot",
    )
    first_bytes = target.read_bytes()
    _publish_immutable_tensor_snapshot(
        target,
        tensors,
        role="initial recurrent adapter snapshot",
    )

    binding = inspect_initial_adapter_snapshot(
        target,
        execution_spec_sha256="a" * 64,
    )

    assert target.read_bytes() == first_bytes
    assert binding["path"] == target.name
    assert binding["tensor_count"] == 2
    assert binding["size_bytes"] == len(first_bytes)
    assert binding["policy_sha256"]

    optimizer_target = tmp_path / "initial_optimizer.safetensors"
    _publish_immutable_tensor_snapshot(
        optimizer_target,
        {
            "step": mx.array(0, dtype=mx.uint64),
            "layer.lora_a.m": mx.zeros((1, 2)),
            "layer.lora_a.v": mx.zeros((1, 2)),
        },
        role="initial recurrent optimizer snapshot",
    )
    optimizer_binding = inspect_initial_optimizer_snapshot(optimizer_target)
    assert optimizer_binding["path"] == optimizer_target.name
    assert optimizer_binding["tensor_count"] == 3


def test_run_metadata_is_immutable_and_idempotent(tmp_path):
    target = tmp_path / "protocol.json"
    _publish_immutable_bytes(target, b"frozen\n", role="protocol")
    _publish_immutable_bytes(target, b"frozen\n", role="protocol")

    with pytest.raises(GRPOCheckpointError, match="differs from the frozen run"):
        _publish_immutable_bytes(target, b"drift\n", role="protocol")
    assert target.read_bytes() == b"frozen\n"


def test_run_metadata_rejects_symlink_indirection(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside\n")
    target = tmp_path / "protocol.json"
    target.symlink_to(outside)

    with pytest.raises(GRPOCheckpointError, match="symlink is forbidden"):
        _publish_immutable_bytes(target, b"frozen\n", role="protocol")
    assert outside.read_bytes() == b"outside\n"


def test_empty_text_completion_uses_eos_for_policy_credit():
    mx = pytest.importorskip("mlx.core")

    class Tokenizer:
        eos_token_id = 2

        @staticmethod
        def encode(text, add_special_tokens=True):
            return [1] if text else []

    class Model:
        @staticmethod
        def __call__(tokens):
            return mx.zeros((1, tokens.shape[1], 4))

    logprobs = completion_logprob(Model(), Tokenizer(), "prompt", "", adapters_on=False)
    assert logprobs.shape == (1, 1)
    assert float(logprobs[0, 0]) < 0.0


def test_empty_text_completion_without_eos_fails_explicitly():
    class Tokenizer:
        eos_token_id = None

        @staticmethod
        def encode(text, add_special_tokens=True):
            return [1] if text else []

    with pytest.raises(RuntimeError, match="no EOS token"):
        completion_logprob(object(), Tokenizer(), "prompt", "", adapters_on=False)


def test_heldout_evaluation_emits_bounded_progress(capsys, monkeypatch):
    class Response:
        text = "FINAL_ANSWER true"

    def fake_stream_generate(*_args, **_kwargs):
        yield Response()

    class LoRALinear:
        pass

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.stream_generate = fake_stream_generate
    mlx_lm_tuner = types.ModuleType("mlx_lm.tuner")
    mlx_lm_lora = types.ModuleType("mlx_lm.tuner.lora")
    mlx_lm_lora.LoRALinear = LoRALinear
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        mlx_lm,
    )
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner", mlx_lm_tuner)
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.lora", mlx_lm_lora)

    class Task:
        domain = "logic"
        depth = 2
        knowledge = "parametric"
        grader = "exact"
        expected = True

        def __init__(self, task_id: str, correct: bool = True) -> None:
            self.task_id = task_id
            self.prompt = "solve"
            self.correct = correct

        def grade(self, _text: str) -> dict[str, bool]:
            return {"correct": self.correct}

    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            return messages[0]["content"]

    tasks = [Task(f"t-{index}", correct=index != 1) for index in range(5)]

    report = evaluate_heldout(
        object(),
        Tokenizer(),
        tasks,
        max_tokens=8,
        envelope=None,
        adapters_on=False,
        progress_label="baseline-standard",
        progress_every=2,
    )

    out = capsys.readouterr().out
    assert "[baseline-standard] 1/5 running=1.000" in out
    assert "[baseline-standard] 2/5 running=0.500" in out
    assert "[baseline-standard] 4/5 running=0.750" in out
    assert "[baseline-standard] 5/5 running=0.800" in out
    assert report["overall"] == pytest.approx(0.8)
    assert report["score_reasons"] == {"correct": 4, "incorrect": 1}


def test_recurrent_heldout_uses_contract_aware_decode(monkeypatch):
    captured_configs = []

    class Random:
        @staticmethod
        def seed(_seed):
            return None

    mlx = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.random = Random()
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    class FakeEngine:
        def __init__(self, _model, *, tokenizer, config, schedule_library):
            assert tokenizer is not None
            assert schedule_library is None
            captured_configs.append(config)

        def reason(self, **kwargs):
            assert kwargs["decode_max_tokens"] == 8
            assert kwargs["decode_sentence_grace_tokens"] == 0
            return types.SimpleNamespace(
                ok=True,
                reason="",
                text='FINAL_ANSWER: {"value":1}',
                tokens=[1, 2, 3],
                receipt=types.SimpleNamespace(
                    selected_branch=0,
                    steps_taken=3,
                    decode_termination="contract_complete",
                ),
            )

    import core.brain.llm.latent_cortex.engine as engine_module

    monkeypatch.setattr(engine_module, "LatentCortexEngine", FakeEngine)
    adapter_module = types.ModuleType("core.brain.llm.latent_cortex.recurrence_adapter")
    adapter_module.recurrence_adapter_disabled = __import__("contextlib").nullcontext
    adapter_module.current_recurrence_adapter_scope = lambda: None
    adapter_module.recurrence_adapter_scope = lambda *_args, **_kwargs: __import__(
        "contextlib"
    ).nullcontext()
    monkeypatch.setitem(
        sys.modules,
        "core.brain.llm.latent_cortex.recurrence_adapter",
        adapter_module,
    )

    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            return messages[0]["content"]

        @staticmethod
        def encode(_text):
            return [1, 2, 3]

    class Task:
        task_id = "contract-task"
        prompt = "solve"
        domain = "logic"
        depth = 2
        knowledge = "parametric"
        grader = "exact_json"
        expected = {"value": 1}

        @staticmethod
        def grade(text):
            return {"correct": text == 'FINAL_ANSWER: {"value":1}'}

    spec = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution", "critical_audit"),
        recurrent_steps=3,
    )

    report = evaluate_recurrent_heldout(
        object(),
        Tokenizer(),
        [Task()],
        spec=spec,
        max_tokens=8,
        envelope=None,
        adapters_on=False,
        seed=12,
    )

    assert report["overall"] == pytest.approx(1.0)
    assert captured_configs[0].decode_contract == "final_answer_v1"
    assert captured_configs[0].decode_contract_grace_tokens == 0
    assert report["episode_receipts"][0]["decode_termination"] == "contract_complete"
    assert report["episode_receipts"][0]["score_reason"] == "correct"
    assert report["episode_receipts"][0]["contract"] == {
        "marker_count": 1,
        "complete": True,
        "valid": True,
        "reason": "complete",
    }
    assert report["score_reasons"] == {"correct": 1}
    assert report["contract_reasons"] == {"complete": 1}


def test_recurrent_heldout_scores_receipted_abstention_as_incorrect(monkeypatch):
    class Random:
        @staticmethod
        def seed(_seed):
            return None

    mlx = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.random = Random()
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    class FakeEngine:
        def __init__(self, _model, *, tokenizer, config, schedule_library):
            assert tokenizer is not None
            assert schedule_library is None
            assert config.answer_replacement_enabled is False

        @staticmethod
        def reason(**_kwargs):
            return types.SimpleNamespace(
                ok=False,
                reason="answer_replacement_abstained",
                text="",
                tokens=[],
                receipt=types.SimpleNamespace(
                    selected_branch=1,
                    steps_taken=4,
                    decode_termination="confidence_bound_abstention",
                ),
            )

    import core.brain.llm.latent_cortex.engine as engine_module

    monkeypatch.setattr(engine_module, "LatentCortexEngine", FakeEngine)
    adapter_module = types.ModuleType("core.brain.llm.latent_cortex.recurrence_adapter")
    adapter_module.recurrence_adapter_disabled = __import__("contextlib").nullcontext
    adapter_module.current_recurrence_adapter_scope = lambda: None
    adapter_module.recurrence_adapter_scope = lambda *_args, **_kwargs: __import__(
        "contextlib"
    ).nullcontext()
    monkeypatch.setitem(
        sys.modules,
        "core.brain.llm.latent_cortex.recurrence_adapter",
        adapter_module,
    )

    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            return messages[0]["content"]

        @staticmethod
        def encode(_text):
            return [1, 2, 3]

    class Task:
        task_id = "abstained-contract-task"
        prompt = "solve"
        domain = "logic"
        depth = 2
        knowledge = "parametric"
        grader = "exact_json"
        expected = {"value": 1}

        @staticmethod
        def grade(text):
            return {
                "correct": False,
                "reason": "missing_answer" if not text else "wrong_answer",
            }

    report = evaluate_recurrent_heldout(
        object(),
        Tokenizer(),
        [Task()],
        spec=RLCExecutionSpec(
            n_slots=4,
            branch_roles=("constructive_solution", "critical_audit"),
            recurrent_steps=4,
        ),
        max_tokens=8,
        envelope=None,
        adapters_on=False,
        seed=12,
    )

    assert report["overall"] == 0.0
    assert report["score_reasons"] == {"missing_answer": 1}
    assert report["contract_reasons"] == {"no_marker": 1}
    assert report["episode_receipts"] == [
        {
            "task_id": "abstained-contract-task",
            "selected_branch": 1,
            "steps_taken": 4,
            "decode_termination": "confidence_bound_abstention",
            "output_tokens": 0,
            "correct": False,
            "score_reason": "missing_answer",
            "episode_ok": False,
            "episode_reason": "answer_replacement_abstained",
            "scored_policy_failure": True,
            "contract": {
                "marker_count": 0,
                "complete": False,
                "valid": False,
                "reason": "no_marker",
            },
        }
    ]


def test_recurrent_heldout_resumes_only_unfinished_identity_bound_suffix(
    monkeypatch,
    tmp_path,
    capsys,
):
    class Random:
        @staticmethod
        def seed(_seed):
            return None

    mlx = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.random = Random()
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    calls = []
    fail_second = [True]

    class FakeEngine:
        def __init__(self, _model, *, tokenizer, config, schedule_library):
            assert tokenizer is not None
            assert schedule_library is None
            assert config.answer_replacement_enabled is False

        @staticmethod
        def reason(*, token_ids, **_kwargs):
            value = token_ids[0]
            calls.append(value)
            if value == 2 and fail_second[0]:
                return types.SimpleNamespace(
                    ok=False,
                    reason="resident_worker_lost",
                    text="",
                    tokens=[],
                    receipt=types.SimpleNamespace(
                        selected_branch=0,
                        steps_taken=1,
                        decode_termination="worker_lost",
                    ),
                )
            return types.SimpleNamespace(
                ok=True,
                reason="",
                text=f'FINAL_ANSWER: {{"value":{value}}}',
                tokens=[value],
                receipt=types.SimpleNamespace(
                    selected_branch=0,
                    steps_taken=4,
                    decode_termination="contract_complete",
                ),
            )

    import core.brain.llm.latent_cortex.engine as engine_module

    monkeypatch.setattr(engine_module, "LatentCortexEngine", FakeEngine)
    adapter_module = types.ModuleType("core.brain.llm.latent_cortex.recurrence_adapter")
    adapter_module.recurrence_adapter_disabled = __import__("contextlib").nullcontext
    adapter_module.current_recurrence_adapter_scope = lambda: None
    adapter_module.recurrence_adapter_scope = lambda *_args, **_kwargs: __import__(
        "contextlib"
    ).nullcontext()
    monkeypatch.setitem(
        sys.modules,
        "core.brain.llm.latent_cortex.recurrence_adapter",
        adapter_module,
    )

    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, **_kwargs):
            return messages[0]["content"]

        @staticmethod
        def encode(text):
            return [1 if "first" in text else 2]

    @dataclass(frozen=True)
    class Task:
        task_id: str
        prompt: str
        expected_value: int
        domain: str = "logic"
        depth: int = 2
        knowledge: str = "parametric"
        grader: str = "exact_json"

        def grade(self, text):
            expected = f'FINAL_ANSWER: {{"value":{self.expected_value}}}'
            return {
                "correct": text == expected,
                "reason": "correct" if text == expected else "wrong_answer",
            }

    tasks = [
        Task("first-task", "solve first", 1),
        Task("second-task", "solve second", 2),
    ]
    spec = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution", "critical_audit"),
        recurrent_steps=4,
    )
    progress_path = tmp_path / "baseline-progress.json"
    identity = {
        "schema": "test.recurrent_baseline_identity.v1",
        "protocol_sha256": "a" * 64,
        "task_ids": [task.task_id for task in tasks],
    }

    with pytest.raises(RuntimeError, match="resident_worker_lost"):
        evaluate_recurrent_heldout(
            object(),
            Tokenizer(),
            tasks,
            spec=spec,
            max_tokens=8,
            envelope=None,
            adapters_on=False,
            seed=12,
            progress_path=progress_path,
            progress_identity=identity,
        )

    interrupted = json.loads(progress_path.read_text(encoding="ascii"))
    assert interrupted["complete"] is False
    assert [record["task_id"] for record in interrupted["records"]] == ["first-task"]
    assert calls == [1, 2]

    fail_second[0] = False
    report = evaluate_recurrent_heldout(
        object(),
        Tokenizer(),
        tasks,
        spec=spec,
        max_tokens=8,
        envelope=None,
        adapters_on=False,
        seed=12,
        progress_label="baseline-recurrent",
        progress_path=progress_path,
        progress_identity=identity,
    )

    assert calls == [1, 2, 2]
    assert report["overall"] == 1.0
    assert json.loads(progress_path.read_text(encoding="ascii"))["complete"] is True
    assert "[baseline-recurrent] resumed 1/2" in capsys.readouterr().out

    with pytest.raises(GRPOCheckpointError, match="identity differs"):
        evaluate_recurrent_heldout(
            object(),
            Tokenizer(),
            tasks,
            spec=spec,
            max_tokens=8,
            envelope=None,
            adapters_on=False,
            seed=12,
            progress_path=progress_path,
            progress_identity={**identity, "protocol_sha256": "b" * 64},
        )
    assert calls == [1, 2, 2]
