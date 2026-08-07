from __future__ import annotations

import copy
import hashlib
import json

import pytest

from core.learning.recurrence_curriculum import task_battery
from core.learning.recurrent_checkpoint_admission import (
    RecurrentCheckpointAdmissionError,
    build_checkpoint_behavioral_admission,
    build_free_generation_report,
    build_recurrence_task_manifest,
    validate_checkpoint_behavioral_admission,
    validate_free_generation_report,
    validate_recurrence_task_free_generation_report,
    validate_recurrence_task_manifest,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


_TASKS = task_battery(["boolean"], [2], 2, seed=31_337)
_TASK_MANIFEST, _TASK_MANIFEST_SHA256 = build_recurrence_task_manifest(_TASKS)
_TASK_BY_ID = {task.task_id: task for task in _TASKS}


def _record(task_id: str, depth: int, correct: bool) -> dict[str, object]:
    task = _TASK_BY_ID[task_id]
    response_text = task.answer if correct else 'FINAL_ANSWER: {"wrong":true}'
    tokens = [depth, int(correct), len(task_id)]
    grade = task.grade(response_text)
    episode_receipt = {
        "episode_id": f"episode:{task_id}:{depth}:{correct}",
        "input_tokens_sha256": _digest(f"prompt:{task_id}"),
        "input_token_count": 11,
        "steps_taken": depth,
        "n_branches": 2,
        "selected_branch": 0,
        "branch_selection_admitted": True,
        "decode_incumbent_policy": "latent",
        "decode_termination": "token_limit",
        "decode_generated_tokens": len(tokens),
        "params_unchanged": True,
        "nonparametric_memory": {"status": "disabled_by_policy"},
        "honest_flags": [],
        "recurrence_adapter": {
            "schema": "aura.recurrence_adapter_activation.v1",
            "scope": "latent_slots_only",
            "active": True,
            "calls": 2,
            "adapted_positions": 8,
        },
    }
    return {
        "task_id": task_id,
        "depth": depth,
        "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
        "response_text": response_text,
        "tokens_sha256": hashlib.sha256(
            json.dumps(
                tokens,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
        "tokens": tokens,
        "token_count": len(tokens),
        "correct": correct,
        "grade_receipt": grade,
        "episode_ok": True,
        "episode_reason": "",
        "decode_termination": "token_limit",
        "branch_selection_admitted": True,
        "decode_incumbent_policy": "latent",
        "episode_receipt_sha256": hashlib.sha256(
            json.dumps(
                episode_receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
        "episode_receipt": episode_receipt,
    }


def _report(arm: str, outcomes: dict[tuple[str, int], bool]) -> dict[str, object]:
    task_ids = tuple(task.task_id for task in _TASKS)
    depths = (1, 2)
    return build_free_generation_report(
        arm=arm,
        adapter_sha256=_digest(arm),
        execution_spec_sha256=_digest("spec"),
        task_manifest_sha256=_TASK_MANIFEST_SHA256,
        task_ids=task_ids,
        depths=depths,
        records=[
            _record(task_id, depth, outcomes[(task_id, depth)])
            for task_id in task_ids
            for depth in depths
        ],
    )


def test_checkpoint_admission_requires_strict_gain_and_depth_interaction():
    task_a, task_b = tuple(task.task_id for task in _TASKS)
    initial = _report(
        "initial_adapter",
        {
            (task_a, 1): True,
            (task_a, 2): False,
            (task_b, 1): False,
            (task_b, 2): False,
        },
    )
    trained = _report(
        "trained_adapter",
        {
            (task_a, 1): True,
            (task_a, 2): True,
            (task_b, 1): False,
            (task_b, 2): True,
        },
    )

    admission = build_checkpoint_behavioral_admission(
        initial_report=initial,
        trained_report=trained,
        task_manifest=_TASK_MANIFEST,
    )

    assert admission["admitted"] is True
    assert admission["aggregate_correct_gain"] == 2
    assert admission["training_by_depth_interaction"] == 2
    assert admission["trained_depth_regressions"] == 0
    assert all(admission["gates"].values())
    assert not any(admission["claim_flags"].values())
    assert validate_free_generation_report(initial) == initial
    assert (
        validate_recurrence_task_free_generation_report(
            initial,
            task_manifest=_TASK_MANIFEST,
        )
        == initial
    )
    assert validate_recurrence_task_manifest(_TASK_MANIFEST) == _TASK_MANIFEST
    assert (
        validate_checkpoint_behavioral_admission(
            admission,
            initial_report=initial,
            trained_report=trained,
            task_manifest=_TASK_MANIFEST,
        )
        == admission
    )


def test_aggregate_gain_without_positive_depth_interaction_is_rejected():
    task_a, task_b = tuple(task.task_id for task in _TASKS)
    initial = _report(
        "initial_adapter",
        {
            (task_a, 1): False,
            (task_a, 2): False,
            (task_b, 1): False,
            (task_b, 2): False,
        },
    )
    trained = _report(
        "trained_adapter",
        {
            (task_a, 1): True,
            (task_a, 2): True,
            (task_b, 1): False,
            (task_b, 2): False,
        },
    )

    admission = build_checkpoint_behavioral_admission(
        initial_report=initial,
        trained_report=trained,
        task_manifest=_TASK_MANIFEST,
    )

    assert admission["aggregate_correct_gain"] == 2
    assert admission["gates"]["strict_heldout_free_generation_gain"] is True
    assert admission["gates"]["positive_training_by_depth_interaction"] is False
    assert admission["admitted"] is False


def test_deeper_regression_is_rejected_even_when_aggregate_score_rises():
    task_a, task_b = tuple(task.task_id for task in _TASKS)
    initial = _report(
        "initial_adapter",
        {
            (task_a, 1): False,
            (task_a, 2): False,
            (task_b, 1): False,
            (task_b, 2): False,
        },
    )
    trained = _report(
        "trained_adapter",
        {
            (task_a, 1): True,
            (task_a, 2): False,
            (task_b, 1): True,
            (task_b, 2): True,
        },
    )

    admission = build_checkpoint_behavioral_admission(
        initial_report=initial,
        trained_report=trained,
        task_manifest=_TASK_MANIFEST,
    )

    assert admission["aggregate_correct_gain"] == 3
    assert admission["trained_depth_regressions"] == 1
    assert admission["admitted"] is False


def test_report_replay_rejects_tampering_and_incomplete_execution():
    task_a, task_b = tuple(task.task_id for task in _TASKS)
    report = _report(
        "initial_adapter",
        {
            (task_a, 1): False,
            (task_a, 2): False,
            (task_b, 1): False,
            (task_b, 2): False,
        },
    )
    tampered = copy.deepcopy(report)
    tampered["records"][0]["correct"] = True
    with pytest.raises(RecurrentCheckpointAdmissionError, match="commitment"):
        validate_free_generation_report(tampered)

    inactive = copy.deepcopy(report)
    inactive["records"][0]["episode_receipt"]["recurrence_adapter"]["calls"] = 0
    inactive["records"][0]["episode_receipt_sha256"] = hashlib.sha256(
        json.dumps(
            inactive["records"][0]["episode_receipt"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    with pytest.raises(RecurrentCheckpointAdmissionError, match="episode_evidence"):
        build_free_generation_report(
            arm=inactive["arm"],
            adapter_sha256=inactive["adapter_sha256"],
            execution_spec_sha256=inactive["execution_spec_sha256"],
            task_manifest_sha256=inactive["task_manifest_sha256"],
            task_ids=inactive["task_ids"],
            depths=inactive["depths"],
            records=inactive["records"],
        )

    incomplete = copy.deepcopy(report)
    incomplete["records"][0]["episode_ok"] = False
    incomplete["records"][0]["episode_reason"] = "budget_exhausted"
    incomplete.pop("report_sha256")
    # Rebuilding is allowed and truthfully preserves the failed episode; the
    # paired admission gate, not serialization, decides that it cannot pass.
    rebuilt = build_free_generation_report(
        arm=incomplete["arm"],
        adapter_sha256=incomplete["adapter_sha256"],
        execution_spec_sha256=incomplete["execution_spec_sha256"],
        task_manifest_sha256=incomplete["task_manifest_sha256"],
        task_ids=incomplete["task_ids"],
        depths=incomplete["depths"],
        records=incomplete["records"],
    )
    assert rebuilt["records"][0]["episode_ok"] is False

    forged = copy.deepcopy(report)
    forged_row = forged["records"][0]
    forged_row["correct"] = True
    forged_row["grade_receipt"] = {
        **forged_row["grade_receipt"],
        "correct": True,
    }
    forged = build_free_generation_report(
        arm=forged["arm"],
        adapter_sha256=forged["adapter_sha256"],
        execution_spec_sha256=forged["execution_spec_sha256"],
        task_manifest_sha256=forged["task_manifest_sha256"],
        task_ids=forged["task_ids"],
        depths=forged["depths"],
        records=forged["records"],
    )
    with pytest.raises(RecurrentCheckpointAdmissionError, match="independent_grade"):
        validate_recurrence_task_free_generation_report(
            forged,
            task_manifest=_TASK_MANIFEST,
        )


def test_task_manifest_rejects_invented_expected_answer():
    forged = copy.deepcopy(_TASK_MANIFEST)
    forged[0]["answer"] = 'FINAL_ANSWER: {"invented":true}'
    with pytest.raises(RecurrentCheckpointAdmissionError, match="replay_mismatch"):
        validate_recurrence_task_manifest(forged)
