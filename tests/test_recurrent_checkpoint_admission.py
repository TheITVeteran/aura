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

    # The vanilla control the trained checkpoint has to beat. Two of its four
    # observations are correct; the trained arm gets three.
    ordinary = _report(
        "ordinary_decode",
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
        ordinary_decode_report=ordinary,
    )

    assert admission["admitted"] is True
    assert admission["gates"]["beats_ordinary_decode"] is True
    assert admission["ordinary_decode_correct"] == 2
    assert admission["trained_correct"] == 3
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
            ordinary_decode_report=ordinary,
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


def _admission_arms(trained_outcomes, ordinary_outcomes):
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
    trained = _report("trained_adapter", trained_outcomes(task_a, task_b))
    ordinary = _report("ordinary_decode", ordinary_outcomes(task_a, task_b))
    return initial, trained, ordinary


def test_admission_is_refused_without_an_ordinary_decode_control():
    """Beating an untrained adapter on a degraded path is not evidence.

    The 2026-08-06 campaign is the case this closes: adapter plus RLC scored
    3/28 while ordinary decode on identical frozen weights scored 13/28. An
    admission that never looks at the vanilla arm cannot see that.
    """
    initial, trained, _ = _admission_arms(
        lambda a, b: {(a, 1): True, (a, 2): True, (b, 1): False, (b, 2): True},
        lambda a, b: {(a, 1): False, (a, 2): False, (b, 1): False, (b, 2): False},
    )

    admission = build_checkpoint_behavioral_admission(
        initial_report=initial,
        trained_report=trained,
        task_manifest=_TASK_MANIFEST,
    )

    assert admission["admitted"] is False
    assert admission["decision"] == "reject_no_ordinary_decode_control"
    assert admission["gates"]["beats_ordinary_decode"] is False
    assert admission["ordinary_decode_correct"] is None
    # Every other gate passed; the missing control alone refused it.
    assert admission["gates"]["strict_heldout_free_generation_gain"] is True
    assert admission["gates"]["positive_training_by_depth_interaction"] is True


def test_admission_is_refused_when_ordinary_decode_answers_more():
    """A real gain over the untrained start still loses to the vanilla floor."""
    initial, trained, ordinary = _admission_arms(
        lambda a, b: {(a, 1): True, (a, 2): True, (b, 1): False, (b, 2): True},
        lambda a, b: {(a, 1): True, (a, 2): True, (b, 1): True, (b, 2): True},
    )

    admission = build_checkpoint_behavioral_admission(
        initial_report=initial,
        trained_report=trained,
        task_manifest=_TASK_MANIFEST,
        ordinary_decode_report=ordinary,
    )

    assert admission["trained_correct"] == 3
    assert admission["ordinary_decode_correct"] == 4
    assert admission["gates"]["beats_ordinary_decode"] is False
    assert admission["admitted"] is False
    assert admission["decision"] == "reject_checkpoint_behavioral_gain_unproven"
    # Improving on itself is still true, and still not enough.
    assert admission["aggregate_correct_gain"] == 2
    assert not any(admission["claim_flags"].values())


def test_admission_refuses_a_control_bound_to_other_tasks():
    """A vanilla arm graded on different questions is not a control."""
    initial, trained, ordinary = _admission_arms(
        lambda a, b: {(a, 1): True, (a, 2): True, (b, 1): False, (b, 2): True},
        lambda a, b: {(a, 1): False, (a, 2): False, (b, 1): False, (b, 2): False},
    )
    forged = copy.deepcopy(ordinary)
    forged["task_manifest_sha256"] = _digest("some other battery")

    with pytest.raises(RecurrentCheckpointAdmissionError):
        build_checkpoint_behavioral_admission(
            initial_report=initial,
            trained_report=trained,
            task_manifest=_TASK_MANIFEST,
            ordinary_decode_report=forged,
        )


def _report_with_texts(arm: str, outcomes, reasoned: bool) -> dict[str, object]:
    """A report whose responses carry a real reasoning prefix, or none at all.

    The answer text itself is the graded one either way, so the only thing
    that varies between arms here is whether the model showed its work.
    """
    task_ids = tuple(task.task_id for task in _TASKS)
    depths = (1, 2)
    records = []
    for task_id in task_ids:
        for depth in depths:
            correct = outcomes[(task_id, depth)]
            row = _record(task_id, depth, correct)
            answer = (
                _TASK_BY_ID[task_id].answer
                if correct
                else 'FINAL_ANSWER: {"wrong":true}'
            )
            text = (
                f"Working through {task_id} at depth {depth}.\n{answer}"
                if reasoned
                else answer
            )
            row["response_text"] = text
            row["response_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            records.append(row)
    return build_free_generation_report(
        arm=arm,
        adapter_sha256=_digest(arm),
        execution_spec_sha256=_digest("spec"),
        task_manifest_sha256=_TASK_MANIFEST_SHA256,
        task_ids=task_ids,
        depths=depths,
        records=records,
    )


def test_admission_refuses_a_checkpoint_that_learned_to_stop_reasoning():
    """The cp796 / role-v6 failure: correct more often, but answering blind.

    Both runs drove validation cross-entropy down smoothly while the model
    learned to emit the answer immediately -- median generated tokens 28
    against 452 for the untrained path. Correctness alone would admit a
    checkpoint like this the moment it got lucky; the structural gate does not.
    """
    task_a, task_b = tuple(task.task_id for task in _TASKS)

    initial = _report_with_texts(
        "initial_adapter",
        {(task_a, 1): True, (task_a, 2): False, (task_b, 1): False, (task_b, 2): False},
        reasoned=True,
    )
    # Strictly better on every count -- and it stopped reasoning to get there.
    trained = _report_with_texts(
        "trained_adapter",
        {(task_a, 1): True, (task_a, 2): True, (task_b, 1): False, (task_b, 2): True},
        reasoned=False,
    )
    ordinary = _report_with_texts(
        "ordinary_decode",
        {(task_a, 1): True, (task_a, 2): True, (task_b, 1): False, (task_b, 2): False},
        reasoned=True,
    )

    admission = build_checkpoint_behavioral_admission(
        initial_report=initial,
        trained_report=trained,
        task_manifest=_TASK_MANIFEST,
        ordinary_decode_report=ordinary,
    )

    # Every correctness gate is satisfied.
    assert admission["gates"]["strict_heldout_free_generation_gain"] is True
    assert admission["gates"]["positive_training_by_depth_interaction"] is True
    assert admission["gates"]["beats_ordinary_decode"] is True
    # And it is still refused.
    assert admission["gates"]["no_answer_only_collapse"] is False
    assert admission["admitted"] is False
    assert admission["trained_answer_only_responses"] == 4
    assert admission["ordinary_answer_only_responses"] == 0


def test_a_reasoning_checkpoint_clears_the_degeneracy_gate():
    task_a, task_b = tuple(task.task_id for task in _TASKS)

    initial = _report_with_texts(
        "initial_adapter",
        {(task_a, 1): True, (task_a, 2): False, (task_b, 1): False, (task_b, 2): False},
        reasoned=True,
    )
    trained = _report_with_texts(
        "trained_adapter",
        {(task_a, 1): True, (task_a, 2): True, (task_b, 1): False, (task_b, 2): True},
        reasoned=True,
    )
    ordinary = _report_with_texts(
        "ordinary_decode",
        {(task_a, 1): True, (task_a, 2): True, (task_b, 1): False, (task_b, 2): False},
        reasoned=True,
    )

    admission = build_checkpoint_behavioral_admission(
        initial_report=initial,
        trained_report=trained,
        task_manifest=_TASK_MANIFEST,
        ordinary_decode_report=ordinary,
    )
    assert admission["gates"]["no_answer_only_collapse"] is True
    assert admission["admitted"] is True
    assert admission["decision"] == "admit_bounded_next_scale_proxy"
    # Admission of a bounded proxy still authorizes nothing on its own.
    assert not any(admission["claim_flags"].values())
