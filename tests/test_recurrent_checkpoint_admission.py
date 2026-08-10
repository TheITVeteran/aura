from __future__ import annotations

import copy
import hashlib
import json

import pytest

from core.brain.llm.latent_cortex.incumbent_artifact import build_incumbent_artifact
from core.learning.recurrence_curriculum import task_battery
from core.learning.recurrent_checkpoint_admission import (
    RecurrentCheckpointAdmissionError,
    build_checkpoint_behavioral_admission,
    build_free_generation_report,
    build_full_engine_behavioral_admission,
    build_recurrence_task_manifest,
    validate_checkpoint_behavioral_admission,
    validate_free_generation_report,
    validate_full_engine_behavioral_admission,
    validate_recurrence_task_free_generation_report,
    validate_recurrence_task_manifest,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


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


def _bind_record_arm(
    row: dict[str, object],
    *,
    arm: str,
) -> dict[str, object]:
    if arm != "ordinary_decode":
        return row
    receipt = {
        "schema": "aura.rlc.ordinary_decode_probe.v2",
        "arm": "ordinary_decode",
        "task_id": row["task_id"],
        "depth_coordinate": row["depth"],
        "generation_seed": 17,
        "recurrent_steps": 0,
        "prompt_tokens_sha256": _digest(f"prompt:{row['task_id']}"),
        "response_sha256": row["response_sha256"],
        "tokens_sha256": row["tokens_sha256"],
        "token_count": row["token_count"],
        "decode_termination": row["decode_termination"],
    }
    row["decode_incumbent_policy"] = "vanilla_incumbent"
    row["episode_receipt"] = receipt
    row["episode_receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return row


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
            _bind_record_arm(
                _record(task_id, depth, outcomes[(task_id, depth)]),
                arm=arm,
            )
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
            records.append(_bind_record_arm(row, arm=arm))
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


def _ordinary_report_for_adapter(
    adapter_label: str,
    outcomes: dict[tuple[str, int], bool],
) -> dict[str, object]:
    report = _report("ordinary_decode", outcomes)
    return build_free_generation_report(
        arm="ordinary_decode",
        adapter_sha256=_digest(adapter_label),
        execution_spec_sha256=report["execution_spec_sha256"],
        task_manifest_sha256=report["task_manifest_sha256"],
        task_ids=report["task_ids"],
        depths=report["depths"],
        records=report["records"],
    )


def _full_engine_report(
    adapter_label: str,
    ordinary: dict[str, object],
    outcomes: dict[tuple[str, int], bool],
    *,
    replacement_source: str = "branch_candidate",
    objective_solver_valid: bool = True,
) -> dict[str, object]:
    ordinary_rows = {
        (row["task_id"], row["depth"]): row for row in ordinary["records"]
    }
    records = []
    for task_id in ordinary["task_ids"]:
        task = _TASK_BY_ID[task_id]
        for depth in ordinary["depths"]:
            coordinate = (task_id, depth)
            ordinary_row = ordinary_rows[coordinate]
            correct = outcomes[coordinate]
            full_text = (
                task.answer
                if correct
                else ordinary_row["response_text"]
                if not ordinary_row["correct"]
                else 'FINAL_ANSWER: {"wrong":true}'
            )
            replaces = full_text != ordinary_row["response_text"]
            full_tokens = (
                [91, depth, len(task_id)]
                if replaces
                else list(ordinary_row["tokens"])
            )
            incumbent = build_incumbent_artifact(
                input_tokens=[7, 8, depth],
                output_tokens=ordinary_row["tokens"],
                output_text=ordinary_row["response_text"],
                checkpoint_fingerprint="c" * 64,
                checkpoint_fingerprint_method="sha256",
                max_tokens=32,
                n_layers=8,
                termination=ordinary_row["decode_termination"],
            )
            baseline = {
                "text_sha256": ordinary_row["response_sha256"],
                "tokens_sha256": ordinary_row["tokens_sha256"],
                "token_count": ordinary_row["token_count"],
            }
            accepted = {
                "source": replacement_source if replaces else "baseline_decode",
                "text_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
                "tokens_sha256": _canonical_digest(full_tokens),
                "token_count": len(full_tokens),
                "binding_status": (
                    "exact_text_token_roundtrip" if replaces else "not_required"
                ),
            }
            replacement_body = {
                "schema": "aura.rlc.answer_replacement.v5",
                "authority": "confidence_bound_answer_replacement",
                "decision": "replace" if replaces else "retain",
                "selected_request_id": (
                    "objective-program"
                    if replaces and replacement_source == "objective_program_solution"
                    else "branch-0"
                    if replaces
                    else ""
                ),
                "baseline_decode": baseline,
                "accepted_output": accepted,
                "candidates": (
                    [
                        {
                            "request_id": "objective-program",
                            "branch": -1,
                            "transaction_status": (
                                "objective_program_solution"
                                if objective_solver_valid
                                else "forged_solution"
                            ),
                            "transaction_sha256": "d" * 64,
                            "required_verifier": "exact_objective_program",
                            "same_verifier_class": True,
                            "replacement_quality": {
                                "basis": "objective_program_exact_complete",
                                "lower_bound": 1.0,
                                "upper_bound": 1.0,
                            },
                            "dominates": True,
                        }
                    ]
                    if replaces and replacement_source == "objective_program_solution"
                    else [{"request_id": "branch-0", "dominates": True}]
                    if replaces
                    else []
                ),
            }
            replacement = {
                **replacement_body,
                "receipt_sha256": _canonical_digest(replacement_body),
            }
            episode_receipt = {
                "episode_id": f"full:{adapter_label}:{task_id}:{depth}",
                "input_tokens_sha256": _digest(f"prompt:{task_id}"),
                "input_token_count": 11,
                "steps_taken": depth,
                "n_branches": 2,
                "selected_branch": 0,
                "branch_selection_admitted": True,
                "decode_incumbent_policy": "vanilla_incumbent",
                "decode_termination": (
                    "confidence_bound_replacement"
                    if replaces
                    else ordinary_row["decode_termination"]
                ),
                "decode_generated_tokens": len(full_tokens),
                "params_unchanged": True,
                "checkpoint_fingerprint": "c" * 64,
                "checkpoint_fingerprint_method": "sha256",
                "nonparametric_memory": {"status": "disabled_by_policy"},
                "honest_flags": [],
                "recurrence_adapter": {
                    "schema": "aura.recurrence_adapter_activation.v1",
                    "scope": "latent_slots_only",
                    "active": True,
                    "calls": 2,
                    "adapted_positions": 8,
                },
                "incumbent_artifact": incumbent.receipt,
                "answer_replacement": replacement,
            }
            grade = task.grade(full_text)
            record = {
                "task_id": task_id,
                "depth": depth,
                "response_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
                "response_text": full_text,
                "tokens_sha256": _canonical_digest(full_tokens),
                "tokens": full_tokens,
                "token_count": len(full_tokens),
                "correct": correct,
                "grade_receipt": grade,
                "episode_ok": True,
                "episode_reason": "",
                "decode_termination": episode_receipt["decode_termination"],
                "branch_selection_admitted": True,
                "decode_incumbent_policy": "vanilla_incumbent",
                "episode_receipt_sha256": _canonical_digest(episode_receipt),
                "episode_receipt": episode_receipt,
            }
            records.append(record)
    return build_free_generation_report(
        arm="full_engine",
        adapter_sha256=_digest(adapter_label),
        execution_spec_sha256=ordinary["execution_spec_sha256"],
        task_manifest_sha256=ordinary["task_manifest_sha256"],
        task_ids=ordinary["task_ids"],
        depths=ordinary["depths"],
        records=records,
    )


def _with_coda_only_episode_evidence(
    report: dict[str, object],
) -> list[dict[str, object]]:
    records = copy.deepcopy(report["records"])
    for row in records:
        receipt = row["episode_receipt"]
        receipt["recurrence_adapter"] = {
            "schema": "aura.recurrence_adapter_activation.v1",
            "scope": "latent_slots_only",
            "active": False,
            "calls": 0,
            "adapted_positions": 0,
        }
        receipt["coda_adapter"] = {
            "schema": "aura.coda_adapter_activation.v1",
            "scope": "rlc_coda_only",
            "active": True,
            "calls": 2,
            "adapted_positions": 8,
            "observed_positions": 8,
            "applied_blocks": {"7": 2},
            "applied_sites": {"model.layers.7.self_attn.o_proj": 2},
        }
        row["episode_receipt_sha256"] = _canonical_digest(receipt)
    return records


def test_full_engine_report_accepts_bound_coda_only_activation():
    task_a, task_b = tuple(task.task_id for task in _TASKS)
    outcomes = {
        (task_a, 1): True,
        (task_a, 2): True,
        (task_b, 1): False,
        (task_b, 2): False,
    }
    ordinary = _ordinary_report_for_adapter("coda-only", outcomes)
    full = _full_engine_report("coda-only", ordinary, outcomes)

    admitted = build_free_generation_report(
        arm="full_engine",
        adapter_sha256=full["adapter_sha256"],
        execution_spec_sha256=full["execution_spec_sha256"],
        task_manifest_sha256=full["task_manifest_sha256"],
        task_ids=full["task_ids"],
        depths=full["depths"],
        records=_with_coda_only_episode_evidence(full),
    )

    assert [row["correct"] for row in admitted["records"]] == [
        row["correct"] for row in full["records"]
    ]
    assert all(
        row["episode_receipt"]["coda_adapter"]["active"]
        for row in admitted["records"]
    )
    persisted = json.loads(json.dumps(admitted))
    assert validate_free_generation_report(persisted) == persisted


def test_recurrence_report_rejects_coda_as_recurrent_activation():
    task_a, task_b = tuple(task.task_id for task in _TASKS)
    outcomes = {
        (task_a, 1): True,
        (task_a, 2): False,
        (task_b, 1): False,
        (task_b, 2): False,
    }
    report = _report("trained_adapter", outcomes)

    with pytest.raises(
        RecurrentCheckpointAdmissionError,
        match="recurrent_checkpoint_episode_evidence_invalid",
    ):
        build_free_generation_report(
            arm="trained_adapter",
            adapter_sha256=report["adapter_sha256"],
            execution_spec_sha256=report["execution_spec_sha256"],
            task_manifest_sha256=report["task_manifest_sha256"],
            task_ids=report["task_ids"],
            depths=report["depths"],
            records=_with_coda_only_episode_evidence(report),
        )


def test_full_engine_evidence_accepts_only_bound_objective_solver_authority():
    task_a, task_b = tuple(task.task_id for task in _TASKS)
    ordinary_outcomes = {
        (task_a, 1): True,
        (task_a, 2): True,
        (task_b, 1): False,
        (task_b, 2): False,
    }
    ordinary = _ordinary_report_for_adapter("objective-solver", ordinary_outcomes)
    solver_outcomes = {**ordinary_outcomes, (task_b, 2): True}

    report = _full_engine_report(
        "objective-solver",
        ordinary,
        solver_outcomes,
        replacement_source="objective_program_solution",
    )
    replaced = [
        row
        for row in report["records"]
        if row["episode_receipt"]["answer_replacement"]["decision"] == "replace"
    ]
    assert len(replaced) == 1
    assert (
        replaced[0]["episode_receipt"]["answer_replacement"]["accepted_output"]["source"]
        == "objective_program_solution"
    )

    with pytest.raises(
        RecurrentCheckpointAdmissionError,
        match="full_engine_replacement_invalid",
    ):
        _full_engine_report(
            "forged-objective-solver",
            ordinary,
            solver_outcomes,
            replacement_source="objective_program_solution",
            objective_solver_valid=False,
        )


def test_full_engine_admission_requires_floor_preservation_and_verified_gain():
    task_a, task_b = tuple(task.task_id for task in _TASKS)
    ordinary_outcomes = {
        (task_a, 1): True,
        (task_a, 2): True,
        (task_b, 1): False,
        (task_b, 2): False,
    }
    initial_ordinary = _ordinary_report_for_adapter("initial-full", ordinary_outcomes)
    trained_ordinary = _ordinary_report_for_adapter("trained-full", ordinary_outcomes)
    initial_full = _full_engine_report(
        "initial-full",
        initial_ordinary,
        ordinary_outcomes,
    )
    trained_full = _full_engine_report(
        "trained-full",
        trained_ordinary,
        {
            **ordinary_outcomes,
            (task_b, 2): True,
        },
    )

    admission = build_full_engine_behavioral_admission(
        initial_full_engine_report=initial_full,
        trained_full_engine_report=trained_full,
        initial_ordinary_decode_report=initial_ordinary,
        trained_ordinary_decode_report=trained_ordinary,
        task_manifest=_TASK_MANIFEST,
    )

    assert admission["admitted"] is True
    assert admission["authorized_correct_gains"] == 1
    assert admission["ordinary_floor_regressions"] == 0
    assert admission["training_correct_gain"] == 1
    assert admission["ordinary_correct_gain"] == 1
    assert all(admission["gates"].values())
    assert not any(admission["claim_flags"].values())
    assert validate_full_engine_behavioral_admission(
        admission,
        initial_full_engine_report=initial_full,
        trained_full_engine_report=trained_full,
        initial_ordinary_decode_report=initial_ordinary,
        trained_ordinary_decode_report=trained_ordinary,
        task_manifest=_TASK_MANIFEST,
    ) == admission


def test_full_engine_admission_rejects_a_gain_that_drops_an_incumbent_answer():
    task_a, task_b = tuple(task.task_id for task in _TASKS)
    ordinary_outcomes = {
        (task_a, 1): True,
        (task_a, 2): True,
        (task_b, 1): False,
        (task_b, 2): False,
    }
    initial_ordinary = _ordinary_report_for_adapter("initial-floor", ordinary_outcomes)
    trained_ordinary = _ordinary_report_for_adapter("trained-floor", ordinary_outcomes)
    initial_full = _full_engine_report(
        "initial-floor",
        initial_ordinary,
        ordinary_outcomes,
    )
    trained_full = _full_engine_report(
        "trained-floor",
        trained_ordinary,
        {
            (task_a, 1): False,
            (task_a, 2): True,
            (task_b, 1): True,
            (task_b, 2): True,
        },
    )

    admission = build_full_engine_behavioral_admission(
        initial_full_engine_report=initial_full,
        trained_full_engine_report=trained_full,
        initial_ordinary_decode_report=initial_ordinary,
        trained_ordinary_decode_report=trained_ordinary,
        task_manifest=_TASK_MANIFEST,
    )

    assert admission["trained_full_engine_correct"] == 3
    assert admission["ordinary_decode_correct"] == 2
    assert admission["ordinary_floor_regressions"] == 1
    assert admission["gates"]["ordinary_correctness_floor_preserved"] is False
    assert admission["admitted"] is False
