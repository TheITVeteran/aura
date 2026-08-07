"""Replayable behavioral admission for recurrent-training checkpoints.

Teacher-forced loss and latent trajectory losses are optimization diagnostics.
They do not establish that a checkpoint generates better answers. This module
binds exact held-out free generations at multiple recurrent depths and admits a
checkpoint only when it shows both a strict aggregate gain and a positive
depth-by-training interaction without depth regressions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final, Never

FREE_GENERATION_REPORT_SCHEMA: Final = "aura.rlc.recurrent_checkpoint_free_generation.v2"
CHECKPOINT_ADMISSION_SCHEMA: Final = "aura.rlc.recurrent_checkpoint_behavioral_admission.v1"
CHECKPOINT_ADMISSION_SCHEMA_V2: Final = (
    "aura.rlc.recurrent_checkpoint_behavioral_admission.v2"
)
# ``ordinary_decode`` is the vanilla control: the same frozen weights answering
# without the recurrent path at all. Without it an admission can only say a
# trained adapter beat an untrained adapter on the same degraded path, which is
# exactly what the 2026-08-06 campaign proved is not enough -- adapter+RLC
# scored 3/28 while ordinary decode on identical weights scored 13/28.
_ARMS: Final = frozenset({"initial_adapter", "trained_adapter", "ordinary_decode"})
_MAX_TASKS: Final = 256
_MAX_DEPTHS: Final = 8
_MAX_RESPONSE_CHARS: Final = 32_768
_MAX_RESPONSE_TOKENS: Final = 8_192
_TASK_MANIFEST_KEYS: Final = frozenset({"task_id", "family", "depth", "seed", "prompt", "answer"})


class RecurrentCheckpointAdmissionError(ValueError):
    """Free-generation evidence is malformed or cannot be replayed."""


def _fail(code: str) -> Never:
    raise RecurrentCheckpointAdmissionError(str(code))


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalize_depths(depths: Sequence[int]) -> tuple[int, ...]:
    values = tuple(depths)
    if (
        not 2 <= len(values) <= _MAX_DEPTHS
        or tuple(sorted(set(values))) != values
        or any(type(depth) is not int or depth < 1 or depth > 64 for depth in values)
    ):
        _fail("recurrent_checkpoint_depths_invalid")
    return values


def _validate_episode_evidence(
    value: Any,
    *,
    depth: int,
    token_count: int,
    decode_termination: str,
    branch_selection_admitted: bool,
    episode_receipt_sha256: str,
) -> dict[str, Any]:
    """Validate the recurrent mechanics behind one graded completion.

    Earlier reports retained only a digest of an unavailable episode receipt.
    That bound bytes but left an independent verifier unable to establish that
    recurrence or the scoped adapter actually executed. The complete public
    receipt is now carried in the report and its promotion-critical fields are
    reconstructed here.
    """

    if not isinstance(value, Mapping):
        _fail("recurrent_checkpoint_episode_evidence_invalid")
    receipt = dict(value)
    activation = receipt.get("recurrence_adapter")
    nonparametric = receipt.get("nonparametric_memory")
    honest_flags = receipt.get("honest_flags")
    selected_branch = receipt.get("selected_branch")
    n_branches = receipt.get("n_branches")
    if (
        _sha(receipt) != episode_receipt_sha256
        or not isinstance(receipt.get("episode_id"), str)
        or not receipt["episode_id"]
        or not _is_sha256(receipt.get("input_tokens_sha256"))
        or type(receipt.get("input_token_count")) is not int
        or receipt["input_token_count"] < 1
        or type(receipt.get("steps_taken")) is not int
        or receipt["steps_taken"] != depth
        or type(n_branches) is not int
        or n_branches < 2
        or type(selected_branch) is not int
        or not 0 <= selected_branch < n_branches
        or receipt.get("branch_selection_admitted") is not branch_selection_admitted
        or receipt.get("decode_incumbent_policy") != "latent"
        or receipt.get("decode_termination") != decode_termination
        or type(receipt.get("decode_generated_tokens")) is not int
        or receipt["decode_generated_tokens"] != token_count
        or receipt.get("params_unchanged") is not True
        or not isinstance(nonparametric, Mapping)
        or nonparametric.get("status") != "disabled_by_policy"
        or not isinstance(honest_flags, list)
        or any(str(flag).startswith("fallback_") for flag in honest_flags)
        or not isinstance(activation, Mapping)
        or activation.get("schema") != "aura.recurrence_adapter_activation.v1"
        or activation.get("scope") != "latent_slots_only"
        or activation.get("active") is not True
        or type(activation.get("calls")) is not int
        or activation["calls"] < 1
        or type(activation.get("adapted_positions")) is not int
        or activation["adapted_positions"] < 1
    ):
        _fail("recurrent_checkpoint_episode_evidence_invalid")
    return receipt


def build_recurrence_task_manifest(
    tasks: Sequence[Any],
) -> tuple[list[dict[str, Any]], str]:
    """Build the canonical, generator-replayable held-out task manifest."""

    rows = [
        {
            "task_id": task.task_id,
            "family": task.family,
            "depth": task.depth,
            "seed": task.seed,
            "prompt": task.prompt,
            "answer": task.answer,
        }
        for task in tasks
    ]
    validated = validate_recurrence_task_manifest(rows)
    return validated, _sha(validated)


def validate_recurrence_task_manifest(
    task_manifest: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Regenerate every task and reject manifests with invented answers."""

    from core.learning.recurrence_curriculum import TASK_GENERATORS

    if isinstance(task_manifest, (str, bytes)) or not 1 <= len(task_manifest) <= _MAX_TASKS:
        _fail("recurrent_checkpoint_task_manifest_invalid")
    rows: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    for item in task_manifest:
        if not isinstance(item, Mapping):
            _fail("recurrent_checkpoint_task_manifest_invalid")
        row = dict(item)
        family = row.get("family")
        depth = row.get("depth")
        seed = row.get("seed")
        if (
            set(row) != _TASK_MANIFEST_KEYS
            or not isinstance(family, str)
            or family not in TASK_GENERATORS
            or type(depth) is not int
            or type(seed) is not int
        ):
            _fail("recurrent_checkpoint_task_manifest_invalid")
        try:
            task = TASK_GENERATORS[family](depth, seed)
        except (TypeError, ValueError) as exc:
            raise RecurrentCheckpointAdmissionError(
                "recurrent_checkpoint_task_manifest_invalid"
            ) from exc
        expected = {
            "task_id": task.task_id,
            "family": task.family,
            "depth": task.depth,
            "seed": task.seed,
            "prompt": task.prompt,
            "answer": task.answer,
        }
        if row != expected or task.task_id in task_ids:
            _fail("recurrent_checkpoint_task_manifest_replay_mismatch")
        task_ids.add(task.task_id)
        rows.append(row)
    return rows


def build_free_generation_report(
    *,
    arm: str,
    adapter_sha256: str,
    execution_spec_sha256: str,
    task_manifest_sha256: str,
    task_ids: Sequence[str],
    depths: Sequence[int],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind one adapter arm's exact held-out outputs and correctness results."""

    normalized_depths = _normalize_depths(depths)
    normalized_tasks = tuple(task_ids)
    if (
        arm not in _ARMS
        or not _is_sha256(adapter_sha256)
        or not _is_sha256(execution_spec_sha256)
        or not _is_sha256(task_manifest_sha256)
        or not 1 <= len(normalized_tasks) <= _MAX_TASKS
        or len(set(normalized_tasks)) != len(normalized_tasks)
        or any(not isinstance(task_id, str) or not task_id for task_id in normalized_tasks)
    ):
        _fail("recurrent_checkpoint_report_identity_invalid")
    expected_coordinates = [
        (task_id, depth) for task_id in normalized_tasks for depth in normalized_depths
    ]
    rows = [dict(record) for record in records]
    if len(rows) != len(expected_coordinates):
        _fail("recurrent_checkpoint_report_coverage_invalid")
    normalized_rows: list[dict[str, Any]] = []
    for row, (task_id, depth) in zip(rows, expected_coordinates, strict=True):
        if (
            set(row)
            != {
                "task_id",
                "depth",
                "response_sha256",
                "response_text",
                "tokens_sha256",
                "tokens",
                "token_count",
                "correct",
                "grade_receipt",
                "episode_ok",
                "episode_reason",
                "decode_termination",
                "branch_selection_admitted",
                "decode_incumbent_policy",
                "episode_receipt_sha256",
                "episode_receipt",
            }
            or row.get("task_id") != task_id
            or row.get("depth") != depth
            or not _is_sha256(row.get("response_sha256"))
            or not isinstance(row.get("response_text"), str)
            or len(row["response_text"]) > _MAX_RESPONSE_CHARS
            or not _is_sha256(row.get("tokens_sha256"))
            or not isinstance(row.get("tokens"), list)
            or len(row["tokens"]) > _MAX_RESPONSE_TOKENS
            or any(type(token) is not int or token < 0 for token in row["tokens"])
            or type(row.get("token_count")) is not int
            or row["token_count"] < 0
            or row["token_count"] != len(row["tokens"])
            or row["response_sha256"]
            != hashlib.sha256(row["response_text"].encode("utf-8")).hexdigest()
            or row["tokens_sha256"] != _sha(row["tokens"])
            or type(row.get("correct")) is not bool
            or not isinstance(row.get("grade_receipt"), dict)
            or type(row["grade_receipt"].get("correct")) is not bool
            or row["grade_receipt"]["correct"] is not row["correct"]
            or type(row.get("episode_ok")) is not bool
            or not isinstance(row.get("episode_reason"), str)
            or not isinstance(row.get("decode_termination"), str)
            or not row["decode_termination"]
            or type(row.get("branch_selection_admitted")) is not bool
            or row.get("decode_incumbent_policy") != "latent"
            or not _is_sha256(row.get("episode_receipt_sha256"))
            or (row["correct"] and not row["episode_ok"])
            or (row["correct"] and not row["branch_selection_admitted"])
        ):
            _fail("recurrent_checkpoint_report_record_invalid")
        row["episode_receipt"] = _validate_episode_evidence(
            row.get("episode_receipt"),
            depth=depth,
            token_count=row["token_count"],
            decode_termination=row["decode_termination"],
            branch_selection_admitted=row["branch_selection_admitted"],
            episode_receipt_sha256=row["episode_receipt_sha256"],
        )
        normalized_rows.append(row)
    correct_by_depth = {
        str(depth): sum(int(row["correct"]) for row in normalized_rows if row["depth"] == depth)
        for depth in normalized_depths
    }
    body = {
        "schema": FREE_GENERATION_REPORT_SCHEMA,
        "arm": arm,
        "adapter_sha256": adapter_sha256,
        "execution_spec_sha256": execution_spec_sha256,
        "task_manifest_sha256": task_manifest_sha256,
        "task_ids": list(normalized_tasks),
        "depths": list(normalized_depths),
        "records": normalized_rows,
        "correct_by_depth": correct_by_depth,
        "total_correct": sum(correct_by_depth.values()),
        "total_observations": len(normalized_rows),
    }
    return {**body, "report_sha256": _sha(body)}


def validate_free_generation_report(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("recurrent_checkpoint_report_invalid")
    report = dict(value)
    body = {key: item for key, item in report.items() if key != "report_sha256"}
    if (
        set(report)
        != {
            "schema",
            "arm",
            "adapter_sha256",
            "execution_spec_sha256",
            "task_manifest_sha256",
            "task_ids",
            "depths",
            "records",
            "correct_by_depth",
            "total_correct",
            "total_observations",
            "report_sha256",
        }
        or report.get("schema") != FREE_GENERATION_REPORT_SCHEMA
        or report.get("report_sha256") != _sha(body)
    ):
        _fail("recurrent_checkpoint_report_commitment_invalid")
    rebuilt = build_free_generation_report(
        arm=report["arm"],
        adapter_sha256=report["adapter_sha256"],
        execution_spec_sha256=report["execution_spec_sha256"],
        task_manifest_sha256=report["task_manifest_sha256"],
        task_ids=report["task_ids"],
        depths=report["depths"],
        records=report["records"],
    )
    if rebuilt != report:
        _fail("recurrent_checkpoint_report_replay_mismatch")
    return report


def validate_recurrence_task_free_generation_report(
    value: Mapping[str, Any],
    *,
    task_manifest: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay task generation and independently grade every exact response."""

    from core.learning.recurrence_curriculum import TASK_GENERATORS

    report = validate_free_generation_report(value)
    manifest = validate_recurrence_task_manifest(task_manifest)
    if report["task_manifest_sha256"] != _sha(manifest) or report["task_ids"] != [
        row["task_id"] for row in manifest
    ]:
        _fail("recurrent_checkpoint_report_task_manifest_mismatch")
    tasks = {
        row["task_id"]: TASK_GENERATORS[row["family"]](row["depth"], row["seed"])
        for row in manifest
    }
    for row in report["records"]:
        response = row["response_text"] if row["episode_ok"] else ""
        grade = dict(tasks[row["task_id"]].grade(response))
        grade["correct"] = bool(row["episode_ok"] and grade.get("correct"))
        if row["grade_receipt"] != grade or row["correct"] is not grade["correct"]:
            _fail("recurrent_checkpoint_report_independent_grade_mismatch")
    return report


_ANSWER_MARKER: Final = "FINAL_ANSWER:"


def _is_answer_only(text: object) -> bool:
    """True when a response states an answer without doing any work first.

    Structural, not a length threshold: everything before the answer marker is
    the reasoning, and a response that has none of it answered without working.
    A response with no marker at all is a different failure (it is graded
    incorrect on its own terms) and is not counted here.
    """

    if not isinstance(text, str):
        return False
    index = text.find(_ANSWER_MARKER)
    if index < 0:
        return False
    return not text[:index].strip()


def _answer_only_count(records: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for row in records if _is_answer_only(row.get("response_text")))


def build_checkpoint_behavioral_admission(
    *,
    initial_report: Mapping[str, Any],
    trained_report: Mapping[str, Any],
    task_manifest: Sequence[Mapping[str, Any]],
    ordinary_decode_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require strict held-out gain over training, depth, and ordinary decode.

    ``ordinary_decode_report`` is the vanilla control on the same frozen
    weights and the same tasks. It is not optional in effect: an admission
    built without it is sealed unadmitted, because improving on an untrained
    adapter proves nothing about whether the recurrent path is worth running
    at all. Historical v1 receipts stay replayable under their own schema.
    """

    initial = validate_recurrence_task_free_generation_report(
        initial_report,
        task_manifest=task_manifest,
    )
    trained = validate_recurrence_task_free_generation_report(
        trained_report,
        task_manifest=task_manifest,
    )
    ordinary: dict[str, Any] | None = None
    if ordinary_decode_report is not None:
        ordinary = validate_recurrence_task_free_generation_report(
            ordinary_decode_report,
            task_manifest=task_manifest,
        )
        if (
            ordinary["arm"] != "ordinary_decode"
            or ordinary["task_manifest_sha256"] != trained["task_manifest_sha256"]
            or ordinary["task_ids"] != trained["task_ids"]
        ):
            _fail("recurrent_checkpoint_admission_ordinary_control_invalid")
    if (
        initial["arm"] != "initial_adapter"
        or trained["arm"] != "trained_adapter"
        or initial["adapter_sha256"] == trained["adapter_sha256"]
        or initial["execution_spec_sha256"] != trained["execution_spec_sha256"]
        or initial["task_manifest_sha256"] != trained["task_manifest_sha256"]
        or initial["task_ids"] != trained["task_ids"]
        or initial["depths"] != trained["depths"]
    ):
        _fail("recurrent_checkpoint_admission_pair_invalid")
    initial_rows = {(row["task_id"], row["depth"]): row for row in initial["records"]}
    trained_rows = {(row["task_id"], row["depth"]): row for row in trained["records"]}
    shallow = initial["depths"][0]
    deep = initial["depths"][-1]
    initial_depth_delta = 0
    trained_depth_delta = 0
    trained_depth_regressions = 0
    for task_id in initial["task_ids"]:
        initial_shallow = int(initial_rows[(task_id, shallow)]["correct"])
        initial_deep = int(initial_rows[(task_id, deep)]["correct"])
        trained_shallow = int(trained_rows[(task_id, shallow)]["correct"])
        trained_deep = int(trained_rows[(task_id, deep)]["correct"])
        initial_depth_delta += initial_deep - initial_shallow
        trained_depth_delta += trained_deep - trained_shallow
        trained_depth_regressions += int(trained_deep < trained_shallow)
    aggregate_gain = trained["total_correct"] - initial["total_correct"]
    depth_interaction = trained_depth_delta - initial_depth_delta
    ordinary_correct = None if ordinary is None else int(ordinary["total_correct"])
    trained_answer_only = _answer_only_count(trained["records"])
    ordinary_answer_only = None if ordinary is None else _answer_only_count(ordinary["records"])
    gates = {
        "complete_episode_execution": all(
            row["episode_ok"] and row["branch_selection_admitted"] for row in trained["records"]
        ),
        "strict_heldout_free_generation_gain": aggregate_gain > 0,
        "positive_training_by_depth_interaction": depth_interaction > 0,
        "no_trained_depth_regressions": trained_depth_regressions == 0,
        # The floor. A checkpoint that answers fewer held-out questions than
        # the same weights answering ordinarily has not earned the recurrent
        # path, however much it improved on its own untrained starting point.
        "beats_ordinary_decode": (
            ordinary_correct is not None
            and int(trained["total_correct"]) > ordinary_correct
        ),
        # Answer-span supervision pays a model to stop reasoning: emitting the
        # answer immediately is the cheapest way to make it predictable. The
        # cp796 and role-v6 runs did exactly that -- median generated tokens
        # fell to 28 against 452 for the untrained path -- while validation
        # cross-entropy fell smoothly and monotonically the entire way. No
        # teacher-forced loss can see this, so it is gated on behavior: the
        # trained arm may not answer without working more often than the
        # ordinary path does.
        "no_answer_only_collapse": (
            ordinary_answer_only is not None
            and trained_answer_only <= ordinary_answer_only
        ),
    }
    decision = (
        "reject_no_ordinary_decode_control"
        if ordinary is None
        else (
            "admit_bounded_next_scale_proxy"
            if all(gates.values())
            else "reject_checkpoint_behavioral_gain_unproven"
        )
    )
    body = {
        "schema": CHECKPOINT_ADMISSION_SCHEMA_V2,
        "ordinary_decode_report_sha256": (
            None if ordinary is None else ordinary["report_sha256"]
        ),
        "ordinary_decode_correct": ordinary_correct,
        "trained_correct": int(trained["total_correct"]),
        "trained_answer_only_responses": trained_answer_only,
        "ordinary_answer_only_responses": ordinary_answer_only,
        "decision": decision,
        "initial_report_sha256": initial["report_sha256"],
        "trained_report_sha256": trained["report_sha256"],
        "execution_spec_sha256": initial["execution_spec_sha256"],
        "task_ids_sha256": _sha(initial["task_ids"]),
        "task_manifest_sha256": initial["task_manifest_sha256"],
        "depths": list(initial["depths"]),
        "aggregate_correct_gain": aggregate_gain,
        "initial_depth_delta": initial_depth_delta,
        "trained_depth_delta": trained_depth_delta,
        "training_by_depth_interaction": depth_interaction,
        "trained_depth_regressions": trained_depth_regressions,
        "gates": gates,
        "admitted": ordinary is not None and all(gates.values()),
        "claim_flags": {
            "resident_32b_gain_proven": False,
            "frontier_level_proven": False,
            "fusion_allowed": False,
            "production_activation_allowed": False,
        },
    }
    return {**body, "admission_sha256": _sha(body)}


def validate_checkpoint_behavioral_admission(
    value: Mapping[str, Any],
    *,
    initial_report: Mapping[str, Any],
    trained_report: Mapping[str, Any],
    task_manifest: Sequence[Mapping[str, Any]],
    ordinary_decode_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("recurrent_checkpoint_admission_invalid")
    expected = build_checkpoint_behavioral_admission(
        initial_report=initial_report,
        trained_report=trained_report,
        task_manifest=task_manifest,
        ordinary_decode_report=ordinary_decode_report,
    )
    if dict(value) != expected:
        _fail("recurrent_checkpoint_admission_replay_mismatch")
    return expected


__all__ = [
    "CHECKPOINT_ADMISSION_SCHEMA",
    "CHECKPOINT_ADMISSION_SCHEMA_V2",
    "FREE_GENERATION_REPORT_SCHEMA",
    "RecurrentCheckpointAdmissionError",
    "build_checkpoint_behavioral_admission",
    "build_free_generation_report",
    "build_recurrence_task_manifest",
    "validate_checkpoint_behavioral_admission",
    "validate_free_generation_report",
    "validate_recurrence_task_free_generation_report",
    "validate_recurrence_task_manifest",
]
