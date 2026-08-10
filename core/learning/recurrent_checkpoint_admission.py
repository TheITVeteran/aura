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
FULL_ENGINE_ADMISSION_SCHEMA: Final = (
    "aura.rlc.recurrent_full_engine_behavioral_admission.v1"
)
# ``ordinary_decode`` is the vanilla control: the same frozen weights answering
# without the recurrent path at all. Without it an admission can only say a
# trained adapter beat an untrained adapter on the same degraded path, which is
# exactly what the 2026-08-06 campaign proved is not enough -- adapter+RLC
# scored 3/28 while ordinary decode on identical weights scored 13/28.
_ARMS: Final = frozenset(
    {
        "initial_adapter",
        "trained_adapter",
        "trained_adapter_lesion",
        "trained_adapter_sham",
        "trained_coda_lesion",
        "trained_coda_sham",
        "ordinary_decode",
        "full_engine",
    }
)
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


def _json_native(value: Any) -> Any:
    """Normalize a validated value to the structure JSON will persist."""

    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


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
    decode_incumbent_policy: str = "latent",
    allow_coda_adapter: bool = False,
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
    coda_activation = receipt.get("coda_adapter")
    nonparametric = receipt.get("nonparametric_memory")
    honest_flags = receipt.get("honest_flags")
    selected_branch = receipt.get("selected_branch")
    n_branches = receipt.get("n_branches")
    recurrence_active = bool(
        isinstance(activation, Mapping)
        and activation.get("schema") == "aura.recurrence_adapter_activation.v1"
        and activation.get("scope") == "latent_slots_only"
        and activation.get("active") is True
        and type(activation.get("calls")) is int
        and activation["calls"] >= 1
        and type(activation.get("adapted_positions")) is int
        and activation["adapted_positions"] >= 1
    )
    coda_active = bool(
        allow_coda_adapter
        and isinstance(coda_activation, Mapping)
        and coda_activation.get("schema") == "aura.coda_adapter_activation.v1"
        and coda_activation.get("scope") == "rlc_coda_only"
        and coda_activation.get("active") is True
        and type(coda_activation.get("calls")) is int
        and coda_activation["calls"] >= 1
        and type(coda_activation.get("adapted_positions")) is int
        and coda_activation["adapted_positions"] >= 1
        and type(coda_activation.get("observed_positions")) is int
        and coda_activation["observed_positions"]
        >= coda_activation["adapted_positions"]
        and isinstance(coda_activation.get("applied_blocks"), Mapping)
        and bool(coda_activation["applied_blocks"])
        and all(
            isinstance(block, str)
            and block.isdigit()
            and type(count) is int
            and count >= 1
            for block, count in coda_activation["applied_blocks"].items()
        )
        and isinstance(coda_activation.get("applied_sites"), Mapping)
        and bool(coda_activation["applied_sites"])
        and all(
            isinstance(site, str)
            and bool(site)
            and type(count) is int
            and count >= 1
            for site, count in coda_activation["applied_sites"].items()
        )
    )
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
        or receipt.get("decode_incumbent_policy") != decode_incumbent_policy
        or receipt.get("decode_termination") != decode_termination
        or type(receipt.get("decode_generated_tokens")) is not int
        or receipt["decode_generated_tokens"] != token_count
        or receipt.get("params_unchanged") is not True
        or not isinstance(nonparametric, Mapping)
        or nonparametric.get("status") != "disabled_by_policy"
        or not isinstance(honest_flags, list)
        or any(str(flag).startswith("fallback_") for flag in honest_flags)
        or not (recurrence_active or coda_active)
    ):
        _fail("recurrent_checkpoint_episode_evidence_invalid")
    return receipt


def _validate_full_engine_episode_evidence(
    value: Any,
    *,
    depth: int,
    response_sha256: str,
    tokens_sha256: str,
    token_count: int,
    decode_termination: str,
    branch_selection_admitted: bool,
    episode_receipt_sha256: str,
) -> dict[str, Any]:
    """Validate recurrence plus the exact incumbent/replacement output floor."""

    receipt = _validate_episode_evidence(
        value,
        depth=depth,
        token_count=token_count,
        decode_termination=decode_termination,
        branch_selection_admitted=branch_selection_admitted,
        episode_receipt_sha256=episode_receipt_sha256,
        decode_incumbent_policy="vanilla_incumbent",
        allow_coda_adapter=True,
    )
    incumbent = receipt.get("incumbent_artifact")
    replacement = receipt.get("answer_replacement")
    if not isinstance(incumbent, Mapping) or not isinstance(replacement, Mapping):
        _fail("recurrent_checkpoint_full_engine_evidence_invalid")
    try:
        from core.brain.llm.latent_cortex.incumbent_artifact import (
            validate_incumbent_receipt,
        )

        incumbent = validate_incumbent_receipt(
            incumbent,
            checkpoint_fingerprint=str(receipt.get("checkpoint_fingerprint") or ""),
            checkpoint_fingerprint_method=str(
                receipt.get("checkpoint_fingerprint_method") or ""
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RecurrentCheckpointAdmissionError(
            "recurrent_checkpoint_full_engine_incumbent_invalid"
        ) from exc
    baseline = replacement.get("baseline_decode")
    accepted = replacement.get("accepted_output")
    incumbent_output = incumbent.get("output")
    decision = replacement.get("decision")
    flags = {str(flag) for flag in receipt.get("honest_flags") or []}
    if (
        replacement.get("schema") != "aura.rlc.answer_replacement.v5"
        or replacement.get("authority") != "confidence_bound_answer_replacement"
        or not _is_sha256(replacement.get("receipt_sha256"))
        or replacement["receipt_sha256"]
        != _sha(
            {
                key: item
                for key, item in replacement.items()
                if key != "receipt_sha256"
            }
        )
        or not isinstance(baseline, Mapping)
        or not isinstance(accepted, Mapping)
        or not isinstance(incumbent_output, Mapping)
        or baseline.get("text_sha256") != incumbent_output.get("text_sha256")
        or baseline.get("tokens_sha256") != incumbent_output.get("tokens_sha256")
        or baseline.get("token_count") != incumbent_output.get("token_count")
        or decision not in {"retain", "replace", "abstain"}
    ):
        _fail("recurrent_checkpoint_full_engine_evidence_invalid")
    output_binding = {
        "text_sha256": response_sha256,
        "tokens_sha256": tokens_sha256,
        "token_count": token_count,
    }
    if decision == "replace":
        candidates = replacement.get("candidates")
        selected_request_id = replacement.get("selected_request_id")
        selected_rows = [
            row
            for row in candidates or []
            if isinstance(row, Mapping)
            and row.get("request_id") == selected_request_id
        ]
        source = accepted.get("source")
        objective_solution_valid = True
        if source == "objective_program_solution":
            objective_solution_valid = bool(
                selected_request_id == "objective-program"
                and len(selected_rows) == 1
                and selected_rows[0].get("branch") == -1
                and selected_rows[0].get("transaction_status")
                == "objective_program_solution"
                and _is_sha256(selected_rows[0].get("transaction_sha256"))
                and selected_rows[0].get("required_verifier")
                == "exact_objective_program"
                and selected_rows[0].get("same_verifier_class") is True
                and isinstance(selected_rows[0].get("replacement_quality"), Mapping)
                and selected_rows[0]["replacement_quality"].get("basis")
                == "objective_program_exact_complete"
                and selected_rows[0]["replacement_quality"].get("lower_bound") == 1.0
                and selected_rows[0]["replacement_quality"].get("upper_bound") == 1.0
            )
        if (
            source
            not in {
                "branch_candidate",
                "objective_program_solution",
                "repaired_candidate",
            }
            or accepted.get("binding_status") != "exact_text_token_roundtrip"
            or any(accepted.get(key) != value for key, value in output_binding.items())
            or len(selected_rows) != 1
            or selected_rows[0].get("dominates") is not True
            or not objective_solution_valid
        ):
            _fail("recurrent_checkpoint_full_engine_replacement_invalid")
    elif (
        any(incumbent_output.get(key) != value for key, value in output_binding.items())
        or (
            decision == "retain"
            and (
                accepted.get("source") != "baseline_decode"
                or any(accepted.get(key) != value for key, value in output_binding.items())
            )
        )
        or (
            decision == "abstain"
            and not {
                "confidence_bound_abstention_declined_under_incumbent",
                "answer_replacement_abstention_declined_under_incumbent",
            }
            & flags
        )
    ):
        _fail("recurrent_checkpoint_full_engine_floor_invalid")
    return receipt


def _validate_ordinary_decode_evidence(
    value: Any,
    *,
    task_id: str,
    depth: int,
    response_sha256: str,
    tokens_sha256: str,
    token_count: int,
    decode_termination: str,
    episode_receipt_sha256: str,
) -> dict[str, Any]:
    """Validate a control generation without inventing recurrent evidence."""

    if not isinstance(value, Mapping):
        _fail("recurrent_checkpoint_ordinary_evidence_invalid")
    receipt = dict(value)
    if (
        set(receipt)
        != {
            "schema",
            "arm",
            "task_id",
            "depth_coordinate",
            "generation_seed",
            "recurrent_steps",
            "prompt_tokens_sha256",
            "response_sha256",
            "tokens_sha256",
            "token_count",
            "decode_termination",
        }
        or _sha(receipt) != episode_receipt_sha256
        or receipt.get("schema") != "aura.rlc.ordinary_decode_probe.v2"
        or receipt.get("arm") != "ordinary_decode"
        or receipt.get("task_id") != task_id
        or receipt.get("depth_coordinate") != depth
        or type(receipt.get("generation_seed")) is not int
        or receipt["generation_seed"] < 0
        or receipt.get("recurrent_steps") != 0
        or not _is_sha256(receipt.get("prompt_tokens_sha256"))
        or receipt.get("response_sha256") != response_sha256
        or receipt.get("tokens_sha256") != tokens_sha256
        or receipt.get("token_count") != token_count
        or receipt.get("decode_termination") != decode_termination
    ):
        _fail("recurrent_checkpoint_ordinary_evidence_invalid")
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
    expected_incumbent_policy = (
        "vanilla_incumbent"
        if arm in {"ordinary_decode", "full_engine"}
        else "latent"
    )
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
            or row.get("decode_incumbent_policy") != expected_incumbent_policy
            or not _is_sha256(row.get("episode_receipt_sha256"))
            or (row["correct"] and not row["episode_ok"])
            or (row["correct"] and not row["branch_selection_admitted"])
        ):
            _fail("recurrent_checkpoint_report_record_invalid")
        if arm == "ordinary_decode":
            row["episode_receipt"] = _validate_ordinary_decode_evidence(
                row.get("episode_receipt"),
                task_id=task_id,
                depth=depth,
                response_sha256=row["response_sha256"],
                tokens_sha256=row["tokens_sha256"],
                token_count=row["token_count"],
                decode_termination=row["decode_termination"],
                episode_receipt_sha256=row["episode_receipt_sha256"],
            )
        elif arm == "full_engine":
            row["episode_receipt"] = _validate_full_engine_episode_evidence(
                row.get("episode_receipt"),
                depth=depth,
                response_sha256=row["response_sha256"],
                tokens_sha256=row["tokens_sha256"],
                token_count=row["token_count"],
                decode_termination=row["decode_termination"],
                branch_selection_admitted=row["branch_selection_admitted"],
                episode_receipt_sha256=row["episode_receipt_sha256"],
            )
        else:
            row["episode_receipt"] = _validate_episode_evidence(
                row.get("episode_receipt"),
                depth=depth,
                token_count=row["token_count"],
                decode_termination=row["decode_termination"],
                branch_selection_admitted=row["branch_selection_admitted"],
                episode_receipt_sha256=row["episode_receipt_sha256"],
            )
        # The input commitment is checked above before normalization. Public
        # report evidence then crosses a JSON boundary, so bind the exact
        # structure that survives persistence (not Python-only integer keys or
        # tuples whose canonical ordering can change after reload).
        row["episode_receipt"] = _json_native(row["episode_receipt"])
        row["episode_receipt_sha256"] = _sha(row["episode_receipt"])
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
    body = _json_native(body)
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


def build_full_engine_behavioral_admission(
    *,
    initial_full_engine_report: Mapping[str, Any],
    trained_full_engine_report: Mapping[str, Any],
    initial_ordinary_decode_report: Mapping[str, Any],
    trained_ordinary_decode_report: Mapping[str, Any],
    task_manifest: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Admit only trained, floor-preserving, independently authorized gain."""

    initial = validate_recurrence_task_free_generation_report(
        initial_full_engine_report,
        task_manifest=task_manifest,
    )
    trained = validate_recurrence_task_free_generation_report(
        trained_full_engine_report,
        task_manifest=task_manifest,
    )
    initial_ordinary = validate_recurrence_task_free_generation_report(
        initial_ordinary_decode_report,
        task_manifest=task_manifest,
    )
    trained_ordinary = validate_recurrence_task_free_generation_report(
        trained_ordinary_decode_report,
        task_manifest=task_manifest,
    )
    reports = (initial, trained, initial_ordinary, trained_ordinary)
    if (
        initial["arm"] != "full_engine"
        or trained["arm"] != "full_engine"
        or initial_ordinary["arm"] != "ordinary_decode"
        or trained_ordinary["arm"] != "ordinary_decode"
        or initial["adapter_sha256"] == trained["adapter_sha256"]
        or initial["adapter_sha256"] != initial_ordinary["adapter_sha256"]
        or trained["adapter_sha256"] != trained_ordinary["adapter_sha256"]
        or len({report["execution_spec_sha256"] for report in reports}) != 1
        or len({report["task_manifest_sha256"] for report in reports}) != 1
        or any(report["task_ids"] != initial["task_ids"] for report in reports[1:])
        or any(report["depths"] != initial["depths"] for report in reports[1:])
    ):
        _fail("recurrent_full_engine_admission_pair_invalid")

    def rows(report: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
        return {
            (str(row["task_id"]), int(row["depth"])): dict(row)
            for row in report["records"]
        }

    initial_rows = rows(initial)
    trained_rows = rows(trained)
    initial_ordinary_rows = rows(initial_ordinary)
    trained_ordinary_rows = rows(trained_ordinary)
    coordinates = tuple(initial_rows)
    ordinary_decode_changed = 0
    incumbent_binding_failures = 0
    ordinary_floor_regressions = 0
    training_regressions = 0
    authorized_correct_gains = 0
    trained_depth_regressions = 0
    for coordinate in coordinates:
        before = initial_rows[coordinate]
        after = trained_rows[coordinate]
        ordinary_before = initial_ordinary_rows[coordinate]
        ordinary_after = trained_ordinary_rows[coordinate]
        if (
            ordinary_before["response_sha256"] != ordinary_after["response_sha256"]
            or ordinary_before["tokens_sha256"] != ordinary_after["tokens_sha256"]
            or ordinary_before["correct"] is not ordinary_after["correct"]
        ):
            ordinary_decode_changed += 1
        for full_row, ordinary_row in (
            (before, ordinary_before),
            (after, ordinary_after),
        ):
            incumbent_output = full_row["episode_receipt"]["incumbent_artifact"][
                "output"
            ]
            if (
                incumbent_output.get("text_sha256")
                != ordinary_row["response_sha256"]
                or incumbent_output.get("tokens_sha256")
                != ordinary_row["tokens_sha256"]
                or incumbent_output.get("token_count")
                != ordinary_row["token_count"]
            ):
                incumbent_binding_failures += 1
        ordinary_floor_regressions += int(
            bool(ordinary_after["correct"]) and not bool(after["correct"])
        )
        training_regressions += int(bool(before["correct"]) and not bool(after["correct"]))
        replacement = after["episode_receipt"]["answer_replacement"]
        authorized_correct_gains += int(
            bool(after["correct"])
            and not bool(ordinary_after["correct"])
            and replacement.get("decision") == "replace"
        )
    shallow = int(initial["depths"][0])
    deep = int(initial["depths"][-1])
    for task_id in initial["task_ids"]:
        trained_depth_regressions += int(
            bool(trained_rows[(task_id, shallow)]["correct"])
            and not bool(trained_rows[(task_id, deep)]["correct"])
        )

    trained_answer_only = _answer_only_count(trained["records"])
    ordinary_answer_only = _answer_only_count(trained_ordinary["records"])
    training_gain = int(trained["total_correct"]) - int(initial["total_correct"])
    ordinary_gain = int(trained["total_correct"]) - int(
        trained_ordinary["total_correct"]
    )
    gates = {
        "complete_full_engine_execution": all(
            row["episode_ok"] and row["branch_selection_admitted"]
            for row in trained["records"]
        ),
        "ordinary_decode_stable_across_adapter": ordinary_decode_changed == 0,
        "canonical_incumbent_bound": incumbent_binding_failures == 0,
        "ordinary_correctness_floor_preserved": ordinary_floor_regressions == 0,
        "no_training_correctness_regressions": training_regressions == 0,
        "no_trained_depth_regressions": trained_depth_regressions == 0,
        "strict_gain_over_initial_full_engine": training_gain > 0,
        "strict_gain_over_ordinary_decode": ordinary_gain > 0,
        "independently_authorized_correct_gain": authorized_correct_gains > 0,
        "no_answer_only_collapse": trained_answer_only <= ordinary_answer_only,
    }
    body = {
        "schema": FULL_ENGINE_ADMISSION_SCHEMA,
        "decision": (
            "admit_bounded_complete_engine_proxy"
            if all(gates.values())
            else "reject_complete_engine_gain_unproven"
        ),
        "initial_full_engine_report_sha256": initial["report_sha256"],
        "trained_full_engine_report_sha256": trained["report_sha256"],
        "initial_ordinary_decode_report_sha256": initial_ordinary["report_sha256"],
        "trained_ordinary_decode_report_sha256": trained_ordinary["report_sha256"],
        "execution_spec_sha256": initial["execution_spec_sha256"],
        "task_manifest_sha256": initial["task_manifest_sha256"],
        "task_ids_sha256": _sha(initial["task_ids"]),
        "depths": list(initial["depths"]),
        "initial_full_engine_correct": int(initial["total_correct"]),
        "trained_full_engine_correct": int(trained["total_correct"]),
        "ordinary_decode_correct": int(trained_ordinary["total_correct"]),
        "training_correct_gain": training_gain,
        "ordinary_correct_gain": ordinary_gain,
        "ordinary_decode_changed": ordinary_decode_changed,
        "incumbent_binding_failures": incumbent_binding_failures,
        "ordinary_floor_regressions": ordinary_floor_regressions,
        "training_regressions": training_regressions,
        "trained_depth_regressions": trained_depth_regressions,
        "authorized_correct_gains": authorized_correct_gains,
        "trained_answer_only_responses": trained_answer_only,
        "ordinary_answer_only_responses": ordinary_answer_only,
        "gates": gates,
        "admitted": all(gates.values()),
        "claim_flags": {
            "resident_32b_gain_proven": False,
            "frontier_level_proven": False,
            "fusion_allowed": False,
            "production_activation_allowed": False,
        },
    }
    return {**body, "admission_sha256": _sha(body)}


def validate_full_engine_behavioral_admission(
    value: Mapping[str, Any],
    *,
    initial_full_engine_report: Mapping[str, Any],
    trained_full_engine_report: Mapping[str, Any],
    initial_ordinary_decode_report: Mapping[str, Any],
    trained_ordinary_decode_report: Mapping[str, Any],
    task_manifest: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("recurrent_full_engine_admission_invalid")
    expected = build_full_engine_behavioral_admission(
        initial_full_engine_report=initial_full_engine_report,
        trained_full_engine_report=trained_full_engine_report,
        initial_ordinary_decode_report=initial_ordinary_decode_report,
        trained_ordinary_decode_report=trained_ordinary_decode_report,
        task_manifest=task_manifest,
    )
    if dict(value) != expected:
        _fail("recurrent_full_engine_admission_replay_mismatch")
    return expected


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
    "FULL_ENGINE_ADMISSION_SCHEMA",
    "RecurrentCheckpointAdmissionError",
    "build_checkpoint_behavioral_admission",
    "build_free_generation_report",
    "build_full_engine_behavioral_admission",
    "build_recurrence_task_manifest",
    "validate_checkpoint_behavioral_admission",
    "validate_free_generation_report",
    "validate_full_engine_behavioral_admission",
    "validate_recurrence_task_free_generation_report",
    "validate_recurrence_task_manifest",
]
