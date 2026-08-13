#!/usr/bin/env python3
"""Decode fresh answers from a frozen unified-recurrence checkpoint.

This is deliberately slower than teacher-forced CE evaluation: every token is
generated from the public prefix and the recurrent window is actually executed.
The result therefore measures emitted exact answers rather than readout affinity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402

from core.brain.canonical_json import canonical_json_bytes  # noqa: E402
from core.learning.recurrent_answer_emission import (  # noqa: E402
    tokenizer_answer_emission_contract,
)
from core.learning.recurrent_literal_grounding import (  # noqa: E402
    LiteralObservationContract,
    tokenizer_digit_token_ids,
)
from core.learning.recurrent_opcode_grounding import (  # noqa: E402
    tokenizer_opcode_contract,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    unified_recurrent_logits,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from tools.evaluate_unified_intrinsic_checkpoint import (  # noqa: E402
    _canonical_sha256,
    _evaluation_layout,
    _evaluation_source_sha256s,
    _file_sha256,
    _fresh_tasks,
    campaign_initial_control_binding,
    load_root_initial_controller,
    unified_evaluation_context,
)
from tools.train_intrinsic_recurrence import encode_example  # noqa: E402
from tools.unified_intrinsic_checkpoint import (  # noqa: E402
    resolve_checkpoint_generation,
)
from tools.unified_intrinsic_decode_journal import (  # noqa: E402
    DecodeProgressError,
    DecodeProgressJournal,
)

DECODE_EVALUATION_SCHEMA = "aura.unified_intrinsic_decode_evaluation.v2"
DECODE_CLAIM_BOUNDARY = (
    "compiled arms measure public typed-state execution plus tokenizer-bound "
    "value emission; trained arms constrain only the public output grammar while "
    "neural state selects every digit. Grammar and digit-pointer lesions test "
    "those mechanisms separately. Resident model identity is established only by "
    "the independently verified campaign binding, not this report in isolation. "
    "This is not a broad reasoning, frontier, fusion, or WOW result"
)


def _logits(value: Any) -> Any:
    if hasattr(value, "logits"):
        return value.logits
    if isinstance(value, tuple):
        return value[0]
    return value


def _greedy_decode(
    tokenizer: Any,
    prompt: Any,
    logits_fn: Callable[[Any], Any],
    *,
    max_tokens: int,
) -> tuple[str, list[int], bool]:
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("decode max_tokens must be positive")
    tokens = prompt
    generated: list[int] = []
    eos = tokenizer.eos_token_id
    stopped = False
    for _index in range(max_tokens):
        logits = logits_fn(tokens)
        token = int(mx.argmax(logits[0, -1]).item())
        if eos is not None and token == eos:
            stopped = True
            break
        generated.append(token)
        tokens = mx.concatenate(
            [tokens, mx.array([[token]], dtype=tokens.dtype)],
            axis=1,
        )
    return tokenizer.decode(generated), generated, stopped


def _candidate_response(bridge: str, decoded: str) -> str:
    if not bridge.strip().endswith("FINAL_ANSWER:"):
        raise ValueError("decoded proof requires the exact answer bridge")
    return f"{bridge}{decoded}".strip()


def _force_next_token(logits: Any, token_id: int) -> Any:
    if type(token_id) is not int or not 0 <= token_id < int(logits.shape[-1]):
        raise ValueError("compiled answer token is outside the model vocabulary")
    vocabulary = mx.arange(int(logits.shape[-1]))[None, :]
    forced = mx.where(vocabulary == token_id, 0.0, -1e9).astype(logits.dtype)
    forced = mx.broadcast_to(forced, logits[:, -1, :].shape)
    return mx.concatenate([logits[:, :-1, :], forced[:, None, :]], axis=1)


def _paired_training_effects(
    candidates: list[dict[str, Any]],
    recurrence_depths: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    """Summarize trained versus initialization-matched recurrent execution."""

    effects: dict[str, dict[str, Any]] = {}
    for depth in (1, *recurrence_depths):
        control_arm = f"untrained_t{depth}"
        trained_arm = f"trained_t{depth}"
        by_task: dict[str, dict[str, bool]] = {}
        for row in candidates:
            arm = row.get("arm")
            if arm not in (control_arm, trained_arm):
                continue
            task_id = str(row["task_id"])
            bucket = by_task.setdefault(task_id, {})
            if arm in bucket:
                raise RuntimeError("paired recurrent control contains a duplicate task arm")
            bucket[str(arm)] = bool(row["correct"])
        if not by_task or any(
            set(result) != {control_arm, trained_arm} for result in by_task.values()
        ):
            raise RuntimeError("paired recurrent control is incomplete")
        wrong_to_right = sum(
            not result[control_arm] and result[trained_arm] for result in by_task.values()
        )
        right_to_wrong = sum(
            result[control_arm] and not result[trained_arm] for result in by_task.values()
        )
        control_correct = sum(result[control_arm] for result in by_task.values())
        trained_correct = sum(result[trained_arm] for result in by_task.values())
        effects[str(depth)] = {
            "tasks": len(by_task),
            "control_arm": control_arm,
            "trained_arm": trained_arm,
            "untrained_correct": control_correct,
            "trained_correct": trained_correct,
            "net_correct_gain": trained_correct - control_correct,
            "wrong_to_right": wrong_to_right,
            "right_to_wrong": right_to_wrong,
        }
    return effects


def _decoded_arm_names(decoded_depths: tuple[int, ...]) -> tuple[str, ...]:
    lesion_depth = max(decoded_depths)
    return (
        "base_t1",
        "untrained_t1",
        "trained_t1",
        *(f"untrained_t{depth}" for depth in decoded_depths),
        *(f"trained_t{depth}" for depth in decoded_depths),
        f"grammar_lesion_t{lesion_depth}",
        f"pointer_lesion_t{lesion_depth}",
        *(f"compiled_t{depth}" for depth in decoded_depths),
    )


def _evaluate_decoding_loaded(
    campaign_dir: Path,
    *,
    bundle: Any,
    initial_controller: Any,
    tokenizer: Any,
    spec: Any,
    identity: dict[str, Any],
    control_binding: dict[str, Any],
    envelope: Any,
    stem: str,
    per_cell: int,
    evaluation_seed: int,
    max_tokens: int,
    task_depths: tuple[int, ...] | None = None,
    recurrence_depths: tuple[int, ...] | None = None,
    progress_dir: Path | None = None,
) -> dict[str, Any]:
    if task_depths is not None and (
        not task_depths
        or len(task_depths) != len(set(task_depths))
        or any(type(depth) is not int or depth < 1 for depth in task_depths)
    ):
        raise ValueError("decode task depths must be unique positive integers")
    if recurrence_depths is not None and (
        not recurrence_depths
        or len(recurrence_depths) != len(set(recurrence_depths))
        or any(type(depth) is not int or depth < 2 for depth in recurrence_depths)
    ):
        raise ValueError("decode recurrence depths must be unique non-anchor campaign depths")
    depths = task_depths or tuple(
        int(value) for value in identity.get("task_depths", (identity.get("task_depth"),))
    )
    decoded_depths = recurrence_depths or (max(spec.heldout_depths),)
    if any(depth not in spec.depths for depth in decoded_depths):
        raise ValueError("decode recurrence depths must be unique non-anchor campaign depths")
    tasks = [
        task
        for depth in depths
        for task in _fresh_tasks(
            identity,
            per_cell=per_cell,
            seed=evaluation_seed + depth * 1_000_003,
            task_depth=depth,
        )
    ]
    layout = _evaluation_layout(campaign_dir)
    resolved = resolve_checkpoint_generation(
        layout.checkpoint_dir,
        stem=stem,
        required=True,
    )
    if resolved is None:  # pragma: no cover - required=True is exhaustive
        raise RuntimeError("unified checkpoint is unavailable")
    checkpoint_sha256 = _file_sha256(resolved.weights_path)
    source_sha256s = _evaluation_source_sha256s()
    arm_names = _decoded_arm_names(decoded_depths)
    task_identities = [
        {
            "task_id": task.task_id,
            "family": task.family,
            "task_depth": task.depth,
            "prompt_sha256": hashlib.sha256(task.prompt.encode()).hexdigest(),
            "expected_sha256": hashlib.sha256(task.answer.encode()).hexdigest(),
        }
        for task in tasks
    ]
    experiment = {
        "schema": "aura.unified_intrinsic.decode_experiment.v1",
        "campaign_identity_sha256": identity["identity_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "matched_control": control_binding,
        "evaluation_source_sha256s": source_sha256s,
        "stem": stem,
        "per_cell": per_cell,
        "evaluation_seed": evaluation_seed,
        "max_tokens": max_tokens,
        "task_depths": list(depths),
        "recurrence_depths": list(decoded_depths),
        "tasks": task_identities,
        "arms": list(arm_names),
    }
    progress = DecodeProgressJournal(progress_dir, experiment) if progress_dir is not None else None
    total_candidates = len(tasks) * len(arm_names)
    resumed_candidates = 0
    bridge = {"assistant_answer": "\n\nFINAL_ANSWER: "}.get(
        identity["bridge"],
        identity["bridge"],
    )
    t1 = spec.plan_at(1)
    answer_contract = tokenizer_answer_emission_contract(
        tokenizer,
        tokenizer_opcode_contract(tokenizer),
    )
    from core.brain.llm.latent_cortex.recurrence_adapter import (
        recurrence_adapter_scope,
    )

    candidates: list[dict[str, Any]] = []
    sequence = 0
    for task in tasks:
        prompt, _answer = encode_example(tokenizer, task, bridge)
        prompt_sha256 = hashlib.sha256(task.prompt.encode()).hexdigest()
        compiled_cache: dict[int, tuple[tuple[int, ...], Any, int]] = {}

        def base_logits(tokens: Any) -> Any:
            return _logits(bundle.model(tokens))

        def recurrent_logits(
            tokens: Any,
            plan: Any,
            *,
            controller: Any = bundle.controller,
            grammar_enabled: bool = True,
            pointer_enabled: bool = True,
            state_slot_start: int = int(prompt.shape[-1]),
        ) -> Any:
            logits, _telemetry = unified_recurrent_logits(
                bundle.model,
                tokens,
                plan,
                controller,
                state_slot_start=state_slot_start,
                answer_emission_contract=(answer_contract if grammar_enabled else None),
                answer_digit_pointer_enabled=pointer_enabled,
            )
            return logits

        def compiled_logits(
            tokens: Any,
            plan: Any,
            state_slot_start: int = int(prompt.shape[-1]),
            cache: dict[int, tuple[tuple[int, ...], Any, int]] = compiled_cache,
        ) -> Any:
            generated = tuple(int(value) for value in tokens[0, state_slot_start:].tolist())
            cached = cache.get(plan.iterations)
            if cached is not None:
                expected, dtype, vocabulary_size = cached
                if len(generated) >= len(expected) or generated != expected[: len(generated)]:
                    raise RuntimeError(
                        "compiled recurrent answer diverged from its terminal program"
                    )
                shell = mx.zeros((1, 1, vocabulary_size), dtype=dtype)
                return _force_next_token(shell, expected[len(generated)])

            state_probabilities: list[Any] = []
            logits, _telemetry = unified_recurrent_logits(
                bundle.model,
                tokens,
                plan,
                bundle.controller,
                state_slot_start=state_slot_start,
                state_probability_trajectory=state_probabilities,
            )
            if not state_probabilities:
                raise RuntimeError("compiled recurrent answer produced no typed state")
            state_values = tuple(
                int(value) for value in mx.argmax(state_probabilities[-1], axis=-1).tolist()[0]
            )
            public_tokens = tuple(int(value) for value in tokens[0, :state_slot_start].tolist())
            expected = answer_contract.expected_tokens(public_tokens, state_values)
            if expected is None:
                raise RuntimeError("compiled recurrent answer did not reach terminal state")
            if generated:
                raise RuntimeError("compiled recurrent answer cache missed after emission began")
            cache[plan.iterations] = (
                expected,
                logits.dtype,
                int(logits.shape[-1]),
            )
            return _force_next_token(logits, expected[len(generated)])

        lesion_depth = max(decoded_depths)
        arms: tuple[tuple[str, Callable[[Any], Any]], ...] = (
            ("base_t1", base_logits),
            (
                "untrained_t1",
                lambda tokens: recurrent_logits(tokens, t1, controller=initial_controller),
            ),
            ("trained_t1", lambda tokens: recurrent_logits(tokens, t1)),
            *(
                (
                    f"untrained_t{depth}",
                    lambda tokens, depth=depth: recurrent_logits(
                        tokens,
                        spec.plan_at(depth),
                        controller=initial_controller,
                    ),
                )
                for depth in decoded_depths
            ),
            *(
                (
                    f"trained_t{depth}",
                    lambda tokens, depth=depth: recurrent_logits(
                        tokens,
                        spec.plan_at(depth),
                    ),
                )
                for depth in decoded_depths
            ),
            (
                f"grammar_lesion_t{lesion_depth}",
                lambda tokens, depth=lesion_depth: recurrent_logits(
                    tokens,
                    spec.plan_at(depth),
                    grammar_enabled=False,
                ),
            ),
            (
                f"pointer_lesion_t{lesion_depth}",
                lambda tokens, depth=lesion_depth: recurrent_logits(
                    tokens,
                    spec.plan_at(depth),
                    pointer_enabled=False,
                ),
            ),
            *(
                (
                    f"compiled_t{depth}",
                    lambda tokens, depth=depth: compiled_logits(
                        tokens,
                        spec.plan_at(depth),
                    ),
                )
                for depth in decoded_depths
            ),
        )
        for arm, operation in arms:
            if tuple(name for name, _operation in arms) != arm_names:
                raise RuntimeError("decoded arm construction differs from experiment identity")
            resumed = (
                progress.load(
                    sequence,
                    task_id=task.task_id,
                    arm=arm,
                    prompt_sha256=prompt_sha256,
                )
                if progress is not None
                else None
            )
            if resumed is not None:
                if (
                    resumed.get("family") != task.family
                    or resumed.get("task_depth") != task.depth
                    or resumed.get("expected") != task.answer
                ):
                    raise DecodeProgressError("decode progress candidate task contract differs")
                candidates.append(resumed)
                resumed_candidates += 1
                sequence += 1
                print(
                    f"[decode] {sequence}/{total_candidates} "
                    f"task={task.task_id} arm={arm} resumed=true",
                    flush=True,
                )
                continue
            if arm == "base_t1":
                decoded, token_ids, stopped = _greedy_decode(
                    tokenizer,
                    prompt,
                    operation,
                    max_tokens=max_tokens,
                )
            else:
                with recurrence_adapter_scope(start=None, stop=None):
                    decoded, token_ids, stopped = _greedy_decode(
                        tokenizer,
                        prompt,
                        operation,
                        max_tokens=max_tokens,
                    )
            response = _candidate_response(bridge, decoded)
            candidate = {
                "task_id": task.task_id,
                "family": task.family,
                "task_depth": task.depth,
                "prompt_sha256": prompt_sha256,
                "arm": arm,
                "decoded": decoded,
                "expected": task.answer,
                "token_ids": token_ids,
                "stopped_on_eos": stopped,
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            }
            if progress is not None:
                progress.commit(sequence, candidate)
            candidates.append(candidate)
            sequence += 1
            print(
                f"[decode] {sequence}/{total_candidates} "
                f"task={task.task_id} arm={arm} tokens={len(token_ids)}",
                flush=True,
            )
        envelope.reclaim(force=True)

    # Grade only after every arm has emitted a committed candidate in memory.
    task_by_id = {task.task_id: task for task in tasks}
    for row in candidates:
        response = _candidate_response(bridge, row["decoded"])
        grade = task_by_id[row["task_id"]].grade(response)
        row["correct"] = bool(grade["correct"])
        row["grade_reason"] = grade.get("reason")

    arm_results: dict[str, dict[str, Any]] = {}
    for arm in dict.fromkeys(row["arm"] for row in candidates):
        selected = [row for row in candidates if row["arm"] == arm]
        arm_results[arm] = {
            "correct": sum(row["correct"] for row in selected),
            "tasks": len(selected),
            "accuracy": sum(row["correct"] for row in selected) / len(selected),
            "eos_stops": sum(row["stopped_on_eos"] for row in selected),
        }
    depth_results: dict[str, dict[str, dict[str, Any]]] = {}
    for task_depth in depths:
        depth_results[str(task_depth)] = {}
        for arm in arm_results:
            selected = [
                row for row in candidates if row["arm"] == arm and row["task_depth"] == task_depth
            ]
            correct = sum(row["correct"] for row in selected)
            depth_results[str(task_depth)][arm] = {
                "correct": correct,
                "tasks": len(selected),
                "accuracy": correct / len(selected),
                "eos_stops": sum(row["stopped_on_eos"] for row in selected),
            }
    body = {
        "schema": DECODE_EVALUATION_SCHEMA,
        "campaign_identity_sha256": identity["identity_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "matched_control": control_binding,
        "evaluation_source_sha256s": source_sha256s,
        "evaluation_seed": evaluation_seed,
        "per_cell": per_cell,
        "task_depths": list(depths),
        "task_count": len(tasks),
        "recurrence_depths": list(decoded_depths),
        "lesion_depth": max(decoded_depths),
        "max_tokens": max_tokens,
        "arm_results": arm_results,
        "paired_training_effects": _paired_training_effects(
            candidates,
            decoded_depths,
        ),
        "depth_results": depth_results,
        "candidates": candidates,
        "decode_progress": {
            "enabled": progress is not None,
            "experiment_sha256": (progress.experiment_sha256 if progress is not None else None),
            "candidates_committed": len(candidates),
            "candidates_resumed": resumed_candidates,
        },
        "claim_boundary": DECODE_CLAIM_BOUNDARY,
    }
    return {**body, "report_sha256": _canonical_sha256(body)}


def evaluate_decoding(
    campaign_dir: Path,
    *,
    stem: str,
    per_cell: int,
    evaluation_seed: int,
    max_tokens: int,
    task_depths: tuple[int, ...] | None = None,
    recurrence_depths: tuple[int, ...] | None = None,
    memory_limit_gb: float = 40.0,
    cache_limit_gb: float = 2.0,
    wired_limit_gb: float = 48.0,
    resource_stage_path: Path | None = None,
    resource_startup_lethal_mb: float | None = None,
    resource_steady_lethal_mb: float | None = None,
    preload_ready_path: Path | None = None,
    preload_release_path: Path | None = None,
    preload_key_path: Path | None = None,
    preload_config_sha256: str | None = None,
    progress_dir: Path | None = None,
    matched_control_campaign: Path | None = None,
    matched_control_stem: str = "checkpoint_latest",
) -> dict[str, Any]:
    if task_depths is not None and (
        not task_depths
        or len(task_depths) != len(set(task_depths))
        or any(type(depth) is not int or depth < 1 for depth in task_depths)
    ):
        raise ValueError("decode task depths must be unique positive integers")
    if recurrence_depths is not None and (
        not recurrence_depths
        or len(recurrence_depths) != len(set(recurrence_depths))
        or any(type(depth) is not int or depth < 2 for depth in recurrence_depths)
    ):
        raise ValueError("decode recurrence depths must be unique non-anchor campaign depths")
    with unified_evaluation_context(
        campaign_dir,
        stem=stem,
        memory_limit_gb=memory_limit_gb,
        cache_limit_gb=cache_limit_gb,
        wired_limit_gb=wired_limit_gb,
        resource_stage_path=resource_stage_path,
        resource_startup_lethal_mb=resource_startup_lethal_mb,
        resource_steady_lethal_mb=resource_steady_lethal_mb,
        preload_ready_path=preload_ready_path,
        preload_release_path=preload_release_path,
        preload_key_path=preload_key_path,
        preload_config_sha256=preload_config_sha256,
    ) as loaded:
        (
            bundle,
            initial_controller,
            tokenizer,
            spec,
            identity,
            envelope,
            resource_receipt,
        ) = loaded
        control_binding = campaign_initial_control_binding(
            campaign_dir,
            identity=identity,
        )
        if matched_control_campaign is not None:
            initial_controller, control_binding = load_root_initial_controller(
                matched_control_campaign,
                stem=matched_control_stem,
                model=bundle.model,
                tokenizer=tokenizer,
                spec=spec,
                target_identity=identity,
                literal_contract=LiteralObservationContract(tokenizer_digit_token_ids(tokenizer)),
                opcode_contract=tokenizer_opcode_contract(tokenizer),
            )
        report = _evaluate_decoding_loaded(
            campaign_dir,
            bundle=bundle,
            initial_controller=initial_controller,
            tokenizer=tokenizer,
            spec=spec,
            identity=identity,
            control_binding=control_binding,
            envelope=envelope,
            stem=stem,
            per_cell=per_cell,
            evaluation_seed=evaluation_seed,
            max_tokens=max_tokens,
            task_depths=task_depths,
            recurrence_depths=recurrence_depths,
            progress_dir=progress_dir,
        )
        body = {
            **{key: value for key, value in report.items() if key != "report_sha256"},
            "resource_envelope": envelope.to_receipt(),
            "resource_guard": resource_receipt,
        }
        return {**body, "report_sha256": _canonical_sha256(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--stem", default="checkpoint_best_heldout")
    parser.add_argument("--per-cell", type=int, default=1)
    parser.add_argument("--evaluation-seed", type=int, default=20260810203)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--task-depths",
        help="comma-separated task difficulties; defaults to campaign depth",
    )
    parser.add_argument(
        "--recurrence-depths",
        help="comma-separated trained/heldout recurrence depths; defaults to deepest heldout",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--memory-limit-gb", type=float, default=40.0)
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    parser.add_argument("--wired-limit-gb", type=float, default=48.0)
    parser.add_argument("--resource-stage-path", type=Path)
    parser.add_argument("--resource-startup-lethal-mb", type=float)
    parser.add_argument("--resource-steady-lethal-mb", type=float)
    parser.add_argument("--preload-ready-path", type=Path)
    parser.add_argument("--preload-release-path", type=Path)
    parser.add_argument("--preload-key-path", type=Path)
    parser.add_argument("--preload-config-sha256")
    parser.add_argument("--progress-dir", type=Path)
    parser.add_argument("--matched-control-campaign", type=Path)
    parser.add_argument("--matched-control-stem", default="checkpoint_latest")
    args = parser.parse_args()
    report = evaluate_decoding(
        args.campaign.expanduser().resolve(strict=True),
        stem=args.stem,
        per_cell=args.per_cell,
        evaluation_seed=args.evaluation_seed,
        max_tokens=args.max_tokens,
        task_depths=(
            tuple(int(value) for value in args.task_depths.split(",")) if args.task_depths else None
        ),
        recurrence_depths=(
            tuple(int(value) for value in args.recurrence_depths.split(","))
            if args.recurrence_depths
            else None
        ),
        memory_limit_gb=args.memory_limit_gb,
        cache_limit_gb=args.cache_limit_gb,
        wired_limit_gb=args.wired_limit_gb,
        resource_stage_path=args.resource_stage_path,
        resource_startup_lethal_mb=args.resource_startup_lethal_mb,
        resource_steady_lethal_mb=args.resource_steady_lethal_mb,
        preload_ready_path=args.preload_ready_path,
        preload_release_path=args.preload_release_path,
        preload_key_path=args.preload_key_path,
        preload_config_sha256=args.preload_config_sha256,
        progress_dir=args.progress_dir,
        matched_control_campaign=args.matched_control_campaign,
        matched_control_stem=args.matched_control_stem,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        target = args.report.expanduser().resolve()
        atomic_write_bytes(target, canonical_json_bytes(report) + b"\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
