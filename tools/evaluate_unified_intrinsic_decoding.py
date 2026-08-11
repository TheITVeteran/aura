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
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402

from core.learning.recurrent_answer_emission import (  # noqa: E402
    tokenizer_answer_emission_contract,
)
from core.learning.recurrent_opcode_grounding import (  # noqa: E402
    tokenizer_opcode_contract,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    unified_recurrent_logits,
)
from tools.evaluate_unified_intrinsic_checkpoint import (  # noqa: E402
    _canonical_sha256,
    _file_sha256,
    _fresh_tasks,
    _load_checkpoint,
)
from tools.train_intrinsic_recurrence import encode_example  # noqa: E402

DECODE_EVALUATION_SCHEMA = "aura.unified_intrinsic_decode_evaluation.v1"


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


def evaluate_decoding(
    campaign_dir: Path,
    *,
    stem: str,
    per_cell: int,
    evaluation_seed: int,
    max_tokens: int,
    task_depths: tuple[int, ...] | None = None,
    recurrence_depths: tuple[int, ...] | None = None,
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
        or any(
            type(depth) is not int or depth < 2
            for depth in recurrence_depths
        )
    ):
        raise ValueError(
            "decode recurrence depths must be unique non-anchor campaign depths"
        )
    bundle, tokenizer, spec, identity = _load_checkpoint(campaign_dir, stem=stem)
    depths = task_depths or tuple(
        int(value)
        for value in identity.get("task_depths", (identity.get("task_depth"),))
    )
    decoded_depths = recurrence_depths or (max(spec.heldout_depths),)
    if (
        any(depth not in spec.depths for depth in decoded_depths)
    ):
        raise ValueError(
            "decode recurrence depths must be unique non-anchor campaign depths"
        )
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
    for task in tasks:
        prompt, _answer = encode_example(tokenizer, task, bridge)
        compiled_cache: dict[int, tuple[tuple[int, ...], Any, int]] = {}

        def base_logits(tokens: Any) -> Any:
            return _logits(bundle.model(tokens))

        def recurrent_logits(
            tokens: Any,
            plan: Any,
            state_slot_start: int = int(prompt.shape[-1]),
        ) -> Any:
            logits, _telemetry = unified_recurrent_logits(
                bundle.model,
                tokens,
                plan,
                bundle.controller,
                state_slot_start=state_slot_start,
            )
            return logits

        def compiled_logits(
            tokens: Any,
            plan: Any,
            state_slot_start: int = int(prompt.shape[-1]),
            cache: dict[int, tuple[tuple[int, ...], Any, int]] = compiled_cache,
        ) -> Any:
            generated = tuple(
                int(value) for value in tokens[0, state_slot_start:].tolist()
            )
            cached = cache.get(plan.iterations)
            if cached is not None:
                expected, dtype, vocabulary_size = cached
                if (
                    len(generated) >= len(expected)
                    or generated != expected[: len(generated)]
                ):
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
                int(value)
                for value in mx.argmax(state_probabilities[-1], axis=-1)
                .tolist()[0]
            )
            public_tokens = tuple(
                int(value) for value in tokens[0, :state_slot_start].tolist()
            )
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

        arms: tuple[tuple[str, Callable[[Any], Any]], ...] = (
            ("base_t1", base_logits),
            ("trained_t1", lambda tokens: recurrent_logits(tokens, t1)),
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
            candidates.append(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "task_depth": task.depth,
                    "prompt_sha256": hashlib.sha256(task.prompt.encode()).hexdigest(),
                    "arm": arm,
                    "decoded": decoded,
                    "expected": task.answer,
                    "token_ids": token_ids,
                    "stopped_on_eos": stopped,
                    "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                }
            )

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
                row
                for row in candidates
                if row["arm"] == arm and row["task_depth"] == task_depth
            ]
            correct = sum(row["correct"] for row in selected)
            depth_results[str(task_depth)][arm] = {
                "correct": correct,
                "tasks": len(selected),
                "accuracy": correct / len(selected),
                "eos_stops": sum(row["stopped_on_eos"] for row in selected),
            }
    checkpoint_path = campaign_dir / f"{stem}.safetensors"
    body = {
        "schema": DECODE_EVALUATION_SCHEMA,
        "campaign_identity_sha256": identity["identity_sha256"],
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "evaluation_seed": evaluation_seed,
        "per_cell": per_cell,
        "task_depths": list(depths),
        "task_count": len(tasks),
        "recurrence_depths": list(decoded_depths),
        "max_tokens": max_tokens,
        "arm_results": arm_results,
        "depth_results": depth_results,
        "candidates": candidates,
        "claim_boundary": (
            "compiled arms measure public typed-state execution plus "
            "tokenizer-bound emission; trained arms remove that compiler and "
            "measure neural internalization. This is not a preregistered broad "
            "reasoning, resident-32B, frontier, fusion, or WOW result"
        ),
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
    args = parser.parse_args()
    report = evaluate_decoding(
        args.campaign.expanduser().resolve(strict=True),
        stem=args.stem,
        per_cell=args.per_cell,
        evaluation_seed=args.evaluation_seed,
        max_tokens=args.max_tokens,
        task_depths=(
            tuple(int(value) for value in args.task_depths.split(","))
            if args.task_depths
            else None
        ),
        recurrence_depths=(
            tuple(int(value) for value in args.recurrence_depths.split(","))
            if args.recurrence_depths
            else None
        ),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        target = args.report.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        scratch = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        scratch.write_text(encoded, encoding="utf-8")
        with scratch.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(scratch, target)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
